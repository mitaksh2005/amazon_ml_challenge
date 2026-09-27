"""P10: P9 hardened for the unseen French data and the denser test decoys, with a Rust feature engine
and the matcher trained on the GPU (XGBoost / CUDA). Evaluated on the same dev folds as P5-Bwide.

    python run_p10.py xtra2    XTRA2 features (Rust) for the wide dev / holdout pairs → artifacts/xtra2_<role>.parquet
    python run_p10.py aug      French-structured copies of 25% of the dev entities   → artifacts/pairs_devaug.parquet
    python run_p10.py evaluate P10-base, P10, P10-iso (5-fold OOF)                   → experiments/runs/<name>/
    python run_p10.py compare  paired bootstrap Δ vs P5-Bwide and P10-base           → reports/p10_compare.json
    python run_p10.py submit   refit, holdout (unique assignment), test, validate, zip

The Rust engine (../rust/xfeat) must be built first: cargo build --release.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import models as M  # noqa: E402
import xfeatures2 as X2  # noqa: E402
from data import Paths, load_records, log  # noqa: E402
from er_eval import Registry, paired_delta, write_json  # noqa: E402
from features import PAIR_FEATURES, PoolIDF, RecordStore, pair_features  # noqa: E402
from pipelines import FULL, PipelineConfig, Workspace, evaluate  # noqa: E402

BASE = "P5-Bwide"
AUG_ROLE = "devaug"
AUG_FRAC = 0.25


def configs(reg: Registry) -> list[PipelineConfig]:
    d = PipelineConfig.from_dict(reg.load(BASE, "config.json")).to_dict()
    gpu = dict(matcher="xgb", params="hpo")
    base = PipelineConfig.from_dict({**d, **gpu, "name": "P10-base", "parent": BASE,
                                     "description": "P5-Bwide with the Optuna-tuned XGBoost matcher on the GPU"})
    p10 = PipelineConfig.from_dict({**d, **gpu, "name": "P10", "parent": "P10-base", "calibration": "blend",
                                    "families": [*FULL, "XTRA2"], "extra": {"aug": AUG_ROLE},
                                    "description": "P10-base + XTRA2 (Rust: distinctive-name, script-conflict, "
                                                   "street-core, ordered address numbers, density context) + "
                                                   "French address augmentation + 0.5 isotonic / 0.5 beta calibration"})
    iso = PipelineConfig.from_dict({**p10.to_dict(), "name": "P10-iso", "parent": "P10", "calibration": "isotonic",
                                    "description": "P10 with isotonic calibration (isolates the calibration change)"})
    return [base, p10, iso]


def stage_xtra2(paths: Paths, roles=("dev", "holdout")):
    for role in roles:
        out = paths.art / f"xtra2_{role}.parquet"
        if out.exists():
            log(f"{out.name} exists")
            continue
        t0 = time.time()
        pairs = pd.read_parquet(paths.art / f"pairs_{role}.parquet", columns=["s1_entity_id", "cand_entity_id"])
        s1 = load_records(paths, "train", ["S1"], ids=pairs.s1_entity_id.unique())
        pool = load_records(paths, "train", ["S2", "S3"], ids=pairs.cand_entity_id.unique())
        eng = X2.XEngine(s1, pool, paths.work / f"xfeat_{role}")
        f = eng.for_ids(pairs.s1_entity_id.to_numpy(), pairs.cand_entity_id.to_numpy())
        pd.concat([pairs, f], axis=1).to_parquet(out, index=False)
        log(f"{out.name}: {len(pairs):,} pairs ({time.time() - t0:.0f}s)")
        log(f.describe().T[["count", "mean"]].to_string())


def _frenchify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["addr_norm"] = df.addr_norm.map(X2.frenchify_addr)
    df["house_no"] = df.addr_norm.str.extract(r"^(\d+[a-z]?(?:[/-]\d+[a-z]?)*)\b", expand=False).fillna("")
    return df


def stage_aug(paths: Paths, seed: int = 7):
    """French-structured copies of AUG_FRAC of the dev entities: every address of the S1 and of all its
    candidates gets French street words in French order, then all pair features are recomputed. The IDF
    is refitted on the equally transformed train pool, so 'rue' is as common as 'street' was."""
    out = paths.art / f"pairs_{AUG_ROLE}.parquet"
    if out.exists():
        log(f"{out.name} exists")
        return
    t0 = time.time()
    ws = Workspace(paths)
    ent = ws.entities("dev")
    rng = np.random.default_rng(seed)
    chosen = ent.s1_entity_id.to_numpy()[rng.random(len(ent)) < AUG_FRAC]
    pairs = pq.read_table(paths.art / "pairs_dev.parquet").to_pandas()
    pairs = pairs[pairs.s1_entity_id.isin(pd.Index(chosen))].reset_index(drop=True)
    log(f"aug: {len(chosen):,} entities, {len(pairs):,} pairs")

    cols = ["entity_id", "country", "name_core", "addr_norm"]
    pool_min = pd.concat([pq.read_table(paths.clean("train", s), columns=cols).to_pandas() for s in ("S2", "S3")],
                         ignore_index=True)
    for c in cols:
        pool_min[c] = pool_min[c].fillna("").astype(str)
    pool_min["addr_norm"] = pool_min.addr_norm.map(X2.frenchify_addr)
    idf = PoolIDF(pool_min)
    del pool_min
    log(f"aug: IDF on the French-structured pool ({time.time() - t0:.0f}s)")

    s1 = _frenchify(load_records(paths, "train", ["S1"], ids=chosen))
    pool = _frenchify(load_records(paths, "train", ["S2", "S3"], ids=pairs.cand_entity_id.unique()))
    ex = pd.concat([s1.addr_norm.head(3), pool.addr_norm.head(3)]).tolist()
    log("aug examples: " + " | ".join(ex))
    L, R = RecordStore(s1).apply_idf(idf), RecordStore(pool).apply_idf(idf)
    f = pair_features(L, R, pairs.s1_entity_id.to_numpy(), pairs.cand_entity_id.to_numpy(), log=log)
    del L, R
    aug = pd.concat([pairs.drop(columns=PAIR_FEATURES), f], axis=1)[pairs.columns]
    aug.to_parquet(out, index=False)
    log(f"{out.name}: {len(aug):,} pairs ({time.time() - t0:.0f}s)")

    eng = X2.XEngine(s1, pool, paths.work / f"xfeat_{AUG_ROLE}")
    x = eng.for_ids(pairs.s1_entity_id.to_numpy(), pairs.cand_entity_id.to_numpy())
    pd.concat([pairs[["s1_entity_id", "cand_entity_id"]], x], axis=1) \
        .to_parquet(paths.art / f"xtra2_{AUG_ROLE}.parquet", index=False)
    log(f"xtra2_{AUG_ROLE}.parquet ({time.time() - t0:.0f}s)")


def stage_evaluate(paths: Paths, force: bool, only=None):
    ws, reg = Workspace(paths), Registry(paths.exp)
    for c in configs(reg):
        if only and c.name not in only:
            continue
        evaluate(c, ws, reg, force=force)


def stage_compare(paths: Paths) -> dict:
    reg = Registry(paths.exp)
    base = reg.load(BASE, "entity.parquet").set_index("s1_entity_id")
    res = {}
    ref = {}
    for c in configs(reg):
        if not reg.exists(c.name):
            continue
        e = reg.load(c.name, "entity.parquet").set_index("s1_entity_id").reindex(base.index)
        ref[c.name] = e
        m = reg.load(c.name, "metrics.json")
        res[c.name] = {"macro_f05": float(e.f05.mean()), "base_macro_f05": float(base.f05.mean()),
                       "delta_vs_P5-Bwide": paired_delta(e.f05.to_numpy(), base.f05.to_numpy()),
                       "folds": [f["macro_f05"] for f in m["folds"]], "loss": m["loss"], "slices": m["slices"],
                       "calibration": m["calibration"], "overall": m["overall"]}
        if "P10-base" in ref and c.name != "P10-base":
            res[c.name]["delta_vs_P10-base"] = paired_delta(e.f05.to_numpy(), ref["P10-base"].f05.to_numpy())
        log(f"{c.name}: {res[c.name]['macro_f05']:.5f} vs {BASE} {res[c.name]['base_macro_f05']:.5f}  "
            f"Δ {res[c.name]['delta_vs_P5-Bwide']}" +
            (f"  Δ vs P10-base {res[c.name]['delta_vs_P10-base']}" if "delta_vs_P10-base" in res[c.name] else ""))
    write_json(paths.reports / "p10_compare.json", res)
    return res


def pick(res: dict) -> str:
    """P10 unless its paired CI vs P5-Bwide has a negative lower bound; then the isotonic variant, which
    keeps the XTRA2 features and augmentation but reverts the calibration."""
    if "P10" in res and res["P10"]["delta_vs_P5-Bwide"]["lo"] > 0:
        return "P10"
    if "P10-iso" in res:
        return "P10-iso"
    return "P10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["xtra2", "aug", "evaluate", "compare", "submit", "all"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--clean-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--only", default=None, help="comma list of pipelines to evaluate")
    ap.add_argument("--run", default=None, help="pipeline to submit (default: pick() over the comparison)")
    ap.add_argument("--team", default="EpochAlypse")
    ap.add_argument("--s1-batch", type=int, default=100_000)
    ap.add_argument("--cpu", action="store_true", help="train XGBoost on the CPU")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not a.cpu:
        err = M.xgb_gpu_check()
        if err:
            sys.exit(err)
        M.set_xgb_device("cuda")
    paths = Paths(Path(a.data_dir), Path(a.work_dir), a.clean_dir, a.cache_dir)
    stages = ["xtra2", "aug", "evaluate", "compare", "submit"] if a.stage == "all" else [a.stage]
    for s in stages:
        log(f"===== {s} (xgb device {M.XGB_DEVICE}) =====")
        if s == "xtra2":
            stage_xtra2(paths)
        elif s == "aug":
            stage_aug(paths)
        elif s == "evaluate":
            stage_evaluate(paths, a.force, a.only.split(",") if a.only else None)
        elif s == "compare":
            stage_compare(paths)
        elif s == "submit":
            import submit
            reg = Registry(paths.exp)
            name = a.run or pick(stage_compare(paths))
            log(f"submitting {name}")
            res = submit.run(paths, Workspace(paths), reg, name=name, team=a.team, repo_root=paths.work,
                             unique=True, s1_batch=a.s1_batch)
            log(json.dumps({k: v for k, v in res.items() if k != "validator_output"}, indent=1, default=str))


if __name__ == "__main__":
    main()
