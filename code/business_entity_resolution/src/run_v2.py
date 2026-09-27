"""P9: P5-Bwide + the fixes from the output audit (RESULTS_AND_PIPELINE.md §6), evaluated on the same dev folds.

    python run_v2.py xtra     XTRA features for the wide dev / holdout pairs      → artifacts/xtra_<role>.parquet
    python run_v2.py evaluate P9a (beta calibration only) and P9 (+ XTRA features)  → experiments/runs/<name>/
    python run_v2.py compare  paired bootstrap Δ vs P5-Bwide (same dev entities)   → reports/v2_compare.json
    python run_v2.py submit   refit the best, holdout with unique assignment, test → output/, zip
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

import xfeatures as X  # noqa: E402
from data import Paths, load_records, log  # noqa: E402
from er_eval import Registry, paired_delta, write_json  # noqa: E402
from pipelines import FULL, PipelineConfig, Workspace, evaluate  # noqa: E402

BASE = "P5-Bwide"


def configs(reg: Registry) -> list[PipelineConfig]:
    base = PipelineConfig.from_dict(reg.load(BASE, "config.json"))
    d = base.to_dict()
    p9a = PipelineConfig.from_dict({**d, "name": "P9a", "parent": BASE, "calibration": "beta",
                                    "description": "P5-Bwide with beta calibration (no isotonic plateau)"})
    p9 = PipelineConfig.from_dict({**d, "name": "P9", "parent": "P9a", "calibration": "beta",
                                   "families": [*FULL, "XTRA"],
                                   "description": "P9a + XTRA: joined-name/acronym/skeleton, house-number "
                                                  "canonical/truncation, native-script flags, candidate-set context"})
    return [p9a, p9]


def stage_xtra(paths: Paths, roles=("dev", "holdout")):
    for role in roles:
        out = paths.art / f"xtra_{role}.parquet"
        if out.exists():
            log(f"{out.name} exists")
            continue
        t0 = time.time()
        pairs = pd.read_parquet(paths.art / f"pairs_{role}.parquet", columns=["s1_entity_id", "cand_entity_id"])
        s1 = load_records(paths, "train", ["S1"], ids=pairs.s1_entity_id.unique())
        pool = load_records(paths, "train", ["S2", "S3"], ids=pairs.cand_entity_id.unique())
        f = X.for_pairs(s1, pool, pairs.s1_entity_id.to_numpy(), pairs.cand_entity_id.to_numpy())
        pd.concat([pairs, f], axis=1).to_parquet(out, index=False)
        log(f"{out.name}: {len(pairs):,} pairs ({time.time() - t0:.0f}s)")
        log(f.describe().T[["count", "mean"]].to_string())


def stage_evaluate(paths: Paths, force: bool):
    ws, reg = Workspace(paths), Registry(paths.exp)
    for c in configs(reg):
        evaluate(c, ws, reg, force=force)


def stage_compare(paths: Paths):
    reg = Registry(paths.exp)
    base = reg.load(BASE, "entity.parquet").set_index("s1_entity_id")
    res = {}
    for n in ("P9a", "P9"):
        if not reg.exists(n):
            continue
        e = reg.load(n, "entity.parquet").set_index("s1_entity_id").reindex(base.index)
        m = reg.load(n, "metrics.json")
        res[n] = {"macro_f05": float(e.f05.mean()), "base_macro_f05": float(base.f05.mean()),
                  "delta": paired_delta(e.f05.to_numpy(), base.f05.to_numpy()),
                  "loss": m["loss"], "slices": m["slices"], "calibration": m["calibration"],
                  "overall": m["overall"]}
        log(f"{n}: {res[n]['macro_f05']:.5f} vs {BASE} {res[n]['base_macro_f05']:.5f}  Δ {res[n]['delta']}")
    write_json(paths.reports / "v2_compare.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["xtra", "evaluate", "compare", "submit", "all"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--clean-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--run", default=None, help="pipeline to submit (default: best of P9a/P9 by OOF, if it beats base)")
    ap.add_argument("--team", default="EpochAlypse")
    ap.add_argument("--s1-batch", type=int, default=100_000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    paths = Paths(Path(a.data_dir), Path(a.work_dir), a.clean_dir, a.cache_dir)
    stages = ["xtra", "evaluate", "compare", "submit"] if a.stage == "all" else [a.stage]
    for s in stages:
        log(f"===== {s} =====")
        if s == "xtra":
            stage_xtra(paths)
        elif s == "evaluate":
            stage_evaluate(paths, a.force)
        elif s == "compare":
            stage_compare(paths)
        elif s == "submit":
            import submit
            reg = Registry(paths.exp)
            name = a.run
            if name is None:
                res = stage_compare(paths)
                ok = {n: r for n, r in res.items() if r["delta"]["lo"] > 0}
                if not ok:
                    sys.exit("no P9 variant beats P5-Bwide (paired CI lower bound <= 0): not submitting")
                name = max(ok, key=lambda n: ok[n]["macro_f05"])
            log(f"submitting {name}")
            res = submit.run(paths, Workspace(paths), reg, name=name, team=a.team, repo_root=paths.work,
                             unique=True, s1_batch=a.s1_batch)
            log(json.dumps({k: v for k, v in res.items() if k != "validator_output"}, indent=1, default=str))


if __name__ == "__main__":
    main()
