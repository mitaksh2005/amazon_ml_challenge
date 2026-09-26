"""Stage `submit`: refit the chosen pipeline on the whole dev sample, score the untouched holdout once,
run it on the test set, write output/{matching_results,candidate_pairs}.tsv, validate, and zip."""
from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import blocking as B
import models as M
from data import PASS_NAMES, Paths, key_chunks, load_records, log, passes_for
from er_eval import Registry, bootstrap_mean, entity_table, summary, write_json
from features import build_stores, pair_features
from pipelines import PipelineConfig, Workspace, add_ret_ctx, apply_bundle, fit_bundle


def choose(reg: Registry, name: str = "best") -> str:
    if name != "best":
        return name
    idx = reg.index()
    idx = idx[~idx.name.isin(["P-oracle"])]
    return idx.sort_values(["macro_f05", "wall_s"], ascending=[False, True]).iloc[0]["name"]


def final_bundle(cfg: PipelineConfig, ws: Workspace, reg: Registry):
    f = reg.dir(cfg.name) / "final_bundle.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    if cfg.kind in ("empty", "oracle"):
        return None
    b = fit_bundle(cfg, ws.pairs_for(cfg, "dev"), ws.entities("dev"), seed=0, ws=ws)
    f.write_bytes(pickle.dumps(b))
    return b


def holdout(cfg: PipelineConfig, b, ws: Workspace, reg: Registry) -> dict:
    ent = ws.entities("holdout")
    df = ws.pairs_for(cfg, "holdout")
    if b is None:
        out = df.assign(pred=False if cfg.kind == "empty" else df.label.astype(bool))
    else:
        out = apply_bundle(b, df, ent)
        if cfg.graph:
            out["pred"] = M.unique_assignment(out, out.pred.to_numpy())
    e = entity_table(ent, out)
    res = {"pipeline": cfg.name, **summary(e), "ci_macro_f05": bootstrap_mean(e.f05)}
    write_json(reg.dir(cfg.name) / "holdout.json", res)
    log(f"[{cfg.name}] HOLDOUT macro F0.5 {res['macro_f05']:.4f} "
        f"[{res['ci_macro_f05']['lo']:.4f}, {res['ci_macro_f05']['hi']:.4f}] on {len(e):,} entities")
    return res


def test_candidates(paths: Paths, cfg: PipelineConfig, ws: Workspace, batch_chunks: int) -> tuple[Path, list[Path]]:
    q_chunks, pool = key_chunks(paths, "test")
    out_dir = paths.art / f"candidates_test_{cfg.blocking}"
    done = out_dir / "_SUCCESS"
    if not done.exists():
        if out_dir.exists():
            shutil.rmtree(out_dir)
        stats = B.pool_stats(pool)
        t0 = time.time()
        B.generate_candidates(q_chunks, pool, stats, out_dir, passes=passes_for(ws.blocking(cfg.blocking)),
                              batch_chunks=batch_chunks, log=log)
        done.write_text(f"{time.time() - t0:.0f}s")
    return out_dir, q_chunks


def predict_test(paths: Paths, cfg: PipelineConfig, b, ws: Workspace, batch_chunks: int = 2,
                 feature_chunk: int = 500_000) -> dict:
    cand_dir, q_chunks = test_candidates(paths, cfg, ws, batch_chunks)
    s1_ids = B.chunk_ids(q_chunks).astype(str)
    t0 = time.time()
    s1 = load_records(paths, "test", ["S1"])
    pool = load_records(paths, "test", ["S2", "S3"])
    parts = sorted(cand_dir.glob("part-*.parquet"))
    keep = pd.concat([pd.read_parquet(p_, columns=["cand_entity_id"]).cand_entity_id for p_ in parts]).unique() \
        if parts else []
    L, R = build_stores(s1, pool, keep_pool_ids=keep)
    del pool
    log(f"test record stores ready ({time.time() - t0:.0f}s)")
    dense = None
    if cfg.dense:
        import neural
        needed = np.r_[s1.entity_id.to_numpy(), np.asarray(keep, dtype=object)]
        dense = neural.embeddings_for(paths, "test", needed, "test")

    paths.out.mkdir(parents=True, exist_ok=True)
    cand_tsv = paths.out / "candidate_pairs.tsv"
    seen, preds = set(), []
    n_pairs = 0
    with open(cand_tsv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for part in parts:
            t1 = time.time()
            c = pd.read_parquet(part)
            for p in PASS_NAMES:                       # passes not used by this config are absent
                for col in (f"score_{p}", f"rank_{p}"):
                    if col not in c:
                        c[col] = np.nan
            c["n_passes"] = c[[f"score_{p}" for p in PASS_NAMES]].notna().sum(axis=1)
            c["label"] = 0
            if len(c):
                f = pair_features(L, R, c.s1_entity_id.to_numpy(), c.cand_entity_id.to_numpy(), chunk=feature_chunk)
                c = pd.concat([c.reset_index(drop=True), f], axis=1)
                if dense is not None:
                    import neural
                    c["dense_cos"] = neural.cosine(dense[0], dense[1], c.s1_entity_id, c.cand_entity_id)
                c = add_ret_ctx(c, dense=dense is not None)
            ent = pd.DataFrame({"s1_entity_id": c.s1_entity_id.unique(), "g": 0})
            if b is None:
                c["p"], c["pred"] = 0.0, False
            else:
                c = apply_bundle(b, c, ent)
            n_pairs += len(c)
            srt = c.sort_values(["s1_entity_id", "score"], ascending=[True, False])
            lists = srt.groupby("s1_entity_id", sort=False).cand_entity_id.agg(",".join)
            for s, l in lists.items():
                fh.write(f"{s}\t{l}\n")
            seen.update(lists.index)
            preds.append(c.loc[c.pred.astype(bool), ["s1_entity_id", "cand_entity_id", "p"]])
            log(f"{part.name}: {len(ent):,} S1, {len(c):,} pairs, {int(c.pred.sum()):,} predicted ({time.time() - t1:.0f}s)")
        for s in s1_ids:                               # S1 with no candidates: empty row
            if s not in seen:
                fh.write(f"{s}\t\n")
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["s1_entity_id", "cand_entity_id", "p"])
    if cfg.graph and len(pred):
        keep = M.unique_assignment(pred.assign(pred=True), np.ones(len(pred), bool))
        pred = pred[keep]
    pred = pred.sort_values(["s1_entity_id", "p"], ascending=[True, False])
    lists = pred.groupby("s1_entity_id").cand_entity_id.agg(",".join)
    res = pd.DataFrame({"source1_entity_id": s1_ids,
                        "matched_entity_ids": pd.Series(s1_ids).map(lists).fillna("").to_numpy()})
    res.to_csv(paths.out / "matching_results.tsv", sep="\t", index=False, lineterminator="\n", encoding="utf-8")
    stats = {"n_s1": int(len(s1_ids)), "n_candidate_pairs": int(n_pairs), "n_predicted_pairs": int(len(pred)),
             "s1_with_prediction": float((res.matched_entity_ids != "").mean()),
             "cands_per_s1": n_pairs / max(len(s1_ids), 1)}
    log(f"test predictions: {stats}")
    return stats


def validate(paths: Paths) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(paths.validator), "--matching", str(paths.out / "matching_results.tsv"),
                        "--candidate", str(paths.out / "candidate_pairs.tsv"), "--test-dir",
                        str(paths.data_dir / "test")], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)[-4000:]


def make_zip(paths: Paths, team: str, repo_root: Path) -> Path:
    code = Path(__file__).resolve().parent.parent          # code/business_entity_resolution
    z = paths.work / f"{team}_submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            zf.write(paths.out / f, f"output/{f}")
        for f in code.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                zf.write(f, f"code/business_entity_resolution/{f.relative_to(code)}")
        # the filled-in methodology document: first non-empty candidate
        docs = [repo_root / "Documentation_template.md", paths.data_dir.parent / "Documentation_template.md"]
        doc = next((d for d in docs if d.exists() and d.stat().st_size > 0), None)
        if doc is not None:
            zf.write(doc, "Documentation_template.md")
    return z


def run(paths: Paths, ws: Workspace, reg: Registry, name: str = "best", batch_chunks: int = 2,
        team: str | None = None, repo_root: Path | None = None, feature_chunk: int = 500_000) -> dict:
    name = choose(reg, name)
    cfg = PipelineConfig.from_dict(reg.load(name, "config.json"))
    log(f"submitting pipeline {name}: {cfg.description}")
    b = final_bundle(cfg, ws, reg)
    hold = holdout(cfg, b, ws, reg)
    stats = predict_test(paths, cfg, b, ws, batch_chunks=batch_chunks, feature_chunk=feature_chunk)
    ok, msg = validate(paths)
    log(f"validator: {'PASS' if ok else 'FAIL'}\n{msg}")
    out = {"pipeline": name, "holdout": hold, "test": stats, "validator_pass": ok, "validator_output": msg}
    if ok and team:
        out["zip"] = str(make_zip(paths, team, repo_root or paths.work))
        log(f"submission package: {out['zip']}")
    write_json(paths.out / "submission_report.json", out)
    return out
