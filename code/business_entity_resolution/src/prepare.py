"""Stage `prepare`: clean → keys → samples/folds → wide labelled candidates → pair features.

Outputs (under <work>/artifacts/, each step skipped when its output exists):
  samples.parquet        sampled train S1 with role (dev/hpo/holdout), fold, country, g, slices
  cand_train_wide.parquet wide candidates of all sampled S1, labelled
  pairs_<role>.parquet   wide candidates + pair features, one file per role
  blocking_misses.parquet true pairs blocking misses (wide and B0), with a heuristic reason
  gold_by_source.parquet gold counts per (S1, source) for recall-by-source plots
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from data import B0, Paths, clean_and_key, load_gold, load_records, log, make_samples, truncate, wide_candidates
from er_eval import write_json
from features import build_stores, pair_features


def run(paths: Paths, n_dev: int, n_hpo: int, n_holdout: int, jobs: int = 2, seed: int = 13,
        feature_chunk: int = 500_000) -> None:
    art = paths.art
    t0 = time.time()
    clean_and_key(paths, jobs=jobs)
    log(f"cleaned + keyed ({time.time() - t0:.0f}s)")

    if not (art / "samples.parquet").exists():
        s = make_samples(paths, n_dev, n_hpo, n_holdout, seed=seed)
        s.to_parquet(art / "samples.parquet", index=False)
        log(f"samples: {s.role.value_counts().to_dict()}, singleton rate {float((s.g == 0).mean()):.3f}")
    samples = pd.read_parquet(art / "samples.parquet")
    _, gold = load_gold(paths)
    gold = gold[gold.s1_entity_id.isin(samples.s1_entity_id)]
    if not (art / "gold_by_source.parquet").exists():
        gold.assign(source=gold.cand_entity_id.str[:2]).groupby(["s1_entity_id", "source"]).size() \
            .rename("n").reset_index().to_parquet(art / "gold_by_source.parquet", index=False)

    if not (art / "cand_train_wide.parquet").exists():
        t0 = time.time()
        cand = wide_candidates(paths, samples.s1_entity_id.to_numpy(), gold)
        cand.to_parquet(art / "cand_train_wide.parquet", index=False)
        log(f"wide candidates: {len(cand):,} pairs, {cand.label.sum():,} true ({time.time() - t0:.0f}s)")

    roles = [r for r in ("dev", "hpo", "holdout") if not (art / f"pairs_{r}.parquet").exists()]
    if roles or not (art / "blocking_misses.parquet").exists():
        cand = pd.read_parquet(art / "cand_train_wide.parquet")
        t0 = time.time()
        s1 = load_records(paths, "train", ["S1"], ids=samples.s1_entity_id.to_numpy())
        pool = load_records(paths, "train", ["S2", "S3"])
        L, R = build_stores(s1, pool, keep_pool_ids=cand.cand_entity_id.unique())
        log(f"record stores: {len(s1):,} S1, {len(pool):,} pool ({time.time() - t0:.0f}s)")
        for role in roles:
            t0 = time.time()
            ids = samples.s1_entity_id[samples.role == role]
            c = cand[cand.s1_entity_id.isin(ids)].reset_index(drop=True)
            f = pair_features(L, R, c.s1_entity_id.to_numpy(), c.cand_entity_id.to_numpy(), chunk=feature_chunk, log=log)
            pd.concat([c, f], axis=1).to_parquet(art / f"pairs_{role}.parquet", index=False)
            log(f"pairs_{role}: {len(c):,} pairs with {f.shape[1]} features ({time.time() - t0:.0f}s, "
                f"{len(c) / max(time.time() - t0, 1e-9):,.0f} pairs/s)")
        if not (art / "blocking_misses.parquet").exists():
            misses(paths, samples, gold, cand, s1, pool)
        del L, R, pool
    write_json(art / "prepare_done.json", {"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                           "n_dev": n_dev, "n_hpo": n_hpo, "n_holdout": n_holdout})


def misses(paths: Paths, samples: pd.DataFrame, gold: pd.DataFrame, cand: pd.DataFrame,
           s1: pd.DataFrame, pool: pd.DataFrame) -> None:
    """Heuristic reason for each dev true pair that blocking misses (plan §6.1 miss taxonomy)."""
    dev = samples.s1_entity_id[samples.role == "dev"]
    g = gold[gold.s1_entity_id.isin(dev)]
    c = cand[cand.s1_entity_id.isin(dev)]
    rows = []
    for level, cc in (("wide", c), ("B0", truncate(c, B0))):
        hit = pd.MultiIndex.from_frame(cc[["s1_entity_id", "cand_entity_id"]])
        m = ~pd.MultiIndex.from_frame(g[["s1_entity_id", "cand_entity_id"]]).isin(hit)
        rows.append(g[m].assign(level=level))
    miss = pd.concat(rows, ignore_index=True)
    a = s1.set_index("entity_id").reindex(miss.s1_entity_id)
    b = pool.set_index("entity_id").reindex(miss.cand_entity_id)
    na, nb = a.name_core.fillna("").to_numpy(), b.name_core.fillna("").to_numpy()
    share = np.array([bool(set(x.split()) & set(y.split())) for x, y in zip(na, nb)])
    reason = np.select(
        [b.addr_norm.fillna("").to_numpy() == "", (na == "") | (nb == ""),
         (a.region.fillna("").to_numpy() != b.region.fillna("").to_numpy()) & (a.region.fillna("").to_numpy() != "")
         & (b.region.fillna("").to_numpy() != ""),
         ~share, (np.char.str_len(na.astype(str)) <= 4) | (np.char.str_len(nb.astype(str)) <= 4)],
        ["candidate has no address", "empty name", "region differs", "no shared name token", "very short name"],
        "shares tokens but ranked out")
    miss["reason"] = reason
    miss.to_parquet(paths.art / "blocking_misses.parquet", index=False)
    log(f"blocking misses: {miss.groupby('level').size().to_dict()}")
