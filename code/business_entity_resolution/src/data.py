"""Data preparation shared by all pipelines.

  * paths of the work directory
  * ground truth → gold counts per S1 entity and encoded true pairs
  * disjoint train samples: `dev` (5 stratified folds, out-of-fold evaluation of every pipeline),
    `hpo` (only seen by hyperparameter studies) and `holdout` (scored once, for the final pipeline)
  * *wide* candidate retrieval for those samples, labelled; every blocking configuration the search
    considers is a truncation of this wide set (per-pass rank ≤ k and score ≥ min_score), which is
    exactly what retrieving with that configuration would return
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import blocking as B
import cleaning as C

WIDE_PASSES = [B.Pass("all", B.W_ALL, k=50, cap=2000), B.Pass("name", B.W_NAME, k=30, cap=2000),
               B.Pass("addr", B.W_ADDR, k=30, cap=2000)]
PASS_NAMES = [p.name for p in WIDE_PASSES]
# FINAL_PASSES of notebook 02 expressed as a truncation of the wide set
B0 = {"all": {"k": 20, "min_score": 0.05}, "name": {"k": 10, "min_score": 0.05},
      "addr": {"k": 10, "min_score": 0.05}}
SPLITS = ("train", "test")
SOURCES = ("S1", "S2", "S3")


@dataclass
class Paths:
    data_dir: Path            # student_resource/dataset
    work: Path
    clean_dir: Path | None = None
    cache_dir: Path | None = None

    def __post_init__(self):
        self.work = Path(self.work)
        self.data_dir = Path(self.data_dir)
        self.clean_dir = Path(self.clean_dir or self.work / "cleaned")
        self.cache_dir = Path(self.cache_dir or self.work / "blocking_cache")
        for d in (self.work, self.art, self.exp, self.reports, self.out):
            d.mkdir(parents=True, exist_ok=True)

    art = property(lambda self: self.work / "artifacts")
    exp = property(lambda self: self.work / "experiments")
    reports = property(lambda self: self.work / "reports")
    out = property(lambda self: self.work / "output")
    validator = property(lambda self: self.data_dir.parent / "utils/validate_submission.py")

    def clean(self, split: str, src: str) -> Path:
        return self.clean_dir / f"{split}_source{src[-1]}.parquet"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- cleaning + keys ------------------------------------------------------------------------------
def clean_and_key(paths: Paths, jobs: int = 2) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    C.clean_all(paths.data_dir, paths.clean_dir, jobs=jobs)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut = {(s, src): ex.submit(B.cache_keys, paths.clean(s, src), paths.cache_dir)
               for s in SPLITS for src in SOURCES}
        return {k: f.result() for k, f in fut.items()}


def key_chunks(paths: Paths, split: str) -> tuple[list[Path], list[Path]]:
    """(S1 chunks, S2+S3 pool chunks) of a split, from the key cache."""
    get = lambda src: B.cache_keys(paths.clean(split, src), paths.cache_dir)   # noqa: E731 (cached: no work)
    return get("S1"), get("S2") + get("S3")


# ---- ground truth ---------------------------------------------------------------------------------
def load_gold(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entities: s1_entity_id, g) for every train S1, (pairs: s1_entity_id, cand_entity_id)."""
    gt = pd.read_csv(paths.data_dir / "train/train_ground_truth.tsv", **C.READ_KW)
    lists = gt.matched_entity_ids.fillna("").str.split(",")
    ex = gt.assign(cand_entity_id=lists).explode("cand_entity_id")
    ex = ex[ex.cand_entity_id.fillna("") != ""]
    pairs = ex[["source1_entity_id", "cand_entity_id"]].rename(columns={"source1_entity_id": "s1_entity_id"})
    pairs = pairs.drop_duplicates().reset_index(drop=True)
    ent = gt[["source1_entity_id"]].rename(columns={"source1_entity_id": "s1_entity_id"})
    ent["g"] = ent.s1_entity_id.map(pairs.groupby("s1_entity_id").size()).fillna(0).astype(np.int64)
    return ent, pairs


def mult_bucket(g: pd.Series) -> pd.Series:
    return pd.cut(g, [-1, 0, 1, 2, 5, 10**9], labels=["0", "1", "2", "3-5", "6+"]).astype(str)


def make_samples(paths: Paths, n_dev: int, n_hpo: int, n_holdout: int, n_folds: int = 5,
                 n_inner: int = 3, seed: int = 13) -> pd.DataFrame:
    """Disjoint random samples of train S1, stratified folds inside `dev` and `hpo`."""
    ent, _ = load_gold(paths)
    s1 = pq.read_table(paths.clean("train", "S1"), columns=["entity_id", "country", "name_core", "addr_norm"]) \
        .to_pandas().rename(columns={"entity_id": "s1_entity_id"})
    ent = ent.merge(s1, on="s1_entity_id", how="left")
    rng = np.random.default_rng(seed)
    ent = ent.iloc[rng.permutation(len(ent))].reset_index(drop=True)
    need = n_dev + n_hpo + n_holdout
    if need > len(ent):
        raise ValueError(f"asked for {need} sampled entities, train has {len(ent)}")
    ent = ent.iloc[:need].copy()
    ent["role"] = np.repeat(["dev", "hpo", "holdout"], [n_dev, n_hpo, n_holdout])
    ent["mult"] = mult_bucket(ent.g)
    ent["stratum"] = ent.country.astype(str) + "|" + ent.mult
    ent["fold"] = -1
    for role, k in (("dev", n_folds), ("hpo", n_inner)):
        m = ent.role == role
        # round-robin inside each stratum (rows already shuffled) → stratified folds
        ent.loc[m, "fold"] = ent[m].groupby("stratum").cumcount().to_numpy() % k
    ent["has_addr"] = ent.addr_norm.fillna("").str.len() > 0
    ent["name_tokens"] = ent.name_core.fillna("").str.split().str.len().fillna(0).astype(int)
    return ent.drop(columns=["name_core", "addr_norm", "stratum"])


# ---- wide candidates for sampled train S1 ---------------------------------------------------------
def label_pairs(s1: pd.Series, cand: pd.Series, gold_pairs: pd.DataFrame) -> np.ndarray:
    """1 where (s1, cand) is a true pair, exact match on encoded ids."""
    key = pd.MultiIndex.from_arrays([B.encode_ids(s1), B.encode_ids(cand)])
    gold = pd.MultiIndex.from_arrays([B.encode_ids(gold_pairs.s1_entity_id), B.encode_ids(gold_pairs.cand_entity_id)])
    return key.isin(gold).astype(np.int8)


def wide_candidates(paths: Paths, s1_ids: np.ndarray, gold_pairs: pd.DataFrame,
                    batch: int = 100_000) -> pd.DataFrame:
    """Retrieve WIDE_PASSES candidates for a subset of train S1 against the full train pool."""
    q_chunks, pool = key_chunks(paths, "train")
    stats = B.pool_stats(pool)
    pool_ids = B.chunk_ids(pool)
    chunk_ids = [B.load_chunk(p).ids.astype(str) for p in q_chunks]
    all_q_ids = np.concatenate(chunk_ids)
    want = pd.Index(s1_ids)
    parts = []
    for b in range(0, len(want), batch):
        sub = set(want[b:b + batch])
        rows = {p.name: np.flatnonzero(np.isin(ids, list(sub))) for p, ids in zip(q_chunks, chunk_ids)}
        u = B.union(B.retrieve(q_chunks, rows, pool, stats, WIDE_PASSES, log=log))
        u.insert(0, "s1_entity_id", all_q_ids[u.q.to_numpy()])
        u.insert(1, "cand_entity_id", pool_ids[u.p.to_numpy()].astype(str))
        parts.append(u.drop(columns=["q", "p"]))
        log(f"wide candidates: {min(b + batch, len(want)):,}/{len(want):,} S1")
    cand = pd.concat(parts, ignore_index=True)
    cand["label"] = label_pairs(cand.s1_entity_id, cand.cand_entity_id, gold_pairs)
    cand["source"] = cand.cand_entity_id.str[:2]
    return cand


def truncate(cand: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Candidates a blocking configuration keeps: a pair survives if some pass p with k_p > 0 ranked
    it within k_p with score ≥ min_score_p. Scores/ranks of dropped passes are blanked and the
    overall score recomputed, so the frame equals what retrieving with those passes returns."""
    keep = np.zeros(len(cand), bool)
    out = {}
    for p in PASS_NAMES:
        c = cfg.get(p, {"k": 0})
        r, s = cand[f"rank_{p}"].to_numpy(), cand[f"score_{p}"].to_numpy()
        m = (r <= c.get("k", 0)) & (s >= c.get("min_score", 0.0)) if c.get("k", 0) > 0 else np.zeros(len(cand), bool)
        out[p] = m
        keep |= m
    df = cand.loc[keep].copy()
    for p in PASS_NAMES:
        m = out[p][keep]
        df.loc[~m, f"rank_{p}"] = np.nan
        df.loc[~m, f"score_{p}"] = np.nan
    df["score"] = df[[f"score_{p}" for p in PASS_NAMES]].max(axis=1)
    df["n_passes"] = df[[f"score_{p}" for p in PASS_NAMES]].notna().sum(axis=1).astype(np.int8)
    return df


def passes_for(cfg: dict) -> list[B.Pass]:
    """Pass objects that retrieve exactly `truncate(wide, cfg)` (for test-time retrieval)."""
    out = []
    for p in WIDE_PASSES:
        c = cfg.get(p.name, {"k": 0})
        if c.get("k", 0) > 0:
            out.append(B.Pass(p.name, p.families, k=int(c["k"]), cap=p.cap,
                              min_score=max(p.min_score, float(c.get("min_score", p.min_score)))))
    return out


def blocking_configs(paths: Paths) -> dict:
    f = paths.exp / "blocking" / "blocking_configs.json"
    cfgs = {"B0": B0}
    if f.exists():
        cfgs.update(json.loads(f.read_text()))
    return cfgs


# ---- record tables --------------------------------------------------------------------------------
REC_COLS = ["entity_id", "country", "business_name", "business_address", "name_norm", "name_core",
            "legal_form", "addr_norm", "region", "postcode", "house_no"]


def load_records(paths: Paths, split: str, sources, ids=None) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    tabs = []
    for src in sources:
        t = pq.read_table(paths.clean(split, src), columns=REC_COLS)
        if ids is not None:
            t = t.filter(pc.is_in(t["entity_id"], value_set=pa.array(np.asarray(ids, dtype=object).astype(str))))
        tabs.append(t)
    df = pa.concat_tables(tabs).to_pandas()
    for c in REC_COLS:
        df[c] = df[c].fillna("").astype(str)
    return df.reset_index(drop=True)
