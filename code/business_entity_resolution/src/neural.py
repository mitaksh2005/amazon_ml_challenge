"""Optional neural stage (plan Stage 4): frozen multilingual bi-encoder similarity as a feature, and
record embeddings for the SAE analysis. Needs `torch` + `sentence-transformers`; a GPU instance
(e.g. ml.g5.xlarge) is strongly recommended — encoding millions of records on CPU is slow.

Model: intfloat/multilingual-e5-small (MIT licence, 118M parameters, 384-d), used symmetrically
with the "query: " prefix on both sides as its authors recommend for symmetric tasks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import Paths, load_records, log

MODEL = "intfloat/multilingual-e5-small"


def record_text(df: pd.DataFrame) -> pd.Series:
    return "query: " + df.name_core.fillna("") + " | " + df.addr_norm.fillna("")


def embed(texts: list[str], model_name: str = MODEL, batch_size: int = 256) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(model_name, device=dev)
    if dev == "cuda":
        m = m.half()
    order = np.argsort([len(t) for t in texts])            # length bucketing: less padding
    out = m.encode([texts[i] for i in order], batch_size=batch_size, normalize_embeddings=True,
                   show_progress_bar=True, convert_to_numpy=True)
    emb = np.empty_like(out)
    emb[order] = out
    return emb.astype(np.float16)


def embeddings_for(paths: Paths, split: str, ids: np.ndarray, tag: str) -> tuple[pd.Index, np.ndarray]:
    """Embeddings of the given records, cached under artifacts/emb_<tag>.npy (+ ids parquet)."""
    f_emb, f_ids = paths.art / f"emb_{tag}.npy", paths.art / f"emb_{tag}_ids.parquet"
    if f_emb.exists() and f_ids.exists():
        return pd.Index(pd.read_parquet(f_ids).entity_id), np.load(f_emb)
    recs = load_records(paths, split, ("S1", "S2", "S3"), ids=np.unique(ids))
    log(f"embedding {len(recs):,} {split} records with {MODEL}")
    emb = embed(record_text(recs).tolist())
    np.save(f_emb, emb)
    recs[["entity_id"]].assign(text=record_text(recs)).to_parquet(f_ids, index=False)
    return pd.Index(recs.entity_id), emb


def cosine(idx: pd.Index, emb: np.ndarray, a_ids, b_ids, chunk: int = 2_000_000) -> np.ndarray:
    a, b = idx.get_indexer(pd.Index(a_ids)), idx.get_indexer(pd.Index(b_ids))
    out = np.full(len(a), np.nan, np.float32)
    for s in range(0, len(a), chunk):
        aa, bb = a[s:s + chunk], b[s:s + chunk]
        ok = (aa >= 0) & (bb >= 0)
        out[s:s + chunk][ok] = (emb[aa[ok]].astype(np.float32) * emb[bb[ok]].astype(np.float32)).sum(1)
    return out


def dense_for_roles(paths: Paths, roles=("dev", "hpo", "holdout")) -> None:
    """dense_<role>.parquet (s1_entity_id, cand_entity_id, dense_cos) for every wide train pair."""
    frames = {r: pd.read_parquet(paths.art / f"pairs_{r}.parquet", columns=["s1_entity_id", "cand_entity_id"])
              for r in roles if (paths.art / f"pairs_{r}.parquet").exists()}
    todo = {r: f for r, f in frames.items() if not (paths.art / f"dense_{r}.parquet").exists()}
    if not todo:
        return
    ids = np.unique(np.concatenate([np.r_[f.s1_entity_id.to_numpy(), f.cand_entity_id.to_numpy()] for f in frames.values()]))
    idx, emb = embeddings_for(paths, "train", ids, "train")
    for r, f in todo.items():
        f.assign(dense_cos=cosine(idx, emb, f.s1_entity_id, f.cand_entity_id)).to_parquet(
            paths.art / f"dense_{r}.parquet", index=False)
        log(f"dense similarity written for {r}: {len(f):,} pairs")
