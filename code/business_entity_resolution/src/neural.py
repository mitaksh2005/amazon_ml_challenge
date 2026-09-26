"""Optional neural stage (plan Stage 4): frozen multilingual bi-encoder similarity as a feature, and
record embeddings for the SAE analysis. Needs `torch` + `sentence-transformers`; a GPU instance
(e.g. ml.g5.xlarge) is strongly recommended — encoding millions of records on CPU is slow.

Model: intfloat/multilingual-e5-small (MIT licence, 118M parameters, 384-d), used symmetrically
with the "query: " prefix on both sides as its authors recommend for symmetric tasks."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from data import Paths, load_records, log

MODEL = "intfloat/multilingual-e5-small"


def record_text(df: pd.DataFrame) -> pd.Series:
    return "query: " + df.name_core.fillna("") + " | " + df.addr_norm.fillna("")


def check(require_cuda: bool = True) -> str | None:
    """None if the neural stage can run, else a message saying what is missing."""
    import importlib.util
    missing = [m for m in ("torch", "sentence_transformers") if importlib.util.find_spec(m) is None]
    if missing:
        return (f"--neural needs {', '.join(missing)} in THIS Python ({sys.executable}). Install with:\n"
                f"  \"{sys.executable}\" -m pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
                f"  \"{sys.executable}\" -m pip install sentence-transformers")
    import torch
    if require_cuda and not torch.cuda.is_available():
        return (f"torch {torch.__version__} sees no CUDA GPU (CPU-only build?). Reinstall the CUDA build with\n"
                f"  \"{sys.executable}\" -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128\n"
                f"or pass --neural-cpu to accept a very slow CPU run.")
    return None


def embed(texts: list[str], model_name: str = MODEL, batch_size: int = 256, block: int = 250_000) -> np.ndarray:
    """L2-normalised FP16 embeddings, encoded in length-sorted blocks so host memory stays ~N x d x 2 bytes."""
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(model_name, device=dev)
    if dev == "cuda":
        m = m.half()
    order = np.argsort(np.fromiter((len(t) for t in texts), np.int32, len(texts)), kind="stable")
    emb = np.empty((len(texts), m.get_sentence_embedding_dimension()), np.float16)
    t0 = time.time()
    for b in range(0, len(order), block):
        idx = order[b:b + block]
        emb[idx] = m.encode([texts[i] for i in idx], batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
        done = min(b + block, len(order))
        rate = done / max(time.time() - t0, 1e-9)
        log(f"embedded {done:,}/{len(order):,} records ({rate:,.0f}/s, ~{(len(order) - done) / rate / 60:.0f} min left)")
    return emb


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
