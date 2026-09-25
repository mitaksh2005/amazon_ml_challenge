"""Candidate generation (blocking) for S1 → S2/S3 entity resolution.

Approach: multi-pass sparse TF-IDF top-K retrieval, partitioned by country.

  1. Every cleaned record emits hashed *keys* from several families (name tokens, phonetic skeletons,
     joined name, region-scoped address tokens, house number + street, initials).
  2. Keys are weighted by IDF computed per country over the S2+S3 pool. Keys held by more than
     `cap` pool records carry little identity and would explode the candidate count, so they are
     dropped from retrieval (they still count in the vector norm).
  3. Each *pass* uses a subset of families: S1 vectors are multiplied against S2/S3 vectors
     (cosine), streaming the pool in chunks and keeping a running top-K per S1 record. Memory is
     bounded by one pool chunk plus the (n_S1 × K) top-K buffers, whatever the pool size.
  4. Passes are unioned. A junk or initials-only name can still be found by its address and vice versa.

Country is a hard partition (100% of true pairs share it); unseen countries (test has France) are
handled like any other label.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

KEY_BITS = 24
N_BUCKETS = 1 << KEY_BITS
FAMILIES = ["name", "phon", "join", "nreg", "init", "addr", "anum", "house"]
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}

# Address words that never identify a place on their own (kept out of address keys entirely).
ADDR_GENERIC = {
    "street", "road", "avenue", "drive", "lane", "court", "place", "boulevard", "highway", "suite", "floor",
    "apartment", "unit", "number", "building", "house", "near", "opposite", "behind", "the", "and", "of", "no",
    "rue", "de", "la", "le", "du", "des", "chemin", "allee", "impasse", "po", "pobox", "box", "city", "district",
    "village", "post", "office", "taluka", "block", "sector", "plot", "ground", "first", "second", "main", "cross",
}


# ---- key generation -----------------------------------------------------------------------------
_SKEL_SUBS = [(re.compile(p), r) for p, r in [
    (r"ph", "f"), (r"(?<=[bcdfgjklmnpqrstvwxz])h", ""), (r"[cq]", "k"), (r"w", "v"), (r"z", "s"), (r"y", "i"),
]]


@lru_cache(maxsize=1 << 20)
def skeleton(tok: str) -> str:
    """Consonant skeleton: robust to vowel typos and transliteration (innovative / inovetiv → nvtv)."""
    t = tok
    for pat, rep in _SKEL_SUBS:
        t = pat.sub(rep, t)
    t = re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiou]", "", t))   # leading vowels too: yunaited / united → ntd
    return t or tok


# Indian states split in 2000/2014: sources disagree on which one a place belongs to (Hyderabad is filed
# under both Telangana and Andhra Pradesh), so region-scoped keys use the pre-split state.
REGION_GROUP = {"telangana": "andhra pradesh", "uttarakhand": "uttar pradesh", "chhattisgarh": "madhya pradesh",
                "jharkhand": "bihar"}

_LEET = str.maketrans("0134567", "oleastt")
_ORDINAL_WORDS = {w: str(i) for i, w in enumerate(
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
    "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
_ORDINAL_RE = re.compile(r"^0*(\d+)(?:st|nd|rd|th)$")


@lru_cache(maxsize=1 << 20)
def name_token(t: str) -> str:
    """Undo digit-for-letter typos inside words: bi0technologies → biotechnologies, 5hop → shop."""
    if t.isalpha() or t.isdigit() or not any(c.isalpha() for c in t):
        return t
    return t.translate(_LEET)


@lru_cache(maxsize=1 << 20)
def addr_tokens(t: str) -> tuple[str, ...]:
    """Canonical address token(s): 0067 → 67, 12th/twelfth → 12, 116st → 116, 1237-1239 → 1237 1239,
    1450-b → 1450 b. Hyphenated non-numeric tokens (lege-cap-ferret, 7-1-112) stay whole."""
    t = t.strip("-/")
    if not t:
        return ()
    if t in _ORDINAL_WORDS:
        return (_ORDINAL_WORDS[t],)
    m = _ORDINAL_RE.match(t)
    if m:
        return (m.group(1),)
    if t.isdigit():
        return (t.lstrip("0") or "0",)
    if "-" in t and t[0].isdigit():
        parts = [p for p in t.split("-") if p]
        if len(parts) == 2 and parts[0].isdigit():   # a range or a unit suffix, not an Indian door id
            return tuple(q for p in parts for q in addr_tokens(p))
    return (t,)


def record_keys(name_core: str, addr_norm: str, region: str) -> list[tuple[str, int]]:
    """All (key, family) pairs for one record, de-duplicated.

    name  n|tok        cleaned name tokens, plus adjacent 2-3 tokens joined (royal intelligence ↔ royalintelligence)
    phon  p|skel       consonant skeleton of name tokens (typos, transliteration)
    join  j|name       whole name joined (reetexportsindia.com ↔ reet exports india)
    nreg  r|region|tok name tokens scoped to the region: rare enough to rank well where "royal" alone is not
    init  i|letters    initials (Mallick Sons ↔ MS)
    addr  a|region|tok address tokens scoped to the region
    anum  g|tok        numeric address tokens country-wide (survives a wrong/missing region)
    house h|num|street house number + first street word
    """
    keys = set()
    own_region, region = region, REGION_GROUP.get(region, region)
    toks = [name_token(t) for t in name_core.split() if len(t) >= 2 or t.isdigit()]
    for t in toks:
        keys.add(("n|" + t, FAM_ID["name"]))
        if region:
            keys.add((f"r|{region}|{t}", FAM_ID["nreg"]))
        if len(t) >= 4 and not t.isdigit():
            keys.add(("p|" + skeleton(t), FAM_ID["phon"]))
    for n in (2, 3):   # joined bigrams/trigrams match concatenated names (rajkamalcomplexdhantoli)
        for i in range(len(toks) - n + 1):
            keys.add(("n|" + "".join(toks[i:i + n]), FAM_ID["name"]))
    joined = "".join(toks)
    if len(joined) >= 6:
        keys.add(("j|" + joined, FAM_ID["join"]))
    if len(toks) >= 2:
        keys.add(("i|" + "".join(t[0] for t in toks), FAM_ID["init"]))
    elif len(joined) in (2, 3, 4) and joined.isalpha():   # a name that *is* initials ("ms", "lp")
        keys.add(("i|" + joined, FAM_ID["init"]))

    house = street = None
    for raw in addr_norm.replace(",", " ").split():
        for t in addr_tokens(raw):
            if t in ADDR_GENERIC or t == own_region or t == region:
                continue
            is_num = t[0].isdigit()
            if is_num:
                house = house or t
            if len(t) >= 3 and any(c.isdigit() for c in t):
                keys.add(("g|" + t, FAM_ID["anum"]))
            if not is_num and len(t) < 3:
                continue
            if not is_num and street is None:
                street = t
            keys.add((f"a|{region}|{t}", FAM_ID["addr"]))
    if house and street:
        keys.add((f"h|{house}|{street}", FAM_ID["house"]))
    return list(keys)


def hash_keys(keys: np.ndarray) -> np.ndarray:
    return (pd.util.hash_array(keys, categorize=False) & np.uint64(N_BUCKETS - 1)).astype(np.int32)


@dataclass
class KeyChunk:
    """Keys of a block of records in CSR layout (row i owns entries indptr[i]:indptr[i+1])."""
    ids: np.ndarray       # entity ids, bytes
    country: np.ndarray   # country label per row, str
    indptr: np.ndarray
    bucket: np.ndarray    # int32 hashed key
    fam: np.ndarray       # int8 family id

    @property
    def n(self) -> int:
        return len(self.ids)


def build_keys(df: pd.DataFrame) -> KeyChunk:
    cols = [df[c].tolist() for c in ("name_core", "addr_norm", "region")]
    per_row = [record_keys(*r) for r in zip(*cols)]
    counts = np.fromiter((len(k) for k in per_row), np.int64, len(per_row))
    flat = [k for ks in per_row for k in ks]
    keys = np.array([k for k, _ in flat], dtype=object)
    fam = np.array([f for _, f in flat], dtype=np.int8)
    indptr = np.zeros(len(per_row) + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    bucket = hash_keys(keys) if len(keys) else np.zeros(0, np.int32)
    return KeyChunk(df.entity_id.to_numpy().astype("S"), df.country.to_numpy().astype(str), indptr, bucket, fam)


# ---- on-disk key cache --------------------------------------------------------------------------
COLS = ["entity_id", "country", "name_core", "addr_norm", "region"]


def cache_keys(parquet: Path, cache_dir: Path, rows_per_chunk: int = 250_000) -> list[Path]:
    """Build keys for a cleaned parquet file, one .npz per chunk. Skips work already on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = cache_dir / f"{parquet.stem}.done.json"
    if done.exists():
        return [cache_dir / p for p in json.loads(done.read_text())]
    paths = []
    for i, batch in enumerate(pq.ParquetFile(parquet).iter_batches(batch_size=rows_per_chunk, columns=COLS)):
        kc = build_keys(batch.to_pandas())
        p = cache_dir / f"{parquet.stem}.{i:03d}.npz"
        np.savez(p, ids=kc.ids, country=kc.country, indptr=kc.indptr, bucket=kc.bucket, fam=kc.fam)
        paths.append(p)
    done.write_text(json.dumps([p.name for p in paths]))
    return paths


def load_chunk(path: Path) -> KeyChunk:
    z = np.load(path)
    return KeyChunk(z["ids"], z["country"], z["indptr"], z["bucket"], z["fam"])


@dataclass
class PoolStats:
    df: dict[str, np.ndarray]   # per-country document frequency of every bucket
    n: dict[str, int]           # per-country pool size
    _idf: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def idf(self, c: str) -> np.ndarray:
        if c not in self._idf:
            self._idf[c] = np.log(np.float32(1 + self.n[c]) / (1 + self.df[c].astype(np.float32))) + 1
        return self._idf[c]


def pool_stats(chunks: list[Path]) -> PoolStats:
    """Document frequencies over the S2+S3 pool (keys are already unique per record)."""
    df, n = {}, {}
    for p in chunks:
        kc = load_chunk(p)
        row_country = np.repeat(kc.country, np.diff(kc.indptr))
        for c, cnt in zip(*np.unique(kc.country, return_counts=True)):
            if c not in df:
                df[c], n[c] = np.zeros(N_BUCKETS, np.uint32), 0
            df[c] += np.bincount(kc.bucket[row_country == c], minlength=N_BUCKETS).astype(np.uint32)
            n[c] += int(cnt)
    return PoolStats(df, n)


# ---- retrieval ----------------------------------------------------------------------------------
@dataclass
class Pass:
    name: str
    families: dict[str, float]    # family → weight multiplier
    k: int = 10
    cap: int = 2_000              # drop keys held by more than `cap` pool records (per country)
    min_score: float = 0.05       # ignore weaker overlaps (also keeps the top-k sort cheap)


def to_matrix(kc: KeyChunk, rows: np.ndarray, p: Pass, idf: np.ndarray,
              stop: np.ndarray | None = None) -> sp.csr_matrix:
    """L2-normalised TF-IDF rows for the selected records, restricted to the pass's families.
    Keys in `stop` count toward the norm but are removed from the returned matrix."""
    counts = np.diff(kc.indptr)
    row_of = np.repeat(np.arange(kc.n), counts)
    fam_w = np.zeros(len(FAMILIES), np.float32)
    for f, w in p.families.items():
        fam_w[FAM_ID[f]] = w
    sel = np.zeros(kc.n, bool)
    sel[rows] = True
    m = sel[row_of] & (fam_w[kc.fam] > 0)
    b, r = kc.bucket[m], row_of[m]
    w = fam_w[kc.fam[m]] * idf[b]
    local = np.full(kc.n, -1, np.int64)
    local[rows] = np.arange(len(rows))
    r = local[r]
    norm = np.sqrt(np.bincount(r, weights=w * w, minlength=len(rows))).astype(np.float32)
    w = w / np.maximum(norm[r], 1e-9)
    if stop is not None:
        keep = ~stop[b]
        b, r, w = b[keep], r[keep], w[keep]
    return sp.csr_matrix((w.astype(np.float32), (r, b)), shape=(len(rows), N_BUCKETS))


def csr_topk(R: sp.csr_matrix, k: int, min_score: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k entries (value >= min_score) of every row of a CSR matrix → (row, col, value),
    rank-ordered within each row."""
    rows = np.repeat(np.arange(R.shape[0], dtype=np.int64), np.diff(R.indptr))
    cols, vals = R.indices, R.data
    if min_score > 0:
        m = vals >= min_score
        rows, cols, vals = rows[m], cols[m], vals[m]
    # one float sort instead of lexsort: row ascending, then value descending (values are in (0, 1])
    order = np.argsort(rows + (1.0 - np.minimum(vals, 1.0)) * 0.999)
    rows_o = rows[order]
    starts = np.r_[0, np.flatnonzero(np.diff(rows_o)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = order[rank < k]
    return rows[keep], cols[keep], vals[keep]


@dataclass
class TopK:
    """Running top-k (score, pool index) per query row."""
    k: int
    n: int
    score: np.ndarray = field(init=False)
    idx: np.ndarray = field(init=False)

    def __post_init__(self):
        self.score = np.full((self.n, self.k), -1.0, np.float32)
        self.idx = np.full((self.n, self.k), -1, np.int64)

    def merge(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, row_offset: int, col_offset: int):
        if not len(rows):
            return
        lo, hi = row_offset + rows.min(), row_offset + rows.max() + 1
        new_s = np.full((hi - lo, self.k), -1.0, np.float32)
        new_i = np.full((hi - lo, self.k), -1, np.int64)
        rr = rows + row_offset - lo
        # rows arrive rank-ordered from csr_topk, so the rank is the position within the row's run
        starts = np.r_[0, np.flatnonzero(np.diff(rr)) + 1]
        rank = np.arange(len(rr)) - np.repeat(starts, np.diff(np.r_[starts, len(rr)]))
        new_s[rr, rank] = vals
        new_i[rr, rank] = cols + col_offset
        s = np.concatenate([self.score[lo:hi], new_s], axis=1)
        i = np.concatenate([self.idx[lo:hi], new_i], axis=1)
        top = np.argpartition(-s, self.k - 1, axis=1)[:, : self.k]
        self.score[lo:hi] = np.take_along_axis(s, top, 1)
        self.idx[lo:hi] = np.take_along_axis(i, top, 1)


def retrieve(query: list[Path], query_rows: dict[str, np.ndarray] | None, pool: list[Path],
             stats: PoolStats, passes: list[Pass], q_step: int = 20_000,
             log=print) -> pd.DataFrame:
    """Run every pass. Returns candidate pairs (q, p, score, pass) with global row indices:
    q indexes the concatenated query chunks, p the concatenated pool chunks.

    query_rows optionally restricts the query to a subset: {chunk path name: row indices}."""
    q_chunks = [load_chunk(p) for p in query]
    q_off = np.r_[0, np.cumsum([c.n for c in q_chunks])]
    n_q = int(q_off[-1])
    q_country = np.concatenate([c.country for c in q_chunks])
    q_sel = np.zeros(n_q, bool)
    if query_rows is None:
        q_sel[:] = True
    else:
        for j, p in enumerate(query):
            q_sel[q_off[j] + query_rows.get(p.name, np.zeros(0, np.int64))] = True
    countries = [c for c in np.unique(q_country[q_sel]) if c in stats.df]
    out = []
    for ps in passes:
        t0 = time.time()
        idf = {c: stats.idf(c) for c in countries}
        stop = {c: stats.df[c] > ps.cap for c in countries}
        # query matrices per country (global query rows, in order)
        Q, Q_rows = {}, {}
        for c in countries:
            mats, rows_g = [], []
            for j, kc in enumerate(q_chunks):
                rows = np.flatnonzero((kc.country == c) & q_sel[q_off[j]:q_off[j + 1]])
                if len(rows):
                    mats.append(to_matrix(kc, rows, ps, idf[c], stop[c]))
                    rows_g.append(rows + q_off[j])
            Q[c], Q_rows[c] = sp.vstack(mats).tocsr(), np.concatenate(rows_g)
        tops = {c: TopK(ps.k, Q[c].shape[0]) for c in countries}
        p_off = 0
        for pi, path in enumerate(pool):
            kc = load_chunk(path)
            for c in countries:
                rows = np.flatnonzero(kc.country == c)
                if not len(rows):
                    continue
                PT = to_matrix(kc, rows, ps, idf[c]).T.tocsr()
                for q0 in range(0, Q[c].shape[0], q_step):
                    r, col, v = csr_topk(Q[c][q0:q0 + q_step] @ PT, ps.k, ps.min_score)
                    tops[c].merge(r, rows[col], v, q0, p_off)
            p_off += kc.n
        for c in countries:
            t = tops[c]
            qq = np.repeat(Q_rows[c], ps.k)
            m = t.idx.ravel() >= 0
            out.append(pd.DataFrame({"q": qq[m], "p": t.idx.ravel()[m], "score": t.score.ravel()[m],
                                     "pass": ps.name}))
        log(f"pass {ps.name}: {time.time() - t0:.0f}s")
    cand = pd.concat(out, ignore_index=True)
    cand["pass"] = cand["pass"].astype("category")
    return cand


def union(cand: pd.DataFrame) -> pd.DataFrame:
    """One row per (q, p) with the best score and the passes that found it."""
    g = cand.groupby(["q", "p"], sort=False)
    return pd.DataFrame({"score": g.score.max(), "passes": g["pass"].agg(lambda s: "+".join(sorted(set(s))))}) \
        .reset_index()


def chunk_ids(paths: list[Path]) -> np.ndarray:
    return np.concatenate([load_chunk(p).ids for p in paths])


# ---- evaluation ---------------------------------------------------------------------------------
def f_beta(precision: np.ndarray, recall: np.ndarray, beta: float = 0.5) -> np.ndarray:
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.where(denom > 0, (1 + b2) * precision * recall / np.where(denom > 0, denom, 1), 0.0)


def evaluate(cand: pd.DataFrame, truth: pd.DataFrame, n_query: int, n_pool: int) -> dict:
    """Blocking quality vs ground truth. `truth` has (q, p) for every true pair of the evaluated
    queries; `n_query` counts evaluated queries including singletons.

    ceiling_f05: the macro F0.5 a *perfect* matcher could reach on these candidates
    (precision 1, recall = blocking recall; singletons score 1 by predicting nothing)."""
    pairs = cand[["q", "p"]].drop_duplicates()
    hit = truth.merge(pairs, on=["q", "p"], how="left", indicator=True)["_merge"].eq("both").to_numpy()
    per_q = pd.DataFrame({"q": truth.q, "hit": hit}).groupby("q").hit.agg(["sum", "size"])
    rec = (per_q["sum"] / per_q["size"]).to_numpy()
    n_single = n_query - len(per_q)
    ceiling = (f_beta(np.ones_like(rec), rec).sum() + n_single) / n_query
    per_s1 = pairs.groupby("q").size()
    return {
        "pair_recall": hit.mean(),
        "entities_fully_covered": (rec == 1).mean(),
        "ceiling_f05": ceiling,
        "cands_per_s1_mean": len(pairs) / n_query,
        "cands_per_s1_p95": float(np.percentile(np.r_[per_s1.to_numpy(), np.zeros(n_query - len(per_s1))], 95)),
        "reduction_ratio": 1 - len(pairs) / (n_query * n_pool),
        "n_candidates": len(pairs),
    }


def write_candidates(path: Path, s1_ids: np.ndarray, pool_ids: np.ndarray, cand: pd.DataFrame) -> None:
    """candidate_pairs.tsv: one row per S1 record (file order), ids sorted by score, empty if none."""
    c = cand.sort_values(["q", "score"], ascending=[True, False])
    lists = pd.Series(pool_ids[c.p.to_numpy()].astype(str)).groupby(c.q.to_numpy()).agg(",".join)
    col = pd.Series("", index=np.arange(len(s1_ids)))
    col[lists.index] = lists.to_numpy()
    pd.DataFrame({"source1_entity_id": s1_ids.astype(str), "candidate_entity_ids": col.to_numpy()}) \
        .to_csv(path, sep="\t", index=False)
