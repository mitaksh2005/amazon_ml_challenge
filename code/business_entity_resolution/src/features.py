"""Pair features (plan §4, feature families C).

Only name, address and country exist in this dataset, so the report's phone/URL/geo/category
features are replaced by what can be read from the text itself (numbers, legal forms, phone
numbers and domains embedded in names). Families:

  STR    string similarity of names (edit, Jaro/JW, token-set, char 3-gram cosine, ...)
  RARE   token rarity: IDF-weighted Jaccard, shared/unshared rare tokens, name frequency
  NUM    numbers in names and addresses: equality / conflict (Hotel21 vs Hotel12, house numbers)
  ADDR   address similarity, region and postcode agreement / conflict
  LEGAL  legal-form agreement / conflict (llc vs private limited)
  TXT    phone numbers / domains embedded in the raw text
  RET    retrieval scores and ranks from blocking          (computed per blocking config, pipelines.py)
  CTX    candidate-set context: rank / gap to best / counts (computed per blocking config, pipelines.py)

IDF statistics come from the S2+S3 pool of the same split (label-free, like blocking), scoped by
country through the token hash, so a token common in India can still be rare in the US.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein, Jaro, JaroWinkler, Levenshtein, Prefix
from rapidfuzz.process import cpdist

import cleaning as C

TOK_BITS = 22
N_TOK = 1 << TOK_BITS
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
PHONE_RE = "(" + C.PHONE_RE + ")"

FAMILIES = {
    "STR": ["name_ratio", "name_lev", "name_dl", "name_jaro", "name_jw", "name_tset", "name_tsort",
            "name_partial", "name_prefix", "name_len_ratio", "name_norm_ratio", "name_char3_cos",
            "name_empty_l", "name_empty_r", "is_s3"],
    "RARE": ["name_tok_jacc", "name_tok_contain", "name_wjacc", "name_shared_idf_sum", "name_shared_idf_max",
             "name_miss_idf_max_l", "name_miss_idf_max_r", "name_freq_l", "name_freq_r",
             "name_ntok_l", "name_ntok_r"],
    "NUM": ["name_num_state", "name_num_eq", "name_num_conflict", "name_num_jacc", "addr_num_jacc",
            "addr_num_conflict", "house_eq", "house_conflict", "house_absdiff", "conflict_count"],
    "ADDR": ["addr_ratio", "addr_tset", "addr_partial", "addr_char3_cos", "addr_tok_jacc", "addr_wjacc",
             "addr_shared_idf_max", "addr_miss_idf_max_l", "addr_miss_idf_max_r", "street_ratio",
             "region_eq", "region_conflict", "addr_state", "postcode_eq", "postcode_conflict"],
    "LEGAL": ["legal_state", "legal_eq", "legal_conflict"],
    "TXT": ["phone_eq", "phone_conflict", "domain_eq", "domain_conflict"],
}
PAIR_FEATURES = [c for fam in FAMILIES.values() for c in fam]


# ---- record-level preprocessing ---------------------------------------------------------------------
def _hash_tokens(tokens: np.ndarray, country: np.ndarray) -> np.ndarray:
    th = pd.util.hash_array(tokens.astype(object), categorize=True)
    ch = pd.util.hash_array(country.astype(object), categorize=True)
    return ((th ^ (ch * _GOLDEN)) & np.uint64(N_TOK - 1)).astype(np.int64)


def token_matrix(text: pd.Series, country: pd.Series, batch: int = 500_000) -> sp.csr_matrix:
    """Binary (record × hashed country|token) matrix, built in batches to bound memory."""
    mats = []
    text = text.reset_index(drop=True)
    country = country.reset_index(drop=True)
    for b in range(0, len(text), batch):
        t = text.iloc[b:b + batch].str.replace(",", " ", regex=False).str.split()
        ex = t.explode()
        ex = ex[ex.notna() & (ex.astype(str).str.len() > 0)]
        rows = ex.index.to_numpy() - b
        cols = _hash_tokens(ex.to_numpy(object), country.iloc[b:b + batch].to_numpy(object)[rows]) \
            if len(ex) else np.zeros(0, np.int64)
        m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(t), N_TOK))
        m.sum_duplicates()
        m.data[:] = 1
        mats.append(m)
    return sp.vstack(mats).tocsr() if mats else sp.csr_matrix((0, N_TOK), dtype=np.float32)


def _name_hash(name: pd.Series, country: pd.Series) -> np.ndarray:
    return _hash_tokens(name.to_numpy(object), country.to_numpy(object))


class PoolIDF:
    """Document frequencies of hashed tokens and exact names over the S2+S3 pool (all records,
    even those that never become candidates)."""

    def __init__(self, pool_df: pd.DataFrame):
        country = pool_df.country.reset_index(drop=True)
        self.n = country.value_counts().to_dict()
        self.n_total = int(sum(self.n.values()))
        self.df = {}
        for f, col in (("name", "name_core"), ("addr", "addr_norm")):
            M = token_matrix(pool_df[col], country)
            self.df[f] = np.asarray(M.sum(axis=0)).ravel().astype(np.float32)
            del M
        self.name_counts = pd.Series(_name_hash(pool_df.name_core, pool_df.country)).value_counts()

    def weigh(self, M: sp.csr_matrix, field: str, country: np.ndarray) -> sp.csr_matrix:
        n_c = pd.Series(country).map(self.n).fillna(self.n_total).to_numpy(np.float32)
        rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
        W = M.copy()
        W.data = np.log((n_c[rows] + 1) / (self.df[field][M.indices] + 1)).astype(np.float32)
        return W

    def name_freq(self, h: np.ndarray, empty: np.ndarray) -> np.ndarray:
        f = np.log1p(self.name_counts.reindex(h).fillna(0).to_numpy(np.float32))
        f[empty] = np.nan
        return f


class RecordStore:
    """Column arrays + token matrices for one set of records, addressable by entity id."""

    def __init__(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        self.ids = pd.Index(df.entity_id.to_numpy())
        self.country = df.country.to_numpy(object)
        self.country_counts = df.country.value_counts().to_dict()
        self.name = df.name_core.to_numpy(object)
        self.name_norm = df.name_norm.to_numpy(object)
        self.addr = df.addr_norm.to_numpy(object)
        self.street = df.addr_norm.str.split(",").str[0].fillna("").to_numpy(object)
        self.legal = df.legal_form.to_numpy(object)
        self.region = df.region.to_numpy(object)
        self.postcode = df.postcode.to_numpy(object)
        self.house = df.house_no.to_numpy(object)
        self.house_int = pd.to_numeric(df.house_no.str.extract(r"^(\d{1,7})", expand=False),
                                       errors="coerce").to_numpy(np.float64)
        self.name_len = df.name_core.str.len().to_numpy(np.float32)
        self.addr_len = df.addr_norm.str.len().to_numpy(np.float32)
        self.name_nums = df.name_core.str.findall(r"\d+").map(frozenset).to_numpy(object)
        self.addr_nums = df.addr_norm.str.findall(r"\d+[a-z]?").map(frozenset).to_numpy(object)
        raw = (df.business_name + " " + df.business_address).str.lower()
        ph = raw.str.extract(PHONE_RE, expand=False).fillna("").str.replace(r"\D", "", regex=True).str[-10:]
        self.phone = ph.where(ph.str.len() >= 7, "").to_numpy(object)
        self.domain = df.business_name.str.lower().str.extract(C.DOMAIN_RE, expand=False).fillna("").to_numpy(object)
        self.name_hash = _name_hash(df.name_core, df.country)
        self.name_B = token_matrix(df.name_core, df.country)
        self.addr_B = token_matrix(df.addr_norm, df.country)
        self.name_W = self.addr_W = None
        self.name_freq = None

    def apply_idf(self, idf: PoolIDF) -> "RecordStore":
        self.name_W = idf.weigh(self.name_B, "name", self.country)
        self.addr_W = idf.weigh(self.addr_B, "addr", self.country)
        self.name_freq = idf.name_freq(self.name_hash, self.name_len == 0)
        return self

    def rows(self, ids) -> np.ndarray:
        r = self.ids.get_indexer(pd.Index(np.asarray(ids)))
        if (r < 0).any():
            raise KeyError(f"{int((r < 0).sum())} ids not in record store, e.g. {np.asarray(ids)[r < 0][:3]}")
        return r


# ---- pair features ------------------------------------------------------------------------------------
def _rowmax(M: sp.csr_matrix) -> np.ndarray:
    M.eliminate_zeros()
    out = np.full(M.shape[0], np.nan, np.float32)
    nz = np.diff(M.indptr) > 0
    if nz.any():
        out[nz] = np.maximum.reduceat(M.data, M.indptr[:-1][nz])
    return out


def _overlap(WL, WR, BL, BR, li, ri, prefix: str) -> dict:
    A, Bw = WL[li], WR[ri]
    Ab, Bb = BL[li], BR[ri]
    SA = A.multiply(Bb).tocsr()                     # shared tokens, weighted by their IDF
    SA.eliminate_zeros()
    n_sh = np.diff(SA.indptr).astype(np.float32)
    w_sh = np.asarray(SA.sum(axis=1)).ravel()
    nA, nB = np.diff(A.indptr).astype(np.float32), np.diff(Bw.indptr).astype(np.float32)
    WA, WB = np.asarray(A.sum(axis=1)).ravel(), np.asarray(Bw.sum(axis=1)).ravel()
    none = (nA == 0) | (nB == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = n_sh / (nA + nB - n_sh)
        contain = n_sh / np.minimum(nA, nB)
        wjacc = w_sh / (WA + WB - w_sh)
    out = {f"{prefix}_tok_jacc": jacc, f"{prefix}_tok_contain": contain, f"{prefix}_wjacc": wjacc,
           f"{prefix}_shared_idf_sum": w_sh, f"{prefix}_shared_idf_max": _rowmax(SA),
           f"{prefix}_miss_idf_max_l": _rowmax((A - SA).tocsr()),
           f"{prefix}_miss_idf_max_r": _rowmax((Bw - Bw.multiply(Ab)).tocsr())}
    for k, v in out.items():
        v = np.asarray(v, np.float32)
        v[none] = np.nan
        if k.endswith(("_miss_idf_max_l", "_miss_idf_max_r")):
            v[~none & np.isnan(v)] = 0.0          # nothing unshared on that side
        if k.endswith("_shared_idf_max"):
            v[~none & np.isnan(v)] = 0.0
        out[k] = v
    return out


def _rf(scorer, a, b, both, scale=1.0) -> np.ndarray:
    v = cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / np.float32(scale)
    v[~both] = np.nan
    return v


def char3_matrix(strings) -> sp.csr_matrix:
    """L2-normalised counts of byte trigrams of " " + s + " " (24-bit codes, no hashing collisions),
    vectorised over the UTF-8 buffer of a pyarrow array: millions of strings per second."""
    import pyarrow as pa
    arr = pa.array([" " + x + " " for x in strings], type=pa.string())
    n = len(arr)
    off = np.frombuffer(arr.buffers()[1], dtype=np.int32)[arr.offset: arr.offset + n + 1].astype(np.int64)
    data = np.frombuffer(arr.buffers()[2], dtype=np.uint8) if arr.buffers()[2] is not None else np.zeros(0, np.uint8)
    n_tri = np.maximum(np.diff(off) - 2, 0)
    total = int(n_tri.sum())
    rows = np.repeat(np.arange(n), n_tri)
    start = np.repeat(off[:-1] - (np.cumsum(n_tri) - n_tri), n_tri) + np.arange(total)
    code = (data[start].astype(np.int64) << 16) | (data[start + 1].astype(np.int64) << 8) | data[start + 2]
    M = sp.csr_matrix((np.ones(total, np.float32), (rows, code)), shape=(n, 1 << 24))
    M.sum_duplicates()
    norm = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    M.data /= np.repeat(np.maximum(norm, 1e-9), np.diff(M.indptr)).astype(np.float32)
    return M


def _char_cos(a, b, both) -> np.ndarray:
    ia, ua = pd.factorize(pd.Series(a, dtype=object))
    ib, ub = pd.factorize(pd.Series(b, dtype=object))
    XL, XR = char3_matrix(ua)[ia], char3_matrix(ub)[ib]
    v = np.asarray(XL.multiply(XR).sum(axis=1)).ravel().astype(np.float32)
    v[~both] = np.nan
    return v


def _eq_conflict(a, b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(state 0 both/1 left only/2 right only/3 neither, equal, conflict); NaN where not comparable."""
    pa, pb = (a != ""), (b != "")
    state = np.where(pa & pb, 0, np.where(pa, 1, np.where(pb, 2, 3))).astype(np.float32)
    both = pa & pb
    eq = np.where(both, (a == b).astype(np.float32), np.nan).astype(np.float32)
    return state, eq, np.where(both, 1 - eq, np.nan).astype(np.float32)


def _set_feats(a, b):
    n = len(a)
    state = np.zeros(n, np.float32)
    eq, conf, jac = (np.full(n, np.nan, np.float32) for _ in range(3))
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            inter = len(x & y)
            state[i] = 2
            eq[i] = float(x == y)
            conf[i] = float(inter == 0)
            jac[i] = inter / len(x | y)
        elif x or y:
            state[i] = 1
    return state, eq, conf, jac


def _chunk(L: RecordStore, R: RecordStore, li: np.ndarray, ri: np.ndarray) -> dict:
    f = {}
    nl, nr = L.name[li], R.name[ri]
    el, er = L.name_len[li] == 0, R.name_len[ri] == 0
    both = ~(el | er)
    f["name_ratio"] = _rf(fuzz.ratio, nl, nr, both, 100)
    f["name_lev"] = _rf(Levenshtein.normalized_similarity, nl, nr, both)
    f["name_dl"] = _rf(DamerauLevenshtein.normalized_similarity, nl, nr, both)
    f["name_jaro"] = _rf(Jaro.normalized_similarity, nl, nr, both)
    f["name_jw"] = _rf(JaroWinkler.normalized_similarity, nl, nr, both)
    f["name_tset"] = _rf(fuzz.token_set_ratio, nl, nr, both, 100)
    f["name_tsort"] = _rf(fuzz.token_sort_ratio, nl, nr, both, 100)
    f["name_partial"] = _rf(fuzz.partial_ratio, nl, nr, both, 100)
    f["name_prefix"] = _rf(Prefix.normalized_similarity, nl, nr, both)
    ll, lr = L.name_len[li], R.name_len[ri]
    with np.errstate(divide="ignore", invalid="ignore"):
        f["name_len_ratio"] = np.where(both, np.minimum(ll, lr) / np.maximum(ll, lr), np.nan).astype(np.float32)
    f["name_norm_ratio"] = _rf(fuzz.ratio, L.name_norm[li], R.name_norm[ri],
                               (L.name_norm[li] != "") & (R.name_norm[ri] != ""), 100)
    f["name_char3_cos"] = _char_cos(nl, nr, both)
    f["name_empty_l"], f["name_empty_r"] = el.astype(np.float32), er.astype(np.float32)
    f["is_s3"] = np.char.startswith(R.ids.to_numpy()[ri].astype(str), "S3").astype(np.float32)

    f.update(_overlap(L.name_W, R.name_W, L.name_B, R.name_B, li, ri, "name"))
    f["name_freq_l"], f["name_freq_r"] = L.name_freq[li], R.name_freq[ri]
    f["name_ntok_l"] = np.diff(L.name_B.indptr)[li].astype(np.float32)
    f["name_ntok_r"] = np.diff(R.name_B.indptr)[ri].astype(np.float32)

    f["name_num_state"], f["name_num_eq"], f["name_num_conflict"], f["name_num_jacc"] = \
        _set_feats(L.name_nums[li], R.name_nums[ri])
    _, _, f["addr_num_conflict"], f["addr_num_jacc"] = _set_feats(L.addr_nums[li], R.addr_nums[ri])
    _, f["house_eq"], f["house_conflict"] = _eq_conflict(L.house[li], R.house[ri])
    f["house_absdiff"] = np.log1p(np.abs(L.house_int[li] - R.house_int[ri])).astype(np.float32)

    al, ar = L.addr[li], R.addr[ri]
    aboth = (L.addr_len[li] > 0) & (R.addr_len[ri] > 0)
    f["addr_ratio"] = _rf(fuzz.ratio, al, ar, aboth, 100)
    f["addr_tset"] = _rf(fuzz.token_set_ratio, al, ar, aboth, 100)
    f["addr_partial"] = _rf(fuzz.partial_ratio, al, ar, aboth, 100)
    f["addr_char3_cos"] = _char_cos(al, ar, aboth)
    ov = _overlap(L.addr_W, R.addr_W, L.addr_B, R.addr_B, li, ri, "addr")
    for k in ("addr_tok_jacc", "addr_wjacc", "addr_shared_idf_max", "addr_miss_idf_max_l", "addr_miss_idf_max_r"):
        f[k] = ov[k]
    sl, sr = L.street[li], R.street[ri]
    f["street_ratio"] = _rf(fuzz.ratio, sl, sr, (sl != "") & (sr != ""), 100)
    _, f["region_eq"], f["region_conflict"] = _eq_conflict(L.region[li], R.region[ri])
    f["addr_state"], _, _ = _eq_conflict(np.where(L.addr_len[li] > 0, "x", ""), np.where(R.addr_len[ri] > 0, "x", ""))
    _, f["postcode_eq"], f["postcode_conflict"] = _eq_conflict(L.postcode[li], R.postcode[ri])
    f["conflict_count"] = np.nansum(np.vstack([f["house_conflict"], f["name_num_conflict"],
                                               f["postcode_conflict"], f["addr_num_conflict"]]), axis=0).astype(np.float32)

    f["legal_state"], f["legal_eq"], f["legal_conflict"] = _eq_conflict(L.legal[li], R.legal[ri])
    _, f["phone_eq"], f["phone_conflict"] = _eq_conflict(L.phone[li], R.phone[ri])
    _, f["domain_eq"], f["domain_conflict"] = _eq_conflict(L.domain[li], R.domain[ri])
    return f


def pair_features(L: RecordStore, R: RecordStore, s1_ids, cand_ids, chunk: int = 500_000,
                  log=None) -> pd.DataFrame:
    """Feature frame (PAIR_FEATURES columns, float32) aligned with the given pairs."""
    li, ri = L.rows(s1_ids), R.rows(cand_ids)
    parts = []
    for b in range(0, len(li), chunk):
        f = _chunk(L, R, li[b:b + chunk], ri[b:b + chunk])
        parts.append(pd.DataFrame({c: np.asarray(f[c], np.float32) for c in PAIR_FEATURES}))
        if log:
            log(f"features: {min(b + chunk, len(li)):,}/{len(li):,} pairs")
    if not parts:
        return pd.DataFrame({c: np.zeros(0, np.float32) for c in PAIR_FEATURES})
    return pd.concat(parts, ignore_index=True)


def build_stores(s1_df: pd.DataFrame, pool_df: pd.DataFrame, keep_pool_ids=None) -> tuple[RecordStore, RecordStore]:
    """IDF from the whole pool; the pool store itself only for `keep_pool_ids` (the candidates)."""
    idf = PoolIDF(pool_df)
    if keep_pool_ids is not None:
        pool_df = pool_df[pool_df.entity_id.isin(pd.Index(keep_pool_ids))]
    return RecordStore(s1_df).apply_idf(idf), RecordStore(pool_df).apply_idf(idf)
