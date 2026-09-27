"""Extra pair features (family XTRA), added after the P5-Bwide output audit (RESULTS_AND_PIPELINE.md §5-6).

  joined name    domain-style names: "bronaughseries.com" ↔ "Bronaugh Series" (the TXT domain features
                 only compared domain to domain and had zero gain)
  acronym        "HGR" ↔ "Holloman, Groseclose and Ramirez"
  skeleton       consonant-skeleton token-set similarity: robust to transliteration of native-script names
  house number   canonical equality (leading zeros stripped: 003905 = 3905) and truncation (4806 ↔ 806,
                 841 ↔ 84): the most confident false positives were near-copies differing only here
  native script  whether either raw name was non-ASCII (so its string similarity is a transliteration)

plus candidate-set context (computed in pipelines.add_ret_ctx, per S1): does *another* candidate match the
house number exactly, and address / street / joined-name / skeleton gaps to the best candidate.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import blocking as B

XTRA_COLS = ["x_join_eq", "x_join_ratio", "x_join_partial", "x_acr", "x_skel_tset", "x_house_eq",
             "x_house_trunc", "x_addrnum_eq", "x_addrnum_jacc", "x_native_l", "x_native_r"]
XCTX_COLS = ["x_house_eq_other", "x_addr_c3_gap_best", "x_street_gap_best", "x_join_ratio_gap_best",
             "x_skel_gap_best"]
NON_ASCII = re.compile(r"[^\x00-\x7f]")
HOUSE_RE = re.compile(r"^0*(\d{1,7})")


def _tokens(name_core: str) -> list[str]:
    return [B.name_token(t) for t in name_core.split() if len(t) >= 2 or t.isdigit()]


def _skel(toks: list[str]) -> str:
    return " ".join(B.skeleton(t) if len(t) >= 4 and not t.isdigit() else t for t in toks)


def _addr_nums(addr: str) -> frozenset:
    return frozenset(t for raw in addr.replace(",", " ").split() for t in B.addr_tokens(raw) if t.isdigit())


class XStore:
    """Per-record derived strings for the XTRA features, addressable by position."""

    def __init__(self, df: pd.DataFrame):
        toks = df.name_core.map(_tokens)
        self.joined = toks.map("".join).to_numpy(object)
        self.initials = toks.map(lambda t: "".join(x[0] for x in t) if len(t) >= 2 else "").to_numpy(object)
        self.skel = toks.map(_skel).to_numpy(object)
        self.house = df.house_no.str.extract(HOUSE_RE, expand=False).fillna("").to_numpy(object)
        self.nums = df.addr_norm.map(_addr_nums).to_numpy(object)
        self.native = df.business_name.str.contains(NON_ASCII).to_numpy(np.float32)
        self.ids = pd.Index(df.entity_id.to_numpy())


def _rf(scorer, a, b, both) -> np.ndarray:
    v = cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / np.float32(100)
    v[~both] = np.nan
    return v


def xtra_features(L: XStore, R: XStore, li: np.ndarray, ri: np.ndarray, chunk: int = 1_000_000) -> pd.DataFrame:
    parts = []
    for b in range(0, len(li), chunk):
        parts.append(_chunk(L, R, li[b:b + chunk], ri[b:b + chunk]))
    if not parts:
        return pd.DataFrame({c: np.zeros(0, np.float32) for c in XTRA_COLS})
    return pd.concat(parts, ignore_index=True)


def _chunk(L: XStore, R: XStore, li: np.ndarray, ri: np.ndarray) -> pd.DataFrame:
    f = {}
    jl, jr = L.joined[li], R.joined[ri]
    both = (np.char.str_len(jl.astype(str)) > 0) & (np.char.str_len(jr.astype(str)) > 0)
    f["x_join_eq"] = np.where(both, (jl == jr).astype(np.float32), np.nan)
    f["x_join_ratio"] = _rf(fuzz.ratio, jl, jr, both)
    f["x_join_partial"] = _rf(fuzz.partial_ratio, jl, jr, both)
    il, ir = L.initials[li], R.initials[ri]
    has = (il != "") | (ir != "")
    acr = ((il != "") & (jr == il)) | ((ir != "") & (jl == ir))
    f["x_acr"] = np.where(has & both, acr.astype(np.float32), np.nan)
    sl, sr = L.skel[li], R.skel[ri]
    f["x_skel_tset"] = _rf(fuzz.token_set_ratio, sl, sr, (sl != "") & (sr != ""))
    hl, hr = L.house[li], R.house[ri]
    hb = (hl != "") & (hr != "")
    f["x_house_eq"] = np.where(hb, (hl == hr).astype(np.float32), np.nan)
    trunc = np.fromiter((a != b and (a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a))
                         for a, b in zip(hl, hr)), bool, len(hl))
    f["x_house_trunc"] = np.where(hb, trunc.astype(np.float32), np.nan)
    nl, nr = L.nums[li], R.nums[ri]
    eq = np.full(len(li), np.nan, np.float32)
    jac = np.full(len(li), np.nan, np.float32)
    for i, (x, y) in enumerate(zip(nl, nr)):
        if x and y:
            eq[i] = float(x == y)
            jac[i] = len(x & y) / len(x | y)
    f["x_addrnum_eq"], f["x_addrnum_jacc"] = eq, jac
    f["x_native_l"], f["x_native_r"] = L.native[li], R.native[ri]
    return pd.DataFrame({c: np.asarray(f[c], np.float32) for c in XTRA_COLS})


def add_xctx(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate-set context for the XTRA features; `df` is sorted by entity (add_ret_ctx)."""
    g = df.s1_entity_id
    heq = (df.x_house_eq == 1).astype(np.float32)
    df["x_house_eq_other"] = (heq.groupby(g).transform("sum") - heq > 0).astype(np.float32)
    for col, name in (("addr_char3_cos", "x_addr_c3_gap_best"), ("street_ratio", "x_street_gap_best"),
                      ("x_join_ratio", "x_join_ratio_gap_best"), ("x_skel_tset", "x_skel_gap_best")):
        v = df[col].fillna(-1)
        df[name] = (v - v.groupby(g).transform("max")).astype(np.float32)
    return df


def for_pairs(s1_df: pd.DataFrame, pool_df: pd.DataFrame, s1_ids, cand_ids) -> pd.DataFrame:
    """XTRA features aligned with (s1_ids, cand_ids); record frames as returned by data.load_records."""
    L, R = XStore(s1_df), XStore(pool_df)
    li, ri = L.ids.get_indexer(pd.Index(s1_ids)), R.ids.get_indexer(pd.Index(cand_ids))
    if (li < 0).any() or (ri < 0).any():
        raise KeyError("pair ids missing from the record frames")
    return xtra_features(L, R, li, ri)
