"""Pair features XTRA2 (pipeline P10), computed by the Rust engine in ../rust/xfeat.

Changes over XTRA (xfeatures.py, P9), aimed at the unseen French test data and its denser decoys:

  distinctive name  joined-name / acronym / skeleton features run on name_core minus *generic* tokens:
                    tokens found in ≥ GENERIC_FRAC of the pool records of their country (data-driven,
                    so it also covers French words that never occur in training), plus a small static
                    list (fils, freres, cie, ...). "Foot & Fils SARL" vs "Aide & Fils SARL" now compare
                    "foot" with "aide". If every token is generic the full token list is kept.
  script conflict   x2_script_conflict: the two raw names are written in different Unicode scripts
                    (Latin incl. accents, Devanagari, Bengali, ...). Replaces x_native_l/r, which flagged
                    any non-ASCII character and so fired on every accented French name.
  street core       first address component without numbers and street-type words (street/rue/road/route
                    ...): a language-neutral street-name comparison.
  ordered numbers   x2_addrnum_ordered: the numeric address tokens are the same *sequence*
                    ("48-30" vs "30-48" have Jaccard 1 but are different buildings).
  genericness       x2_generic_frac_l/r: share of a name's tokens that are generic.
  density (ctx)     per S1: variance of the retrieval score, how many candidates have a near-identical
                    distinctive name or the same house number (the candidate count is already n_cand_ctx).

The Rust engine reads per-record strings (XStore2) and pair positions, and computes all pair
features in one parallel pass (rayon, every core) instead of rapidfuzz calls plus Python loops.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import blocking as B
from data import log

RUST_COLS = ["x2_join_eq", "x2_join_ratio", "x2_join_partial", "x2_acr", "x2_skel_tset", "x2_dtok_tset",
             "x2_house_eq", "x2_house_trunc", "x2_addrnum_eq", "x2_addrnum_jacc", "x2_addrnum_ordered",
             "x2_script_conflict", "x2_street_ratio", "x2_street_tset"]
XTRA2_COLS = RUST_COLS + ["x2_generic_frac_l", "x2_generic_frac_r"]
XCTX2_COLS = ["x2_house_eq_other", "x2_n_house_eq", "x2_n_join_hi", "x2_score_var", "x2_addr_c3_gap_best",
              "x2_street_gap_best", "x2_join_ratio_gap_best", "x2_dtok_gap_best"]

GENERIC_FRAC = 0.004
GENERIC_STATIC = {"fils", "freres", "cie", "compagnie", "societe", "ste", "ets", "etablissements", "entreprise",
                  "groupe", "group", "holding", "services", "service", "international", "enterprises",
                  "enterprise", "industries", "solutions", "trading", "traders", "and", "et", "sons", "brothers"}
STREET_TYPES = {"street", "road", "avenue", "boulevard", "drive", "lane", "court", "place", "highway", "parkway",
                "circle", "terrace", "trail", "square", "way", "rue", "route", "chemin", "allee", "impasse", "quai",
                "cours", "cour", "faubourg", "voie", "sentier", "terrasse", "marg", "gali", "path", "rond", "point",
                "north", "south", "east", "west", "nord", "sud", "est", "ouest", "de", "du", "des", "la", "le", "les",
                "number", "numero", "no"}
_DIGIT = re.compile(r"\d")

XFEAT_EXE = Path(os.environ.get("ER_XFEAT", Path(__file__).resolve().parent.parent / "rust" / "xfeat" / "target" /
                                "release" / ("xfeat.exe" if os.name == "nt" else "xfeat")))


def _tokens(name_core: str) -> list[str]:
    return [B.name_token(t) for t in name_core.split() if len(t) >= 2 or t.isdigit()]


def generic_tokens(pool: pd.DataFrame, frac: float = GENERIC_FRAC, chunk: int = 500_000) -> dict[str, frozenset]:
    """Per country: tokens present in at least `frac` of that country's pool records."""
    from collections import Counter
    counts, n = {}, {}
    for b in range(0, len(pool), chunk):
        part = pool.iloc[b:b + chunk]
        for c, names in part.groupby(part.country.fillna("")).name_core:
            cnt = counts.setdefault(c, Counter())
            for s in names:
                cnt.update(set(_tokens(s)))
            n[c] = n.get(c, 0) + len(names)
    return {c: frozenset(t for t, k in cnt.items() if k >= max(frac * n[c], 20)) | GENERIC_STATIC
            for c, cnt in counts.items()}


def _street_core(addr_norm: str) -> str:
    first = addr_norm.split(",", 1)[0]
    return " ".join(t for t in first.split() if not _DIGIT.search(t) and t not in STREET_TYPES)


def _addr_num_seq(addr: str) -> str:
    return " ".join(t for raw in addr.replace(",", " ").split() for t in B.addr_tokens(raw) if t.isdigit())


STORE_COLS = ["joined", "initials", "skel", "dtok", "house", "nums", "street", "raw"]


def _record_strings(df: pd.DataFrame, generic: dict[str, frozenset]) -> tuple[dict, np.ndarray]:
    empty = frozenset(GENERIC_STATIC)
    gen = df.country.fillna("").map(lambda c: generic.get(c, empty)).tolist()
    toks = df.name_core.map(_tokens).tolist()
    dist = [[x for x in tk if x not in g] or tk for tk, g in zip(toks, gen)]
    gfrac = np.array([sum(x in g for x in tk) / len(tk) if tk else np.nan for tk, g in zip(toks, gen)], np.float32)
    cols = {
        "joined": ["".join(t) for t in dist],
        "initials": ["".join(x[0] for x in t) if len(t) >= 2 else "" for t in dist],
        "skel": [" ".join(B.skeleton(x) if len(x) >= 4 and not x.isdigit() else x for x in t) for t in dist],
        "dtok": [" ".join(t) for t in dist],
        "house": df.house_no.str.extract(r"^0*(\d{1,7})", expand=False).fillna("").tolist(),
        "nums": df.addr_norm.map(_addr_num_seq).tolist(),
        "street": df.addr_norm.map(_street_core).tolist(),
        "raw": df.business_name.fillna("").tolist(),
    }
    return cols, gfrac


class XStore2:
    """Per-record derived strings for XTRA2, streamed to `d` (<col>.off u64[n + 1], <col>.dat UTF-8) in
    chunks so that ~10M test records never sit in memory as Python strings; positions = df row order."""

    def __init__(self, df: pd.DataFrame, generic: dict[str, frozenset], d: Path, chunk: int = 250_000):
        df = df.reset_index(drop=True)
        self.ids = pd.Index(df.entity_id.to_numpy())
        self.dir = Path(d)
        self.dir.mkdir(parents=True, exist_ok=True)
        gfr = []
        fh = {c: (open(self.dir / f"{c}.off", "wb"), open(self.dir / f"{c}.dat", "wb")) for c in STORE_COLS}
        pos = dict.fromkeys(STORE_COLS, 0)
        try:
            for c in STORE_COLS:
                fh[c][0].write(np.zeros(1, "<u8").tobytes())
            for b in range(0, len(df), chunk):
                cols, g = _record_strings(df.iloc[b:b + chunk], generic)
                gfr.append(g)
                for c in STORE_COLS:
                    enc = [s.encode("utf-8") for s in cols[c]]
                    off = pos[c] + np.cumsum([len(x) for x in enc], dtype=np.uint64)
                    fh[c][0].write(off.astype("<u8").tobytes())
                    fh[c][1].write(b"".join(enc))
                    pos[c] = int(off[-1]) if len(off) else pos[c]
        finally:
            for o, a in fh.values():
                o.close()
                a.close()
        self.gfrac = np.concatenate(gfr) if gfr else np.zeros(0, np.float32)


def rust_features(ldir: Path, rdir: Path, li: np.ndarray, ri: np.ndarray, work: Path) -> pd.DataFrame:
    """Run the Rust engine on (li, ri) record positions of two written stores."""
    if not XFEAT_EXE.exists():
        raise FileNotFoundError(f"{XFEAT_EXE} not built: cargo build --release in rust/xfeat")
    work.mkdir(parents=True, exist_ok=True)
    pdir = Path(tempfile.mkdtemp(prefix="pairs_", dir=work))
    try:
        np.asarray(li, "<u4").tofile(pdir / "li.u32")
        np.asarray(ri, "<u4").tofile(pdir / "ri.u32")
        r = subprocess.run([str(XFEAT_EXE), str(ldir), str(rdir), str(pdir)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"xfeat failed ({r.returncode}): {r.stderr[-2000:]}")
        log(r.stderr.strip())
        v = np.fromfile(pdir / "out.f32", "<f4").reshape(len(RUST_COLS), len(li))
    finally:
        shutil.rmtree(pdir, ignore_errors=True)
    return pd.DataFrame({c: v[k] for k, c in enumerate(RUST_COLS)})


class XEngine:
    """Two written stores (left = S1, right = pool) plus the Rust engine, for repeated pair batches."""

    def __init__(self, s1_df: pd.DataFrame, pool_df: pd.DataFrame, work: Path, generic=None):
        t0 = time.time()
        self.work = Path(work)
        generic = generic if generic is not None else generic_tokens(pool_df)
        if self.work.exists():
            shutil.rmtree(self.work)
        self.L, self.R = XStore2(s1_df, generic, self.work / "L"), XStore2(pool_df, generic, self.work / "R")
        self.ldir, self.rdir = self.L.dir, self.R.dir
        log(f"xfeat stores: {len(self.L.ids):,} left, {len(self.R.ids):,} right ({time.time() - t0:.0f}s)")

    def features(self, li: np.ndarray, ri: np.ndarray) -> pd.DataFrame:
        f = rust_features(self.ldir, self.rdir, li, ri, self.work)
        f["x2_generic_frac_l"], f["x2_generic_frac_r"] = self.L.gfrac[li], self.R.gfrac[ri]
        return f[XTRA2_COLS]

    def for_ids(self, s1_ids, cand_ids) -> pd.DataFrame:
        li = self.L.ids.get_indexer(pd.Index(s1_ids))
        ri = self.R.ids.get_indexer(pd.Index(cand_ids))
        if (li < 0).any() or (ri < 0).any():
            raise KeyError("pair ids missing from the xfeat stores")
        return self.features(li, ri)


def add_xctx2(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate-set context for XTRA2; `df` is sorted by entity (pipelines.add_ret_ctx)."""
    g = df.s1_entity_id
    heq = (df.x2_house_eq == 1).astype(np.float32)
    n_heq = heq.groupby(g).transform("sum")
    df["x2_n_house_eq"] = n_heq.astype(np.float32)
    df["x2_house_eq_other"] = (n_heq - heq > 0).astype(np.float32)
    df["x2_n_join_hi"] = (df.x2_join_ratio.fillna(0) >= 0.9).groupby(g).transform("sum").astype(np.float32)
    df["x2_score_var"] = df.score.groupby(g).transform("var").fillna(0).astype(np.float32)
    for col, name in (("addr_char3_cos", "x2_addr_c3_gap_best"), ("x2_street_ratio", "x2_street_gap_best"),
                      ("x2_join_ratio", "x2_join_ratio_gap_best"), ("x2_dtok_tset", "x2_dtok_gap_best")):
        v = df[col].fillna(-1)
        df[name] = (v - v.groupby(g).transform("max")).astype(np.float32)
    return df


# ---- French-structured address augmentation --------------------------------------------------------
FR_STREET = {"street": "rue", "road": "route", "drive": "allee", "lane": "chemin", "court": "cour",
             "highway": "route nationale", "parkway": "cours", "circle": "rond point", "terrace": "terrasse",
             "trail": "sentier", "way": "voie", "avenue": "avenue", "boulevard": "boulevard", "place": "place",
             "square": "square", "marg": "rue", "gali": "ruelle"}
FR_WORDS = {"suite": "bureau", "floor": "etage", "apartment": "appartement", "building": "batiment",
            "north": "nord", "south": "sud", "east": "est", "west": "ouest", "near": "pres", "opposite": "face",
            "number": "numero", "first": "premier", "ground": "rez", "and": "et", "of": "de", "the": "le"}


def frenchify_addr(addr_norm: str) -> str:
    """'1450 main street, springfield, illinois' → '1450 rue main, springfield, illinois': French street
    words and the French order number–type–name, so the matcher cannot lean on English street words."""
    out = []
    for comp in addr_norm.split(", "):
        t = comp.split()
        if not t:
            continue
        typ = next((i for i in range(len(t) - 1, -1, -1) if t[i] in FR_STREET), None)
        if typ is not None and typ > 0:
            head = [x for x in t[:typ] if _DIGIT.search(x)] if _DIGIT.search(t[0]) else []
            name = [x for x in t[:typ] if x not in head]
            t = head + [FR_STREET[t[typ]]] + name + t[typ + 1:]
        out.append(" ".join(FR_WORDS.get(x, x) for x in t))
    return ", ".join(out)
