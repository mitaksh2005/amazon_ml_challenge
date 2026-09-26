"""Leaderboard metric and its diagnostics.

Everything is computed from counts per Source-1 entity:
  g  = true matches (complete gold set, including matches blocking never retrieved)
  c  = true matches among the candidates
  k  = predicted matches
  tp = correct predicted matches
Per-entity F0.5 = 1.25 tp / (0.25 g + k); an entity with g = k = 0 (correct singleton) scores 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def entity_f05(g, k, tp) -> np.ndarray:
    g, k, tp = (np.asarray(x, dtype=np.float64) for x in (g, k, tp))
    den = 0.25 * g + k
    return np.where(den > 0, 1.25 * tp / np.where(den > 0, den, 1.0), 1.0)


def threshold_curve(q: np.ndarray, score: np.ndarray, label: np.ndarray, g: np.ndarray):
    """Macro F0.5 for *every* global threshold, exactly, in O(M log M).

    q: entity index (0..n-1) of each candidate pair; score/label per pair; g: gold count per entity.
    Returns (thresholds, macro_f05) with thresholds descending; predicting `score >= t` gives the
    value at t. The first entry (t = +inf) predicts nothing. Entities without candidates count too.

    Pairs are added in descending score order; within one entity that order is also its own
    order, so the number of pairs (k) and true pairs (tp) it already had before each addition are
    a cumcount / cumsum and every step's change of the macro sum is vectorised."""
    n = len(g)
    base = float(np.sum(g == 0))
    if len(score) == 0:
        return np.array([np.inf]), np.array([base / max(n, 1)])
    order = np.lexsort((q, -score))
    qs, ss, ys = q[order], score[order], label[order].astype(np.int64)
    k_before = pd.Series(qs).groupby(qs).cumcount().to_numpy()
    tp_before = pd.Series(ys).groupby(qs).cumsum().to_numpy() - ys
    gq = g[qs]
    delta = entity_f05(gq, k_before + 1, tp_before + ys) - entity_f05(gq, k_before, tp_before)
    total = base + np.cumsum(delta)
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]   # evaluate after whole tie groups
    return np.r_[np.inf, ss[ends]], np.r_[base, total[ends]] / n


def best_threshold(q, score, label, g) -> tuple[float, float]:
    """Global threshold maximising macro F0.5 (highest threshold among exact ties)."""
    t, v = threshold_curve(q, score, label, g)
    i = int(np.argmax(v))
    return float(t[i]), float(v[i])


def entity_table(entities: pd.DataFrame, pairs: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Per-entity counts. `entities`: s1_entity_id, g (+ any slice columns); `pairs`: s1_entity_id,
    label, pred for the candidates that were scored. Entities without candidates get zeros."""
    grp = pairs.groupby("s1_entity_id", sort=False)
    agg = pd.DataFrame({
        "n_cand": grp.size(),
        "c": grp["label"].sum(),
        "k": grp[pred_col].sum(),
        "tp": (pairs["label"].astype(bool) & pairs[pred_col].astype(bool)).groupby(pairs["s1_entity_id"]).sum(),
    })
    e = entities.merge(agg, left_on="s1_entity_id", right_index=True, how="left")
    for col in ("n_cand", "c", "k", "tp"):
        e[col] = e[col].fillna(0).astype(np.int64)
    e["f05"] = entity_f05(e.g, e.k, e.tp)
    e["ceiling_f05"] = entity_f05(e.g, e.c, e.c)
    return e


def summary(e: pd.DataFrame) -> dict:
    """Headline metrics from an entity table (see entity_table)."""
    g, c, k, tp = (e[x].to_numpy(np.float64) for x in ("g", "c", "k", "tp"))
    single, multi = g == 0, g > 0
    n0 = max(int(single.sum()), 1)
    ent_p = np.where(k > 0, tp / np.maximum(k, 1), 1.0)
    ent_r = np.where(multi, tp / np.maximum(g, 1), np.where(k == 0, 1.0, 0.0))
    return {
        "macro_f05": float(e.f05.mean()),
        "ceiling_f05": float(e.ceiling_f05.mean()),
        "n_entities": int(len(e)),
        "singleton_rate": float(single.mean()),
        "singleton_accuracy": float((k[single] == 0).mean()) if single.any() else float("nan"),
        "fp_per_1000_singletons": float(1000 * k[single].sum() / n0),
        "affected_per_1000_singletons": float(1000 * (k[single] > 0).sum() / n0),
        "macro_f05_non_singleton": float(e.f05[multi].mean()) if multi.any() else float("nan"),
        "pair_precision": float(tp.sum() / k.sum()) if k.sum() else float("nan"),
        "pair_recall": float(tp.sum() / g.sum()) if g.sum() else float("nan"),
        "entity_precision": float(ent_p.mean()),
        "entity_recall": float(ent_r.mean()),
        "candidate_pair_recall": float(c.sum() / g.sum()) if g.sum() else float("nan"),
        "candidate_macro_recall": float((c[multi] / g[multi]).mean()) if multi.any() else float("nan"),
        "any_match_coverage": float((c[multi] > 0).mean()) if multi.any() else float("nan"),
        "all_match_coverage": float((c[multi] == g[multi]).mean()) if multi.any() else float("nan"),
        "cands_per_s1_mean": float(e.n_cand.mean()),
        "cands_per_s1_p95": float(e.n_cand.quantile(0.95)),
        "cands_per_s1_max": int(e.n_cand.max()) if len(e) else 0,
        "zero_candidate_rate": float((e.n_cand == 0).mean()),
        "pred_per_s1_mean": float(k.mean()),
    }


def error_type(e: pd.DataFrame) -> pd.Series:
    """Primary reason an entity lost score, in priority order (see plan §6.4)."""
    out = np.full(len(e), "correct", dtype=object)
    g, c, k, tp = (e[x].to_numpy() for x in ("g", "c", "k", "tp"))
    lost = e.f05.to_numpy() < 1
    out[lost & (g == 0)] = "singleton_fp"
    rest = lost & (g > 0)
    fp = k - tp
    fn_cand = c - tp
    out[rest & (c < g) & (fp == 0) & (fn_cand == 0)] = "blocking_miss"
    out[rest & (fp > 0)] = "matcher_fp"
    out[rest & (fp == 0) & (fn_cand > 0)] = "matcher_fn"
    return pd.Series(out, index=e.index)


def loss_decomposition(e: pd.DataFrame) -> dict:
    """Split 1 - macro F0.5 into stage losses. Each entity's shortfall is attributed
    sequentially: blocking miss (ceiling < 1), then matcher FN (missed retrieved matches),
    then matcher FP (wrong matches); singleton FPs are their own bucket."""
    n = len(e)
    g, c, k, tp = (e[x].to_numpy(np.float64) for x in ("g", "c", "k", "tp"))
    single = g == 0
    f_ceiling = entity_f05(g, c, c)          # perfect matcher on these candidates
    f_no_fp = entity_f05(g, tp, tp)          # drop this entity's false positives
    f = e.f05.to_numpy()
    out = {
        "singleton_fp": float((1 - f)[single].sum() / n),
        "blocking_miss": float((1 - f_ceiling)[~single].sum() / n),
        "matcher_fn": float((f_ceiling - f_no_fp)[~single].sum() / n),
        "matcher_fp": float((f_no_fp - f)[~single].sum() / n),
    }
    out["achieved"] = float(f.mean())
    return out
