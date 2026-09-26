"""Candidate pipelines (plan §4–5): configuration, leakage-safe fitting and out-of-fold evaluation.

A pipeline = blocking config × feature families × matcher (+ params) × negatives × calibration ×
set policy × graph refinement. Every pipeline is evaluated on the same `dev` sample with the same
5 grouped folds; inside each training fold the entities are split again into
  fit (70%)  → matcher (10% of it held out for early stopping)
  cal (15%)  → calibrator + singleton model
  tune (15%) → set policy / thresholds
so no score used by a later stage was produced by a model trained on the same labels.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import models as M
from data import PASS_NAMES, Paths, blocking_configs, log, truncate
from er_eval import Registry, bootstrap_mean, entity_table, error_type, loss_decomposition, summary, threshold_curve
from features import FAMILIES

RET_COLS = ["score", *[f"score_{p}" for p in PASS_NAMES], *[f"rr_{p}" for p in PASS_NAMES], "rrf", "n_passes"]
CTX_COLS = ["n_cand_ctx", "score_rank", "score_gap_best", "score_top2_gap", "score_z", "name_ratio_rank",
            "name_ratio_gap_best", "n_name_hi", "wjacc_gap_best", "addr_ratio_gap_best"]
DENSE_COLS = ["dense_cos", "dense_rank", "dense_gap_best"]
FULL = ("STR", "RARE", "NUM", "ADDR", "LEGAL", "TXT", "RET", "CTX")
KEEP_PAIR_COLS = ["s1_entity_id", "cand_entity_id", "label", "fold", "score", "raw", "p", "q", "pred",
                  "conflict_count", "is_s3", "name_ratio", "name_freq_l"]


@dataclass
class PipelineConfig:
    name: str
    parent: str | None = None
    kind: str = "model"                  # empty | oracle | rule | model
    blocking: str = "B0"
    families: tuple = ("STR", "RET")
    matcher: str = "lgbm"                # lr | lgbm | xgb | cat
    params: str | dict = "default"       # "default" | "hpo" | explicit dict
    negatives: str = "E0"                # E0 all | E1 capped 12/anchor | E2 all + singleton-negative weight
    singleton_weight: float = 3.0
    calibration: str = "isotonic"        # none | platt | isotonic | beta
    policy: str = "A"                    # A threshold | B singleton gate | C gate + ambiguity/conflict
    graph: bool = False                  # H1 unique assignment of external records
    dense: bool = False                  # add frozen bi-encoder similarity (neural.py)
    description: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["families"] = list(self.families)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["families"] = tuple(d.get("families", ()))
        return cls(**d)


# ---- workspace: the prepared data every pipeline reads -------------------------------------------------
class Workspace:
    def __init__(self, paths: Paths):
        self.paths = paths
        self.samples = pd.read_parquet(paths.art / "samples.parquet")
        self._pairs: dict[str, pd.DataFrame] = {}

    def entities(self, role: str) -> pd.DataFrame:
        return self.samples[self.samples.role == role].sort_values("s1_entity_id").reset_index(drop=True)

    def wide_pairs(self, role: str) -> pd.DataFrame:
        if role not in self._pairs:
            df = pq.read_table(self.paths.art / f"pairs_{role}.parquet").to_pandas()
            if (self.paths.art / f"dense_{role}.parquet").exists():
                d = pd.read_parquet(self.paths.art / f"dense_{role}.parquet")
                df = df.merge(d, on=["s1_entity_id", "cand_entity_id"], how="left")
            self._pairs[role] = df
        return self._pairs[role]

    def blocking(self, name: str) -> dict:
        return blocking_configs(self.paths)[name]

    def hpo_params(self, lib: str) -> dict | None:
        f = self.paths.exp / "hpo" / lib / "best_params.json"
        return json.loads(f.read_text())["params"] if f.exists() else None

    def hpo_best_lib(self) -> str | None:
        vals = {}
        for lib in ("lgbm", "xgb", "cat"):
            f = self.paths.exp / "hpo" / lib / "best_params.json"
            if f.exists():
                vals[lib] = json.loads(f.read_text())["robust_score"]
        return max(vals, key=vals.get) if vals else None

    def pairs_for(self, cfg: PipelineConfig, role: str) -> pd.DataFrame:
        """Candidates of `role` under the pipeline's blocking config, with RET/CTX features."""
        df = truncate(self.wide_pairs(role), self.blocking(cfg.blocking))
        return add_ret_ctx(df, dense="dense_cos" in df.columns)


def add_ret_ctx(df: pd.DataFrame, dense: bool = False) -> pd.DataFrame:
    df = df.sort_values(["s1_entity_id", "score"], ascending=[True, False], kind="stable").reset_index(drop=True)
    rrf = np.zeros(len(df))
    for p in PASS_NAMES:
        r = df[f"rank_{p}"].to_numpy()
        df[f"rr_{p}"] = np.where(np.isnan(r), 0.0, 1.0 / np.where(np.isnan(r), 1, r)).astype(np.float32)
        rrf += np.where(np.isnan(r), 0.0, 1.0 / (60 + np.nan_to_num(r)))
    df["rrf"] = rrf.astype(np.float32)
    g = df.groupby("s1_entity_id", sort=False)
    df["n_cand_ctx"] = g["score"].transform("size").astype(np.float32)
    df["score_rank"] = g.cumcount().astype(np.float32) + 1           # rows sorted by score within entity
    best = g["score"].transform("max")
    df["score_gap_best"] = (df.score - best).astype(np.float32)
    second = df.score.where(df.score_rank == 2).groupby(df.s1_entity_id).transform("max")
    df["score_top2_gap"] = (best - second.fillna(0)).astype(np.float32)
    std = g["score"].transform("std").fillna(0)
    df["score_z"] = ((df.score - g["score"].transform("mean")) / (std + 1e-6)).clip(-10, 10).astype(np.float32)
    for col, name in (("name_ratio", "name_ratio"), ("name_wjacc", "wjacc"), ("addr_ratio", "addr_ratio")):
        v = df[col].fillna(-1)
        mx = v.groupby(df.s1_entity_id).transform("max")
        df[f"{name}_gap_best"] = (v - mx).astype(np.float32)
        if name == "name_ratio":
            df["name_ratio_rank"] = v.groupby(df.s1_entity_id).rank(ascending=False, method="min").astype(np.float32)
            df["n_name_hi"] = (v >= 0.9).groupby(df.s1_entity_id).transform("sum").astype(np.float32)
    if dense:
        v = df.dense_cos.fillna(-1)
        df["dense_rank"] = v.groupby(df.s1_entity_id).rank(ascending=False, method="min").astype(np.float32)
        df["dense_gap_best"] = (v - v.groupby(df.s1_entity_id).transform("max")).astype(np.float32)
    return df


def feature_columns(cfg: PipelineConfig) -> list[str]:
    cols = []
    for fam in cfg.families:
        cols += RET_COLS if fam == "RET" else CTX_COLS if fam == "CTX" else FAMILIES[fam]
    if cfg.dense:
        cols += DENSE_COLS
    return list(dict.fromkeys(cols))


def resolve_params(cfg: PipelineConfig, ws: Workspace | None) -> dict:
    if isinstance(cfg.params, dict):
        return cfg.params
    if cfg.params == "hpo" and ws is not None:
        p = ws.hpo_params(cfg.matcher)
        if p is None:
            log(f"[{cfg.name}] no HPO result for {cfg.matcher}: using defaults")
        return p or {}
    return {}


# ---- fitting ------------------------------------------------------------------------------------------
@dataclass
class Bundle:
    cfg: PipelineConfig
    features: list
    model: M.Matcher | None
    calibrator: M.Calibrator | None
    calibrators: dict
    singleton: M.SingletonModel | None
    policy: dict
    timings: dict


def split_entities(ids: np.ndarray, seed: int, fracs=(0.70, 0.15, 0.15)) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)[rng.permutation(len(ids))]
    a = int(round(fracs[0] * len(ids)))
    b = a + int(round(fracs[1] * len(ids)))
    return [ids[:a], ids[a:b], ids[b:]]


def training_rows(df: pd.DataFrame, cfg: PipelineConfig, g_of: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row mask and sample weights for the matcher's training set (negative curriculum, axis E)."""
    y = df.label.to_numpy()
    w = np.ones(len(df), np.float32)
    if cfg.negatives == "E1":
        rng = np.random.default_rng(seed)
        hard = (df.score_rank <= 8).to_numpy()   # rows are score-sorted within entity (add_ret_ctx)
        rand = rng.random(len(df)) < np.clip(4.0 / np.maximum(df.n_cand_ctx.to_numpy() - 8, 1), 0, 1)
        return (y == 1) | hard | rand, w
    if cfg.negatives == "E2":
        single = df.s1_entity_id.map(g_of).to_numpy() == 0
        w[single & (y == 0)] = cfg.singleton_weight
    return np.ones(len(df), bool), w


def fit_bundle(cfg: PipelineConfig, df: pd.DataFrame, ent: pd.DataFrame, seed: int = 0,
               ws: Workspace | None = None) -> Bundle:
    """Fit every stage of a pipeline on the pairs/entities of one training set."""
    t = {}
    fit_ids, cal_ids, tune_ids = split_entities(ent.s1_entity_id.to_numpy(), seed)
    role = pd.Series(0, index=pd.Index(ent.s1_entity_id))
    role[cal_ids], role[tune_ids] = 1, 2
    r = df.s1_entity_id.map(role).to_numpy()
    g_of = ent.set_index("s1_entity_id").g
    feats = feature_columns(cfg)

    if cfg.kind == "rule":
        m = r > 0
        pol = M.fit_policy("A", df[m].assign(p=df.score[m]), ent[ent.s1_entity_id.isin(np.r_[cal_ids, tune_ids])], None)
        return Bundle(cfg, ["score"], None, None, {}, None, pol, t)

    t0 = time.time()
    fit = df[r == 0]
    es_ids = split_entities(fit_ids, seed + 1, (0.9, 0.1, 0.0))[1]
    es = fit.s1_entity_id.isin(es_ids).to_numpy()
    rows, w = training_rows(fit, cfg, g_of, seed)
    tr = rows & ~es
    model = M.Matcher(cfg.matcher, resolve_params(cfg, ws), seed=seed)
    model.fit(fit.loc[tr, feats], fit.label.to_numpy()[tr], w[tr],
              fit.loc[es, feats] if es.any() else None, fit.label.to_numpy()[es] if es.any() else None)
    t["fit_s"] = time.time() - t0

    cal = df[r == 1]
    raw_cal = model.predict(cal[feats])
    calibrators = {m_: M.Calibrator(m_).fit(raw_cal, cal.label.to_numpy()) for m_ in ("none", "platt", "isotonic", "beta")}
    calibrator = calibrators[cfg.calibration]
    singleton = None
    if cfg.policy in ("B", "C"):
        cal_ent = ent[ent.s1_entity_id.isin(cal_ids)]
        ctx = M.entity_context(cal.assign(p=calibrator(raw_cal)), cal_ent)
        singleton = M.SingletonModel().fit(ctx, (cal_ent.set_index("s1_entity_id").g.reindex(ctx.s1_entity_id) > 0).to_numpy())

    tune = df[r == 2].copy()
    tune_ent = ent[ent.s1_entity_id.isin(tune_ids)].reset_index(drop=True)
    tune["p"] = calibrator(model.predict(tune[feats]))
    q = None
    if singleton is not None:
        q = singleton.predict(M.entity_context(tune, tune_ent))
    tune["amb"] = M.ambiguity(tune)
    t0 = time.time()
    pol = M.fit_policy(cfg.policy, tune, tune_ent, q, seed=seed)
    t["policy_s"] = time.time() - t0
    return Bundle(cfg, feats, model, calibrator, calibrators, singleton, pol, t)


def apply_bundle(b: Bundle, df: pd.DataFrame, ent: pd.DataFrame) -> pd.DataFrame:
    """Score pairs and apply the policy (graph refinement is applied by the caller, over all S1)."""
    out = df.copy()
    if b.cfg.kind == "rule":
        out["raw"] = out["p"] = out.score.to_numpy()
        out["q"] = np.nan
        out["pred"] = out.p.to_numpy() >= b.policy["t"]
        return out
    out["raw"] = b.model.predict(out[b.features])
    out["p"] = b.calibrator(out.raw.to_numpy())
    out["amb"] = M.ambiguity(out)
    q_pair = None
    if b.singleton is not None:
        ctx = M.entity_context(out, ent)
        q_ent = pd.Series(b.singleton.predict(ctx), index=ctx.s1_entity_id.to_numpy())
        q_pair = out.s1_entity_id.map(q_ent).to_numpy()
    out["q"] = q_pair if q_pair is not None else np.nan
    out["pred"] = M.apply_policy(b.policy, out, q_pair)
    return out


# ---- out-of-fold evaluation -----------------------------------------------------------------------------
def _rss_gb() -> float:
    """Peak resident memory of this process in GB (`resource` is Unix-only; psutil covers Windows)."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024 ** 3 if sys.platform == "darwin" else kb / 1024 ** 2   # macOS reports bytes
    except ImportError:
        pass
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 1024 ** 3
    except ImportError:
        return float("nan")


def _slices(e: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    nf = pairs.groupby("s1_entity_id").name_freq_l.first() if "name_freq_l" in pairs else pd.Series(dtype=float)
    e["name_freq"] = e.s1_entity_id.map(nf)
    e["name_freq_bin"] = pd.qcut(e.name_freq.rank(method="first"), 5, labels=["q1 rare", "q2", "q3", "q4", "q5 common"]) \
        .astype(str).where(e.name_freq.notna(), "no cands") if e.name_freq.notna().sum() >= 5 else "all"
    return e


def evaluate(cfg: PipelineConfig, ws: Workspace, reg: Registry, n_folds: int | None = None,
             seed: int = 0, force: bool = False) -> dict | None:
    if reg.exists(cfg.name) and not force:
        log(f"[{cfg.name}] already evaluated: skipping (use --force to redo)")
        return reg.load(cfg.name, "metrics.json")
    t_start = time.time()
    log(f"[{cfg.name}] {cfg.description}")
    ent = ws.entities("dev")
    df = ws.pairs_for(cfg, "dev")
    fold_of = ent.set_index("s1_entity_id").fold
    df["fold"] = df.s1_entity_id.map(fold_of).to_numpy()
    folds = sorted(ent.fold.unique())[: n_folds or None]
    ent = ent[ent.fold.isin(folds)].reset_index(drop=True)
    df = df[df.fold.isin(folds)].reset_index(drop=True)

    parts, fold_rows, cal_rows, importances, bundles_t = [], [], [], [], []
    xsample = None
    for f in folds:
        val = df[df.fold == f]
        val_ent = ent[ent.fold == f]
        if cfg.kind in ("empty", "oracle"):
            out = val.assign(raw=np.nan, p=val.label.astype(float) if cfg.kind == "oracle" else 0.0, q=np.nan,
                             pred=val.label.astype(bool) if cfg.kind == "oracle" else False)
        else:
            tr_ent = ent[ent.fold != f]
            b = fit_bundle(cfg, df[df.fold != f], tr_ent, seed=seed + int(f), ws=ws)
            out = apply_bundle(b, val, val_ent)
            if cfg.graph:
                out["pred"] = M.unique_assignment(out, out.pred.to_numpy())
            bundles_t.append(b.timings)
            if b.model is not None:
                importances.append(b.model.importance().rename(f"fold{f}"))
                for name, c in b.calibrators.items():
                    cal_rows.append({"fold": f, "method": name,
                                     **M.calibration_stats(c(out.raw.to_numpy()), out.label.to_numpy())})
                if xsample is None:
                    xsample = _xsample(b, out)
        e_f = entity_table(val_ent, out)
        fold_rows.append({"fold": int(f), **summary(e_f)})
        parts.append(out[[c for c in KEEP_PAIR_COLS if c in out.columns]])
        log(f"[{cfg.name}] fold {f}: macro F0.5 {fold_rows[-1]['macro_f05']:.4f}")

    pairs = pd.concat(parts, ignore_index=True)
    pairs["pred"] = pairs.pred.astype(bool)
    e = entity_table(ent, pairs)
    e["error_type"] = error_type(e).to_numpy()
    e = _slices(e, pairs)
    metrics = _metrics(cfg, e, pairs, fold_rows, cal_rows)
    ledger = {"total_s": time.time() - t_start, "peak_rss_gb": _rss_gb(), "folds": bundles_t,
              "n_pairs": int(len(pairs)), "n_entities": int(len(e))}
    extra = {}
    if importances:
        imp = pd.concat(importances, axis=1).fillna(0)
        imp = imp / imp.sum(axis=0).replace(0, 1)
        extra["importance.csv"] = imp.assign(mean=imp.mean(axis=1)).sort_values("mean", ascending=False) \
            .rename_axis("feature").reset_index()
    if xsample is not None:
        extra["xsample.parquet"] = xsample
    config = {**cfg.to_dict(), "blocking_config": ws.blocking(cfg.blocking), "resolved_params": resolve_params(cfg, ws)
              if cfg.kind == "model" else {}, "n_folds": len(folds), "dev_entities": int(len(e))}
    reg.save(cfg.name, config=config, entity=e, pairs=pairs, metrics=metrics, ledger=ledger, extra=extra)
    log(f"[{cfg.name}] OOF macro F0.5 {metrics['overall']['macro_f05']:.4f} "
        f"[{metrics['ci']['macro_f05']['lo']:.4f}, {metrics['ci']['macro_f05']['hi']:.4f}] in {ledger['total_s']:.0f}s")
    return metrics


def _xsample(b: Bundle, out: pd.DataFrame, n: int = 20_000) -> pd.DataFrame:
    s = out.sample(min(n, len(out)), random_state=0)
    x = s[b.features].astype(np.float32).reset_index(drop=True)
    meta = s[["s1_entity_id", "cand_entity_id", "label", "p", "pred"]].reset_index(drop=True)
    contrib = b.model.contributions(s[b.features])
    if contrib is not None:
        x = pd.concat([x, pd.DataFrame(contrib.astype(np.float32), columns=[f"shap__{c}" for c in b.features])], axis=1)
    return pd.concat([meta, x], axis=1)


def _thin(t, v, n=400):
    if len(t) <= n:
        return t, v
    idx = np.unique(np.r_[np.linspace(0, len(t) - 1, n).astype(int), np.argmax(v)])
    return t[idx], v[idx]


def _metrics(cfg, e, pairs, fold_rows, cal_rows) -> dict:
    overall = summary(e)
    ci = {m: bootstrap_mean(v) for m, v in (("macro_f05", e.f05), ("ceiling_f05", e.ceiling_f05))}
    single = e.g == 0
    ci["singleton_accuracy"] = bootstrap_mean((e.k[single] == 0).astype(float))
    slices = {}
    for col in ("country", "mult", "name_freq_bin", "has_addr"):
        if col in e:
            slices[col] = e.groupby(col).agg(macro_f05=("f05", "mean"), ceiling_f05=("ceiling_f05", "mean"),
                                             n=("f05", "size")).reset_index().to_dict("records")
    # descriptive curves on OOF scores (thresholds re-optimised on OOF, so optimistic by construction)
    idx = pd.Index(e.s1_entity_id)
    qi = idx.get_indexer(pairs.s1_entity_id)
    curves = {}
    if pairs.p.notna().any() and cfg.kind not in ("empty", "oracle"):
        t, v = threshold_curve(qi, pairs.p.to_numpy(), pairs.label.to_numpy(), e.g.to_numpy())
        t, v = _thin(t, v)
        curves["threshold"] = {"t": t, "macro_f05": v}
        bins = np.linspace(0, 1, 21)
        b = np.clip(np.digitize(pairs.p, bins) - 1, 0, 19)
        rel = pairs.groupby(b).agg(p_mean=("p", "mean"), frac_pos=("label", "mean"), n=("label", "size"))
        curves["reliability"] = rel.reset_index(names="bin").to_dict("records")
        if pairs.q.notna().any():
            q_ent = pairs.groupby("s1_entity_id").q.first().reindex(e.s1_entity_id).fillna(0).to_numpy()
            sweep = []
            for tau in np.linspace(0, 0.95, 20):
                m = q_ent[qi] >= tau
                _, vv = threshold_curve(qi[m], pairs.p.to_numpy()[m], pairs.label.to_numpy()[m], e.g.to_numpy())
                gate_empty = q_ent < tau
                sweep.append({"tau": tau, "macro_f05": float(vv.max()),
                              "singleton_gated": float(gate_empty[single.to_numpy()].mean()),
                              "non_singleton_gated": float(gate_empty[~single.to_numpy()].mean())})
            curves["singleton_gate"] = sweep
    cal = pd.DataFrame(cal_rows).groupby("method")[["nll", "brier"]].mean().reset_index().to_dict("records") \
        if cal_rows else []
    return {"overall": overall, "ci": ci, "folds": fold_rows, "slices": slices,
            "loss": loss_decomposition(e), "error_types": e.error_type.value_counts().to_dict(),
            "calibration": cal, "curves": curves}


# ---- cross-country robustness (France proxy) --------------------------------------------------------------
def cross_country(cfg: PipelineConfig, ws: Workspace, seed: int = 0) -> list[dict]:
    ent = ws.entities("dev")
    df = ws.pairs_for(cfg, "dev")
    countries = [c for c, n in ent.country.value_counts().items() if n >= 200]
    rows = []
    for tr in countries:
        tr_ent = ent[ent.country == tr]
        b = None if cfg.kind in ("empty", "oracle") else \
            fit_bundle(cfg, df[df.s1_entity_id.isin(tr_ent.s1_entity_id)], tr_ent, seed=seed, ws=ws)
        for ev in countries:
            if ev == tr:
                continue
            ev_ent = ent[ent.country == ev]
            val = df[df.s1_entity_id.isin(ev_ent.s1_entity_id)]
            out = apply_bundle(b, val, ev_ent) if b else val.assign(pred=val.label.astype(bool) if cfg.kind == "oracle" else False)
            if cfg.graph and b:
                out["pred"] = M.unique_assignment(out, out.pred.to_numpy())
            rows.append({"train": tr, "eval": ev, "macro_f05": summary(entity_table(ev_ent, out))["macro_f05"]})
    return rows


# ---- the default candidate pipelines (plan §5.1) -----------------------------------------------------------
def default_pipelines(ws: Workspace, neural: bool = False) -> list[PipelineConfig]:
    P = PipelineConfig
    out = [
        P("P-empty", kind="empty", description="predict no match for anyone (= singleton rate)"),
        P("P-oracle", kind="oracle", description="perfect matcher on B0 candidates (= blocking ceiling)"),
        P("P-rule", kind="rule", description="blocking score above a tuned global threshold"),
        P("P0", parent="P-rule", families=("STR", "RET"), description="LightGBM on name similarity + retrieval scores"),
        P("P0-lr", parent="P0", families=("STR", "RET"), matcher="lr", description="P0 with logistic regression (sanity)"),
        P("P1", parent="P0", families=("STR", "RARE", "NUM", "RET"), description="+ rarity and numeric conflicts"),
        P("P2", parent="P1", families=FULL, description="full engineered feature set"),
    ]
    for lib, nm in (("xgb", "XGBoost"), ("cat", "CatBoost")):
        if M.available(lib):
            out.append(P(f"P3-{lib}", parent="P2", families=FULL, matcher=lib, description=f"P2 with {nm}"))
    hpo_libs = [lib for lib in ("lgbm", "xgb", "cat") if ws.hpo_params(lib) is not None and M.available(lib)]
    for lib in hpo_libs:
        base = "P2" if lib == "lgbm" else f"P3-{lib}"
        out.append(P(f"P2-hpo-{lib}", parent=base, families=FULL, matcher=lib, params="hpo",
                     description=f"P2 features, {lib} with Optuna-tuned hyperparameters"))
    best_lib = ws.hpo_best_lib() if hpo_libs else None
    lib4 = best_lib or "lgbm"
    parent4 = f"P2-hpo-{lib4}" if best_lib else "P2"
    params4 = "hpo" if best_lib else "default"
    common = dict(families=FULL, matcher=lib4, params=params4)
    out += [
        P("P4", parent=parent4, negatives="E2", policy="B", **common,
          description="+ singleton-weighted negatives and singleton gate (policy B)"),
        P("P4-E1", parent="P4", negatives="E1", policy="B", **common, description="P4 with capped hard negatives (E1)"),
        P("P4-C", parent="P4", negatives="E2", policy="C", **common,
          description="P4 with ambiguity/conflict-adjusted threshold (policy C, Bayesian-optimised)"),
        P("P8", parent="P4", negatives="E2", policy="B", graph=True, **common,
          description="P4 + unique assignment of external records (graph H1)"),
    ]
    for bname in blocking_configs(ws.paths):
        if bname != "B0":
            out.append(P(f"P5-{bname}", parent="P4", blocking=bname, negatives="E2", policy="B", **common,
                         description=f"P4 on blocking config {bname}"))
    if neural and (ws.paths.art / "dense_dev.parquet").exists():
        out.append(P("P6", parent="P4", negatives="E2", policy="B", dense=True, **common,
                     description="P4 + frozen multilingual-E5 similarity"))
    return out
