"""Matchers (axis D), calibrators (F), singleton model and set policies (G), graph refinement (H)."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd

from er_eval.metrics import best_threshold

N_THREADS = int(os.environ.get("ER_THREADS", os.cpu_count() or 1))


def available(lib: str) -> bool:
    mod = {"lgbm": "lightgbm", "xgb": "xgboost", "cat": "catboost", "lr": "sklearn"}[lib]
    return importlib.util.find_spec(mod) is not None


# ---- matchers --------------------------------------------------------------------------------------
LGBM_DEFAULT = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, pos_weight=1.0)
XGB_DEFAULT = dict(eta=0.08, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                   gamma=0.0, alpha=1e-3, reg_lambda=1.0, pos_weight=1.0)
CAT_DEFAULT = dict(learning_rate=0.08, depth=8, l2_leaf_reg=3.0, border_count=254, random_strength=1.0,
                   bagging_temperature=0.5, bootstrap_type="Bayesian", pos_weight=1.0)
DEFAULTS = {"lgbm": LGBM_DEFAULT, "xgb": XGB_DEFAULT, "cat": CAT_DEFAULT, "lr": {"C": 1.0}}
MAX_ROUNDS, EARLY_STOP = 3000, 100


class Matcher:
    """fit(X, y, w, X_es, y_es) with early stopping on (X_es, y_es); predict → P(match) in [0, 1]."""

    def __init__(self, lib: str, params: dict | None = None, seed: int = 0):
        self.lib, self.seed = lib, seed
        self.params = {**DEFAULTS[lib], **(params or {})}
        self.model = None
        self.best_iter = None
        self.features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None = None,
            X_es: pd.DataFrame | None = None, y_es: np.ndarray | None = None) -> "Matcher":
        self.features = list(X.columns)
        p = dict(self.params)
        pos_w = float(p.pop("pos_weight", 1.0))
        w = np.ones(len(y), np.float32) if w is None else np.asarray(w, np.float32)
        w = w * np.where(y == 1, pos_w, 1.0).astype(np.float32)
        if self.lib == "lgbm":
            import lightgbm as lgb
            params = {"objective": "binary", "verbosity": -1, "num_threads": N_THREADS, "seed": self.seed,
                      "deterministic": False, **p}
            dtr = lgb.Dataset(X, y, weight=w, free_raw_data=True)
            valid = [lgb.Dataset(X_es, y_es, reference=dtr)] if X_es is not None else []
            cbs = [lgb.early_stopping(EARLY_STOP, verbose=False)] if valid else []
            self.model = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=valid, callbacks=cbs)
            self.best_iter = self.model.best_iteration or self.model.current_iteration()
        elif self.lib == "xgb":
            import xgboost as xgb
            params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                      "nthread": N_THREADS, "seed": self.seed, **p}
            dtr = xgb.DMatrix(X, y, weight=w)
            evals = [(xgb.DMatrix(X_es, y_es), "es")] if X_es is not None else []
            self.model = xgb.train(params, dtr, MAX_ROUNDS, evals=evals,
                                   early_stopping_rounds=EARLY_STOP if evals else None, verbose_eval=False)
            self.best_iter = getattr(self.model, "best_iteration", None)
        elif self.lib == "cat":
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(iterations=MAX_ROUNDS, thread_count=N_THREADS, random_seed=self.seed,
                                            verbose=False, allow_writing_files=False,
                                            od_type="Iter" if X_es is not None else None,
                                            od_wait=EARLY_STOP if X_es is not None else None, **p)
            self.model.fit(X, y, sample_weight=w, eval_set=(X_es, y_es) if X_es is not None else None)
            self.best_iter = self.model.get_best_iteration()
        elif self.lib == "lr":
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                       LogisticRegression(C=p.get("C", 1.0), max_iter=2000))
            self.model.fit(X, y, logisticregression__sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        if self.lib == "lgbm":
            return self.model.predict(X, num_iteration=self.best_iter)
        if self.lib == "xgb":
            import xgboost as xgb
            it = (0, self.best_iter + 1) if self.best_iter is not None else (0, 0)
            return self.model.predict(xgb.DMatrix(X), iteration_range=it)
        return self.model.predict_proba(X)[:, 1]

    def importance(self) -> pd.Series:
        if self.lib == "lgbm":
            return pd.Series(self.model.feature_importance("gain"), index=self.features)
        if self.lib == "xgb":
            s = pd.Series(self.model.get_score(importance_type="total_gain"))
            return s.reindex(self.features).fillna(0)
        if self.lib == "cat":
            return pd.Series(self.model.get_feature_importance(), index=self.features)
        coef = self.model[-1].coef_.ravel()[: len(self.features)]
        return pd.Series(np.abs(coef), index=self.features)

    def contributions(self, X: pd.DataFrame) -> np.ndarray | None:
        """Per-feature additive contributions (TreeSHAP) for LightGBM/XGBoost, without the bias column."""
        X = X[self.features]
        if self.lib == "lgbm":
            return self.model.predict(X, pred_contrib=True, num_iteration=self.best_iter)[:, :-1]
        if self.lib == "xgb":
            import xgboost as xgb
            return self.model.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        if self.lib == "cat":
            from catboost import Pool
            return self.model.get_feature_importance(Pool(X), type="ShapValues")[:, :-1]
        return None


# ---- calibration -----------------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.m = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "none" or len(np.unique(y)) < 2:
            self.method_used = "none"
            return self
        self.method_used = self.method
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        else:
            from sklearn.linear_model import LogisticRegression
            self.m = LogisticRegression(C=1e4, max_iter=1000).fit(self._x(p), y)
        return self

    def _x(self, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if self.method == "platt":
            return _logit(p)[:, None]
        return np.c_[np.log(p), -np.log(1 - p)]        # beta calibration (Kull et al. 2017)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self.method_used == "none":
            return np.asarray(p, np.float64)
        if self.method == "isotonic":
            return self.m.predict(p)
        return self.m.predict_proba(self._x(p))[:, 1]


def calibration_stats(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2))}


# ---- singleton / any-match model -------------------------------------------------------------------
def entity_context(pairs: pd.DataFrame, entities: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    """One row per entity (zero-candidate entities included) summarising its candidate scores."""
    g = pairs.groupby("s1_entity_id", sort=False)
    top = pairs[["s1_entity_id", p_col]].sort_values(["s1_entity_id", p_col], ascending=[True, False])
    rank = top.groupby("s1_entity_id").cumcount().to_numpy()
    second = top[rank == 1].set_index("s1_entity_id")[p_col]
    ctx = pd.DataFrame({
        "p_max": g[p_col].max(), "p_sum": g[p_col].sum(), "p_mean": g[p_col].mean(), "n_cand": g.size(),
        "n_p50": (pairs[p_col] >= 0.5).groupby(pairs.s1_entity_id).sum(),
        "n_p80": (pairs[p_col] >= 0.8).groupby(pairs.s1_entity_id).sum(),
        "score_max": g["score"].max(),
    })
    ctx["p_second"] = second.reindex(ctx.index).fillna(0.0)
    for c in ("name_freq_l", "name_ntok_l", "name_empty_l"):
        if c in pairs:
            ctx[c] = g[c].first()
    out = entities[["s1_entity_id"]].merge(ctx, left_on="s1_entity_id", right_index=True, how="left")
    out[["p_max", "p_sum", "p_mean", "n_cand", "n_p50", "n_p80", "score_max", "p_second"]] = \
        out[["p_max", "p_sum", "p_mean", "n_cand", "n_p50", "n_p80", "score_max", "p_second"]].fillna(0.0)
    out["gap"] = out.p_max - out.p_second
    return out


class SingletonModel:
    """q_i = P(entity has at least one true match | its candidate-score summary)."""

    FEATS = ["p_max", "p_second", "gap", "p_sum", "p_mean", "n_cand", "n_p50", "n_p80", "score_max",
             "name_freq_l", "name_ntok_l", "name_empty_l"]

    def fit(self, ctx: pd.DataFrame, has_match: np.ndarray) -> "SingletonModel":
        import lightgbm as lgb
        self.cols = [c for c in self.FEATS if c in ctx]
        if len(np.unique(has_match)) < 2:
            self.const = float(has_match.mean())
            return self
        self.const = None
        self.m = lgb.train({"objective": "binary", "num_leaves": 15, "learning_rate": 0.05,
                            "min_data_in_leaf": 50, "verbosity": -1, "num_threads": N_THREADS},
                           lgb.Dataset(ctx[self.cols], has_match), num_boost_round=300)
        return self

    def predict(self, ctx: pd.DataFrame) -> np.ndarray:
        if self.const is not None:
            return np.full(len(ctx), self.const)
        return self.m.predict(ctx[self.cols])


# ---- set policies ------------------------------------------------------------------------------------
def _codes(pairs: pd.DataFrame, entities: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    idx = pd.Index(entities.s1_entity_id)
    return idx.get_indexer(pairs.s1_entity_id), entities.g.to_numpy()


def ambiguity(pairs: pd.DataFrame, p_col: str = "p") -> np.ndarray:
    """Per pair: second-best / best calibrated score of its entity (1 = two equally strong candidates)."""
    g = pairs.groupby("s1_entity_id")[p_col]
    best = g.transform("max").to_numpy()
    rank = g.rank(method="first", ascending=False).to_numpy()
    second = pairs[p_col].where(rank == 2).groupby(pairs.s1_entity_id).transform("max").fillna(0).to_numpy()
    return np.where(best > 0, second / np.maximum(best, 1e-9), 0.0)


def adjusted_score(pairs: pd.DataFrame, alpha: float, beta: float) -> np.ndarray:
    return pairs.p.to_numpy() - alpha * pairs.amb.to_numpy() - beta * pairs.conflict_count.fillna(0).to_numpy()


def fit_policy(kind: str, pairs: pd.DataFrame, entities: pd.DataFrame, q: np.ndarray | None,
               n_trials: int = 60, seed: int = 0) -> dict:
    """Tune a set policy on (pairs with calibrated `p`, entity gold counts g, entity q).

    A: global threshold on p                       (exact optimiser)
    B: singleton gate q ≥ τ, then global threshold  (grid over τ, exact threshold per τ)
    C: gate + threshold on p - α·ambiguity - β·conflicts (α, β, τ by Bayesian optimisation)"""
    qi, g = _codes(pairs, entities)
    y = pairs.label.to_numpy()
    if kind == "A":
        t, v = best_threshold(qi, pairs.p.to_numpy(), y, g)
        return {"kind": "A", "t": t, "tune_macro_f05": v}
    q_pair = q[qi]
    if kind == "B":
        best = None
        for tau in np.r_[0.0, np.quantile(q, np.linspace(0.02, 0.9, 45))]:
            m = q_pair >= tau
            t, v = best_threshold(qi[m], pairs.p.to_numpy()[m], y[m], g)
            if best is None or v > best["tune_macro_f05"] + 1e-12:
                best = {"kind": "B", "t": t, "tau": float(tau), "tune_macro_f05": v}
        return best
    if kind == "C":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            a = trial.suggest_float("alpha", 0.0, 0.5)
            b = trial.suggest_float("beta", 0.0, 0.5)
            tau = trial.suggest_float("tau", 0.0, 0.9)
            m = q_pair >= tau
            t, v = best_threshold(qi[m], adjusted_score(pairs, a, b)[m], y[m], g)
            trial.set_user_attr("t", t)
            return v

        study = optuna.create_study(direction="maximize", sampler=_decision_sampler(seed))
        study.enqueue_trial({"alpha": 0.0, "beta": 0.0, "tau": 0.0})   # policy A as a starting point
        study.optimize(objective, n_trials=n_trials)
        bp = study.best_params
        return {"kind": "C", "t": study.best_trial.user_attrs["t"], **bp, "tune_macro_f05": study.best_value}
    raise ValueError(kind)


def _decision_sampler(seed: int):
    """GP-based Bayesian optimisation when available (needs torch), else multivariate TPE."""
    import optuna
    if importlib.util.find_spec("torch") is not None and hasattr(optuna.samplers, "GPSampler"):
        return optuna.samplers.GPSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=10)


def apply_policy(policy: dict, pairs: pd.DataFrame, q_pair: np.ndarray | None) -> np.ndarray:
    kind = policy["kind"]
    s = adjusted_score(pairs, policy["alpha"], policy["beta"]) if kind == "C" else pairs.p.to_numpy()
    pred = s >= policy["t"]
    if kind in ("B", "C"):
        pred &= q_pair >= policy["tau"]
    return pred


def unique_assignment(pairs: pd.DataFrame, pred: np.ndarray, p_col: str = "p") -> np.ndarray:
    """Graph refinement H1: an external record predicted for several S1 entities keeps only its
    highest-scoring S1 (S1 is deduplicated, so one S2/S3 record describes at most one S1 business)."""
    pred = pred.copy()
    idx = np.flatnonzero(pred)
    if not len(idx):
        return pred
    sub = pairs.iloc[idx]
    order = np.lexsort((-sub[p_col].to_numpy(), sub.cand_entity_id.to_numpy()))
    cand_sorted = sub.cand_entity_id.to_numpy()[order]
    first = np.r_[True, cand_sorted[1:] != cand_sorted[:-1]]
    keep = np.zeros(len(idx), bool)
    keep[order[first]] = True
    pred[idx[~keep]] = False
    return pred
