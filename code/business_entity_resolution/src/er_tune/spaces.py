"""Search spaces (plan §5.2). Each function maps an Optuna trial to a parameter dict."""
from __future__ import annotations


def blocking(trial) -> dict:
    cfg = {}
    for p, (lo, hi) in (("all", (5, 50)), ("name", (0, 30)), ("addr", (0, 30))):
        k = trial.suggest_int(f"k_{p}", lo, hi)
        ms = trial.suggest_float(f"min_{p}", 0.05, 0.30)
        cfg[p] = {"k": k, "min_score": ms}
    return cfg


def lgbm(trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
        "pos_weight": trial.suggest_float("pos_weight", 1.0, 20.0, log=True),
    }


def xgb(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 200.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "pos_weight": trial.suggest_float("pos_weight", 1.0, 20.0, log=True),
    }


def cat(trial) -> dict:
    return {
        "depth": trial.suggest_int("depth", 4, 8),        # deeper symmetric trees get very slow
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "pos_weight": trial.suggest_float("pos_weight", 1.0, 20.0, log=True),
    }


MATCHER_SPACES = {"lgbm": lgbm, "xgb": xgb, "cat": cat}


def params_from_trial_params(lib: str, tp: dict) -> dict:
    """Rebuild a full param dict from a finished trial's `params` (fixed params re-added)."""
    p = dict(tp)
    if lib == "lgbm":
        p["bagging_freq"] = 1
    return p
