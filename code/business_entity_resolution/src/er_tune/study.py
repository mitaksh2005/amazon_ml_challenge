"""Optuna studies (plan §5.2).

Blocking: multi-objective (max blocking-ceiling F0.5, min candidates per S1), NSGA-II, on the `hpo`
sample's wide candidates (no training needed: the ceiling is label-based but nothing is fitted).
Matchers: macro F0.5 after the exact threshold optimiser, mean over the `hpo` sample's inner folds,
pruned with a median pruner after each fold; TPE (Bayesian) sampler plus an equal-budget random
sampler baseline. The top trials are re-evaluated with fresh splits/seeds and the configuration
with the best mean − std is kept (winner's-curse control).

Studies live in <work>/experiments/hpo/optuna.db, so interrupted runs resume."""
from __future__ import annotations

import json
import time

import numpy as np
import optuna
import pandas as pd

from data import B0, log, truncate
from er_eval import entity_f05, summary, write_json
from er_eval.metrics import entity_table
from pipelines import FULL, PipelineConfig, Workspace, add_ret_ctx, apply_bundle, fit_bundle

from . import spaces


def _storage(ws: Workspace) -> str:
    d = ws.paths.exp / "hpo"
    d.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{d / 'optuna.db'}"


def _sampler(kind: str, seed: int, multi_objective: bool = False):
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if multi_objective:
        return optuna.samplers.NSGAIISampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=10)


def _remaining(study, n_trials: int) -> int:
    done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    return max(0, n_trials - len(done))


# ---- blocking -------------------------------------------------------------------------------------------
def blocking_metrics(wide: pd.DataFrame, ent: pd.DataFrame, cfg: dict) -> dict:
    c = truncate(wide, cfg)
    agg = c.groupby("s1_entity_id").agg(n=("label", "size"), c=("label", "sum"))
    e = ent[["s1_entity_id", "g"]].merge(agg, left_on="s1_entity_id", right_index=True, how="left").fillna(0)
    return {"ceiling_f05": float(entity_f05(e.g, e.c, e.c).mean()), "cands_per_s1": float(e.n.mean()),
            "pair_recall": float(e.c.sum() / max(e.g.sum(), 1))}


def tune_blocking(ws: Workspace, n_trials: int = 80, seed: int = 0, random_baseline: bool = True) -> dict:
    wide, ent = ws.wide_pairs("hpo"), ws.entities("hpo")
    out_dir = ws.paths.exp / "blocking"
    base = blocking_metrics(wide, ent, B0)
    log(f"blocking B0 on hpo sample: {base}")

    def objective(trial):
        m = blocking_metrics(wide, ent, spaces.blocking(trial))
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m["ceiling_f05"], m["cands_per_s1"]

    studies = {}
    for kind in ["tpe"] + (["random"] if random_baseline else []):
        st = optuna.create_study(study_name=f"blocking-{kind}", storage=_storage(ws), load_if_exists=True,
                                 directions=["maximize", "minimize"], sampler=_sampler(kind, seed, True))
        if kind == "tpe" and not st.trials:
            st.enqueue_trial({"k_all": 20, "min_all": 0.05, "k_name": 10, "min_name": 0.05,
                              "k_addr": 10, "min_addr": 0.05})
        st.optimize(objective, n_trials=_remaining(st, n_trials))
        studies[kind] = st

    front = [t for t in studies["tpe"].best_trials]
    rows = [{"trial": t.number, **t.user_attrs, "cfg": spaces.blocking(optuna.trial.FixedTrial(t.params))} for t in front]
    front_df = pd.DataFrame(rows).sort_values("cands_per_s1")

    def pick(max_cands):
        ok = front_df[front_df.cands_per_s1 <= max_cands]
        return None if ok.empty else ok.sort_values("ceiling_f05").iloc[-1]

    chosen = {}
    for name, budget in (("Btuned", base["cands_per_s1"]), ("Bwide", 1.75 * base["cands_per_s1"]),
                         ("Blean", 0.7 * base["cands_per_s1"])):
        r = pick(budget)
        if r is not None and r.ceiling_f05 > base["ceiling_f05"] - (0.002 if name == "Blean" else -1e-9):
            chosen[name] = r.cfg
    write_json(out_dir / "blocking_configs.json", chosen)
    write_json(out_dir / "blocking_front.json", {"base": base, "front": front_df.drop(columns="cfg").to_dict("records"),
                                                "chosen": {k: blocking_metrics(wide, ent, v) for k, v in chosen.items()}})
    log(f"blocking configs chosen: { {k: blocking_metrics(wide, ent, v) for k, v in chosen.items()} }")
    return chosen


# ---- matchers -------------------------------------------------------------------------------------------
def _hpo_frames(ws: Workspace):
    ent = ws.entities("hpo")
    df = add_ret_ctx(truncate(ws.wide_pairs("hpo"), B0))
    return ent, df


def _eval_split(lib, params, df, ent, val_fold, seed) -> float:
    cfg = PipelineConfig("hpo", families=FULL, matcher=lib, params=params, policy="A")
    tr_ent, va_ent = ent[ent.fold != val_fold], ent[ent.fold == val_fold]
    b = fit_bundle(cfg, df[df.s1_entity_id.isin(tr_ent.s1_entity_id)], tr_ent, seed=seed)
    out = apply_bundle(b, df[df.s1_entity_id.isin(va_ent.s1_entity_id)], va_ent)
    return summary(entity_table(va_ent, out))["macro_f05"]


def tune_matcher(ws: Workspace, lib: str, n_trials: int = 30, seed: int = 0, random_baseline: bool = False,
                 top_k: int = 5, n_seeds: int = 3, timeout: float | None = None) -> dict:
    ent, df = _hpo_frames(ws)
    folds = sorted(ent.fold.unique())
    space = spaces.MATCHER_SPACES[lib]

    def objective(trial):
        params = space(trial)
        vals = []
        for i, f in enumerate(folds):
            vals.append(_eval_split(lib, params, df, ent, f, seed + i))
            trial.report(float(np.mean(vals)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_std", float(np.std(vals)))
        return float(np.mean(vals))

    studies = {}
    for kind in ["tpe"] + (["random"] if random_baseline else []):
        st = optuna.create_study(study_name=f"matcher-{lib}-{kind}", storage=_storage(ws), load_if_exists=True,
                                 direction="maximize", sampler=_sampler(kind, seed),
                                 pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
        if kind == "tpe" and not st.trials:
            from models import DEFAULTS
            st.enqueue_trial({k: v for k, v in DEFAULTS[lib].items() if k != "bagging_freq"},
                             skip_if_exists=True)
        t0 = time.time()
        st.optimize(objective, n_trials=_remaining(st, n_trials), timeout=timeout, catch=(ValueError,))
        log(f"study matcher-{lib}-{kind}: best {st.best_value:.4f} after {len(st.trials)} trials ({time.time() - t0:.0f}s)")
        studies[kind] = st

    # winner's curse: re-evaluate the top trials on reshuffled inner folds with new seeds
    done = [t for t in studies["tpe"].trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:top_k]
    rows = []
    for t in top:
        params = spaces.params_from_trial_params(lib, t.params)
        scores = []
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 + s)
            e2 = ent.assign(fold=rng.integers(0, len(folds), len(ent)))
            scores.append(_eval_split(lib, params, df, e2, 0, 1000 + s))
        rows.append({"trial": t.number, "study_value": t.value, "reeval_mean": float(np.mean(scores)),
                     "reeval_std": float(np.std(scores)), "robust_score": float(np.mean(scores) - np.std(scores)),
                     "params": params})
        log(f"  re-eval trial {t.number}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    best = max(rows, key=lambda r: r["robust_score"])
    out_dir = ws.paths.exp / "hpo" / lib
    write_json(out_dir / "best_params.json", best)
    write_json(out_dir / "reevaluation.json", rows)
    return best


def load_study(ws: Workspace, name: str):
    try:
        return optuna.load_study(study_name=name, storage=_storage(ws))
    except KeyError:
        return None


def study_names(ws: Workspace) -> list[str]:
    try:
        return [s.study_name for s in optuna.get_all_study_summaries(storage=_storage(ws))]
    except Exception:
        return []


def dump_json(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
