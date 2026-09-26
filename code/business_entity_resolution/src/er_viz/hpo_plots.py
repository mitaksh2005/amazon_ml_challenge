"""HPO diagnostics (plan §5.2): convergence vs random search, importances, slices, contours, re-evaluation."""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from .common import INK2, MUTED, NEUTRAL_BAR, SERIES, FigureSink

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def _complete(study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def history_vs_random(sink: FigureSink, tag: str, tpe, rnd):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    rows = []
    for i, (lab, st) in enumerate((("TPE (Bayesian)", tpe), ("random search", rnd))):
        if st is None:
            continue
        tr = _complete(st)
        if not tr:
            continue
        n = [t.number for t in tr]
        v = [t.value for t in tr]
        best = np.maximum.accumulate(v)
        ax.scatter(n, v, s=12, color=SERIES[i], alpha=0.35, linewidths=0)
        ax.step(n, best, where="post", color=SERIES[i], label=f"{lab}: best {best[-1]:.4f}")
        rows += [{"sampler": lab, "trial": a, "value": b, "best_so_far": c} for a, b, c in zip(n, v, best)]
    ax.set(xlabel="trial", ylabel="objective (macro F0.5)", title=f"Optimisation history — {tag}")
    ax.legend()
    sink.save(fig, f"hpo_{tag}_history", f"Best-so-far objective, TPE vs random ({tag})", "Tuning", pd.DataFrame(rows))


def edf(sink: FigureSink, tag: str, tpe, rnd):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    rows = []
    for i, (lab, st) in enumerate((("TPE", tpe), ("random", rnd))):
        if st is None or not _complete(st):
            continue
        v = np.sort([t.value for t in _complete(st)])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=SERIES[i], label=lab)
        rows += [{"sampler": lab, "value": x} for x in v]
    ax.set(xlabel="objective", ylabel="share of trials ≤ value", title=f"Empirical distribution of trial scores — {tag}")
    ax.legend()
    sink.save(fig, f"hpo_{tag}_edf", f"EDF of trial objectives ({tag})", "Tuning", pd.DataFrame(rows),
              note="A curve further right = the sampler finds good regions more often.")


def importances(sink: FigureSink, tag: str, st):
    if st is None or len(_complete(st)) < 5:
        return
    rows = []
    evals = {"fANOVA": optuna.importance.FanovaImportanceEvaluator(seed=0)}
    if hasattr(optuna.importance, "PedAnovaImportanceEvaluator"):
        evals["PED-ANOVA"] = optuna.importance.PedAnovaImportanceEvaluator()
    for name, ev in evals.items():
        try:
            imp = optuna.importance.get_param_importances(st, evaluator=ev)
            rows += [{"method": name, "param": k, "importance": v} for k, v in imp.items()]
        except Exception as ex:   # noqa: BLE001
            rows.append({"method": name, "param": f"error: {ex}", "importance": np.nan})
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return
    params = df.groupby("param").importance.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(params) + 1.4))
    methods = df.method.unique().tolist()
    h = 0.8 / len(methods)
    for i, m in enumerate(methods):
        d = df[df.method == m].set_index("param").reindex(params)
        ax.barh(np.arange(len(params)) + i * h, d.importance, height=h * 0.9, color=SERIES[i], label=m)
    ax.set_yticks(np.arange(len(params)) + h * (len(methods) - 1) / 2, params)
    ax.set(xlabel="importance", title=f"Which hyperparameters matter — {tag}")
    ax.legend()
    ax.grid(axis="y", visible=False)
    sink.save(fig, f"hpo_{tag}_importance", f"Hyperparameter importance ({tag})", "Tuning", df)
    return params[::-1]


def slices(sink: FigureSink, tag: str, st, params: list[str]):
    tr = _complete(st)
    if not tr or not params:
        return
    ps = params[:6]
    fig, axes = plt.subplots(1, len(ps), figsize=(3 * len(ps), 3), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, p in zip(axes, ps):
        x = [t.params.get(p) for t in tr]
        v = [t.value for t in tr]
        numeric = all(isinstance(a, (int, float)) for a in x if a is not None)
        xs = x if numeric else pd.Categorical([str(a) for a in x]).codes
        ax.scatter(xs, v, s=14, color=SERIES[0], alpha=0.6, linewidths=0)
        dist = tr[0].distributions.get(p)
        if numeric and getattr(dist, "log", False):
            ax.set_xscale("log")
        if numeric and dist is not None and hasattr(dist, "low"):
            for edge in (dist.low, dist.high):
                ax.axvline(edge, color=MUTED, lw=1, ls=":")
        ax.set(title=p, xlabel="value")
        rows += [{"param": p, "value": a, "objective": b} for a, b in zip(x, v)]
    axes[0].set_ylabel("objective")
    sink.save(fig, f"hpo_{tag}_slices", f"Objective vs each important parameter ({tag})", "Tuning", pd.DataFrame(rows),
              note="Dotted lines = search-range edges; a best value on an edge means the range should widen.")


def contour(sink: FigureSink, tag: str, st, params: list[str]):
    if not params or len(params) < 2 or len(_complete(st)) < 8:
        return
    try:
        from optuna.visualization.matplotlib import plot_contour
        ax = plot_contour(st, params=params[:2])
        fig = np.ravel(ax)[0].figure if isinstance(ax, np.ndarray) else ax.figure
        fig.set_size_inches(6, 4.5)
        sink.save(fig, f"hpo_{tag}_contour", f"Interaction of the two most important parameters ({tag})", "Tuning")
    except Exception:   # noqa: BLE001
        pass


def parallel(sink: FigureSink, tag: str, st, params: list[str]):
    tr = _complete(st)
    if len(tr) < 5 or not params:
        return
    ps = params[:6]
    df = pd.DataFrame([{**{p: t.params.get(p) for p in ps}, "value": t.value} for t in tr])
    num = df[ps].apply(pd.to_numeric, errors="coerce")
    norm = (num - num.min()) / (num.max() - num.min()).replace(0, 1)
    top = df.value >= df.value.quantile(0.8)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i in range(len(df)):
        ax.plot(range(len(ps)), norm.iloc[i], color=SERIES[0] if top.iloc[i] else NEUTRAL_BAR,
                lw=1.5 if top.iloc[i] else 0.8, alpha=0.9 if top.iloc[i] else 0.5, zorder=3 if top.iloc[i] else 1)
    ax.set_xticks(range(len(ps)), ps, rotation=15)
    ax.set(ylabel="value (min–max scaled)", title=f"Top-20% trials (blue) across parameters — {tag}")
    sink.save(fig, f"hpo_{tag}_parallel", f"Parallel coordinates of trials ({tag})", "Tuning", df)


def intermediate(sink: FigureSink, tag: str, st):
    rows = []
    for t in st.trials:
        for step, v in t.intermediate_values.items():
            rows.append({"trial": t.number, "step": step, "value": v, "state": t.state.name})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for (tn, stt), d in df.groupby(["trial", "state"]):
        ax.plot(d.step, d.value, color=SERIES[1] if stt == "PRUNED" else SERIES[0], lw=1, alpha=0.6,
                marker="o", ms=3)
    ax.set(xlabel="inner fold (step)", ylabel="running mean objective",
           title=f"Pruning: orange trials were stopped early — {tag}")
    ax.set_xticks(sorted(df.step.unique()))
    sink.save(fig, f"hpo_{tag}_pruning", f"Intermediate values and pruned trials ({tag})", "Tuning", df)


def reevaluation(sink: FigureSink, tag: str, path):
    if not path.exists():
        return
    df = pd.DataFrame(json.loads(path.read_text()))
    if df.empty:
        return
    best = df.robust_score.idxmax()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.study_value, width=0.38, color=NEUTRAL_BAR, label="study value (optimistic)")
    ax.bar(x + 0.2, df.reeval_mean, width=0.38, yerr=df.reeval_std, color=[SERIES[0] if i == best else SERIES[2] for i in x],
           ecolor=INK2, capsize=3, label="re-evaluated mean ± std")
    ax.set_xticks(x, [f"trial {t}" for t in df.trial])
    lo = np.nanmin(np.r_[df.study_value, df.reeval_mean - df.reeval_std])
    ax.set_ylim(lo - 0.005, np.nanmax(np.r_[df.study_value, df.reeval_mean + df.reeval_std]) + 0.003)
    ax.set(ylabel="macro F0.5", title=f"Winner's-curse check, top trials — {tag}")
    ax.legend(fontsize=8)
    sink.save(fig, f"hpo_{tag}_reeval", f"Top trials re-evaluated on fresh splits ({tag}); blue = chosen", "Tuning",
              df.drop(columns="params"))


def blocking_front(sink: FigureSink, tpe, rnd):
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    for i, (lab, st) in enumerate((("NSGA-II", tpe), ("random", rnd))):
        if st is None:
            continue
        tr = [t for t in st.trials if t.values]
        x = [t.values[1] for t in tr]
        y = [t.values[0] for t in tr]
        ax.scatter(x, y, s=12, color=SERIES[i], alpha=0.5, linewidths=0, label=f"{lab} trials")
        front = sorted(st.best_trials, key=lambda t: t.values[1])
        ax.plot([t.values[1] for t in front], [t.values[0] for t in front], color=SERIES[i], lw=1.5)
        rows += [{"sampler": lab, "cands_per_s1": a, "ceiling_f05": b} for a, b in zip(x, y)]
    ax.set(xlabel="mean candidates per S1", ylabel="blocking ceiling F0.5", title="Blocking study: Pareto fronts")
    ax.legend()
    sink.save(fig, "hpo_blocking_front", "Blocking multi-objective study, NSGA-II vs random", "Tuning", pd.DataFrame(rows))


def all_studies(sink: FigureSink, ws):
    from er_tune.study import load_study
    blk_t, blk_r = load_study(ws, "blocking-tpe"), load_study(ws, "blocking-random")
    if blk_t is not None:
        blocking_front(sink, blk_t, blk_r)
    for lib in ("lgbm", "xgb", "cat"):
        tpe = load_study(ws, f"matcher-{lib}-tpe")
        if tpe is None:
            continue
        rnd = load_study(ws, f"matcher-{lib}-random")
        history_vs_random(sink, lib, tpe, rnd)
        edf(sink, lib, tpe, rnd)
        params = importances(sink, lib, tpe) or []
        slices(sink, lib, tpe, params)
        contour(sink, lib, tpe, params)
        parallel(sink, lib, tpe, params)
        intermediate(sink, lib, tpe)
        reevaluation(sink, lib, ws.paths.exp / "hpo" / lib / "reevaluation.json")
