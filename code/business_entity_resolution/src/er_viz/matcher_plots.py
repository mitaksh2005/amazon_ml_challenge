"""Matcher and decision-layer analysis (plan §6.2, §6.3), from registry runs."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import FAMILIES
from pipelines import CTX_COLS, DENSE_COLS, RET_COLS

from .common import INK2, MUTED, NEUTRAL_BAR, SERIES, FigureSink, seq_cmap

FAMILY_OF = {c: f for f, cols in FAMILIES.items() for c in cols}
FAMILY_OF.update({c: "RET" for c in RET_COLS})
FAMILY_OF.update({c: "CTX" for c in CTX_COLS})
FAMILY_OF.update({c: "DENSE" for c in DENSE_COLS})
FAMILY_ORDER = ["STR", "RARE", "NUM", "ADDR", "LEGAL", "TXT", "RET", "CTX", "DENSE"]


def _pr(label, score):
    order = np.argsort(-score, kind="stable")
    y = label[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    prec, rec = tp / k, tp / max(y.sum(), 1)
    idx = np.unique(np.linspace(0, len(y) - 1, min(len(y), 500)).astype(int))
    return prec[idx], rec[idx]


def pr_curves(sink: FigureSink, runs: dict[str, pd.DataFrame]):
    """runs: name → OOF pairs (≤ 6 pipelines, one colour slot each)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    rows = []
    for i, (name, p) in enumerate(list(runs.items())[:6]):
        if p.raw.isna().all():
            continue
        prec, rec = _pr(p.label.to_numpy(), p.raw.fillna(-1).to_numpy())
        ax.plot(rec, prec, color=SERIES[i], label=name)
        sel = p.pred.to_numpy()
        if sel.any():
            op_p = p.label[sel].mean()
            op_r = p.label[sel].sum() / max(p.label.sum(), 1)
            ax.scatter([op_r], [op_p], color=SERIES[i], s=50, edgecolor="white", linewidth=1.5, zorder=5)
        rows += [{"pipeline": name, "recall": a, "precision": b} for a, b in zip(rec, prec)]
    ax.set(xlabel="pair recall (among retrieved candidates)", ylabel="pair precision",
           title="Precision–recall of pair scores", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower left")
    sink.save(fig, "matcher_pr_curves", "Precision-recall curves (OOF), dots = chosen operating point", "Matcher",
              pd.DataFrame(rows))


def score_density(sink: FigureSink, name: str, pairs: pd.DataFrame, entity: pd.DataFrame):
    single = set(entity.s1_entity_id[entity.g == 0])
    kind = np.where(pairs.label == 1, "true match",
                    np.where(pairs.s1_entity_id.isin(single), "neg: singleton's candidate",
                             np.where(pairs.name_ratio.fillna(0) >= 0.8, "neg: hard (name ≥ 0.8)", "neg: other")))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, 1, 41)
    rows = []
    for i, k in enumerate(["true match", "neg: hard (name ≥ 0.8)", "neg: singleton's candidate", "neg: other"]):
        v = pairs.p[kind == k].dropna()
        if v.empty:
            continue
        h, _ = np.histogram(v, bins=bins, density=True)
        ax.step(bins[:-1], h, where="post", color=SERIES[i], label=f"{k} (n={len(v):,})")
        rows += [{"kind": k, "bin_lo": b, "density": d} for b, d in zip(bins[:-1], h)]
    ax.set_yscale("log")
    ax.set(xlabel="calibrated match probability", ylabel="density (log)", title=f"Score distributions — {name}")
    ax.legend(fontsize=8)
    sink.save(fig, "matcher_score_density", f"Score density by pair type ({name})", "Matcher", pd.DataFrame(rows),
              note="Hard negatives overlapping the true-match mass are where precision is lost.")


def importance(sink: FigureSink, name: str, imp: pd.DataFrame, top: int = 30):
    d = imp.head(top).iloc[::-1]
    fam = d.feature.map(FAMILY_OF).fillna("Other")
    colors = [SERIES[FAMILY_ORDER.index(f)] if f in FAMILY_ORDER[:8] else NEUTRAL_BAR for f in fam]
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(d) + 1.2))
    ax.barh(d.feature, d["mean"], color=colors, height=0.7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[FAMILY_ORDER.index(f)]) for f in FAMILY_ORDER[:8] if f in set(fam)]
    ax.legend(handles, [f for f in FAMILY_ORDER[:8] if f in set(fam)], loc="lower right", title="family")
    ax.set(xlabel="share of total gain (mean over folds)", title=f"Feature importance — {name}")
    ax.grid(axis="y", visible=False)
    sink.save(fig, "matcher_importance", f"Top-{top} features by gain ({name})", "Matcher",
              imp.assign(family=imp.feature.map(FAMILY_OF)))


def family_importance(sink: FigureSink, name: str, imp: pd.DataFrame):
    fam = imp.assign(family=imp.feature.map(FAMILY_OF).fillna("Other")).groupby("family")["mean"].sum() \
        .reindex([f for f in FAMILY_ORDER if f in set(imp.feature.map(FAMILY_OF))]).dropna()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(fam.index[::-1], fam.values[::-1], color=SERIES[0], height=0.6)
    ax.set(xlabel="share of total gain", title=f"Importance by feature family — {name}")
    ax.grid(axis="y", visible=False)
    sink.save(fig, "matcher_family_importance", f"Gain by feature family ({name})", "Matcher",
              fam.rename("gain_share").reset_index())


def shap_summary(sink: FigureSink, name: str, xs: pd.DataFrame, top: int = 15):
    shap_cols = [c for c in xs.columns if c.startswith("shap__")]
    if not shap_cols:
        return
    mean_abs = xs[shap_cols].abs().mean().sort_values(ascending=False).head(top)
    feats = [c[len("shap__"):] for c in mean_abs.index]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(feats) + 1.3))
    rng = np.random.default_rng(0)
    cmap = seq_cmap()
    sample = xs.sample(min(4000, len(xs)), random_state=0)
    for i, f in enumerate(feats[::-1]):
        v = sample[f"shap__{f}"].to_numpy()
        x = sample[f].to_numpy(dtype=float)
        rank = pd.Series(x).rank(pct=True).fillna(0.5).to_numpy()
        ax.scatter(v, i + rng.uniform(-0.3, 0.3, len(v)), c=cmap(rank), s=6, linewidths=0)
    ax.set_yticks(range(len(feats)), feats[::-1])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set(xlabel="contribution to log-odds (TreeSHAP)", title=f"Feature contributions — {name}")
    ax.text(1.0, -0.12, "colour: feature value percentile (light = low, dark = high)", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    sink.save(fig, "matcher_shap_summary", f"TreeSHAP summary ({name})", "Matcher",
              mean_abs.rename("mean_abs_contribution").rename_axis("feature").reset_index())


def shap_dependence(sink: FigureSink, name: str, xs: pd.DataFrame):
    feats = [f for f in ("name_num_conflict", "house_conflict", "conflict_count", "name_wjacc", "name_miss_idf_max_r",
                         "name_freq_l") if f"shap__{f}" in xs]
    if not feats:
        return
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, f in zip(axes, feats):
        x, v = xs[f].to_numpy(float), xs[f"shap__{f}"].to_numpy()
        ok = ~np.isnan(x)
        ax.scatter(x[ok] + np.random.default_rng(0).normal(0, 0.01, ok.sum()), v[ok], s=5, color=SERIES[0], alpha=0.4, linewidths=0)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(title=f, xlabel="feature value")
        rows.append({"feature": f, "corr": float(pd.Series(x[ok]).corr(pd.Series(v[ok]))) if ok.sum() > 2 else np.nan})
    axes[0].set_ylabel("contribution")
    sink.save(fig, "matcher_shap_dependence", f"Do conflict / rarity features push towards non-match? ({name})",
              "Matcher", pd.DataFrame(rows))


def slice_bars(sink: FigureSink, metrics: dict[str, dict], col: str = "country", top: list[str] | None = None):
    rows = []
    for name, m in metrics.items():
        for r in m.get("slices", {}).get(col, []):
            rows.append({"pipeline": name, col: r[col], "macro_f05": r["macro_f05"], "ceiling_f05": r["ceiling_f05"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    names = top or list(metrics)[:4]
    df = df[df.pipeline.isin(names)]
    cats = sorted(df[col].astype(str).unique())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        d = df[df.pipeline == n].set_index(df[df.pipeline == n][col].astype(str)).reindex(cats)
        ax.bar(np.arange(len(cats)) + i * w, d.macro_f05, width=w * 0.92, color=SERIES[i], label=n)
    ax.set_xticks(np.arange(len(cats)) + w * (len(names) - 1) / 2, cats)
    lo = np.nanmin(df.macro_f05) if len(df) else 0
    ax.set_ylim(max(0, lo - 0.05), 1)
    ax.set(ylabel="macro F0.5", title=f"Macro F0.5 by {col}")
    ax.legend(ncol=min(4, len(names)))
    sink.save(fig, f"slice_{col}", f"Macro F0.5 by {col}", "Matcher", df)


# ---- decision layer ---------------------------------------------------------------------------------------
def reliability(sink: FigureSink, name: str, m: dict):
    rel = pd.DataFrame(m.get("curves", {}).get("reliability", []))
    if rel.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [2, 1.2]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.plot(rel.p_mean, rel.frac_pos, color=SERIES[0], marker="o", ms=4)
    ax.set(xlabel="mean predicted probability", ylabel="observed match rate", title=f"Reliability — {name}")
    cal = pd.DataFrame(m.get("calibration", []))
    if not cal.empty:
        ax2 = axes[1]
        ax2.barh(cal.method, cal.nll, color=[SERIES[0] if c == "isotonic" else NEUTRAL_BAR for c in cal.method], height=0.55)
        ax2.set(xlabel="NLL on held-out folds (lower = better)", title="Calibrators")
        ax2.grid(axis="y", visible=False)
    else:
        axes[1].axis("off")
    sink.save(fig, "decision_reliability", f"Reliability diagram and calibrator comparison ({name})", "Decision",
              rel.merge(cal, how="cross") if not cal.empty else rel)


def threshold_curves(sink: FigureSink, metrics: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    for i, (name, m) in enumerate(list(metrics.items())[:6]):
        c = m.get("curves", {}).get("threshold")
        if not c:
            continue
        t = np.array([np.nan if x is None else x for x in c["t"]], float)
        v = np.array(c["macro_f05"], float)
        ok = np.isfinite(t)
        ax.plot(t[ok], v[ok], color=SERIES[i], label=name)
        j = int(np.nanargmax(v))
        if np.isfinite(t[j]):
            ax.scatter([t[j]], [v[j]], color=SERIES[i], s=40, edgecolor="white", zorder=5)
        rows += [{"pipeline": name, "threshold": a, "macro_f05": b} for a, b in zip(t, v)]
    ax.set(xlabel="global threshold on calibrated probability", ylabel="macro F0.5",
           title="Macro F0.5 vs threshold (OOF, descriptive)")
    ax.legend()
    sink.save(fig, "decision_threshold_curves", "Macro F0.5 as a function of the global threshold", "Decision",
              pd.DataFrame(rows), note="A flat top means the chosen threshold is robust.")


def singleton_gate(sink: FigureSink, name: str, m: dict):
    sw = pd.DataFrame(m.get("curves", {}).get("singleton_gate", []))
    if sw.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
    for ax, col, title, c in zip(axes, ["macro_f05", "singleton_gated", "non_singleton_gated"],
                                 ["macro F0.5", "true singletons gated to ∅", "entities with matches gated to ∅"],
                                 [SERIES[0], SERIES[2], SERIES[1]]):
        ax.plot(sw.tau, sw[col], color=c, marker="o", ms=3)
        ax.set(title=title, xlabel="gate τ on q = P(has a match)")
    sink.save(fig, "decision_singleton_gate", f"Singleton gate sweep ({name})", "Decision", sw,
              note="Separate panels, one scale each: gating more singletons trades against losing real matches.")


def entity_f_hist(sink: FigureSink, name: str, e: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(0, 1, 21)
    for i, (lab, m) in enumerate((("singletons", e.g == 0), ("has matches", e.g > 0))):
        ax.hist(e.f05[m], bins=bins, histtype="step", lw=2, color=SERIES[i], label=f"{lab} (n={int(m.sum()):,})")
    ax.set_yscale("log")
    ax.set(xlabel="per-entity F0.5", ylabel="entities (log)", title=f"Where the macro score is made — {name}")
    ax.legend()
    h = e.groupby(pd.cut(e.f05, bins, include_lowest=True), observed=False).size()
    sink.save(fig, "decision_entity_f_hist", f"Per-entity F0.5 distribution ({name})", "Decision",
              h.rename("n").reset_index().astype({"f05": str}))


def f_by_multiplicity(sink: FigureSink, name: str, e: pd.DataFrame):
    t = e.groupby("mult").agg(macro_f05=("f05", "mean"), ceiling=("ceiling_f05", "mean"), n=("f05", "size"))
    t = t.reindex([x for x in ["0", "1", "2", "3-5", "6+"] if x in t.index])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.ceiling, width=0.38, color=NEUTRAL_BAR, label="blocking ceiling")
    ax.bar(x + 0.2, t.macro_f05, width=0.38, color=SERIES[0], label="achieved")
    ax.set_xticks(x, [f"{i}\n(n={n:,})" for i, n in zip(t.index, t.n)])
    ax.set(xlabel="true matches per S1", ylabel="macro F0.5", title=f"Score by match multiplicity — {name}", ylim=(0, 1.02))
    ax.legend(loc="lower left")
    sink.save(fig, "decision_f_by_multiplicity", f"Macro F0.5 by number of true matches ({name})", "Decision",
              t.reset_index())
