"""Pipeline comparison (plan §6.4–6.5) from the experiment registry."""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from er_eval import paired_delta

from .common import (BASELINE, DIV_MID, HIGHLIGHT, INK2, MUTED, NEUTRAL_BAR, SERIES, FigureSink, div_cmap, seq_cmap,
                     pipeline_colors)

BASELINES = {"P-empty", "P-oracle"}


def leaderboard(sink: FigureSink, idx: pd.DataFrame, best: str, ref: str = "P0"):
    full = idx.sort_values("macro_f05").reset_index(drop=True)
    # trivial baselines far below the rest (e.g. P-empty) would flatten the scale: CSV only
    floor = full.macro_f05.median() - 0.25
    d = full[full.macro_f05 >= floor].reset_index(drop=True)
    dropped = full[full.macro_f05 < floor]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(d) + 1.4))
    colors = pipeline_colors(d.name.tolist(), highlight=best, baseline=ref)
    ax.barh(d.name, d.macro_f05, color=colors, height=0.62)
    err = np.vstack([d.macro_f05 - d.macro_f05_lo, d.macro_f05_hi - d.macro_f05]).clip(0)
    ax.errorbar(d.macro_f05, d.name, xerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    lo = np.nanmin(d.macro_f05_lo.fillna(d.macro_f05))
    ax.set_xlim(max(0, lo - 0.02), np.nanmax(d.macro_f05_hi.fillna(d.macro_f05)) + 0.012)
    for y, (v, hi) in enumerate(zip(d.macro_f05, d.macro_f05_hi.fillna(d.macro_f05))):
        ax.text(max(v, hi), y, f"  {v:.4f}", va="center", fontsize=8, color=INK2)   # right of the CI whisker
    ax.set(xlabel="OOF macro F0.5 (95% entity-bootstrap CI)", title="Pipeline leaderboard")
    ax.grid(axis="y", visible=False)
    note = ("Not drawn (far below the rest): " + ", ".join(f"{n} = {v:.4f}" for n, v in zip(dropped.name, dropped.macro_f05))
            if len(dropped) else "")
    sink.save(fig, "compare_leaderboard", "Leaderboard (blue = selected, orange = reference P0)", "Comparison", full,
              note=note)


def forest(sink: FigureSink, idx: pd.DataFrame, entities: dict[str, pd.DataFrame], admission: float = 0.001):
    rows = []
    for _, r in idx.iterrows():
        par = r.parent
        if not isinstance(par, str) or par not in entities or r["name"] not in entities:
            continue
        a = entities[r["name"]].set_index("s1_entity_id").f05
        b = entities[par].set_index("s1_entity_id").f05
        common = a.index.intersection(b.index)
        d = paired_delta(a[common].to_numpy(), b[common].to_numpy())
        rows.append({"pipeline": r["name"], "parent": par, "delta": d["mean"], "lo": d["lo"], "hi": d["hi"],
                     "p_le_0": d["p_le_0"], "admitted": bool(d["lo"] > 0 and d["mean"] >= admission)})
    if not rows:
        return pd.DataFrame()
    all_rows = pd.DataFrame(rows)
    # changes measured against rule/empty baselines are huge and would flatten every other interval
    df = all_rows[~all_rows.parent.isin(["P-rule", "P-empty", "P-oracle"])].sort_values("delta").reset_index(drop=True)
    if df.empty:
        df = all_rows.sort_values("delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(df) + 1.4))
    y = np.arange(len(df))
    col = [SERIES[0] if a else NEUTRAL_BAR for a in df.admitted]
    ax.errorbar(df.delta, y, xerr=np.vstack([df.delta - df.lo, df.hi - df.delta]).clip(0), fmt="none",
                ecolor=MUTED, elinewidth=1.5, capsize=2)
    ax.scatter(df.delta, y, color=col, s=40, zorder=3, edgecolor="white", linewidth=1)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(admission, color=SERIES[1], lw=1, ls="--")
    ax.annotate(f"admission +{admission}", (admission, -0.6), xytext=(3, 0), textcoords="offset points",
                fontsize=8, color=INK2, va="top")
    ax.set_yticks(y, [f"{p}  vs  {q}" for p, q in zip(df.pipeline, df.parent)])
    ax.set(xlabel="paired Δ macro F0.5 vs parent (95% CI)", title="What each change buys")
    ax.grid(axis="y", visible=False)
    base = all_rows[all_rows.parent.isin(["P-rule", "P-empty", "P-oracle"])]
    note = "; ".join(f"{r.pipeline} vs {r.parent}: {r.delta:+.4f}" for r in base.itertuples())
    sink.save(fig, "compare_forest", "Paired bootstrap Δ vs parent (blue = passes the gate)", "Comparison", all_rows,
              note=("Not drawn (vs trivial baselines): " + note) if note else "")
    return all_rows


def ablation_waterfall(sink: FigureSink, idx: pd.DataFrame, best: str):
    by = idx.set_index("name")
    chain = [best]
    while isinstance(by.parent.get(chain[-1]), str) and by.parent.get(chain[-1]) in by.index and len(chain) < 20:
        chain.append(by.parent[chain[-1]])
    chain = chain[::-1]
    if len(chain) < 2:
        return
    vals = by.macro_f05.reindex(chain).to_numpy()
    deltas = np.r_[vals[0], np.diff(vals)]
    fig, ax = plt.subplots(figsize=(1.1 * len(chain) + 2, 3.8))
    bottom = np.r_[0, vals[:-1]]
    colors = [NEUTRAL_BAR] + [SERIES[0] if d >= 0 else SERIES[7] for d in deltas[1:]]
    ax.bar(range(len(chain)), np.where(np.arange(len(chain)) == 0, vals[0], deltas),
           bottom=np.where(np.arange(len(chain)) == 0, 0, bottom), color=colors, width=0.6)
    for i, (v, d) in enumerate(zip(vals, deltas)):
        ax.text(i, v, f"{v:.4f}" if i == 0 else f"{d:+.4f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(chain)), chain, rotation=20, ha="right")
    ax.set_ylim(max(0, np.nanmin(vals) - 0.03), min(1, np.nanmax(vals) + 0.01))
    ax.set(ylabel="OOF macro F0.5", title=f"Ablation path to {best}")
    sink.save(fig, "compare_ablation_waterfall", "Cumulative gains along the parent chain of the selected pipeline",
              "Comparison", pd.DataFrame({"pipeline": chain, "macro_f05": vals, "delta": deltas}))


def pareto_time(sink: FigureSink, idx: pd.DataFrame, best: str):
    d = idx.dropna(subset=["wall_s"])
    d = d[~d.name.isin(BASELINES)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(d.wall_s / 60, d.macro_f05, color=[HIGHLIGHT if n == best else NEUTRAL_BAR for n in d.name], s=40,
               edgecolor="white", zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["name"], (r.wall_s / 60, r.macro_f05), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK2)
    srt = d.sort_values("wall_s")
    front = srt[srt.macro_f05 >= srt.macro_f05.cummax()]
    ax.step(front.wall_s / 60, front.macro_f05, where="post", color=SERIES[0], lw=1.5)
    ax.set(xlabel="evaluation wall time (minutes, 5 folds)", ylabel="OOF macro F0.5", title="Score vs cost")
    sink.save(fig, "compare_pareto_time", "Macro F0.5 vs wall time (step = Pareto front)", "Comparison",
              d[["name", "macro_f05", "wall_s", "cands_per_s1_mean"]])


def slice_heatmap(sink: FigureSink, entities: dict[str, pd.DataFrame], ref: str = "P0"):
    if ref not in entities:
        return
    cols = ["country", "mult", "name_freq_bin", "has_addr"]
    base = entities[ref]
    rows = []
    for name, e in entities.items():
        if name in BASELINES or name == ref:
            continue
        for c in cols:
            if c not in e:
                continue
            a = e.groupby(c).f05.mean()
            b = base.groupby(c).f05.mean()
            for k in a.index:
                rows.append({"pipeline": name, "slice": f"{c}={k}", "delta": a[k] - b.get(k, np.nan)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    piv = df.pivot(index="pipeline", columns="slice", values="delta")
    v = np.nanmax(np.abs(piv.to_numpy())) or 1e-3
    fig, ax = plt.subplots(figsize=(0.55 * piv.shape[1] + 3, 0.4 * piv.shape[0] + 1.8))
    im = ax.imshow(piv.to_numpy(), cmap=div_cmap(), vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"Δ macro F0.5 vs {ref}")
    ax.set_title(f"Where each pipeline wins or loses vs {ref}")
    sink.save(fig, "compare_slice_heatmap", f"Δ macro F0.5 vs {ref} per slice (blue = better, red = worse)",
              "Comparison", df, note=f"Neutral grey ({DIV_MID}) = no change.")


def fold_lines(sink: FigureSink, metrics: dict[str, dict], top: list[str]):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    rows = []
    for i, n in enumerate(top[:6]):
        f = pd.DataFrame(metrics[n]["folds"])
        ax.plot(f.fold, f.macro_f05, color=SERIES[i], marker="o", ms=4, label=n)
        rows += [{"pipeline": n, "fold": a, "macro_f05": b} for a, b in zip(f.fold, f.macro_f05)]
    ax.set(xlabel="outer fold", ylabel="macro F0.5", title="Fold-to-fold stability of the top pipelines")
    ax.set_xticks(sorted({r["fold"] for r in rows}))
    ax.legend(fontsize=8)
    df = pd.DataFrame(rows)
    ranks = df.pivot(index="fold", columns="pipeline", values="macro_f05").rank(axis=1, ascending=False)
    sink.save(fig, "compare_fold_lines", "Per-fold macro F0.5 (crossing lines = unstable ranking)", "Comparison",
              df.merge(ranks.stack().rename("rank_in_fold").reset_index(), on=["fold", "pipeline"]))


def entity_scatter(sink: FigureSink, a_name: str, a: pd.DataFrame, b_name: str, b: pd.DataFrame):
    m = a[["s1_entity_id", "f05"]].merge(b[["s1_entity_id", "f05"]], on="s1_entity_id", suffixes=("_a", "_b"))
    better, worse = (m.f05_a > m.f05_b + 1e-9).sum(), (m.f05_a < m.f05_b - 1e-9).sum()
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    hb = ax.hexbin(m.f05_b, m.f05_a, gridsize=25, bins="log", cmap=seq_cmap(), mincnt=1)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.set(xlabel=f"per-entity F0.5: {b_name}", ylabel=f"per-entity F0.5: {a_name}",
           title=f"{a_name} better on {better:,}, worse on {worse:,}")
    ax.grid(False)
    fig.colorbar(hb, ax=ax, shrink=0.8, label="entities (log)")
    flips = m[(m.f05_a - m.f05_b).abs() > 1e-9]
    sink.save(fig, "compare_entity_scatter", f"Per-entity agreement: {a_name} vs {b_name}", "Comparison", flips)


def xcountry(sink: FigureSink, rows: dict[str, list[dict]], within: dict[str, dict]):
    data = []
    for name, rr in rows.items():
        for r in rr:
            ins = next((s["macro_f05"] for s in within.get(name, {}).get("slices", {}).get("country", [])
                        if s["country"] == r["eval"]), np.nan)
            data.append({**r, "pipeline": name, "in_country_oof": ins})
    if not data:
        return
    df = pd.DataFrame(data)
    df["transfer"] = df["train"] + " → " + df["eval"]
    names = df.pipeline.unique().tolist()
    cats = df.transfer.unique().tolist()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.8 / len(names)
    for i, n in enumerate(names[:6]):
        d = df[df.pipeline == n].set_index("transfer").reindex(cats)
        ax.bar(np.arange(len(cats)) + i * w, d.macro_f05, width=w * 0.9, color=SERIES[i], label=n)
        ax.scatter(np.arange(len(cats)) + i * w, d.in_country_oof, marker="_", s=200, color=INK2, zorder=4)
    ax.set_xticks(np.arange(len(cats)) + w * (len(names[:6]) - 1) / 2, cats)
    ax.set_ylim(max(0, np.nanmin(df.macro_f05) - 0.05), 1)
    ax.set(ylabel="macro F0.5", title="Cross-country transfer (France proxy); dash = in-country OOF")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    sink.save(fig, "compare_xcountry", "Train on one country, evaluate on another", "Comparison", df)


# ---- error decomposition (plan §6.4) ----------------------------------------------------------------------
LOSS_KEYS = ["singleton_fp", "blocking_miss", "matcher_fn", "matcher_fp"]
LOSS_LABELS = {"singleton_fp": "singleton false positive", "blocking_miss": "blocking miss",
               "matcher_fn": "matcher false negative", "matcher_fp": "matcher false positive"}


def loss_bars(sink: FigureSink, metrics: dict[str, dict]):
    rows = [{"pipeline": n, **{k: m["loss"][k] for k in LOSS_KEYS}, "achieved": m["loss"]["achieved"]}
            for n, m in metrics.items() if n not in BASELINES]
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("achieved")
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(df) + 1.6))
    left = np.zeros(len(df))
    for i, k in enumerate(LOSS_KEYS):
        ax.barh(df.pipeline, df[k], left=left, color=SERIES[i], label=LOSS_LABELS[k], height=0.6,
                edgecolor="#fcfcfb", linewidth=2)
        left += df[k].to_numpy()
    ax.set(xlabel="lost macro F0.5 (1 − score), by cause", title="Where each pipeline loses score")
    ax.legend(ncol=2, bbox_to_anchor=(0, -0.12), loc="upper left", fontsize=8)
    ax.grid(axis="y", visible=False)
    sink.save(fig, "errors_loss_decomposition", "Loss decomposition per pipeline", "Errors", df)


def loss_waterfall(sink: FigureSink, name: str, m: dict):
    L = m["loss"]
    steps = [("perfect", 1.0)] + [(LOSS_LABELS[k], -L[k]) for k in LOSS_KEYS]
    vals = np.cumsum([s[1] for s in steps])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (lab, d) in enumerate(steps):
        if i == 0:
            ax.bar(i, 1.0, color=NEUTRAL_BAR, width=0.6)
        else:
            ax.bar(i, d, bottom=vals[i - 1], color=SERIES[i - 1], width=0.6)
            ax.text(i, vals[i - 1] + d, f"{d:+.4f}", ha="center", va="top", fontsize=8, color=INK2)
    ax.bar(len(steps), L["achieved"], color=HIGHLIGHT, width=0.6)
    ax.text(len(steps), L["achieved"], f"{L['achieved']:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(steps) + 1), [s[0] for s in steps] + ["achieved"], rotation=15, ha="right", fontsize=8)
    ax.set_ylim(max(0, L["achieved"] - 0.05), 1.005)
    ax.set(ylabel="macro F0.5", title=f"From a perfect score to {name}")
    sink.save(fig, "errors_waterfall", f"Loss waterfall ({name})", "Errors",
              pd.DataFrame([{"step": s[0], "delta": s[1]} for s in steps] + [{"step": "achieved", "delta": L["achieved"]}]))


def error_heatmap(sink: FigureSink, name: str, e: pd.DataFrame):
    rows = []
    for c in ("country", "mult", "name_freq_bin"):
        if c in e:
            t = pd.crosstab(e[c], e.error_type, normalize="index")
            for k in t.index:
                for et in t.columns:
                    rows.append({"slice": f"{c}={k}", "error_type": et, "share": t.loc[k, et]})
    df = pd.DataFrame(rows)
    df = df[df.error_type != "correct"]
    if df.empty:
        return
    piv = df.pivot(index="slice", columns="error_type", values="share").fillna(0)
    fig, ax = plt.subplots(figsize=(0.9 * piv.shape[1] + 3, 0.35 * piv.shape[0] + 1.5))
    from .common import seq_cmap
    im = ax.imshow(piv.to_numpy(), cmap=seq_cmap(), aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of entities in slice")
    ax.set_title(f"Error types by slice — {name}")
    sink.save(fig, "errors_by_slice", f"Share of entities per error type and slice ({name})", "Errors", df)


def error_gallery(sink: FigureSink, name: str, pairs: pd.DataFrame, records: pd.DataFrame, n: int = 40):
    fp = pairs[(pairs.pred) & (pairs.label == 0)].sort_values("p", ascending=False).head(n)
    fn = pairs[(~pairs.pred) & (pairs.label == 1)].sort_values("p", ascending=False).head(n)
    rec = records.set_index("entity_id")

    def table(df, title):
        rows = []
        for _, r in df.iterrows():
            a = rec.loc[r.s1_entity_id] if r.s1_entity_id in rec.index else None
            b = rec.loc[r.cand_entity_id] if r.cand_entity_id in rec.index else None
            cell = lambda x, c: html.escape(str(x[c])) if x is not None else ""   # noqa: E731
            rows.append(f"<tr><td>{r.p:.3f}</td><td>{cell(a, 'business_name')}<br><small>{cell(a, 'business_address')}</small></td>"
                        f"<td>{cell(b, 'business_name')}<br><small>{cell(b, 'business_address')}</small></td>"
                        f"<td>{html.escape(str(r.s1_entity_id))}<br>{html.escape(str(r.cand_entity_id))}</td></tr>")
        return (f"<h2>{title}</h2><table><tr><th>p</th><th>Source 1</th><th>candidate</th><th>ids</th></tr>"
                + "".join(rows) + "</table>")

    body = (f"<h1>Error gallery — {html.escape(name)}</h1><p class='note'>Most confident mistakes on out-of-fold "
            f"predictions. Use them to find cleaning or feature gaps.</p>"
            + table(fp, "Most confident false positives") + table(fn, "Highest-scored missed matches"))
    sink.html("errors_gallery", f"Error gallery ({name})", "Errors", body)
