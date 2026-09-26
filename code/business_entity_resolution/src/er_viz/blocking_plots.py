"""Blocking analysis (plan §6.1), from the wide dev candidates and the blocking study."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import B0, PASS_NAMES, truncate

from .common import INK2, MUTED, NEUTRAL_BAR, SERIES, FigureSink, seq_cmap


def _gold_total(ent):
    return max(int(ent.g.sum()), 1)


def recall_at_k(sink: FigureSink, wide: pd.DataFrame, ent: pd.DataFrame):
    pos = wide[wide.label == 1]
    total = _gold_total(ent)
    rows = []
    ks = np.arange(1, 51)
    for p in PASS_NAMES:
        r = pos[f"rank_{p}"].to_numpy()
        rows += [{"pass": p, "k": k, "pair_recall": float((r <= k).sum() / total)} for k in ks]
    ranks = pos[[f"rank_{p}" for p in PASS_NAMES]].to_numpy()
    rows += [{"pass": "union (same k per pass)", "k": k, "pair_recall": float((ranks <= k).any(axis=1).sum() / total)} for k in ks]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (name, d) in enumerate(df.groupby("pass", sort=False)):
        ax.plot(d.k, d.pair_recall, color=SERIES[i], label=name)
        ax.annotate(name, (d.k.iloc[-1], d.pair_recall.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.set(xlabel="K (candidates kept per pass)", ylabel="pair recall", title="Blocking recall@K per pass")
    ax.legend(loc="lower right")
    sink.save(fig, "blocking_recall_at_k", "Pair recall vs K for each pass and their union", "Blocking", df)


def pareto(sink: FigureSink, front_file, configs: dict, wide: pd.DataFrame, ent: pd.DataFrame, study=None):
    from er_tune.study import blocking_metrics
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rows = []
    if study is not None:
        trials = [t for t in study.trials if t.values is not None]
        x = [t.user_attrs.get("cands_per_s1") for t in trials]
        y = [t.user_attrs.get("ceiling_f05") for t in trials]
        ax.scatter(x, y, s=14, color=NEUTRAL_BAR, label="trials", zorder=2)
        rows += [{"kind": "trial", "cands_per_s1": a, "ceiling_f05": b} for a, b in zip(x, y)]
    if front_file.exists():
        fr = pd.DataFrame(json.loads(front_file.read_text())["front"]).sort_values("cands_per_s1")
        ax.plot(fr.cands_per_s1, fr.ceiling_f05, color=SERIES[0], marker="o", ms=4, label="Pareto front", zorder=3)
        rows += [{"kind": "front", **r} for r in fr[["cands_per_s1", "ceiling_f05"]].to_dict("records")]
    for i, (name, cfg) in enumerate({"B0": B0, **configs}.items()):
        m = blocking_metrics(wide, ent, cfg)
        ax.scatter([m["cands_per_s1"]], [m["ceiling_f05"]], s=70, color=SERIES[1] if name == "B0" else SERIES[2],
                   edgecolor="white", linewidth=1.5, zorder=4)
        ax.annotate(name, (m["cands_per_s1"], m["ceiling_f05"]), xytext=(6, -10), textcoords="offset points", fontsize=9)
        rows.append({"kind": name, **m})
    ax.set(xlabel="mean candidates per S1", ylabel="blocking ceiling F0.5",
           title="Recall/cost trade-off of blocking configurations")
    ax.legend(loc="lower right")
    sink.save(fig, "blocking_pareto", "Blocking ceiling F0.5 vs candidates per S1", "Blocking", pd.DataFrame(rows),
              note="Grey: Optuna NSGA-II trials; blue: Pareto front; labelled points: configs carried into the search.")


def pass_combinations(sink: FigureSink, wide: pd.DataFrame):
    c = truncate(wide, B0)
    pos = c[c.label == 1]
    found = pos[[f"score_{p}" for p in PASS_NAMES]].notna()
    combo = found.apply(lambda r: " + ".join(p for p, f in zip(PASS_NAMES, r) if f), axis=1)
    cnt = combo.value_counts().rename_axis("passes").reset_index(name="true_pairs")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(cnt) + 1.2))
    ax.barh(cnt.passes[::-1], cnt.true_pairs[::-1], color=SERIES[0], height=0.6)
    ax.set(xlabel="true pairs retrieved (B0)", title="Which passes retrieve each true pair")
    ax.grid(axis="y", visible=False)
    sink.save(fig, "blocking_pass_combinations", "True pairs by combination of retrieving passes (B0)", "Blocking", cnt,
              note="Pairs found only by one pass show that pass's unique contribution.")


def leave_one_out(sink: FigureSink, wide: pd.DataFrame, ent: pd.DataFrame):
    from er_tune.study import blocking_metrics
    base = blocking_metrics(wide, ent, B0)
    rows = []
    for p in PASS_NAMES:
        cfg = {k: dict(v) for k, v in B0.items()}
        cfg[p]["k"] = 0
        m = blocking_metrics(wide, ent, cfg)
        rows.append({"dropped_pass": p, "delta_ceiling_f05": m["ceiling_f05"] - base["ceiling_f05"],
                     "delta_cands": m["cands_per_s1"] - base["cands_per_s1"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(df.dropped_pass, df.delta_ceiling_f05, color=SERIES[7], height=0.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set(xlabel="Δ blocking ceiling F0.5 when the pass is removed", title="Leave-one-pass-out (B0)")
    ax.grid(axis="y", visible=False)
    sink.save(fig, "blocking_leave_one_out", "Ceiling loss when each pass is dropped", "Blocking", df)


def rank_hist(sink: FigureSink, wide: pd.DataFrame):
    pos = wide[wide.label == 1]
    fig, axes = plt.subplots(1, len(PASS_NAMES), figsize=(10, 3), sharey=True)
    rows = []
    for ax, p in zip(axes, PASS_NAMES):
        r = pos[f"rank_{p}"].dropna()
        cnt = r.value_counts().sort_index()
        ax.bar(cnt.index, cnt.values, color=SERIES[0], width=0.8)
        ax.set_yscale("log")
        ax.set(title=f"{p}", xlabel="rank of true match")
        k = B0[p]["k"]
        ax.axvline(k + 0.5, color=SERIES[1], lw=1.5)
        ax.annotate(f"B0 k={k}", (k + 0.5, ax.get_ylim()[1]), xytext=(3, -12), textcoords="offset points",
                    color=INK2, fontsize=8)
        rows += [{"pass": p, "rank": int(i), "true_pairs": int(v)} for i, v in cnt.items()]
    axes[0].set_ylabel("true pairs (log)")
    sink.save(fig, "blocking_rank_hist", "Rank of true matches within each pass", "Blocking", pd.DataFrame(rows),
              note="Mass just past the orange line = recall a larger k buys; missing mass = needs a new channel.")


def candidate_counts(sink: FigureSink, wide: pd.DataFrame, ent: pd.DataFrame):
    c = truncate(wide, B0)
    n = c.groupby("s1_entity_id").size().reindex(ent.s1_entity_id).fillna(0)
    df = pd.DataFrame({"country": ent.country.to_numpy(), "n": n.to_numpy()})
    countries = df.country.value_counts().index.tolist()[:3]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, df.n.max() + 2)
    for i, cty in enumerate(countries):
        ax.hist(df.n[df.country == cty], bins=bins, histtype="step", lw=2, color=SERIES[i], label=cty)
    for q in (0.95, 0.99):
        v = df.n.quantile(q)
        ax.axvline(v, color=MUTED, lw=1, ls="--")
        ax.annotate(f"p{int(q * 100)}={v:.0f}", (v, ax.get_ylim()[1] * 0.9), xytext=(3, 0), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set(xlabel="candidates per S1 (B0)", ylabel="S1 entities", title="Candidate-set size distribution")
    ax.legend()
    sink.save(fig, "blocking_candidate_counts", "Candidates per S1 by country (B0)", "Blocking",
              df.groupby("country").n.describe().reset_index())


def miss_taxonomy(sink: FigureSink, misses: pd.DataFrame):
    if misses is None or misses.empty:
        return
    t = misses.groupby(["level", "reason"]).size().unstack(fill_value=0)
    t = t.div(t.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    left = np.zeros(len(t))
    for i, col in enumerate(t.columns[:8]):
        ax.barh(t.index, t[col], left=left, color=SERIES[i], label=col, height=0.55, edgecolor="#fcfcfb", linewidth=2)
        left += t[col].to_numpy()
    ax.set(xlabel="share of missed true pairs", title="Why blocking misses true pairs")
    ax.legend(ncol=4, bbox_to_anchor=(0, -0.35), loc="upper left")
    ax.grid(axis="y", visible=False)
    sink.save(fig, "blocking_miss_taxonomy", "Missed true pairs by heuristic reason", "Blocking",
              t.reset_index(), note="'wide' = not in the widest candidate set; 'B0' = missed by the base config.")


def recall_by_slice(sink: FigureSink, wide: pd.DataFrame, ent: pd.DataFrame, gold_src: pd.DataFrame | None):
    c = truncate(wide, B0)
    if gold_src is None or gold_src.empty:
        return
    found = c[c.label == 1].groupby(["s1_entity_id", "source"]).size().rename("found")
    g = gold_src.merge(ent[["s1_entity_id", "country"]], on="s1_entity_id")
    g = g.merge(found, left_on=["s1_entity_id", "source"], right_index=True, how="left").fillna({"found": 0})
    t = g.groupby(["country", "source"]).agg(found=("found", "sum"), gold=("n", "sum"))
    t["recall"] = t.found / t.gold
    piv = t.recall.unstack()
    fig, ax = plt.subplots(figsize=(4.5, 0.6 * len(piv) + 1.5))
    im = ax.imshow(piv.to_numpy(), cmap=seq_cmap(), vmin=max(0.0, float(np.nanmin(piv.to_numpy())) - 0.02), vmax=1)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color="white" if v > np.nanmean(piv.to_numpy()) else "#0b0b0b")
    ax.grid(False)
    ax.set_title("B0 pair recall by country × source")
    fig.colorbar(im, ax=ax, shrink=0.8)
    sink.save(fig, "blocking_recall_by_slice", "Pair recall by country and source (B0)", "Blocking", t.reset_index())
