"""Feature-space views of what the tree matchers learned (plan §6.6a).

Uses the registry's xsample.parquet: a sample of OOF pairs with their feature values and TreeSHAP
contributions. Embeddings: UMAP if installed, else PCA → t-SNE (scikit-learn)."""
from __future__ import annotations

import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import INK2, OUTCOME_COLORS, FigureSink, outcome


def embed_2d(X: np.ndarray, seed: int = 0) -> tuple[np.ndarray, str]:
    X = np.nan_to_num(np.asarray(X, np.float32))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    if importlib.util.find_spec("umap") is not None:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(X), "UMAP"
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Z = PCA(n_components=min(30, X.shape[1]), random_state=seed).fit_transform(X)
    return TSNE(n_components=2, init="pca", perplexity=40, random_state=seed).fit_transform(Z), "PCA→t-SNE"


def _scatter(ax, Z, cls):
    order = ["TN", "TP", "FN", "FP"]      # draw the rare error classes last so they stay visible
    for c in order:
        m = cls == c
        if m.any():
            ax.scatter(Z[m, 0], Z[m, 1], s=4 if c == "TN" else 7, color=OUTCOME_COLORS[c], linewidths=0,
                       alpha=0.35 if c == "TN" else 0.8, label=f"{c} ({int(m.sum()):,})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(markerscale=2.5, fontsize=8, loc="best")


def shap_space(sink: FigureSink, name: str, xs: pd.DataFrame, n: int = 8000):
    shap_cols = [c for c in xs.columns if c.startswith("shap__")]
    feat_cols = [c[len("shap__"):] for c in shap_cols]
    if not shap_cols:
        return
    s = xs.sample(min(n, len(xs)), random_state=0).reset_index(drop=True)
    cls = outcome(s.label, s.pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    Zs, meth = embed_2d(s[shap_cols].to_numpy())
    _scatter(axes[0], Zs, cls)
    axes[0].set_title(f"TreeSHAP space ({meth})")
    Zx, _ = embed_2d(s[feat_cols].to_numpy())
    _scatter(axes[1], Zx, cls)
    axes[1].set_title(f"Raw feature space ({meth})")
    fig.suptitle(f"Pairs coloured by outcome — {name}", x=0.01, ha="left", fontsize=12, fontweight="bold")
    data = pd.DataFrame({"outcome": cls, "shap_x": Zs[:, 0], "shap_y": Zs[:, 1], "raw_x": Zx[:, 0], "raw_y": Zx[:, 1],
                         "s1_entity_id": s.s1_entity_id, "cand_entity_id": s.cand_entity_id})
    sink.save(fig, "space_shap_vs_raw", f"Where false positives sit: SHAP space vs raw features ({name})",
              "Feature space", data,
              note="Separation that exists in SHAP space but not in raw space was created by the model; "
                   "an FP cluster in SHAP space shares one failure reason.")


def fp_reasons(sink: FigureSink, name: str, xs: pd.DataFrame, top: int = 12):
    shap_cols = [c for c in xs.columns if c.startswith("shap__")]
    if not shap_cols:
        return
    cls = outcome(xs.label, xs.pred)
    fp, tp = xs[cls == "FP"], xs[cls == "TP"]
    if len(fp) < 5 or len(tp) < 5:
        return
    d = (fp[shap_cols].mean() - tp[shap_cols].mean()).sort_values()
    d = pd.concat([d.head(top // 2), d.tail(top // 2)])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2))
    ax.barh([c[len("shap__"):] for c in d.index], d.values, color=["#e34948" if v < 0 else "#2a78d6" for v in d.values],
            height=0.6)
    ax.axvline(0, color=INK2, lw=1)
    ax.set(xlabel="mean contribution: false positives − true positives",
           title=f"What makes false positives look like matches — {name}")
    ax.grid(axis="y", visible=False)
    sink.save(fig, "space_fp_reasons", f"Contribution gap between false and true positives ({name})",
              "Feature space", d.rename("fp_minus_tp").rename_axis("feature").reset_index(),
              note="Blue: features pushing FPs up as much as or more than real matches; red: evidence the model saw against them.")
