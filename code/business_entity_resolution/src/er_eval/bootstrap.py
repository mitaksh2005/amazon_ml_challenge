"""Entity-level bootstrap. Resampling uses a fixed seed, so two pipelines evaluated on the same
entity order get the same resamples; `paired_delta` bootstraps the per-entity difference directly."""
from __future__ import annotations

import numpy as np


def bootstrap_mean(x: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                   batch: int = 50) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan"),
                "p_le_0": float("nan")}
    rng = np.random.default_rng(seed)
    means = []
    for b in range(0, n_boot, batch):
        idx = rng.integers(0, n, size=(min(batch, n_boot - b), n))
        means.append(x[idx].mean(axis=1))
    means = np.concatenate(means)
    return {"mean": float(x.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "se": float(means.std(ddof=1)),
            "p_le_0": float((means <= 0).mean())}


def paired_delta(a: np.ndarray, b: np.ndarray, **kw) -> dict:
    """Bootstrap CI of mean(a - b) over the same entities (a, b aligned)."""
    return bootstrap_mean(np.asarray(a, np.float64) - np.asarray(b, np.float64), **kw)
