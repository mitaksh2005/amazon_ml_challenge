"""Metric sanity checks: `python -m pytest tests` or `python tests/test_metrics.py`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er_eval.metrics import entity_f05, entity_table, loss_decomposition, threshold_curve  # noqa: E402


def test_entity_f05_readme_example():
    # README: predicted 3, 2 correct, gold 2 → 0.714
    assert abs(entity_f05(2, 3, 2) - 1.25 * 2 / (0.25 * 2 + 3)) < 1e-12
    assert abs(float(entity_f05(2, 3, 2)) - 0.7142857) < 1e-6
    assert entity_f05(0, 0, 0) == 1.0      # correct singleton
    assert entity_f05(0, 1, 0) == 0.0      # false merge on a singleton
    assert entity_f05(3, 0, 0) == 0.0      # missed everything


def _brute(q, s, y, g, t):
    k = np.bincount(q[s >= t], minlength=len(g))
    tp = np.bincount(q[(s >= t) & (y == 1)], minlength=len(g))
    return entity_f05(g, k, tp).mean()


def test_threshold_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(1, 30))
        g = rng.integers(0, 4, n)
        m = int(rng.integers(0, 80))
        q = rng.integers(0, n, m)
        y = (rng.random(m) < 0.3).astype(int)
        s = np.round(rng.random(m), 1)            # many ties on purpose
        ts, vs = threshold_curve(q, s, y, g)
        for t, v in zip(ts, vs):
            assert abs(_brute(q, s, y, g, t) - v) < 1e-9, (t, v)
        # no threshold outside the curve beats its best value
        for t in np.linspace(-0.1, 1.1, 25):
            assert _brute(q, s, y, g, t) <= vs.max() + 1e-9


def test_entity_table_and_loss_decomposition():
    ent = pd.DataFrame({"s1_entity_id": ["a", "b", "c", "d"], "g": [0, 2, 1, 3]})
    pairs = pd.DataFrame({"s1_entity_id": ["a", "b", "b", "b", "c"], "label": [0, 1, 0, 1, 0],
                          "pred": [True, True, True, False, False]})
    e = entity_table(ent, pairs)
    assert e.set_index("s1_entity_id").loc["d", "n_cand"] == 0
    L = loss_decomposition(e)
    total = L["singleton_fp"] + L["blocking_miss"] + L["matcher_fn"] + L["matcher_fp"]
    assert abs(total - (1 - e.f05.mean())) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
