"""Shared plotting style (reference palette of the dataviz method) and figure saving.

Every figure is saved as PNG plus a CSV of the plotted numbers (the table view)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
# categorical slots, fixed order (never cycled; > 8 series fold into "Other" or facets)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"
NEUTRAL_BAR = "#b9b8b2"
HIGHLIGHT = SERIES[0]
BASELINE = SERIES[1]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "lines.markersize": 6, "font.size": 10, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "figure.dpi": 110,
})


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", SEQ)


class FigureSink:
    """Saves figures + their data under one directory and keeps an index for the HTML report."""

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.items: list[dict] = []

    def save(self, fig, name: str, title: str, section: str, data: pd.DataFrame | None = None,
             note: str = "") -> Path:
        p = self.dir / f"{name}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        if data is not None:
            data.to_csv(self.dir / f"{name}.csv", index=False)
        self.items.append({"file": p.name, "title": title, "section": section, "note": note,
                           "csv": f"{name}.csv" if data is not None else None})
        return p

    def html(self, name: str, title: str, section: str, body: str, note: str = "") -> Path:
        p = self.dir / f"{name}.html"
        p.write_text(page(title, body))
        self.items.append({"file": p.name, "title": title, "section": section, "note": note, "csv": None, "html": True})
        return p


def page(title: str, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--surface:{SURFACE};--ink:{INK};--ink2:{INK2};--grid:{GRID}}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#383835}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#383835}}
body{{background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}}
h1,h2{{font-weight:600}} .note{{color:var(--ink2)}} img{{max-width:100%;height:auto;border:1px solid var(--grid);border-radius:6px;background:#fcfcfb}}
table{{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}} td,th{{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:left;vertical-align:top}}
figure{{margin:24px 0}} figcaption{{font-weight:600;margin-bottom:6px}}
</style></head><body>{body}</body></html>"""


def pipeline_colors(names: list[str], highlight: str | None = None, baseline: str | None = None) -> list[str]:
    return [HIGHLIGHT if n == highlight else BASELINE if n == baseline else NEUTRAL_BAR for n in names]


def outcome(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    label, pred = np.asarray(label).astype(bool), np.asarray(pred).astype(bool)
    return np.where(label & pred, "TP", np.where(~label & pred, "FP", np.where(label & ~pred, "FN", "TN")))


# scatter-type charts: only the first three slots validate all-pairs, the fourth class is neutral
OUTCOME_COLORS = {"TP": SERIES[0], "FP": SERIES[1], "FN": SERIES[2], "TN": NEUTRAL_BAR}
