"""Stitch every saved figure into one self-contained-ish HTML report (images referenced relatively)."""
from __future__ import annotations

import html

import pandas as pd

from .common import FigureSink, page

SECTIONS = ["Comparison", "Errors", "Blocking", "Matcher", "Decision", "Feature space", "Tuning", "SAE"]


def write(sink: FigureSink, leaderboard: pd.DataFrame, best: str, extra_html: str = "") -> str:
    cols = [c for c in ["name", "parent", "macro_f05", "macro_f05_lo", "macro_f05_hi", "ceiling_f05",
                        "singleton_accuracy", "fp_per_1000_singletons", "pair_precision", "pair_recall",
                        "cands_per_s1_mean", "wall_s"] if c in leaderboard]
    lb = leaderboard.sort_values("macro_f05", ascending=False)[cols]
    table = lb.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0)
    body = [f"<h1>Entity-resolution pipeline comparison</h1><p class='note'>Selected pipeline: "
            f"<b>{html.escape(best)}</b>. All scores are out-of-fold on the dev sample (5 grouped folds) unless stated. "
            f"Every figure has a CSV of its numbers next to it.</p>", extra_html, "<h2>Leaderboard</h2>", table]
    for sec in SECTIONS:
        items = [i for i in sink.items if i["section"] == sec]
        if not items:
            continue
        body.append(f"<h2>{sec}</h2>")
        for i in items:
            link = f" · <a href='{i['csv']}'>data</a>" if i.get("csv") else ""
            note = f"<div class='note'>{html.escape(i['note'])}</div>" if i.get("note") else ""
            if i.get("html"):
                body.append(f"<figure><figcaption><a href='{i['file']}'>{html.escape(i['title'])}</a></figcaption>{note}</figure>")
            else:
                body.append(f"<figure><figcaption>{html.escape(i['title'])}{link}</figcaption>{note}"
                            f"<img src='{i['file']}' alt='{html.escape(i['title'])}'></figure>")
    out = sink.dir / "index.html"
    out.write_text(page("Pipeline comparison", "\n".join(body)))
    return str(out)
