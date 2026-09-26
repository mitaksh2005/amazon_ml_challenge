"""Experiment registry: one directory per pipeline run under <work>/experiments/runs/<name>/.

  config.json    pipeline config (+ git sha, sample sizes)
  entity.parquet per-entity OOF counts, F0.5, slices, error type
  pairs.parquet  OOF candidate pairs with raw/calibrated scores and predictions
  metrics.json   headline metrics, bootstrap CIs, per-fold metrics, calibration diagnostics, curves
  ledger.json    wall time and peak memory per stage
  importance.csv feature importance (model pipelines)
  xsample.parquet feature matrix sample with SHAP-style contributions (for feature-space plots)
The index is rebuilt by scanning, so parallel runs never fight over one file."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2))
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def git_sha(repo: Path | None = None) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


class Registry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.runs = self.root / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)

    def dir(self, name: str) -> Path:
        return self.runs / name

    def exists(self, name: str) -> bool:
        return (self.dir(name) / "metrics.json").exists()

    def save(self, name: str, *, config: dict, entity: pd.DataFrame, pairs: pd.DataFrame | None,
             metrics: dict, ledger: dict, extra: dict | None = None) -> Path:
        d = self.dir(name)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "config.json", config)
        entity.to_parquet(d / "entity.parquet", index=False)
        if pairs is not None:
            pairs.to_parquet(d / "pairs.parquet", index=False)
        write_json(d / "ledger.json", ledger)
        for fname, obj in (extra or {}).items():
            if isinstance(obj, pd.DataFrame):
                obj.to_parquet(d / fname, index=False) if fname.endswith(".parquet") \
                    else obj.to_csv(d / fname, index=False)
            else:
                write_json(d / fname, obj)
        write_json(d / "metrics.json", metrics)   # written last: marks the run complete
        return d

    def names(self) -> list[str]:
        return sorted(p.name for p in self.runs.iterdir() if (p / "metrics.json").exists())

    def load(self, name: str, what: str):
        p = self.dir(name) / what
        if what.endswith(".parquet"):
            return pd.read_parquet(p)
        if what.endswith(".csv"):
            return pd.read_csv(p)
        return read_json(p)

    def index(self) -> pd.DataFrame:
        rows = []
        for n in self.names():
            m, c, led = self.load(n, "metrics.json"), self.load(n, "config.json"), self.load(n, "ledger.json")
            row = {"name": n, "parent": c.get("parent"), "kind": c.get("kind"),
                   "description": c.get("description", "")}
            row.update({k: v for k, v in m.get("overall", {}).items() if not isinstance(v, (dict, list))})
            ci = m.get("ci", {}).get("macro_f05", {})
            row["macro_f05_lo"], row["macro_f05_hi"] = ci.get("lo"), ci.get("hi")
            row["wall_s"] = led.get("total_s")
            row["peak_rss_gb"] = led.get("peak_rss_gb")
            rows.append(row)
        return pd.DataFrame(rows)
