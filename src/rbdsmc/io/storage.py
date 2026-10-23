"""Run storage: one self-contained folder per run.

    results/runs/<run_id>/
        config.yaml               exact input
        meta.json                 run id, code version, seed, wall time, backend
        timeseries.{parquet|csv}  one row per recorded time
        band_populations.{...}    long format: t_ms, n, fraction
        loss_events.{...}         one row per lost atom
        summary.json              fits, final fractions, checks (+ summary.{parquet|csv}, 1 row)

Tables are written as Parquet when ``pyarrow`` is installed, otherwise as CSV.  DuckDB reads
both with the same SQL, e.g.  SELECT * FROM 'results/runs/*/summary.parquet'.
The run id is deterministic: <label>_<config-hash>_s<seed>, so rerunning the same
configuration overwrites its folder.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
    BACKEND = "parquet"
except ImportError:        # pragma: no cover - depends on the environment
    BACKEND = "csv"


def run_id_for(cfg) -> str:
    label = cfg.output.label or "run"
    return f"{label}_{cfg.hash()}_s{cfg.numerics.seed}"


def git_version(root) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


class RunWriter:
    def __init__(self, run_dir, backend: str | None = None):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend or BACKEND

    def table(self, name: str, data) -> Path:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        for other in ("parquet", "csv"):          # never leave a stale twin behind
            p = self.dir / f"{name}.{other}"
            if p.exists():
                p.unlink()
        if self.backend == "parquet":
            path = self.dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = self.dir / f"{name}.csv"
            df.to_csv(path, index=False, float_format="%.10g")
        return path

    def json(self, name: str, obj) -> Path:
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(_jsonable(obj), indent=2))
        return path

    def config(self, cfg) -> Path:
        path = self.dir / "config.yaml"
        cfg.dump_yaml(path)
        return path


def read_table(run_dir, name: str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    p = run_dir / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = run_dir / f"{name}.csv"
    if p.exists():
        return pd.read_csv(p)
    raise FileNotFoundError(f"no table {name!r} in {run_dir}")


def read_json(run_dir, name: str) -> dict:
    return json.loads((Path(run_dir) / f"{name}.json").read_text())
