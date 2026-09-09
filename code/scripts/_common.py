"""Shared script plumbing: import path, run directories, logging, provenance."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def results_root() -> Path:
    """Where run outputs go.

    This prefers an explicit override, then an optional project scratch root, then a local fallback.
    """
    env = os.environ.get("SOCIAL_COLLUSION_RUNS")
    if env:
        return Path(env)
    blue = os.environ.get("SIGCOL_BLUE")
    if blue and Path(blue).exists():
        return Path(blue) / "runs" / "social_collusion"
    return REPO / "results"


def run_dir(name: str, make: bool = True) -> Path:
    d = results_root() / name
    if make:
        d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_clean(obj), fh, indent=2, sort_keys=True)
    return path


def _clean(o: Any) -> Any:
    import numpy as np

    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (o != o):
        return None
    return o


def save_csv(rows: list[dict], path: str | Path) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return path


def write_provenance(d: str | Path, extra: dict | None = None) -> Path:
    from social_collusion import runmeta

    return runmeta.write(d, extra)


def limit_threads(n: int = 4) -> None:
    """Keep BLAS/torch from spawning a thread per core on a shared node."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)
