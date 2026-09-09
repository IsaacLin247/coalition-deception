"""Shared plumbing for the extension experiments (multigen, dependence, theory figures).

Re-exports the run-directory / provenance helpers of `scripts/_common.py` so the new experiments
write results exactly where the F1-F4 launchers do (`$SOCIAL_COLLUSION_RUNS`, default
`results/`), and adds one plotting style used by every new figure: matplotlib only, vector PDF
plus a PNG preview, colours matching the published F1-F4 figures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from _common import (  # noqa: E402,F401
    banner,
    limit_threads,
    results_root,
    run_dir,
    save_csv,
    save_json,
    write_provenance,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """CSV rows with numeric coercion (bools become 0/1 floats, empty cells become None)."""
    import csv

    rows: list[dict[str, Any]] = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if v is None or v == "":
                    row[k] = None
                    continue
                low = str(v).lower()
                if low in ("true", "false"):
                    row[k] = 1.0 if low == "true" else 0.0
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
#: colours shared with analysis/analyze_tenseed_sweep.py and visualization/static_plot.py
COLORS = {
    "coalition": "#c62828",
    "crew": "#1565c0",
    "C0": "#90a4ae",
    "D": "#1565c0",
    "C": "#c62828",
    "none": "#546e7a",
    "dependence": "#00897b",
    "mean": "#8e44ad",
    "median": "#2e9d52",
    "trimmed": "#2166c2",
    "soft_credibility": "#e0a900",
    "sharp_credibility": "#d62728",
    "dependence_aware": "#00897b",
    "hypothesis": "#37474f",
    "credibility": "#e0a900",
    "credibility_dependence": "#00897b",
    "note": "#546e7a",
}
RULE_LABELS = {
    "mean": "mean",
    "median": "median",
    "trimmed": "trimmed mean",
    "soft_credibility": "soft credibility",
    "sharp_credibility": "sharp credibility",
    "dependence_aware": "dependence-aware credibility",
    "hypothesis": "hypothesis elimination",
}


def use_paper_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 180,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # embed TrueType so LaTeX / Illustrator can edit text
            "ps.fonttype": 42,
        }
    )


def save_fig(fig, stem: str | Path, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Save `stem.pdf` (vector, for LaTeX) and `stem.png` (preview); returns the paths."""
    import matplotlib.pyplot as plt

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def mean_sd(values) -> tuple[float, float]:
    import numpy as np

    x = np.asarray([v for v in values if v is not None], dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0


__all__ = [
    "REPO",
    "COLORS",
    "RULE_LABELS",
    "banner",
    "limit_threads",
    "results_root",
    "run_dir",
    "save_csv",
    "save_json",
    "write_provenance",
    "load_yaml",
    "read_csv",
    "use_paper_style",
    "save_fig",
    "mean_sd",
]
