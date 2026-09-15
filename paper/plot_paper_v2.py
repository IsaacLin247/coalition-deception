#!/usr/bin/env python3
"""Draw two readable v2 figures from the completed, validated seed-level data.

This is a presentation-only script: no games, training, resampling, or statistical
tests are run. Original manuscript figures are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(10))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(data: Path, out: Path, manifest_path: Path) -> dict:
    names = (
        "status.json", "cell_summary.json", "per_seed.json",
        "publication_inference_verification.json",
    )
    inputs = {name: json.loads((data / name).read_text()) for name in names}
    status = inputs["status.json"]
    if (
        status["status"] != "complete" or not status["inference_enabled"]
        or status["completed_jobs"] != 240 or status["expected_jobs"] != 240
        or status["problems"] or status["missing_comparisons"]
    ):
        raise ValueError("Figures require the complete 240-job analysis")
    receipt = inputs["publication_inference_verification.json"]
    if receipt["status"] != "passed" or receipt["source_sha256"] != status["source_sha256"]:
        raise ValueError("Independent inference verification is missing or mismatched")
    for name in ("status.json", "per_seed.json"):
        if digest(data / name) != receipt["input_sha256"][name]:
            raise ValueError(f"Validated input hash mismatch: {name}")

    def identity(row: dict) -> tuple:
        return row["group"], row["table"], row["cell"], row["metric"]

    summaries = {identity(row): row for row in inputs["cell_summary.json"]["rows"]}
    if len(summaries) != len(inputs["cell_summary.json"]["rows"]):
        raise ValueError("Duplicate summary identities")
    seed_rows: dict[tuple, list] = {}
    for row in inputs["per_seed.json"]["rows"]:
        seed_rows.setdefault(identity(row), []).append(row)

    specs = [
        dict(
            file="hypothesis_crossplay.pdf", group="counterattack_hyp_n7_r1",
            metric="false_ejection_rate", rows=["mean", "soft", "hypothesis"],
            columns=["mean", "soft", "hypothesis"],
            row_labels=["Mean", "Soft", "Hypothesis"],
            column_labels=["Mean", "Soft", "Hypothesis"],
            episodes_per_seed_cell=1000, annotation="mean_and_sample_sd_3_decimals",
        ),
        dict(
            file="finite_crossplay.pdf", group="multigen_none_n7_r8_k10",
            metric="coalition_game_win_rate", rows=[f"C{i}" for i in range(11)],
            columns=[f"D{i}" for i in range(11)],
            row_labels=[f"C{i}" for i in range(11)],
            column_labels=[f"D{i}" for i in range(11)],
            episodes_per_seed_cell=500, annotation="mean_2_decimals",
        ),
    ]
    figures, flat = {}, []
    for spec in specs:
        cells, means, sds = [], [], []
        for row_name in spec["rows"]:
            mean_row, sd_row = [], []
            for column_name in spec["columns"]:
                key = (spec["group"], "crossplay", f"{row_name}|{column_name}", spec["metric"])
                summary = summaries[key]
                if (not summary["complete"] or summary["n_seeds"] != 10
                        or summary["expected_seeds"] != SEEDS or summary["included_seeds"] != SEEDS):
                    raise ValueError(f"Incomplete ten-seed figure cohort: {key}")
                raw = sorted(seed_rows[key], key=lambda item: item["seed"])
                if [item["seed"] for item in raw] != SEEDS:
                    raise ValueError(f"Missing or duplicate seed in figure cohort: {key}")
                if any(item["job"] != f"{spec['group']}_s{item['seed']}" for item in raw):
                    raise ValueError(f"Unexpected source job: {key}")
                values = np.array([item["value"] for item in raw], dtype=float)
                if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError(f"Invalid per-seed probabilities: {key}")
                recomputed = (float(values.mean()), float(values.std(ddof=1)))
                if not np.allclose(recomputed, (summary["mean"], summary["sd"]), rtol=0, atol=1e-12):
                    raise ValueError(f"Summary does not match unrounded seed values: {key}")
                # Plot the published, unrounded summary values, independently
                # reconstructed above from the verified seed-level input.
                mean_row.append(summary["mean"])
                sd_row.append(summary["sd"])
                cells.append(dict(row=row_name, column=column_name, summary=summary, per_seed=raw))
                flat.append(dict(
                    figure=spec["file"], group=spec["group"], table="crossplay",
                    cell=key[2], metric=spec["metric"], row=row_name, column=column_name,
                    n_seeds=10, mean=summary["mean"], sd=summary["sd"],
                ))
            means.append(mean_row)
            sds.append(sd_row)
        figures[spec["file"]] = dict(
            **spec, seeds=SEEDS, mean_matrix=means, sample_sd_matrix=sds, cells=cells,
        )

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "pdf.fonttype": 42})
    for name, spec in figures.items():
        is_hypothesis = name == "hypothesis_crossplay.pdf"
        if is_hypothesis:
            fig = plt.figure(figsize=(6.5, 4.2))
            ax = fig.add_axes((.23, .19, .63, .67))
            cax = fig.add_axes((.90, .19, .025, .67))
        else:
            fig = plt.figure(figsize=(6.5, 6.1))
            ax = fig.add_axes((.14, .12, .75, .80))
            cax = fig.add_axes((.91, .15, .020, .74))
        matrix = np.array(spec["mean_matrix"])
        sds = np.array(spec["sample_sd_matrix"])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(spec["columns"])), spec["column_labels"])
        ax.set_yticks(range(len(spec["rows"])), spec["row_labels"])
        ax.set_xlabel("Evaluation defense" if is_hypothesis else "Defender policy", labelpad=9)
        ax.set_ylabel("Training defense" if is_hypothesis else "Coalition policy", labelpad=9)
        ax.set_title("False-ejection probability" if is_hypothesis else "Coalition-victory probability", pad=10)
        fig.colorbar(im, cax=cax, ticks=[0, .5, 1])
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            label = f"{value:.3f}\n±{sds[i, j]:.3f}" if is_hypothesis else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=12,
                    color="black" if value > .52 else "white")
        # Keep the exact 6.5-inch page width: inclusion at the article's text
        # width therefore preserves the intended 12-point labels.
        fig.savefig(out / name, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    csv_path = manifest_path.with_suffix(".csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    manifest = dict(
        schema_version=1, kind="Exact plotted values for the two simplified v2 figures",
        original_source_sha256=status["source_sha256"],
        input_sha256={f"final_analysis/{name}": digest(data / name) for name in names},
        plotter_sha256=digest(Path(__file__)),
        libraries=dict(matplotlib=matplotlib.__version__, numpy=np.__version__),
        canvas_width_inches=6.5, label_font_points=12,
        figures=figures, csv_file=csv_path.name, csv_sha256=digest(csv_path),
        note="All 130 cells retain exact unrounded published means, sample SDs, and ten seed records. Means and SDs are independently checked against validated seed-level values. Figure labels alone are rounded; no new inference or simulation is performed. Finite crossplay displays means only to avoid dense two-line annotations; its SDs remain in this manifest and CSV. The original figures are unchanged.",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "final_analysis")
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "figures" / "v2")
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper" / "data" / "paper_v2_figure_sources.json")
    args = parser.parse_args()
    result = generate(args.data, args.out, args.manifest)
    print(f"Generated {len(result['figures'])} figures from 130 complete ten-seed cells; no simulation or inference.")


if __name__ == "__main__":
    main()
