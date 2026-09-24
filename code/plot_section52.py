"""Create the Section 5.2 figures from the packaged result CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["DE-MILP", "SS-MILP", "SS-BW"]
LABELS = ["DE-MILP", "SS-MILP", "SS-BW"]
K_VALUES = [15, 50, 200, 500]
COLORS = ["#4C78A8", "#F58518", "#54A24B"]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def figure4(raw_rows: List[Dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.35), sharey=True)
    for ax, K in zip(axes, K_VALUES):
        direct = []
        indirect = []
        for method in METHODS:
            group = [r for r in raw_rows if int(r["K"]) == K and r["method"] == method]
            direct.append(float(np.mean([float(r["avg_direct_burden"]) for r in group])))
            indirect.append(float(np.mean([float(r["avg_indirect_burden"]) for r in group])))
        x = np.arange(len(METHODS))
        ax.bar(x, direct, color=COLORS, edgecolor="black", linewidth=0.35, label="Direct")
        ax.bar(
            x, indirect, bottom=direct, color="white", edgecolor=COLORS,
            hatch="///", linewidth=1.0, label="Inter-stage",
        )
        ax.set_xticks(x, LABELS, rotation=25, ha="right")
        ax.set_title(f"$K={K}$")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Average burden per attended stage")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#808080", edgecolor="black", label="Direct"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#4C78A8", hatch="///", label="Inter-stage"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure5(patient_rows: List[Dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.35), sharey=True)
    rng = np.random.default_rng(2025)
    for ax, K in zip(axes, K_VALUES):
        data = [
            np.asarray([
                float(r["total_burden"]) for r in patient_rows
                if int(r["K"]) == K and r["method"] == method
            ], dtype=float)
            for method in METHODS
        ]
        box = ax.boxplot(
            data, patch_artist=True, widths=0.58, showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
        )
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
        for index, values in enumerate(data, start=1):
            jitter = (rng.random(values.size) - 0.5) * 0.20
            ax.scatter(
                np.full(values.size, index) + jitter, values,
                color=COLORS[index - 1], s=8, alpha=0.35, linewidths=0,
            )
        ax.set_xticks(np.arange(1, 4), LABELS, rotation=25, ha="right")
        ax.set_title(f"$K={K}$")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Total burden per patient")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(PACKAGE_ROOT / "data" / "section_5_2" / "individual_runs.csv"))
    parser.add_argument("--patients", default=str(PACKAGE_ROOT / "data" / "section_5_2" / "patient_burdens.csv"))
    parser.add_argument("--outdir", action="append")
    args = parser.parse_args()
    raw_rows = read_csv(args.raw)
    patient_rows = read_csv(args.patients)
    output_dirs = args.outdir or [str(PACKAGE_ROOT / "figures" / "section_5_2")]
    for directory in output_dirs:
        outdir = Path(directory)
        outdir.mkdir(parents=True, exist_ok=True)
        figure4(raw_rows, outdir / "figure4.pdf")
        figure5(patient_rows, outdir / "figure5.pdf")
        print(f"wrote {outdir / 'figure4.pdf'} and {outdir / 'figure5.pdf'}")


if __name__ == "__main__":
    main()
