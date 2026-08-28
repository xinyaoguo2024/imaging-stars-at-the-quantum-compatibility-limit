from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SOURCE = (
    WORKSPACE
    / "27_strict_physical_near_paircombine_20260613"
    / "results"
    / "paircombine_strict_near_loop_gains_worst_ratio.csv"
)
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open()))
    labels = [row["loop"] for row in rows]
    direct = np.asarray([float(row["gain_direct_optimized_vs_edge"]) for row in rows])
    near = np.asarray([float(row["gain_paircombine_strict_near_vs_edge"]) for row in rows])

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.7, 3.6), constrained_layout=True)
    ax.bar(x - 0.5 * width, direct, width, color="#9e2b22", label="direct closure")
    ax.bar(x + 0.5 * width, near, width, color="#245a9e", label="near optimal")
    ax.axhline(math.sqrt(1.5), color="0.30", lw=1.1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(1.0, 1.42)
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))

    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"sm_loop_gain_benchmark.{suffix}", dpi=280)
    plt.close(fig)


if __name__ == "__main__":
    main()
