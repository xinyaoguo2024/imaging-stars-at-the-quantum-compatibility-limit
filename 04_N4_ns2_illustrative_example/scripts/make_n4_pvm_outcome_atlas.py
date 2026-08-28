#!/usr/bin/env python3
"""Create a compact 16-outcome atlas for the optimized N=4, n_s=2 PVM."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2878A8"
ORANGE = "#D97712"
INK = "#18242D"


# A convenient bijection between the 10+6 Schur-sector ports and the two
# four-valued terminal digits.  It makes the sector structure visible:
# a <= b labels H_+, while a > b labels H_-.
SYMMETRIC_CELLS = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
]
ANTISYMMETRIC_CELLS = [
    (1, 0),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (3, 2),
]


def to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (16,):
        raise ValueError(f"expected 16 outcomes, received {values.shape}")
    matrix = np.empty((4, 4), dtype=float)
    for value, (row, column) in zip(values[:10], SYMMETRIC_CELLS, strict=True):
        matrix[row, column] = value
    for value, (row, column) in zip(values[10:], ANTISYMMETRIC_CELLS, strict=True):
        matrix[row, column] = value
    return matrix


def text_color(value: float, norm, cmap) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.52 else INK


def draw_panel(ax, matrix, title, cmap_name, value_format, cbar_label, indices=None):
    cmap = plt.get_cmap(cmap_name)
    vmin = 0.0
    vmax = float(matrix.max()) * 1.04
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for row in range(4):
        for column in range(4):
            sector_color = BLUE if row <= column else ORANGE
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=sector_color,
                    linewidth=1.35,
                )
            )
            ax.text(
                column,
                row,
                value_format.format(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="medium",
                color=text_color(matrix[row, column], norm, cmap),
            )
            if indices is not None:
                ax.text(
                    column - 0.43,
                    row - 0.39,
                    rf"$y={int(indices[row, column])}$",
                    ha="left",
                    va="top",
                    fontsize=4.5,
                    color=text_color(matrix[row, column], norm, cmap),
                    alpha=0.88,
                )

    ax.set_title(title, fontsize=8.4, pad=5)
    ax.set_xticks(range(4), [1, 2, 3, 4])
    ax.set_yticks(range(4), [1, 2, 3, 4])
    ax.set_xlabel(r"output label $b$", labelpad=2)
    ax.set_ylabel(r"output label $a$", labelpad=2)
    ax.tick_params(length=0, labelsize=7.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.ax.tick_params(labelsize=5.9, length=2)
    colorbar.set_label(cbar_label, fontsize=6.4, labelpad=2)


def make_figure(npz_path: Path, output: Path) -> None:
    saved = np.load(npz_path)
    probabilities = np.asarray(saved["probabilities"], dtype=float)
    derivatives = np.asarray(saved["derivatives"], dtype=float)
    scores = derivatives / probabilities[:, None]
    score_norm = np.linalg.norm(scores, axis=1)
    fisher_trace = probabilities * score_norm**2
    p_plus, p_minus = probabilities[:10].sum(), probabilities[10:].sum()
    j_plus, j_minus = fisher_trace[:10].sum(), fisher_trace[10:].sum()

    matrices = [
        to_matrix(probabilities),
        to_matrix(score_norm),
        to_matrix(fisher_trace),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55))
    draw_panel(
        axes[0],
        matrices[0],
        r"(a) joint probability $p_{ab}$",
        "Blues",
        "{:.3f}",
        r"$p_{ab}$",
        indices=to_matrix(np.arange(1, 17)),
    )
    draw_panel(
        axes[1],
        matrices[1],
        r"(b) likelihood-score strength $\|\mathbf{s}_{ab}\|_2$",
        "Purples",
        "{:.2f}",
        r"$\|\mathbf{s}_{ab}\|_2$",
    )
    draw_panel(
        axes[2],
        matrices[2],
        r"(c) Fisher contribution $\operatorname{Tr}J_{ab}$",
        "Greens",
        "{:.3f}",
        r"$\operatorname{Tr}J_{ab}$",
    )

    fig.suptitle(
        r"$\rho_0^{\otimes 2}$   --- $U_{\rm coll}$ $\longrightarrow$   $y=(a,b)$",
        x=0.50,
        y=1.015,
        fontsize=8.8,
        color=INK,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor=BLUE,
                linewidth=1.35,
                label=(
                    r"$\mathcal{H}_+$: 10 ports, "
                    + rf"$P_+={p_plus:.5f}$, $\mathrm{{Tr}}J_+={j_plus:.5f}$"
                ),
            ),
            Patch(
                facecolor="white",
                edgecolor=ORANGE,
                linewidth=1.35,
                label=(
                    r"$\mathcal{H}_-$: 6 ports, "
                    + rf"$P_-={p_minus:.5f}$, $\mathrm{{Tr}}J_-={j_minus:.5f}$"
                ),
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        fontsize=6.15,
        handlelength=1.2,
        handletextpad=0.45,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.13, top=0.79, wspace=0.42)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "n4_ns2_phase_pvm_complexwp.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "n4_ns2_pvm_outcome_atlas",
    )
    args = parser.parse_args()
    make_figure(args.input, args.output)


if __name__ == "__main__":
    main()
