#!/usr/bin/env python3
"""Draw the compiled N=4, n_s=2 collective receiver as a measurement flow.

The figure deliberately emphasizes the physical organization of the PVM:
two copies are coherently sorted into exchange-symmetry sectors, each sector
is rotated in its optimized basis, and the output ports are detected.  The
actual port probabilities are retained only as a compact visual encoding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


BLUE = "#2474A6"
BLUE_LIGHT = "#D9ECF7"
ORANGE = "#D97918"
ORANGE_LIGHT = "#F9E5CC"
INK = "#1B2630"
MUTED = "#5D6B76"


def box(ax, xy, width, height, text, *, fc="white", ec=INK, fontsize=8.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )
    return patch


def arrow(ax, start, end, *, color=MUTED, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def detector_row(ax, x0, y, probabilities, color, label):
    probabilities = np.asarray(probabilities, dtype=float)
    spacing = 0.027 if len(probabilities) == 10 else 0.045
    radius_scale = 0.022 / np.sqrt(max(probabilities.max(), 1e-15))
    xs = x0 + spacing * np.arange(len(probabilities))
    for index, (x, probability) in enumerate(zip(xs, probabilities, strict=True), 1):
        radius = max(0.010, radius_scale * np.sqrt(max(probability, 0.0)))
        ax.plot([x, x], [y - 0.050, y - 0.014], color=MUTED, lw=0.75)
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.95,
            )
        )
        ax.text(x, y - 0.070, str(index), ha="center", va="top", fontsize=5.3, color=MUTED)
    if label:
        ax.text(
            x0 - 0.018,
            y,
            label,
            ha="right",
            va="center",
            fontsize=6.6,
            color=INK,
        )


def make_figure(probabilities: np.ndarray, output: Path) -> None:
    if probabilities.shape != (16,):
        raise ValueError(f"expected 16 PVM probabilities, received {probabilities.shape}")
    p_plus = probabilities[:10]
    p_minus = probabilities[10:]

    fig, ax = plt.subplots(figsize=(7.05, 2.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Stage I: two indistinguishable copies.
    box(ax, (0.025, 0.60), 0.105, 0.16, "copy 1\n" + r"$\rho_0$", fc="#F1F4F6")
    box(ax, (0.025, 0.25), 0.105, 0.16, "copy 2\n" + r"$\rho_0$", fc="#F1F4F6")
    ax.text(0.078, 0.88, "encoded inputs", ha="center", fontsize=8.0, color=MUTED)
    arrow(ax, (0.13, 0.68), (0.185, 0.56))
    arrow(ax, (0.13, 0.33), (0.185, 0.48))

    # Stage II: coherent symmetry sorting.
    box(
        ax,
        (0.185, 0.405),
        0.14,
        0.20,
        "coherent copy sorter\n" + r"$U_{\rm Schur}$",
        fc="#EEF0FA",
        ec="#5966A4",
    )
    ax.text(0.255, 0.88, "joint processing", ha="center", fontsize=8.0, color=MUTED)
    arrow(ax, (0.325, 0.52), (0.37, 0.70), color=BLUE)
    arrow(ax, (0.325, 0.49), (0.37, 0.30), color=ORANGE)

    # Stage III: sector-conditioned basis rotations.
    ax.text(0.455, 0.88, "sector rotations", ha="center", fontsize=8.0, color=MUTED)
    box(
        ax,
        (0.37, 0.60),
        0.13,
        0.18,
        "symmetric sector\n" + r"$\mathcal{H}_+\;(d=10)$",
        fc=BLUE_LIGHT,
        ec=BLUE,
        fontsize=7.4,
    )
    box(
        ax,
        (0.37, 0.21),
        0.13,
        0.18,
        "antisymmetric sector\n" + r"$\mathcal{H}_-\;(d=6)$",
        fc=ORANGE_LIGHT,
        ec=ORANGE,
        fontsize=7.4,
    )
    box(ax, (0.535, 0.625), 0.095, 0.13, "optimized\n" + r"$U_+^\dagger$", fc="white", ec=BLUE)
    box(ax, (0.535, 0.235), 0.095, 0.13, "optimized\n" + r"$U_-^\dagger$", fc="white", ec=ORANGE)
    arrow(ax, (0.50, 0.69), (0.535, 0.69), color=BLUE)
    arrow(ax, (0.50, 0.30), (0.535, 0.30), color=ORANGE)
    arrow(ax, (0.63, 0.69), (0.682, 0.69), color=BLUE)
    arrow(ax, (0.63, 0.30), (0.682, 0.30), color=ORANGE)

    # Stage IV: detector ports; circle area is proportional to probability.
    detector_row(ax, 0.700, 0.69, p_plus, BLUE, "")
    detector_row(ax, 0.700, 0.30, p_minus, ORANGE, "")
    ax.text(0.825, 0.88, "resolved output ports", ha="center", fontsize=8.0, color=MUTED)
    ax.text(
        0.825,
        0.055,
        r"circle area $\propto p_y$;  "
        + rf"$P_+={p_plus.sum():.3f}$, $P_-={p_minus.sum():.3f}$",
        ha="center",
        va="center",
        fontsize=7.0,
        color=MUTED,
    )

    # Minimal stage labels make the reading order explicit without a title.
    for x, number in [(0.012, "1"), (0.172, "2"), (0.357, "3"), (0.665, "4")]:
        ax.add_patch(Circle((x, 0.91), 0.014, facecolor=INK, edgecolor="none"))
        ax.text(x, 0.91, number, ha="center", va="center", fontsize=6.4, color="white")

    fig.tight_layout(pad=0.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "n4_ns2_phase_pvm.npz",
        help="PVM NPZ containing a length-16 `probabilities` array",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "n4_ns2_collective_receiver",
    )
    args = parser.parse_args()
    saved = np.load(args.input)
    make_figure(np.asarray(saved["probabilities"], dtype=float), args.output)


if __name__ == "__main__":
    main()
