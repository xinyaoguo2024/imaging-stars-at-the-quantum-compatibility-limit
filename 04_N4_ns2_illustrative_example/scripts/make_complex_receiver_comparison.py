#!/usr/bin/env python3
"""Grouped edge-by-edge Fisher comparison at the complex N=4 workpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("N4_DATA_DIR", str(ROOT / "data")))
FIGURES = Path(os.environ.get("N4_FIGURE_DIR", str(ROOT / "figures")))


def gains_from_fisher(path: Path) -> np.ndarray:
    fisher = np.asarray(np.load(path)["fisher"], dtype=float)
    covariance = np.linalg.inv(fisher)
    return 24.0 / np.diag(covariance)


def main() -> None:
    series = [
        gains_from_fisher(DATA / "n4_ns2_phase_pvm_complexwp.npz"),
        gains_from_fisher(DATA / "n4_ns2_phase_povm_q2_complexwp.npz"),
        gains_from_fisher(DATA / "n4_ns3_phase_povm_q2_complexwp.npz"),
    ]
    with (DATA / "n4_complexwp_holevo_summary.json").open() as handle:
        holevo = json.load(handle)
    series.append(
        np.asarray(
            holevo["unit_weight_holevo_optimum"]["edge_directional_fisher_gains"],
            dtype=float,
        )
    )

    labels = [
        r"$n_s=2$ PVM",
        r"$n_s=2$ POVM",
        r"$n_s=3$ POVM",
        "asymptotic Holevo",
    ]
    colors = ["#4477AA", "#66A6D9", "#228B73", "#CC6677"]
    hatches = ["", "///", "xx", ".."]
    edge_labels = ["12", "13", "14", "23", "24", "34", "Avg."]
    values = [np.r_[g, g.mean()] for g in series]

    x = np.arange(len(edge_labels), dtype=float)
    width = 0.19
    offsets = (np.arange(len(values)) - (len(values) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(3.42, 2.72))
    for offset, data, label, color, hatch in zip(
        offsets, values, labels, colors, hatches, strict=True
    ):
        ax.bar(
            x + offset,
            data,
            width=width,
            label=label,
            color=color,
            edgecolor="white" if not hatch else "#27323A",
            linewidth=0.45,
            hatch=hatch,
            zorder=3,
        )

    ax.axhline(1.0, color="0.42", lw=0.8, ls="--", zorder=2)
    ax.axvline(5.52, color="0.80", lw=0.65, zorder=1)
    ax.set_xticks(x, edge_labels)
    ax.set_xlabel("edge phase", labelpad=2)
    ax.set_ylabel(r"nuisance-aware gain $F_e/F_e^{\rm rep}$")
    ax.set_ylim(0, max(3.45, 1.08 * max(v.max() for v in values)))
    ax.set_xlim(-0.58, 6.58)
    ax.grid(axis="y", color="0.88", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.2)
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=6.25,
        loc="upper left",
        handlelength=1.35,
        handletextpad=0.4,
        columnspacing=0.75,
        borderaxespad=0.25,
    )
    fig.tight_layout(pad=0.35)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "n4_finite_receiver_edge_gains.pdf")
    fig.savefig(FIGURES / "n4_finite_receiver_edge_gains.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
