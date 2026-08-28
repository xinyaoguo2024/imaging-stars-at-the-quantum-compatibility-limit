from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUNDLE = Path(__file__).resolve().parents[2]
OUTDIR = BUNDLE / "figures" / "main"
ROOT_FIGDIR = BUNDLE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
ROOT_FIGDIR.mkdir(parents=True, exist_ok=True)


def fisher_gain_equal_s(r23: np.ndarray, r31: np.ndarray, x: float) -> np.ndarray:
    """Direct/edge-first Fisher gain for s1=s2=s3=s."""
    numerator = 2.0 * (r23**2 * r31**2 + r23**2 + r31**2)
    denominator = r23**2 * r31**2 + r23**2 + r31**2 - x * r23 * r31 * (1.0 + r23**2 + r31**2)
    return numerator / denominator


def main() -> None:
    nu12 = 0.5
    epsilon = 1.0e-9
    eta_u_over_epsilon = 50.0
    x = nu12 * eta_u_over_epsilon / (eta_u_over_epsilon + 1.0)

    ratios = np.logspace(-2.0, 0.0, 241)
    r23, r31 = np.meshgrid(ratios, ratios, indexing="xy")
    snr_gain = np.sqrt(fisher_gain_equal_s(r23, r31, x))

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
        }
    )
    fig, ax = plt.subplots(figsize=(3.38, 2.82), constrained_layout=True)
    image = ax.pcolormesh(
        r23,
        r31,
        snr_gain,
        shading="auto",
        cmap="magma",
        edgecolors="none",
        linewidth=0.0,
        rasterized=True,
        vmin=float(np.nanmin(snr_gain)),
        vmax=float(np.nanmax(snr_gain)),
    )
    levels = np.linspace(float(np.nanmin(snr_gain)), float(np.nanmax(snr_gain)), 5)[1:-1]
    contours = ax.contour(r23, r31, snr_gain, levels=levels, colors="white", linewidths=0.55, alpha=0.75)
    ax.clabel(contours, inline=True, fontsize=6.8, fmt="%.2f")
    ax.plot([0.01, 1.0], [0.01, 1.0], color="white", lw=0.8, ls="--", alpha=0.55)
    ax.scatter([1.0], [1.0], color="white", s=16, edgecolor="0.2", linewidth=0.45, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0.01, 1.0)
    ax.set_xlabel(r"$g_{23}/g_{12}$")
    ax.set_ylabel(r"$g_{31}/g_{12}$")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("SNR gain")

    png = OUTDIR / "prl_closure_gain_g_ratio_single.png"
    pdf = OUTDIR / "prl_closure_gain_g_ratio_single.pdf"
    root_png = ROOT_FIGDIR / "prl_closure_gain_g_ratio_single.png"
    root_pdf = ROOT_FIGDIR / "prl_closure_gain_g_ratio_single.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(root_png, dpi=300, bbox_inches="tight")
    fig.savefig(root_pdf, bbox_inches="tight")

    stats = {
        "model": "closed_form_equal_s_direct_over_equal_split_edge_fisher_gain",
        "epsilon": epsilon,
        "nu12": nu12,
        "eta_u_over_epsilon": eta_u_over_epsilon,
        "x_g12_over_s": x,
        "snr_gain_min": float(np.nanmin(snr_gain)),
        "snr_gain_max": float(np.nanmax(snr_gain)),
        "symmetric_snr_gain": float(np.sqrt(2.0 / (1.0 - x))),
        "equal_weak_edge_fisher_limit": float(4.0 / (2.0 - x)),
        "single_weak_edge_fisher_limit": 2.0,
    }
    stats_path = OUTDIR / "prl_closure_gain_g_ratio_single_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    (ROOT_FIGDIR / "prl_closure_gain_g_ratio_single_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(pdf)
    print(png)
    print(stats_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
