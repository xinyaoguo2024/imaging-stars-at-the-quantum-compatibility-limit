from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NU12 = 0.3
ALPHA_MIN = 0.01
ALPHA_MAX = 1.0
BETA_MIN = 0.01
BETA_MAX = 1.0
N_GRID = 300


def snr_gain(alpha: np.ndarray, beta: np.ndarray, nu12: float = NU12) -> np.ndarray:
    nu23 = nu12 * alpha
    nu31 = nu12 * beta
    denom = (
        nu12**2 * nu23**2
        + nu23**2 * nu31**2
        + nu31**2 * nu12**2
        - nu12 * nu23 * nu31 * (nu12**2 + nu23**2 + nu31**2)
    )
    num = (
        nu12**2
        * nu23**2
        * nu31**2
        * (1.0 / nu12 + 1.0 / nu23 + 1.0 / nu31) ** 2
    )
    gain_q = num / denom
    return np.sqrt(gain_q)


def main() -> None:
    alpha = np.linspace(ALPHA_MIN, ALPHA_MAX, N_GRID)
    beta = np.linspace(BETA_MIN, BETA_MAX, N_GRID)
    aa, bb = np.meshgrid(alpha, beta)
    gain = snr_gain(aa, bb)

    min_idx = np.unravel_index(np.nanargmin(gain), gain.shape)
    max_idx = np.unravel_index(np.nanargmax(gain), gain.shape)
    min_gain = gain[min_idx]
    max_gain = gain[max_idx]
    min_point = (aa[min_idx], bb[min_idx])
    max_point = (aa[max_idx], bb[max_idx])

    outdir = Path("output/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
    im = ax.imshow(
        gain,
        origin="lower",
        extent=[ALPHA_MIN, ALPHA_MAX, BETA_MIN, BETA_MAX],
        aspect="auto",
        cmap="viridis",
    )
    contours = ax.contour(
        aa,
        bb,
        gain,
        levels=[1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0],
        colors="white",
        linewidths=0.8,
        alpha=0.8,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

    ax.scatter(*min_point, color="white", s=24, marker="o")
    ax.scatter(*max_point, color="white", s=24, marker="s")

    ax.set_title(
        r"Closure-phase SNR gain: $\mathrm{SNR}_Q / \mathrm{SNR}_{\mathrm{sep}}$"
        "\n"
        r"$\nu_{12}=0.3,\ \nu_{23}=0.3\alpha,\ \nu_{31}=0.3\beta$"
    )
    ax.set_xlabel(r"$\alpha = \nu_{23}/\nu_{12}$")
    ax.set_ylabel(r"$\beta = \nu_{31}/\nu_{12}$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\mathrm{SNR}_Q / \mathrm{SNR}_{\mathrm{sep}}$")

    note = (
        f"min = {min_gain:.3f} at "
        + rf"$(\alpha,\beta)=({min_point[0]:.2f},{min_point[1]:.2f})$"
        + "\n"
        + f"max = {max_gain:.3f} at "
        + rf"$(\alpha,\beta)=({max_point[0]:.2f},{max_point[1]:.2f})$"
    )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
    )

    png_path = outdir / "closure_snr_gain_alpha_beta.png"
    pdf_path = outdir / "closure_snr_gain_alpha_beta.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(
        "Gain range:",
        f"{min_gain:.6f} to {max_gain:.6f}",
        "for alpha,beta in",
        f"[{ALPHA_MIN}, {ALPHA_MAX}]",
    )


if __name__ == "__main__":
    main()
