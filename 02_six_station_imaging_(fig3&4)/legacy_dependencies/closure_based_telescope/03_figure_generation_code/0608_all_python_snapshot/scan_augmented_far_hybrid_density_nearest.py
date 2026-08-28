from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from plot_monochromatic_uniform_stack import (
    aggregate_cells,
    nearest_label_map,
    normalize_stack,
    support_mask_from_occupied,
)


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

TARGET_KEY = "direct"
SNR_BOOST = 1.0
COARSE_GRIDS = [40, 50]
DENSITY_POWERS = [0.0, 0.25, 0.5, 0.75, 1.0]
FILL_ALPHAS = [1.0, 0.5, 0.25]
DENSITY_SMOOTH_CELLS = 0.75
WEIGHT_CLIP = (0.25, 4.0)


def coarse_density_on_fine_grid(
    u: np.ndarray,
    v: np.ndarray,
    *,
    n: int,
    fov_rad: float,
    n_bin: int,
    power: float,
) -> np.ndarray:
    """Estimate uv-density compensation on the final FFT grid.

    The actual visibility samples keep their true coordinates.  The coarse grid
    is used only to estimate how crowded each region of Fourier space is.  A
    fine-grid Fourier cell then receives a multiplicative area weight
    proportional to rho^{-power}.
    """
    if power == 0.0:
        return np.ones((n, n), dtype=float)

    u_full = np.concatenate([u, -u])
    v_full = np.concatenate([v, -v])
    max_u = max(float(np.max(np.abs(u_full))), 1.0)
    max_v = max(float(np.max(np.abs(v_full))), 1.0)
    u_edges = np.linspace(-1.0001 * max_u, 1.0001 * max_u, n_bin + 1)
    v_edges = np.linspace(-1.0001 * max_v, 1.0001 * max_v, n_bin + 1)
    iu = np.searchsorted(u_edges, u_full, side="right") - 1
    iv = np.searchsorted(v_edges, v_full, side="right") - 1
    valid = (iu >= 0) & (iu < n_bin) & (iv >= 0) & (iv < n_bin)
    density = np.zeros((n_bin, n_bin), dtype=float)
    np.add.at(density, (iv[valid], iu[valid]), 1.0)
    density = base.gaussian_filter(density, DENSITY_SMOOTH_CELLS)
    positive = density[density > 0.0]
    if len(positive) == 0:
        return np.ones((n, n), dtype=float)
    floor = 0.15 * float(np.median(positive))
    density = np.maximum(density, floor)

    du = 1.0 / fov_rad
    mid = n // 2
    coords = (np.arange(n) - mid) * du
    uu, vv = np.meshgrid(coords, coords)
    fine_iu = np.searchsorted(u_edges, uu, side="right") - 1
    fine_iv = np.searchsorted(v_edges, vv, side="right") - 1
    inside = (fine_iu >= 0) & (fine_iu < n_bin) & (fine_iv >= 0) & (fine_iv < n_bin)
    fine_density = np.ones((n, n), dtype=float)
    fine_density[inside] = density[fine_iv[inside], fine_iu[inside]]
    reference = float(np.median(fine_density[inside])) if np.any(inside) else 1.0
    area_weight = (fine_density / max(reference, 1e-30)) ** (-power)
    return np.clip(area_weight, WEIGHT_CLIP[0], WEIGHT_CLIP[1])


def reconstruct_band_hybrid(
    band: dict[str, np.ndarray],
    key: str,
    fov_rad: float,
    *,
    n_bin: int,
    power: float,
    fill_alpha: float,
) -> tuple[np.ndarray, float]:
    grid, occupied, cell_var = aggregate_cells(
        band["u"],
        band["v"],
        band[f"vis_{key}"],
        band[f"sigma_{key}"],
        n=wt.N_PIX,
        fov_rad=fov_rad,
        average_mode="noise",
    )
    support = support_mask_from_occupied(occupied, du=1.0 / fov_rad, mode=wt.aug.SUPPORT_MODE)
    label_y, label_x, fillable = nearest_label_map(occupied, support)
    filled_grid = np.zeros_like(grid)
    filled_grid[fillable] = grid[label_y[fillable], label_x[fillable]]
    filled_grid[occupied] = grid[occupied]

    area_weight = coarse_density_on_fine_grid(
        band["u"],
        band["v"],
        n=wt.N_PIX,
        fov_rad=fov_rad,
        n_bin=n_bin,
        power=power,
    )
    support_weight = np.zeros_like(area_weight)
    support_weight[fillable] = fill_alpha * area_weight[fillable]
    support_weight[occupied] = area_weight[occupied]
    support_weight[wt.N_PIX // 2, wt.N_PIX // 2] = max(support_weight[wt.N_PIX // 2, wt.N_PIX // 2], 1.0)

    image_grid = filled_grid * support_weight
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image_grid))).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(support_weight))).real
    peak = psf[wt.N_PIX // 2, wt.N_PIX // 2]
    if peak > 0.0:
        image /= peak

    assigned_var = cell_var[label_y[fillable], label_x[fillable]]
    assigned_weight = support_weight[fillable]
    finite = np.isfinite(assigned_var) & (assigned_var > 0.0) & (assigned_weight > 0.0)
    if np.any(finite):
        mean_var = np.average(assigned_var[finite], weights=assigned_weight[finite])
        band_weight = 1.0 / float(mean_var)
    else:
        band_weight = 1.0
    return image, band_weight


def reconstruct_stack(
    bands: list[dict[str, np.ndarray]],
    truth: np.ndarray,
    *,
    n_bin: int,
    power: float,
    fill_alpha: float,
) -> np.ndarray:
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    stack = np.zeros_like(truth)
    total_weight = 0.0
    for band in bands:
        image, weight = reconstruct_band_hybrid(
            band,
            TARGET_KEY,
            fov_rad,
            n_bin=n_bin,
            power=power,
            fill_alpha=fill_alpha,
        )
        stack += weight * image
        total_weight += weight
    return normalize_stack(stack / max(total_weight, 1e-30))


def image_metrics(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> dict[str, float]:
    ring_mask, core_mask = opt.blr_masks(axis_uas)
    return {
        "global_corr": float(base.corrcoef_positive(truth, image)),
        "blr_corr": float(opt.masked_corr(truth, image, ring_mask)),
        "ring_contrast": float(opt.ring_contrast(image, ring_mask, core_mask)),
    }


def run_case(case: wt.aug.NetworkCase) -> dict:
    bands, _, truth, axis_uas = wt.simulate_bands(case)
    images: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    labels: dict[str, str] = {}

    for n_bin in COARSE_GRIDS:
        for fill_alpha in FILL_ALPHAS:
            for power in DENSITY_POWERS:
                key = f"bin{n_bin}_p{power:.2f}_fill{fill_alpha:.2f}"
                images[key] = reconstruct_stack(
                    bands,
                    truth,
                    n_bin=n_bin,
                    power=power,
                    fill_alpha=fill_alpha,
                )
                metrics[key] = image_metrics(truth, images[key], axis_uas)
                labels[key] = f"{n_bin}x{n_bin}, p={power:.2f}, f={fill_alpha:.2f}"

    # p=0, fill=1 is exactly the nearest-filled baseline up to numerical
    # roundoff.  Keep an explicit alias to make the comparison readable.
    nearest_key = "bin40_p0.00_fill1.00"
    best_key = max(
        (key for key in metrics if not key.endswith("p0.00_fill1.00")),
        key=lambda k: (metrics[k]["blr_corr"], metrics[k]["global_corr"]),
    )
    return {
        "case": case.key,
        "label": case.key.replace("_", " "),
        "truth": truth,
        "axis_uas": axis_uas,
        "images": images,
        "metrics": metrics,
        "labels": labels,
        "nearest_key": nearest_key,
        "best_key": best_key,
    }


def plot_best_montage(results: dict[str, dict]) -> tuple[Path, Path]:
    case_keys = list(results)
    columns = ["input", "nearest", "best"]
    axis_uas = results[case_keys[0]]["axis_uas"]
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
        }
    )
    fig, axes = plt.subplots(len(case_keys), len(columns), figsize=(8.2, 7.0), constrained_layout=True)
    for row, case_key in enumerate(case_keys):
        payload = results[case_key]
        for col, column in enumerate(columns):
            ax = axes[row, col]
            if column == "input":
                image = payload["truth"]
                title = payload["label"]
            elif column == "nearest":
                key = payload["nearest_key"]
                image = payload["images"][key]
                m = payload["metrics"][key]
                title = f"nearest only\nBLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}"
            else:
                key = payload["best_key"]
                image = payload["images"][key]
                m = payload["metrics"][key]
                title = f"best hybrid: {payload['labels'][key]}\nBLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}"
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            ax.set_title(title)
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 0:
                ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            else:
                ax.set_yticklabels([])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=axes,
        fraction=0.025,
        pad=0.012,
    )
    cbar.set_label("norm. brightness\n(BLR arcsinh)", fontsize=7.0)
    fig.suptitle("Hybrid coarse-density / nearest-fill reconstruction, real SNR", fontsize=10.5, weight="bold")
    png = OUTFIG / "augmented_existing_telescope_far_snr1_hybrid_density_nearest_best.png"
    pdf = OUTFIG / "augmented_existing_telescope_far_snr1_hybrid_density_nearest_best.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_metric_scan(results: dict[str, dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(FILL_ALPHAS), len(COARSE_GRIDS), figsize=(9.6, 8.0), constrained_layout=True)
    for i, fill_alpha in enumerate(FILL_ALPHAS):
        for j, n_bin in enumerate(COARSE_GRIDS):
            ax = axes[i, j]
            for case_key, payload in results.items():
                y = [
                    payload["metrics"][f"bin{n_bin}_p{power:.2f}_fill{fill_alpha:.2f}"]["blr_corr"]
                    for power in DENSITY_POWERS
                ]
                ax.plot(DENSITY_POWERS, y, marker="o", lw=1.5, label=payload["label"])
                nearest = payload["metrics"][payload["nearest_key"]]["blr_corr"]
                ax.axhline(nearest, color=ax.lines[-1].get_color(), ls="--", lw=0.85, alpha=0.5)
            ax.set_title(f"{n_bin}x{n_bin}, fill={fill_alpha:.2f}")
            ax.set_xlabel(r"density power $p$")
            ax.set_ylabel("BLR correlation")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Hybrid scan: dashed lines are nearest-only baselines", fontsize=10.5, weight="bold")
    png = OUTFIG / "augmented_existing_telescope_far_snr1_hybrid_density_nearest_scan.png"
    pdf = OUTFIG / "augmented_existing_telescope_far_snr1_hybrid_density_nearest_scan.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    wt.SNR_BOOST = SNR_BOOST
    stats_paths = [
        OUTFIG / "augmented_existing_telescope_maunakea_plus5_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus3_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus4_far_stats.json",
    ]
    results: dict[str, dict] = {}
    for path in stats_paths:
        case = wt.case_from_stats(path)
        print(f"simulate and scan {case.key}")
        results[case.key] = run_case(case)

    best_pdf, best_png = plot_best_montage(results)
    scan_pdf, scan_png = plot_metric_scan(results)
    serializable = {}
    for case_key, payload in results.items():
        serializable[case_key] = {
            "label": payload["label"],
            "nearest_key": payload["nearest_key"],
            "best_key": payload["best_key"],
            "best_label": payload["labels"][payload["best_key"]],
            "metrics": payload["metrics"],
        }
    out_json = OUTFIG / "augmented_existing_telescope_far_snr1_hybrid_density_nearest_scan.json"
    out_json.write_text(json.dumps(serializable, indent=2) + "\n")
    print(best_pdf)
    print(best_png)
    print(scan_pdf)
    print(scan_png)
    print(out_json)
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
