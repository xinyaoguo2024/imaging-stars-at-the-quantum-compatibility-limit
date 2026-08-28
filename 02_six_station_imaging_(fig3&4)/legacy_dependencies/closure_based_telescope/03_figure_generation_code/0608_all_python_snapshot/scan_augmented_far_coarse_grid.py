from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from plot_monochromatic_uniform_stack import normalize_stack
from plot_uv_weighting_diagnostic import aggregate_to_coarse_uv_grid


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

GRID_SIZES = [20, 30, 40, 50, 60, 80, 100]
SNR_BOOST = 1.0
TARGET_KEY = "direct"


def reconstruct_band_coarse(
    band: dict[str, np.ndarray],
    key: str,
    fov_rad: float,
    grid_size: int,
) -> tuple[np.ndarray, float]:
    u_cell, v_cell, vis_cell, weights = aggregate_to_coarse_uv_grid(
        band["u"],
        band["v"],
        band[f"vis_{key}"],
        band[f"sigma_{key}"],
        n_bin_u=grid_size,
        n_bin_v=grid_size,
        mode="coarse_ivar_briggs",
        smooth_cells=wt.COARSE_SMOOTH_CELLS,
    )
    image = wt.wiener_from_uv(u_cell, v_cell, vis_cell, weights, fov_rad)
    return image, float(np.median(weights[weights > 0.0])) if np.any(weights > 0.0) else 1.0


def reconstruct_stack(
    bands: list[dict[str, np.ndarray]],
    truth: np.ndarray,
    *,
    grid_size: int | None,
) -> np.ndarray:
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    stack = np.zeros_like(truth)
    total_weight = 0.0
    for band in bands:
        if grid_size is None:
            image, weight = wt.reconstruct_band_nearest(band, TARGET_KEY, fov_rad)
        else:
            image, weight = reconstruct_band_coarse(band, TARGET_KEY, fov_rad, grid_size)
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


def plot_montage(results: dict, truth_ref: np.ndarray, axis_uas: np.ndarray) -> tuple[Path, Path]:
    case_keys = list(results)
    columns = ["input", "nearest"] + [str(size) for size in GRID_SIZES]
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
        }
    )
    fig, axes = plt.subplots(len(case_keys), len(columns), figsize=(17.4, 5.9), constrained_layout=True)
    for row, case_key in enumerate(case_keys):
        payload = results[case_key]
        for col, column in enumerate(columns):
            ax = axes[row, col]
            if column == "input":
                image = truth_ref
                title = payload["short_label"]
            elif column == "nearest":
                image = payload["images"]["nearest"]
                m = payload["metrics"]["nearest"]
                title = f"nearest\nBLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}"
            else:
                image = payload["images"][column]
                m = payload["metrics"][column]
                title = f"{column}x{column}\nBLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}"
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            ax.set_title(title)
            ax.set_xlabel(r"$\Delta\alpha$")
            if col == 0:
                ax.set_ylabel(r"$\Delta\delta$")
            else:
                ax.set_yticklabels([])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=axes,
        fraction=0.012,
        pad=0.010,
    )
    cbar.set_label("norm. brightness\n(BLR arcsinh)", fontsize=7.0)
    fig.suptitle("Direct-closure coarse-grid interpolation scan, SNR x1", fontsize=11, weight="bold")
    png = OUTFIG / "augmented_existing_telescope_far_snr1_coarse_grid_scan.png"
    pdf = OUTFIG / "augmented_existing_telescope_far_snr1_coarse_grid_scan.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_metric_curves(results: dict) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0), constrained_layout=True)
    metric_labels = [
        ("global_corr", "global correlation"),
        ("blr_corr", "BLR correlation"),
        ("ring_contrast", "ring contrast"),
    ]
    x = np.array(GRID_SIZES, dtype=float)
    for ax, (metric, label) in zip(axes, metric_labels):
        for case_key, payload in results.items():
            y = np.array([payload["metrics"][str(size)][metric] for size in GRID_SIZES])
            ax.plot(x, y, marker="o", lw=1.5, label=payload["short_label"])
            ax.axhline(payload["metrics"]["nearest"][metric], color=ax.lines[-1].get_color(), ls="--", lw=0.9, alpha=0.55)
        ax.set_xlabel("coarse grid size")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Coarse-grid interpolation metrics; dashed lines are nearest-fill baselines", fontsize=10.5)
    png = OUTFIG / "augmented_existing_telescope_far_snr1_coarse_grid_scan_metrics.png"
    pdf = OUTFIG / "augmented_existing_telescope_far_snr1_coarse_grid_scan_metrics.pdf"
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
    results = {}
    truth_ref = None
    axis_ref = None
    for path in stats_paths:
        case = wt.case_from_stats(path)
        print(f"simulate {case.key}")
        bands, _, truth, axis_uas = wt.simulate_bands(case)
        images: dict[str, np.ndarray] = {}
        metrics: dict[str, dict[str, float]] = {}
        images["nearest"] = reconstruct_stack(bands, truth, grid_size=None)
        metrics["nearest"] = image_metrics(truth, images["nearest"], axis_uas)
        for size in GRID_SIZES:
            print(f"  grid {size}x{size}")
            key = str(size)
            images[key] = reconstruct_stack(bands, truth, grid_size=size)
            metrics[key] = image_metrics(truth, images[key], axis_uas)
        results[case.key] = {
            "short_label": case.key.replace("_", " "),
            "metrics": metrics,
            "images": images,
        }
        truth_ref = truth
        axis_ref = axis_uas
    assert truth_ref is not None and axis_ref is not None
    montage_pdf, montage_png = plot_montage(results, truth_ref, axis_ref)
    curves_pdf, curves_png = plot_metric_curves(results)
    serializable = {
        case_key: {
            "short_label": payload["short_label"],
            "metrics": payload["metrics"],
        }
        for case_key, payload in results.items()
    }
    out_path = OUTFIG / "augmented_existing_telescope_far_snr1_coarse_grid_scan.json"
    out_path.write_text(json.dumps(serializable, indent=2) + "\n")
    print(montage_pdf)
    print(montage_png)
    print(curves_pdf)
    print(curves_png)
    print(out_path)
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
