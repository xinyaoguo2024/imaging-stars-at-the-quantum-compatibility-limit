from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import parametric_closure_rml_crescent_core as prm
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from plot_monochromatic_uniform_stack import aggregate_cells


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

N_PIX = int(os.environ.get("EHT_CLEAN_N_PIX", "128"))
MAX_ITER = int(os.environ.get("EHT_CLEAN_MAX_ITER", "2600"))
GAIN = float(os.environ.get("EHT_CLEAN_GAIN", "0.065"))
THRESHOLD_FACTOR = float(os.environ.get("EHT_CLEAN_THRESHOLD", "1.05"))
CLEAN_WINDOW_UAS = float(os.environ.get("EHT_CLEAN_WINDOW_UAS", "78.0"))
RESTORE_FWHM_UAS = float(os.environ.get("EHT_CLEAN_RESTORE_FWHM_UAS", "8.0"))
UV_WEIGHT_EXPONENT = float(os.environ.get("EHT_CLEAN_UV_WEIGHT_EXPONENT", "0.0"))
ZERO_SPACING_WEIGHT = float(os.environ.get("EHT_CLEAN_ZERO_SPACING_WEIGHT", "0.08"))
INCLUDE_RESIDUAL = os.environ.get("EHT_CLEAN_INCLUDE_RESIDUAL", "0") == "1"


def robust_rms_masked(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = image[mask] if mask is not None else image.ravel()
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def normalize_positive(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    out -= np.percentile(out, 1.0)
    out = np.clip(out, 0.0, None)
    total = float(np.sum(out))
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros_like(out)
    return out / total


def make_clean_window(axis_uas: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    return xx * xx + yy * yy <= CLEAN_WINDOW_UAS**2


def grid_multifrequency_uniform(
    bands: list[dict[str, np.ndarray]],
    strategy: str,
    *,
    n: int,
    fov_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_num = np.zeros((n, n), dtype=complex)
    weight_sum = np.zeros((n, n), dtype=float)
    occupied_any = np.zeros((n, n), dtype=bool)
    for band in bands:
        grid, occupied, cell_var = aggregate_cells(
            band["u"],
            band["v"],
            band[f"vis_{strategy}"],
            band[f"sigma_{strategy}"],
            n=n,
            fov_rad=fov_rad,
            average_mode="noise",
        )
        valid = occupied & np.isfinite(cell_var) & (cell_var > 0.0)
        if UV_WEIGHT_EXPONENT == 0.0:
            cell_weight = np.ones_like(cell_var)
        else:
            sigma = np.sqrt(cell_var)
            cell_weight = np.power(np.maximum(sigma, 1e-12), UV_WEIGHT_EXPONENT)
        cell_weight = np.where(valid, cell_weight, 0.0)
        positive = cell_weight[cell_weight > 0.0]
        if positive.size:
            cell_weight = np.clip(cell_weight / np.median(positive), 0.05, 20.0)
        data_num[valid] += cell_weight[valid] * grid[valid]
        weight_sum[valid] += cell_weight[valid]
        occupied_any |= valid

    mid = n // 2
    data_num[mid, mid] += ZERO_SPACING_WEIGHT
    weight_sum[mid, mid] += ZERO_SPACING_WEIGHT
    occupied_any[mid, mid] = True

    data_grid = np.zeros_like(data_num)
    valid = weight_sum > 0.0
    data_grid[valid] = data_num[valid] / weight_sum[valid]
    # EHT-style uniform density weighting: every occupied uv cell contributes
    # once after intra-cell averaging.  This avoids dense arcs dominating.
    psf_weight = np.zeros_like(weight_sum)
    psf_weight[valid] = 1.0
    return data_grid, psf_weight, occupied_any


def dirty_psf_from_grid(data_grid: np.ndarray, psf_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dirty = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data_grid * psf_weight))).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(psf_weight))).real
    peak = psf[psf.shape[0] // 2, psf.shape[1] // 2]
    if peak > 0.0:
        dirty /= peak
        psf /= peak
    return dirty, psf


def masked_positive_multiscale_clean(
    dirty: np.ndarray,
    psf: np.ndarray,
    clean_window: np.ndarray,
    *,
    scales_pix: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0, 14.0, 22.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = dirty.shape[0]
    center = n // 2
    residual = dirty.copy()
    model = np.zeros_like(dirty)
    threshold = THRESHOLD_FACTOR * robust_rms_masked(dirty, ~clean_window)
    yy, xx = np.indices((n, n))
    rr2 = (xx - center) ** 2 + (yy - center) ** 2
    kernels = []
    responses = []
    for scale in scales_pix:
        if scale <= 0.0:
            kernel = np.zeros_like(dirty)
            kernel[center, center] = 1.0
        else:
            kernel = np.exp(-0.5 * rr2 / scale**2)
            kernel /= np.sum(kernel)
        response = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(psf)) * np.fft.fft2(np.fft.ifftshift(kernel)))
        ).real
        response_peak = response[center, center]
        if abs(response_peak) > 1e-30:
            response /= response_peak
        kernels.append(kernel)
        responses.append(response)

    for _ in range(MAX_ITER):
        best = None
        for scale, kernel, response in zip(scales_pix, kernels, responses):
            smoothed = residual if scale <= 0.0 else base.gaussian_filter(residual, scale)
            candidate = np.where(clean_window, smoothed, -np.inf)
            iy, ix = np.unravel_index(np.argmax(candidate), candidate.shape)
            peak = candidate[iy, ix]
            score = peak / (1.0 + 0.02 * scale)
            if best is None or score > best[0]:
                best = (score, peak, iy, ix, kernel, response)
        assert best is not None
        _score, peak, iy, ix, kernel, response = best
        if not np.isfinite(peak) or peak < threshold:
            break
        amp = GAIN * peak
        model += amp * np.roll(np.roll(kernel, iy - center, axis=0), ix - center, axis=1)
        residual -= amp * np.roll(np.roll(response, iy - center, axis=0), ix - center, axis=1)

    return model, residual, clean_window


def restore_clean_model(model: np.ndarray, residual: np.ndarray, axis_uas: np.ndarray) -> np.ndarray:
    pix_uas = float(axis_uas[1] - axis_uas[0])
    sigma_pix = RESTORE_FWHM_UAS / (2.355 * max(pix_uas, 1e-12))
    clean = base.gaussian_filter(model, sigma_pix)
    if INCLUDE_RESIDUAL:
        clean = clean + residual
    return normalize_positive(clean)


def run_case(case) -> dict:
    print(f"[simulate] {case.key}", flush=True)
    prm.N_MODEL = N_PIX
    bands, stats, truth, axis_uas = prm.simulate_case(case)
    fov_rad = 2.0 * prm.wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    clean_window = make_clean_window(axis_uas)
    images = {"truth": truth}
    metrics = {}
    clean_stats = {}
    for strategy in prm.STRATEGIES:
        print(f"[eht-clean] {case.key} strategy={strategy}", flush=True)
        data_grid, psf_weight, occupied = grid_multifrequency_uniform(bands, strategy, n=N_PIX, fov_rad=fov_rad)
        dirty, psf = dirty_psf_from_grid(data_grid, psf_weight)
        model, residual, _ = masked_positive_multiscale_clean(dirty, psf, clean_window)
        clean = restore_clean_model(model, residual, axis_uas)
        images[f"{strategy}_dirty_uniform"] = normalize_positive(dirty)
        images[f"{strategy}_eht_clean"] = clean
        metrics[f"{strategy}_dirty_uniform"] = prm.image_metrics(truth, images[f"{strategy}_dirty_uniform"], axis_uas)
        metrics[f"{strategy}_eht_clean"] = prm.image_metrics(truth, clean, axis_uas)
        clean_stats[strategy] = {
            "occupied_cells": int(np.count_nonzero(occupied)),
            "psf_peak": float(psf[N_PIX // 2, N_PIX // 2]),
            "residual_rms_off_window": robust_rms_masked(residual, ~clean_window),
            "model_flux": float(np.sum(model)),
        }
    stats.update(
        {
            "method": "EHT-style masked positive multiscale CLEAN diagnostic",
            "important_caveat": (
                "CLEAN is a calibrated-complex-visibility deconvolver. For split/direct columns, "
                "the input is a gauge-fixed closure-space pseudo-visibility diagnostic rather than "
                "a mathematically exact closure-only likelihood. Closure-only imaging should be done with RML."
            ),
            "n_pix": N_PIX,
            "max_iter": MAX_ITER,
            "gain": GAIN,
            "threshold_factor": THRESHOLD_FACTOR,
            "clean_window_uas": CLEAN_WINDOW_UAS,
            "restore_fwhm_uas": RESTORE_FWHM_UAS,
            "uv_weight_exponent": UV_WEIGHT_EXPONENT,
            "zero_spacing_weight": ZERO_SPACING_WEIGHT,
            "include_residual": INCLUDE_RESIDUAL,
            "metrics": metrics,
            "clean_stats": clean_stats,
        }
    )
    return {"case": case, "stats": stats, "truth": truth, "axis_uas": axis_uas, "images": images, "metrics": metrics}


def plot_results(results: list[dict], tag: str) -> tuple[Path, Path]:
    rows = []
    for result in results:
        for strategy in ("all", "split", "direct"):
            rows.append((result, strategy))
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 1.85 * len(rows)), constrained_layout=True)
    case_labels = {
        "optimal8_ngc4151_hub_m2_m5": "Optimal 8",
        "hawaii_top4_remote3_ngc4151": "Hawaii+3",
        "hawaii_top4_remote4_ngc4151": "Hawaii+4",
    }
    cols = [("truth", "Input"), ("dirty_uniform", "Uniform dirty"), ("eht_clean", "EHT-style CLEAN")]
    image_axes = []
    for row, (result, strategy) in enumerate(rows):
        axis_uas = result["axis_uas"]
        extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
        for col, (suffix, title) in enumerate(cols):
            ax = axes[row, col]
            key = suffix if suffix == "truth" else f"{strategy}_{suffix}"
            image = result["images"][key]
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            if suffix == "truth":
                ax.set_title(title)
            else:
                metric = result["metrics"][key]
                ax.set_title(f"{title}\nBLR={metric['blr_corr']:.2f}, all={metric['global_corr']:.2f}")
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 0:
                ax.set_ylabel(
                    f"{case_labels.get(result['case'].key, result['case'].key)} {strategy}\n"
                    + r"$\Delta\delta$ ($\mu$as)"
                )
            else:
                ax.set_yticklabels([])
            image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap="inferno"),
        ax=image_axes,
        fraction=0.015,
        pad=0.010,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.6)
    fig.suptitle(
        (
            "EHT-style CLEAN diagnostic: uniform uv weighting, compact window, restoring beam; "
            f"{prm.SOURCE.name}, {prm.OBSERVING_DAYS} d"
        ),
        fontsize=9.0,
        weight="bold",
    )
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_outputs(results: list[dict], tag: str, pdf: Path, png: Path) -> tuple[Path, Path]:
    rows = []
    for result in results:
        for key, metric in result["metrics"].items():
            rows.append({"case": result["case"].key, "image": key, **metric})
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(csv_path),
        "results": [result["stats"] for result in results],
    }
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def main() -> None:
    results = [run_case(case) for case in prm.make_cases()]
    tag = (
        f"eht_style_clean_fig3_compare_{prm.SOURCE.key}_{prm.OBSERVING_DAYS}d_"
        f"snr{prm.SNR_BOOST:g}_loss{prm.FIBER_LOSS_DB_PER_KM:g}_fp{prm.MODE_FALSE_POSITIVE:g}_"
        f"n{N_PIX}_beam{RESTORE_FWHM_UAS:g}"
    ).replace(".", "p")
    pdf, png = plot_results(results, tag)
    csv_path, json_path = write_outputs(results, tag, pdf, png)
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(result["case"].key)
        for strategy in prm.STRATEGIES:
            for suffix in ("dirty_uniform", "eht_clean"):
                key = f"{strategy}_{suffix}"
                print(" ", key, result["metrics"][key])


if __name__ == "__main__":
    main()
