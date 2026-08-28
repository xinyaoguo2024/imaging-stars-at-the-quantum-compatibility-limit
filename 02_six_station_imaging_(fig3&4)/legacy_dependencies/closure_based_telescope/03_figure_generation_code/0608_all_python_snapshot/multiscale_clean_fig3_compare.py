from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import latest_maunakea_closure_snr_clean_rml as latest
import parametric_closure_rml_crescent_core as prm
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

N_CLEAN_PIX = int(os.environ.get("MULTISCALE_CLEAN_N_PIX", "128"))
MAX_ITER = int(os.environ.get("MULTISCALE_CLEAN_MAX_ITER", "1800"))
GAIN = float(os.environ.get("MULTISCALE_CLEAN_GAIN", "0.08"))
THRESHOLD_FACTOR = float(os.environ.get("MULTISCALE_CLEAN_THRESHOLD", "1.15"))
SCALES_PIX = tuple(
    float(x)
    for x in os.environ.get("MULTISCALE_CLEAN_SCALES", "0,1.5,3,6,10,16,24").split(",")
)


def normalize_positive(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    out -= np.percentile(out, 1.0)
    out = np.clip(out, 0.0, None)
    total = float(np.sum(out))
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros_like(out)
    return out / total


def set_shared_resolution() -> None:
    prm.N_MODEL = N_CLEAN_PIX


def run_case(case) -> dict:
    print(f"[simulate] {case.key}", flush=True)
    set_shared_resolution()
    bands, stats, truth, axis_uas = prm.simulate_case(case)
    images: dict[str, np.ndarray] = {"truth": truth}
    metrics: dict[str, dict[str, float]] = {}
    psf_stats: dict[str, dict[str, float]] = {}
    for strategy in prm.STRATEGIES:
        print(f"[dirty/clean] {case.key} strategy={strategy}", flush=True)
        filled_dirty, filled_psf = latest.stack_dirty_psf(bands, strategy, truth, fill=True)
        sparse_dirty, sparse_psf = latest.stack_dirty_psf(bands, strategy, truth, fill=False)
        clean, residual = base.multiscale_clean(
            sparse_dirty,
            sparse_psf,
            scales_pix=SCALES_PIX,
            gain=GAIN,
            max_iter=MAX_ITER,
            threshold_factor=THRESHOLD_FACTOR,
        )
        for label, image in (
            (f"{strategy}_nearest_dirty", filled_dirty),
            (f"{strategy}_sparse_dirty", sparse_dirty),
            (f"{strategy}_multiscale_clean", clean),
        ):
            images[label] = normalize_positive(image)
            metrics[label] = prm.image_metrics(truth, images[label], axis_uas)
        psf_stats[strategy] = {
            "sparse_residual_rms": float(base.robust_rms(residual)),
            "sparse_psf_peak": float(sparse_psf[sparse_psf.shape[0] // 2, sparse_psf.shape[1] // 2]),
            "filled_psf_peak": float(filled_psf[filled_psf.shape[0] // 2, filled_psf.shape[1] // 2]),
        }
    stats.update(
        {
            "method": "measured-cell multiscale CLEAN compared with nearest-fill and sparse dirty maps",
            "n_clean_pix": N_CLEAN_PIX,
            "clean_scales_pix": SCALES_PIX,
            "clean_gain": GAIN,
            "clean_max_iter": MAX_ITER,
            "clean_threshold_factor": THRESHOLD_FACTOR,
            "metrics": metrics,
            "psf_stats": psf_stats,
        }
    )
    return {"case": case, "stats": stats, "truth": truth, "axis_uas": axis_uas, "images": images, "metrics": metrics}


def plot_results(results: list[dict], tag: str) -> tuple[Path, Path]:
    rows = []
    for result in results:
        for strategy in ("split", "direct"):
            rows.append((result, strategy))
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.7, 2.05 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None, :]

    case_labels = {
        "optimal8_ngc4151_hub_m2_m5": "Optimal 8",
        "hawaii_top4_remote3_ngc4151": "Hawaii+3",
        "hawaii_top4_remote4_ngc4151": "Hawaii+4",
    }
    cols = [
        ("truth", "Input"),
        ("nearest_dirty", "Nearest-fill dirty"),
        ("sparse_dirty", "Sparse dirty"),
        ("multiscale_clean", "Multiscale CLEAN"),
    ]
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
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.015,
        pad=0.010,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.6)
    fig.suptitle(
        (
            "Prior-light deconvolution test: multiscale CLEAN vs dirty maps; "
            f"NGC 4151, {prm.OBSERVING_DAYS} d, loss {prm.FIBER_LOSS_DB_PER_KM:g} dB/km"
        ),
        fontsize=9.4,
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
        f"multiscale_clean_fig3_compare_{prm.SOURCE.key}_{prm.OBSERVING_DAYS}d_"
        f"snr{prm.SNR_BOOST:g}_loss{prm.FIBER_LOSS_DB_PER_KM:g}_fp{prm.MODE_FALSE_POSITIVE:g}_"
        f"n{N_CLEAN_PIX}"
    ).replace(".", "p")
    pdf, png = plot_results(results, tag)
    csv_path, json_path = write_outputs(results, tag, pdf, png)
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(result["case"].key)
        for strategy in ("split", "direct"):
            for suffix in ("nearest_dirty", "sparse_dirty", "multiscale_clean"):
                key = f"{strategy}_{suffix}"
                print(" ", key, result["metrics"][key])


if __name__ == "__main__":
    main()
