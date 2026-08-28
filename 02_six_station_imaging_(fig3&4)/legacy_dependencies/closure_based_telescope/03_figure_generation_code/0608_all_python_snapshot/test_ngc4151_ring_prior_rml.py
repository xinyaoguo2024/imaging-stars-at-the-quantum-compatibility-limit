from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import latest_maunakea_closure_snr_clean_rml as latest
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from plot_monochromatic_uniform_stack import normalize_stack


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

OBSERVING_DAYS = 30
SNR_BOOST = 1.0
SOURCE = ngc.NGC4151


def ring_prior_fields(axis_uas: np.ndarray, source: ngc.SourceModel) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx**2 + yy**2)
    ring_sigma = max(2.2 * source.blr_width_uas, 12.0)
    core_sigma = max(0.32 * source.blr_radius_uas, 14.0)
    ring = np.exp(-0.5 * ((rr - source.blr_radius_uas) / ring_sigma) ** 2)
    core = np.exp(-0.5 * (rr / core_sigma) ** 2)
    support = np.clip(np.maximum(0.85 * ring, core), 0.0, 1.0)
    template = 0.52 * core / np.sum(core) + 0.48 * ring / np.sum(ring)
    template /= max(np.max(template), 1e-30)
    return support, template


def centered_fft_convolve(image: np.ndarray, otf: np.ndarray, *, adjoint: bool = False) -> np.ndarray:
    kernel = np.conj(otf) if adjoint else otf
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(image)) * kernel)).real


def rml_ring_prior(
    dirty: np.ndarray,
    psf: np.ndarray,
    support: np.ndarray,
    template: np.ndarray,
    *,
    n_iter: int = 430,
    step: float = 0.22,
    tv_weight: float = 0.035,
    l2_weight: float = 0.006,
    outside_weight: float = 0.0,
    template_weight: float = 0.0,
) -> np.ndarray:
    otf = np.fft.fft2(np.fft.ifftshift(psf))
    d = dirty.copy()
    d -= np.percentile(d, 2.0)
    d_scale = np.percentile(np.abs(d), 99.5)
    if d_scale > 0:
        d /= d_scale

    x = np.clip(base.gaussian_filter(d, 1.0), 0.0, None)
    x_scale = np.percentile(x, 99.5)
    if x_scale > 0:
        x /= x_scale

    def rms(arr: np.ndarray) -> float:
        return max(float(np.sqrt(np.mean(arr * arr))), 1e-30)

    outside = (1.0 - support) ** 2
    residual0 = centered_fft_convolve(x, otf) - d
    grad_data0 = centered_fft_convolve(residual0, otf, adjoint=True)
    grad_tv0 = latest.tv_gradient(x)
    grad_out0 = outside * x
    scale0 = np.sum(x) / max(np.sum(template), 1e-30)
    grad_template0 = x - scale0 * template
    norm_data = rms(grad_data0)
    norm_tv = rms(grad_tv0)
    norm_l2 = rms(x)
    norm_out = rms(grad_out0)
    norm_template = rms(grad_template0)

    for iteration in range(n_iter):
        residual = centered_fft_convolve(x, otf) - d
        grad = centered_fft_convolve(residual, otf, adjoint=True) / norm_data
        grad += tv_weight * latest.tv_gradient(x) / norm_tv
        grad += l2_weight * x / norm_l2
        if outside_weight > 0.0:
            grad += outside_weight * outside * x / norm_out
        if template_weight > 0.0:
            # Weak shape prior: match the broad RM-informed core+ring template
            # at the current total flux scale, without fixing asymmetries.
            scale = np.sum(x) / max(np.sum(template), 1e-30)
            grad += template_weight * (x - scale * template) / norm_template
        x -= step * grad
        x = np.clip(x, 0.0, None)
        if (iteration + 1) % 50 == 0:
            x = base.gaussian_filter(x, 0.20)
    return normalize_stack(x)


def run() -> dict:
    case = latest.load_case(latest.LAYOUT)
    wt.SNR_BOOST = SNR_BOOST
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    with ngc.patched_source(SOURCE):
        bands, stats, truth, axis_uas = wt.simulate_bands(case)

    support, template = ring_prior_fields(axis_uas, SOURCE)
    methods = {
        "no_prior": {"outside_weight": 0.0, "template_weight": 0.0},
        "support_medium": {"outside_weight": 0.12, "template_weight": 0.0},
        "support_strong": {"outside_weight": 0.40, "template_weight": 0.0},
        "support_very_strong": {"outside_weight": 1.00, "template_weight": 0.0},
        "template_medium": {"outside_weight": 0.20, "template_weight": 0.16},
        "template_strong": {"outside_weight": 0.35, "template_weight": 0.45},
    }
    images: dict[str, np.ndarray] = {"truth": truth, "support": support, "template": template}
    metrics: dict[str, dict[str, float]] = {}
    for strategy in ("direct", "split"):
        sparse_dirty, sparse_psf = latest.stack_dirty_psf(bands, strategy, truth, fill=False)
        filled_dirty, _ = latest.stack_dirty_psf(bands, strategy, truth, fill=True)
        images[f"{strategy}_filled_dirty"] = normalize_stack(filled_dirty)
        metrics[f"{strategy}_filled_dirty"] = latest.image_metrics(SOURCE, truth, images[f"{strategy}_filled_dirty"], axis_uas)
        for name, params in methods.items():
            image = rml_ring_prior(sparse_dirty, sparse_psf, support, template, **params)
            key = f"{strategy}_{name}"
            images[key] = image
            metrics[key] = latest.image_metrics(SOURCE, truth, image, axis_uas)
    pdf, png = plot(images, metrics, axis_uas)
    stats.update(
        {
            "ring_prior_methods": methods,
            "metrics": metrics,
            "figure_pdf": str(pdf),
            "figure_png": str(png),
        }
    )
    return stats


def plot(images: dict[str, np.ndarray], metrics: dict[str, dict[str, float]], axis_uas: np.ndarray) -> tuple[Path, Path]:
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    rows = [
        ("direct", "Direct closure-space"),
        ("split", "Edge-first closure"),
    ]
    cols = [
        ("truth", "Input"),
        ("filled_dirty", "Filled dirty"),
        ("no_prior", "RML no prior"),
        ("support_strong", "RML strong support"),
        ("template_medium", "RML medium template"),
        ("template_strong", "RML strong template"),
    ]
    fig, axes = plt.subplots(2, len(cols), figsize=(13.0, 4.9), constrained_layout=True)
    plt.rcParams.update(
        {
            "font.size": 6.6,
            "axes.labelsize": 6.6,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
        }
    )
    image_axes = []
    for row_idx, (strategy, row_title) in enumerate(rows):
        for col_idx, (suffix, col_title) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            key = "truth" if suffix == "truth" else f"{strategy}_{suffix}"
            ax.imshow(opt.normalize_blr_display(images[key]), origin="lower", extent=extent, cmap="inferno")
            if suffix == "truth":
                ax.set_title(f"{row_title}\n{col_title}")
            else:
                m = metrics[key]
                ax.set_title(f"{col_title}\nBLR r={m['blr_corr']:.2f}, all r={m['global_corr']:.2f}")
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col_idx == 0:
                ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.018,
        pad=0.01,
    )
    cbar.set_label("norm. brightness", fontsize=6.3)
    fig.suptitle("Ring-prior RML test: Maunakea top4+5, NGC 4151, 30 days, SNR boost = 1", fontsize=10.0, weight="bold")
    png = OUTFIG / "maunakea_top4_plus5_ngc4151_ring_prior_rml_test.png"
    pdf = OUTFIG / "maunakea_top4_plus5_ngc4151_ring_prior_rml_test.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    stats = run()
    out = OUTFIG / "maunakea_top4_plus5_ngc4151_ring_prior_rml_test_summary.json"
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(out)
    print(stats["figure_pdf"])
    print(stats["figure_png"])
    print(json.dumps(stats["metrics"], indent=2))


if __name__ == "__main__":
    main()
