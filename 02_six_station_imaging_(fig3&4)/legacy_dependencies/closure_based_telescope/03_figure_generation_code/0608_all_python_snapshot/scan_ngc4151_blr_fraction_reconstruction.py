from __future__ import annotations

import json
from dataclasses import replace
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

BLR_FRACTION_PARAMS = [0.42, 0.55, 0.70, 0.90]
OBSERVING_DAYS = 30
SNR_BOOST = 1.0


def effective_blr_fraction(blr_fraction_param: float) -> float:
    compact = 0.39
    outer = 0.14
    asym = max(1.0 - blr_fraction_param - compact - outer, 0.0)
    return blr_fraction_param / (compact + outer + blr_fraction_param + asym)


def make_source(blr_fraction_param: float) -> ngc.SourceModel:
    return replace(
        ngc.NGC4151,
        key=f"ngc4151_blr{int(round(100 * blr_fraction_param)):02d}",
        name=f"NGC 4151, BLR param {blr_fraction_param:.2f}",
        blr_fraction=blr_fraction_param,
    )


def reconstruct_case(source: ngc.SourceModel) -> dict:
    case = latest.load_case(latest.LAYOUT)
    wt.SNR_BOOST = SNR_BOOST
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    with ngc.patched_source(source):
        bands, stats, truth, axis_uas = wt.simulate_bands(case)

    images: dict[str, np.ndarray] = {"truth": truth}
    metrics: dict[str, dict[str, float]] = {}
    for strategy in ("direct", "split"):
        filled_dirty, _ = latest.stack_dirty_psf(bands, strategy, truth, fill=True)
        sparse_dirty, sparse_psf = latest.stack_dirty_psf(bands, strategy, truth, fill=False)
        clean, _ = base.multiscale_clean(
            sparse_dirty,
            sparse_psf,
            scales_pix=(0.0, 2.0, 4.5, 8.0, 14.0, 22.0),
            gain=0.10,
            max_iter=900,
            threshold_factor=1.3,
        )
        rml = latest.rml_tv_reconstruct(sparse_dirty, sparse_psf)
        images[f"{strategy}_filled_dirty"] = normalize_stack(filled_dirty)
        images[f"{strategy}_multiscale_clean"] = normalize_stack(clean)
        images[f"{strategy}_rml_tv"] = normalize_stack(rml)
        for key in (f"{strategy}_filled_dirty", f"{strategy}_multiscale_clean", f"{strategy}_rml_tv"):
            metrics[key] = latest.image_metrics(source, truth, images[key], axis_uas)

    stats.update(
        {
            "source_key": source.key,
            "blr_fraction_param": source.blr_fraction,
            "effective_blr_fraction": effective_blr_fraction(source.blr_fraction),
            "metrics": metrics,
        }
    )
    return {"source": source, "stats": stats, "truth": truth, "axis_uas": axis_uas, "images": images}


def plot_scan(results: list[dict]) -> tuple[Path, Path]:
    axis_uas = results[0]["axis_uas"]
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    n_rows = len(results)
    col_specs = [
        ("truth", "Input"),
        ("direct_filled_dirty", "Direct filled dirty"),
        ("direct_multiscale_clean", "Direct multiscale CLEAN"),
        ("direct_rml_tv", "Direct RML-TV"),
        ("split_multiscale_clean", "Edge-first CLEAN"),
    ]
    fig, axes = plt.subplots(n_rows, len(col_specs), figsize=(11.2, 2.45 * n_rows), constrained_layout=True)
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
        }
    )
    image_axes = []
    for row, result in enumerate(results):
        eff = result["stats"]["effective_blr_fraction"]
        param = result["stats"]["blr_fraction_param"]
        for col, (key, title) in enumerate(col_specs):
            ax = axes[row, col]
            display = opt.normalize_blr_display(result["images"][key])
            ax.imshow(display, origin="lower", extent=extent, cmap="inferno")
            if key == "truth":
                ax.set_title(f"{title}\nparam={param:.2f}, eff={eff:.2f}")
            else:
                m = result["stats"]["metrics"][key]
                ax.set_title(f"{title}\nBLR r={m['blr_corr']:.2f}, all r={m['global_corr']:.2f}")
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 0:
                ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.018,
        pad=0.01,
    )
    cbar.set_label("norm. brightness", fontsize=6.6)
    fig.suptitle("NGC 4151 BLR brightness-fraction scan, Maunakea top4+5, 30 days, SNR boost = 1", fontsize=10.5, weight="bold")
    png = OUTFIG / "maunakea_top4_plus5_ngc4151_blr_fraction_scan_clean_rml.png"
    pdf = OUTFIG / "maunakea_top4_plus5_ngc4151_blr_fraction_scan_clean_rml.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    results = []
    for fraction in BLR_FRACTION_PARAMS:
        print(f"running BLR fraction parameter {fraction:.2f} (effective {effective_blr_fraction(fraction):.3f})")
        results.append(reconstruct_case(make_source(fraction)))
    pdf, png = plot_scan(results)
    summary = {
        "observing_days": OBSERVING_DAYS,
        "snr_boost": SNR_BOOST,
        "blr_fraction_params": BLR_FRACTION_PARAMS,
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "rows": [
            {
                "source_key": result["stats"]["source_key"],
                "blr_fraction_param": result["stats"]["blr_fraction_param"],
                "effective_blr_fraction": result["stats"]["effective_blr_fraction"],
                "metrics": result["stats"]["metrics"],
            }
            for result in results
        ],
    }
    out = OUTFIG / "maunakea_top4_plus5_ngc4151_blr_fraction_scan_clean_rml_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out)
    print(pdf)
    print(png)
    print(json.dumps(summary["rows"], indent=2))


if __name__ == "__main__":
    main()
