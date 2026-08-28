from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
import run_hawaii3_hawaii4_amp_closure_rml as hrun
import test_midbaseline_amp_closure_rml as mid


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)


def run_strategy_multistart(case: aug.NetworkCase, strategy: str, *, zero_piston: bool) -> dict:
    old_drift = aug.POST_AVERAGE_DRIFT_STD
    if zero_piston:
        aug.POST_AVERAGE_DRIFT_STD = 0.0
    try:
        bands, stats, truth, axis = amp_rml.simulate_case(case)
    finally:
        aug.POST_AVERAGE_DRIFT_STD = old_drift

    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    prior = amp_rml.broad_gaussian_prior(axis)
    starts = {
        f"{strategy}_dirty": amp_rml.quick_dirty_start(bands, strategy, truth),
        "all_dirty": amp_rml.quick_dirty_start(bands, "all", truth),
        "split_dirty": amp_rml.quick_dirty_start(bands, "split", truth),
        "direct_dirty": amp_rml.quick_dirty_start(bands, "direct", truth),
        "prior": amp_rml.project_flux_positive(prior, smooth_pix=0.0),
    }
    best = None
    for start_name, start in starts.items():
        print(f"[rml] {case.key} strategy={strategy} zero_piston={zero_piston} start={start_name}", flush=True)
        image, history = amp_rml.amplitude_closure_rml(
            bands,
            case,
            strategy,
            prior,
            start,
            fov_rad=fov_rad,
        )
        objective, amp_obj, phase_obj = hrun.data_objective(image, bands, case, strategy, axis, prior)
        item = (objective, start_name, image, history, amp_obj, phase_obj)
        if best is None or objective < best[0]:
            best = item
    assert best is not None
    objective, start_name, image, history, amp_obj, phase_obj = best
    metric = amp_rml.metrics_for(image, truth, axis)
    sigma = np.concatenate([band[f"sigma_{strategy}"] for band in bands])
    return {
        "case": case,
        "strategy": strategy,
        "zero_piston": zero_piston,
        "truth": truth,
        "axis_uas": axis,
        "image": image,
        "metric": metric,
        "selected_start": start_name,
        "objective": objective,
        "amp_objective": amp_obj,
        "phase_objective": phase_obj,
        "sigma_median_rad": float(np.median(sigma)),
        "sigma_p90_rad": float(np.percentile(sigma, 90.0)),
        "stats": stats,
    }


def make_cases() -> list[aug.NetworkCase]:
    return [
        hrun.make_hawaii_remote_case(3),
        hrun.make_hawaii_remote_case(4),
        mid.make_mid_case(4, np.array([1.6, 3.0, 5.0, 6.0]), key="hawaii_mid4_r1p6_3_5_6"),
    ]


def plot_results(results: list[dict], tag: str) -> tuple[Path, Path]:
    cases = []
    for result in results:
        if result["case"].key not in [case.key for case in cases]:
            cases.append(result["case"])
    fig, axes = plt.subplots(len(cases), 4, figsize=(9.0, 2.35 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = axes[None, :]
    image_axes = []
    by_case = {}
    for result in results:
        by_case.setdefault(result["case"].key, {})[
            "raw_phase" if result["strategy"] == "all" and result["zero_piston"] else result["strategy"]
        ] = result

    labels = {
        "raw_phase": "Ideal baseline phases\n(no atmospheric piston)",
        "direct": "Direct closure phases",
    }
    for row, case in enumerate(cases):
        case_results = by_case[case.key]
        stations, _, _, is_added = aug.station_table_from_case(case)
        axis = next(iter(case_results.values()))["axis_uas"]
        extent = [axis[0], axis[-1], axis[0], axis[-1]]

        ax = axes[row, 0]
        ax.scatter(stations[~is_added, 0], stations[~is_added, 1], s=42, color="#1f77b4", label="existing")
        ax.scatter(stations[is_added, 0], stations[is_added, 1], s=50, marker="^", color="#d62728", label="remote")
        ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=66, marker="*", color="#ffb000", label="hub")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(case.key)
        ax.set_xlabel("x east (km)")
        ax.set_ylabel("y north (km)")
        if row == 0:
            ax.legend(fontsize=6)

        ax = axes[row, 1]
        ax.imshow(opt.normalize_blr_display(case_results["raw_phase"]["truth"]), origin="lower", extent=extent, cmap="inferno")
        ax.set_title("Input")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        image_axes.append(ax)

        for col, key in enumerate(("raw_phase", "direct"), start=2):
            result = case_results[key]
            metric = result["metric"]
            ax = axes[row, col]
            ax.imshow(opt.normalize_blr_display(result["image"]), origin="lower", extent=extent, cmap="inferno")
            ax.set_title(
                f"{labels[key]}\n"
                f"BLR={metric['blr_corr']:.2f}, global={metric['global_corr']:.2f}, "
                f"sigma50={result['sigma_median_rad']:.2f}"
            )
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            image_axes.append(ax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.018,
        pad=0.012,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.5)
    fig.suptitle(
        "Upper-bound test: replace closure phases by individual baseline phases",
        fontsize=10.0,
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
        rows.append(
            {
                "case": result["case"].key,
                "strategy": "ideal_raw_baseline_phase" if result["strategy"] == "all" and result["zero_piston"] else result["strategy"],
                "selected_start": result["selected_start"],
                "objective": result["objective"],
                "amp_objective": result["amp_objective"],
                "phase_objective": result["phase_objective"],
                "sigma_median_rad": result["sigma_median_rad"],
                "sigma_p90_rad": result["sigma_p90_rad"],
                **result["metric"],
            }
        )
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(csv_path),
        "interpretation": (
            "The raw-baseline-phase column is an upper bound with station piston set to zero. "
            "It is not an atmosphere-protected observable."
        ),
        "results": [
            {
                "case": row["case"],
                "strategy": row["strategy"],
                "metrics": {k: row[k] for k in ("global_corr", "blr_corr", "ring_contrast", "radial_corr")},
                "sigma_median_rad": row["sigma_median_rad"],
                "selected_start": row["selected_start"],
            }
            for row in rows
        ],
    }
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def main() -> None:
    cases = make_cases()
    results = []
    for case in cases:
        print(f"[simulate] {case.key}", flush=True)
        results.append(run_strategy_multistart(case, "all", zero_piston=True))
        results.append(run_strategy_multistart(case, "direct", zero_piston=False))
    tag = (
        f"raw_baseline_phase_upper_bound_{ngc.NGC4151.key}_{amp_rml.OBSERVING_DAYS}d_"
        f"ampw{amp_rml.AMP_GRAD_WEIGHT:g}_n{amp_rml.N_RML}"
    ).replace(".", "p")
    pdf, png = plot_results(results, tag)
    csv_path, json_path = write_outputs(results, tag, pdf, png)
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(
            result["case"].key,
            "raw_phase" if result["strategy"] == "all" else result["strategy"],
            result["metric"],
            "sigma50",
            result["sigma_median_rad"],
        )


if __name__ == "__main__":
    main()
