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
import run_rml_validation_pipeline as val
import test_fig3_split_objective_imaging as split_sim
from make_all_closure_global_benchmark_note import AllClosureBenchmark
from make_all_closure_global_benchmark_note_v2 import optimize_split


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "output" / "figures" / "fig3_split_objective_rml"
OUT.mkdir(parents=True, exist_ok=True)


STRATEGY_ORDER = [
    ("edge_uniform", "edge uniform"),
    ("edge_meanrms", "edge mean-RMS opt."),
    ("edge_maxrms", "edge max-RMS opt."),
    ("nmode_joint_scheduled", "N-mode scheduled"),
    ("nmode_joint_rawQFI", "N-mode raw-QFI bound"),
]


def configure_rml() -> None:
    """Keep this diagnostic fast but still use the actual amp+closure RML solver."""
    split_sim.configure()
    # Use the same observing/noise model as the split-objective benchmark.
    amp_rml.SOURCE = ngc.NGC4151
    amp_rml.N_RML = int(getattr(amp_rml, "N_RML", 96))
    amp_rml.OBSERVING_DAYS = 30
    amp_rml.N_TIME_WINDOWS = 36
    amp_rml.EXPOSURE_S = 600.0
    amp_rml.EXPOSURE_GAP_S = 150.0
    amp_rml.FIBER_LOSS_DB_PER_KM = split_sim.FIBER_LOSS_DB_PER_KM
    amp_rml.MODE_FALSE_POSITIVE = split_sim.EPS_STATION
    amp_rml.PAIR_FALSE_POSITIVE = split_sim.EPS_PAIR
    amp_rml.AMP_SIGMA_MODE = "physical"
    amp_rml.PHASE_FLOOR_RAD = 0.0
    amp_rml.AMP_GRAD_WEIGHT = 1.20
    amp_rml.PHASE_GRAD_WEIGHT = 1.00
    # Validation RML: use Adam; 220 steps is enough for a strategy-level check
    # and keeps this script usable in iteration.
    val.OPTIMIZER = "adam"
    val.ADAM_ITER = 220
    val.ADAM_LR = 0.035
    val.FIT_N_PIX = min(96, amp_rml.N_RML)
    val.DISPLAY_SMOOTH_PIX = 1.0
    wt.N_PIX = amp_rml.N_RML


def split_matrices() -> dict[str, np.ndarray]:
    bm = AllClosureBenchmark()
    matrices = {
        "edge_uniform": bm.uniform_split_matrix(),
    }
    for objective, key in (
        ("mean_rms", "edge_meanrms"),
        ("max_rms", "edge_maxrms"),
    ):
        split, _info = optimize_split(bm, objective)
        matrices[key] = split
    return matrices


def build_strategy_start(
    bands: list[dict[str, np.ndarray]],
    strategy: str,
    truth: np.ndarray,
    prior: np.ndarray,
) -> dict[str, np.ndarray]:
    starts = {
        f"{strategy}_dirty": val.rebin_image_average(amp_rml.quick_dirty_start(bands, strategy, truth), val.FIT_N_PIX),
        "prior": amp_rml.project_flux_positive(prior, smooth_pix=0.0),
    }
    return starts


def run_one_strategy(
    *,
    case: aug.NetworkCase,
    bands: list[dict[str, np.ndarray]],
    truth: np.ndarray,
    axis_uas: np.ndarray,
    strategy: str,
    label: str,
) -> dict:
    old_strategy = val.STRATEGY
    val.STRATEGY = strategy
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = build_strategy_start(bands, strategy, truth, prior)
    config = {
        "label": "split_objective_rml",
        "prior": amp_rml.PRIOR_WEIGHT,
        "tv": amp_rml.TV_WEIGHT,
        "entropy": amp_rml.ENTROPY_WEIGHT,
        "step": amp_rml.STEP,
    }
    candidates = []
    for start_name, start in starts.items():
        print(f"[rml] strategy={strategy} start={start_name}", flush=True)
        candidates.append(
            val.run_single_reconstruction(
                case=case,
                bands=bands,
                truth=truth,
                axis_uas=axis_uas,
                prior=prior,
                start_name=start_name,
                start=start,
                config=config,
                split_label="fig3_split_objective_rml",
            )
        )
    val.STRATEGY = old_strategy
    best = min(candidates, key=lambda item: item["validation_score"])
    return {
        "strategy": strategy,
        "label": label,
        "best": best,
        "candidates": candidates,
    }


def phase_diagnostics(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    truth: np.ndarray,
) -> dict[str, dict[str, float]]:
    return split_sim.phase_residual_diagnostics(
        bands,
        case,
        truth,
        [key for key, _ in STRATEGY_ORDER],
    )


def serializable_run(run: dict) -> dict:
    return {
        "strategy": run["strategy"],
        "start": run["start"],
        "validation_score": float(run["validation_score"]),
        "metrics": {k: float(v) for k, v in run["metrics"].items()},
        "residuals": {k: float(v) for k, v in run["residuals"].items()},
    }


def plot_results(
    case: aug.NetworkCase,
    stats: dict,
    truth: np.ndarray,
    axis_uas: np.ndarray,
    results: list[dict],
    phase_stats: dict[str, dict[str, float]],
) -> tuple[Path, Path]:
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    fig = plt.figure(figsize=(14.2, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[0.78, 1.0])

    stations, diameters, _names, is_added = aug.station_table_from_case(case)
    ax_layout = fig.add_subplot(gs[0, 0])
    for added, marker, color, label in (
        (False, "o", "#005f73", "existing"),
        (True, "^", "#ae2012", "remote 5 m"),
    ):
        mask = is_added == added
        if np.any(mask):
            ax_layout.scatter(
                stations[mask, 0],
                stations[mask, 1],
                s=34,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
            )
    ax_layout.scatter([case.hub_km[0]], [case.hub_km[1]], s=58, marker="*", color="#ca6702", label="hub")
    for i, j in base.edge_list(len(stations)):
        ax_layout.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.84", lw=0.42)
    for idx, (x, y) in enumerate(stations):
        ax_layout.text(x + 0.25, y + 0.25, f"S{idx + 1}\n{diameters[idx]:g}m", fontsize=5.4)
    ax_layout.set_aspect("equal", adjustable="box")
    ax_layout.set_xlabel("east (km)")
    ax_layout.set_ylabel("north (km)")
    ax_layout.set_title("array + hub")
    ax_layout.legend(frameon=False, fontsize=5.8, loc="best")

    ax_uv = fig.add_subplot(gs[0, 1])
    for wavelength, color, alpha in (("400", "#005f73", 0.52), ("800", "#ee9b00", 0.45)):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax_uv.scatter(uu, vv, s=1.2, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax_uv.scatter(-uu, -vv, s=1.2, color=color, alpha=alpha * 0.65)
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    for theta_uas, ls in ((60.0, ":"), (30.0, "--"), (10.0, "-.")):
        radius = 1.0 / (theta_uas * base.UAS_TO_RAD) / 1.0e9
        ax_uv.plot(radius * np.cos(theta), radius * np.sin(theta), ls=ls, lw=0.55, color="0.35")
    ax_uv.set_aspect("equal", adjustable="box")
    ax_uv.set_xlabel(r"$u$ (G$\lambda$)")
    ax_uv.set_ylabel(r"$v$ (G$\lambda$)")
    ax_uv.set_title("UV coverage")
    ax_uv.legend(frameon=False, fontsize=5.8, loc="upper right")

    ax_input = fig.add_subplot(gs[0, 2])
    ax_input.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax_input.set_title("input source")
    ax_input.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax_input.set_ylabel(r"$\Delta\delta$ ($\mu$as)")

    ax_text = fig.add_subplot(gs[0, 3:])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        (
            "Same amplitude+closure RML solver for all panels.\n"
            "The previous dirty/coarse plot was only a failed diagnostic.\n\n"
            f"{stats['n_station']} stations, {stats['n_closure']} closure coordinates\n"
            f"rank-share for scheduled N-mode = {stats['rank_share']:.2f}\n"
            f"loss = {stats['fiber_loss_db_per_km']:.2f} dB/km, "
            f"eps_station={stats['eps_station']:.3g}, eps_pair={stats['eps_pair']:.3g}\n"
            "raw-QFI is an upper bound, not a constructed simultaneous POVM."
        ),
        ha="left",
        va="top",
        fontsize=7.0,
    )

    image_axes = [ax_input]
    for idx, result in enumerate(results):
        ax = fig.add_subplot(gs[1, idx])
        image = result["best"]["image"]
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        metric = result["best"]["metrics"]
        residual = result["best"]["residuals"]
        ph = phase_stats[result["strategy"]]
        ax.set_title(
            f"{result['label']}\n"
            f"BLR={metric['blr_corr']:.2f}, all={metric['global_corr']:.2f}; "
            rf"$\chi^2_A$={residual['amp_reduced_chi2']:.2g}, $\chi^2_\phi$={residual['phase_reduced_chi2']:.2g}"
            f"\nphase RMS={ph['phase_resid_rms_rad']:.2f} rad",
            fontsize=7.2,
        )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if idx == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        else:
            ax.set_yticklabels([])
        image_axes.append(ax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.020,
        pad=0.012,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.8)
    fig.suptitle(
        "Fig.3 split-objective RML diagnostic: edge splitting vs N-mode benchmark",
        fontsize=10.4,
        weight="bold",
    )
    tag = "fig3_split_objective_amp_closure_rml"
    png = OUT / f"{tag}.png"
    pdf = OUT / f"{tag}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    configure_rml()
    case = amp_rml.load_maunakea_plus3_case()
    splits = split_matrices()
    with ngc.patched_source(ngc.NGC4151):
        bands, stats, truth, axis_uas = split_sim.simulate_bands_with_strategies(case, splits)
    phase_stats = phase_diagnostics(bands, case, truth)
    results = [
        run_one_strategy(
            case=case,
            bands=bands,
            truth=truth,
            axis_uas=axis_uas,
            strategy=strategy,
            label=label,
        )
        for strategy, label in STRATEGY_ORDER
    ]
    pdf, png = plot_results(case, stats, truth, axis_uas, results, phase_stats)

    csv_path = OUT / "fig3_split_objective_amp_closure_rml_metrics.csv"
    with csv_path.open("w", newline="") as f:
        rows = []
        for result in results:
            best = serializable_run(result["best"])
            rows.append(
                {
                    "strategy": result["strategy"],
                    "label": result["label"],
                    "best_start": best["start"],
                    "validation_score": best["validation_score"],
                    **{f"metric_{k}": v for k, v in best["metrics"].items()},
                    **{f"resid_{k}": v for k, v in best["residuals"].items()},
                    **phase_stats[result["strategy"]],
                }
            )
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(csv_path),
        "stats": stats,
        "phase_diagnostics": phase_stats,
        "results": [
            {
                "strategy": result["strategy"],
                "label": result["label"],
                "best": serializable_run(result["best"]),
                "candidates": [serializable_run(item) for item in result["candidates"]],
            }
            for result in results
        ],
        "method_note": (
            "This is the amplitude+closure-phase RML solver, not the dirty/coarse-interp diagnostic. "
            "N-mode rawQFI is plotted as an upper-bound benchmark; scheduled N-mode applies the rank-share factor."
        ),
    }
    json_path = OUT / "fig3_split_objective_amp_closure_rml_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(result["strategy"], result["best"]["metrics"], result["best"]["residuals"])


if __name__ == "__main__":
    main()
