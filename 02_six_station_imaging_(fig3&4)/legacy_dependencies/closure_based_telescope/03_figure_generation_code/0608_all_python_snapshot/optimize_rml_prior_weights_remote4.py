from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compare_remote3_remote4_phaseled_rml as remote_cmp
import eht_style_amplitude_closure_rml as amp_rml
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_blr_optimized as opt
import run_hawaii3_rml_strategy_comparison as strategy_run
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_prior_weight_optimization_remote4_20260525"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = [
    ("all", "All visibilities + drift", "all_dirty", "#8d99ae"),
    ("split", "Edge-first closure", "split_dirty", "#0077b6"),
    ("direct", "Direct closure", "direct_dirty", "#d00000"),
]


@dataclass(frozen=True)
class PriorConfig:
    key: str
    prior: float
    tv: float
    entropy: float


def configure_runtime(*, scan: bool) -> None:
    amp_rml.SOURCE = ngc.NGC4151
    ngc.SOURCE_MORPHOLOGY = "lopsided_crescent"
    amp_rml.N_RML = 64
    amp_rml.OBSERVING_DAYS = 30
    amp_rml.N_TIME_WINDOWS = 36
    amp_rml.EXPOSURE_S = 600.0
    amp_rml.EXPOSURE_GAP_S = 150.0
    amp_rml.FIBER_LOSS_DB_PER_KM = 0.2
    amp_rml.MODE_FALSE_POSITIVE = 0.05
    amp_rml.PAIR_FALSE_POSITIVE = 0.0
    amp_rml.SNR_BOOST = 1.0
    amp_rml.AMP_SIGMA_MODE = "physical"
    amp_rml.PHASE_FLOOR_RAD = 0.0
    amp_rml.AMP_GRAD_WEIGHT = 0.7
    amp_rml.PHASE_GRAD_WEIGHT = 2.4
    amp_rml.STEP = 0.018
    val.FIT_N_PIX = 32
    val.ADAM_ITER = 650 if scan else 1800
    val.ADAM_LR = 0.012
    val.ADAM_TARGET_AMP_CHI2 = 0.0
    val.ADAM_TARGET_PHASE_CHI2 = 0.0
    val.DISPLAY_SMOOTH_PIX = 1.0
    val.OPTIMIZER = "adam"


def make_configs() -> list[PriorConfig]:
    configs = [
        PriorConfig("baseline", 0.10, 0.045, 0.010),
    ]
    for prior in (0.0, 0.03, 0.08, 0.15):
        for tv in (0.0, 0.010, 0.030, 0.060):
            for entropy in (0.0, 0.005):
                key = f"p{prior:g}_tv{tv:g}_e{entropy:g}".replace(".", "p")
                configs.append(PriorConfig(key, prior, tv, entropy))
    dedup: dict[tuple[float, float, float], PriorConfig] = {}
    for cfg in configs:
        dedup[(cfg.prior, cfg.tv, cfg.entropy)] = cfg
    return list(dedup.values())


def angular_profile(image: np.ndarray, axis_uas: np.ndarray, n_bin: int = 72):
    source = ngc.NGC4151
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    half_width = max(2.2 * source.blr_width_uas, 10.0)
    mask = (rr > source.blr_radius_uas - half_width) & (rr < source.blr_radius_uas + half_width)
    bins = np.linspace(-np.pi, np.pi, n_bin + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    prof = np.zeros(n_bin, dtype=float)
    for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = mask & (theta >= lo) & (theta < hi)
        prof[k] = float(np.mean(image[m])) if np.any(m) else 0.0
    prof /= max(float(np.mean(prof[prof > 0])), 1e-30) if np.any(prof > 0) else 1.0
    return centers, prof


def profile_rmse(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> float:
    _, truth_prof = angular_profile(truth, axis_uas)
    _, image_prof = angular_profile(image, axis_uas)
    return float(np.sqrt(np.mean((image_prof - truth_prof) ** 2)))


def simulate_case():
    configure_runtime(scan=True)
    case = remote_cmp.make_hawaii4_compact_remote_case()
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    return case, bands, stats, truth, axis_uas, prior, starts


def run_single_direct_scan(
    cfg: PriorConfig,
    *,
    case,
    bands,
    truth,
    axis_uas,
    prior,
    starts,
) -> dict:
    configure_runtime(scan=True)
    val.STRATEGY = "direct"
    config = {"label": cfg.key, "prior": cfg.prior, "tv": cfg.tv, "entropy": cfg.entropy, "step": amp_rml.STEP}
    result = val.run_single_reconstruction(
        case=case,
        bands=bands,
        truth=truth,
        axis_uas=axis_uas,
        prior=prior,
        start_name="direct_dirty",
        start=starts["direct_dirty"],
        config=config,
        split_label="prior_scan_direct",
    )
    metrics = result["metrics"]
    residuals = result["residuals"]
    prof = profile_rmse(truth, result["image"], axis_uas)
    chi_penalty = max(float(residuals["amp_reduced_chi2"]) - 1.2, 0.0) + max(
        float(residuals["phase_reduced_chi2"]) - 1.2, 0.0
    )
    score = prof + 0.20 * max(1.0 - float(metrics["blr_corr"]), 0.0) + 0.25 * chi_penalty
    return {
        "config": cfg.key,
        "prior": cfg.prior,
        "tv": cfg.tv,
        "entropy": cfg.entropy,
        "score": float(score),
        "profile_rmse": float(prof),
        "global_corr": float(metrics["global_corr"]),
        "blr_corr": float(metrics["blr_corr"]),
        "radial_corr": float(metrics.get("radial_corr", np.nan)),
        "amp_chi2": float(residuals["amp_reduced_chi2"]),
        "phase_chi2": float(residuals["phase_reduced_chi2"]),
        "result": result,
    }


def evaluate_all_strategies(best_cfg: PriorConfig, *, case, bands, truth, axis_uas, prior, starts) -> list[dict]:
    configure_runtime(scan=False)
    amp_rml.PRIOR_WEIGHT = best_cfg.prior
    amp_rml.TV_WEIGHT = best_cfg.tv
    amp_rml.ENTROPY_WEIGHT = best_cfg.entropy
    results = []
    for strategy, label, start_name, _color in STRATEGIES:
        print(f"[final-rml] {strategy} with {best_cfg.key}", flush=True)
        results.append(
            strategy_run.run_strategy(
                strategy,
                label,
                start_name,
                case,
                bands,
                truth,
                axis_uas,
                prior,
                starts,
            )
        )
    return results


def final_rows(best_cfg: PriorConfig, results: list[dict], truth: np.ndarray, axis_uas: np.ndarray) -> list[dict]:
    rows = []
    for result in results:
        best = result["best"]
        metrics = best["metrics"]
        residuals = best["residuals"]
        rows.append(
            {
                "config": best_cfg.key,
                "prior": best_cfg.prior,
                "tv": best_cfg.tv,
                "entropy": best_cfg.entropy,
                "strategy": result["strategy"],
                "best_start": best["start"],
                "global_corr": float(metrics["global_corr"]),
                "blr_corr": float(metrics["blr_corr"]),
                "profile_rmse": profile_rmse(truth, best["image"], axis_uas),
                "amp_chi2": float(residuals["amp_reduced_chi2"]),
                "phase_chi2": float(residuals["phase_reduced_chi2"]),
            }
        )
    return rows


def plot_scan_summary(scan_rows: list[dict]) -> tuple[Path, Path]:
    rows_sorted = sorted(scan_rows, key=lambda row: row["score"])
    top = rows_sorted[:12]
    labels = [row["config"] for row in top]
    y = np.arange(len(top))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.2), constrained_layout=True)
    for ax, key, title in (
        (axes[0], "score", "selection score"),
        (axes[1], "profile_rmse", "profile RMSE"),
        (axes[2], "blr_corr", "BLR correlation"),
    ):
        values = [float(row[key]) for row in top][::-1]
        ax.barh(y, values, color="#d00000" if key != "blr_corr" else "#0077b6", alpha=0.82)
        ax.set_yticks(y)
        ax.set_yticklabels(labels[::-1], fontsize=7)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Direct-closure prior-weight scan: best 12 configurations", weight="bold")
    png = OUT / "prior_scan_top12.png"
    pdf = OUT / "prior_scan_top12.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_final(best_cfg: PriorConfig, results: list[dict], truth: np.ndarray, axis_uas: np.ndarray) -> tuple[Path, Path]:
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    result_by = {item["strategy"]: item for item in results}
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.3), constrained_layout=True)
    panels = [("truth", "Input source", truth)] + [
        (strategy, label, result_by[strategy]["best"]["image"]) for strategy, label, *_ in STRATEGIES
    ]
    for col, (strategy, label, image) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        if strategy == "truth":
            ax.set_title(label)
        else:
            m = result_by[strategy]["best"]["metrics"]
            r = result_by[strategy]["best"]["residuals"]
            ax.set_title(
                f"{label}\nBLR r={m['blr_corr']:.2f}, all r={m['global_corr']:.2f}\n"
                rf"$\chi^2_A$={r['amp_reduced_chi2']:.2f}, $\chi^2_\phi$={r['phase_reduced_chi2']:.2f}",
                fontsize=8.2,
            )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    theta, truth_prof = angular_profile(truth, axis_uas)
    ax = axes[1, 0]
    ax.plot(np.rad2deg(theta), truth_prof, color="black", lw=2.2, label="input")
    for strategy, label, _start, color in STRATEGIES:
        _, prof = angular_profile(result_by[strategy]["best"]["image"], axis_uas)
        ax.plot(np.rad2deg(theta), prof, color=color, lw=1.5, label=label)
    ax.set_title("BLR annular profile")
    ax.set_xlabel("azimuth angle (deg)")
    ax.set_ylabel("mean-normalized brightness")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    truth_disp = opt.normalize_blr_display(truth)
    for col, (strategy, label, *_rest) in enumerate(STRATEGIES, start=1):
        ax = axes[1, col]
        residual = opt.normalize_blr_display(result_by[strategy]["best"]["image"]) - truth_disp
        vmax = max(0.08, float(np.percentile(np.abs(residual), 99.0)))
        im = ax.imshow(residual, origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label} residual", fontsize=8.2)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"Optimized-prior comparison: {best_cfg.key} "
        f"(prior={best_cfg.prior:g}, TV={best_cfg.tv:g}, entropy={best_cfg.entropy:g})",
        weight="bold",
    )
    png = OUT / "optimized_prior_strategy_comparison.png"
    pdf = OUT / "optimized_prior_strategy_comparison.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    case, bands, stats, truth, axis_uas, prior, starts = simulate_case()
    configs = make_configs()
    scan_rows = []
    scan_results = []
    for idx, cfg in enumerate(configs, start=1):
        print(f"[scan {idx}/{len(configs)}] {cfg}", flush=True)
        row = run_single_direct_scan(cfg, case=case, bands=bands, truth=truth, axis_uas=axis_uas, prior=prior, starts=starts)
        scan_results.append(row)
        scan_rows.append({key: value for key, value in row.items() if key != "result"})
    best_row = min(scan_rows, key=lambda row: row["score"])
    best_cfg = PriorConfig(
        str(best_row["config"]),
        float(best_row["prior"]),
        float(best_row["tv"]),
        float(best_row["entropy"]),
    )
    print(f"[best] {best_cfg}", flush=True)
    final_results = evaluate_all_strategies(best_cfg, case=case, bands=bands, truth=truth, axis_uas=axis_uas, prior=prior, starts=starts)
    rows = final_rows(best_cfg, final_results, truth, axis_uas)
    scan_pdf, scan_png = plot_scan_summary(scan_rows)
    final_pdf, final_png = plot_final(best_cfg, final_results, truth, axis_uas)

    scan_csv = OUT / "direct_prior_scan_metrics.csv"
    with scan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scan_rows)
    final_csv = OUT / "optimized_prior_strategy_metrics.csv"
    with final_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "runtime": {
            "case": case.key,
            "source": amp_rml.SOURCE.name,
            "morphology": ngc.SOURCE_MORPHOLOGY,
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "exposure_s": amp_rml.EXPOSURE_S,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
            "amp_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_weight": amp_rml.PHASE_GRAD_WEIGHT,
            "fit_n_pix": val.FIT_N_PIX,
            "shown_n_pix": amp_rml.N_RML,
            "scan_adam_iter": 650,
            "final_adam_iter": val.ADAM_ITER,
        },
        "selection_note": (
            "The prior grid is selected using direct-closure profile RMSE plus a mild BLR-correlation "
            "and chi-square penalty. This is an oracle simulation diagnostic, not a blind observing prescription."
        ),
        "best_prior": best_row,
        "final_rows": rows,
        "figures": {
            "scan_top12_png": str(scan_png),
            "scan_top12_pdf": str(scan_pdf),
            "optimized_comparison_png": str(final_png),
            "optimized_comparison_pdf": str(final_pdf),
            "scan_csv": str(scan_csv),
            "final_csv": str(final_csv),
        },
    }
    json_path = OUT / "optimized_prior_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    readme = OUT / "README.md"
    readme.write_text(
        "# RML prior-weight optimization for compact Hawaii+4\n\n"
        "This is a simulation-level diagnostic.  It scans broad-prior, TV, and entropy weights using the\n"
        "direct-closure reconstruction of the lopsided-crescent NGC 4151 benchmark, then freezes the best\n"
        "weights and compares all-vis, edge-first closure, and direct closure on the same simulated data.\n"
    )
    print(final_png)
    print(scan_csv)
    print(final_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
