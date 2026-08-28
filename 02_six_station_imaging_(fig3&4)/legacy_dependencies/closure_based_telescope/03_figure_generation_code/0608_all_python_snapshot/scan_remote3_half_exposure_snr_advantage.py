from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import optimize_rml_prior_weights_remote4 as prior_scan
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_blr_optimized as opt
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_remote3_half_exposure_snr_scan_20260525"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = [
    ("all", "All visibilities + drift", "all_dirty", "#8d99ae"),
    ("split", "Edge-first closure", "split_dirty", "#0077b6"),
    ("direct", "Direct closure", "direct_dirty", "#d00000"),
]

REMOTE3_PRIOR = prior_scan.PriorConfig("remote3_opt", prior=0.03, tv=0.03, entropy=0.005)
SNR_GRID = [0.25, 0.35, 0.50, 0.70, 1.00, 1.40, 2.00, 3.00]


def configure_runtime(*, exposure_s: float, snr_boost: float, adam_iter: int) -> None:
    amp_rml.SOURCE = ngc.NGC4151
    ngc.SOURCE_MORPHOLOGY = "lopsided_crescent"
    amp_rml.N_RML = 64
    amp_rml.OBSERVING_DAYS = 30
    amp_rml.N_TIME_WINDOWS = 36
    amp_rml.EXPOSURE_S = exposure_s
    amp_rml.EXPOSURE_GAP_S = 150.0
    amp_rml.SNR_BOOST = snr_boost
    amp_rml.FIBER_LOSS_DB_PER_KM = 0.2
    amp_rml.MODE_FALSE_POSITIVE = 0.05
    amp_rml.PAIR_FALSE_POSITIVE = 0.0
    amp_rml.AMP_SIGMA_MODE = "physical"
    amp_rml.PHASE_FLOOR_RAD = 0.0
    amp_rml.AMP_GRAD_WEIGHT = 0.7
    amp_rml.PHASE_GRAD_WEIGHT = 2.4
    amp_rml.PRIOR_WEIGHT = REMOTE3_PRIOR.prior
    amp_rml.TV_WEIGHT = REMOTE3_PRIOR.tv
    amp_rml.ENTROPY_WEIGHT = REMOTE3_PRIOR.entropy
    amp_rml.STEP = 0.018
    val.FIT_N_PIX = 32
    val.ADAM_ITER = adam_iter
    val.ADAM_LR = 0.012
    val.ADAM_TARGET_AMP_CHI2 = 0.0
    val.ADAM_TARGET_PHASE_CHI2 = 0.0
    val.DISPLAY_SMOOTH_PIX = 1.0
    val.OPTIMIZER = "adam"


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


def simulate_dataset(*, exposure_s: float, snr_boost: float, adam_iter: int):
    configure_runtime(exposure_s=exposure_s, snr_boost=snr_boost, adam_iter=adam_iter)
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    print(f"[simulate] exposure={exposure_s:g}s snr={snr_boost:g}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    return case, bands, stats, truth, axis_uas, prior, starts


def run_strategy_single(
    strategy: str,
    label: str,
    start_name: str,
    *,
    case,
    bands,
    truth,
    axis_uas,
    prior,
    starts,
) -> dict:
    old_strategy = val.STRATEGY
    old_optimizer = val.OPTIMIZER
    val.STRATEGY = strategy
    val.OPTIMIZER = "adam"
    config = {
        "label": REMOTE3_PRIOR.key,
        "prior": REMOTE3_PRIOR.prior,
        "tv": REMOTE3_PRIOR.tv,
        "entropy": REMOTE3_PRIOR.entropy,
        "step": amp_rml.STEP,
    }
    print(f"[rml] strategy={strategy} start={start_name}", flush=True)
    result = val.run_single_reconstruction(
        case=case,
        bands=bands,
        truth=truth,
        axis_uas=axis_uas,
        prior=prior,
        start_name=start_name,
        start=starts[start_name],
        config=config,
        split_label="remote3_snr_scan",
    )
    val.STRATEGY = old_strategy
    val.OPTIMIZER = old_optimizer
    return {"strategy": strategy, "label": label, "best": result, "candidates": [result]}


def run_strategy_two_start(
    strategy: str,
    label: str,
    start_name: str,
    *,
    case,
    bands,
    truth,
    axis_uas,
    prior,
    starts,
) -> dict:
    first = run_strategy_single(
        strategy,
        label,
        start_name,
        case=case,
        bands=bands,
        truth=truth,
        axis_uas=axis_uas,
        prior=prior,
        starts=starts,
    )
    second = run_strategy_single(
        strategy,
        label,
        "prior",
        case=case,
        bands=bands,
        truth=truth,
        axis_uas=axis_uas,
        prior=prior,
        starts=starts,
    )
    candidates = [first["best"], second["best"]]
    best = min(candidates, key=lambda item: item["validation_score"])
    return {"strategy": strategy, "label": label, "best": best, "candidates": candidates}


def rows_from_results(*, exposure_s: float, snr_boost: float, results: list[dict], truth, axis_uas, scan_mode: str) -> list[dict]:
    rows = []
    for result in results:
        best = result["best"]
        metrics = best["metrics"]
        residuals = best["residuals"]
        rows.append(
            {
                "exposure_s": exposure_s,
                "snr_boost": snr_boost,
                "scan_mode": scan_mode,
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


def summarize_advantage(rows: list[dict]) -> list[dict]:
    by_snr: dict[float, dict[str, dict]] = {}
    for row in rows:
        by_snr.setdefault(float(row["snr_boost"]), {})[str(row["strategy"])] = row
    out = []
    for snr, items in sorted(by_snr.items()):
        if "direct" not in items or "split" not in items:
            continue
        d = items["direct"]
        s = items["split"]
        a = items.get("all")
        out.append(
            {
                "snr_boost": snr,
                "direct_minus_split_blr_corr": float(d["blr_corr"] - s["blr_corr"]),
                "split_over_direct_profile_rmse": float(s["profile_rmse"] / max(d["profile_rmse"], 1e-12)),
                "split_over_direct_phase_chi2": float(s["phase_chi2"] / max(d["phase_chi2"], 1e-12)),
                "direct_minus_all_blr_corr": float(d["blr_corr"] - a["blr_corr"]) if a else float("nan"),
                "all_over_direct_profile_rmse": float(a["profile_rmse"] / max(d["profile_rmse"], 1e-12)) if a else float("nan"),
                "direct_blr_corr": float(d["blr_corr"]),
                "split_blr_corr": float(s["blr_corr"]),
                "all_blr_corr": float(a["blr_corr"]) if a else float("nan"),
                "direct_profile_rmse": float(d["profile_rmse"]),
                "split_profile_rmse": float(s["profile_rmse"]),
                "all_profile_rmse": float(a["profile_rmse"]) if a else float("nan"),
            }
        )
    return out


def run_one_setting(*, exposure_s: float, snr_boost: float, adam_iter: int, two_start: bool, scan_mode: str):
    case, bands, stats, truth, axis_uas, prior, starts = simulate_dataset(
        exposure_s=exposure_s,
        snr_boost=snr_boost,
        adam_iter=adam_iter,
    )
    results = []
    runner = run_strategy_two_start if two_start else run_strategy_single
    for strategy, label, start_name, _color in STRATEGIES:
        results.append(
            runner(
                strategy,
                label,
                start_name,
                case=case,
                bands=bands,
                truth=truth,
                axis_uas=axis_uas,
                prior=prior,
                starts=starts,
            )
        )
    rows = rows_from_results(
        exposure_s=exposure_s,
        snr_boost=snr_boost,
        results=results,
        truth=truth,
        axis_uas=axis_uas,
        scan_mode=scan_mode,
    )
    return {
        "case": case,
        "bands": bands,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "results": results,
        "rows": rows,
    }


def plot_half_exposure(dataset: dict) -> tuple[Path, Path]:
    axis = dataset["axis_uas"]
    truth = dataset["truth"]
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    result_by = {item["strategy"]: item for item in dataset["results"]}
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), constrained_layout=True)
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
                rf"$\chi_A^2$={r['amp_reduced_chi2']:.2f}, $\chi_\phi^2$={r['phase_reduced_chi2']:.2f}",
                fontsize=8.2,
            )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    theta, truth_prof = angular_profile(truth, axis)
    axes[1, 0].plot(np.rad2deg(theta), truth_prof, color="black", lw=2.2, label="input")
    for strategy, label, _start, color in STRATEGIES:
        _, prof = angular_profile(result_by[strategy]["best"]["image"], axis)
        axes[1, 0].plot(np.rad2deg(theta), prof, color=color, lw=1.5, label=label)
    axes[1, 0].set_title("BLR annular profile")
    axes[1, 0].set_xlabel("azimuth angle (deg)")
    axes[1, 0].set_ylabel("mean-normalized brightness")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=7)
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
        "Remote3, half per-sample exposure: 36 x 5 min, fixed optimized prior",
        weight="bold",
    )
    png = OUT / "remote3_half_exposure_strategy_comparison.png"
    pdf = OUT / "remote3_half_exposure_strategy_comparison.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_snr_scan(rows: list[dict], advantage_rows: list[dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), constrained_layout=True)
    colors = {"all": "#8d99ae", "split": "#0077b6", "direct": "#d00000"}
    for strategy in ("all", "split", "direct"):
        srows = sorted([row for row in rows if row["strategy"] == strategy], key=lambda row: row["snr_boost"])
        x = [row["snr_boost"] for row in srows]
        axes[0, 0].plot(x, [row["blr_corr"] for row in srows], "o-", color=colors[strategy], label=strategy)
        axes[0, 1].plot(x, [row["profile_rmse"] for row in srows], "o-", color=colors[strategy], label=strategy)
        axes[1, 0].plot(x, [row["phase_chi2"] for row in srows], "o-", color=colors[strategy], label=strategy)
    adv = sorted(advantage_rows, key=lambda row: row["snr_boost"])
    x = [row["snr_boost"] for row in adv]
    axes[1, 1].plot(x, [row["direct_minus_split_blr_corr"] for row in adv], "o-", color="#d00000", label=r"$\Delta$ BLR corr")
    axes[1, 1].plot(
        x,
        [row["split_over_direct_profile_rmse"] - 1.0 for row in adv],
        "s-",
        color="#0077b6",
        label="profile RMSE gain - 1",
    )
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xlabel("global SNR boost")
        ax.grid(alpha=0.25)
    axes[0, 0].set_title("BLR correlation")
    axes[0, 1].set_title("BLR profile RMSE")
    axes[1, 0].set_title("phase reduced chi-square")
    axes[1, 1].set_title("direct advantage over edge-first")
    axes[0, 0].set_ylabel("higher is better")
    axes[0, 1].set_ylabel("lower is better")
    axes[1, 0].set_ylabel("lower is better")
    axes[1, 1].set_ylabel("positive favors direct")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Remote3 SNR scan at half per-sample exposure", weight="bold")
    png = OUT / "remote3_half_exposure_snr_scan.png"
    pdf = OUT / "remote3_half_exposure_snr_scan.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    print("[half exposure full comparison]", flush=True)
    half_dataset = run_one_setting(exposure_s=300.0, snr_boost=1.0, adam_iter=1600, two_start=True, scan_mode="half_exposure_two_start")
    half_pdf, half_png = plot_half_exposure(half_dataset)

    scan_rows: list[dict] = []
    for snr in SNR_GRID:
        print(f"[snr scan] boost={snr:g}", flush=True)
        dataset = run_one_setting(exposure_s=300.0, snr_boost=snr, adam_iter=900, two_start=False, scan_mode="snr_scan_dirty_start")
        scan_rows.extend(dataset["rows"])
    advantage_rows = summarize_advantage(scan_rows)
    snr_pdf, snr_png = plot_snr_scan(scan_rows, advantage_rows)

    half_rows = half_dataset["rows"]
    half_csv = OUT / "remote3_half_exposure_metrics.csv"
    with half_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(half_rows[0].keys()))
        writer.writeheader()
        writer.writerows(half_rows)
    scan_csv = OUT / "remote3_half_exposure_snr_scan_metrics.csv"
    with scan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scan_rows)
    advantage_csv = OUT / "remote3_half_exposure_snr_advantage.csv"
    with advantage_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(advantage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(advantage_rows)

    best_blr = max(advantage_rows, key=lambda row: row["direct_minus_split_blr_corr"])
    best_rmse = max(advantage_rows, key=lambda row: row["split_over_direct_profile_rmse"])
    payload = {
        "runtime": {
            "case": half_dataset["case"].key,
            "source": amp_rml.SOURCE.name,
            "morphology": ngc.SOURCE_MORPHOLOGY,
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "half_exposure_s": 300.0,
            "reference_exposure_s": 600.0,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
            "amp_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_weight": amp_rml.PHASE_GRAD_WEIGHT,
            "prior": REMOTE3_PRIOR.prior,
            "tv": REMOTE3_PRIOR.tv,
            "entropy": REMOTE3_PRIOR.entropy,
            "fit_n_pix": val.FIT_N_PIX,
            "shown_n_pix": amp_rml.N_RML,
            "snr_grid": SNR_GRID,
        },
        "notes": {
            "half_exposure": "Complete two-start comparison at SNR_BOOST=1 and EXPOSURE_S=300 s.",
            "snr_scan": "Dirty-start single-start scan with 900 Adam steps for speed; use it as a trend diagnostic.",
        },
        "half_exposure_rows": half_rows,
        "advantage_rows": advantage_rows,
        "best_direct_minus_split_blr_corr": best_blr,
        "best_split_over_direct_profile_rmse": best_rmse,
        "figures": {
            "half_png": str(half_png),
            "half_pdf": str(half_pdf),
            "snr_scan_png": str(snr_png),
            "snr_scan_pdf": str(snr_pdf),
            "half_csv": str(half_csv),
            "scan_csv": str(scan_csv),
            "advantage_csv": str(advantage_csv),
        },
    }
    json_path = OUT / "remote3_half_exposure_snr_scan_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Remote3 half-exposure and SNR scan\n\n"
        "This diagnostic keeps the remote3 optimized prior fixed at p=0.03, TV=0.03, entropy=0.005.\n"
        "It halves the per-sample exposure from 600 s to 300 s and compares all-vis, edge-first,\n"
        "and direct closure.  It then scans a global SNR boost at the same 300 s exposure using a\n"
        "fast dirty-start RML run to locate where direct closure has the largest advantage.\n"
    )
    print(half_png)
    print(snr_png)
    print(advantage_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
