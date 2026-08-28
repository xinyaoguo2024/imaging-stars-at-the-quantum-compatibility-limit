from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import optimize_remote3_source_morphology_priors as morph
import optimize_rml_prior_weights_remote4 as prior_scan
import plot_prl_broadband_blr_optimized as opt
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_remote3_broad_plume_snr_gain_scan_20260525"
OUT.mkdir(parents=True, exist_ok=True)

SNR_GAINS = (0.3, 0.5, 0.8, 1.5)
VARIANT = morph.VARIANTS[0]  # broad_plume_irregular_blr
STRATEGIES = prior_scan.STRATEGIES


def configure_runtime(*, scan: bool, snr_gain: float) -> None:
    morph.configure_runtime(scan=scan)
    amp_rml.SNR_BOOST = float(snr_gain)
    val.ADAM_ITER = 650 if scan else 1800


def simulate_variant(snr_gain: float) -> dict:
    configure_runtime(scan=True, snr_gain=snr_gain)
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    with morph.patched_variant(VARIANT):
        print(f"[simulate] {VARIANT.key} snr_gain={snr_gain:g}", flush=True)
        bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    return {
        "case": case,
        "bands": bands,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "prior": prior,
        "starts": starts,
    }


def scan_config_rows(snr_gain: float, data: dict):
    rows = []
    configs = prior_scan.make_configs()
    for idx, cfg in enumerate(configs, start=1):
        print(f"[scan snr={snr_gain:g} {idx}/{len(configs)}] {cfg.key}", flush=True)
        with morph.patched_variant(VARIANT):
            row = prior_scan.run_single_direct_scan(
                cfg,
                case=data["case"],
                bands=data["bands"],
                truth=data["truth"],
                axis_uas=data["axis_uas"],
                prior=data["prior"],
                starts=data["starts"],
            )
        clean = {key: value for key, value in row.items() if key != "result"}
        clean["snr_gain"] = float(snr_gain)
        clean["variant"] = VARIANT.key
        clean["variant_label"] = VARIANT.label
        rows.append(clean)
    best = min(rows, key=lambda row: row["score"])
    cfg = prior_scan.PriorConfig(str(best["config"]), float(best["prior"]), float(best["tv"]), float(best["entropy"]))
    return cfg, rows


def evaluate_gain(snr_gain: float, data: dict, cfg: prior_scan.PriorConfig):
    configure_runtime(scan=False, snr_gain=snr_gain)
    with morph.patched_variant(VARIANT):
        return prior_scan.evaluate_all_strategies(
            cfg,
            case=data["case"],
            bands=data["bands"],
            truth=data["truth"],
            axis_uas=data["axis_uas"],
            prior=data["prior"],
            starts=data["starts"],
        )


def final_rows(snr_gain: float, cfg: prior_scan.PriorConfig, results: list[dict], truth: np.ndarray, axis_uas: np.ndarray):
    rows = prior_scan.final_rows(cfg, results, truth, axis_uas)
    for row in rows:
        row["snr_gain"] = float(snr_gain)
        row["variant"] = VARIANT.key
        row["variant_label"] = VARIANT.label
    return rows


def plot_gain_final(snr_gain: float, data: dict, cfg: prior_scan.PriorConfig, results: list[dict]) -> tuple[Path, Path]:
    axis = data["axis_uas"]
    truth = data["truth"]
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    result_by = {item["strategy"]: item for item in results}
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.25), constrained_layout=True)
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
                f"{label}\nBLR r={m['blr_corr']:.3f}, all r={m['global_corr']:.3f}\n"
                rf"$\chi_A^2$={r['amp_reduced_chi2']:.2f}, $\chi_\phi^2$={r['phase_reduced_chi2']:.2f}",
                fontsize=8.0,
            )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")

    theta, truth_prof = morph.angular_profile(truth, axis)
    axes[1, 0].plot(np.rad2deg(theta), truth_prof, color="black", lw=2.2, label="input")
    for strategy, label, _start, color in STRATEGIES:
        _, prof = morph.angular_profile(result_by[strategy]["best"]["image"], axis)
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
        ax.set_title(f"{label} residual", fontsize=8.0)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"{VARIANT.label}, SNR gain={snr_gain:g}: optimized prior p={cfg.prior:g}, TV={cfg.tv:g}, e={cfg.entropy:g}",
        weight="bold",
    )
    stem = f"broad_plume_snr_gain_{str(snr_gain).replace('.', 'p')}"
    png = OUT / f"{stem}_optimized_strategy_comparison.png"
    pdf = OUT / f"{stem}_optimized_strategy_comparison.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_summary(rows: list[dict]) -> tuple[Path, Path]:
    gains = list(SNR_GAINS)
    colors = {"all": "#8d99ae", "split": "#0077b6", "direct": "#d00000"}
    labels = {"all": "All visibilities + drift", "split": "Edge-first closure", "direct": "Direct closure"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), constrained_layout=True)
    metrics = [
        ("blr_corr", "BLR correlation", "higher is better"),
        ("profile_rmse", "BLR profile RMSE", "lower is better"),
        ("phase_chi2", "phase chi-square", "lower is better"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for strategy in ("all", "split", "direct"):
            vals = [
                float(next(row[metric] for row in rows if row["snr_gain"] == gain and row["strategy"] == strategy))
                for gain in gains
            ]
            ax.plot(gains, vals, marker="o", color=colors[strategy], lw=1.8, label=labels[strategy])
        ax.set_xscale("log")
        ax.set_xticks(gains)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _pos: f"{x:g}"))
        ax.set_title(title)
        ax.set_xlabel("global SNR gain")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{VARIANT.label}: SNR-gain scan with per-gain prior optimization", weight="bold")
    png = OUT / "broad_plume_snr_gain_metric_summary.png"
    pdf = OUT / "broad_plume_snr_gain_metric_summary.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    all_scan_rows: list[dict] = []
    all_final_rows: list[dict] = []
    best_priors: dict[str, dict] = {}
    figures: dict[str, dict] = {}
    for snr_gain in SNR_GAINS:
        data = simulate_variant(snr_gain)
        cfg, scan_rows = scan_config_rows(snr_gain, data)
        key = f"{snr_gain:g}"
        best_priors[key] = {"config": cfg.key, "prior": cfg.prior, "tv": cfg.tv, "entropy": cfg.entropy}
        all_scan_rows.extend(scan_rows)
        print(f"[best snr={snr_gain:g}] {cfg}", flush=True)
        results = evaluate_gain(snr_gain, data, cfg)
        rows = final_rows(snr_gain, cfg, results, data["truth"], data["axis_uas"])
        all_final_rows.extend(rows)
        pdf, png = plot_gain_final(snr_gain, data, cfg, results)
        figures[key] = {"png": str(png), "pdf": str(pdf)}

    summary_pdf, summary_png = plot_summary(all_final_rows)
    scan_csv = OUT / "broad_plume_snr_gain_prior_scan_metrics.csv"
    with scan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_scan_rows)
    final_csv = OUT / "broad_plume_snr_gain_optimized_strategy_metrics.csv"
    with final_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_final_rows)
    payload = {
        "runtime": {
            "case": hawaii3_compact_case.make_hawaii3_compact_remote_case().key,
            "variant": VARIANT.__dict__,
            "snr_gains": list(SNR_GAINS),
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "exposure_s": amp_rml.EXPOSURE_S,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
            "amp_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_weight": amp_rml.PHASE_GRAD_WEIGHT,
            "fit_n_pix": val.FIT_N_PIX,
            "shown_n_pix": amp_rml.N_RML,
        },
        "best_priors": best_priors,
        "final_rows": all_final_rows,
        "figures": {
            **figures,
            "summary_png": str(summary_png),
            "summary_pdf": str(summary_pdf),
            "scan_csv": str(scan_csv),
            "final_csv": str(final_csv),
        },
    }
    json_path = OUT / "broad_plume_snr_gain_scan_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Broad-plume remote3 SNR-gain scan\n\n"
        "This diagnostic fixes the source morphology to the broad plume + irregular BLR variant and reruns\n"
        "the full simulated data generation, direct-closure prior scan, and all/edge/direct RML comparison\n"
        "for global SNR gains 0.3, 0.5, 0.8, and 1.5.\n"
    )
    print(summary_png)
    print(final_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
