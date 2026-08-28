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
import hawaii3_compact_case
import optimize_rml_prior_weights_remote4 as prior_scan
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_blr_optimized as opt
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_prior_weight_optimization_remote3_remote4_20260525"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = prior_scan.STRATEGIES


@dataclass(frozen=True)
class LayoutSpec:
    key: str
    label: str
    case: object


def simulate_for_layout(spec: LayoutSpec):
    prior_scan.configure_runtime(scan=True)
    print(f"[simulate] {spec.key}: {spec.case.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(spec.case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    return {
        "spec": spec,
        "bands": bands,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "prior": prior,
        "starts": starts,
    }


def strip_result(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "result"}


def scan_layout(data: dict) -> tuple[prior_scan.PriorConfig, list[dict]]:
    configs = prior_scan.make_configs()
    rows = []
    for idx, cfg in enumerate(configs, start=1):
        print(f"[scan {data['spec'].key} {idx}/{len(configs)}] {cfg.key}", flush=True)
        row = prior_scan.run_single_direct_scan(
            cfg,
            case=data["spec"].case,
            bands=data["bands"],
            truth=data["truth"],
            axis_uas=data["axis_uas"],
            prior=data["prior"],
            starts=data["starts"],
        )
        clean = strip_result(row)
        clean["layout"] = data["spec"].key
        clean["layout_label"] = data["spec"].label
        rows.append(clean)
    best = min(rows, key=lambda row: row["score"])
    best_cfg = prior_scan.PriorConfig(
        str(best["config"]),
        float(best["prior"]),
        float(best["tv"]),
        float(best["entropy"]),
    )
    print(f"[best {data['spec'].key}] {best_cfg}", flush=True)
    return best_cfg, rows


def run_final(data: dict, best_cfg: prior_scan.PriorConfig) -> tuple[list[dict], list[dict]]:
    results = prior_scan.evaluate_all_strategies(
        best_cfg,
        case=data["spec"].case,
        bands=data["bands"],
        truth=data["truth"],
        axis_uas=data["axis_uas"],
        prior=data["prior"],
        starts=data["starts"],
    )
    rows = prior_scan.final_rows(best_cfg, results, data["truth"], data["axis_uas"])
    for row in rows:
        row["layout"] = data["spec"].key
        row["layout_label"] = data["spec"].label
        row["n_station"] = len(data["spec"].case.telescopes)
        row["n_baseline"] = len(data["spec"].case.telescopes) * (len(data["spec"].case.telescopes) - 1) // 2
        row["n_closure"] = (len(data["spec"].case.telescopes) - 1) * (len(data["spec"].case.telescopes) - 2) // 2
    return results, rows


def plot_scan_comparison(all_scan_rows: list[dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)
    metrics = [
        ("score", "selection score", "lower is better"),
        ("profile_rmse", "profile RMSE", "lower is better"),
        ("blr_corr", "BLR correlation", "higher is better"),
    ]
    for row_idx, layout in enumerate(("remote3", "remote4")):
        rows = sorted([row for row in all_scan_rows if row["layout"] == layout], key=lambda item: item["score"])[:10]
        labels = [row["config"] for row in rows]
        y = np.arange(len(rows))[::-1]
        for col_idx, (key, title, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            values = [float(row[key]) for row in rows][::-1]
            ax.barh(y, values, color="#d00000" if key != "blr_corr" else "#0077b6", alpha=0.84)
            ax.set_yticks(y)
            ax.set_yticklabels(labels[::-1], fontsize=6.8)
            ax.set_title(f"{layout}: {title}")
            ax.set_xlabel(ylabel)
            ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Prior-weight scan: top direct-closure candidates for each layout", weight="bold")
    png = OUT / "paired_prior_scan_top10.png"
    pdf = OUT / "paired_prior_scan_top10.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_final_comparison(datasets: list[dict], result_map: dict[str, list[dict]], best_cfgs: dict[str, prior_scan.PriorConfig]):
    fig, axes = plt.subplots(len(datasets), 5, figsize=(14.0, 3.6 * len(datasets)), constrained_layout=True)
    if len(datasets) == 1:
        axes = axes[None, :]
    for row_idx, data in enumerate(datasets):
        axis = data["axis_uas"]
        extent = [axis[0], axis[-1], axis[0], axis[-1]]
        results = {item["strategy"]: item for item in result_map[data["spec"].key]}
        best_cfg = best_cfgs[data["spec"].key]
        panels = [("truth", "Input", data["truth"])] + [
            (strategy, label, results[strategy]["best"]["image"]) for strategy, label, *_ in STRATEGIES
        ]
        for col_idx, (key, label, image) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            if key == "truth":
                ax.set_title(f"{data['spec'].label}\nInput")
            else:
                m = results[key]["best"]["metrics"]
                r = results[key]["best"]["residuals"]
                ax.set_title(
                    f"{label}\nBLR r={m['blr_corr']:.2f}, all r={m['global_corr']:.2f}\n"
                    rf"$\chi_A^2$={r['amp_reduced_chi2']:.2f}, $\chi_\phi^2$={r['phase_reduced_chi2']:.2f}",
                    fontsize=7.8,
                )
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col_idx == 0:
                ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        ax = axes[row_idx, 4]
        theta, truth_prof = prior_scan.angular_profile(data["truth"], axis)
        ax.plot(np.rad2deg(theta), truth_prof, color="black", lw=2.2, label="input")
        for strategy, label, _start, color in STRATEGIES:
            _, prof = prior_scan.angular_profile(results[strategy]["best"]["image"], axis)
            ax.plot(np.rad2deg(theta), prof, color=color, lw=1.4, label=label)
        ax.set_title(
            f"BLR profile\nbest: p={best_cfg.prior:g}, TV={best_cfg.tv:g}, e={best_cfg.entropy:g}",
            fontsize=7.8,
        )
        ax.set_xlabel("azimuth (deg)")
        ax.set_ylabel("mean-norm. brightness")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6.8)
    fig.suptitle("Optimized-prior RML comparison: remote3 versus remote4", weight="bold")
    png = OUT / "optimized_prior_remote3_remote4_final.png"
    pdf = OUT / "optimized_prior_remote3_remote4_final.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_metric_bars(final_rows: list[dict]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), constrained_layout=True)
    metrics = [
        ("blr_corr", "BLR correlation", "higher is better"),
        ("profile_rmse", "profile RMSE", "lower is better"),
        ("phase_chi2", "phase chi-square", "lower is better"),
    ]
    layouts = ["remote3", "remote4"]
    x = np.arange(len(layouts))
    width = 0.23
    colors = {"all": "#8d99ae", "split": "#0077b6", "direct": "#d00000"}
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for offset, strategy in zip((-width, 0.0, width), ("all", "split", "direct")):
            vals = [
                float(next(row[metric] for row in final_rows if row["layout"] == layout and row["strategy"] == strategy))
                for layout in layouts
            ]
            ax.bar(x + offset, vals, width=width, label=strategy, color=colors[strategy], alpha=0.86)
        ax.set_xticks(x)
        ax.set_xticklabels(layouts)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("After independent prior optimization for each layout", weight="bold")
    png = OUT / "optimized_prior_metric_bars.png"
    pdf = OUT / "optimized_prior_metric_bars.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    specs = [
        LayoutSpec("remote3", "top4 + remote3", hawaii3_compact_case.make_hawaii3_compact_remote_case()),
        LayoutSpec("remote4", "top4 + remote4", remote_cmp.make_hawaii4_compact_remote_case()),
    ]
    datasets = [simulate_for_layout(spec) for spec in specs]
    all_scan_rows: list[dict] = []
    final_rows: list[dict] = []
    result_map: dict[str, list[dict]] = {}
    best_cfgs: dict[str, prior_scan.PriorConfig] = {}
    for data in datasets:
        best_cfg, scan_rows = scan_layout(data)
        best_cfgs[data["spec"].key] = best_cfg
        all_scan_rows.extend(scan_rows)
        results, rows = run_final(data, best_cfg)
        result_map[data["spec"].key] = results
        final_rows.extend(rows)

    scan_pdf, scan_png = plot_scan_comparison(all_scan_rows)
    final_pdf, final_png = plot_final_comparison(datasets, result_map, best_cfgs)
    bars_pdf, bars_png = plot_metric_bars(final_rows)

    scan_csv = OUT / "paired_direct_prior_scan_metrics.csv"
    with scan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_scan_rows)
    final_csv = OUT / "paired_optimized_prior_strategy_metrics.csv"
    with final_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    payload = {
        "runtime": {
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
        },
        "selection_note": (
            "Each layout is optimized independently using direct-closure profile RMSE plus mild "
            "BLR-correlation and chi-square penalties, then all three readout strategies are rerun "
            "with that layout's fixed best prior weights."
        ),
        "best_priors": {
            key: {"prior": cfg.prior, "tv": cfg.tv, "entropy": cfg.entropy, "config": cfg.key}
            for key, cfg in best_cfgs.items()
        },
        "final_rows": final_rows,
        "figures": {
            "scan_top10_png": str(scan_png),
            "scan_top10_pdf": str(scan_pdf),
            "final_png": str(final_png),
            "final_pdf": str(final_pdf),
            "metric_bars_png": str(bars_png),
            "metric_bars_pdf": str(bars_pdf),
            "scan_csv": str(scan_csv),
            "final_csv": str(final_csv),
        },
    }
    json_path = OUT / "paired_optimized_prior_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Remote3/remote4 independent RML-prior optimization\n\n"
        "Both layouts use the same NGC 4151 lopsided-crescent source, true SNR, 30 days, 0.2 dB/km fiber loss,\n"
        "and phase-led amplitude/closure RML data weights.  For each layout, broad-prior, TV, and entropy\n"
        "weights are optimized on direct closure, then frozen for all-vis, edge-first, and direct comparisons.\n"
    )
    print(final_png)
    print(bars_png)
    print(scan_csv)
    print(final_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
