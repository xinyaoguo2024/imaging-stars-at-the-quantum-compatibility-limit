from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED_RUNS = RESULTS / "fig3a_seed_runs"
SEEDS = [20260529, 20260530, 20260531, 20260532, 20260533]
EXPOSURES = ["50ms", "100ms"]

STRATEGY_ORDER = [
    ("all", "All visibilities\n+ drift", "#7b2cbf"),
    ("edge_uniform", "Edge-first\nclosure", "#0077b6"),
    ("core4_remote_optimized", "Strict near\npair-combine", "#2a9d8f"),
    ("nmode_joint_scheduled", "Direct optimized\nclosure", "#9d0208"),
]


def run_tag(label: str, seed: int) -> str:
    return f"fig3a_{label}_seed{seed}_paired10loop"


def summary_path(label: str, seed: int) -> Path:
    stem = f"broad_plume_split_objective_nmode_rml_{run_tag(label, seed)}_summary.json"
    return SEED_RUNS / label / f"seed_{seed}" / "rml_outputs" / stem


def load_seed_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in EXPOSURES:
        for seed in SEEDS:
            path = summary_path(label, seed)
            data = json.loads(path.read_text())
            paired = bool(data.get("stats", {}).get("paired_loop_noise", False))
            exposure = float(data.get("stats", {}).get("sample_stress_test", {}).get("effective_exposure_s", 0.0))
            for row in data["rows"]:
                rows.append(
                    {
                        "exposure_label": label,
                        "exposure_s": exposure,
                        "seed": seed,
                        "paired_loop_noise": paired,
                        "strategy": row["strategy"],
                        "label": row["label"],
                        "blr_corr": float(row["blr_corr"]),
                        "global_corr": float(row["global_corr"]),
                        "profile_rmse": float(row["profile_rmse"]),
                        "amp_chi2": float(row["amp_chi2"]),
                        "phase_chi2": float(row["phase_chi2"]),
                        "summary_path": str(path),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["exposure_label"]), str(row["strategy"]))].append(row)
    out: list[dict[str, object]] = []
    for exposure in EXPOSURES:
        for strategy, label, _color in STRATEGY_ORDER:
            vals = grouped[(exposure, strategy)]
            if len(vals) != len(SEEDS):
                raise ValueError(f"Missing rows for {exposure} {strategy}: found {len(vals)}")
            for metric in ("blr_corr", "global_corr", "profile_rmse", "amp_chi2", "phase_chi2"):
                arr = np.asarray([float(row[metric]) for row in vals], dtype=float)
                out.append(
                    {
                        "exposure_label": exposure,
                        "strategy": strategy,
                        "label": label.replace("\n", " "),
                        "metric": metric,
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1)),
                        "sem": float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
                        "n_seed": len(arr),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
                )
    return out


def lookup(summary_rows: list[dict[str, object]], exposure: str, strategy: str, metric: str) -> dict[str, object]:
    for row in summary_rows:
        if row["exposure_label"] == exposure and row["strategy"] == strategy and row["metric"] == metric:
            return row
    raise KeyError((exposure, strategy, metric))


def plot_bars(summary_rows: list[dict[str, object]]) -> tuple[Path, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), sharey="row", constrained_layout=True)
    metric_rows = [("blr_corr", "BLR correlation r"), ("global_corr", "All-pixel correlation r")]
    x = np.arange(len(STRATEGY_ORDER))
    for row_idx, (metric, ylabel) in enumerate(metric_rows):
        for col_idx, exposure in enumerate(EXPOSURES):
            ax = axes[row_idx, col_idx]
            means = []
            errs = []
            colors = []
            labels = []
            for strategy, label, color in STRATEGY_ORDER:
                item = lookup(summary_rows, exposure, strategy, metric)
                means.append(float(item["mean"]))
                errs.append(float(item["std"]))
                colors.append(color)
                labels.append(label)
            ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor="0.2", linewidth=0.6)
            ax.set_title(f"{exposure}, 5 seeds")
            ax.set_xticks(x, labels, fontsize=8)
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, axis="y", color="0.88", linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            for xpos, mean in zip(x, means):
                ax.text(xpos, mean + 0.025, f"{mean:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Fig.3(a) RML benchmark correlations; bars show mean, error bars show seed-to-seed std", fontsize=11)
    png = FIGURES / "fig3a_5seed_50ms_100ms_correlation_bars.png"
    pdf = FIGURES / "fig3a_5seed_50ms_100ms_correlation_bars.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return pdf, png


def plot_bars_100ms(summary_rows: list[dict[str, object]]) -> tuple[Path, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True, constrained_layout=True)
    metric_rows = [("blr_corr", "BLR annulus"), ("global_corr", "all pixels")]
    x = np.arange(len(STRATEGY_ORDER))
    exposure = "100ms"
    short_labels = {
        "all": "All vis.\n+ drift",
        "edge_uniform": "Edge-first",
        "core4_remote_optimized": "Strict\nnear",
        "nmode_joint_scheduled": "Direct\nopt.",
    }
    for ax, (metric, title) in zip(axes, metric_rows):
        means = []
        errs = []
        colors = []
        labels = []
        for strategy, label, color in STRATEGY_ORDER:
            item = lookup(summary_rows, exposure, strategy, metric)
            means.append(float(item["mean"]))
            errs.append(float(item["std"]))
            colors.append(color)
            labels.append(short_labels[strategy])
        ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor="0.2", linewidth=0.6)
        ax.set_title(title)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", color="0.88", linewidth=0.8)
        for xpos, mean in zip(x, means):
            ax.text(xpos, mean + 0.025, f"{mean:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("correlation r")
    fig.suptitle("100 ms RML benchmark, five noise seeds", fontsize=10)
    png = FIGURES / "fig3a_5seed_100ms_correlation_bars.png"
    pdf = FIGURES / "fig3a_5seed_100ms_correlation_bars.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return pdf, png


def main() -> None:
    rows = load_seed_rows()
    write_csv(RESULTS / "fig3a_5seed_seed_metrics.csv", rows)
    summary_rows = summarize(rows)
    write_csv(RESULTS / "fig3a_5seed_summary.csv", summary_rows)
    (RESULTS / "fig3a_5seed_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    pdf, png = plot_bars(summary_rows)
    pdf_100, png_100 = plot_bars_100ms(summary_rows)
    print(pdf)
    print(png)
    print(pdf_100)
    print(png_100)
    print(RESULTS / "fig3a_5seed_seed_metrics.csv")
    print(RESULTS / "fig3a_5seed_summary.csv")


if __name__ == "__main__":
    main()
