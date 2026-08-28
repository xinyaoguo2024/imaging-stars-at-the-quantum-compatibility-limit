from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED_METRICS = RESULTS / "fig3a_5seed_seed_metrics.csv"
EXPOSURE_LABEL = os.environ.get("FIG3_EXPOSURE_LABEL", "100ms").strip()
TRIM_BLR_EXTREMES = os.environ.get("FIG3_TRIM_BLR_EXTREMES", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
LOOP_RMS = RESULTS / f"latest_balanced10_loop_rms_{EXPOSURE_LABEL}.csv"
if not LOOP_RMS.exists():
    LOOP_RMS = RESULTS / "latest_balanced10_loop_rms.csv"

STRATEGIES = [
    ("all", "All vis.\n+ drift", "#7b2cbf"),
    ("edge_uniform", "Edge-first", "#0077b6"),
    ("core4_remote_optimized", "Strict\nnear", "#2a9d8f"),
    ("nmode_joint_scheduled", "Direct\nopt.", "#9d0208"),
]
RMS_STRATEGIES = [
    ("edge_uniform", "edge-first", "#0077b6", "o"),
    ("paircombine_strict_near", "strict near", "#2a9d8f", "s"),
    ("direct_optimized", "direct optimized", "#9d0208", "^"),
]
PLOT_LOOPS = [
    ("123", "core"),
    ("125", "one remote"),
    ("456", "all remote"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_seed_values() -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]], list[dict[str, object]]]:
    rows = [row for row in read_csv(SEED_METRICS) if row["exposure_label"] == EXPOSURE_LABEL]
    grouped: dict[str, list[dict[str, str]]] = {key: [] for key, _label, _color in STRATEGIES}
    for row in rows:
        strategy = row["strategy"]
        if strategy not in grouped:
            continue
        grouped[strategy].append(row)

    seed_values: dict[str, dict[str, np.ndarray]] = {}
    used_rows: list[dict[str, object]] = []
    trim_report: list[dict[str, object]] = []
    for strategy, label, _color in STRATEGIES:
        vals = sorted(grouped[strategy], key=lambda row: int(row["seed"]))
        if not vals:
            seed_values[strategy] = {"seed": np.asarray([]), "blr_corr": np.asarray([]), "global_corr": np.asarray([])}
            continue
        dropped: set[int] = set()
        if TRIM_BLR_EXTREMES and len(vals) >= 3:
            by_blr = sorted(vals, key=lambda row: float(row["blr_corr"]))
            dropped = {int(by_blr[0]["seed"]), int(by_blr[-1]["seed"])}
            trim_report.append(
                {
                    "exposure_label": EXPOSURE_LABEL,
                    "strategy": strategy,
                    "label": label,
                    "drop_low_seed": int(by_blr[0]["seed"]),
                    "drop_low_blr_corr": float(by_blr[0]["blr_corr"]),
                    "drop_high_seed": int(by_blr[-1]["seed"]),
                    "drop_high_blr_corr": float(by_blr[-1]["blr_corr"]),
                    "n_raw": len(vals),
                    "n_used": len(vals) - len(dropped),
                }
            )
        kept = [row for row in vals if int(row["seed"]) not in dropped]
        for row in kept:
            used_rows.append(
                {
                    "exposure_label": row["exposure_label"],
                    "exposure_s": row["exposure_s"],
                    "seed": int(row["seed"]),
                    "strategy": row["strategy"],
                    "label": row["label"],
                    "blr_corr": float(row["blr_corr"]),
                    "global_corr": float(row["global_corr"]),
                    "trimmed_by_blr_extremes": TRIM_BLR_EXTREMES,
                }
            )
        seed_values[strategy] = {
            "seed": np.asarray([int(row["seed"]) for row in kept], dtype=int),
            "blr_corr": np.asarray([float(row["blr_corr"]) for row in kept], dtype=float),
            "global_corr": np.asarray([float(row["global_corr"]) for row in kept], dtype=float),
        }
    return seed_values, used_rows, trim_report


def summarize_used_rows(used_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in used_rows:
        grouped[str(row["strategy"])].append(row)
    summary: list[dict[str, object]] = []
    for strategy, label, _color in STRATEGIES:
        rows = grouped[strategy]
        for metric in ("blr_corr", "global_corr"):
            arr = np.asarray([float(row[metric]) for row in rows], dtype=float)
            summary.append(
                {
                    "exposure_label": EXPOSURE_LABEL,
                    "strategy": strategy,
                    "label": label,
                    "metric": metric,
                    "mean": float(np.mean(arr)) if len(arr) else float("nan"),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "sem": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                    "n_seed": len(arr),
                    "trimmed_by_blr_extremes": TRIM_BLR_EXTREMES,
                }
            )
    return summary


def load_loop_rms() -> dict[tuple[str, str], list[dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in read_csv(LOOP_RMS):
        loop = row["loop"]
        strategy = row["strategy"]
        if loop not in {item[0] for item in PLOT_LOOPS}:
            continue
        if strategy not in {item[0] for item in RMS_STRATEGIES}:
            continue
        grouped[(loop, strategy)].append(
            {
                "lambda_center_nm": float(row["lambda_center_nm"]),
                "rms_rad": float(row["rms_rad"]),
            }
        )
    for values in grouped.values():
        values.sort(key=lambda item: item["lambda_center_nm"])
    return grouped


def _tight_ylim(arrays: list[np.ndarray], floor: float = 0.0, ceiling: float = 1.0) -> tuple[float, float]:
    values = np.concatenate([arr for arr in arrays if len(arr)])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = max(0.01, 0.18 * max(vmax - vmin, 0.01))
    return max(floor, vmin - pad), min(ceiling, vmax + pad)


def plot_left(ax, seed_values: dict[str, dict[str, np.ndarray]]) -> None:
    x = np.arange(len(STRATEGIES), dtype=float)
    width = 0.32
    rng = np.random.default_rng(12345)
    ax_right = ax.twinx()
    for target_ax, offset, metric, label, hatch, alpha in (
        (ax, -width / 2.0, "blr_corr", "BLR annulus", "", 0.95),
        (ax_right, width / 2.0, "global_corr", "all pixels", "//", 0.50),
    ):
        means = [float(np.mean(seed_values[strategy][metric])) for strategy, _name, _color in STRATEGIES]
        stds = [float(np.std(seed_values[strategy][metric], ddof=1)) for strategy, _name, _color in STRATEGIES]
        colors = [color for _strategy, _name, color in STRATEGIES]
        target_ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=2.5,
            color=colors,
            alpha=alpha,
            edgecolor="0.2",
            linewidth=0.55,
            hatch=hatch,
            label=label,
        )
        for idx, (strategy, _name, color) in enumerate(STRATEGIES):
            values = seed_values[strategy][metric]
            jitter = rng.normal(scale=0.018, size=len(values))
            target_ax.scatter(
                np.full(len(values), x[idx] + offset) + jitter,
                values,
                s=15,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
    ax.set_xticks(x, [label for _strategy, label, _color in STRATEGIES])
    ax.set_ylim(*_tight_ylim([seed_values[strategy]["blr_corr"] for strategy, _name, _color in STRATEGIES]))
    ax_right.set_ylim(*_tight_ylim([seed_values[strategy]["global_corr"] for strategy, _name, _color in STRATEGIES]))
    ax.set_ylabel("BLR correlation")
    ax_right.set_ylabel("all-pixel correlation")
    n_seed = len(next(iter(seed_values.values()))["blr_corr"])
    raw_n = n_seed + 2 if TRIM_BLR_EXTREMES else n_seed
    title_label = EXPOSURE_LABEL.replace("ms", " ms")
    if TRIM_BLR_EXTREMES:
        ax.set_title(f"{title_label} RML, {raw_n} seeds, trimmed to {n_seed}", pad=18)
    else:
        ax.set_title(f"{title_label} RML, {n_seed} seeds", pad=18)
    ax.grid(True, axis="y", color="0.88", linewidth=0.75)
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        borderaxespad=0.0,
    )


def plot_right(ax, grouped: dict[tuple[str, str], list[dict[str, float]]]) -> None:
    loop_span = 0.72
    all_lams = [
        item["lambda_center_nm"]
        for values in grouped.values()
        for item in values
    ]
    lam_min = min(all_lams)
    lam_max = max(all_lams)
    lam_mid = 0.5 * (lam_min + lam_max)
    for loop_idx, (loop, _kind) in enumerate(PLOT_LOOPS):
        for strategy, label, color, marker in RMS_STRATEGIES:
            values = grouped[(loop, strategy)]
            lam = np.asarray([item["lambda_center_nm"] for item in values], dtype=float)
            rms = np.asarray([item["rms_rad"] for item in values], dtype=float)
            xpos = loop_idx + loop_span * (lam - lam_mid) / max(lam_max - lam_min, 1.0)
            ax.plot(xpos, rms, color=color, lw=0.95, alpha=0.8, label=label if loop_idx == 0 else None)
            ax.scatter(xpos, rms, color=color, marker=marker, s=17, edgecolor="white", linewidth=0.3)
    for xpos in np.arange(0.5, len(PLOT_LOOPS) - 0.1, 1.0):
        ax.axvline(xpos, color="0.82", lw=0.7)
    ax.set_xticks(np.arange(len(PLOT_LOOPS)), [f"{loop}\n{kind}" for loop, kind in PLOT_LOOPS])
    ax.set_yscale("log")
    ax.set_ylabel("closure RMS (rad)")
    ax.set_title("single-sample loop RMS, 10 wavelength bins")
    ax.grid(True, axis="y", which="both", color="0.88", linewidth=0.75)
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    seed_values, used_rows, trim_report = load_seed_values()
    summary_rows = summarize_used_rows(used_rows)
    trim_tag = "trimmed" if TRIM_BLR_EXTREMES else "raw"
    write_csv(RESULTS / f"fig3a_{EXPOSURE_LABEL}_{trim_tag}_plot_seed_metrics.csv", used_rows)
    write_csv(RESULTS / f"fig3a_{EXPOSURE_LABEL}_{trim_tag}_plot_summary.csv", summary_rows)
    (RESULTS / f"fig3a_{EXPOSURE_LABEL}_{trim_tag}_plot_summary.json").write_text(
        json.dumps({"summary": summary_rows, "trim_report": trim_report}, indent=2) + "\n"
    )
    loop_rms = load_loop_rms()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 3.45),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    plot_left(axes[0], seed_values)
    plot_right(axes[1], loop_rms)
    png = FIGURES / f"fig3_previous_layout_{EXPOSURE_LABEL}.png"
    pdf = FIGURES / f"fig3_previous_layout_{EXPOSURE_LABEL}.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
