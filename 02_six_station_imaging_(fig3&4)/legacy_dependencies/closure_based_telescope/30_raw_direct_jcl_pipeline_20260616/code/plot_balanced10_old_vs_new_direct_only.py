from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

OLD_LOOP_GAINS = (
    WORKSPACE
    / "18_balanced_10loop_independent_set_20260611"
    / "results"
    / "remote_star_joint_loop_gains.csv"
)
NEW_LOOP_GAINS = RESULTS / "balanced10_generalized_station_budget_rawje_loop_gains.csv"
STEM = "balanced10_old_vs_new_direct_only"


def gain_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    old_rows = list(csv.DictReader(OLD_LOOP_GAINS.open()))
    new_rows = list(csv.DictReader(NEW_LOOP_GAINS.open()))
    new_by_loop = {row["loop"]: row for row in new_rows}

    rows: list[dict[str, object]] = []
    for old in old_rows:
        loop = old["loop"]
        new = new_by_loop[loop]
        old_gain = float(old["snr_gain_direct_optimized_schedule_vs_edge"])
        new_gain = float(new["gain_corrected_raw_generalized_opt_vs_edge"])
        rows.append(
            {
                "loop": loop,
                "old_3port_direct_gain_vs_edge": old_gain,
                "new_generalized_3port_direct_gain_vs_edge": new_gain,
                "new_over_old_snr": new_gain / max(old_gain, 1.0e-300),
            }
        )

    csv_path = RESULTS / f"{STEM}_loop_gains.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [str(row["loop"]) for row in rows]
    old_values = np.asarray([float(row["old_3port_direct_gain_vs_edge"]) for row in rows], dtype=float)
    new_values = np.asarray([float(row["new_generalized_3port_direct_gain_vs_edge"]) for row in rows], dtype=float)
    n_station = 6
    three_port_limit = math.sqrt(n_station / (n_station - 2.0))
    global_n_limit = math.sqrt(n_station / 2.0)

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.0, 4.1), constrained_layout=True)
    ax.bar(
        x - 0.5 * width,
        old_values,
        width,
        color="#3a3a3a",
        label="old 3-port direct",
    )
    ax.bar(
        x + 0.5 * width,
        new_values,
        width,
        color="#2a9d8f",
        label=r"new 3-port direct, optimized $a_{\ell i}$",
    )
    ax.axhline(1.0, color="0.28", lw=1.0, ls="--")
    ax.axhline(
        three_port_limit,
        color="#6a4c93",
        lw=1.2,
        ls=":",
        label=r"$\sqrt{N/(N-2)}$",
    )
    ax.axhline(
        global_n_limit,
        color="#005f73",
        lw=1.2,
        ls="-.",
        label=r"$\sqrt{N/2}$",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.set_ylim(0.95, 1.08 * max(float(np.max(new_values)), global_n_limit))
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.20))

    pdf_path = FIGURES / f"{STEM}.pdf"
    png_path = FIGURES / f"{STEM}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=280)
    plt.close(fig)

    payload = {
        "definition": {
            "snr_gain": "sigma_edge_uniform / sigma_direct for each balanced-10 loop",
            "old_3port_direct": "direct_optimized_schedule from 18_balanced_10loop_independent_set_20260611",
            "new_3port_direct": "corrected raw Je generalized station-budget direct benchmark with optimized a_{loop,station}",
            "loop_set": labels,
        },
        "summary": {
            "old_3port_direct_gain_vs_edge": gain_summary(old_values),
            "new_generalized_3port_direct_gain_vs_edge": gain_summary(new_values),
            "new_over_old_snr": gain_summary(new_values / np.maximum(old_values, 1.0e-300)),
        },
        "outputs": {
            "loop_gains_csv": str(csv_path),
            "figure_pdf": str(pdf_path),
            "figure_png": str(png_path),
        },
    }
    summary_path = RESULTS / f"{STEM}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(csv_path)
    print(summary_path)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
