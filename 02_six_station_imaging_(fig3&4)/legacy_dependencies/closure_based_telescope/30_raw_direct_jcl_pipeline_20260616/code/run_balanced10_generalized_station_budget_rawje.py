from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import generalized_loop_station_budget_helpers as gh
import rawje_balanced10_helpers as h


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

STEM = "balanced10_generalized_station_budget_rawje"


def make_rows(
    case: gh.PreparedBalanced10,
    legacy_sigma: np.ndarray,
    legacy_gains: np.ndarray,
    corrected_symmetric_sigma: np.ndarray,
    corrected_symmetric_gains: np.ndarray,
    optimized_sigma: np.ndarray,
    optimized_gains: np.ndarray,
    old_local_opt_gains: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, tri in enumerate(case.triangles):
        label = h.loop_label(tri)
        old_local_gain = float(old_local_opt_gains[label])
        rows.append(
            {
                "loop": label,
                "loop_class": gh.loop_class(tri),
                "rms_edge_uniform_rad": float(case.edge_sigma[idx]),
                "rms_legacy_raw_linear_omega02_rad": float(legacy_sigma[idx]),
                "rms_corrected_raw_symmetric_omega02_rad": float(corrected_symmetric_sigma[idx]),
                "rms_corrected_raw_generalized_opt_rad": float(optimized_sigma[idx]),
                "gain_legacy_raw_linear_omega02_vs_edge": float(legacy_gains[idx]),
                "gain_corrected_raw_symmetric_omega02_vs_edge": float(corrected_symmetric_gains[idx]),
                "gain_corrected_raw_generalized_opt_vs_edge": float(optimized_gains[idx]),
                "gain_old_local_schur_opt_vs_edge": old_local_gain,
                "gain_ratio_corrected_symmetric_over_legacy": float(
                    corrected_symmetric_gains[idx] / max(legacy_gains[idx], 1.0e-300)
                ),
                "gain_ratio_generalized_over_corrected_symmetric": float(
                    optimized_gains[idx] / max(corrected_symmetric_gains[idx], 1.0e-300)
                ),
                "gain_ratio_generalized_over_old_local_opt": float(
                    optimized_gains[idx] / max(old_local_gain, 1.0e-300)
                ),
            }
        )
    return rows


def plot_rows(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    labels = [str(row["loop"]) for row in rows]
    legacy = np.asarray([float(row["gain_legacy_raw_linear_omega02_vs_edge"]) for row in rows], dtype=float)
    corrected = np.asarray([float(row["gain_corrected_raw_symmetric_omega02_vs_edge"]) for row in rows], dtype=float)
    optimized = np.asarray([float(row["gain_corrected_raw_generalized_opt_vs_edge"]) for row in rows], dtype=float)
    old_local = np.asarray([float(row["gain_old_local_schur_opt_vs_edge"]) for row in rows], dtype=float)

    x = np.arange(len(rows))
    width = 0.24
    n_station = 6
    three_port_limit = math.sqrt(n_station / (n_station - 2.0))
    global_n_limit = math.sqrt(n_station / 2.0)

    fig, ax = plt.subplots(figsize=(9.7, 4.2), constrained_layout=True)
    ax.bar(
        x - width,
        legacy,
        width,
        color="#9d0208",
        label=r"legacy raw $J_e$, linear $\omega_\ell=0.2$",
    )
    ax.bar(
        x,
        corrected,
        width,
        color="#f4a261",
        label=r"corrected raw $J_e$, symmetric $a_{\ell i}=0.2$",
    )
    ax.bar(
        x + width,
        optimized,
        width,
        color="#2a9d8f",
        label=r"corrected raw $J_e$, optimized $a_{\ell i}$",
    )
    ax.plot(
        x,
        old_local,
        color="0.15",
        lw=1.1,
        marker="o",
        ms=4.0,
        label="old local-Schur optimal",
    )
    ax.axhline(1.0, color="0.28", lw=1.0, ls="--")
    ax.axhline(
        three_port_limit,
        color="#6a4c93",
        lw=1.15,
        ls=":",
        label=r"$\sqrt{N/(N-2)}$",
    )
    ax.axhline(
        global_n_limit,
        color="#005f73",
        lw=1.15,
        ls="-.",
        label=r"$\sqrt{N/2}$",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ymax = max(
        1.42,
        1.08
        * float(
            max(
                np.max(legacy),
                np.max(corrected),
                np.max(optimized),
                np.max(old_local),
                three_port_limit,
                global_n_limit,
            )
        ),
    )
    ax.set_ylim(0.80, ymax)
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24))

    pdf_path = FIGURES / f"{STEM}.pdf"
    png_path = FIGURES / f"{STEM}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=280)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    case = gh.prepare_balanced10_case()

    legacy_sigma, legacy_gains = gh.legacy_raw_gain_rows(case)
    corrected_symmetric_sigma, corrected_symmetric_gains = gh.corrected_symmetric_gain_rows(case)
    optimized_schedule, optimized_sigma, optimized_info = gh.optimize_generalized_station_schedule(case)
    optimized_fisher = gh.corrected_raw_fisher_from_station_schedule(case, optimized_schedule)
    optimized_sigma_check, optimized_gains = gh.gains_from_fisher(case, optimized_fisher)
    if float(np.max(np.abs(optimized_sigma_check - optimized_sigma))) > 1.0e-10:
        raise RuntimeError("optimized sigma mismatch between cached result and recomputed Fisher")

    old_local_opt_gains = gh.load_old_local_opt_gains()
    rows = make_rows(
        case,
        legacy_sigma,
        legacy_gains,
        corrected_symmetric_sigma,
        corrected_symmetric_gains,
        optimized_sigma,
        optimized_gains,
        old_local_opt_gains,
    )
    csv_path = RESULTS / f"{STEM}_loop_gains.csv"
    gh.write_csv(csv_path, rows)
    pdf_path, png_path = plot_rows(rows)

    schedule_json = RESULTS / f"{STEM}_optimized_station_schedule.json"
    gh.dump_station_schedule_json(schedule_json, case, optimized_schedule, optimized_info)

    summary = {
        "definition": {
            "loop_set": [h.loop_label(tri) for tri in case.triangles],
            "legacy_raw_linear_model": "Old shortcut: build unit-budget raw triangle edge Fisher with equal station fractions and multiply the integrated branch by omega_l=0.2 before the final global Schur complement.",
            "corrected_raw_symmetric_model": "Correct physical raw Je schedule with a_{loop,i}=0.2 for the three stations in each balanced-10 loop and zero otherwise, then one final global Schur complement.",
            "corrected_raw_generalized_model": "Correct physical raw Je schedule with independently optimized a_{loop,i} on each incident loop of each station, subject to one unit of total station budget per station.",
            "old_local_opt_reference": "Loaded from 18_balanced_10loop_independent_set_20260611/results/remote_star_joint_loop_gains.csv.",
            "snr_gain_definition": "sigma_edge_uniform / sigma_strategy for each balanced-10 loop",
            "n_samples": int(len(case.total_modes)),
            "exposure_s": float(h.aug.EXPOSURE_S),
        },
        "optimized_schedule_info": optimized_info,
        "summary": {
            "legacy_raw_linear_gain_vs_edge": gh.ratio_summary(legacy_gains),
            "corrected_raw_symmetric_gain_vs_edge": gh.ratio_summary(corrected_symmetric_gains),
            "corrected_raw_generalized_gain_vs_edge": gh.ratio_summary(optimized_gains),
            "old_local_schur_opt_gain_vs_edge": gh.ratio_summary(np.asarray(list(old_local_opt_gains.values()), dtype=float)),
            "corrected_symmetric_over_legacy": gh.ratio_summary(
                corrected_symmetric_gains / np.maximum(legacy_gains, 1.0e-300)
            ),
            "generalized_over_corrected_symmetric": gh.ratio_summary(
                optimized_gains / np.maximum(corrected_symmetric_gains, 1.0e-300)
            ),
            "generalized_over_old_local_opt": gh.ratio_summary(
                optimized_gains
                / np.maximum(np.asarray([old_local_opt_gains[h.loop_label(tri)] for tri in case.triangles], dtype=float), 1.0e-300)
            ),
        },
        "outputs": {
            "loop_gains_csv": str(csv_path),
            "optimized_station_schedule_json": str(schedule_json),
            "figure_pdf": str(pdf_path),
            "figure_png": str(png_path),
        },
    }
    summary_path = RESULTS / f"{STEM}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary["summary"], indent=2))
    print(csv_path)
    print(schedule_json)
    print(summary_path)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
