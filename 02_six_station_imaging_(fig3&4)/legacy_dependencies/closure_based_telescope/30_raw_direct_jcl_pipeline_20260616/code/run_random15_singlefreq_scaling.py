from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for folder in (RESULTS, FIGURES, LOGS, LOGS / "mplconfig"):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOGS / "mplconfig"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(THIS_DIR))

from random_singlefreq_scaling_helpers import (  # noqa: E402
    DEFAULT_DIAMETER_M,
    DEFAULT_MIN_SEPARATION_KM,
    DEFAULT_RADIUS_KM,
    DEFAULT_WAVELENGTH_NM,
    edge_uniform_fisher,
    global_raw_qfi_fisher,
    make_random_case,
    make_single_frequency_benchmark,
    physical_alltriangle_direct_fisher,
    ratio_summary,
    root_loop_gain_rows,
)


TAG = "random15_singlefreq_current_source_scaling"
N_STATION = 15
WAVELENGTH_NM = DEFAULT_WAVELENGTH_NM
SOURCE_SCALE = 1.0
SEEDS = [20260616, 20260617, 20260618, 20260619, 20260620]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(
    representative_case,
    root_rows: list[dict[str, object]],
    *,
    theory_scalar: float,
    theory_raw: float,
    summary: dict[str, object],
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(10.0, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.65])
    ax_layout = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1])

    stations = np.asarray([[tel.x_km, tel.y_km] for tel in representative_case.telescopes], dtype=float)
    hub = np.asarray(representative_case.hub_km, dtype=float)
    ax_layout.scatter(stations[:, 0], stations[:, 1], s=36, color="#005f73")
    ax_layout.scatter([hub[0]], [hub[1]], s=70, marker="*", color="#ca6702", zorder=4)
    for idx, (x, y) in enumerate(stations):
        ax_layout.text(x, y, f"S{idx + 1}", fontsize=6.5, ha="center", va="center", color="white", weight="bold")
    ax_layout.set_aspect("equal", adjustable="box")
    ax_layout.set_title(f"Representative random layout\nseed={summary['representative_seed']}")
    ax_layout.set_xlabel("x east (km)")
    ax_layout.set_ylabel("y north (km)")
    ax_layout.grid(True, color="0.90", lw=0.6)

    direct = np.asarray([float(row["snr_gain_direct_alltriangle_vs_edge"]) for row in root_rows], dtype=float)
    raw = np.asarray([float(row["snr_gain_global_raw_qfi_vs_edge"]) for row in root_rows], dtype=float)
    order = np.argsort(raw)
    x = np.arange(1, direct.size + 1)
    ax_gain.plot(x, direct[order], color="#0a9396", lw=1.8, label="3-mode direct (all triangles)")
    ax_gain.plot(x, raw[order], color="#ae2012", lw=1.8, label="global raw QFI upper bound")
    ax_gain.axhline(theory_scalar, color="#0a9396", lw=1.0, ls="--", alpha=0.70, label=r"$\sqrt{N/(N-2)}$")
    ax_gain.axhline(theory_raw, color="#ae2012", lw=1.0, ls="--", alpha=0.70, label=r"$\sqrt{N/2}$")
    ax_gain.set_xlabel("root-loop rank (sorted by raw-QFI gain)")
    ax_gain.set_ylabel("SNR gain vs uniform edge")
    ax_gain.set_title(
        f"N={summary['n_station']}, single-frequency current-source surrogate\n"
        f"{summary['wavelength_nm']:.1f} nm, equal {summary['diameter_m']:.1f} m stations"
    )
    ax_gain.grid(True, color="0.90", lw=0.6)
    ax_gain.legend(fontsize=8, loc="best")

    png = FIGURES / f"{TAG}.png"
    pdf = FIGURES / f"{TAG}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    representative_seed = SEEDS[0]
    representative_case = None
    representative_rows = None
    representative_summary = None
    representative_equiv_bandwidth = None
    seed_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        print(f"[seed {seed}] building random case", flush=True)
        case = make_random_case(
            N_STATION,
            seed,
            radius_km=DEFAULT_RADIUS_KM,
            min_separation_km=DEFAULT_MIN_SEPARATION_KM,
            diameter_m=DEFAULT_DIAMETER_M,
        )
        print(f"[seed {seed}] simulating single-frequency benchmark", flush=True)
        bm = make_single_frequency_benchmark(case, wavelength_nm=WAVELENGTH_NM, source_scale=SOURCE_SCALE)
        print(f"[seed {seed}] assembling edge/direct/raw Fisher matrices", flush=True)
        edge_fisher_q = edge_uniform_fisher(bm)
        direct_fisher_q = physical_alltriangle_direct_fisher(bm)
        raw_fisher_q = global_raw_qfi_fisher(bm)
        rows = root_loop_gain_rows(bm, edge_fisher_q, direct_fisher_q, raw_fisher_q)

        direct_gains = np.asarray([float(row["snr_gain_direct_alltriangle_vs_edge"]) for row in rows], dtype=float)
        raw_gains = np.asarray([float(row["snr_gain_global_raw_qfi_vs_edge"]) for row in rows], dtype=float)
        raw_scheduled_proxy_gains = math.sqrt(max(bm.rank_share, 0.0)) * raw_gains
        seed_row = {
            "seed": int(seed),
            "n_station": int(bm.n),
            "n_root_loops": int(len(rows)),
            "rank_share": float(bm.rank_share),
            "equivalent_bandwidth_hz": float(bm.equivalent_bandwidth_hz),
            "direct_gain_min": float(np.min(direct_gains)),
            "direct_gain_mean": float(np.mean(direct_gains)),
            "direct_gain_median": float(np.median(direct_gains)),
            "direct_gain_max": float(np.max(direct_gains)),
            "raw_gain_min": float(np.min(raw_gains)),
            "raw_gain_mean": float(np.mean(raw_gains)),
            "raw_gain_median": float(np.median(raw_gains)),
            "raw_gain_max": float(np.max(raw_gains)),
            "raw_scheduled_proxy_gain_min": float(np.min(raw_scheduled_proxy_gains)),
            "raw_scheduled_proxy_gain_mean": float(np.mean(raw_scheduled_proxy_gains)),
            "raw_scheduled_proxy_gain_median": float(np.median(raw_scheduled_proxy_gains)),
            "raw_scheduled_proxy_gain_max": float(np.max(raw_scheduled_proxy_gains)),
        }
        seed_rows.append(seed_row)

        if seed == representative_seed:
            representative_case = case
            representative_rows = rows
            representative_summary = {
                "direct_gain_vs_edge": ratio_summary(direct_gains),
                "raw_gain_vs_edge": ratio_summary(raw_gains),
                "raw_scheduled_proxy_gain_vs_edge": ratio_summary(raw_scheduled_proxy_gains),
            }
            representative_equiv_bandwidth = float(bm.equivalent_bandwidth_hz)

    assert representative_case is not None
    assert representative_rows is not None
    assert representative_summary is not None
    assert representative_equiv_bandwidth is not None

    direct_medians = np.asarray([float(row["direct_gain_median"]) for row in seed_rows], dtype=float)
    raw_medians = np.asarray([float(row["raw_gain_median"]) for row in seed_rows], dtype=float)
    proxy_medians = np.asarray([float(row["raw_scheduled_proxy_gain_median"]) for row in seed_rows], dtype=float)

    summary = {
        "tag": TAG,
        "n_station": N_STATION,
        "n_root_loops": int((N_STATION - 1) * (N_STATION - 2) // 2),
        "wavelength_nm": float(WAVELENGTH_NM),
        "source_scale": float(SOURCE_SCALE),
        "diameter_m": float(DEFAULT_DIAMETER_M),
        "radius_km": float(DEFAULT_RADIUS_KM),
        "min_separation_km": float(DEFAULT_MIN_SEPARATION_KM),
        "seeds": [int(seed) for seed in SEEDS],
        "representative_seed": int(representative_seed),
        "representative_equivalent_bandwidth_hz": float(representative_equiv_bandwidth),
        "theory_lines": {
            "three_mode_direct_limit_sqrt_N_over_Nminus2": float(math.sqrt(N_STATION / (N_STATION - 2.0))),
            "global_raw_qfi_limit_sqrt_N_over_2": float(math.sqrt(N_STATION / 2.0)),
        },
        "representative_summary": representative_summary,
        "multi_seed_summary": {
            "direct_gain_median_over_seeds": ratio_summary(direct_medians),
            "raw_gain_median_over_seeds": ratio_summary(raw_medians),
            "raw_scheduled_proxy_gain_median_over_seeds": ratio_summary(proxy_medians),
        },
    }

    seed_csv = RESULTS / f"{TAG}_seed_summaries.csv"
    root_csv = RESULTS / f"{TAG}_representative_root_loop_gains.csv"
    summary_json = RESULTS / f"{TAG}_summary.json"

    write_csv(seed_csv, seed_rows)
    write_csv(root_csv, representative_rows)
    pdf, png = plot_outputs(
        representative_case,
        representative_rows,
        theory_scalar=summary["theory_lines"]["three_mode_direct_limit_sqrt_N_over_Nminus2"],
        theory_raw=summary["theory_lines"]["global_raw_qfi_limit_sqrt_N_over_2"],
        summary=summary,
    )
    summary["figure_pdf"] = str(pdf)
    summary["figure_png"] = str(png)
    summary["seed_csv"] = str(seed_csv)
    summary["representative_root_loop_csv"] = str(root_csv)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
