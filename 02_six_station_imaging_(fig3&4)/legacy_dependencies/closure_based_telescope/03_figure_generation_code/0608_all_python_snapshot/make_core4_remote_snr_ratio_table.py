from __future__ import annotations

import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
BUNDLE = THIS_DIR.parents[1]
CORE_DIR = THIS_DIR.parent / "core"
for _path in (CORE_DIR, THIS_DIR):
    _path_text = str(_path)
    if _path_text in sys.path:
        sys.path.remove(_path_text)
    sys.path.insert(0, _path_text)

import hawaii3_compact_case  # noqa: E402
import run_broad_plume_split_objective_rml as fig_run  # noqa: E402


OUT = BUNDLE / "exploration" / "core4_joint_remote_split"
OUT.mkdir(parents=True, exist_ok=True)

BENCHMARK_FACTOR = 2.0 / 5.0


def configure_active_benchmark_physics() -> None:
    """Use the active Fig.2 resource model inside AllClosureBenchmark.

    The upstream benchmark helper has historical defaults of 30 nights and
    600 s samples.  This table is used as a companion to the active Fig.2 run,
    so keep the Fisher normalization on the same exposure, cadence, wavelength,
    and false-positive settings as the RML simulator.
    """
    exposure_s = fig_run.BASE_EXPOSURE_S * fig_run.EXPOSURE_SCALE
    fig_run.aug.OBSERVING_DAYS = fig_run.OBSERVING_DAYS
    fig_run.aug.N_TIME_WINDOWS = fig_run.N_TIME_WINDOWS_RUN
    fig_run.aug.EXPOSURE_S = exposure_s
    fig_run.aug.EXPOSURE_GAP_S = fig_run.EXPOSURE_GAP_S_RUN
    fig_run.aug.LAMBDA_MIN_NM = fig_run.LAMBDA_MIN_NM_RUN
    fig_run.aug.LAMBDA_MAX_NM = fig_run.LAMBDA_MAX_NM_RUN
    fig_run.aug.LAMBDA_STEP_NM = fig_run.LAMBDA_STEP_NM_RUN
    fig_run.aug.FIBER_LOSS_DB_PER_KM = fig_run.closure_bm.FIBER_LOSS_DB_PER_KM
    fig_run.aug.FIBER_LENGTH_SCALE = fig_run.closure_bm.FIBER_LENGTH_SCALE
    fig_run.aug.MODE_FALSE_POSITIVE = fig_run.EPS_STATION_RUN
    fig_run.aug.PAIR_FALSE_POSITIVE = fig_run.EPS_PAIR_RUN
    fig_run.aug.BASELINE_FALSE_POSITIVE = fig_run.EPS_PAIR_RUN
    fig_run.wt.OBSERVING_DAYS = fig_run.OBSERVING_DAYS
    fig_run.wt.SNR_BOOST = 1.0


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def closure_kind(tri: tuple[int, int, int]) -> str:
    n_remote = sum(i in fig_run.core4_remote.REMOTE for i in tri)
    if n_remote == 0:
        return "core only"
    if n_remote == 1:
        return "one remote"
    if n_remote == 2:
        return "two remote"
    return "three remote"


def snr_for_vector(fisher: np.ndarray, d: np.ndarray) -> float:
    cov = np.linalg.pinv(0.5 * (fisher + fisher.T), rcond=1e-12)
    var = float(d @ cov @ d)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0
    return 1.0 / math.sqrt(var)


def independent_root_triangles(n: int) -> list[tuple[int, int, int]]:
    return [(0, i, j) for i in range(1, n) for j in range(i + 1, n)]


def make_payload() -> dict:
    fig_run.configure_good_runtime()
    fig_run.apply_sample_stress_runtime()
    case = fig_run.scale_remote_coordinates(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    old_configure = fig_run.closure_bm.configure_physics
    old_loader = fig_run.closure_bm.rml_cases.load_maunakea_plus3_case
    fig_run.closure_bm.configure_physics = configure_active_benchmark_physics
    fig_run.closure_bm.rml_cases.load_maunakea_plus3_case = lambda: case
    try:
        with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT):
            bm = fig_run.closure_bm.AllClosureBenchmark()
            loop_specs = fig_run.core4_remote.optimize_root_loop_splits(bm)
            fisher_approx = fig_run.core4_remote.fisher_for_loop_specs(bm, loop_specs)
    finally:
        fig_run.closure_bm.configure_physics = old_configure
        fig_run.closure_bm.rml_cases.load_maunakea_plus3_case = old_loader

    fisher_direct_benchmark = BENCHMARK_FACTOR * bm.direct_raw
    metrics_approx = fig_run.closure_bm.stable_metrics(fisher_approx)
    metrics_benchmark = fig_run.closure_bm.stable_metrics(fisher_direct_benchmark)

    rows = []
    for tri in independent_root_triangles(bm.n):
        d = fig_run.core4_remote.measurement_vector(bm, tri)
        snr_approx = snr_for_vector(fisher_approx, d)
        snr_benchmark = snr_for_vector(fisher_direct_benchmark, d)
        rows.append(
            {
                "closure": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
                "stations": " / ".join(bm.names[i] for i in tri),
                "type": closure_kind(tri),
                "snr_core4_remote": snr_approx,
                "snr_direct_2over5": snr_benchmark,
                "snr_ratio": snr_approx / snr_benchmark if snr_benchmark > 0.0 else math.inf,
            }
        )

    close_factor, remote_loop_factor = fig_run.core4_remote.equal_loop_budget_factors(loop_specs)
    split_rows = []
    for spec in loop_specs:
        tri = tuple(spec["tri"])
        if all(station in fig_run.core4_remote.CORE for station in tri):
            continue
        directed = fig_run.core4_remote.noncore_directed_fractions(
            tri,
            tuple(float(value) for value in spec["split"]),
        )
        for (station, target), fraction in sorted(directed.items()):
            split_rows.append(
                {
                    "closure": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
                    "loop_budget_fraction": float(remote_loop_factor),
                    "from": f"S{station + 1}",
                    "station": bm.names[station],
                    "to": f"S{target + 1}",
                    "target": bm.names[target],
                    "loop_internal_fraction": float(fraction),
                    "effective_total_fraction": float(remote_loop_factor * fraction),
                    "station_budget_inside_loop": float(fig_run.core4_remote.station_loop_budget(station)),
                }
            )

    return {
        "case": bm.case.key,
        "station_names": bm.names,
        "run_tag": fig_run.RUN_TAG,
        "observing_days": int(fig_run.OBSERVING_DAYS),
        "samples_per_night": int(fig_run.N_TIME_WINDOWS_RUN),
        "exposure_s": float(fig_run.BASE_EXPOSURE_S * fig_run.EXPOSURE_SCALE),
        "sample_cadence_s": float(fig_run.SAMPLE_CADENCE_S_RUN),
        "lambda_min_nm": float(fig_run.LAMBDA_MIN_NM_RUN),
        "lambda_max_nm": float(fig_run.LAMBDA_MAX_NM_RUN),
        "lambda_step_nm": float(fig_run.LAMBDA_STEP_NM_RUN),
        "benchmark_factor": BENCHMARK_FACTOR,
        "benchmark_factor_note": "2/5 = 6/15 for N=7, C=15 independent closures.",
        "remote_x_scale": fig_run.REMOTE_X_SCALE,
        "remote_y_scale": fig_run.REMOTE_Y_SCALE,
        "eps_station": fig_run.EPS_STATION_RUN,
        "eps_pair": fig_run.EPS_PAIR_RUN,
        "eps_direct_extra": fig_run.EPS_DIRECT_EXTRA_RUN,
        "fiber_length_scale": fig_run.closure_bm.FIBER_LENGTH_SCALE,
        "fiber_loss_db_per_km": fig_run.closure_bm.FIBER_LOSS_DB_PER_KM,
        "core_joint_fraction": fig_run.core4_remote.CORE_JOINT_FRACTION,
        "core_remote_fraction": fig_run.core4_remote.CORE_REMOTE_FRACTION,
        "remote_total_fraction": fig_run.core4_remote.REMOTE_TOTAL_FRACTION,
        "optimization_info": {
            "strategy": "equal root-loop photon budget; close4 closure for close-only loops; loop-internal optimized remote-edge readout for remote-involved loops",
            "n_root_loops": len(loop_specs),
            "close4_budget_fraction": float(close_factor),
            "remote_loop_budget_fraction": float(remote_loop_factor),
        },
        "matrix_metrics": {
            "core4_remote": metrics_approx,
            "direct_2over5": metrics_benchmark,
        },
        "independent_closure_rows": rows,
        "equal_loop_split_rows": split_rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(payload: dict, tex_path: Path) -> None:
    rows = payload["independent_closure_rows"]
    metric = payload["matrix_metrics"]
    table_lines = []
    for row in rows:
        table_lines.append(
            f"{latex_escape(row['closure'])} & {latex_escape(row['type'])} & "
            f"{row['snr_core4_remote']:.3g} & {row['snr_direct_2over5']:.3g} & "
            f"{row['snr_ratio']:.3f} \\\\"
        )
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.55in,landscape]{{geometry}}
\usepackage{{booktabs,amsmath,array,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{Independent-closure SNR comparison for the core4+remote strategy}}
\author{{Codex diagnostic table}}
\date{{\today}}
\begin{{document}}
\maketitle

\noindent
This table compares the active approximate implementation, equal root-loop
budget with a shared close-four direct closure block for close-only loops and
loop-internal optimized remote-related edge readout for remote-involved loops,
against the
benchmark requested in the discussion: the unsplit direct closure Fisher
multiplied by the optimal polling factor \(2/5=6/15\).  The benchmark is not
used as the image data model; it is only the scalar Fisher reference for the
same 15 root-cycle independent closure coordinates.  Both Fisher matrices are
scaled to the active Fig.~2 setting: {payload["observing_days"]} observing
nights, {payload["samples_per_night"]} samples per night, {payload["exposure_s"]:.3g} s
per sample, {payload["lambda_min_nm"]:.0f}--{payload["lambda_max_nm"]:.0f} nm split into
{payload["lambda_step_nm"]:.0f} nm bins, remote \(x\) scale {payload["remote_x_scale"]:.3g}, remote \(y\) scale
{payload["remote_y_scale"]:.3g}, \(\epsilon_i={payload["eps_station"]:.3g}\),
\(\epsilon_{{ij}}=\epsilon_{{\rm dir}}={payload["eps_pair"]:.3g}\), and
fibre attenuation {payload["fiber_loss_db_per_km"]:.3g} dB/km.

\vspace{{0.6em}}
\noindent
Matrix mean RMS: core4+remote \(={metric["core4_remote"]["mean_coord_rms"]:.3g}\)
rad, direct \(2/5\) benchmark \(={metric["direct_2over5"]["mean_coord_rms"]:.3g}\)
rad.  Thus the mean-coordinate RMS ratio is
\({metric["direct_2over5"]["mean_coord_rms"] / metric["core4_remote"]["mean_coord_rms"]:.3f}\);
values above unity favor the approximate implementation in this RMS metric.

\begin{{center}}
\begin{{tabular}}{{llrrr}}
\toprule
closure & class & SNR core4+remote & SNR direct \(2/5\) & ratio \\
\midrule
{chr(10).join(table_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\vspace{{0.4em}}
\noindent
The root-cycle basis uses station S1 as the root, so two-remote rows include
remote-remote baselines.  Each of the 15 root loops receives the same photon
budget.  The three close-only root loops are supplied by the shared close4
direct block, giving that block a total budget fraction
{payload["optimization_info"]["close4_budget_fraction"]:.3g}.  Each
remote-involved loop has budget fraction
{payload["optimization_info"]["remote_loop_budget_fraction"]:.3g} and then
optimizes only its internal remote-related edge split.  A close-close baseline
appearing in such a loop is supplied by close4 closure information plus
station-gauge nuisance before marginalization.  The corresponding optimized
per-loop station fractions are written to the companion CSV file.

\end{{document}}
"""
    tex_path.write_text(tex)


def main() -> None:
    payload = make_payload()
    stem = "core4_remote_vs_direct_2over5_independent_closure_snr"
    csv_path = OUT / f"{stem}.csv"
    split_csv_path = OUT / f"{stem}_station_split.csv"
    json_path = OUT / f"{stem}.json"
    tex_path = OUT / f"{stem}.tex"
    pdf_path = OUT / f"{stem}.pdf"
    write_csv(csv_path, payload["independent_closure_rows"])
    write_csv(split_csv_path, payload["equal_loop_split_rows"])
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_tex(payload, tex_path)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUT, check=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUT, check=True)
    print(pdf_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
