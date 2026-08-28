from __future__ import annotations

import csv
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
import plot_prl_broadband_clean as base  # noqa: E402
import run_broad_plume_split_objective_rml as fig_run  # noqa: E402
import test_fig3_split_objective_imaging as split_sim  # noqa: E402


OUT = BUNDLE / "exploration" / "fig2_seed_scan"
SCAN_JSON = OUT / "fig2_seed_scan_fit40_i2600.json"
REPORT_STEM = "fig2_seed_scan_fit40_i2600_summary"

STRATEGY_KEYS = [
    ("all", "all-vis"),
    ("edge_uniform", "edge"),
    ("core4_remote_optimized", "core4"),
    ("nmode_joint_scheduled", "direct 6/15"),
]


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt(value: float, ndigit: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{ndigit}f}"


def closure_edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_idx[(a, b)]] = 1.0
    out[edge_to_idx[(b, c)]] = 1.0
    out[edge_to_idx[(a, c)]] = -1.0
    return out


def root_triangles(n_station: int) -> list[tuple[int, int, int]]:
    return [(0, i, j) for i in range(1, n_station) for j in range(i + 1, n_station)]


def circular_rms(values: list[np.ndarray]) -> float:
    if not values:
        return math.nan
    residual = np.concatenate(values)
    residual = np.angle(np.exp(1j * residual))
    return float(np.sqrt(np.mean(residual * residual)))


def load_seed_rows() -> list[dict]:
    rows = json.loads(SCAN_JSON.read_text())
    return [row for row in rows if row.get("status") == "ok"]


def simulate_loop_rms_for_seed(seed: int, case, splits: dict[str, np.ndarray]) -> list[dict[str, float | str | int]]:
    split_sim.RNG_SEED = int(seed)
    split_sim.configure()
    fig_run.apply_sample_stress_runtime()
    with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.ngc.NGC4151):
        bands, _stats, _truth, _axis_uas = split_sim.simulate_bands_with_strategies(case, splits)
        stations, _diameters, names, _is_added = fig_run.aug.station_table_from_case(case)
        edges = base.edge_list(len(stations))
        loop_vectors = {tri: closure_edge_vector(edges, tri) for tri in root_triangles(len(stations))}
        values = {
            (tri, key): []
            for tri in loop_vectors
            for key, _label in STRATEGY_KEYS
        }
        fov_rad = 2.0 * fig_run.aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
        lam_edges_nm = np.arange(
            fig_run.aug.LAMBDA_MIN_NM,
            fig_run.aug.LAMBDA_MAX_NM + 0.5 * fig_run.aug.LAMBDA_STEP_NM,
            fig_run.aug.LAMBDA_STEP_NM,
        )
        lam_edges_nm[-1] = fig_run.aug.LAMBDA_MAX_NM
        wavelength_source = getattr(base, "make_source_at_wavelength_nm")
        for band, lo_nm, hi_nm in zip(bands, lam_edges_nm[:-1], lam_edges_nm[1:]):
            center_nm = float(math.sqrt(lo_nm * hi_nm))
            image, _ = wavelength_source(fig_run.aug.N_PIX, fig_run.aug.HALF_WIDTH_UAS, center_nm)
            vgrid, uv_axis = base.visibility_grid(image, fov_rad)
            vtrue = base.interp_vis(vgrid, uv_axis, band["u"], band["v"])
            true_phase = np.angle(vtrue).reshape(-1, len(edges))
            for key, _label in STRATEGY_KEYS:
                observed_phase = np.angle(band[f"vis_{key}"]).reshape(-1, len(edges))
                delta_phase = observed_phase - true_phase
                for tri, vec in loop_vectors.items():
                    values[(tri, key)].append(delta_phase @ vec)

    rows: list[dict[str, float | str | int]] = []
    for tri in root_triangles(len(stations)):
        row: dict[str, float | str | int] = {
            "seed": int(seed),
            "loop": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
            "stations": " / ".join(names[idx] for idx in tri),
        }
        for key, _label in STRATEGY_KEYS:
            row[f"rms_{key}_rad"] = circular_rms(values[(tri, key)])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(seed_rows: list[dict], loop_rows: list[dict], tex_path: Path) -> None:
    best = max(
        seed_rows,
        key=lambda row: (
            float(row["core4_remote_optimized_blr_corr"]),
            float(row["core4_remote_optimized_global_corr"]),
        ),
    )
    metric_lines = []
    for row in seed_rows:
        marker = r"\textbf{yes}" if int(row["seed"]) == int(best["seed"]) else ""
        metric_lines.append(
            f"{row['seed']} & {marker} & "
            f"{fmt(row['all_global_corr'])}/{fmt(row['all_blr_corr'])} & "
            f"{fmt(row['edge_uniform_global_corr'])}/{fmt(row['edge_uniform_blr_corr'])} & "
            f"{fmt(row['core4_remote_optimized_global_corr'])}/{fmt(row['core4_remote_optimized_blr_corr'])} & "
            f"{fmt(row['core4_remote_optimized_amp_chi2'])}/{fmt(row['core4_remote_optimized_phase_chi2'])} \\\\"
        )

    loop_lines = []
    for row in loop_rows:
        loop_lines.append(
            f"{row['seed']} & {latex_escape(row['loop'])} & "
            f"{fmt(row['rms_all_rad'])} & "
            f"{fmt(row['rms_edge_uniform_rad'])} & "
            f"{fmt(row['rms_core4_remote_optimized_rad'])} & "
            f"{fmt(row['rms_nmode_joint_scheduled_rad'])} \\\\"
        )

    summary_by_loop = []
    loops = sorted({row["loop"] for row in loop_rows})
    for loop in loops:
        vals = [row for row in loop_rows if row["loop"] == loop]
        summary_by_loop.append(
            f"{latex_escape(loop)} & "
            f"{fmt(float(np.mean([row['rms_all_rad'] for row in vals])))} & "
            f"{fmt(float(np.mean([row['rms_edge_uniform_rad'] for row in vals])))} & "
            f"{fmt(float(np.mean([row['rms_core4_remote_optimized_rad'] for row in vals])))} & "
            f"{fmt(float(np.mean([row['rms_nmode_joint_scheduled_rad'] for row in vals])))} \\\\"
        )

    tex = rf"""\documentclass[9pt]{{article}}
\usepackage[margin=0.45in,landscape]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,array,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{Fig. 2 Seed Scan: Image Metrics and Per-Loop Closure-Phase Noise}}
\date{{\today}}
\begin{{document}}
\maketitle

\noindent
All runs use the component-resolved chromatic NGC~4151 source, five observing nights,
36 ten-minute samples per night, remote \(y\)-scale 0.85, \(RML\_FIT\_N\_PIX=40\),
Adam 2600 steps, and learning rate 0.010.  The selected seed is the one with the
largest core4+remote BLR-annulus image correlation, using the global correlation as
a tie-breaker.  Loop RMS values are circular RMS phase residuals, in radians,
relative to the true closure phase for the same chromatic source realization.

\begin{{center}}
\begin{{tabular}}{{rcllll}}
\toprule
seed & selected & all-vis \(r_{{\rm all}}/r_{{\rm BLR}}\) &
edge \(r_{{\rm all}}/r_{{\rm BLR}}\) &
core4 \(r_{{\rm all}}/r_{{\rm BLR}}\) &
core4 \(\chi_A^2/\chi_\phi^2\) \\
\midrule
{chr(10).join(metric_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Mean Loop RMS Over Seeds}}
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
loop & all-vis & edge & core4 & direct \(6/15\) \\
\midrule
{chr(10).join(summary_by_loop)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Per-Seed Loop RMS}}
\scriptsize
\begin{{longtable}}{{rlrrrr}}
\toprule
seed & loop & all-vis & edge & core4 & direct \(6/15\) \\
\midrule
\endfirsthead
\toprule
seed & loop & all-vis & edge & core4 & direct \(6/15\) \\
\midrule
\endhead
{chr(10).join(loop_lines)}
\bottomrule
\end{{longtable}}
\end{{document}}
"""
    tex_path.write_text(tex)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_rows = load_seed_rows()
    fig_run.configure_good_runtime()
    case = fig_run.scale_remote_coordinates(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    splits = fig_run.make_split_matrices(case)
    loop_rows: list[dict] = []
    for row in seed_rows:
        seed = int(row["seed"])
        print(f"[loop-rms] seed={seed}", flush=True)
        loop_rows.extend(simulate_loop_rms_for_seed(seed, case, splits))

    metrics_csv = OUT / f"{REPORT_STEM}_metrics.csv"
    loop_csv = OUT / f"{REPORT_STEM}_loop_rms.csv"
    json_path = OUT / f"{REPORT_STEM}.json"
    tex_path = OUT / f"{REPORT_STEM}.tex"
    pdf_path = OUT / f"{REPORT_STEM}.pdf"

    write_csv(metrics_csv, seed_rows)
    write_csv(loop_csv, loop_rows)
    json_path.write_text(json.dumps({"seed_rows": seed_rows, "loop_rows": loop_rows}, indent=2) + "\n")
    write_tex(seed_rows, loop_rows, tex_path)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUT, check=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUT, check=True)
    print(pdf_path)
    print(metrics_csv)
    print(loop_csv)
    print(json_path)


if __name__ == "__main__":
    main()
