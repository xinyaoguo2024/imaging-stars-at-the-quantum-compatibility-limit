from __future__ import annotations

import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

import hawaii3_compact_case
import make_all_closure_global_benchmark_note as closure_bm
import plot_augmented_existing_telescope_closure_networks as aug
import plot_prl_broadband_clean as base
from make_all_closure_global_benchmark_note import stable_metrics
from make_all_closure_global_benchmark_note_v2 import optimize_split


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
MORPH_DIR = ROOT / "rml_remote3_source_morphology_optimization_20260525"
if str(MORPH_DIR) not in sys.path:
    sys.path.insert(0, str(MORPH_DIR))
import optimize_remote3_source_morphology_priors as morph  # noqa: E402


OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

TAG = "hawaii3_compact_split_ratios_closure_gains_20260527"


def configure_benchmark_physics() -> None:
    """Use the same physics knobs as the broad-plume Hawaii+3 RML benchmark."""
    closure_bm.EPS_STATION = 0.05
    closure_bm.EPS_PAIR = 0.0
    closure_bm.EPS_DIRECT_EXTRA = 0.0
    closure_bm.SPLIT_FLOOR = 0.02
    closure_bm.FIBER_LENGTH_SCALE = 0.75
    closure_bm.FIBER_LOSS_DB_PER_KM = 0.20


def edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


def closure_rows(bm: closure_bm.AllClosureBenchmark, matrices: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    covariances = {key: np.linalg.pinv(mat, rcond=1e-12) for key, mat in matrices.items()}
    for tri in itertools.combinations(range(bm.n), 3):
        c = edge_vector(bm.edges, tri)
        d = bm.q_basis.T @ c
        row: dict[str, float | str] = {
            "loop": f"{tri[0] + 1}-{tri[1] + 1}-{tri[2] + 1}",
            "stations": " / ".join(bm.names[i] for i in tri),
            "type": "core" if all(not bm.is_added[i] for i in tri) else "remote",
        }
        for key, cov in covariances.items():
            var = float(d @ cov @ d)
            row[f"rms_{key}_rad"] = math.sqrt(max(var, 0.0)) if np.isfinite(var) else math.inf
        ref = float(row["rms_edge_uniform_rad"])
        for key in matrices:
            row[f"gain_{key}_vs_uniform"] = ref / max(float(row[f"rms_{key}_rad"]), 1e-300)
        rows.append(row)
    return rows


def matrix_tex(matrix: np.ndarray, names: list[str], caption: str) -> str:
    cols = "l" + "r" * len(names)
    header = "from $\\backslash$ to & " + " & ".join(short_name(name) for name in names) + r" \\"
    lines = [rf"\begin{{table}}[!ht]", r"\centering", r"\scriptsize", rf"\caption*{{{caption}}}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header, r"\midrule"]
    for i, name in enumerate(names):
        vals = []
        for j in range(len(names)):
            vals.append("--" if i == j else f"{100.0 * matrix[i, j]:.1f}")
        lines.append(short_name(name) + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def short_name(name: str) -> str:
    return (
        name.replace("Remote-", "R")
        .replace("Gemini North", "Gemini")
        .replace("Subaru", "Subaru")
        .replace("Keck I", "Keck-I")
        .replace("Keck II", "Keck-II")
    )


def fmt_gain(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def fmt_rms(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{1e3 * value:.2f}"


def write_note() -> tuple[Path, Path, Path, Path, dict]:
    configure_benchmark_physics()
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    old_loader = closure_bm.rml_cases.load_maunakea_plus3_case
    closure_bm.rml_cases.load_maunakea_plus3_case = lambda: case
    try:
        with morph.patched_variant(morph.VARIANTS[0]):
            bm = closure_bm.AllClosureBenchmark()
            splits = {
                "edge_uniform": bm.uniform_split_matrix(),
                "edge_meanrms": optimize_split(bm, "mean_rms")[0],
                "edge_maxrms": optimize_split(bm, "max_rms")[0],
            }
    finally:
        closure_bm.rml_cases.load_maunakea_plus3_case = old_loader

    matrices = {
        "edge_uniform": bm.edge_closure_fisher(splits["edge_uniform"]),
        "edge_meanrms": bm.edge_closure_fisher(splits["edge_meanrms"]),
        "edge_maxrms": bm.edge_closure_fisher(splits["edge_maxrms"]),
        "nmode_scheduled": bm.rank_share * bm.direct_raw,
        "nmode_raw_qfi": bm.direct_raw,
    }
    metrics = {key: stable_metrics(value) for key, value in matrices.items()}
    rows = closure_rows(bm, matrices)

    split_csv = OUT / f"{TAG}_split_ratios.csv"
    loop_csv = OUT / f"{TAG}_closure_gains.csv"
    json_path = OUT / f"{TAG}.json"
    tex_path = OUT / f"{TAG}.tex"
    pdf_path = OUT / f"{TAG}.pdf"

    with split_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "from_station", "to_station", "fraction"])
        for strategy, p in splits.items():
            for i, ni in enumerate(bm.names):
                for j, nj in enumerate(bm.names):
                    if i != j:
                        writer.writerow([strategy, ni, nj, p[i, j]])

    fieldnames = list(rows[0].keys())
    with loop_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "case": case.key,
        "station_names": bm.names,
        "station_xy_km": bm.stations.tolist(),
        "hub_km": list(case.hub_km),
        "physics": {
            "eps_station": closure_bm.EPS_STATION,
            "eps_pair": closure_bm.EPS_PAIR,
            "eps_direct_extra": closure_bm.EPS_DIRECT_EXTRA,
            "split_floor": closure_bm.SPLIT_FLOOR,
            "fiber_length_scale": closure_bm.FIBER_LENGTH_SCALE,
            "fiber_loss_db_per_km": closure_bm.FIBER_LOSS_DB_PER_KM,
            "n_time_windows": aug.N_TIME_WINDOWS,
            "exposure_s": aug.EXPOSURE_S,
            "observing_days": aug.OBSERVING_DAYS,
            "note": "Exposure-scale changes multiply all Fisher matrices nearly uniformly; loop SNR gains are therefore quoted for the full 36 x 10 min reference.",
        },
        "split_ratios": {key: value.tolist() for key, value in splits.items()},
        "metrics": metrics,
        "closure_rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    metric_lines = []
    labels = {
        "edge_uniform": "edge uniform",
        "edge_meanrms": "edge mean-RMS split",
        "edge_maxrms": "edge max-RMS split",
        "nmode_scheduled": "N-mode scheduled proxy",
        "nmode_raw_qfi": "N-mode raw-QFI bound",
    }
    ref = metrics["edge_uniform"]
    for key, label in labels.items():
        m = metrics[key]
        metric_lines.append(
            f"{label} & {m['mean_coord_rms']:.3e} & {m['max_coord_rms']:.3e} & "
            f"{math.sqrt(m['trace_fisher'] / ref['trace_fisher']):.2f} & "
            f"{ref['mean_coord_rms'] / m['mean_coord_rms']:.2f} \\\\"
        )

    loop_lines = []
    for row in rows:
        loop_lines.append(
            f"{row['loop']} & {short_name(str(row['stations']))} & {row['type']} & "
            f"{fmt_rms(float(row['rms_edge_uniform_rad']))} & "
            f"{fmt_gain(float(row['gain_edge_meanrms_vs_uniform']))} & "
            f"{fmt_gain(float(row['gain_edge_maxrms_vs_uniform']))} & "
            f"{fmt_gain(float(row['gain_nmode_scheduled_vs_uniform']))} & "
            f"{fmt_gain(float(row['gain_nmode_raw_qfi_vs_uniform']))} \\\\"
        )

    split_tables = "\n\n".join(
        matrix_tex(splits[key], bm.names, f"{labels[key]} split ratios $100p_{{i\\to j}}$ (percent)")
        for key in ("edge_uniform", "edge_meanrms", "edge_maxrms")
    )

    tex = rf"""\documentclass[9pt]{{article}}
\usepackage[margin=0.45in,landscape]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,caption,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\begin{{document}}
\title{{Split Ratios and Per-Closure SNR Gains for the Hawaii+3 Compact Benchmark}}
\author{{Codex diagnostic note}}
\date{{\today}}
\maketitle

\section*{{Definition and caveat}}
This note uses the same Hawaii top-four plus remote-three compact array and broad-plume NGC~4151 source model as the recent RML benchmark.  The full-reference observing setup is \(36\) time windows, \(10\,\mathrm{{min}}\) per window, \(30\) observing days, \(0.2\,\mathrm{{dB/km}}\) fiber attenuation, \(\epsilon_i=0.05\), and \(\epsilon_{{ij}}=0\).

For each triangular closure \(c=(i,j,k)\), the marginal closure variance is
\[
\sigma_c^2 = d_c^T F^+ d_c,\qquad d_c=Q^T(e_{{ij}}+e_{{jk}}-e_{{ik}}),
\]
where \(F\) is the all-closure Fisher matrix after station-gauge marginalization.  The tabulated gain is
\[
G_c({{\rm strategy}})=\sigma_c({{\rm edge\ uniform}})/\sigma_c({{\rm strategy}}).
\]
Thus \(G_c>1\) means better than uniform edge-first.  The N-mode raw-QFI column is an upper bound; the scheduled column multiplies the raw QFI by \((N-1)/M={bm.rank_share:.2f}\).

Changing only the exposure scale from \(0.5\) to \(1\) multiplies all Fisher matrices almost uniformly, so it changes absolute RMS but not these dimensionless gains.  The separate full-reference RML image rerun is therefore the visual reference, while this PDF reports the dimensionless closure-gain bookkeeping.

\section*{{Global metrics}}
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
strategy & mean RMS & max RMS & trace-SNR gain & mean-RMS gain \\
\midrule
{chr(10).join(metric_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\clearpage
\section*{{Station split ratios}}
Rows are transmitting stations and columns are the stations receiving a split of that station's photon budget. Entries are percentages; diagonal entries are absent.

{split_tables}

\clearpage
\section*{{Per triangular closure SNR gains}}
The reference RMS is given in mrad for the uniform edge-first split.  Other columns are gains relative to that reference.
\scriptsize
\begin{{longtable}}{{lllrrrrr}}
\toprule
loop & stations & type & RMS$_{{\rm uni}}$ [mrad] & mean-RMS & max-RMS & N-mode sched. & raw QFI \\
\midrule
\endfirsthead
\toprule
loop & stations & type & RMS$_{{\rm uni}}$ [mrad] & mean-RMS & max-RMS & N-mode sched. & raw QFI \\
\midrule
\endhead
{chr(10).join(loop_lines)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Reading the table}}
Mean-RMS and max-RMS split optimization do not simply improve every triangular loop.  They redistribute station photon budget to improve the global all-closure covariance, so some already-strong loops can lose SNR while weak loops gain.  This is why the loop-by-loop table is more informative than a single averaged gain.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tex_path, pdf_path, split_csv, loop_csv, payload


def main() -> None:
    tex_path, pdf_path, split_csv, loop_csv, payload = write_note()
    print(tex_path)
    print(pdf_path)
    print(split_csv)
    print(loop_csv)
    print(json.dumps(payload["physics"], indent=2))


if __name__ == "__main__":
    main()
