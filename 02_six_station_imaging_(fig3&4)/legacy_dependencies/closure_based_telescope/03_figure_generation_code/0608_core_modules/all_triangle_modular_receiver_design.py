from __future__ import annotations

import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


BUNDLE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import make_all_closure_global_benchmark_note as bm_lib  # noqa: E402
import plot_augmented_existing_telescope_closure_networks as aug  # noqa: E402
import plot_augmented_existing_telescope_ngc_sources as ngc  # noqa: E402
import plot_prl_broadband_clean as base  # noqa: E402
from plot_prl_broadband_blr_realnight import project_enu_baselines  # noqa: E402


OUT = BUNDLE / "exploration" / "all_triangle_modular_receiver"
OUT.mkdir(parents=True, exist_ok=True)


def edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


def measurement_vector(bm: bm_lib.AllClosureBenchmark, tri: tuple[int, int, int]) -> np.ndarray:
    return bm.q_basis.T @ edge_vector(bm.edges, tri)


def edge_fisher_from_arrays(arrays: dict[str, np.ndarray], fi: float, fj: float) -> float:
    if fi <= 0.0 or fj <= 0.0:
        return 0.0
    denom = fi * arrays["ai"] + fj * arrays["aj"] + arrays["pair"]
    return float(np.sum(arrays["k"] * fi * fj / np.maximum(denom, 1e-300)))


def closure_fisher_from_edge_fishers(fab: float, fbc: float, fac: float) -> float:
    if min(fab, fbc, fac) <= 0.0:
        return 0.0
    return 1.0 / (1.0 / fab + 1.0 / fbc + 1.0 / fac)


def triangle_direct_fisher(
    bm: bm_lib.AllClosureBenchmark,
    tri_global: tuple[int, int, int],
    station_fractions: tuple[float, float, float],
) -> float:
    """Three-mode direct closure Fisher for one triangle.

    station_fractions are intensity fractions of each participating station mode
    sent to this 3-mode receiver. Input station false positives are split with
    the optical field. Receiver-added direct false positives are not split.
    """
    subset = tuple(tri_global)
    local_edges = base.edge_list(3)
    local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, 3))
    local_stations = bm.stations[list(subset)]
    local_baselines = np.asarray([local_stations[j] - local_stations[i] for i, j in local_edges], dtype=float)
    frac = np.asarray(station_fractions, dtype=float)
    eta = frac * bm.eta[list(subset)]
    noise = frac * bm_lib.EPS_STATION + bm_lib.EPS_DIRECT_EXTRA
    local_fq = np.zeros((local_q.shape[1], local_q.shape[1]), dtype=float)
    for lam, freq, total_modes in bm.iter_bands():
        u_station = aug.station_u_modes(freq, bm.diameters[list(subset)])
        uu_rows, vv_rows = project_enu_baselines(
            local_baselines,
            bm.hour_angles,
            lam,
            latitude_deg=bm.case.latitude_deg,
            declination_deg=ngc.NGC4151.dec_deg,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            vlocal = base.interp_vis(bm.vgrid, bm.uv_axis, uu, vv)
            local_fq += total_modes * aug.noisy_closure_fisher_station_u(
                vlocal,
                eta,
                noise,
                u_station,
                local_q,
                local_edges,
            )
    c = edge_vector(local_edges, (0, 1, 2))
    d = local_q.T @ c
    cov = np.linalg.pinv(local_fq, rcond=1e-12)
    var = float(d @ cov @ d)
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def optimize_triangle_edge_split_with_budgets(
    bm: bm_lib.AllClosureBenchmark,
    tri: tuple[int, int, int],
    station_budgets: tuple[float, float, float],
) -> tuple[float, tuple[float, float, float], tuple[float, float, float, float, float, float]]:
    """Optimize edge-first readout inside one triangle under station budgets.

    For tri=(a,b,c), x_a sends station a's triangle budget to baseline ab and
    1-x_a to ac; x_b sends b's budget to ab and 1-x_b to bc; x_c sends c's
    budget to bc and 1-x_c to ac.
    """
    a, b, c = tri
    ga, gb, gc = station_budgets
    eab = bm.edge_arrays[(a, b)]
    ebc = bm.edge_arrays[(b, c)]
    eac = bm.edge_arrays[(a, c)]

    def score(xa: float, xb: float, xc: float) -> float:
        fab = edge_fisher_from_arrays(eab, ga * xa, gb * xb)
        fbc = edge_fisher_from_arrays(ebc, gb * (1.0 - xb), gc * xc)
        fac = edge_fisher_from_arrays(eac, ga * (1.0 - xa), gc * (1.0 - xc))
        return closure_fisher_from_edge_fishers(fab, fbc, fac)

    floor = bm_lib.SPLIT_FLOOR
    grid = np.linspace(floor, 1.0 - floor, 121)
    seeds = [
        (0.5, 0.5, 0.5),
        (floor, floor, 0.5),
        (1.0 - floor, 1.0 - floor, 0.5),
        (0.5, floor, 1.0 - floor),
        (0.5, 1.0 - floor, floor),
    ]
    best = seeds[0]
    best_score = score(*best)
    for seed in seeds:
        xa, xb, xc = seed
        local_score = score(xa, xb, xc)
        for _ in range(16):
            local_score, xa = max((score(v, xb, xc), v) for v in grid)
            local_score, xb = max((score(xa, v, xc), v) for v in grid)
            local_score, xc = max((score(xa, xb, v), v) for v in grid)
        if local_score > best_score:
            best_score = local_score
            best = (xa, xb, xc)
    xa, xb, xc = best
    absolute_edge_fractions = (ga * xa, gb * xb, gb * (1.0 - xb), gc * xc, ga * (1.0 - xa), gc * (1.0 - xc))
    return best_score, best, absolute_edge_fractions


def add_scalar_measurement(
    fisher: np.ndarray,
    bm: bm_lib.AllClosureBenchmark,
    tri: tuple[int, int, int],
    scalar_fisher: float,
) -> None:
    d = measurement_vector(bm, tri)
    fisher += scalar_fisher * np.outer(d, d)


def closure_rms_rows(
    bm: bm_lib.AllClosureBenchmark,
    matrices: dict[str, np.ndarray],
    per_triangle: dict[tuple[int, int, int], dict[str, float | str]],
) -> list[dict[str, float | str]]:
    covs = {key: np.linalg.pinv(0.5 * (mat + mat.T), rcond=1e-12) for key, mat in matrices.items()}
    rows: list[dict[str, float | str]] = []
    for tri in itertools.combinations(range(bm.n), 3):
        d = measurement_vector(bm, tri)
        row: dict[str, float | str] = {
            "closure": f"{tri[0] + 1}-{tri[1] + 1}-{tri[2] + 1}",
            "stations": " | ".join(bm.names[i] for i in tri),
            "type": "core_only" if all(not bm.is_added[i] for i in tri) else "remote_involved",
        }
        row.update(per_triangle[tri])
        for key, cov in covs.items():
            var = float(d @ cov @ d)
            row[f"rms_{key}_rad"] = math.sqrt(var) if np.isfinite(var) and var > 0.0 else math.inf
        for key in matrices:
            row[f"gain_{key}_vs_edge_uniform"] = float(row["rms_edge_uniform_rad"]) / max(
                float(row[f"rms_{key}_rad"]), 1e-300
            )
            row[f"gain_{key}_vs_direct_scheduled"] = float(row["rms_direct_scheduled_rad"]) / max(
                float(row[f"rms_{key}_rad"]), 1e-300
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def write_note(
    bm: bm_lib.AllClosureBenchmark,
    payload: dict,
    rows: list[dict[str, float | str]],
    core_rows: list[dict[str, float | str]],
    remote_rows: list[dict[str, float | str]],
) -> tuple[Path, Path]:
    tex_path = OUT / "all_triangle_modular_receiver_note.tex"
    pdf_path = OUT / "all_triangle_modular_receiver_note.pdf"

    metric_lines = []
    for item in payload["matrix_metrics"]:
        metric_lines.append(
            f"{latex_escape(item['strategy'])} & {item['mean_coord_rms']:.3g} & "
            f"{item['median_coord_rms']:.3g} & {item['max_coord_rms']:.3g} & "
            f"{item['mean_rms_gain_vs_edge_uniform']:.2f} & "
            f"{item['mean_rms_gain_vs_direct_scheduled']:.2f} \\\\"
        )

    def compact_lines(selected: list[dict[str, float | str]], gain_key: str) -> list[str]:
        lines = []
        for row in selected:
            lines.append(
                f"{row['closure']} & {latex_escape(str(row['type']))} & "
                f"{float(row['receiver_weight_per_station']):.4f} & "
                f"{float(row['scalar_fisher_all3_direct']):.3e} & "
                f"{float(row['scalar_fisher_modular']):.3e} & "
                f"{float(row[gain_key]):.2f} \\\\"
            )
        return lines

    core_table = "\n".join(compact_lines(core_rows, "gain_parallel_modular_hybrid_vs_direct_scheduled"))
    remote_table = "\n".join(compact_lines(remote_rows, "gain_parallel_modular_hybrid_vs_direct_scheduled"))

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.65in]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{All-triangle modular receiver design for the 7-station closure network}}
\author{{Codex diagnostic note}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Resource model}}
The network has \(N=7\) stations, \(E=21\) baselines, and \(C=15\) independent
closure degrees of freedom.  We use all \(M=\binom{{7}}{{3}}=35\) triangle
closures as an overcomplete closure frame.  Each station participates in
\[
\binom{{6}}{{2}}=15
\]
triangles.  The resource-conserving parallel all-triangle design therefore
sends
\[
f_{{i\Delta}}=\frac{{1}}{{15}}
\]
of station \(i\)'s optical mode to each triangle receiver \(\Delta\) containing
that station.  This removes time duty-cycle loss but keeps station-side photon
budget conservation.

The noise model is the active manuscript diagnostic model:
\[
\epsilon_i={bm_lib.EPS_STATION:.2f},\qquad
\epsilon_{{\rm pair}}=\epsilon_{{\rm dir}}={bm_lib.EPS_PAIR:.2f},\qquad
L_{{\rm fib}}={bm_lib.FIBER_LOSS_DB_PER_KM:.2f}\ {{\rm dB/km}}.
\]
Fibre attenuation is signal loss.  Station-local false positives are split with
the field; receiver-added false positives are not split.

\section*{{Receivers compared}}
\begin{{enumerate}}
\item \textbf{{All-triangle 3-mode direct:}} all 35 closures are measured by
parallel 3-mode direct closure receivers with \(f_{{i\Delta}}=1/15\).
\item \textbf{{Parallel modular hybrid:}} core-only triangles inside the four
current telescopes use the same 3-mode direct receivers.  Any triangle involving
a remote station is measured by optimized edge-first readout, using the same
station triangle budget \(1/15\) and optimizing how that budget is split between
the two local baselines of the triangle.
\item \textbf{{Reference:}} edge-uniform is simultaneous all-baseline edge-first
with station split \(1/(N-1)=1/6\).  Direct-scheduled is the previous
\((N-1)/C=6/15\) rank-share proxy applied to the full 7-mode closure QFI.
\end{{enumerate}}

\section*{{Matrix-level result}}
\begin{{center}}
\begin{{tabular}}{{lrrrrr}}
\toprule
strategy & mean RMS & median RMS & max RMS & gain vs edge & gain vs scheduled direct \\
\midrule
{chr(10).join(metric_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Core-only closures}}
\begin{{center}}
\begin{{tabular}}{{llrrrr}}
\toprule
closure & type & station weight & \(F_\Delta^{{3m}}\) & \(F_\Delta^{{mod}}\) & modular/scheduled SNR \\
\midrule
{core_table}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Remote-involved closures}}
\small
\begin{{longtable}}{{llrrrr}}
\toprule
closure & type & station weight & \(F_\Delta^{{3m}}\) & \(F_\Delta^{{mod}}\) & modular/scheduled SNR \\
\midrule
{remote_table}
\bottomrule
\end{{longtable}}

\section*{{Interpretation}}
The all-triangle set is the natural physical version of the closure-frame idea:
no triangle adds a new parameter, but each gives a noisy projection in the
15-dimensional closure subspace.  Uniform all-triangle weighting avoids the
special burden on station 1 that appears in a root-loop basis.

The important implementation caveat is that putting many receivers in parallel
does not copy the optical mode.  The price is the \(1/15\) station-side split.
In the current weak-light false-positive model this split is costly; therefore
the resource-conserving all-triangle receiver need not equal the idealized
\(6/15\) scheduled-direct proxy.  The modular hybrid is the lower-complexity
version of the same closure frame: use small joint receivers where the three
baselines are comparable, and use optimized edge-first readout for asymmetric
remote triangles.

Full per-closure rows, including optimized split coefficients, are written to
\texttt{{all\_triangle\_closure\_weights\_and\_gains.csv}}.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUT, check=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUT, check=True)
    return tex_path, pdf_path


def main() -> None:
    bm = bm_lib.AllClosureBenchmark()
    triangles = list(itertools.combinations(range(bm.n), 3))
    station_triangle_count = math.comb(bm.n - 1, 2)
    station_weight = 1.0 / station_triangle_count

    all3 = np.zeros((bm.n_closure, bm.n_closure), dtype=float)
    modular = np.zeros_like(all3)
    per_triangle: dict[tuple[int, int, int], dict[str, float | str]] = {}

    for tri in triangles:
        fractions = (station_weight, station_weight, station_weight)
        f_direct = triangle_direct_fisher(bm, tri, fractions)
        add_scalar_measurement(all3, bm, tri, f_direct)
        core_only = all(not bm.is_added[i] for i in tri)
        if core_only:
            f_mod = f_direct
            receiver = "3-mode direct"
            split_xa = split_xb = split_xc = math.nan
            abs_fracs = (math.nan,) * 6
        else:
            f_mod, (split_xa, split_xb, split_xc), abs_fracs = optimize_triangle_edge_split_with_budgets(
                bm, tri, fractions
            )
            receiver = "optimized edge-first"
        add_scalar_measurement(modular, bm, tri, f_mod)
        single_gain = math.sqrt(f_direct / f_mod) if f_mod > 0.0 else math.inf
        per_triangle[tri] = {
            "receiver_weight_per_station": station_weight,
            "receiver": receiver,
            "scalar_fisher_all3_direct": f_direct,
            "scalar_fisher_modular": f_mod,
            "single_measurement_snr_direct3_over_modular": single_gain,
            "split_xa_a_to_ab": split_xa,
            "split_xb_b_to_ab": split_xb,
            "split_xc_c_to_bc": split_xc,
            "abs_frac_a_to_ab": abs_fracs[0],
            "abs_frac_b_to_ab": abs_fracs[1],
            "abs_frac_b_to_bc": abs_fracs[2],
            "abs_frac_c_to_bc": abs_fracs[3],
            "abs_frac_a_to_ac": abs_fracs[4],
            "abs_frac_c_to_ac": abs_fracs[5],
        }

    edge_uniform = bm.edge_closure_fisher(bm.uniform_split_matrix())
    direct_scheduled = bm.rank_share * bm.direct_raw
    direct_raw = bm.direct_raw
    matrices = {
        "edge_uniform": edge_uniform,
        "direct_scheduled": direct_scheduled,
        "direct_raw_qfi_upper": direct_raw,
        "all_triangle_3mode_parallel": all3,
        "parallel_modular_hybrid": modular,
    }
    rows = closure_rms_rows(bm, matrices, per_triangle)

    metrics_ref_edge = bm_lib.stable_metrics(edge_uniform)
    metrics_ref_sched = bm_lib.stable_metrics(direct_scheduled)
    matrix_metrics = []
    for key, fisher in matrices.items():
        metrics = bm_lib.stable_metrics(fisher)
        matrix_metrics.append(
            {
                "strategy": key,
                **metrics,
                "mean_rms_gain_vs_edge_uniform": metrics_ref_edge["mean_coord_rms"] / metrics["mean_coord_rms"],
                "mean_rms_gain_vs_direct_scheduled": metrics_ref_sched["mean_coord_rms"] / metrics["mean_coord_rms"],
            }
        )

    csv_path = OUT / "all_triangle_closure_weights_and_gains.csv"
    write_csv(csv_path, rows)

    core_rows = [row for row in rows if row["type"] == "core_only"]
    remote_rows = [row for row in rows if row["type"] == "remote_involved"]
    split_rows = [
        {
            "closure": row["closure"],
            "stations": row["stations"],
            "receiver": row["receiver"],
            "station_weight": row["receiver_weight_per_station"],
            "split_xa_a_to_ab": row["split_xa_a_to_ab"],
            "split_xb_b_to_ab": row["split_xb_b_to_ab"],
            "split_xc_c_to_bc": row["split_xc_c_to_bc"],
            "abs_frac_a_to_ab": row["abs_frac_a_to_ab"],
            "abs_frac_b_to_ab": row["abs_frac_b_to_ab"],
            "abs_frac_b_to_bc": row["abs_frac_b_to_bc"],
            "abs_frac_c_to_bc": row["abs_frac_c_to_bc"],
            "abs_frac_a_to_ac": row["abs_frac_a_to_ac"],
            "abs_frac_c_to_ac": row["abs_frac_c_to_ac"],
        }
        for row in rows
    ]
    split_csv = OUT / "parallel_modular_hybrid_split_coefficients.csv"
    write_csv(split_csv, split_rows)

    payload = {
        "case": bm.case.key,
        "station_names": bm.names,
        "station_is_added": bm.is_added.tolist(),
        "n_stations": bm.n,
        "n_edges": len(bm.edges),
        "n_independent_closures": bm.n_closure,
        "n_triangle_receivers": len(triangles),
        "station_triangle_count": station_triangle_count,
        "station_weight_per_triangle": station_weight,
        "rank_share_direct_proxy": bm.rank_share,
        "eps_station": bm_lib.EPS_STATION,
        "eps_pair": bm_lib.EPS_PAIR,
        "eps_direct_extra": bm_lib.EPS_DIRECT_EXTRA,
        "fiber_loss_db_per_km": bm_lib.FIBER_LOSS_DB_PER_KM,
        "matrix_metrics": matrix_metrics,
        "closure_csv": str(csv_path),
        "split_csv": str(split_csv),
    }
    json_path = OUT / "all_triangle_modular_receiver_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    tex_path, pdf_path = write_note(bm, payload, rows, core_rows, remote_rows)
    print(json.dumps(payload, indent=2))
    print(tex_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
