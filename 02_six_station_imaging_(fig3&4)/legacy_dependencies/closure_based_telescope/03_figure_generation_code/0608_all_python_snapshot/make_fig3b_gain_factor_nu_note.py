from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np


BUNDLE = Path(__file__).resolve().parents[2]
OUT = BUNDLE / "isolated_reference_runs_20260609_near_direct_100ms" / "notes"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("EPS_STATION", "1e-9")
os.environ.setdefault("EPS_PAIR", "0.0")
os.environ.setdefault("EPS_DIRECT_EXTRA", "0.0")
os.environ.setdefault("FIG2_EXPOSURE_S", os.environ.get("EXPOSURE_S", "0.050"))
os.environ.setdefault("EXPOSURE_S", os.environ.get("FIG2_EXPOSURE_S", "0.050"))

import make_fig2_current_seed_diagnostics as diag  # noqa: E402


LOOPS = diag.LOOPS
PDF = OUT / "fig3b_three_station_gain_connection_with_nu_50ms.pdf"
TEX = OUT / "fig3b_three_station_gain_connection_with_nu_50ms.tex"
CSV_SUMMARY = OUT / "fig3b_three_station_gain_connection_with_nu_50ms.csv"
CSV_BAND_NU = OUT / "fig3b_loop_nu_per_band_50ms.csv"
JSON_SUMMARY = OUT / "fig3b_three_station_gain_connection_with_nu_50ms.json"


def weighted_percentiles(values: list[float], weights: list[float], qs: tuple[float, ...]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(arr)
    arr = arr[order]
    w = w[order]
    cdf = np.cumsum(w) / max(float(np.sum(w)), 1e-300)
    return [float(np.interp(q, cdf, arr)) for q in qs]


def scalar_from_cycle(fisher: np.ndarray, q_basis: np.ndarray, edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> float:
    return diag.scalar_fisher_from_cycle_matrix(fisher, q_basis, edges, tri)


def compute() -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], dict[str, float | int | str]]:
    diag.fig_run.configure_good_runtime()
    case = diag.fig_run.scale_remote_coordinates(diag.hawaii3_compact_case.make_hawaii3_compact_remote_case())
    splits = diag.fig_run.make_split_matrices(case)
    diag.split_sim.configure()
    diag.fig_run.apply_sample_stress_runtime()

    stations, diameters, names, _is_added = diag.fig_run.aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n_station = len(stations)
    edges = diag.base.edge_list(n_station)
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = diag.base.orthonormal_cycle_basis(diag.base.root_cycle_basis(edges, n_station))
    rank_share = min(1.0, (n_station - 1.0) / q_basis.shape[1])
    fov_rad = 2.0 * diag.fig_run.aug.HALF_WIDTH_UAS * diag.base.UAS_TO_RAD
    effective_hub_dist = diag.fig_run.aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-diag.fig_run.aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n_station, diag.fig_run.EPS_STATION_RUN, dtype=float)
    direct_noise = np.full(n_station, diag.fig_run.EPS_STATION_RUN + diag.fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    hour_angles = diag.realnight_hour_angles(
        diag.fig_run.aug.N_TIME_WINDOWS,
        diag.fig_run.aug.EXPOSURE_S,
        diag.fig_run.aug.EXPOSURE_GAP_S,
    )

    lam_edges_nm = np.arange(
        diag.fig_run.aug.LAMBDA_MIN_NM,
        diag.fig_run.aug.LAMBDA_MAX_NM + 0.5 * diag.fig_run.aug.LAMBDA_STEP_NM,
        diag.fig_run.aug.LAMBDA_STEP_NM,
    )
    lam_edges_nm[-1] = diag.fig_run.aug.LAMBDA_MAX_NM

    totals = {
        loop_label: {
            "edge_full": 0.0,
            "edge_tri": 0.0,
            "direct_tri": 0.0,
            "nu": {(tri[0], tri[1]): [], (tri[1], tri[2]): [], (tri[0], tri[2]): []},
        }
        for tri, loop_label, _kind in LOOPS
    }
    per_band: list[dict[str, float | str]] = []
    near_cycle = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
    direct_raw_cycle = np.zeros_like(near_cycle)
    direct_sched_cycle = np.zeros_like(near_cycle)

    with diag.fig_run.morph.patched_variant(diag.fig_run.GOOD_VARIANT), diag.fig_run.ngc.patched_source(diag.fig_run.GOOD_SOURCE):
        for band_index, (lo_nm, hi_nm) in enumerate(zip(lam_edges_nm[:-1], lam_edges_nm[1:])):
            center_nm = float(math.sqrt(lo_nm * hi_nm))
            lam_m = center_nm * 1e-9
            freq = diag.base.C_LIGHT / lam_m
            freq_lo = diag.base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = diag.base.C_LIGHT / (lo_nm * 1e-9)
            total_modes = diag.fig_run.aug.EXPOSURE_S * diag.fig_run.OBSERVING_DAYS * (freq_hi - freq_lo)
            u_station = diag.fig_run.aug.station_u_modes(freq, diameters)
            band_truth, _axis = diag.base.make_source_at_wavelength_nm(
                diag.fig_run.aug.N_PIX,
                diag.fig_run.aug.HALF_WIDTH_UAS,
                center_nm,
            )
            band_vgrid, band_uv_axis = diag.base.visibility_grid(band_truth, fov_rad)
            uu_rows, vv_rows = diag.project_enu_baselines(
                baselines,
                hour_angles,
                lam_m,
                latitude_deg=case.latitude_deg,
                declination_deg=diag.fig_run.GOOD_SOURCE.dec_deg,
            )
            band_nu_values: dict[tuple[str, tuple[int, int]], list[float]] = {}
            for uu, vv in zip(uu_rows, vv_rows):
                vtrue = diag.base.interp_vis(band_vgrid, band_uv_axis, uu, vv)
                nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
                near_cycle += diag.split_sim.core4_remote_loop_fisher_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    direct_noise=direct_noise,
                    nu_eff=nu_eff,
                    q_basis=q_basis,
                    edges=edges,
                )
                direct_raw_cycle += total_modes * diag.fig_run.aug.noisy_closure_fisher_station_u(
                    vtrue,
                    eta,
                    direct_noise,
                    u_station,
                    q_basis,
                    edges,
                )
                direct_sched_cycle += diag.split_sim.direct_root_weighted_fisher_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    direct_noise=direct_noise,
                    q_basis=q_basis,
                    edges=edges,
                )

                for tri, loop_label, _kind in LOOPS:
                    a, b, c = tri
                    totals[loop_label]["edge_full"] += diag.uniform_edge_scalar_loop_fisher_for_sample(
                        splits["edge_uniform"],
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )
                    edge_vals = []
                    for i, j in ((a, b), (b, c), (a, c)):
                        edge_vals.append(
                            diag.split_sim.edge_pair_fisher_for_sample(
                                i,
                                j,
                                0.5,
                                0.5,
                                total_modes=total_modes,
                                u_station=u_station,
                                eta=eta,
                                station_noise=station_noise,
                                nu_eff=nu_eff,
                                edge_to_index=edge_to_index,
                            )
                        )
                        nu = float(nu_eff[edge_to_index[(i, j)]])
                        totals[loop_label]["nu"][(i, j)].append((nu, float(total_modes)))
                        band_nu_values.setdefault((loop_label, (i, j)), []).append(nu)
                    totals[loop_label]["edge_tri"] += diag.split_sim.scalar_closure_fisher_from_edges(*edge_vals)
                    totals[loop_label]["direct_tri"] += diag.split_sim.core_triangle_direct_fisher_for_sample(
                        tri,
                        total_modes=total_modes,
                        vtrue=vtrue,
                        u_station=u_station,
                        eta=eta,
                        direct_noise=direct_noise,
                        edges=edges,
                        edge_to_index=edge_to_index,
                    )
            for (loop_label, pair), values in sorted(band_nu_values.items()):
                per_band.append(
                    {
                        "loop": loop_label,
                        "band_index": band_index,
                        "lambda_center_nm": center_nm,
                        "edge": f"{pair[0] + 1}{pair[1] + 1}",
                        "nu_mean": float(np.mean(values)),
                        "nu_min": float(np.min(values)),
                        "nu_max": float(np.max(values)),
                    }
                )

    rows: list[dict[str, float | str]] = []
    for tri, loop_label, _kind in LOOPS:
        f_edge_full = float(totals[loop_label]["edge_full"])
        f_edge_tri = float(totals[loop_label]["edge_tri"])
        f_direct_tri = float(totals[loop_label]["direct_tri"])
        f_direct_full = scalar_from_cycle(direct_raw_cycle, q_basis, edges, tri)
        f_sched = scalar_from_cycle(direct_sched_cycle, q_basis, edges, tri)
        f_near = scalar_from_cycle(near_cycle, q_basis, edges, tri)
        row: dict[str, float | str] = {
            "loop": loop_label,
            "G_fullraw_over_full_edge": math.sqrt(f_direct_full / f_edge_full),
            "G_tri_direct_over_tri_edge": math.sqrt(f_direct_tri / f_edge_tri),
            "sqrt_edge_tri_over_full_edge": math.sqrt(f_edge_tri / f_edge_full),
            "G_sched_over_full_edge": math.sqrt(f_sched / f_edge_full),
            "G_sched_over_tri_edge": math.sqrt(f_sched / f_edge_tri),
            "G_near_over_full_edge": math.sqrt(f_near / f_edge_full),
            "G_near_over_sched": math.sqrt(f_near / f_sched),
            "F_edge_full": f_edge_full,
            "F_edge_tri": f_edge_tri,
            "F_direct_tri": f_direct_tri,
            "F_direct_full_scalar": f_direct_full,
            "F_direct_scheduled_scalar": f_sched,
            "F_near": f_near,
        }
        for pair, values in totals[loop_label]["nu"].items():
            vals = [float(v) for v, _w in values]
            weights = [float(w) for _v, w in values]
            p16, p50, p84 = weighted_percentiles(vals, weights, (0.16, 0.50, 0.84))
            label = f"nu_{pair[0] + 1}{pair[1] + 1}"
            row[f"{label}_mean"] = float(np.average(vals, weights=weights))
            row[f"{label}_p16"] = p16
            row[f"{label}_p50"] = p50
            row[f"{label}_p84"] = p84
        rows.append(row)

    meta: dict[str, float | int | str] = {
        "n_station": n_station,
        "n_closure": int(q_basis.shape[1]),
        "rank_share": float(rank_share),
        "exposure_s": float(diag.fig_run.aug.EXPOSURE_S),
        "n_time_windows": int(diag.fig_run.aug.N_TIME_WINDOWS),
        "lambda_min_nm": float(diag.fig_run.aug.LAMBDA_MIN_NM),
        "lambda_max_nm": float(diag.fig_run.aug.LAMBDA_MAX_NM),
        "lambda_step_nm": float(diag.fig_run.aug.LAMBDA_STEP_NM),
        "eps_station": float(diag.fig_run.EPS_STATION_RUN),
        "eps_pair": float(os.environ.get("EPS_PAIR", "0.0")),
        "eps_direct_extra": float(diag.fig_run.EPS_DIRECT_EXTRA_RUN),
        "note": "nu summaries use clipped |V_ij| values weighted by spectral mode count.",
    }
    return rows, per_band, meta


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: float | str, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def nu_cell(row: dict[str, float | str], edge: str) -> str:
    prefix = f"nu_{edge}"
    mean = float(row[f"{prefix}_mean"])
    p16 = float(row[f"{prefix}_p16"])
    p84 = float(row[f"{prefix}_p84"])
    return f"{mean:.3f} [{p16:.3f},{p84:.3f}]"


def render_tex(rows: list[dict[str, float | str]], meta: dict[str, float | int | str]) -> str:
    nu_lines = []
    for row in rows:
        loop = str(row["loop"])
        if loop == "123":
            edges = ("12", "23", "13")
        elif loop == "125":
            edges = ("12", "25", "15")
        else:
            edges = ("12", "27", "17")
        nu_lines.append(
            f"{loop} & {nu_cell(row, edges[0])} & {nu_cell(row, edges[1])} & {nu_cell(row, edges[2])} \\\\"
        )

    gain_lines = []
    for row in rows:
        gain_lines.append(
            f"{row['loop']} & "
            f"{fmt(row['G_fullraw_over_full_edge'])} & "
            f"{fmt(row['sqrt_edge_tri_over_full_edge'])} & "
            f"{fmt(row['G_tri_direct_over_tri_edge'])} & "
            f"{fmt(row['G_sched_over_full_edge'])} & "
            f"{fmt(row['G_sched_over_tri_edge'])} & "
            f"{fmt(row['G_near_over_full_edge'])} & "
            f"{fmt(row['G_near_over_sched'])} \\\\"
        )

    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.68in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,array}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}

\begin{{document}}

\begin{{center}}
{{\Large Fig. 3(b) gain normalization and $\nu_{{ij}}$ check}}\\[0.3em]
{{\normalsize 50 ms, 600--700 nm in 10 nm bins, $\epsilon_i=10^{{-9}}$, $\epsilon_{{ij}}=\epsilon_{{\rm dir}}=0$}}
\end{{center}}

\section*{{Executive check}}

The large values in the old column $G_{{\rm raw}}/G_{{\rm edge}}$ do not come from an accidental extra
$\sqrt{{N-1}}$ multiplier in the direct Fisher. They come from the denominator: the plotted edge-first
reference is the full seven-station uniform edge readout, where each station sends $1/(N-1)=1/6$
of its light to each baseline. The isolated three-station triangle used in the analytic discussion
instead sends $1/2$ of each station to each of the two triangle baselines. With photon-dominated
noise this changes the edge closure Fisher by exactly a factor of three,
\[
  {{F_{{\rm edge}}^{{\rm tri}}\over F_{{\rm edge}}^{{\rm full}}}}=3,\qquad
  {{\sigma_{{\rm edge}}^{{\rm full}}\over \sigma_{{\rm edge}}^{{\rm tri}}}}=\sqrt{{3}}.
\]
The factor is therefore $\sqrt{{(N-1)/2}}=\sqrt{{3}}$ when comparing the full-array edge denominator
to an isolated three-station equal-split denominator. It would be $\sqrt{{N-1}}=\sqrt{{6}}$ only if
one compared to an unsplit single-baseline reference, which is not the three-station closure
benchmark in Fig. 1(a).

\section*{{$\nu_{{ij}}$ values for the representative loops}}

Entries are weighted mean $[16,84]\%$ of the clipped visibility amplitude $\nu_{{ij}}=|V_{{ij}}|$
over the 10 wavelength bins and {int(meta['n_time_windows'])} scheduled samples.

\begin{{center}}
\small
\begin{{tabular}}{{c c c c}}
\toprule
Loop & first edge & second edge & third edge\\
\midrule
{chr(10).join(nu_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Gain dictionary}}

Here ``full edge'' is the actual Fig. 3(b) edge-first denominator with station split $1/6$.
``tri edge'' is the isolated three-station equal-split denominator with split $1/2$.
``full raw'' is the unscheduled full-array scalar direct-QFI projection used by the old diagnostic,
whereas ``tri direct'' is the strict local three-station direct receiver.

\begin{{center}}
\scriptsize
\begin{{tabular}}{{c c c c c c c c}}
\toprule
Loop &
$G_{{\rm full\,raw/full\,edge}}$ &
$\sqrt{{F_{{\rm tri\,edge}}/F_{{\rm full\,edge}}}}$ &
$G_{{\rm tri\,direct/tri\,edge}}$ &
$G_{{\rm sched/full\,edge}}$ &
$G_{{\rm sched/tri\,edge}}$ &
$G_{{\rm near/full\,edge}}$ &
$G_{{\rm near/sched}}$\\
\midrule
{chr(10).join(gain_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Interpretation}}

For loop 123, the old full-array raw number is about $5.21$, but removing the full-array
edge splitting denominator gives the strict three-station value about $3.01$. For loops 125
and 127 the corresponding strict three-station gains are about $2.24$ and $2.21$. These
are still substantial because the remote baselines have low $\nu_{{ij}}$ and the direct
closure receiver uses the joint three-mode covariance instead of three separately noisy
baseline estimates.

The scheduled direct benchmark in Fig. 3(b) additionally applies the capacity-relaxed scalar
weight
\[
  w_\ell=(N-1)/C=({int(meta['n_station']) - 1})/{int(meta['n_closure'])}={float(meta['rank_share']):.3f}.
\]
Thus the plotted red gain obeys
\[
  G_{{\rm sched/full\,edge}}
  \simeq
  \sqrt{{w_\ell}}\,
  \sqrt{{F_{{\rm tri\,edge}}\over F_{{\rm full\,edge}}}}\,
  G_{{\rm tri\,direct/tri\,edge}},
\]
up to the small distinction between strict local-triangle direct Fisher and the full-array scalar
direct projection.

The important conclusion is: the old table was not numerically multiplying by an extra
$\sqrt{{N-1}}$ by mistake, but its label was easy to misread. The full-array gain column included
the full-array edge splitting penalty in the denominator. When making contact with the analytic
three-station Fig. 1(a), use $G_{{\rm tri\,direct/tri\,edge}}$, not
$G_{{\rm full\,raw/full\,edge}}$.

\end{{document}}
"""


def main() -> None:
    rows, per_band, meta = compute()
    write_csv(CSV_SUMMARY, rows)
    write_csv(CSV_BAND_NU, per_band)
    JSON_SUMMARY.write_text(json.dumps({"meta": meta, "rows": rows, "per_band_nu": per_band}, indent=2) + "\n")
    TEX.write_text(render_tex(rows, meta))
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TEX.name],
        cwd=OUT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(json.dumps({"pdf": str(PDF), "tex": str(TEX), "csv": str(CSV_SUMMARY)}, indent=2))


if __name__ == "__main__":
    main()
