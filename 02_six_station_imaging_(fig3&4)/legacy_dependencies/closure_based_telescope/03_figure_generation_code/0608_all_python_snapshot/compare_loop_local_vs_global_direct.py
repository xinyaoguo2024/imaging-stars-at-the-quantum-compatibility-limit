from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

import eht_style_amplitude_closure_rml as rml_cases
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_clean as base
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def configure_fig3_physics() -> None:
    """Match the latest Hawaii+remote3 Fig. 3 detector convention."""
    aug.OBSERVING_DAYS = 30
    aug.N_TIME_WINDOWS = 36
    aug.EXPOSURE_S = 600.0
    aug.EXPOSURE_GAP_S = 150.0
    aug.FIBER_LOSS_DB_PER_KM = 0.20
    aug.FIBER_LENGTH_SCALE = 0.75
    aug.MODE_FALSE_POSITIVE = 0.05
    aug.PAIR_FALSE_POSITIVE = 0.0
    aug.BASELINE_FALSE_POSITIVE = 0.0
    wt.OBSERVING_DAYS = 30
    wt.SNR_BOOST = 1.0


def edge_vector_for_triangle(
    edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
    tri: tuple[int, int, int],
) -> np.ndarray:
    """Return c such that c.phi = phi_ij + phi_jk + phi_ki for i<j<k."""
    i, j, k = tri
    c = np.zeros(len(edges), dtype=float)
    c[edge_to_index[(i, j)]] = 1.0
    c[edge_to_index[(j, k)]] = 1.0
    c[edge_to_index[(i, k)]] = -1.0
    return c


def scalar_fisher_for_closure(fq: np.ndarray, q_basis: np.ndarray, c_edge: np.ndarray) -> float:
    """Convert a closure-subspace Fisher matrix into Fisher for c_edge.phi."""
    d = q_basis.T @ c_edge
    cov = np.linalg.pinv(fq, rcond=1e-12)
    var = float(d @ cov @ d)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0
    return 1.0 / var


def scalar_sld_fisher_for_closure_path(fq: np.ndarray, q_basis: np.ndarray, c_edge: np.ndarray) -> float:
    """Fisher for a single-parameter path whose physical closure is c_edge.phi.

    This is the relevant benchmark for a receiver optimized only for one
    selected loop.  If C=c_edge.phi and ||c_edge||^2=3 for a triangle, then
    d phi/dC = c_edge/3.
    """
    denom = float(c_edge @ c_edge)
    dq_dclosure = q_basis.T @ (c_edge / denom)
    return float(dq_dclosure @ fq @ dq_dclosure)


def edge_fisher_from_arrays(arrays: dict[str, np.ndarray], fi: float, fj: float) -> float:
    if fi <= 0.0 or fj <= 0.0:
        return 0.0
    denom = fi * arrays["ai"] + fj * arrays["aj"] + arrays["pair"]
    return float(np.sum(arrays["k"] * fi * fj / np.maximum(denom, 1e-300)))


def closure_fisher_from_edges(f1: float, f2: float, f3: float) -> float:
    if min(f1, f2, f3) <= 0.0:
        return 0.0
    return 1.0 / (1.0 / f1 + 1.0 / f2 + 1.0 / f3)


def optimize_triangle_split(
    edge_arrays: dict[tuple[int, int], dict[str, np.ndarray]],
    tri: tuple[int, int, int],
) -> tuple[float, tuple[float, float, float]]:
    """Optimize station-side splitting for edge-first closure on one triangle.

    Variables are xa, xb, xc:
      station a sends xa to ab and 1-xa to ac,
      station b sends xb to ab and 1-xb to bc,
      station c sends xc to bc and 1-xc to ac.
    """
    a, b, c = tri
    eab = edge_arrays[(a, b)]
    ebc = edge_arrays[(b, c)]
    eac = edge_arrays[(a, c)]

    grid = np.linspace(0.02, 0.98, 49)

    def score(xa: float, xb: float, xc: float) -> float:
        fab = edge_fisher_from_arrays(eab, xa, xb)
        fbc = edge_fisher_from_arrays(ebc, 1.0 - xb, xc)
        fac = edge_fisher_from_arrays(eac, 1.0 - xa, 1.0 - xc)
        return closure_fisher_from_edges(fab, fbc, fac)

    best = (0.5, 0.5, 0.5)
    best_score = score(*best)
    # Coordinate-search is enough here because this is only a diagnostic table.
    for seed in [(0.5, 0.5, 0.5), (0.25, 0.5, 0.75), (0.75, 0.5, 0.25), (0.2, 0.8, 0.5), (0.8, 0.2, 0.5)]:
        xa, xb, xc = seed
        local_best = score(xa, xb, xc)
        for _ in range(8):
            candidates = [(score(v, xb, xc), v) for v in grid]
            local_best, xa = max(candidates, key=lambda item: item[0])
            candidates = [(score(xa, v, xc), v) for v in grid]
            local_best, xb = max(candidates, key=lambda item: item[0])
            candidates = [(score(xa, xb, v), v) for v in grid]
            local_best, xc = max(candidates, key=lambda item: item[0])
        if local_best > best_score:
            best_score = local_best
            best = (xa, xb, xc)
    return best_score, best


def main() -> None:
    configure_fig3_physics()
    case = rml_cases.load_maunakea_plus3_case()
    stations, diameters, station_names, _ = aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n_station = len(stations)
    edges = base.edge_list(n_station)
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    n_closure = q_basis.shape[1]
    closure_rank_share = min(1.0, (n_station - 1.0) / n_closure)
    split_global = 1.0 / (n_station - 1.0)

    with ngc.patched_source(ngc.NGC4151):
        truth, _axis_uas = base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)

    hub_dist = np.linalg.norm(stations - hub, axis=1)
    effective_hub_dist = aug.FIBER_LENGTH_SCALE * hub_dist
    station_eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n_station, aug.MODE_FALSE_POSITIVE, dtype=float)
    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)

    fq_full_raw = np.zeros((n_closure, n_closure), dtype=float)
    edge_arrays: dict[tuple[int, int], dict[str, list[float]]] = {
        edge: {"k": [], "ai": [], "aj": [], "pair": []} for edge in edges
    }

    lam_edges_nm = np.arange(aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM, aug.LAMBDA_STEP_NM)
    lam_edges_nm[-1] = aug.LAMBDA_MAX_NM
    for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
        lam = math.sqrt(lo_nm * hi_nm) * 1e-9
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
        freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
        df = freq_hi - freq_lo
        total_modes = aug.EXPOSURE_S * aug.OBSERVING_DAYS * df
        u_station = aug.station_u_modes(freq, diameters)
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            lam,
            latitude_deg=case.latitude_deg,
            declination_deg=ngc.NGC4151.dec_deg,
        )
        ai_station = station_eta * u_station + station_noise
        for uu, vv in zip(uu_rows, vv_rows):
            vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
            nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
            fq_full_raw += total_modes * aug.noisy_closure_fisher_station_u(
                vtrue,
                station_eta,
                station_noise,
                u_station,
                q_basis,
                edges,
            )
            for edge_index, (i, j) in enumerate(edges):
                arrays = edge_arrays[(i, j)]
                arrays["k"].append(
                    total_modes
                    * 4.0
                    * station_eta[i]
                    * station_eta[j]
                    * u_station[i]
                    * u_station[j]
                    * nu_eff[edge_index] ** 2
                )
                arrays["ai"].append(ai_station[i])
                arrays["aj"].append(ai_station[j])
                arrays["pair"].append(aug.PAIR_FALSE_POSITIVE)

    edge_arrays_np = {
        edge: {key: np.asarray(values, dtype=float) for key, values in payload.items()}
        for edge, payload in edge_arrays.items()
    }

    # Precompute edge-first Fisher for two common station-side budgets.
    edge_fisher_all = {
        edge: edge_fisher_from_arrays(arrays, 1.0, 1.0) for edge, arrays in edge_arrays_np.items()
    }
    edge_fisher_global_split = {
        edge: edge_fisher_from_arrays(arrays, split_global, split_global)
        for edge, arrays in edge_arrays_np.items()
    }

    rows = []
    for tri in itertools.combinations(range(n_station), 3):
        i, j, k = tri
        c_global = edge_vector_for_triangle(edges, edge_to_index, tri)
        f_full_raw = scalar_fisher_for_closure(fq_full_raw, q_basis, c_global)
        f_full_sched = closure_rank_share * f_full_raw
        f_full_scalar_raw = scalar_sld_fisher_for_closure_path(fq_full_raw, q_basis, c_global)
        f_full_scalar_sched = closure_rank_share * f_full_scalar_raw

        local_edges = base.edge_list(3)
        local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, 3))
        local_c = edge_vector_for_triangle(local_edges, {e: idx for idx, e in enumerate(local_edges)}, (0, 1, 2))
        fq_tri = np.zeros((local_q.shape[1], local_q.shape[1]), dtype=float)
        # Reuse the edge-array integrated Fisher for edge-first, but recompute the
        # three-mode SLD with the same wavelength/hour-angle source coherences.
        for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
            lam = math.sqrt(lo_nm * hi_nm) * 1e-9
            freq = base.C_LIGHT / lam
            freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
            df = freq_hi - freq_lo
            total_modes = aug.EXPOSURE_S * aug.OBSERVING_DAYS * df
            u_local = aug.station_u_modes(freq, diameters[[i, j, k]])
            local_baselines = np.asarray(
                [stations[j] - stations[i], stations[k] - stations[i], stations[k] - stations[j]],
                dtype=float,
            )
            uu_rows, vv_rows = project_enu_baselines(
                local_baselines,
                hour_angles,
                lam,
                latitude_deg=case.latitude_deg,
                declination_deg=ngc.NGC4151.dec_deg,
            )
            for uu, vv in zip(uu_rows, vv_rows):
                vlocal = base.interp_vis(vgrid, uv_axis, uu, vv)
                fq_tri += total_modes * aug.noisy_closure_fisher_station_u(
                    vlocal,
                    station_eta[[i, j, k]],
                    station_noise[[i, j, k]],
                    u_local,
                    local_q,
                    local_edges,
                )
        f_3mode = scalar_fisher_for_closure(fq_tri, local_q, local_c)

        f_edge_global = closure_fisher_from_edges(
            edge_fisher_global_split[(i, j)],
            edge_fisher_global_split[(j, k)],
            edge_fisher_global_split[(i, k)],
        )
        f_edge_all = closure_fisher_from_edges(
            edge_fisher_all[(i, j)],
            edge_fisher_all[(j, k)],
            edge_fisher_all[(i, k)],
        )
        f_edge_opt, split_opt = optimize_triangle_split(edge_arrays_np, tri)

        edge_snrs_all = [
            math.sqrt(edge_fisher_all[(i, j)]),
            math.sqrt(edge_fisher_all[(j, k)]),
            math.sqrt(edge_fisher_all[(i, k)]),
        ]
        edge_snr_minmax = min(edge_snrs_all) / max(edge_snrs_all)
        xa, xb, xc = split_opt
        rows.append(
            {
                "loop": f"{i+1}-{j+1}-{k+1}",
                "stations": f"{station_names[i]} | {station_names[j]} | {station_names[k]}",
                "edge_snr_allphot_ij": edge_snrs_all[0],
                "edge_snr_allphot_jk": edge_snrs_all[1],
                "edge_snr_allphot_ik": edge_snrs_all[2],
                "edge_snr_min_over_max": edge_snr_minmax,
                "global_edge_split_per_station": split_global,
                "tri_opt_station_i_to_ij": xa,
                "tri_opt_station_i_to_ik": 1.0 - xa,
                "tri_opt_station_j_to_ij": xb,
                "tri_opt_station_j_to_jk": 1.0 - xb,
                "tri_opt_station_k_to_jk": xc,
                "tri_opt_station_k_to_ik": 1.0 - xc,
                "F_fullN_raw": f_full_raw,
                "F_fullN_scheduled": f_full_sched,
                "F_fullN_scalarSLD_raw": f_full_scalar_raw,
                "F_fullN_scalarSLD_scheduled": f_full_scalar_sched,
                "F_3mode_direct": f_3mode,
                "F_edge_first_global_uniform": f_edge_global,
                "F_edge_first_allphot_no_split": f_edge_all,
                "F_edge_first_tri_opt_split": f_edge_opt,
                "rms_fullN_raw_rad": 1.0 / math.sqrt(max(f_full_raw, 1e-300)),
                "rms_fullN_scheduled_rad": 1.0 / math.sqrt(max(f_full_sched, 1e-300)),
                "rms_fullN_scalarSLD_raw_rad": 1.0 / math.sqrt(max(f_full_scalar_raw, 1e-300)),
                "rms_fullN_scalarSLD_scheduled_rad": 1.0 / math.sqrt(max(f_full_scalar_sched, 1e-300)),
                "rms_3mode_direct_rad": 1.0 / math.sqrt(max(f_3mode, 1e-300)),
                "rms_edge_global_uniform_rad": 1.0 / math.sqrt(max(f_edge_global, 1e-300)),
                "rms_edge_tri_opt_rad": 1.0 / math.sqrt(max(f_edge_opt, 1e-300)),
                "snr_3mode_over_fullN_raw": math.sqrt(f_3mode / f_full_raw) if f_full_raw > 0 else math.nan,
                "snr_3mode_over_fullN_scheduled": math.sqrt(f_3mode / f_full_sched) if f_full_sched > 0 else math.nan,
                "snr_3mode_over_fullN_scalarSLD_raw": math.sqrt(f_3mode / f_full_scalar_raw)
                if f_full_scalar_raw > 0
                else math.nan,
                "snr_3mode_over_fullN_scalarSLD_scheduled": math.sqrt(f_3mode / f_full_scalar_sched)
                if f_full_scalar_sched > 0
                else math.nan,
                "snr_triopt_edge_over_fullN_raw": math.sqrt(f_edge_opt / f_full_raw) if f_full_raw > 0 else math.nan,
                "snr_triopt_edge_over_fullN_scheduled": math.sqrt(f_edge_opt / f_full_sched) if f_full_sched > 0 else math.nan,
                "snr_triopt_edge_over_fullN_scalarSLD_raw": math.sqrt(f_edge_opt / f_full_scalar_raw)
                if f_full_scalar_raw > 0
                else math.nan,
                "snr_triopt_edge_over_fullN_scalarSLD_scheduled": math.sqrt(f_edge_opt / f_full_scalar_sched)
                if f_full_scalar_sched > 0
                else math.nan,
                "snr_3mode_over_edge_global_uniform": math.sqrt(f_3mode / f_edge_global) if f_edge_global > 0 else math.nan,
                "snr_triopt_edge_over_edge_global_uniform": math.sqrt(f_edge_opt / f_edge_global)
                if f_edge_global > 0
                else math.nan,
                "best_local_snr_over_fullN_scheduled": math.sqrt(max(f_3mode, f_edge_opt) / f_full_sched)
                if f_full_sched > 0
                else math.nan,
                "best_local_label": "3mode_direct" if f_3mode >= f_edge_opt else "edge_tri_opt",
            }
        )

    csv_path = OUTDIR / "loop_local_vs_global_direct_hawaii3_ngc4151.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def pct(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
        }

    summary = {
        "case": case.key,
        "n_station": n_station,
        "n_baseline": len(edges),
        "n_independent_closure": n_closure,
        "n_triangular_loops": len(rows),
        "global_edge_split_per_station": split_global,
        "direct_closure_rank_share_used_in_fig3": closure_rank_share,
        "station_names": station_names,
        "station_eta": station_eta.tolist(),
        "hub_km": list(case.hub_km),
        "physics": {
            "source": "NGC 4151 patched source",
            "observing_days": aug.OBSERVING_DAYS,
            "n_time_windows": aug.N_TIME_WINDOWS,
            "exposure_s": aug.EXPOSURE_S,
            "lambda_nm": [aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM, aug.LAMBDA_STEP_NM],
            "fiber_loss_db_per_km": aug.FIBER_LOSS_DB_PER_KM,
            "fiber_length_scale": aug.FIBER_LENGTH_SCALE,
            "mode_false_positive": aug.MODE_FALSE_POSITIVE,
            "pair_false_positive": aug.PAIR_FALSE_POSITIVE,
        },
        "snr_ratio_percentiles": {
            "3mode_over_fullN_raw": pct([r["snr_3mode_over_fullN_raw"] for r in rows]),
            "3mode_over_fullN_scheduled": pct([r["snr_3mode_over_fullN_scheduled"] for r in rows]),
            "3mode_over_fullN_scalarSLD_raw": pct([r["snr_3mode_over_fullN_scalarSLD_raw"] for r in rows]),
            "3mode_over_fullN_scalarSLD_scheduled": pct(
                [r["snr_3mode_over_fullN_scalarSLD_scheduled"] for r in rows]
            ),
            "triopt_edge_over_fullN_raw": pct([r["snr_triopt_edge_over_fullN_raw"] for r in rows]),
            "triopt_edge_over_fullN_scheduled": pct([r["snr_triopt_edge_over_fullN_scheduled"] for r in rows]),
            "triopt_edge_over_fullN_scalarSLD_raw": pct([r["snr_triopt_edge_over_fullN_scalarSLD_raw"] for r in rows]),
            "triopt_edge_over_fullN_scalarSLD_scheduled": pct(
                [r["snr_triopt_edge_over_fullN_scalarSLD_scheduled"] for r in rows]
            ),
            "3mode_over_edge_global_uniform": pct([r["snr_3mode_over_edge_global_uniform"] for r in rows]),
            "triopt_edge_over_edge_global_uniform": pct([r["snr_triopt_edge_over_edge_global_uniform"] for r in rows]),
            "best_local_over_fullN_scheduled": pct([r["best_local_snr_over_fullN_scheduled"] for r in rows]),
        },
        "example_most_symmetric": min(rows, key=lambda r: abs(r["edge_snr_min_over_max"] - 1.0)),
        "example_most_asymmetric": min(rows, key=lambda r: r["edge_snr_min_over_max"]),
        "csv": str(csv_path),
    }
    json_path = OUTDIR / "loop_local_vs_global_direct_hawaii3_ngc4151_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(csv_path)
    print(json_path)
    print(json.dumps(summary["snr_ratio_percentiles"], indent=2))
    print("most symmetric", summary["example_most_symmetric"]["loop"], summary["example_most_symmetric"]["stations"])
    print("most asymmetric", summary["example_most_asymmetric"]["loop"], summary["example_most_asymmetric"]["stations"])


if __name__ == "__main__":
    main()
