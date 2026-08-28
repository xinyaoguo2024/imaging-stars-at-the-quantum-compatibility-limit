from __future__ import annotations

import math

import numpy as np

from audit_20260608_common import OUT, write_csv

import hawaii3_compact_case
import make_fig2_current_seed_diagnostics as diag
import plot_prl_broadband_clean as base
import run_broad_plume_split_objective_rml as fig_run
import test_fig3_split_objective_imaging as split_sim
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


def triangle_half_split(n_station: int, tri: tuple[int, int, int]) -> np.ndarray:
    split = np.zeros((n_station, n_station), dtype=float)
    a, b, c = tri
    for i, j, k in ((a, b, c), (b, a, c), (c, a, b)):
        split[i, j] = 0.5
        split[i, k] = 0.5
    return split


def run_gain_factor_audit() -> dict[str, object]:
    fig_run.configure_good_runtime()
    case = fig_run.scale_remote_coordinates(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    splits = fig_run.make_split_matrices(case)
    split_sim.configure()
    fig_run.apply_sample_stress_runtime()

    stations, diameters, _names, _is_added = fig_run.aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n = len(stations)
    edges = base.edge_list(n)
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, n))
    rank_share = min(1.0, (n - 1.0) / q_basis.shape[1])
    fov_rad = 2.0 * fig_run.aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    effective_hub_dist = fig_run.aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-fig_run.aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n, fig_run.EPS_STATION_RUN, dtype=float)
    direct_noise = np.full(n, fig_run.EPS_STATION_RUN + fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    hour_angles = realnight_hour_angles(fig_run.aug.N_TIME_WINDOWS, fig_run.aug.EXPOSURE_S, fig_run.aug.EXPOSURE_GAP_S)

    lam_edges_nm = np.arange(
        fig_run.aug.LAMBDA_MIN_NM,
        fig_run.aug.LAMBDA_MAX_NM + 0.5 * fig_run.aug.LAMBDA_STEP_NM,
        fig_run.aug.LAMBDA_STEP_NM,
    )
    lam_edges_nm[-1] = fig_run.aug.LAMBDA_MAX_NM

    accum: dict[str, dict[str, float]] = {
        loop: {
            "edge_uniform": 0.0,
            "edge_triangle_half": 0.0,
            "direct_raw": 0.0,
        }
        for _tri, loop, _kind in diag.LOOPS
    }

    with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.GOOD_SOURCE):
        for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
            center_nm = float(math.sqrt(lo_nm * hi_nm))
            lam_m = center_nm * 1e-9
            freq = base.C_LIGHT / lam_m
            freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
            total_modes = fig_run.aug.EXPOSURE_S * fig_run.OBSERVING_DAYS * (freq_hi - freq_lo)
            u_station = fig_run.aug.station_u_modes(freq, diameters)
            band_truth, _axis = base.make_source_at_wavelength_nm(
                fig_run.aug.N_PIX,
                fig_run.aug.HALF_WIDTH_UAS,
                center_nm,
            )
            band_vgrid, band_uv_axis = base.visibility_grid(band_truth, fov_rad)
            uu_rows, vv_rows = project_enu_baselines(
                baselines,
                hour_angles,
                lam_m,
                latitude_deg=case.latitude_deg,
                declination_deg=fig_run.GOOD_SOURCE.dec_deg,
            )
            for uu, vv in zip(uu_rows, vv_rows):
                vtrue = base.interp_vis(band_vgrid, band_uv_axis, uu, vv)
                nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
                for tri, loop_label, _kind in diag.LOOPS:
                    half_split = triangle_half_split(n, tri)
                    accum[loop_label]["edge_uniform"] += diag.uniform_edge_scalar_loop_fisher_for_sample(
                        splits["edge_uniform"],
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )
                    accum[loop_label]["edge_triangle_half"] += diag.uniform_edge_scalar_loop_fisher_for_sample(
                        half_split,
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )
                    accum[loop_label]["direct_raw"] += split_sim.core_triangle_direct_fisher_for_sample(
                        tri,
                        total_modes=total_modes,
                        vtrue=vtrue,
                        u_station=u_station,
                        eta=eta,
                        direct_noise=direct_noise,
                        edges=edges,
                        edge_to_index=edge_to_index,
                    )

    rows: list[dict[str, object]] = []
    for tri, loop_label, kind in diag.LOOPS:
        f_uniform = accum[loop_label]["edge_uniform"]
        f_half = accum[loop_label]["edge_triangle_half"]
        f_direct = accum[loop_label]["direct_raw"]
        f_scheduled = rank_share * f_direct
        fig1_like_actual = math.sqrt(f_direct / f_half)
        fig3_scheduled_vs_uniform = math.sqrt(f_scheduled / f_uniform)
        resource_factor = fig3_scheduled_vs_uniform / fig1_like_actual
        split_penalty_fisher = f_half / f_uniform
        rows.append(
            {
                "loop": loop_label,
                "loop_class": kind,
                "rank_share": rank_share,
                "edge_uniform_fisher": f_uniform,
                "edge_triangle_half_fisher": f_half,
                "direct_raw_fisher": f_direct,
                "direct_scheduled_fisher": f_scheduled,
                "fig1_like_raw_direct_over_triangle_half_edge_snr": fig1_like_actual,
                "fig3_scheduled_direct_over_uniform_edge_snr": fig3_scheduled_vs_uniform,
                "resource_factor_fig3_over_fig1_like": resource_factor,
                "edge_half_over_uniform_fisher": split_penalty_fisher,
                "sqrt_rank_share": math.sqrt(rank_share),
                "sqrt_edge_half_over_uniform": math.sqrt(split_penalty_fisher),
            }
        )

    csv_path = OUT / "fig1_vs_fig3_gain_factor_decomposition_200ms_minphase.csv"
    write_csv(csv_path, rows)
    return {
        "csv": str(csv_path),
        "rank_share": rank_share,
        "rows": rows,
    }
