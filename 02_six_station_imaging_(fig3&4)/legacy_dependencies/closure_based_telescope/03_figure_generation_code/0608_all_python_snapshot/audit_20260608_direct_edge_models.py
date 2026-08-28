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


def scalar_info_from_cycle_fisher(fisher: np.ndarray, q_basis: np.ndarray, edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> float:
    d = q_basis.T @ split_sim.closure_edge_vector(edges, tri)
    cov = np.linalg.pinv(0.5 * (fisher + fisher.T), rcond=1e-12)
    var = float(d @ cov @ d)
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def run_direct_edge_model_audit() -> dict[str, object]:
    """Compare old loop-local proxies with full-array scalar Fisher forms."""
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
            "edge_three_edge_harmonic": 0.0,
            "edge_full_cycle_projection": 0.0,
            "direct_local_triangle_raw": 0.0,
            "direct_old_rankshare_local": 0.0,
            "direct_full_array_qfi_scalar": 0.0,
        }
        for _tri, loop, _kind in diag.LOOPS
    }
    edge_rel_diffs: list[float] = []

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
                edge_values = split_sim.edge_fisher_for_sample(
                    splits["edge_uniform"],
                    total_modes=total_modes,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    nu_eff=nu_eff,
                    edges=edges,
                )
                edge_cycle_fisher = base.closure_fisher_after_gauge_marginalization(
                    np.diag(edge_values),
                    q_basis,
                    edges,
                    n,
                )
                direct_full_fisher = total_modes * fig_run.aug.noisy_closure_fisher_station_u(
                    vtrue,
                    eta,
                    direct_noise,
                    u_station,
                    q_basis,
                    edges,
                )
                for tri, loop_label, _kind in diag.LOOPS:
                    f_edge_harm = diag.uniform_edge_scalar_loop_fisher_for_sample(
                        splits["edge_uniform"],
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )
                    f_edge_full = scalar_info_from_cycle_fisher(edge_cycle_fisher, q_basis, edges, tri)
                    f_direct_local = split_sim.core_triangle_direct_fisher_for_sample(
                        tri,
                        total_modes=total_modes,
                        vtrue=vtrue,
                        u_station=u_station,
                        eta=eta,
                        direct_noise=direct_noise,
                        edges=edges,
                        edge_to_index=edge_to_index,
                    )
                    f_direct_full = scalar_info_from_cycle_fisher(direct_full_fisher, q_basis, edges, tri)
                    accum[loop_label]["edge_three_edge_harmonic"] += f_edge_harm
                    accum[loop_label]["edge_full_cycle_projection"] += f_edge_full
                    accum[loop_label]["direct_local_triangle_raw"] += f_direct_local
                    accum[loop_label]["direct_old_rankshare_local"] += rank_share * f_direct_local
                    accum[loop_label]["direct_full_array_qfi_scalar"] += f_direct_full
                    edge_rel_diffs.append(abs(f_edge_full - f_edge_harm) / max(abs(f_edge_harm), 1e-300))

    rows: list[dict[str, object]] = []
    for _tri, loop_label, kind in diag.LOOPS:
        values = accum[loop_label]
        old_direct = values["direct_old_rankshare_local"]
        full_direct = values["direct_full_array_qfi_scalar"]
        edge_harm = values["edge_three_edge_harmonic"]
        edge_full = values["edge_full_cycle_projection"]
        rows.append(
            {
                "loop": loop_label,
                "loop_class": kind,
                "rank_share": float(rank_share),
                **values,
                "edge_full_over_harmonic_fisher": edge_full / max(edge_harm, 1e-300),
                "direct_full_qfi_over_old_rankshare_fisher": full_direct / max(old_direct, 1e-300),
                "direct_full_qfi_over_local_raw_fisher": full_direct / max(values["direct_local_triangle_raw"], 1e-300),
                "old_edge_rms_over_old_direct_rms": math.sqrt(old_direct / max(edge_harm, 1e-300)),
                "edge_rms_over_full_qfi_direct_rms": math.sqrt(full_direct / max(edge_harm, 1e-300)),
                "old_direct_rms_over_full_qfi_direct_rms": math.sqrt(full_direct / max(old_direct, 1e-300)),
            }
        )

    csv_path = OUT / "direct_edge_full_array_scalar_audit_200ms_minphase.csv"
    write_csv(csv_path, rows)
    return {
        "csv": str(csv_path),
        "rank_share": float(rank_share),
        "max_edge_full_vs_harmonic_rel_diff_per_sample": float(max(edge_rel_diffs) if edge_rel_diffs else 0.0),
        "rows": rows,
    }


if __name__ == "__main__":
    result = run_direct_edge_model_audit()
    print(result["csv"])
    print(f"rank_share={result['rank_share']}")
    print(f"max_edge_rel_diff={result['max_edge_full_vs_harmonic_rel_diff_per_sample']:.3e}")
    for row in result["rows"]:
        print(
            row["loop"],
            f"edge_full/harm={row['edge_full_over_harmonic_fisher']:.6g}",
            f"full_direct/old={row['direct_full_qfi_over_old_rankshare_fisher']:.6g}",
        )
