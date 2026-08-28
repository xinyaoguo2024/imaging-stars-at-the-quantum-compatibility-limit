from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import plot_prl_broadband_clean as base
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
TABLE_DIR = ROOT / "output" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FIG3_STATS_PATH = (
    ROOT
    / "output"
    / "figures"
    / "prl_mono_uniform_stack_stats_fig3_u50v50_lowuv15_near10_5_ydown1p5_hub_m3_m4_30d_loss015_6panel.json"
)


def triangle_direct_fisher(
    g12: float,
    g23: float,
    g31: float,
    s1: float,
    s2: float,
    s3: float,
) -> float:
    """Exact three-station SLD FI density for the physical closure phase."""
    matrix = np.array(
        [
            [s1 + s2, -g31, -g23],
            [-g31, s2 + s3, -g12],
            [-g23, -g12, s3 + s1],
        ],
        dtype=float,
    )
    rhs = (2.0 / 3.0) * np.array([g12, g23, g31], dtype=float)
    x, y, z = np.linalg.solve(matrix, rhs)
    return float((2.0 / 3.0) * (g12 * x + g23 * y + g31 * z))


def latex_number(value: float, *, sig: int = 2) -> str:
    """Compact table formatting with TeX scientific notation for large/small values."""
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 100.0 or abs_value < 1e-2:
        exponent = int(np.floor(np.log10(abs_value)))
        mantissa = value / 10.0**exponent
        return rf"${mantissa:.{sig}g}\!\times\!10^{{{exponent}}}$"
    if abs_value >= 10.0:
        return f"{value:.3g}"
    return f"{value:.{sig + 1}g}"


def main() -> None:
    stats = json.loads(FIG3_STATS_PATH.read_text())
    layout_path = ROOT / stats["station_layout_file"]
    layout = json.loads(layout_path.read_text())
    stations_km = np.array(layout["stations_km"], dtype=float)
    hub_km = np.array(stats["hub_km"], dtype=float)

    n_station = len(stations_km)
    edges = base.edge_list(n_station)
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    n_closure = w_basis.shape[1]
    closure_rank_share = min(1.0, (n_station - 1.0) / n_closure)

    fiber_length_scale = float(stats["fiber_length_scale"])
    fiber_loss_db_per_km = float(stats["fiber_loss_db_per_km"])
    hub_distances_km = np.linalg.norm(stations_km - hub_km, axis=1)
    effective_hub_distances_km = fiber_length_scale * hub_distances_km
    station_link_eff = 10.0 ** (-fiber_loss_db_per_km * effective_hub_distances_km / 10.0)
    station_channel_noise = np.full_like(station_link_eff, base.MODE_FALSE_POSITIVE)
    baseline_link_eff = np.array([np.sqrt(station_link_eff[i] * station_link_eff[j]) for i, j in edges])
    baseline_load_eff = np.array([(station_link_eff[i] + station_link_eff[j]) / 2.0 for i, j in edges])
    baseline_noise_eff = np.array([(station_channel_noise[i] + station_channel_noise[j]) / 2.0 for i, j in edges])

    split_fraction = 1.0 / (n_station - 1.0)
    edge_split_coherence_eff = split_fraction * baseline_link_eff
    edge_split_load_eff = 2.0 * split_fraction * baseline_load_eff
    edge_split_channel_noise = 2.0 * split_fraction * baseline_noise_eff
    direct_channel_noise = float(np.sum(station_channel_noise))

    n_pix = int(stats["n_pix"])
    half_width_uas = float(stats["half_width_uas"])
    fov_rad = 2.0 * half_width_uas * base.UAS_TO_RAD
    truth, _ = base.make_source(n_pix, half_width_uas)
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)

    n_time = int(stats["n_time_windows"])
    mag_ab = base.SOURCE_AB_MAG
    observing_days = int(stats["observing_days"])
    exposure_s = float(stats["exposure_s"])
    exposure_gap_s = float(stats["exposure_gap_s"])

    lam_edges_nm = np.arange(
        float(stats["lambda_min_nm"]),
        float(stats["lambda_max_nm"]) + 0.5 * float(stats["lambda_step_nm"]),
        float(stats["lambda_step_nm"]),
    )
    lam_edges_nm[-1] = float(stats["lambda_max_nm"])
    lam_edges = lam_edges_nm * 1e-9
    lam_centers = np.sqrt(lam_edges[:-1] * lam_edges[1:])
    n_lambda = len(lam_centers)
    hour_angles = realnight_hour_angles(n_time, exposure_s, exposure_gap_s)

    j_edge_total = np.zeros((n_closure, n_closure))
    j_direct_total = np.zeros((n_closure, n_closure))
    loop_fsep_total = np.zeros(n_closure)
    loop_fopt_total = np.zeros(n_closure)
    loop_edge_fi_total = np.zeros((n_closure, 3))

    baseline_vectors_km = np.array([stations_km[j] - stations_km[i] for i, j in edges])
    edge_index = {edge: index for index, edge in enumerate(edges)}
    loop_defs = []
    for i in range(1, n_station):
        for j in range(i + 1, n_station):
            loop_defs.append(
                {
                    "label": (0, i, j),
                    "edge_indices": (edge_index[(0, i)], edge_index[(i, j)], edge_index[(0, j)]),
                }
            )

    for lam, lam_lo, lam_hi in zip(lam_centers, lam_edges[:-1], lam_edges[1:]):
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_mode = base.mode_occupation_ab(mag_ab, freq, diameter_m=base.TELESCOPE_DIAMETER_M)
        total_modes = exposure_s * observing_days * df
        uu_rows, vv_rows = project_enu_baselines(
            baseline_vectors_km,
            hour_angles,
            lam,
            latitude_deg=float(stats["array_latitude_deg"]),
            declination_deg=float(stats["source_declination_deg"]),
        )

        for u, v in zip(uu_rows, vv_rows):
            vtrue = base.interp_vis(vgrid, uv_axis, u, v)
            amp = np.abs(vtrue)
            nu_eff = np.clip(amp, 1e-4, 0.98)
            station_load = station_link_eff * u_mode + station_channel_noise

            coherent_signal_split = edge_split_coherence_eff * u_mode
            noise_load_split = edge_split_load_eff * u_mode
            edge_fisher = (
                total_modes
                * 4.0
                * coherent_signal_split**2
                * nu_eff**2
                / (noise_load_split + edge_split_channel_noise)
            )
            j_edge_total += q_basis.T @ (edge_fisher[:, None] * q_basis)

            j_direct_total += (
                total_modes
                * base.noisy_closure_fisher_from_station_modes(
                    vtrue, station_link_eff, station_channel_noise, u_mode, q_basis, edges
                )
            )

            for loop_index, loop_def in enumerate(loop_defs):
                a, b, c_station = loop_def["label"]
                e_ab, e_bc, e_ac = loop_def["edge_indices"]
                g_ab = u_mode * np.sqrt(station_link_eff[a] * station_link_eff[b]) * nu_eff[e_ab]
                g_bc = u_mode * np.sqrt(station_link_eff[b] * station_link_eff[c_station]) * nu_eff[e_bc]
                g_ca = u_mode * np.sqrt(station_link_eff[c_station] * station_link_eff[a]) * nu_eff[e_ac]

                f_edges = np.array(
                    [
                        4.0
                        * (split_fraction * g_ab) ** 2
                        / (split_fraction * (station_load[a] + station_load[b])),
                        4.0
                        * (split_fraction * g_bc) ** 2
                        / (split_fraction * (station_load[b] + station_load[c_station])),
                        4.0
                        * (split_fraction * g_ca) ** 2
                        / (split_fraction * (station_load[c_station] + station_load[a])),
                    ],
                    dtype=float,
                )
                f_edges = np.maximum(f_edges, 1e-300)
                loop_fsep_total[loop_index] += total_modes / np.sum(1.0 / f_edges)
                loop_fopt_total[loop_index] += total_modes * triangle_direct_fisher(
                    g_ab,
                    g_bc,
                    g_ca,
                    station_load[a],
                    station_load[b],
                    station_load[c_station],
                )
                loop_edge_fi_total[loop_index] += total_modes * f_edges

    inv_edge = np.linalg.pinv(0.5 * (j_edge_total + j_edge_total.T), rcond=1e-14)
    inv_direct = np.linalg.pinv(0.5 * (j_direct_total + j_direct_total.T), rcond=1e-14)

    rows = []
    labels = [(0, i, j) for i in range(1, n_station) for j in range(i + 1, n_station)]
    for index, (label, w_loop) in enumerate(zip(labels, w_basis.T), start=1):
        beta = q_basis.T @ w_loop
        var_edge = float(beta.T @ inv_edge @ beta)
        var_direct = float(beta.T @ inv_direct @ beta)
        snr_gain_full_qfi = float(np.sqrt(var_edge / var_direct))
        sigma_direct_scheduled = float(np.sqrt(var_direct / closure_rank_share))
        snr_gain_full_scheduled = snr_gain_full_qfi * np.sqrt(closure_rank_share)
        loop_fsep = max(float(loop_fsep_total[index - 1]), 1e-300)
        loop_fopt = max(float(loop_fopt_total[index - 1]), 1e-300)
        snr_gain_loop = float(np.sqrt(loop_fopt / loop_fsep))
        snr_gain_loop_scheduled = snr_gain_loop * np.sqrt(closure_rank_share)
        sorted_edge_fi = np.sort(loop_edge_fi_total[index - 1])[::-1]
        chi2 = float(np.sqrt(sorted_edge_fi[1] / sorted_edge_fi[0]))
        chi3 = float(np.sqrt(sorted_edge_fi[2] / sorted_edge_fi[0]))
        rows.append(
            {
                "index": index,
                "loop": f"S{label[0] + 1}-S{label[1] + 1}-S{label[2] + 1}",
                "chi2": chi2,
                "chi3": chi3,
                "sigma_loop_edge_rad": float(1.0 / np.sqrt(loop_fsep)),
                "sigma_loop_direct_rad": float(1.0 / np.sqrt(loop_fopt)),
                "sigma_loop_direct_scheduled_rad": float(1.0 / np.sqrt(loop_fopt * closure_rank_share)),
                "snr_gain_loop_qfi": snr_gain_loop,
                "snr_gain_loop_scheduled": snr_gain_loop_scheduled,
                "variance_reduction_loop_scheduled": snr_gain_loop_scheduled**2,
                "sigma_full_marginal_edge_rad": float(np.sqrt(var_edge)),
                "sigma_full_marginal_direct_rad": float(np.sqrt(var_direct)),
                "sigma_full_marginal_direct_scheduled_rad": sigma_direct_scheduled,
                "snr_gain_full_subspace_qfi": snr_gain_full_qfi,
                "snr_gain_full_subspace_scheduled": snr_gain_full_scheduled,
            }
        )

    payload = {
        "definition": "loop-local three-station SLD gain for root-loop closure phases in the eight-station Fig. 3 benchmark",
        "n_stations": n_station,
        "n_baselines": len(edges),
        "n_closure": n_closure,
        "closure_rank_share": closure_rank_share,
        "split_fraction": split_fraction,
        "n_time": n_time,
        "n_lambda": n_lambda,
        "lambda_step_nm": float(stats["lambda_step_nm"]),
        "mag_ab": mag_ab,
        "telescope_diameter_m": base.TELESCOPE_DIAMETER_M,
        "fiber_length_scale": fiber_length_scale,
        "fiber_loss_db_per_km": fiber_loss_db_per_km,
        "observing_days": observing_days,
        "exposure_s": exposure_s,
        "exposure_gap_s": exposure_gap_s,
        "station_layout_file": stats["station_layout_file"],
        "hub_km": hub_km.tolist(),
        "array_latitude_deg": float(stats["array_latitude_deg"]),
        "source_declination_deg": float(stats["source_declination_deg"]),
        "channel_noise_model": "quantum_limited_station_to_hub_fibre_output_referred",
        "station_channel_noise_min": float(np.min(station_channel_noise)),
        "station_channel_noise_max": float(np.max(station_channel_noise)),
        "edge_split_channel_noise_percentiles": np.percentile(
            edge_split_channel_noise, [5, 25, 50, 75, 95]
        ).tolist(),
        "direct_channel_noise_sum": direct_channel_noise,
        "rows": rows,
    }

    json_path = TABLE_DIR / "closure_loop_gain_table.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        r"\begin{table*}[t]",
        r"\caption{\label{tab:loop_gains}",
        r"Loop-local closure-phase gain for the eight-station benchmark.",
        r"The root-loop observable is $\Gamma_{1ij}=\phi_{1i}+\phi_{ij}-\phi_{1j}$.",
        r"For each triangle we accumulate the exact three-station SLD Fisher information and the",
        r"edge-first harmonic-sum Fisher information over the same broadband samples as Fig.~3.",
        r"The columns $\chi_2,\chi_3$ are the two weaker accumulated baseline SNRs normalized to",
        r"the strongest baseline SNR in that loop; they are the loop-specific coordinates of",
        r"Fig.~2(a).  The single-loop gain is",
        r"$G_{\rm loop}=\sqrt{F_{\opt}^{\rm loop}/F_{\sep}^{\rm loop}}$.",
        r"The scheduled gain $G_{\rm sched}=G_{\rm loop}\sqrt{(N-1)/C}$ applies when the receiver",
        r"cycles through a complete 21-dimensional closure basis in three seven-direction settings.",
        r"Large gains occur when edge-first closure is harmonic-mean limited by a near-null baseline,",
        r"so the absolute error bars should be read together with the gain.  All entries are evaluated from the accumulated Fisher",
        rf"matrices over the same {n_time} ten-minute real-night samples, {n_lambda} wavelength bins of width {stats['lambda_step_nm']:g} nm, and {observing_days} observing nights",
        r"used in Fig.~3 of the main text, with an AB 12.8 source, $D=5~\mathrm{m}$ stations,",
        rf"${fiber_loss_db_per_km:.2f}~\mathrm{{dB/km}}$ fibre attenuation, and station-to-hub fibre lengths scaled by ${fiber_length_scale:.2f}$.  The edge-first benchmark includes simultaneous",
        r"all-baseline splitting with $f=1/(N-1)$.}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{ccccccc}",
        r"Index & Loop & $\chi_2$ & $\chi_3$ & $\sigma_{\sep}$ & $G_{\rm sched}$ & $G_{\rm loop}$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            rf"{row['index']} & {row['loop']} & "
            rf"{latex_number(row['chi2'])} & {latex_number(row['chi3'])} & "
            rf"{latex_number(row['sigma_loop_edge_rad'])} & "
            rf"{latex_number(row['snr_gain_loop_scheduled'])} & {latex_number(row['snr_gain_loop_qfi'])} \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{ruledtabular}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = TABLE_DIR / "closure_loop_gain_table.tex"
    tex_path.write_text("\n".join(lines))

    print(tex_path)
    print(json_path)


if __name__ == "__main__":
    main()
