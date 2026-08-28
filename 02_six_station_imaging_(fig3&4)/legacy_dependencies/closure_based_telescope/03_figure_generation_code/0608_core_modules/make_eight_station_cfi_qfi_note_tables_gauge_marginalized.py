from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import plot_prl_broadband_clean as base
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
TABLE_DIR = ROOT / "output" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

STATS_PATH = (
    ROOT
    / "output"
    / "figures"
    / "prl_mono_uniform_stack_stats_hub_m2_m5_30d_loss020_6panel.json"
)

LOOP_JSON = TABLE_DIR / "eight_station_ideal_loop_gain_loss020_gauge_marginalized.json"
LOOP_TEX = TABLE_DIR / "eight_station_ideal_loop_gain_loss020_gauge_marginalized.tex"
MISMATCH_JSON = TABLE_DIR / "eight_station_phase_frame_cfi_loss020_gauge_marginalized.json"
MISMATCH_TEX = TABLE_DIR / "eight_station_phase_frame_cfi_loss020_gauge_marginalized.tex"

SOURCE_MODEL_DESCRIPTION = (
    "3C273-like parametric image used for Fig. 3 with a NED median optical SED photon budget, "
    "45% compact accretion-disc continuum, 10% diffuse continuum, 40% BLR ring, "
    "and 5% inner optical jet/knot.  This is not a source-independent uniform-visibility benchmark."
)
LOCK_MONTE_CARLO_DRAWS = 16


def latex_number(value: float, *, sig: int = 2) -> str:
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


TRIANGLE_EDGES = [(0, 1), (1, 2), (2, 0)]
TRIANGLE_CLOSURE_WEIGHTS = np.full(3, 1.0 / 3.0, dtype=float)


def triangle_bmat_and_edge_derivatives(
    g12: float,
    g23: float,
    g31: float,
    s1: float,
    s2: float,
    s3: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Three-station one-click matrix and derivatives for the three edge phases.

    The third edge is oriented as 3 -> 1, so the protected phase is
    phi_12 + phi_23 + phi_31.  Keeping the edge phases explicit lets us remove
    the two station-gauge nuisance directions by a Schur complement.
    """
    bmat = np.array(
        [
            [s1, g12, g31],
            [g12, s2, g23],
            [g31, g23, s3],
        ],
        dtype=complex,
    )
    a12 = np.zeros((3, 3), dtype=complex)
    a23 = np.zeros((3, 3), dtype=complex)
    a31 = np.zeros((3, 3), dtype=complex)
    a12[0, 1] = 1j
    a12[1, 0] = -1j
    a23[1, 2] = 1j
    a23[2, 1] = -1j
    a31[2, 0] = 1j
    a31[0, 2] = -1j
    return bmat, [g12 * a12, g23 * a23, g31 * a31]


def triangle_effective_closure_derivative(
    bmat: np.ndarray,
    edge_derivatives: list[np.ndarray],
    *,
    eig_floor: float = 1e-12,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Efficient closure derivative after eliminating station-gauge nuisance.

    The returned derivative is the SLD-efficient tangent for the scalar closure
    phase Phi_cl = phi_12 + phi_23 + phi_31.  It is not the naive equal-share
    tangent when the edge Fisher matrix mixes closure and station-piston
    directions.
    """
    edge_fisher = base.qfi_from_bmat_derivatives(bmat, edge_derivatives, eig_floor=eig_floor)
    cut = base.edge_cut_basis(TRIANGLE_EDGES, 3)
    v = TRIANGLE_CLOSURE_WEIGHTS
    jgg = cut.T @ edge_fisher @ cut
    beta = np.linalg.pinv(jgg, rcond=1e-12) @ (cut.T @ edge_fisher @ v)
    efficient_weights = v - cut @ beta
    fisher = float(efficient_weights.T @ edge_fisher @ efficient_weights)
    deriv = sum(float(weight) * deriv for weight, deriv in zip(efficient_weights, edge_derivatives))
    return 0.5 * (deriv + deriv.conj().T), max(fisher, 0.0), efficient_weights, edge_fisher


def triangle_direct_fisher(g12: float, g23: float, g31: float, s1: float, s2: float, s3: float) -> float:
    bmat, edge_derivatives = triangle_bmat_and_edge_derivatives(g12, g23, g31, s1, s2, s3)
    _deriv, fisher, _weights, _edge_fisher = triangle_effective_closure_derivative(bmat, edge_derivatives)
    return fisher


def triangle_bmat_and_deriv(
    g12: float,
    g23: float,
    g31: float,
    s1: float,
    s2: float,
    s3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauge-nuisance-marginalized three-station closure derivative."""
    bmat, edge_derivatives = triangle_bmat_and_edge_derivatives(g12, g23, g31, s1, s2, s3)
    deriv, _fisher, _weights, _edge_fisher = triangle_effective_closure_derivative(bmat, edge_derivatives)
    return bmat, deriv


def triangle_cfi_with_frame_error(
    g12: float,
    g23: float,
    g31: float,
    s1: float,
    s2: float,
    s3: float,
    residual_phases: np.ndarray,
) -> tuple[float, float]:
    """Return CFI and ideal QFI for a three-station SLD sorter with phase-frame mismatch."""
    bmat, deriv = triangle_bmat_and_deriv(g12, g23, g31, s1, s2, s3)
    lmat, qfi = sld_from_b_and_d(bmat, deriv)
    if qfi <= 1e-30:
        return 0.0, 0.0
    _evals, evecs = np.linalg.eigh(lmat)
    projectors = [evecs[:, r : r + 1] @ evecs[:, r : r + 1].conj().T for r in range(3)]
    phase = np.diag(np.exp(1j * residual_phases))
    b_true = phase @ bmat @ phase.conj().T
    d_true = phase @ deriv @ phase.conj().T
    return projective_cfi(projectors, b_true, d_true), qfi


def triangle_cfi_ratio_from_covariance(
    rng: np.random.Generator,
    g12: float,
    g23: float,
    g31: float,
    s1: float,
    s2: float,
    s3: float,
    covariance_3: np.ndarray,
    *,
    draws: int = LOCK_MONTE_CARLO_DRAWS,
) -> tuple[float, float, float]:
    """Average CFI/QFI over a three-station residual gauge covariance."""
    bmat, deriv = triangle_bmat_and_deriv(g12, g23, g31, s1, s2, s3)
    lmat, qfi = sld_from_b_and_d(bmat, deriv)
    if qfi <= 1e-30:
        return 0.0, 0.0, 0.0
    _evals, evecs = np.linalg.eigh(lmat)
    projectors = [evecs[:, r : r + 1] @ evecs[:, r : r + 1].conj().T for r in range(3)]
    cov = 0.5 * (covariance_3 + covariance_3.T)
    evals, evecs_cov = np.linalg.eigh(cov)
    evals = np.maximum(evals, 0.0)
    transform = evecs_cov @ np.diag(np.sqrt(evals))
    cfi_values = []
    for _ in range(draws):
        residual = transform @ rng.normal(size=3)
        residual -= np.mean(residual)
        phase = np.diag(np.exp(1j * residual))
        b_true = phase @ bmat @ phase.conj().T
        d_true = phase @ deriv @ phase.conj().T
        cfi_values.append(projective_cfi(projectors, b_true, d_true))
    cfi = float(np.mean(cfi_values))
    return cfi, qfi, min(cfi / qfi, 1.0)


def one_click_matrices(
    visibilities: np.ndarray,
    station_efficiencies: np.ndarray,
    station_noise: np.ndarray,
    u_mode: float,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    n_station = len(station_efficiencies)
    bmat = np.diag(u_mode * station_efficiencies + station_noise).astype(complex)
    source_coherences = []
    for edge_index, (i, j) in enumerate(edges):
        coherence = u_mode * np.sqrt(station_efficiencies[i] * station_efficiencies[j]) * visibilities[edge_index]
        bmat[i, j] = coherence
        bmat[j, i] = np.conj(coherence)
        source_coherences.append(coherence)
    bmat = 0.5 * (bmat + bmat.conj().T)

    derivatives = []
    for coord in range(q_basis.shape[1]):
        deriv = np.zeros_like(bmat, dtype=complex)
        for edge_index, (i, j) in enumerate(edges):
            coeff = q_basis[edge_index, coord]
            if abs(coeff) < 1e-14:
                continue
            coherence = source_coherences[edge_index]
            deriv[i, j] += 1j * coeff * coherence
            deriv[j, i] += -1j * coeff * np.conj(coherence)
        derivatives.append(0.5 * (deriv + deriv.conj().T))
    return bmat, derivatives


def sld_from_b_and_d(bmat: np.ndarray, deriv: np.ndarray, eig_floor: float = 1e-12) -> tuple[np.ndarray, float]:
    evals, evecs = np.linalg.eigh(0.5 * (bmat + bmat.conj().T))
    evals = np.maximum(evals, eig_floor)
    deriv_p = evecs.conj().T @ deriv @ evecs
    l_p = np.zeros_like(deriv_p, dtype=complex)
    for m, lm in enumerate(evals):
        for n, ln in enumerate(evals):
            denom = lm + ln
            if denom > eig_floor:
                l_p[m, n] = 2.0 * deriv_p[m, n] / denom
    lmat = evecs @ l_p @ evecs.conj().T
    lmat = 0.5 * (lmat + lmat.conj().T)
    qfi = float(np.real(np.trace(bmat @ lmat @ lmat)))
    return lmat, max(qfi, 0.0)


def projective_cfi(projectors: list[np.ndarray], bmat: np.ndarray, deriv: np.ndarray) -> float:
    cfi = 0.0
    for proj in projectors:
        p = float(np.real(np.trace(proj @ bmat)))
        dp = float(np.real(np.trace(proj @ deriv)))
        if p > 1e-20:
            cfi += dp * dp / p
    return max(cfi, 0.0)


def setup_from_stats() -> dict:
    stats = json.loads(STATS_PATH.read_text())
    layout = json.loads((ROOT / stats["station_layout_file"]).read_text())
    stations_km = np.array(layout["stations_km"], dtype=float)
    hub_km = np.array(stats["hub_km"], dtype=float)
    n_station = len(stations_km)
    edges = base.edge_list(n_station)
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    closure_rank_share = min(1.0, (n_station - 1.0) / w_basis.shape[1])

    hub_distances_km = np.linalg.norm(stations_km - hub_km, axis=1)
    effective_hub_distances_km = float(stats["fiber_length_scale"]) * hub_distances_km
    station_link_eff = 10.0 ** (-float(stats["fiber_loss_db_per_km"]) * effective_hub_distances_km / 10.0)
    station_channel_noise = np.full_like(station_link_eff, base.MODE_FALSE_POSITIVE)

    n_pix = int(stats["n_pix"])
    half_width_uas = float(stats["half_width_uas"])
    fov_rad = 2.0 * half_width_uas * base.UAS_TO_RAD
    truth, _ = base.make_source(n_pix, half_width_uas)
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
    baseline_vectors_km = np.array([stations_km[j] - stations_km[i] for i, j in edges])

    lambda_edges_nm = np.arange(
        float(stats["lambda_min_nm"]),
        float(stats["lambda_max_nm"]) + 0.5 * float(stats["lambda_step_nm"]),
        float(stats["lambda_step_nm"]),
    )
    lambda_edges_nm[-1] = float(stats["lambda_max_nm"])
    lambda_edges_m = lambda_edges_nm * 1e-9
    lambda_centers_m = np.sqrt(lambda_edges_m[:-1] * lambda_edges_m[1:])
    hour_angles = realnight_hour_angles(
        int(stats["n_time_windows"]),
        float(stats["exposure_s"]),
        float(stats["exposure_gap_s"]),
    )

    return {
        "stats": stats,
        "stations_km": stations_km,
        "hub_km": hub_km,
        "edges": edges,
        "w_basis": w_basis,
        "q_basis": q_basis,
        "closure_rank_share": closure_rank_share,
        "station_link_eff": station_link_eff,
        "station_channel_noise": station_channel_noise,
        "baseline_vectors_km": baseline_vectors_km,
        "vgrid": vgrid,
        "uv_axis": uv_axis,
        "lambda_edges_m": lambda_edges_m,
        "lambda_centers_m": lambda_centers_m,
        "hour_angles": hour_angles,
    }


def accumulated_samples(context: dict, *, lambda_stride: int = 1, time_stride: int = 1):
    stats = context["stats"]
    for band_index, (lam, lam_lo, lam_hi) in enumerate(
        zip(context["lambda_centers_m"], context["lambda_edges_m"][:-1], context["lambda_edges_m"][1:])
    ):
        if band_index % lambda_stride:
            continue
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_mode = base.source_mode_occupation(freq, diameter_m=base.TELESCOPE_DIAMETER_M)
        total_modes = float(stats["exposure_s"]) * int(stats["observing_days"]) * df
        uu_rows, vv_rows = project_enu_baselines(
            context["baseline_vectors_km"],
            context["hour_angles"],
            lam,
            latitude_deg=float(stats["array_latitude_deg"]),
            declination_deg=float(stats["source_declination_deg"]),
        )
        for time_index, (u, v) in enumerate(zip(uu_rows, vv_rows)):
            if time_index % time_stride:
                continue
            vis = base.interp_vis(context["vgrid"], context["uv_axis"], u, v)
            yield band_index, time_index, vis, u_mode, total_modes


def incidence_matrix(edges: list[tuple[int, int]], n_station: int) -> np.ndarray:
    matrix = np.zeros((len(edges), n_station), dtype=float)
    for edge_index, (i, j) in enumerate(edges):
        matrix[edge_index, i] = 1.0
        matrix[edge_index, j] = -1.0
    return matrix


def compute_science_fast_lock_covariances(context: dict, split_fraction: float) -> list[np.ndarray]:
    """Science-only sample-averaged fast-loop covariance for each time window.

    The locking Fisher information is accumulated over the 400-800 nm spectral bins within one
    ten-minute sample, but not over the number of observing days.  This approximates the residual
    gauge uncertainty for each nightly sample before the science SLD score is accumulated.
    """
    stats = context["stats"]
    edges = context["edges"]
    n_station = len(context["station_link_eff"])
    n_time = int(stats["n_time_windows"])
    edge_lock_fisher = np.zeros((n_time, len(edges)), dtype=float)
    station_eff = context["station_link_eff"]
    station_noise = context["station_channel_noise"]

    for lam, lam_lo, lam_hi in zip(
        context["lambda_centers_m"],
        context["lambda_edges_m"][:-1],
        context["lambda_edges_m"][1:],
    ):
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_mode = base.source_mode_occupation(freq, diameter_m=base.TELESCOPE_DIAMETER_M)
        lock_modes = float(stats["exposure_s"]) * df
        uu_rows, vv_rows = project_enu_baselines(
            context["baseline_vectors_km"],
            context["hour_angles"],
            lam,
            latitude_deg=float(stats["array_latitude_deg"]),
            declination_deg=float(stats["source_declination_deg"]),
        )
        for time_index, (u, v) in enumerate(zip(uu_rows, vv_rows)):
            vis = base.interp_vis(context["vgrid"], context["uv_axis"], u, v)
            nu_eff = np.clip(np.abs(vis), 1e-4, 0.98)
            station_load = station_eff * u_mode + station_noise
            for edge_index, (i, j) in enumerate(edges):
                g = u_mode * np.sqrt(station_eff[i] * station_eff[j]) * nu_eff[edge_index]
                fisher_density = 4.0 * (split_fraction * g) ** 2 / (
                    split_fraction * (station_load[i] + station_load[j]) + base.PAIR_FALSE_POSITIVE
                )
                edge_lock_fisher[time_index, edge_index] += lock_modes * fisher_density

    inc = incidence_matrix(edges, n_station)
    covariances = []
    for time_index in range(n_time):
        laplacian = inc.T @ (edge_lock_fisher[time_index, :, None] * inc)
        evals, evecs = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
        inv_evals = np.zeros_like(evals)
        positive = evals > 1e-12
        inv_evals[positive] = 1.0 / evals[positive]
        covariance = (evecs * inv_evals) @ evecs.T
        covariance = 0.5 * (covariance + covariance.T)
        covariances.append(covariance)
    context["science_fast_lock_edge_fisher"] = edge_lock_fisher
    return covariances


def make_loop_gain_table(context: dict) -> dict:
    rng = np.random.default_rng(20260514)
    stats = context["stats"]
    edges = context["edges"]
    w_basis = context["w_basis"]
    q_basis = context["q_basis"]
    station_link_eff = context["station_link_eff"]
    station_noise = context["station_channel_noise"]
    n_station = len(station_link_eff)
    n_closure = w_basis.shape[1]
    closure_rank_share = context["closure_rank_share"]
    split_fraction = 1.0 / (n_station - 1.0)

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

    loop_fsep_total = np.zeros(n_closure)
    loop_fopt_total = np.zeros(n_closure)
    loop_edge_fi_total = np.zeros((n_closure, 3))

    j_edge_total = np.zeros((n_closure, n_closure))
    j_direct_total = np.zeros((n_closure, n_closure))

    baseline_link_eff = np.array([np.sqrt(station_link_eff[i] * station_link_eff[j]) for i, j in edges])
    baseline_load_eff = np.array([(station_link_eff[i] + station_link_eff[j]) / 2.0 for i, j in edges])
    baseline_noise_eff = np.array([(station_noise[i] + station_noise[j]) / 2.0 for i, j in edges])
    edge_split_coherence_eff = split_fraction * baseline_link_eff
    edge_split_load_eff = 2.0 * split_fraction * baseline_load_eff
    edge_split_channel_noise = 2.0 * split_fraction * baseline_noise_eff + base.PAIR_FALSE_POSITIVE
    lock_covariances = compute_science_fast_lock_covariances(context, split_fraction)
    loop_fcfi_total = np.zeros(n_closure)
    loop_retention_weight = np.zeros(n_closure)

    for _band_index, time_index, vis, u_mode, total_modes in accumulated_samples(context):
        nu_eff = np.clip(np.abs(vis), 1e-4, 0.98)
        station_load = station_link_eff * u_mode + station_noise

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
        j_direct_total += total_modes * base.noisy_closure_fisher_from_station_modes(
            vis, station_link_eff, station_noise, u_mode, q_basis, edges
        )

        for loop_index, loop_def in enumerate(loop_defs):
            a, b, c_station = loop_def["label"]
            e_ab, e_bc, e_ac = loop_def["edge_indices"]
            g_ab = u_mode * np.sqrt(station_link_eff[a] * station_link_eff[b]) * nu_eff[e_ab]
            g_bc = u_mode * np.sqrt(station_link_eff[b] * station_link_eff[c_station]) * nu_eff[e_bc]
            g_ca = u_mode * np.sqrt(station_link_eff[c_station] * station_link_eff[a]) * nu_eff[e_ac]
            f_edges = np.array(
                [
                    4.0 * (split_fraction * g_ab) ** 2 / (split_fraction * (station_load[a] + station_load[b])),
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
                g_ab, g_bc, g_ca, station_load[a], station_load[b], station_load[c_station]
            )
            cfi_density, qfi_density, _retention = triangle_cfi_ratio_from_covariance(
                rng,
                g_ab,
                g_bc,
                g_ca,
                station_load[a],
                station_load[b],
                station_load[c_station],
                lock_covariances[time_index][np.ix_([a, b, c_station], [a, b, c_station])],
            )
            loop_fcfi_total[loop_index] += total_modes * cfi_density
            loop_retention_weight[loop_index] += total_modes * qfi_density
            loop_edge_fi_total[loop_index] += total_modes * f_edges

    inv_edge = np.linalg.pinv(0.5 * (j_edge_total + j_edge_total.T), rcond=1e-14)
    inv_direct = np.linalg.pinv(0.5 * (j_direct_total + j_direct_total.T), rcond=1e-14)

    rows = []
    labels = [(0, i, j) for i in range(1, n_station) for j in range(i + 1, n_station)]
    split_snr_factor = float(np.sqrt(n_station - 1.0))
    for index, (label, w_loop) in enumerate(zip(labels, w_basis.T), start=1):
        beta = q_basis.T @ w_loop
        var_edge = float(beta.T @ inv_edge @ beta)
        var_direct = float(beta.T @ inv_direct @ beta)
        loop_fsep = max(float(loop_fsep_total[index - 1]), 1e-300)
        loop_fopt = max(float(loop_fopt_total[index - 1]), 1e-300)
        sorted_edge_fi = np.sort(loop_edge_fi_total[index - 1])[::-1]
        chi2 = float(np.sqrt(sorted_edge_fi[1] / sorted_edge_fi[0]))
        chi3 = float(np.sqrt(sorted_edge_fi[2] / sorted_edge_fi[0]))
        g_loop = float(np.sqrt(loop_fopt / loop_fsep))
        g_joint = float(g_loop / split_snr_factor)
        g_sched = g_loop * np.sqrt(closure_rank_share)
        loop_fcfi = max(float(loop_fcfi_total[index - 1]), 1e-300)
        g_cfi = float(np.sqrt(loop_fcfi / loop_fsep) * np.sqrt(closure_rank_share))
        retention = float(loop_fcfi / max(float(loop_retention_weight[index - 1]), 1e-300))
        rows.append(
            {
                "index": index,
                "loop": f"S{label[0] + 1}-S{label[1] + 1}-S{label[2] + 1}",
                "chi2": chi2,
                "chi3": chi3,
                "sigma_loop_edge_rad": float(1.0 / np.sqrt(loop_fsep)),
                "sigma_loop_direct_rad": float(1.0 / np.sqrt(loop_fopt)),
                "sigma_loop_direct_scheduled_rad": float(1.0 / np.sqrt(loop_fopt * closure_rank_share)),
                "sigma_loop_direct_cfi_scheduled_rad": float(1.0 / np.sqrt(loop_fcfi * closure_rank_share)),
                "snr_gain_loop_qfi": g_loop,
                "snr_gain_loop_joint_qfi": g_joint,
                "snr_gain_split_factor": split_snr_factor,
                "snr_gain_loop_scheduled": g_sched,
                "snr_gain_loop_cfi_scheduled": g_cfi,
                "fast_lock_cfi_retention": retention,
                "snr_gain_full_subspace_qfi": float(np.sqrt(var_edge / var_direct)),
                "snr_gain_full_subspace_scheduled": float(np.sqrt(var_edge / (var_direct / closure_rank_share))),
            }
        )

    payload = {
        "definition": "source-dependent ideal phase-frame loop-local SLD gain for root-loop closure phases",
        "source_model": SOURCE_MODEL_DESCRIPTION,
        "source_component_fractions": base.SOURCE_COMPONENT_FRACTIONS,
        "source_spectrum_name": base.SOURCE_SPECTRUM_NAME,
        "source_spectrum_note": base.SOURCE_SPECTRUM_NOTE,
        "source_spectrum_ned_points_400_800nm": base.SOURCE_SED_NED_POINT_COUNT,
        "source_half_width_uas": float(stats["half_width_uas"]),
        "stats_path": str(STATS_PATH.relative_to(ROOT)),
        "fiber_loss_db_per_km": float(stats["fiber_loss_db_per_km"]),
        "fiber_length_scale": float(stats["fiber_length_scale"]),
        "mode_false_positive": float(base.MODE_FALSE_POSITIVE),
        "pair_false_positive": float(base.PAIR_FALSE_POSITIVE),
        "observing_days": int(stats["observing_days"]),
        "n_time_windows": int(stats["n_time_windows"]),
        "exposure_s": float(stats["exposure_s"]),
        "lambda_step_nm": float(stats["lambda_step_nm"]),
        "n_closure": n_closure,
        "closure_rank_share": closure_rank_share,
        "split_snr_factor": split_snr_factor,
        "gain_factorization": (
            "G_loop is the total ideal loop-local gain over simultaneous split edge-first. "
            "G_split=sqrt(N-1) is the common no-splitting factor, and "
            "G_joint=G_loop/G_split is the intrinsic three-mode SLD gain over an unsplit edge-first closure."
        ),
        "g_cfi_model": (
        "G_CFI propagates a science-flux-equivalent fast-loop gauge covariance.  For each ten-minute "
            "time sample, J_psi^lock is built from the 400-800 nm summed split-baseline "
            "locking Fisher information with f=1/(N-1), without accumulating over observing days."
        ),
        "fast_lock_edge_fisher_percentiles": np.percentile(
            context["science_fast_lock_edge_fisher"], [5, 16, 50, 84, 95]
        ).tolist(),
        "rows": rows,
    }
    LOOP_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        r"\begingroup",
        r"\refstepcounter{table}\label{tab:eight_station_loop_gain_note}",
        r"\begin{center}",
        r"\noindent\textbf{Table~\thetable.}",
        r"Source-dependent ideal-workpoint loop-local SNR gains for the 21 root-loop closures in the eight-station benchmark.",
        r"The visibility amplitudes are evaluated from the same 3C-273-like parametric source used in Fig. 3,",
        r"and the photon budget uses the NED median optical SED rather than a flat-$F_\nu$ source.",
        r"This is not a source-independent uniform-visibility benchmark.",
        r"The table uses the same array and broadband sampling as Fig. 3, with pure fibre attenuation",
        rf"${float(stats['fiber_loss_db_per_km']):.1f}\,\mathrm{{dB/km}}$, independent station-mode",
        rf"background $p_{{\rm fp}}={float(base.MODE_FALSE_POSITIVE):.2f}$, and pair-combiner background",
        rf"${float(base.PAIR_FALSE_POSITIVE):.2f}$.",
        r"$G_{\rm loop}$ compares an individually optimized three-station SLD closure readout with",
        r"simultaneous split edge-first post-processing for that loop.  We also list",
        rf"$G_{{\rm joint}}=G_{{\rm loop}}/\sqrt{{N-1}}=G_{{\rm loop}}/{split_snr_factor:.3g}$,",
        r"which removes the common no-splitting factor and isolates the intrinsic three-mode SLD gain.",
        r"$G_{\rm CFI}$ includes the conservative",
        r"rank-sharing duty factor for cycling through the full $C=21$ closure space of an",
        r"$N=8$ array and further propagates the",
        r"sample-by-sample gauge uncertainty from a science-flux-equivalent fast loop, with",
        r"$J_\psi^{\rm lock}=A^{\mathsf T}\operatorname{diag}(F_{ij}^{\rm lock})A$ built by",
        r"summing the 400-800 nm split-baseline locking Fisher information within one ten-minute sample.",
        r"\par\medskip",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{clrrrrrrr}",
        r"\hline\hline",
        r"Index & Loop & $\chi_2$ & $\chi_3$ & $\sigma_{\rm edge}$ & $\sigma_{\rm dir,CFI}$ & $G_{\rm CFI}$ & $G_{\rm joint}$ & $G_{\rm loop}$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['index']} & {row['loop']} & "
            f"{latex_number(row['chi2'])} & {latex_number(row['chi3'])} & "
            f"{latex_number(row['sigma_loop_edge_rad'])} & {latex_number(row['sigma_loop_direct_cfi_scheduled_rad'])} & "
            f"{latex_number(row['snr_gain_loop_cfi_scheduled'])} & "
            f"{latex_number(row['snr_gain_loop_joint_qfi'])} & {latex_number(row['snr_gain_loop_qfi'])} \\\\"
        )
    lines.extend([r"\hline\hline", r"\end{tabular}%", r"}", r"\end{center}", r"\endgroup"])
    LOOP_TEX.write_text("\n".join(lines) + "\n")
    return payload


def make_mismatch_table(context: dict) -> dict:
    rng = np.random.default_rng(20260513)
    edges = context["edges"]
    q_basis = context["q_basis"]
    w_basis = context["w_basis"]
    station_eff = context["station_link_eff"]
    station_noise = context["station_channel_noise"]
    n_station = len(station_eff)
    loop_betas = [q_basis.T @ w_basis[:, i] for i in range(w_basis.shape[1])]
    sigmas = [0.05, 0.10, np.pi / 20.0, 0.20, np.pi / 10.0, 0.50, np.pi / 4.0]
    ratios = {sigma: [] for sigma in sigmas}
    weights = {sigma: [] for sigma in sigmas}

    # A representative subset is enough for the dimensionless phase-frame diagnostic.
    # The ideal loop-gain table above still uses the full broadband grid.
    for _band_index, _time_index, vis, u_mode, _total_modes in accumulated_samples(
        context, lambda_stride=4, time_stride=3
    ):
        bmat, derivs = one_click_matrices(vis, station_eff, station_noise, u_mode, q_basis, edges)
        for beta in loop_betas:
            deriv = sum(float(beta[k]) * derivs[k] for k in range(len(derivs)))
            lmat, qfi = sld_from_b_and_d(bmat, deriv)
            if qfi <= 1e-24:
                continue
            evals, evecs = np.linalg.eigh(lmat)
            projectors = [evecs[:, r : r + 1] @ evecs[:, r : r + 1].conj().T for r in range(n_station)]
            for sigma in sigmas:
                for _ in range(12):
                    delta = rng.normal(0.0, sigma, n_station)
                    delta -= np.mean(delta)
                    phase = np.diag(np.exp(1j * delta))
                    b_true = phase @ bmat @ phase.conj().T
                    d_true = phase @ deriv @ phase.conj().T
                    cfi = projective_cfi(projectors, b_true, d_true)
                    ratios[sigma].append(min(cfi / qfi, 1.0))
                    weights[sigma].append(qfi)

    rows = []
    for sigma in sigmas:
        arr = np.array(ratios[sigma], dtype=float)
        w = np.array(weights[sigma], dtype=float)
        weighted = float(np.sum(w * arr) / np.sum(w))
        row = {
            "sigma_station_rad": float(sigma),
            "sigma_station_deg": float(np.rad2deg(sigma)),
            "weighted_mean_cfi_over_qfi": weighted,
            "median_cfi_over_qfi": float(np.median(arr)),
            "p16_cfi_over_qfi": float(np.percentile(arr, 16)),
            "p84_cfi_over_qfi": float(np.percentile(arr, 84)),
            "p05_cfi_over_qfi": float(np.percentile(arr, 5)),
            "snr_retention_weighted": float(np.sqrt(max(weighted, 0.0))),
        }
        rows.append(row)

    payload = {
        "definition": "source-dependent CFI/QFI retained by a fixed ideal SLD eigenbasis under random station phase-frame residuals",
        "source_model": SOURCE_MODEL_DESCRIPTION,
        "source_component_fractions": base.SOURCE_COMPONENT_FRACTIONS,
        "source_spectrum_name": base.SOURCE_SPECTRUM_NAME,
        "source_spectrum_note": base.SOURCE_SPECTRUM_NOTE,
        "source_spectrum_ned_points_400_800nm": base.SOURCE_SED_NED_POINT_COUNT,
        "source_half_width_uas": float(context["stats"]["half_width_uas"]),
        "stats_path": str(STATS_PATH.relative_to(ROOT)),
        "fiber_loss_db_per_km": float(context["stats"]["fiber_loss_db_per_km"]),
        "diagnostic_grid": "lambda_stride=4 and time_stride=3, all 21 root loops, 12 random frames per sigma",
        "rows": rows,
    }
    MISMATCH_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        r"\begin{table}[t]",
        r"\caption{\label{tab:eight_station_cfi_loss_note}",
        r"Source-dependent classical Fisher information retained when the eight-mode SLD sorter is programmed at",
        r"the ideal phase frame but the true station frame has independent residual piston errors.",
        r"The common phase is removed.  The weighted mean uses the ideal scalar QFI as weight over",
        r"a representative subset of the broadband samples and all 21 root-loop SLDs for the",
        r"3C-273-like parametric source used in Fig. 3, with the same NED median optical SED photon budget.}",
        r"\centering",
        r"\begin{tabular}{rrrrr}",
        r"\hline\hline",
        r"$\sigma_\psi$ (rad) & $\sigma_\psi$ (deg) & weighted CFI/QFI & median CFI/QFI & SNR retained \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['sigma_station_rad']:.3f} & {row['sigma_station_deg']:.1f} & "
            f"{row['weighted_mean_cfi_over_qfi']:.3f} & {row['median_cfi_over_qfi']:.3f} & "
            f"{row['snr_retention_weighted']:.3f} \\\\"
        )
    lines.extend([r"\hline\hline", r"\end{tabular}", r"\end{table}"])
    MISMATCH_TEX.write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    context = setup_from_stats()
    loop_payload = make_loop_gain_table(context)
    mismatch_payload = make_mismatch_table(context)
    print(LOOP_JSON)
    print(LOOP_TEX)
    print(MISMATCH_JSON)
    print(MISMATCH_TEX)
    print(json.dumps({"loop_rows": len(loop_payload["rows"]), "mismatch_rows": mismatch_payload["rows"]}, indent=2))


if __name__ == "__main__":
    main()
