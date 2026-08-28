from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eht_style_amplitude_closure_rml as rml_cases
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from make_all_closure_global_benchmark_note_v2 import optimize_split
from make_all_closure_global_benchmark_note import (
    AllClosureBenchmark,
    EPS_DIRECT_EXTRA,
    EPS_PAIR,
    EPS_STATION,
    FIBER_LENGTH_SCALE,
    FIBER_LOSS_DB_PER_KM,
)
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
BUNDLE = Path(__file__).resolve().parents[2]
OUT = BUNDLE / "output" / "figures" / "fig3_split_objective_test"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ngc.NGC4151
RNG_SEED = int(os.environ.get("FIG2_RNG_SEED", os.environ.get("RNG_SEED", "20260528")))
RECON_MODE = "coarse_interp"
USE_TRUE_AMPLITUDE = os.environ.get("USE_TRUE_AMPLITUDE", "0") == "1"
PAIR_CORE_DIRECT_NOISE = os.environ.get("PAIR_CORE_DIRECT_NOISE", "1") != "0"
CORE_STATIONS = tuple(range(4))
CORE_JOINT_FRACTION = 0.5
SWITCHED_MODULAR_HYBRID_KEY = "switched_modular_hybrid"
MODULAR_LOOP_SPECS: list[dict] = []
MODULAR_SCHEDULE_FACTOR = 1.0
MODULAR_CLOSE_FACTOR: float | None = None
DIRECT_ROOT_WEIGHTS: dict[tuple[int, int, int], float] = {}
DIRECT_ROOT_WEIGHT_MODEL = "capacity_relaxed_scalar_schedule_default"


def configure() -> None:
    aug.OBSERVING_DAYS = 30
    aug.N_TIME_WINDOWS = 36
    aug.EXPOSURE_S = 600.0
    aug.EXPOSURE_GAP_S = 150.0
    aug.FIBER_LOSS_DB_PER_KM = FIBER_LOSS_DB_PER_KM
    aug.FIBER_LENGTH_SCALE = FIBER_LENGTH_SCALE
    aug.MODE_FALSE_POSITIVE = EPS_STATION
    aug.PAIR_FALSE_POSITIVE = EPS_PAIR
    aug.BASELINE_FALSE_POSITIVE = EPS_PAIR
    wt.SNR_BOOST = 1.0
    wt.OBSERVING_DAYS = aug.OBSERVING_DAYS
    wt.N_PIX = aug.N_PIX
    wt.BASELINE_FALSE_POSITIVE = EPS_PAIR
    wt.AMPLITUDE_MODE_FALSE_POSITIVE = EPS_STATION
    wt.AMPLITUDE_SIGMA_ABS = None


def edge_fisher_for_sample(
    split: np.ndarray,
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(len(edges), dtype=float)
    for idx, (i, j) in enumerate(edges):
        pi = split[i, j]
        pj = split[j, i]
        signal2 = pi * pj * eta[i] * eta[j] * u_station[i] * u_station[j]
        load = pi * (eta[i] * u_station[i] + station_noise[i]) + pj * (
            eta[j] * u_station[j] + station_noise[j]
        ) + EPS_PAIR
        out[idx] = total_modes * 4.0 * signal2 * nu_eff[idx] ** 2 / max(load, 1e-300)
    return out


def core4_remote_edge_fisher_for_sample(
    split: np.ndarray,
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Edge Fisher proxy for the core-four joint plus remote optimized split model."""
    out = np.zeros(len(edges), dtype=float)
    for idx, (i, j) in enumerate(edges):
        if i in CORE_STATIONS and j in CORE_STATIONS:
            pi = pj = CORE_JOINT_FRACTION
        else:
            pi = split[i, j]
            pj = split[j, i]
        signal2 = pi * pj * eta[i] * eta[j] * u_station[i] * u_station[j]
        load = pi * (eta[i] * u_station[i] + station_noise[i]) + pj * (
            eta[j] * u_station[j] + station_noise[j]
        ) + EPS_PAIR
        out[idx] = total_modes * 4.0 * signal2 * nu_eff[idx] ** 2 / max(load, 1e-300)
    return out


def core4_remote_global_split_fisher_for_sample(
    split: np.ndarray,
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    direct_noise: np.ndarray,
    nu_eff: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Cycle Fisher for close4 closure plus globally shared remote-edge splits."""
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    edge_fisher = core_direct_edge_fisher_for_sample(
        total_modes=total_modes,
        vtrue=vtrue,
        u_station=u_station,
        eta=eta,
        direct_noise=direct_noise,
        edges=edges,
        edge_to_index=edge_to_index,
    )
    for idx, (i, j) in enumerate(edges):
        if i in CORE_STATIONS and j in CORE_STATIONS:
            continue
        f_edge = edge_pair_fisher_for_sample(
            i,
            j,
            split[i, j],
            split[j, i],
            total_modes=total_modes,
            u_station=u_station,
            eta=eta,
            station_noise=station_noise,
            nu_eff=nu_eff,
            edge_to_index=edge_to_index,
        )
        edge_fisher[idx, idx] += f_edge
    n_station = max(max(edge) for edge in edges) + 1
    fisher = base.closure_fisher_after_gauge_marginalization(edge_fisher, q_basis, edges, n_station)
    return 0.5 * (fisher + fisher.T)


def set_modular_loop_specs(
    specs: list[dict],
    schedule_factor: float = 1.0,
    *,
    close_factor: float | None = None,
) -> None:
    """Install loop-wise receiver specs used by approximate closure strategies."""
    global MODULAR_LOOP_SPECS, MODULAR_SCHEDULE_FACTOR, MODULAR_CLOSE_FACTOR
    MODULAR_LOOP_SPECS = list(specs)
    MODULAR_SCHEDULE_FACTOR = float(schedule_factor)
    MODULAR_CLOSE_FACTOR = None if close_factor is None else float(close_factor)


def set_direct_root_weights(
    weights: dict[tuple[int, int, int], float],
    *,
    model: str = "custom_root_closure_weights",
) -> None:
    """Install scheduled direct root-closure weights for the benchmark column."""
    global DIRECT_ROOT_WEIGHTS, DIRECT_ROOT_WEIGHT_MODEL
    DIRECT_ROOT_WEIGHTS = {tuple(key): float(value) for key, value in weights.items()}
    DIRECT_ROOT_WEIGHT_MODEL = str(model)


def equal_loop_budget_factors(specs: list[dict]) -> tuple[float, float]:
    n_loop = max(len(specs), 1)
    n_close = sum(1 for spec in specs if all(station in CORE_STATIONS for station in tuple(spec["tri"])))
    return n_close / n_loop, 1.0 / n_loop


def sample_cycle_noise_from_fisher_with_q_standard_normal(
    fisher: np.ndarray,
    q_basis: np.ndarray,
    standard_normal_q: np.ndarray,
    *,
    max_std: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cycle noise using a fixed standard-normal vector in q coordinates.

    This pairs approximate and benchmark closure receivers in the same
    independent-closure coordinate system.  Each receiver keeps its own
    covariance, but the underlying random realization is identical.
    """
    evals, evecs = np.linalg.eigh(0.5 * (fisher + fisher.T))
    eval_floor = 1.0 / max_std**2
    safe = np.maximum(evals, eval_floor)
    cov_sqrt = (evecs / np.sqrt(safe)) @ evecs.T
    coeff = cov_sqrt @ np.asarray(standard_normal_q, dtype=float)
    edge_noise = q_basis @ coeff
    coord_cov = (evecs / safe) @ evecs.T
    edge_cov = q_basis @ coord_cov @ q_basis.T
    return edge_noise, np.sqrt(np.maximum(np.diag(edge_cov), 0.0))


def cycle_sigma_from_fisher(fisher: np.ndarray, *, max_std: float = 2.5) -> np.ndarray:
    evals, evecs = np.linalg.eigh(0.5 * (fisher + fisher.T))
    eval_floor = 1.0 / max_std**2
    safe = np.maximum(evals, eval_floor)
    coord_cov = (evecs / safe) @ evecs.T
    return np.sqrt(np.maximum(np.diag(0.5 * (coord_cov + coord_cov.T)), 0.0))


def closure_edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


def root_independent_triangles(n_station: int) -> list[tuple[int, int, int]]:
    return [(0, i, j) for i in range(1, n_station) for j in range(i + 1, n_station)]


def default_capacity_relaxed_root_weights(
    n_station: int,
    n_closure: int,
) -> dict[tuple[int, int, int], float]:
    triangles = root_independent_triangles(n_station)
    if len(triangles) != n_closure:
        raise ValueError(f"root closure count {len(triangles)} does not match q-basis dimension {n_closure}")
    per_loop_weight = (n_station - 1.0) / max(n_closure, 1)
    return {tri: float(per_loop_weight) for tri in triangles}


def scalar_closure_fisher_from_edges(fab: float, fbc: float, fac: float) -> float:
    if min(fab, fbc, fac) <= 0.0:
        return 0.0
    return 1.0 / (1.0 / fab + 1.0 / fbc + 1.0 / fac)


def edge_pair_fisher_for_sample(
    i: int,
    j: int,
    fi: float,
    fj: float,
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    edge_to_index: dict[tuple[int, int], int],
) -> float:
    idx = edge_to_index[(i, j)]
    signal2 = fi * fj * eta[i] * eta[j] * u_station[i] * u_station[j]
    load = fi * (eta[i] * u_station[i] + station_noise[i]) + fj * (
        eta[j] * u_station[j] + station_noise[j]
    ) + EPS_PAIR
    return float(total_modes * 4.0 * signal2 * nu_eff[idx] ** 2 / max(load, 1e-300))


def noncore_directed_fractions(
    tri: tuple[int, int, int],
    split: tuple[float, float, float],
) -> dict[tuple[int, int], float]:
    """Directed fractions for remote-involved edge readout.

    The close four-station subarray supplies close-only closure information.
    Therefore close-close baselines inside remote-involved loops are not given
    separate edge splits here.
    """
    a, b, c = tri
    pairs = [(a, b), (b, c), (a, c)]
    incident: dict[int, list[int]] = {station: [] for station in tri}
    for i, j in pairs:
        if i in CORE_STATIONS and j in CORE_STATIONS:
            continue
        incident[i].append(j)
        incident[j].append(i)

    directed: dict[tuple[int, int], float] = {}
    for value, station in zip(split, tri):
        total = 1.0 if station not in CORE_STATIONS else CORE_JOINT_FRACTION
        neighbors = incident[station]
        if len(neighbors) == 0:
            continue
        if len(neighbors) == 1:
            directed[(station, neighbors[0])] = total
            continue
        first, second = neighbors
        value = min(max(float(value), 0.0), total)
        directed[(station, first)] = value
        directed[(station, second)] = total - value
    return directed


def raw_edge_phase_fisher_station_u(
    visibilities: np.ndarray,
    station_efficiencies: np.ndarray,
    station_noise: np.ndarray,
    station_u: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    eig_floor: float = 1e-12,
) -> np.ndarray:
    n_station = len(station_efficiencies)
    bmat = np.diag(station_efficiencies * station_u + station_noise).astype(complex)
    source_coherences = np.zeros(len(edges), dtype=complex)
    for edge_index, (i, j) in enumerate(edges):
        coherence = (
            math.sqrt(station_efficiencies[i] * station_efficiencies[j] * station_u[i] * station_u[j])
            * visibilities[edge_index]
        )
        source_coherences[edge_index] = coherence
        bmat[i, j] = coherence
        bmat[j, i] = np.conj(coherence)

    edge_derivatives = []
    for edge_index, (i, j) in enumerate(edges):
        deriv = np.zeros((n_station, n_station), dtype=complex)
        deriv[i, j] = 1j * source_coherences[edge_index]
        deriv[j, i] = -1j * np.conj(source_coherences[edge_index])
        edge_derivatives.append(deriv)
    return base.qfi_from_bmat_derivatives(bmat, edge_derivatives, eig_floor=eig_floor)


def core_direct_edge_fisher_for_sample(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
) -> np.ndarray:
    """Embed one sample of the four-core phase-frame Fisher in edge space."""
    local_edges = base.edge_list(len(CORE_STATIONS))
    local_vis = np.asarray([vtrue[edge_to_index[edge]] for edge in local_edges], dtype=complex)
    subset = list(CORE_STATIONS)
    eta_core = CORE_JOINT_FRACTION * eta[subset]
    station_part = np.maximum(direct_noise[subset] - EPS_DIRECT_EXTRA, 0.0)
    noise_core = CORE_JOINT_FRACTION * station_part + EPS_DIRECT_EXTRA
    local_edge_fisher = total_modes * raw_edge_phase_fisher_station_u(
        local_vis,
        eta_core,
        noise_core,
        u_station[subset],
        local_edges,
    )
    out = np.zeros((len(edges), len(edges)), dtype=float)
    for local_i, edge_i in enumerate(local_edges):
        global_i = edge_to_index[edge_i]
        for local_j, edge_j in enumerate(local_edges):
            global_j = edge_to_index[edge_j]
            out[global_i, global_j] += local_edge_fisher[local_i, local_j]
    return 0.5 * (out + out.T)


def core_triangle_direct_fisher_for_sample(
    tri: tuple[int, int, int],
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
) -> float:
    subset = list(tri)
    local_edges = base.edge_list(3)
    local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, 3))
    local_vis = np.asarray(
        [
            vtrue[edge_to_index[(tri[i], tri[j])]]
            for i, j in local_edges
        ],
        dtype=complex,
    )
    fisher_q = total_modes * aug.noisy_closure_fisher_station_u(
        local_vis,
        eta[subset],
        direct_noise[subset],
        u_station[subset],
        local_q,
        local_edges,
    )
    c_local = closure_edge_vector(local_edges, (0, 1, 2))
    d_local = local_q.T @ c_local
    cov = np.linalg.pinv(fisher_q, rcond=1e-12)
    var = float(d_local @ cov @ d_local)
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def scalar_info_from_cycle_fisher(fisher: np.ndarray, d: np.ndarray) -> float:
    cov = np.linalg.pinv(0.5 * (fisher + fisher.T), rcond=1e-12)
    var = float(d @ cov @ d)
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def direct_root_weighted_fisher_for_sample(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Capacity-relaxed scalar direct Fisher using explicit root-closure weights."""
    direct_raw = total_modes * aug.noisy_closure_fisher_station_u(
        vtrue,
        eta,
        direct_noise,
        u_station,
        q_basis,
        edges,
    )
    n = max(max(edge) for edge in edges) + 1
    weights = DIRECT_ROOT_WEIGHTS or default_capacity_relaxed_root_weights(n, q_basis.shape[1])
    fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
    for tri, weight in weights.items():
        c_global = closure_edge_vector(edges, tri)
        d_global = q_basis.T @ c_global
        scalar = scalar_info_from_cycle_fisher(direct_raw, d_global)
        fisher += float(weight) * scalar * np.outer(d_global, d_global)
    return 0.5 * (fisher + fisher.T)


def modular_hybrid_fisher_for_sample(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    direct_noise: np.ndarray,
    nu_eff: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """SNR-derived closure information for a switched modular receiver.

    Each switch state produces a scalar closure SNR; the scalar information
    below is SNR^2 for that actual approximate hardware path.  The full
    direct/scheduled Fisher matrix is kept outside this routine as a benchmark,
    not used as the strategy itself.
    """
    if not MODULAR_LOOP_SPECS:
        raise RuntimeError("modular hybrid receiver requested before loop specs were installed")

    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
    for spec in MODULAR_LOOP_SPECS:
        tri = tuple(spec["tri"])
        c_global = closure_edge_vector(edges, tri)
        d_global = q_basis.T @ c_global
        if spec["receiver"] == "core_3mode_direct":
            scalar_f = core_triangle_direct_fisher_for_sample(
                tri,
                total_modes=total_modes,
                vtrue=vtrue,
                u_station=u_station,
                eta=eta,
                direct_noise=direct_noise,
                edges=edges,
                edge_to_index=edge_to_index,
            )
        elif spec["receiver"] == "remote_optimized_edge_first":
            a, b, c = tri
            xa, xb, xc = spec["split"]
            fab = edge_pair_fisher_for_sample(
                a,
                b,
                xa,
                xb,
                total_modes=total_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                nu_eff=nu_eff,
                edge_to_index=edge_to_index,
            )
            fbc = edge_pair_fisher_for_sample(
                b,
                c,
                1.0 - xb,
                xc,
                total_modes=total_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                nu_eff=nu_eff,
                edge_to_index=edge_to_index,
            )
            fac = edge_pair_fisher_for_sample(
                a,
                c,
                1.0 - xa,
                1.0 - xc,
                total_modes=total_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                nu_eff=nu_eff,
                edge_to_index=edge_to_index,
            )
            scalar_f = scalar_closure_fisher_from_edges(fab, fbc, fac)
        else:
            raise ValueError(f"unknown modular receiver {spec['receiver']!r}")
        # scalar_f is the actual strategy SNR^2 for this switched closure.
        fisher += MODULAR_SCHEDULE_FACTOR * scalar_f * np.outer(d_global, d_global)
    return 0.5 * (fisher + fisher.T)


def core4_remote_loop_fisher_for_sample(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    direct_noise: np.ndarray,
    nu_eff: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Cycle Fisher for close4 direct closure plus remote-related edge readout."""
    if not MODULAR_LOOP_SPECS:
        raise RuntimeError("core4+remote loop strategy requested before loop specs were installed")

    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    equal_close_factor, remote_loop_factor = equal_loop_budget_factors(MODULAR_LOOP_SPECS)
    close_factor = equal_close_factor if MODULAR_CLOSE_FACTOR is None else MODULAR_CLOSE_FACTOR
    edge_fisher = close_factor * core_direct_edge_fisher_for_sample(
        total_modes=total_modes,
        vtrue=vtrue,
        u_station=u_station,
        eta=eta,
        direct_noise=direct_noise,
        edges=edges,
        edge_to_index=edge_to_index,
    )
    for spec in MODULAR_LOOP_SPECS:
        tri = tuple(spec["tri"])
        if all(station in CORE_STATIONS for station in tri):
            continue
        schedule_weight = float(spec.get("schedule_weight", remote_loop_factor))
        a, b, c = tri
        pairs = [(a, b), (b, c), (a, c)]
        directed = noncore_directed_fractions(tri, tuple(spec["split"]))
        for i, j in pairs:
            if i in CORE_STATIONS and j in CORE_STATIONS:
                continue
            f_edge = edge_pair_fisher_for_sample(
                i,
                j,
                directed[(i, j)],
                directed[(j, i)],
                total_modes=total_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                nu_eff=nu_eff,
                edge_to_index=edge_to_index,
            )
            edge = (i, j) if i < j else (j, i)
            edge_fisher[edge_to_index[edge], edge_to_index[edge]] += schedule_weight * f_edge
    n_station = max(max(edge) for edge in edges) + 1
    fisher = base.closure_fisher_after_gauge_marginalization(edge_fisher, q_basis, edges, n_station)
    return MODULAR_SCHEDULE_FACTOR * 0.5 * (fisher + fisher.T)


def amplitude_sigma_for_sample(
    split: np.ndarray,
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(len(edges), dtype=float)
    for idx, (i, j) in enumerate(edges):
        pi = split[i, j]
        pj = split[j, i]
        signal2 = pi * pj * eta[i] * eta[j] * u_station[i] * u_station[j]
        load = pi * (eta[i] * u_station[i] + station_noise[i]) + pj * (
            eta[j] * u_station[j] + station_noise[j]
        ) + EPS_PAIR
        out[idx] = total_modes * 4.0 * signal2 / max(load, 1e-300)
    return 1.0 / np.sqrt(np.maximum(out, 1e-300))


def simulate_bands_with_strategies(
    case: aug.NetworkCase,
    splits: dict[str, np.ndarray],
) -> tuple[list[dict[str, np.ndarray]], dict, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    drift_rng = np.random.default_rng(RNG_SEED + 37)
    stations, diameters, _, _ = aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n = len(stations)
    edges = base.edge_list(n)
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, n))
    rank_share = min(1.0, (n - 1.0) / q_basis.shape[1])

    truth, axis_uas = base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
    wavelength_source = getattr(base, "make_source_at_wavelength_nm", None)

    effective_hub_dist = aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n, EPS_STATION, dtype=float)
    direct_noise = np.full(n, EPS_STATION + EPS_DIRECT_EXTRA, dtype=float)
    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    station_piston_std = aug.POST_AVERAGE_DRIFT_STD / np.sqrt(2.0)

    endpoint_coverage = {}
    for wavelength_nm in (aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM):
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            wavelength_nm * 1e-9,
            latitude_deg=case.latitude_deg,
            declination_deg=SOURCE.dec_deg,
        )
        endpoint_coverage[f"{wavelength_nm:g}"] = {
            "u": (uu_rows.reshape(-1) / 1e9).tolist(),
            "v": (vv_rows.reshape(-1) / 1e9).tolist(),
        }

    lam_edges_nm = np.arange(aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM, aug.LAMBDA_STEP_NM)
    lam_edges_nm[-1] = aug.LAMBDA_MAX_NM
    strategy_keys = ["all"] + list(splits) + ["nmode_joint_scheduled", "nmode_joint_rawQFI"]
    uniform_split = splits["edge_uniform"]

    bands: list[dict[str, np.ndarray]] = []
    all_amp_sigma = []
    all_amp_true = []
    all_amp_data = []
    for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
        center_nm = math.sqrt(lo_nm * hi_nm)
        lam = math.sqrt(lo_nm * hi_nm) * 1e-9
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
        freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
        total_modes = aug.EXPOSURE_S * aug.OBSERVING_DAYS * (freq_hi - freq_lo)
        u_station = aug.station_u_modes(freq, diameters)
        if callable(wavelength_source):
            band_truth, _ = wavelength_source(aug.N_PIX, aug.HALF_WIDTH_UAS, center_nm)
            band_vgrid, band_uv_axis = base.visibility_grid(band_truth, fov_rad)
        else:
            band_vgrid, band_uv_axis = vgrid, uv_axis
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            lam,
            latitude_deg=case.latitude_deg,
            declination_deg=SOURCE.dec_deg,
        )
        band = {"u": [], "v": []}
        vis = {key: [] for key in strategy_keys}
        sig = {key: [] for key in strategy_keys}
        sig_q = {key: [] for key in strategy_keys}
        amp_all = []
        amp_true_all = []
        amp_sigma_all = []
        for uu, vv in zip(uu_rows, vv_rows):
            vtrue = base.interp_vis(band_vgrid, band_uv_axis, uu, vv)
            amp = np.abs(vtrue)
            phase = np.angle(vtrue)
            phase_closure = q_basis @ (q_basis.T @ phase)
            nu_eff = np.clip(amp, 1e-4, 0.98)

            sigma_amp = amplitude_sigma_for_sample(
                uniform_split,
                total_modes=total_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                edges=edges,
            )
            measured_amp = amp.copy() if USE_TRUE_AMPLITUDE else np.maximum(amp + rng.normal(scale=sigma_amp), 0.0)
            phase_amp = np.maximum(measured_amp, 1e-8)
            uniform_raw_noise = None
            uniform_sigma_raw = None
            common_edge_z = rng.normal(size=len(edges)) if PAIR_CORE_DIRECT_NOISE else None
            common_q_z = q_basis.T @ common_edge_z if common_edge_z is not None else None

            for key, split in splits.items():
                if key == SWITCHED_MODULAR_HYBRID_KEY:
                    fisher_cycle = modular_hybrid_fisher_for_sample(
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
                    noise, sigma_projected = base.sample_cycle_noise_from_fisher(rng, fisher_cycle, q_basis)
                elif key == "core4_remote_optimized":
                    if MODULAR_LOOP_SPECS:
                        fisher_cycle = core4_remote_loop_fisher_for_sample(
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
                    else:
                        fisher_cycle = core4_remote_global_split_fisher_for_sample(
                            split,
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
                    if PAIR_CORE_DIRECT_NOISE:
                        noise, sigma_projected = sample_cycle_noise_from_fisher_with_q_standard_normal(
                            fisher_cycle,
                            q_basis,
                            common_q_z,
                        )
                    else:
                        noise, sigma_projected = base.sample_cycle_noise_from_fisher(rng, fisher_cycle, q_basis)
                    sig_q[key].append(cycle_sigma_from_fisher(fisher_cycle))
                elif key == "core4_remote_global_split":
                    fisher_cycle = core4_remote_global_split_fisher_for_sample(
                        split,
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
                    noise, sigma_projected = base.sample_cycle_noise_from_fisher(rng, fisher_cycle, q_basis)
                    sig_q[key].append(cycle_sigma_from_fisher(fisher_cycle))
                else:
                    fisher_edge = edge_fisher_for_sample(
                        split,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edges=edges,
                    )
                    sigma_raw = np.minimum(1.0 / np.sqrt(np.maximum(fisher_edge, 1e-300)), aug.SIGMA_CLIP_RAD)
                    if common_edge_z is None:
                        raw_noise = rng.normal(scale=sigma_raw)
                    else:
                        raw_noise = sigma_raw * common_edge_z
                    noise = q_basis @ (q_basis.T @ raw_noise)
                    cov_cycle = q_basis.T @ ((sigma_raw**2)[:, None] * q_basis)
                    cov_edge = q_basis @ cov_cycle @ q_basis.T
                    sigma_projected = np.sqrt(np.maximum(np.diag(cov_edge), 0.0))
                    sig_q[key].append(np.sqrt(np.maximum(np.diag(cov_cycle), 0.0)))
                    if key == "edge_uniform":
                        uniform_raw_noise = raw_noise
                        uniform_sigma_raw = sigma_raw
                vis[key].append(phase_amp * np.exp(1j * (phase_closure + noise)))
                sig[key].append(sigma_projected)

            if uniform_raw_noise is None or uniform_sigma_raw is None:
                raise KeyError("simulate_bands_with_strategies requires an edge_uniform split")
            station_pistons = drift_rng.normal(scale=station_piston_std, size=n)
            station_pistons -= np.mean(station_pistons)
            residual_drift = np.array([station_pistons[i] - station_pistons[j] for i, j in edges])
            noise_all = uniform_raw_noise + residual_drift
            sigma_all = np.sqrt(uniform_sigma_raw**2 + aug.POST_AVERAGE_DRIFT_STD**2)
            vis["all"].append(phase_amp * np.exp(1j * (phase + noise_all)))
            sig["all"].append(sigma_all)

            fisher_direct_raw = total_modes * aug.noisy_closure_fisher_station_u(
                vtrue,
                eta,
                direct_noise,
                u_station,
                q_basis,
                edges,
            )
            fisher_direct_scheduled = direct_root_weighted_fisher_for_sample(
                total_modes=total_modes,
                vtrue=vtrue,
                u_station=u_station,
                eta=eta,
                direct_noise=direct_noise,
                q_basis=q_basis,
                edges=edges,
            )
            for key, fisher_direct in (
                ("nmode_joint_rawQFI", fisher_direct_raw),
                ("nmode_joint_scheduled", fisher_direct_scheduled),
            ):
                if common_q_z is None:
                    noise, sigma_direct = base.sample_cycle_noise_from_fisher(rng, fisher_direct, q_basis)
                else:
                    noise, sigma_direct = sample_cycle_noise_from_fisher_with_q_standard_normal(
                        fisher_direct,
                        q_basis,
                        common_q_z,
                    )
                vis[key].append(phase_amp * np.exp(1j * (phase_closure + noise)))
                sig[key].append(sigma_direct)
                sig_q[key].append(cycle_sigma_from_fisher(fisher_direct))

            band["u"].append(uu)
            band["v"].append(vv)
            amp_all.append(measured_amp)
            amp_true_all.append(amp)
            amp_sigma_all.append(sigma_amp)
        band["u"] = np.concatenate(band["u"])
        band["v"] = np.concatenate(band["v"])
        band["amp"] = np.concatenate(amp_all)
        band["amp_true"] = np.concatenate(amp_true_all)
        band["amp_sigma"] = np.concatenate(amp_sigma_all)
        for key in strategy_keys:
            band[f"vis_{key}"] = np.concatenate(vis[key])
            band[f"sigma_{key}"] = np.concatenate(sig[key])
            if sig_q[key]:
                band[f"sigmaq_{key}"] = np.concatenate(sig_q[key])
        bands.append(band)
        all_amp_sigma.append(band["amp_sigma"])
        all_amp_true.append(band["amp_true"])
        all_amp_data.append(band["amp"])

    stats = {
        "case": case.key,
        "n_station": n,
        "n_closure": int(q_basis.shape[1]),
        "rank_share": float(rank_share),
        "capacity_relaxed_weight_per_closure": float((n - 1.0) / q_basis.shape[1]),
        "capacity_relaxed_total_root_weight": float(n - 1.0),
        "direct_schedule_model": DIRECT_ROOT_WEIGHT_MODEL,
        "direct_root_weights": {
            f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}": float(weight)
            for tri, weight in sorted(
                (
                    DIRECT_ROOT_WEIGHTS
                    or default_capacity_relaxed_root_weights(n, q_basis.shape[1])
                ).items()
            )
        },
        "modular_close_factor": MODULAR_CLOSE_FACTOR,
        "modular_schedule_factor": float(MODULAR_SCHEDULE_FACTOR),
        "modular_loop_specs": [
            {
                "closure": f"S{tuple(spec['tri'])[0] + 1}-S{tuple(spec['tri'])[1] + 1}-S{tuple(spec['tri'])[2] + 1}",
                "receiver": str(spec.get("receiver", "")),
                "split": [float(value) for value in spec.get("split", [])],
                "schedule_weight": float(spec.get("schedule_weight", 0.0)),
            }
            for spec in MODULAR_LOOP_SPECS
        ],
        "eps_station": EPS_STATION,
        "eps_pair": EPS_PAIR,
        "eps_direct_extra": EPS_DIRECT_EXTRA,
        "post_average_drift_std_rad": float(getattr(aug, "POST_AVERAGE_DRIFT_STD", 0.0)),
        "fiber_length_scale": FIBER_LENGTH_SCALE,
        "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
        "amplitude_sigma_model": "common uniform-edge amplitude realization shared by all strategies",
        "use_true_amplitude": USE_TRUE_AMPLITUDE,
        "pair_core_direct_noise": PAIR_CORE_DIRECT_NOISE,
        "phase_noise_pairing": (
            "shared edge-space standard normal z_edge; all/edge use sigma_edge*z_edge, "
            "core4/direct use Q^T z_edge in the fixed independent-closure coordinates"
            if PAIR_CORE_DIRECT_NOISE
            else "independent strategy noise draws"
        ),
        "endpoint_coverage_g_lambda": endpoint_coverage,
        "source_spectral_model": getattr(base, "SOURCE_COMPONENT_SPECTRAL_MODEL", "achromatic source morphology"),
    }
    component_fractions = getattr(base, "source_component_flux_fractions", None)
    if callable(component_fractions):
        stats["source_component_flux_fractions"] = {
            f"{wavelength_nm:g}": component_fractions(wavelength_nm)
            for wavelength_nm in (400.0, 488.0, 510.0, 550.0, 659.0, 800.0)
        }
    amp_sigma = np.concatenate(all_amp_sigma)
    amp_true = np.concatenate(all_amp_true)
    stats["amplitude_snr_median"] = float(np.median(amp_true / np.maximum(amp_sigma, 1e-300)))
    stats["amplitude_noise_rms"] = float(np.sqrt(np.mean((np.concatenate(all_amp_data) - amp_true) ** 2)))
    return bands, stats, truth, axis_uas


def reconstruct_strategies(
    bands: list[dict[str, np.ndarray]],
    truth: np.ndarray,
    strategy_keys: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    images = {key: np.zeros_like(truth) for key in strategy_keys}
    weights = {key: 0.0 for key in strategy_keys}
    for band in bands:
        for key in strategy_keys:
            image, weight = wt.reconstruct_band_coarse_interp(band, key, fov_rad)
            images[key] += weight * image
            weights[key] += weight
    for key in strategy_keys:
        images[key] = wt.normalize_stack(images[key] / max(weights[key], 1e-30))
    return images, weights


def reconstruct_strategies_with_common_weights(
    bands: list[dict[str, np.ndarray]],
    truth: np.ndarray,
    strategy_keys: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Reconstruct with identical uv weights to isolate phase-noise effects."""
    common_bands: list[dict[str, np.ndarray]] = []
    for band in bands:
        copied = dict(band)
        for key in strategy_keys:
            copied[f"sigma_{key}"] = np.ones_like(band[f"sigma_{key}"])
        common_bands.append(copied)
    return reconstruct_strategies(common_bands, truth, strategy_keys)


def phase_residual_diagnostics(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    truth: np.ndarray,
    strategy_keys: list[str],
) -> dict[str, dict[str, float]]:
    """Compare sampled pseudo-visibility phases to the true closure projection."""
    stations, _, _, _ = aug.station_table_from_case(case)
    edges = base.edge_list(len(stations))
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, len(stations)))
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
    phase_residuals = {key: [] for key in strategy_keys}
    sigma_values = {key: [] for key in strategy_keys}

    for band in bands:
        n_row = len(band["u"]) // len(edges)
        true_closure_phase = []
        for row in range(n_row):
            row_slice = slice(row * len(edges), (row + 1) * len(edges))
            vtrue = base.interp_vis(vgrid, uv_axis, band["u"][row_slice], band["v"][row_slice])
            true_closure_phase.append(q_basis @ (q_basis.T @ np.angle(vtrue)))
        true_closure_phase = np.concatenate(true_closure_phase)
        for key in strategy_keys:
            residual = np.angle(band[f"vis_{key}"] * np.exp(-1j * true_closure_phase))
            phase_residuals[key].append(residual)
            sigma_values[key].append(band[f"sigma_{key}"])

    diagnostics: dict[str, dict[str, float]] = {}
    for key in strategy_keys:
        residual = np.concatenate(phase_residuals[key])
        sigma = np.concatenate(sigma_values[key])
        diagnostics[key] = {
            "phase_resid_rms_rad": float(np.sqrt(np.mean(residual**2))),
            "phase_resid_median_abs_rad": float(np.median(np.abs(residual))),
            "sigma_p10_rad": float(np.percentile(sigma, 10.0)),
            "sigma_median_rad": float(np.median(sigma)),
            "sigma_p90_rad": float(np.percentile(sigma, 90.0)),
        }
    return diagnostics


def plot_strategy_grid(
    images: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    truth: np.ndarray,
    axis_uas: np.ndarray,
) -> tuple[Path, Path]:
    strategy_order = [
        "edge_uniform",
        "edge_logdet",
        "edge_meanrms",
        "edge_maxrms",
        "edge_trace",
        "nmode_joint_scheduled",
        "nmode_joint_rawQFI",
    ]
    labels = {
        "edge_uniform": "edge uniform",
        "edge_logdet": "edge logdet-opt",
        "edge_meanrms": "edge mean-RMS-opt",
        "edge_maxrms": "edge max-RMS-opt",
        "edge_trace": "edge trace-opt",
        "nmode_joint_scheduled": "N-mode scheduled",
        "nmode_joint_rawQFI": "N-mode raw QFI",
    }
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 5.0), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    axes[0].set_title("input NGC 4151")
    for ax, key in zip(axes[1:], strategy_order):
        ax.imshow(opt.normalize_blr_display(images[key]), origin="lower", extent=extent, cmap="inferno")
        m = metrics[key]
        ax.set_title(f"{labels[key]}\nBLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}")
    for ax in axes:
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    tag = "trueamp" if USE_TRUE_AMPLITUDE else "physical_amp"
    png = OUT / f"fig3_split_objective_strategy_grid_{tag}.png"
    pdf = OUT / f"fig3_split_objective_strategy_grid_{tag}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    configure()
    case = rml_cases.load_maunakea_plus3_case()
    split_benchmark = AllClosureBenchmark()
    split_matrices = {
        "edge_uniform": split_benchmark.uniform_split_matrix(),
    }
    split_info = {}
    for objective, key in (
        ("logdet", "edge_logdet"),
        ("mean_rms", "edge_meanrms"),
        ("max_rms", "edge_maxrms"),
        ("trace", "edge_trace"),
    ):
        split, info = optimize_split(split_benchmark, objective)
        split_matrices[key] = split
        split_info[key] = info

    with ngc.patched_source(SOURCE):
        bands, stats, truth, axis_uas = simulate_bands_with_strategies(case, split_matrices)
    strategy_keys = list(split_matrices) + ["nmode_joint_scheduled", "nmode_joint_rawQFI"]
    images, weights = reconstruct_strategies(bands, truth, strategy_keys)
    common_images, common_weights = reconstruct_strategies_with_common_weights(bands, truth, strategy_keys)
    metrics = {
        key: ngc.image_metrics(truth, image, axis_uas, SOURCE)
        for key, image in images.items()
    }
    common_metrics = {
        key: ngc.image_metrics(truth, image, axis_uas, SOURCE)
        for key, image in common_images.items()
    }
    phase_diagnostics = phase_residual_diagnostics(bands, case, truth, strategy_keys)
    pdf, png = plot_strategy_grid(images, metrics, truth, axis_uas)

    tag = "trueamp" if USE_TRUE_AMPLITUDE else "physical_amp"
    metrics_path = OUT / f"fig3_split_objective_metrics_{tag}.csv"
    with metrics_path.open("w", newline="") as f:
        fieldnames = ["strategy", "global_corr", "blr_corr", "ring_contrast", "stack_weight"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in strategy_keys:
            writer.writerow({"strategy": key, **metrics[key], "stack_weight": weights[key]})
    diagnostics_path = OUT / f"fig3_split_objective_diagnostics_{tag}.csv"
    with diagnostics_path.open("w", newline="") as f:
        fieldnames = [
            "strategy",
            "phase_resid_rms_rad",
            "phase_resid_median_abs_rad",
            "sigma_p10_rad",
            "sigma_median_rad",
            "sigma_p90_rad",
            "common_weight_global_corr",
            "common_weight_blr_corr",
            "common_weight_ring_contrast",
            "common_stack_weight",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in strategy_keys:
            writer.writerow(
                {
                    "strategy": key,
                    **phase_diagnostics[key],
                    "common_weight_global_corr": common_metrics[key]["global_corr"],
                    "common_weight_blr_corr": common_metrics[key]["blr_corr"],
                    "common_weight_ring_contrast": common_metrics[key]["ring_contrast"],
                    "common_stack_weight": common_weights[key],
                }
            )
    summary = {
        "stats": stats,
        "split_objectives": split_info,
        "metrics": metrics,
        "common_weight_metrics": common_metrics,
        "phase_diagnostics": phase_diagnostics,
        "weights": weights,
        "common_weights": common_weights,
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(metrics_path),
        "diagnostics_csv": str(diagnostics_path),
        "note": (
            "N-mode rawQFI is an upper-bound image benchmark sampled from the raw closure QFI. "
            "N-mode scheduled uses an explicit capacity-relaxed scalar root-closure schedule "
            "with sum_l w_l=N-1 and uniform w_l=(N-1)/C; rawQFI remains only an upper-bound diagnostic."
        ),
    }
    summary_path = OUT / f"fig3_split_objective_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(pdf)
    print(png)
    print(metrics_path)
    print(summary_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
