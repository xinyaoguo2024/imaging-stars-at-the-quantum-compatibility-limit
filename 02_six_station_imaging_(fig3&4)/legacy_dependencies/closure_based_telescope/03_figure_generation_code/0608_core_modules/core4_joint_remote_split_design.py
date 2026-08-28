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
SNAPSHOT_DIR = BUNDLE / "code" / "all_python_snapshot"
CORE_DIR = BUNDLE / "code" / "core"
for _path in (CORE_DIR, SNAPSHOT_DIR):
    _path_text = str(_path)
    if _path_text in sys.path:
        sys.path.remove(_path_text)
    sys.path.insert(0, _path_text)

import make_all_closure_global_benchmark_note as bm_lib  # noqa: E402
import plot_augmented_existing_telescope_closure_networks as aug  # noqa: E402
import plot_augmented_existing_telescope_ngc_sources as ngc  # noqa: E402
import plot_prl_broadband_clean as base  # noqa: E402
from plot_prl_broadband_blr_realnight import project_enu_baselines  # noqa: E402


OUT = BUNDLE / "exploration" / "core4_joint_remote_split"
OUT.mkdir(parents=True, exist_ok=True)

CORE = tuple(range(4))
REMOTE = tuple(range(4, 7))
CORE_JOINT_FRACTION = 0.5
CORE_REMOTE_FRACTION = 0.5
REMOTE_TOTAL_FRACTION = 1.0
SPLIT_FLOOR = bm_lib.SPLIT_FLOOR
ROOT_UNIFORM_LOG_PENALTY = 2.0
DEFAULT_CLOSE4_PHASE_FRAME_SCHEDULE_WEIGHT = 1.0
DEFAULT_REMOTE_LOOP_TOTAL_WEIGHT = 8.8
DEFAULT_REMOTE_LOOP_WEIGHT_CAP = 1.0
DEFAULT_DIRECT_ALLCLOSURE_MODE = "capacity_relaxed_scalar"


def edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


def edge_fisher_from_arrays(arrays: dict[str, np.ndarray], fi: float, fj: float) -> float:
    if fi <= 0.0 or fj <= 0.0:
        return 0.0
    denom = fi * arrays["ai"] + fj * arrays["aj"] + arrays["pair"]
    return float(np.sum(arrays["k"] * fi * fj / np.maximum(denom, 1e-300)))


def raw_edge_phase_fisher_station_u(
    visibilities: np.ndarray,
    station_efficiencies: np.ndarray,
    station_noise: np.ndarray,
    station_u: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    eig_floor: float = 1e-12,
) -> np.ndarray:
    """Raw edge-phase Fisher before station-gauge marginalization.

    For the compact close-four receiver we need both closure directions and
    station-gauge/nuisance directions.  The full-array Schur complement later
    removes the nuisance coordinates; projecting to the close-four closure
    basis here would discard the close-baseline phase components required by
    remote-involving loops.
    """
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


def measurement_vector(bm: bm_lib.AllClosureBenchmark, tri: tuple[int, int, int]) -> np.ndarray:
    return bm.q_basis.T @ edge_vector(bm.edges, tri)


def root_independent_triangles(n_station: int) -> list[tuple[int, int, int]]:
    """Root-cycle triangle coordinates using station 0 as the gauge root."""
    return [(0, i, j) for i in range(1, n_station) for j in range(i + 1, n_station)]


def equal_loop_budget_factors(specs: list[dict[str, object]]) -> tuple[float, float]:
    """Return Fisher scale factors for equal-budget root-loop scheduling.

    Each independent root loop receives the same photon/time budget.  The
    close-only root loops are read by the shared close4 closure receiver, so
    their equal shares are added into one close4 block.  Every remote-involved
    loop contributes its own optimized edge-readout block with one loop share.
    """
    n_loop = max(len(specs), 1)
    n_close = sum(1 for spec in specs if all(station in CORE for station in tuple(spec["tri"])))
    return n_close / n_loop, 1.0 / n_loop


def row_softmax(raw: np.ndarray, allowed: np.ndarray, total: float, floor: float) -> np.ndarray:
    out = np.zeros(raw.shape[0], dtype=float)
    n_allowed = int(np.sum(allowed))
    if n_allowed == 0:
        return out
    remaining = total - n_allowed * floor
    if remaining < -1e-12:
        raise ValueError("floor exceeds row budget")
    x = raw[allowed]
    weights = np.exp(x - np.max(x))
    weights /= np.sum(weights)
    out[allowed] = floor + max(0.0, remaining) * weights
    return out


def project_remote_split(raw: np.ndarray, bm: bm_lib.AllClosureBenchmark) -> np.ndarray:
    """Station-side split matrix for the proposed core4+remote strategy.

    Core stations use half their light in the core 4-mode receiver.  The other
    half is split only among the three remote stations.  Remote stations split
    all their light among the other six stations.
    """
    p = np.zeros((bm.n, bm.n), dtype=float)
    for i in CORE:
        allowed = np.zeros(bm.n, dtype=bool)
        allowed[list(REMOTE)] = True
        p[i] = row_softmax(raw[i], allowed, CORE_REMOTE_FRACTION, SPLIT_FLOOR)
    for i in REMOTE:
        allowed = np.ones(bm.n, dtype=bool)
        allowed[i] = False
        p[i] = row_softmax(raw[i], allowed, REMOTE_TOTAL_FRACTION, SPLIT_FLOOR)
    return p


def fisher_for_split(bm: bm_lib.AllClosureBenchmark, p: np.ndarray) -> np.ndarray:
    """Closure Fisher for close4 closure receiver plus global remote-edge split."""
    edge_fisher = core_direct_edge_fisher_matrix(bm) + remote_edge_fisher_matrix_for_split(bm, p)
    return base.closure_fisher_after_gauge_marginalization(
        edge_fisher,
        bm.q_basis,
        bm.edges,
        bm.n,
    )


def remote_edge_fisher_matrix_for_split(bm: bm_lib.AllClosureBenchmark, p: np.ndarray) -> np.ndarray:
    """Edge Fisher matrix for globally shared remote-related edge readout."""
    edge_diag = np.zeros(len(bm.edges), dtype=float)
    for idx, (i, j) in enumerate(bm.edges):
        if i in CORE and j in CORE:
            continue
        edge_diag[idx] = edge_fisher_from_arrays(bm.edge_arrays[(i, j)], p[i, j], p[j, i])
    return np.diag(edge_diag)


def station_loop_budget(station: int) -> float:
    return REMOTE_TOTAL_FRACTION if station in REMOTE else CORE_REMOTE_FRACTION


def edge_fisher_for_pair(bm: bm_lib.AllClosureBenchmark, i: int, j: int, fi: float, fj: float) -> float:
    if i > j:
        i, j = j, i
        fi, fj = fj, fi
    return edge_fisher_from_arrays(bm.edge_arrays[(i, j)], fi, fj)


def noncore_directed_fractions(
    tri: tuple[int, int, int],
    split: tuple[float, float, float],
) -> dict[tuple[int, int], float]:
    """Directed station fractions for remote-involved edge readout.

    The close four-station subarray supplies close-only closure information.
    Therefore a close-close baseline inside a remote-involved loop is not given
    a separate edge split here; it is later handled as close-closure information
    plus gauge nuisance before marginalization.
    """
    a, b, c = tri
    pairs = [(a, b), (b, c), (a, c)]
    incident: dict[int, list[int]] = {station: [] for station in tri}
    for i, j in pairs:
        if i in CORE and j in CORE:
            continue
        incident[i].append(j)
        incident[j].append(i)

    directed: dict[tuple[int, int], float] = {}
    for value, station in zip(split, tri):
        neighbors = incident[station]
        total = station_loop_budget(station)
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


def canonical_noncore_split(
    tri: tuple[int, int, int],
    split: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Make unused one-neighbor split coordinates explicit in diagnostics."""
    a, b, c = tri
    pairs = [(a, b), (b, c), (a, c)]
    incident: dict[int, list[int]] = {station: [] for station in tri}
    for i, j in pairs:
        if i in CORE and j in CORE:
            continue
        incident[i].append(j)
        incident[j].append(i)

    out = []
    for value, station in zip(split, tri):
        total = station_loop_budget(station)
        if len(incident[station]) <= 1:
            out.append(total)
        else:
            out.append(min(max(float(value), 0.0), total))
    return tuple(float(x) for x in out)


def core_direct_edge_fisher_matrix(
    bm: bm_lib.AllClosureBenchmark,
    fraction: float = CORE_JOINT_FRACTION,
) -> np.ndarray:
    """Embed the shared four-core phase-frame Fisher as an edge-space block.

    The receiver reads the six close-close edge phases as three closure
    coordinates plus three station-gauge/nuisance coordinates.  We therefore
    embed the raw six-dimensional close-edge Fisher and let the full seven
    station Schur complement eliminate gauge nuisance directions.  This keeps
    the close-baseline phase combinations needed by remote-involving loops.
    """
    local_edges = base.edge_list(len(CORE))
    global_edge_index = {edge: idx for idx, edge in enumerate(bm.edges)}
    core_stations = bm.stations[list(CORE)]
    core_baselines = np.asarray([core_stations[j] - core_stations[i] for i, j in local_edges], dtype=float)
    eta = fraction * bm.eta[list(CORE)]
    noise = np.full(len(CORE), fraction * bm_lib.EPS_STATION + bm_lib.EPS_DIRECT_EXTRA, dtype=float)

    edge_fisher = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
    for lam, freq, total_modes in bm.iter_bands():
        vgrid, uv_axis = bm.visibility_grid_for_wavelength(lam * 1e9)
        u_station = aug.station_u_modes(freq, bm.diameters[list(CORE)])
        uu_rows, vv_rows = project_enu_baselines(
            core_baselines,
            bm.hour_angles,
            lam,
            latitude_deg=bm.case.latitude_deg,
            declination_deg=ngc.NGC4151.dec_deg,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            vlocal = base.interp_vis(vgrid, uv_axis, uu, vv)
            local_edge_fisher = total_modes * raw_edge_phase_fisher_station_u(
                vlocal,
                eta,
                noise,
                u_station,
                local_edges,
            )
            for local_i, edge_i in enumerate(local_edges):
                global_i = global_edge_index[edge_i]
                for local_j, edge_j in enumerate(local_edges):
                    global_j = global_edge_index[edge_j]
                    edge_fisher[global_i, global_j] += local_edge_fisher[local_i, local_j]
    return 0.5 * (edge_fisher + edge_fisher.T)


def remote_edge_fisher_matrix_for_loop_specs(
    bm: bm_lib.AllClosureBenchmark,
    specs: list[dict[str, object]],
    *,
    apply_schedule_weights: bool = False,
) -> np.ndarray:
    edge_diag = np.zeros(len(bm.edges), dtype=float)
    edge_index = {edge: idx for idx, edge in enumerate(bm.edges)}
    for spec in specs:
        schedule_weight = float(spec.get("schedule_weight", 1.0)) if apply_schedule_weights else 1.0
        tri = tuple(spec["tri"])
        if all(station in CORE for station in tri):
            continue
        split = tuple(float(x) for x in spec["split"])
        directed = noncore_directed_fractions(tri, split)
        a, b, c = tri
        for i, j in ((a, b), (b, c), (a, c)):
            if i in CORE and j in CORE:
                continue
            edge = (i, j) if i < j else (j, i)
            fi = directed[(i, j)]
            fj = directed[(j, i)]
            if i > j:
                fi, fj = fj, fi
            edge_diag[edge_index[edge]] += schedule_weight * edge_fisher_for_pair(bm, edge[0], edge[1], fi, fj)
    return np.diag(edge_diag)


def scalar_loop_fisher_for_split(
    bm: bm_lib.AllClosureBenchmark,
    tri: tuple[int, int, int],
    split: tuple[float, float, float],
) -> float:
    """Remote-edge scalar objective for one loop under loop-wise splitting."""
    a, b, c = tri
    pairs = [(a, b), (b, c), (a, c)]

    directed = noncore_directed_fractions(tri, split)
    edge_values = []
    for i, j in pairs:
        if i in CORE and j in CORE:
            continue
        edge_values.append(edge_fisher_for_pair(bm, i, j, directed[(i, j)], directed[(j, i)]))
    if not edge_values:
        return 0.0
    if min(edge_values) <= 0.0:
        return 0.0
    return 1.0 / sum(1.0 / value for value in edge_values)


def optimize_loop_split(
    bm: bm_lib.AllClosureBenchmark,
    tri: tuple[int, int, int],
    seed: int = 20260601,
) -> tuple[float, tuple[float, float, float]]:
    """Optimize one independent closure's edge-first split under loop budgets."""
    budgets = np.asarray([station_loop_budget(station) for station in tri], dtype=float)
    rng = np.random.default_rng(seed + sum((idx + 3) * station for idx, station in enumerate(tri)))
    best_split = tuple(0.5 * budgets)
    best_info = scalar_loop_fisher_for_split(bm, tri, best_split)

    for _ in range(2500):
        candidate = tuple(rng.random(3) * budgets)
        info = scalar_loop_fisher_for_split(bm, tri, candidate)
        if info > best_info:
            best_info = info
            best_split = candidate

    split = np.asarray(best_split, dtype=float)
    for width in (0.30, 0.15, 0.07, 0.03, 0.012, 0.005):
        improved = True
        while improved:
            improved = False
            for idx in range(3):
                for sign in (-1.0, 1.0):
                    candidate = split.copy()
                    candidate[idx] = np.clip(candidate[idx] + sign * width * budgets[idx], 0.0, budgets[idx])
                    info = scalar_loop_fisher_for_split(bm, tri, tuple(candidate))
                    if info > best_info:
                        best_info = info
                        split = candidate
                        improved = True
    return best_info, canonical_noncore_split(tri, tuple(float(x) for x in split))


def optimize_root_loop_splits(
    bm: bm_lib.AllClosureBenchmark,
    seed: int = 20260601,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for tri in root_independent_triangles(bm.n):
        if all(station in CORE for station in tri):
            info = 0.0
            split = tuple(0.0 for _station in tri)
            receiver = "core_direct"
        else:
            info, split = optimize_loop_split(bm, tri, seed=seed)
            receiver = "core_direct_plus_remote_edge"
        specs.append(
            {
                "tri": tri,
                "receiver": receiver,
                "split": split,
                "integrated_scalar_fisher": float(info),
            }
        )
    return specs


def fisher_for_loop_specs(
    bm: bm_lib.AllClosureBenchmark,
    specs: list[dict[str, object]],
    schedule_factor: float = 1.0,
    *,
    close_factor: float | None = None,
    remote_loop_factor: float | None = None,
    apply_spec_schedule_weights: bool = False,
) -> np.ndarray:
    equal_close_factor, equal_remote_loop_factor = equal_loop_budget_factors(specs)
    if close_factor is None:
        close_factor = equal_close_factor
    if remote_loop_factor is None:
        remote_loop_factor = equal_remote_loop_factor
    edge_fisher = float(close_factor) * core_direct_edge_fisher_matrix(bm)
    if apply_spec_schedule_weights:
        edge_fisher += remote_edge_fisher_matrix_for_loop_specs(bm, specs, apply_schedule_weights=True)
    else:
        edge_fisher += float(remote_loop_factor) * remote_edge_fisher_matrix_for_loop_specs(bm, specs)
    fisher = base.closure_fisher_after_gauge_marginalization(edge_fisher, bm.q_basis, bm.edges, bm.n)
    return schedule_factor * 0.5 * (fisher + fisher.T)


def root_closure_log_uniform_objective(
    bm: bm_lib.AllClosureBenchmark,
    fisher: np.ndarray,
    penalty: float = ROOT_UNIFORM_LOG_PENALTY,
) -> float:
    """Balance average root-closure sensitivity against closure-to-closure spread."""
    cov = np.linalg.pinv(0.5 * (fisher + fisher.T), rcond=1e-12)
    log_rms = []
    for tri in root_independent_triangles(bm.n):
        d = measurement_vector(bm, tri)
        var = float(d @ cov @ d)
        if not np.isfinite(var) or var <= 0.0:
            return -math.inf
        log_rms.append(0.5 * math.log(var))
    values = np.asarray(log_rms, dtype=float)
    return -float(np.mean(values) + penalty * np.std(values))


def metrics_objective(fisher: np.ndarray, objective: str, bm: bm_lib.AllClosureBenchmark | None = None) -> float:
    if objective == "root_uniform_log":
        if bm is None:
            raise ValueError("root_uniform_log objective requires benchmark context")
        return root_closure_log_uniform_objective(bm, fisher)
    metrics = bm_lib.stable_metrics(fisher)
    if objective == "mean_rms":
        return -metrics["mean_coord_rms"]
    if objective == "max_rms":
        return -metrics["max_coord_rms"]
    if objective == "logdet":
        return metrics["logdet_fisher"]
    raise ValueError(objective)


def optimize_split(bm: bm_lib.AllClosureBenchmark, objective: str, seed: int = 20260529) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    raw = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw, -np.inf)
    core_edge_fisher = core_direct_edge_fisher_matrix(bm)

    def fisher_for_candidate(p: np.ndarray) -> np.ndarray:
        edge_fisher = core_edge_fisher + remote_edge_fisher_matrix_for_split(bm, p)
        return base.closure_fisher_after_gauge_marginalization(edge_fisher, bm.q_basis, bm.edges, bm.n)

    best_raw = raw.copy()
    best_p = project_remote_split(best_raw, bm)
    best_f = fisher_for_candidate(best_p)
    best_score = metrics_objective(best_f, objective, bm)

    # Random starts.
    for scale in (0.7, 1.4, 2.2, 3.0):
        for _ in range(500):
            cand = raw + rng.normal(scale=scale, size=(bm.n, bm.n))
            np.fill_diagonal(cand, -np.inf)
            p = project_remote_split(cand, bm)
            score = metrics_objective(fisher_for_candidate(p), objective, bm)
            if score > best_score:
                best_score = score
                best_raw = cand
                best_p = p

    # Coordinate refinement.
    active = []
    for i in CORE:
        for j in REMOTE:
            active.append((i, j))
    for i in REMOTE:
        for j in range(bm.n):
            if i != j:
                active.append((i, j))

    for width in (1.2, 0.6, 0.28, 0.12, 0.05, 0.02):
        improved = True
        while improved:
            improved = False
            for i, j in active:
                for sign in (-1.0, 1.0):
                    cand = best_raw.copy()
                    cand[i, j] += sign * width
                    p = project_remote_split(cand, bm)
                    score = metrics_objective(fisher_for_candidate(p), objective, bm)
                    if score > best_score:
                        best_score = score
                        best_raw = cand
                        best_p = p
                        improved = True
    return best_p, {"objective": objective, "score": best_score}


def strict_mean_coord_rms(fisher: np.ndarray, rel_floor: float = 1e-12) -> float:
    fisher = 0.5 * (fisher + fisher.T)
    eig = np.linalg.eigvalsh(fisher)
    max_eig = float(np.max(eig)) if eig.size else 0.0
    min_allowed = max(rel_floor * max(max_eig, 1.0), 1e-300)
    if max_eig <= 0.0 or float(np.min(eig)) <= min_allowed:
        return math.inf
    cov = np.linalg.inv(fisher)
    diag = np.diag(0.5 * (cov + cov.T))
    if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
        return math.inf
    return float(np.mean(np.sqrt(diag)))


def scalar_closure_info_from_cycle_fisher(fisher: np.ndarray, d: np.ndarray) -> float:
    cov = np.linalg.pinv(0.5 * (fisher + fisher.T), rcond=1e-12)
    var = float(d @ cov @ d)
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def direct_fisher_from_root_weights(
    bm: bm_lib.AllClosureBenchmark,
    weights: dict[tuple[int, int, int], float],
) -> np.ndarray:
    fisher = np.zeros((bm.q_basis.shape[1], bm.q_basis.shape[1]), dtype=float)
    for tri, weight in weights.items():
        d = measurement_vector(bm, tri)
        scalar = scalar_closure_info_from_cycle_fisher(bm.direct_raw, d)
        fisher += float(weight) * scalar * np.outer(d, d)
    return 0.5 * (fisher + fisher.T)


def direct_allclosure_total_weight(
    bm: bm_lib.AllClosureBenchmark,
    mode: str = DEFAULT_DIRECT_ALLCLOSURE_MODE,
) -> tuple[float, str]:
    """Total scalar direct-closure budget for the explicit all-closure schedule."""
    normalized = mode.strip().lower().replace("-", "_")
    if normalized in {"strict", "strict_scalar", "strict_scalar_polling", "polling"}:
        return 1.0, (
            "strict scalar polling: one scalar direct-closure setting is read at a time, "
            "so sum_l w_l = 1"
        )
    if normalized in {
        "capacity",
        "capacity_relaxed",
        "capacity_relaxed_scalar",
        "nminus1",
        "n_minus_1",
    }:
        return float(bm.n - 1), (
            "capacity-relaxed scalar schedule: the N-mode direct readout is allowed to "
            "carry N-1 independent scalar score directions per unit resource, so "
            "sum_l w_l = N-1"
        )
    raise ValueError(f"unknown direct all-closure mode {mode!r}")


def uniform_direct_root_weights(
    bm: bm_lib.AllClosureBenchmark,
    *,
    mode: str = DEFAULT_DIRECT_ALLCLOSURE_MODE,
) -> tuple[dict[tuple[int, int, int], float], dict[str, float | str]]:
    """Explicit uniform root-closure weights for the direct all-closure benchmark.

    This replaces the old implicit ``rank_share * direct_raw`` shortcut.  For
    the capacity-relaxed mode the numerical weight is still 6/15 in the
    seven-station case, but it is now an explicit scalar schedule with
    ``sum_l w_l=N-1`` rather than an unexamined raw-QFI rescaling.
    """
    triangles = root_independent_triangles(bm.n)
    total_weight, interpretation = direct_allclosure_total_weight(bm, mode)
    weight = total_weight / max(len(triangles), 1)
    weights = {tri: float(weight) for tri in triangles}
    fisher = direct_fisher_from_root_weights(bm, weights)
    return weights, {
        "objective": "uniform_root_closure",
        "description": "equal explicit scalar weight for each independent root closure",
        "schedule_mode": mode,
        "interpretation": interpretation,
        "strict_mean_coord_rms": float(strict_mean_coord_rms(fisher)),
        "total_weight": float(sum(weights.values())),
        "per_closure_weight": float(weight),
        "n_root_closures": float(len(triangles)),
        "rank_share_obsolete_proxy_value": float(bm.rank_share),
        "snr_vs_single_loop_expected_for_uniform_weights": float(math.sqrt(max(weight, 0.0))),
    }


def snr_values_for_root_triangles(
    bm: bm_lib.AllClosureBenchmark,
    fisher: np.ndarray,
    triangles: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    if triangles is None:
        triangles = root_independent_triangles(bm.n)
    fisher = 0.5 * (fisher + fisher.T)
    cov = np.linalg.pinv(fisher, rcond=1e-12)
    vectors = np.asarray([measurement_vector(bm, tri) for tri in triangles], dtype=float)
    var = np.einsum("ij,jk,ik->i", vectors, cov, vectors)
    return 1.0 / np.sqrt(np.maximum(var, 1e-300))


def project_capped_simplex(values: np.ndarray, total: float, cap: float = 1.0) -> np.ndarray:
    lo = float(np.min(values) - cap)
    hi = float(np.max(values))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        projected = np.clip(values - mid, 0.0, cap)
        if float(np.sum(projected)) > total:
            lo = mid
        else:
            hi = mid
    out = np.clip(values - hi, 0.0, cap)
    scale = total / max(float(np.sum(out)), 1e-300)
    out = np.clip(out * scale, 0.0, cap)
    for _ in range(20):
        deficit = total - float(np.sum(out))
        if abs(deficit) < 1e-12:
            break
        movable = (out < cap - 1e-12) if deficit > 0.0 else (out > 1e-12)
        if not np.any(movable):
            break
        out[movable] += deficit / float(np.count_nonzero(movable))
        out = np.clip(out, 0.0, cap)
    return out


def optimize_strict_v1_schedule_weights(
    bm: bm_lib.AllClosureBenchmark,
    *,
    seed: int = 20260605,
    close_grid: np.ndarray | None = None,
    remote_total_grid: np.ndarray | None = None,
    remote_weight_cap: float = DEFAULT_REMOTE_LOOP_WEIGHT_CAP,
    max_ratio_allowed: float = 1.03,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Optimize the strict V1 close4+remote loop-wise schedule.

    The architecture is fixed: the close-four stations are read by the shared
    phase-frame direct receiver, while every root loop involving a remote station
    is read by an edge-first receiver with loop-internal optimized splitting.
    This routine only tunes the schedule weights of those already-fixed blocks.
    """
    if close_grid is None:
        close_grid = np.arange(0.85, 1.16, 0.05)
    if remote_total_grid is None:
        remote_total_grid = np.arange(8.2, 8.81, 0.2)

    triangles = root_independent_triangles(bm.n)
    remote_mask = np.asarray([not all(station in CORE for station in tri) for tri in triangles], dtype=bool)
    direct_weights, direct_info = uniform_direct_root_weights(bm)
    direct_fisher = direct_fisher_from_root_weights(bm, direct_weights)
    direct_snr = snr_values_for_root_triangles(bm, direct_fisher, triangles)

    specs = optimize_root_loop_splits(bm, seed=seed)
    remote_specs = [spec for spec in specs if not all(station in CORE for station in tuple(spec["tri"]))]
    remote_tris = [tuple(spec["tri"]) for spec in remote_specs]
    remote_edge_blocks = [remote_edge_fisher_matrix_for_loop_specs(bm, [spec]) for spec in remote_specs]
    close_edge_block = core_direct_edge_fisher_matrix(bm)

    def fisher_from_weights(close_factor: float, weights: np.ndarray) -> np.ndarray:
        edge_fisher = float(close_factor) * close_edge_block.copy()
        for weight, block in zip(weights, remote_edge_blocks):
            edge_fisher += float(weight) * block
        return base.closure_fisher_after_gauge_marginalization(edge_fisher, bm.q_basis, bm.edges, bm.n)

    def ratios_for(close_factor: float, weights: np.ndarray) -> np.ndarray:
        return snr_values_for_root_triangles(bm, fisher_from_weights(close_factor, weights), triangles) / np.maximum(direct_snr, 1e-300)

    def objective(ratios: np.ndarray) -> float:
        under = np.maximum(0.0, 0.98 - ratios)
        over = np.maximum(0.0, ratios - max_ratio_allowed)
        spread = np.std(np.log(np.maximum(ratios, 1e-12)))
        remote_min = float(np.min(ratios[remote_mask]))
        close_min = float(np.min(ratios[~remote_mask]))
        return float(1.4 * np.mean(under**2) + 8.0 * np.mean(over**2) + 0.30 * spread**2 - 0.18 * remote_min - 0.10 * close_min)

    def optimize_remote_weights(total: float, close_factor: float, trial_seed: int) -> tuple[np.ndarray, np.ndarray, float]:
        rng = np.random.default_rng(trial_seed)
        n_remote = len(remote_specs)
        starts = [project_capped_simplex(np.full(n_remote, total / n_remote), total, remote_weight_cap)]
        for _ in range(10):
            starts.append(project_capped_simplex(rng.random(n_remote) * remote_weight_cap, total, remote_weight_cap))

        best_weights = starts[0]
        best_ratios = ratios_for(close_factor, best_weights)
        best_score = objective(best_ratios)
        for weights in starts[1:]:
            ratios = ratios_for(close_factor, weights)
            score = objective(ratios)
            if score < best_score:
                best_weights = weights
                best_ratios = ratios
                best_score = score

        for width in (0.18, 0.09, 0.045, 0.022, 0.011, 0.005):
            improved = True
            while improved:
                improved = False
                for i in range(n_remote):
                    for j in range(n_remote):
                        if i == j:
                            continue
                        movable = min(width, best_weights[j], remote_weight_cap - best_weights[i])
                        if movable <= 1e-13:
                            continue
                        candidate = best_weights.copy()
                        candidate[i] += movable
                        candidate[j] -= movable
                        ratios = ratios_for(close_factor, candidate)
                        score = objective(ratios)
                        if score < best_score - 1e-13:
                            best_weights = candidate
                            best_ratios = ratios
                            best_score = score
                            improved = True
        return best_weights, best_ratios, best_score

    candidates: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for total in np.asarray(remote_total_grid, dtype=float):
        for close_factor in np.asarray(close_grid, dtype=float):
            trial_seed = seed + int(round(1000.0 * total + 100.0 * close_factor))
            weights, ratios, score = optimize_remote_weights(float(total), float(close_factor), trial_seed)
            close_ratios = ratios[~remote_mask]
            remote_ratios = ratios[remote_mask]
            valid = (
                float(np.min(close_ratios)) >= 0.90
                and float(np.min(remote_ratios)) >= 0.80
                and float(np.max(ratios)) <= max_ratio_allowed + 1e-9
            )
            quality = (
                0 if valid else 1,
                abs(float(np.median(ratios)) - 0.99)
                + 0.8 * float(np.std(ratios))
                + 1.5 * max(0.0, float(np.max(ratios)) - max_ratio_allowed)
                - 0.5 * float(np.min(ratios)),
                float(score),
            )
            item = {
                "quality": quality,
                "valid": bool(valid),
                "close_factor": float(close_factor),
                "remote_total_weight": float(total),
                "remote_weights": weights,
                "ratios": ratios,
                "score": float(score),
            }
            candidates.append(item)
            if best is None or item["quality"] < best["quality"]:  # type: ignore[operator]
                best = item

    if best is None:
        raise RuntimeError("strict V1 schedule optimization produced no candidates")

    selected_weights = np.asarray(best["remote_weights"], dtype=float)
    selected_ratios = np.asarray(best["ratios"], dtype=float)
    schedule_by_tri = {tri: float(weight) for tri, weight in zip(remote_tris, selected_weights)}
    optimized_specs: list[dict[str, object]] = []
    for spec in specs:
        tri = tuple(spec["tri"])
        item = dict(spec)
        if tri in schedule_by_tri:
            item["schedule_weight"] = schedule_by_tri[tri]
        else:
            item["schedule_weight"] = 0.0
        optimized_specs.append(item)

    ratio_rows = []
    for tri, ratio in zip(triangles, selected_ratios):
        ratio_rows.append(
            {
                "closure": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
                "type": "core_only" if all(station in CORE for station in tri) else "remote_involved",
                "near_over_direct_snr": float(ratio),
                "remote_schedule_weight": float(schedule_by_tri.get(tri, 0.0)),
            }
        )

    budget_diagnostics = strict_v1_photon_budget_diagnostics(
        bm,
        optimized_specs,
        close_factor=float(best["close_factor"]),
        remote_total_weight=float(best["remote_total_weight"]),
        remote_weight_cap=remote_weight_cap,
    )

    top_candidates = []
    for item in sorted(candidates, key=lambda value: value["quality"])[:6]:  # type: ignore[index]
        ratios = np.asarray(item["ratios"], dtype=float)
        top_candidates.append(
            {
                "close_factor": float(item["close_factor"]),
                "remote_total_weight": float(item["remote_total_weight"]),
                "valid": bool(item["valid"]),
                "score": float(item["score"]),
                "ratio_min": float(np.min(ratios)),
                "ratio_median": float(np.median(ratios)),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_max": float(np.max(ratios)),
            }
        )

    info: dict[str, object] = {
        "objective": "strict_v1_loopwise_schedule_ratio_balancing",
        "description": (
            "Close4 is read as a phase-frame direct+nuisance block; every root loop involving "
            "a remote station is read by edge-first with loop-internal optimized splitting. "
            "Only the schedule weights are optimized against the capacity-relaxed scalar "
            "direct root-closure benchmark."
        ),
        "close4_phase_frame_schedule_weight": float(best["close_factor"]),
        "remote_total_schedule_weight": float(best["remote_total_weight"]),
        "remote_weight_cap": float(remote_weight_cap),
        "max_ratio_allowed": float(max_ratio_allowed),
        "ratio_min": float(np.min(selected_ratios)),
        "ratio_median": float(np.median(selected_ratios)),
        "ratio_mean": float(np.mean(selected_ratios)),
        "ratio_max": float(np.max(selected_ratios)),
        "close_ratio_min": float(np.min(selected_ratios[~remote_mask])),
        "close_ratio_max": float(np.max(selected_ratios[~remote_mask])),
        "remote_ratio_min": float(np.min(selected_ratios[remote_mask])),
        "remote_ratio_max": float(np.max(selected_ratios[remote_mask])),
        "ratio_rows": ratio_rows,
        "photon_budget_diagnostics": budget_diagnostics,
        "top_candidates": top_candidates,
        "direct_weight_info": direct_info,
    }
    return optimized_specs, info


def strict_v1_photon_budget_diagnostics(
    bm: bm_lib.AllClosureBenchmark,
    specs: list[dict[str, object]],
    *,
    close_factor: float,
    remote_total_weight: float,
    remote_weight_cap: float,
    atol: float = 5e-10,
) -> dict[str, object]:
    """Validate and summarize the strict V1 schedule photon-budget profile."""
    demand = np.zeros((bm.n, bm.n), dtype=float)
    active_weight_by_station = np.zeros(bm.n, dtype=float)
    loop_rows: list[dict[str, object]] = []
    remote_weights: list[float] = []
    max_loop_row_error = 0.0

    for spec in specs:
        tri = tuple(spec["tri"])
        if all(station in CORE for station in tri):
            continue
        schedule_weight = float(spec.get("schedule_weight", 0.0))
        remote_weights.append(schedule_weight)
        if schedule_weight < -atol or schedule_weight > remote_weight_cap + atol:
            raise ValueError(f"remote loop {tri} schedule_weight={schedule_weight} violates cap {remote_weight_cap}")

        directed = noncore_directed_fractions(tri, tuple(float(value) for value in spec["split"]))
        stations_with_remote_edges = sorted({station for station, _target in directed})
        for station in stations_with_remote_edges:
            row_sum = sum(value for (origin, _target), value in directed.items() if origin == station)
            expected = station_loop_budget(station)
            row_error = abs(float(row_sum) - expected)
            max_loop_row_error = max(max_loop_row_error, row_error)
            if row_error > atol:
                raise ValueError(
                    f"remote loop {tri} station {station} row sum {row_sum} does not match budget {expected}"
                )
            active_weight_by_station[station] += schedule_weight
        for (origin, target), value in directed.items():
            demand[origin, target] += schedule_weight * float(value)
        loop_rows.append(
            {
                "closure": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
                "schedule_weight": schedule_weight,
                "instantaneous_station_row_sums": {
                    bm.names[station]: float(sum(value for (origin, _target), value in directed.items() if origin == station))
                    for station in stations_with_remote_edges
                },
                "instantaneous_station_row_budgets": {
                    bm.names[station]: float(station_loop_budget(station)) for station in stations_with_remote_edges
                },
            }
        )

    remote_weight_sum = float(np.sum(remote_weights))
    if abs(remote_weight_sum - remote_total_weight) > 1e-8:
        raise ValueError(f"remote schedule weights sum to {remote_weight_sum}, expected {remote_total_weight}")

    station_rows: list[dict[str, object]] = []
    max_active_row_error = 0.0
    max_global_row_fraction = 0.0
    split_profile_rows: list[dict[str, object]] = []
    for station in range(bm.n):
        branch_budget = station_loop_budget(station)
        weighted_row_sum = float(np.sum(demand[station]))
        active_weight = float(active_weight_by_station[station])
        active_denominator = branch_budget * active_weight
        if active_denominator > 0.0:
            active_row_sum = weighted_row_sum / active_denominator
            active_error = abs(active_row_sum - 1.0)
            max_active_row_error = max(max_active_row_error, active_error)
            if active_error > 1e-8:
                raise ValueError(
                    f"station {station} active-normalized split row sums to {active_row_sum}, not 1"
                )
        else:
            active_row_sum = 0.0

        global_denominator = branch_budget * max(remote_total_weight, 1e-300)
        global_row_fraction = weighted_row_sum / global_denominator
        max_global_row_fraction = max(max_global_row_fraction, global_row_fraction)
        if global_row_fraction > 1.0 + 1e-8:
            raise ValueError(
                f"station {station} schedule-averaged row fraction {global_row_fraction} exceeds 1"
            )

        active_profile = {}
        schedule_profile = {}
        for target in range(bm.n):
            if target == station:
                continue
            weighted_value = float(demand[station, target])
            if active_denominator > 0.0 and weighted_value > 0.0:
                active_profile[bm.names[target]] = weighted_value / active_denominator
            if weighted_value > 0.0:
                schedule_profile[bm.names[target]] = weighted_value / global_denominator
                split_profile_rows.append(
                    {
                        "from_station": bm.names[station],
                        "to_station": bm.names[target],
                        "weighted_fraction_sum": weighted_value,
                        "active_normalized_fraction": weighted_value / active_denominator if active_denominator > 0.0 else 0.0,
                        "schedule_averaged_fraction": weighted_value / global_denominator,
                    }
                )

        station_rows.append(
            {
                "station": bm.names[station],
                "group": "core" if station in CORE else "remote",
                "remote_branch_budget": float(branch_budget),
                "remote_active_schedule_weight": active_weight,
                "weighted_remote_branch_row_sum": weighted_row_sum,
                "active_normalized_row_sum": float(active_row_sum),
                "schedule_averaged_row_fraction": float(global_row_fraction),
                "schedule_averaged_idle_fraction": float(max(0.0, 1.0 - global_row_fraction)),
                "active_normalized_profile": active_profile,
                "schedule_averaged_profile": schedule_profile,
            }
        )

    return {
        "status": "passed",
        "interpretation": (
            "Loop rows validate instantaneous photon-budget conservation.  Station rows validate "
            "the schedule-weighted profile after normalizing by each station's active remote-loop "
            "time.  The schedule-averaged row fraction can be below one because a station is idle "
            "during remote-loop slots that do not include it."
        ),
        "close4_phase_frame_schedule_weight": float(close_factor),
        "close4_instantaneous_core_fraction": float(CORE_JOINT_FRACTION),
        "remote_total_schedule_weight": remote_weight_sum,
        "remote_weight_cap": float(remote_weight_cap),
        "remote_weight_min": float(np.min(remote_weights)) if remote_weights else 0.0,
        "remote_weight_max": float(np.max(remote_weights)) if remote_weights else 0.0,
        "max_loop_row_sum_error": float(max_loop_row_error),
        "max_active_normalized_row_sum_error": float(max_active_row_error),
        "max_schedule_averaged_row_fraction": float(max_global_row_fraction),
        "loop_rows": loop_rows,
        "station_rows": station_rows,
        "split_profile_rows": split_profile_rows,
    }


def optimize_direct_root_weights_mean_rms(
    bm: bm_lib.AllClosureBenchmark,
    seed: int = 20260605,
) -> tuple[dict[tuple[int, int, int], float], dict[str, float | str]]:
    """Mean-RMS optimized scheduled direct root-closure weights.

    The total weight is the capacity-relaxed scalar resource count, (N-1)=6 for the
    seven-station, 15-root-closure case, with each scalar closure setting capped
    at unit weight.  Rank-deficient schedules are rejected by the strict
    covariance target.
    """
    triangles = root_independent_triangles(bm.n)
    total_weight, _interpretation = direct_allclosure_total_weight(bm, "capacity_relaxed_scalar")
    rng = np.random.default_rng(seed)

    def fisher_for_weight_vec(weight_vec: np.ndarray) -> np.ndarray:
        return direct_fisher_from_root_weights(bm, dict(zip(triangles, weight_vec)))

    def score(weight_vec: np.ndarray) -> float:
        return -strict_mean_coord_rms(fisher_for_weight_vec(weight_vec))

    best_w = np.full(len(triangles), total_weight / len(triangles), dtype=float)
    best_score = score(best_w)
    for scale in (0.5, 1.0, 2.0):
        for _ in range(700):
            cand = project_capped_simplex(best_w + rng.normal(scale=scale, size=len(best_w)), total_weight)
            value = score(cand)
            if value > best_score:
                best_score = value
                best_w = cand

    for width in (0.30, 0.15, 0.07, 0.03, 0.012):
        improved = True
        while improved:
            improved = False
            for i in range(len(best_w)):
                for j in range(len(best_w)):
                    if i == j:
                        continue
                    movable = min(width, best_w[j], 1.0 - best_w[i])
                    if movable <= 0.0:
                        continue
                    cand = best_w.copy()
                    cand[i] += movable
                    cand[j] -= movable
                    value = score(cand)
                    if value > best_score:
                        best_score = value
                        best_w = cand
                        improved = True

    final_fisher = fisher_for_weight_vec(best_w)
    final_mean = strict_mean_coord_rms(final_fisher)
    weights = dict(zip(triangles, (float(value) for value in best_w)))
    return weights, {
        "objective": "mean_coord_rms",
        "strict_mean_coord_rms": float(final_mean),
        "score": float(best_score),
        "total_weight": float(total_weight),
        "max_per_closure_weight": 1.0,
        "min_weight": float(np.min(best_w)),
        "max_weight": float(np.max(best_w)),
    }


def closure_rows(bm: bm_lib.AllClosureBenchmark, matrices: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    covs = {key: np.linalg.pinv(0.5 * (value + value.T), rcond=1e-12) for key, value in matrices.items()}
    rows: list[dict[str, float | str]] = []
    for tri in itertools.combinations(range(bm.n), 3):
        d = measurement_vector(bm, tri)
        row: dict[str, float | str] = {
            "closure": f"{tri[0] + 1}-{tri[1] + 1}-{tri[2] + 1}",
            "stations": " | ".join(bm.names[i] for i in tri),
            "type": "core_only" if all(i in CORE for i in tri) else "remote_involved",
        }
        for key, cov in covs.items():
            var = float(d @ cov @ d)
            row[f"rms_{key}_rad"] = math.sqrt(max(var, 0.0)) if np.isfinite(var) else math.inf
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


def split_rows(bm: bm_lib.AllClosureBenchmark, p: np.ndarray) -> list[dict[str, float | str]]:
    rows = []
    for i, name_i in enumerate(bm.names):
        for j, name_j in enumerate(bm.names):
            if i == j:
                continue
            rows.append(
                {
                    "from_station": name_i,
                    "to_station": name_j,
                    "fraction": p[i, j],
                    "from_group": "core" if i in CORE else "remote",
                    "to_group": "core" if j in CORE else "remote",
                }
            )
    return rows


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_note(payload: dict, rows: list[dict[str, float | str]], p: np.ndarray, bm: bm_lib.AllClosureBenchmark) -> tuple[Path, Path]:
    tex_path = OUT / "core4_joint_remote_split_note.tex"
    pdf_path = OUT / "core4_joint_remote_split_note.pdf"
    metric_lines = []
    for item in payload["matrix_metrics"]:
        metric_lines.append(
            f"{latex_escape(item['strategy'])} & {item['mean_coord_rms']:.3g} & {item['median_coord_rms']:.3g} & "
            f"{item['max_coord_rms']:.3g} & {item['mean_rms_gain_vs_edge_uniform']:.2f} & "
            f"{item['mean_rms_gain_vs_direct_scheduled']:.2f} \\\\"
        )
    closure_lines = []
    for row in rows:
        closure_lines.append(
            f"{row['closure']} & {latex_escape(str(row['type']))} & "
            f"{float(row['rms_core4_remote_optimized_rad']):.3g} & "
            f"{float(row['gain_core4_remote_optimized_vs_edge_uniform']):.2f} & "
            f"{float(row['gain_core4_remote_optimized_vs_direct_scheduled']):.2f} \\\\"
        )
    split_lines = []
    for i in range(bm.n):
        vals = [p[i, j] for j in range(bm.n) if j != i]
        split_lines.append(
            f"{latex_escape(bm.names[i])} & {sum(vals):.3f} & "
            + ", ".join(f"{v:.3f}" for v in vals)
            + r" \\"
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.65in]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{Core-four joint receiver plus remote optimized splitting}}
\author{{Codex diagnostic note}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Strategy tested}}
The four current Maunakea stations are treated as a near-core subarray.  Each
core station sends a fraction \(1/2\) of its field into a 4-mode joint receiver.
For this diagnostic the core receiver is represented as providing the six
core-core baseline phase Fisher informations with half-station budget and no
additional per-baseline split.  The remaining \(1/2\) of each core station is
split among the three remote stations.  Each remote station uses its full field
for remote-related readout.  The remote split is optimized under the row-budget
constraints.

\section*{{Matrix-level comparison}}
\begin{{center}}
\begin{{tabular}}{{lrrrrr}}
\toprule
strategy & mean RMS & median RMS & max RMS & gain vs edge & gain vs scheduled direct \\
\midrule
{chr(10).join(metric_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Optimized station-side split}}
The entries are row fractions excluding the diagonal.  Core rows sum to \(0.5\)
for the remote readout because their other \(0.5\) is consumed by the 4-mode
core receiver.
\begin{{longtable}}{{lrl}}
\toprule
station & row sum & fractions to other stations \\
\midrule
{chr(10).join(split_lines)}
\bottomrule
\end{{longtable}}

\section*{{Closure-level performance}}
\small
\begin{{longtable}}{{llrrr}}
\toprule
closure & type & RMS & gain vs edge & gain vs scheduled direct \\
\midrule
{chr(10).join(closure_lines)}
\bottomrule
\end{{longtable}}

\section*{{Interpretation}}
This strategy is much more plausible than the all-triangle design with 35
independent 3-mode receivers, because the near-core information is collected by
one shared 4-mode receiver rather than by duplicating receiver noise over many
triangles.  However, it still cannot saturate the capacity-relaxed scalar
direct schedule in the current noise model.  The limiting directions are dominated
by weak remote-involved closures; improving the core block alone does not fix
the far-baseline Fisher bottleneck.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUT, check=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=OUT, check=True)
    return tex_path, pdf_path


def main() -> None:
    bm = bm_lib.AllClosureBenchmark()
    p_fixed = project_remote_split(np.zeros((bm.n, bm.n), dtype=float), bm)
    f_fixed = fisher_for_split(bm, p_fixed)
    candidates = {"fixed_equal_remote": (p_fixed, f_fixed, {"objective": "fixed"})}
    for objective in ("mean_rms", "max_rms", "logdet"):
        p, info = optimize_split(bm, objective)
        candidates[f"optimized_{objective}"] = (p, fisher_for_split(bm, p), info)

    edge_uniform = bm.edge_closure_fisher(bm.uniform_split_matrix())
    direct_weights, direct_info = uniform_direct_root_weights(bm, mode="capacity_relaxed_scalar")
    direct_scheduled = direct_fisher_from_root_weights(bm, direct_weights)
    direct_raw = bm.direct_raw
    ref_edge = bm_lib.stable_metrics(edge_uniform)
    ref_sched = bm_lib.stable_metrics(direct_scheduled)
    candidate_metrics = {}
    for key, (_, fisher, info) in candidates.items():
        metrics = bm_lib.stable_metrics(fisher)
        candidate_metrics[key] = {
            **metrics,
            "mean_rms_gain_vs_edge_uniform": ref_edge["mean_coord_rms"] / metrics["mean_coord_rms"],
            "mean_rms_gain_vs_direct_scheduled": ref_sched["mean_coord_rms"] / metrics["mean_coord_rms"],
            "info": info,
        }
    best_key = min(candidate_metrics, key=lambda key: candidate_metrics[key]["mean_coord_rms"])
    best_p, best_fisher, _ = candidates[best_key]

    matrices = {
        "edge_uniform": edge_uniform,
        "direct_scheduled": direct_scheduled,
        "direct_raw_qfi_upper": direct_raw,
        "core4_remote_fixed": f_fixed,
        "core4_remote_optimized": best_fisher,
    }
    matrix_metrics = []
    for key, fisher in matrices.items():
        metrics = bm_lib.stable_metrics(fisher)
        matrix_metrics.append(
            {
                "strategy": key,
                **metrics,
                "mean_rms_gain_vs_edge_uniform": ref_edge["mean_coord_rms"] / metrics["mean_coord_rms"],
                "mean_rms_gain_vs_direct_scheduled": ref_sched["mean_coord_rms"] / metrics["mean_coord_rms"],
            }
        )
    rows = closure_rows(bm, matrices)

    closure_csv = OUT / "core4_joint_remote_split_closure_gains.csv"
    split_csv = OUT / "core4_joint_remote_split_station_fractions.csv"
    write_csv(closure_csv, rows)
    write_csv(split_csv, split_rows(bm, best_p))

    payload = {
        "case": bm.case.key,
        "station_names": bm.names,
        "best_candidate": best_key,
        "core_joint_fraction": CORE_JOINT_FRACTION,
        "core_remote_fraction": CORE_REMOTE_FRACTION,
        "remote_total_fraction": REMOTE_TOTAL_FRACTION,
        "eps_station": bm_lib.EPS_STATION,
        "eps_pair": bm_lib.EPS_PAIR,
        "eps_direct_extra": bm_lib.EPS_DIRECT_EXTRA,
        "fiber_loss_db_per_km": bm_lib.FIBER_LOSS_DB_PER_KM,
        "candidate_metrics": candidate_metrics,
        "matrix_metrics": matrix_metrics,
        "closure_csv": str(closure_csv),
        "split_csv": str(split_csv),
        "direct_weight_info": direct_info,
        "direct_root_weights": {
            f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}": float(weight)
            for tri, weight in sorted(direct_weights.items())
        },
    }
    json_path = OUT / "core4_joint_remote_split_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    tex_path, pdf_path = write_note(payload, rows, best_p, bm)
    print(json.dumps(payload, indent=2))
    print(tex_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
