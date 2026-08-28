from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE = THIS_DIR.parents[1]
ROOT = WORKSPACE / "18_balanced_10loop_independent_set_20260611"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
NOTES = ROOT / "notes"
LOGS = ROOT / "logs"
for folder in (RESULTS, FIGURES, NOTES, LOGS, LOGS / "mplconfig"):
    folder.mkdir(parents=True, exist_ok=True)

for path in (
    THIS_DIR,
    WORKSPACE / "16_six_station_reduced_from7_20260611" / "code",
    WORKSPACE / "03_figure_generation_code" / "0608_core_modules",
    WORKSPACE / "03_figure_generation_code" / "0608_all_python_snapshot",
):
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

import core4_joint_remote_split_design as core_remote  # noqa: E402
import plot_prl_broadband_clean as base  # noqa: E402
from plot_prl_broadband_blr_realnight import project_enu_baselines  # noqa: E402
import sixstation_base_variants as variants  # noqa: E402


CURRENT_PAYLOAD = (
    WORKSPACE
    / "16_six_station_reduced_from7_20260611"
    / "results"
    / "near_match_direct_split_payload.json"
)

BALANCED_INDEPENDENT_TRIANGLES = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 5),
    (1, 3, 4),
    (1, 4, 5),
    (2, 3, 5),
    (2, 4, 5),
    (3, 4, 5),
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def edge_index(edges: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {edge: idx for idx, edge in enumerate(edges)}


def embed_local_edge_fisher(
    out: np.ndarray,
    local_fisher: np.ndarray,
    subset: list[int],
    local_edges: list[tuple[int, int]],
    global_edge_index: dict[tuple[int, int], int],
) -> None:
    for local_i, (ai, bi) in enumerate(local_edges):
        edge_i = tuple(sorted((subset[ai], subset[bi])))
        gi = global_edge_index[edge_i]
        for local_j, (aj, bj) in enumerate(local_edges):
            edge_j = tuple(sorted((subset[aj], subset[bj])))
            gj = global_edge_index[edge_j]
            out[gi, gj] += local_fisher[local_i, local_j]


def marginalize_local_core_core_edges(local_fisher: np.ndarray, local_edges: list[tuple[int, int]]) -> np.ndarray:
    """Keep only remote-edge information after marginalizing local core-core phases."""
    remote_local = max(max(edge) for edge in local_edges)
    desired = [idx for idx, edge in enumerate(local_edges) if remote_local in edge]
    nuisance = [idx for idx, edge in enumerate(local_edges) if remote_local not in edge]
    out = np.zeros_like(local_fisher)
    fdd = local_fisher[np.ix_(desired, desired)]
    if nuisance:
        fdn = local_fisher[np.ix_(desired, nuisance)]
        fnn = local_fisher[np.ix_(nuisance, nuisance)]
        efficient = fdd - fdn @ np.linalg.pinv(fnn, rcond=1e-12) @ fdn.T
    else:
        efficient = fdd
    for a, ia in enumerate(desired):
        for b, ib in enumerate(desired):
            out[ia, ib] = efficient[a, b]
    return 0.5 * (out + out.T)


def split_gamma_parameters(gamma: np.ndarray | float | None, n_station: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    n_core = len(core_remote.CORE)
    n_remote = len(core_remote.REMOTE)
    if gamma is None:
        core_gamma = np.ones((n_core, n_remote), dtype=float)
        remote_gamma = np.ones((n_remote, n_core), dtype=float)
        kind = "default_all_star"
    else:
        arr = np.asarray(gamma, dtype=float)
        if arr.ndim == 0:
            value = float(np.clip(arr, 0.0, 1.0))
            core_gamma = np.full((n_core, n_remote), value, dtype=float)
            remote_gamma = np.full((n_remote, n_core), value, dtype=float)
            kind = "scalar"
        elif arr.shape == (n_remote,):
            values = np.clip(arr, 0.0, 1.0)
            core_gamma = np.tile(values.reshape(1, n_remote), (n_core, 1))
            remote_gamma = np.tile(values.reshape(n_remote, 1), (1, n_core))
            kind = "by_remote"
        elif arr.shape == (n_station, n_station):
            core_gamma = arr[np.ix_(core_remote.CORE, core_remote.REMOTE)]
            remote_gamma = arr[np.ix_(core_remote.REMOTE, core_remote.CORE)]
            kind = "directed_station_matrix"
        else:
            flat = arr.reshape(-1)
            expected = 2 * n_core * n_remote
            if flat.size != expected:
                raise ValueError(f"gamma must be scalar, {n_remote} remote values, {n_station}x{n_station}, or {expected} directed values")
            core_gamma = flat[: n_core * n_remote].reshape(n_core, n_remote)
            remote_gamma = flat[n_core * n_remote :].reshape(n_remote, n_core)
            kind = "independent_core_remote_and_remote_core"
        core_gamma = np.clip(core_gamma, 0.0, 1.0)
        remote_gamma = np.clip(remote_gamma, 0.0, 1.0)
    return core_gamma, remote_gamma, {
        "gamma_kind": kind,
        "gamma_core_to_remote": core_gamma.tolist(),
        "gamma_remote_to_core": remote_gamma.tolist(),
    }


def remote_star_joint_edge_fisher_matrix(
    bm,
    p: np.ndarray,
    gamma_by_remote: np.ndarray | None = None,
    *,
    core_core_handling: str = "nuisance",
) -> tuple[np.ndarray, dict[str, object]]:
    """Remote-star coherent/joint replacement for core-remote edge-first beats.

    For each remote station r, the receiver takes the three core fractions
    p[c,r] and a single remote fraction sum_c p[r,c].  It performs one joint
    raw phase-frame measurement on the four modes {core1, core2, core3, r}.
    This contains pair-coherent observables such as (core1 + core2) beaten
    against r, while preserving station-side photon budgets.
    """
    variants.configure_six_station_constants()
    global_edges = list(bm.edges)
    global_index = edge_index(global_edges)
    out = np.zeros((len(global_edges), len(global_edges)), dtype=float)
    star_rows: list[dict[str, object]] = []
    core_gamma, remote_gamma, gamma_info = split_gamma_parameters(gamma_by_remote, bm.n)

    old_source = core_remote.ngc.NGC4151
    core_remote.ngc.NGC4151 = variants.fig_run.GOOD_SOURCE
    try:
        for remote_index, remote in enumerate(core_remote.REMOTE):
            core_g = core_gamma[:, remote_index]
            remote_g = remote_gamma[remote_index, :]
            subset = list(core_remote.CORE) + [remote]
            local_edges = base.edge_list(len(subset))
            local_stations = bm.stations[subset]
            local_baselines = np.asarray(
                [local_stations[j] - local_stations[i] for i, j in local_edges],
                dtype=float,
            )
            core_available = np.asarray([float(p[core, remote]) for core in core_remote.CORE], dtype=float)
            remote_available = np.asarray([float(p[remote, core]) for core in core_remote.CORE], dtype=float)
            star_core = core_g * core_available
            remote_star = float(np.sum(remote_g * remote_available))
            star_fractions = np.concatenate([star_core, [remote_star]])
            eta = star_fractions * bm.eta[subset]
            noise = star_fractions * variants.fig_run.EPS_STATION_RUN + variants.fig_run.EPS_DIRECT_EXTRA_RUN
            star_rows.append(
                {
                    "remote": bm.names[remote],
                    "gamma_core_to_remote": ",".join(f"{x:.8g}" for x in core_g),
                    "gamma_remote_to_core": ",".join(f"{x:.8g}" for x in remote_g),
                    "subset": "|".join(str(bm.names[i]) for i in subset),
                    "core_to_remote_available_fractions": ",".join(f"{x:.8g}" for x in core_available),
                    "core_to_remote_star_fractions": ",".join(f"{x:.8g}" for x in star_core),
                    "remote_to_core_available_fractions": ",".join(f"{x:.8g}" for x in remote_available),
                    "remote_to_core_joint_available_fraction": float(np.sum(remote_available)),
                    "remote_to_core_joint_star_fraction": float(star_fractions[-1]),
                    "total_star_receiver_input_fraction": float(np.sum(star_fractions)),
                }
            )
            if remote_star > 0.0 and float(np.max(star_core)) > 0.0:
                for lam, freq, total_modes in bm.iter_bands():
                    vgrid, uv_axis = bm.visibility_grid_for_wavelength(lam * 1e9)
                    u_station = core_remote.aug.station_u_modes(freq, bm.diameters[subset])
                    uu_rows, vv_rows = project_enu_baselines(
                        local_baselines,
                        bm.hour_angles,
                        lam,
                        latitude_deg=bm.case.latitude_deg,
                        declination_deg=variants.fig_run.GOOD_SOURCE.dec_deg,
                    )
                    for uu, vv in zip(uu_rows, vv_rows):
                        vlocal = base.interp_vis(vgrid, uv_axis, uu, vv)
                        local_fisher = total_modes * core_remote.raw_edge_phase_fisher_station_u(
                            vlocal,
                            eta,
                            noise,
                            u_station,
                            local_edges,
                        )
                        if core_core_handling == "nuisance":
                            local_fisher = marginalize_local_core_core_edges(local_fisher, local_edges)
                        elif core_core_handling != "full":
                            raise ValueError(f"Unknown core_core_handling={core_core_handling!r}")
                        embed_local_edge_fisher(out, local_fisher, subset, local_edges, global_index)

            for core_idx, core in enumerate(core_remote.CORE):
                core_residual = (1.0 - core_g[core_idx]) * p[core, remote]
                remote_residual = (1.0 - remote_g[core_idx]) * p[remote, core]
                if core_residual > 0.0 and remote_residual > 0.0:
                    edge = tuple(sorted((core, remote)))
                    idx = global_index[edge]
                    out[idx, idx] += core_remote.edge_fisher_from_arrays(
                        bm.edge_arrays[edge],
                        core_residual,
                        remote_residual,
                    )

        for i, j in itertools.combinations(core_remote.REMOTE, 2):
            idx = global_index[(i, j)]
            out[idx, idx] += core_remote.edge_fisher_from_arrays(bm.edge_arrays[(i, j)], p[i, j], p[j, i])
    finally:
        core_remote.ngc.NGC4151 = old_source

    station_totals = {}
    for i in range(bm.n):
        alpha = 0.0
        if i in core_remote.CORE:
            # Alpha is accounted for outside this helper; report only split rows here.
            alpha = 0.0
        station_totals[bm.names[i]] = float(np.sum(p[i]) + alpha)
    return (
        0.5 * (out + out.T),
        {
            "remote_star_receivers": star_rows,
            "split_row_sums": station_totals,
            "gamma_model": "gamma fraction of every core-remote directed split enters a remote-star joint receiver; the residual remains pairwise edge-first",
            "core_core_handling": core_core_handling,
            **gamma_info,
        },
    )


def closure_fisher_from_edge_matrix(bm, edge_fisher: np.ndarray) -> np.ndarray:
    return base.closure_fisher_after_gauge_marginalization(edge_fisher, bm.q_basis, bm.edges, bm.n)


def fisher_current_near(bm, p: np.ndarray, alpha_core: np.ndarray) -> np.ndarray:
    edge_fisher = variants.core_direct_edge_fisher_matrix_alpha(bm, alpha_core)
    edge_fisher += core_remote.remote_edge_fisher_matrix_for_split(bm, p)
    return closure_fisher_from_edge_matrix(bm, edge_fisher)


def fisher_remote_star_near(
    bm,
    p: np.ndarray,
    alpha_core: np.ndarray,
    gamma_by_remote: np.ndarray | None = None,
    core_core_handling: str = "nuisance",
) -> tuple[np.ndarray, dict[str, object]]:
    core_edge = variants.core_direct_edge_fisher_matrix_alpha(bm, alpha_core)
    star_edge, info = remote_star_joint_edge_fisher_matrix(
        bm,
        p,
        gamma_by_remote,
        core_core_handling=core_core_handling,
    )
    return closure_fisher_from_edge_matrix(bm, core_edge + star_edge), info


def root_sigmas(bm, fisher_q: np.ndarray) -> np.ndarray:
    return loop_sigmas_from_q_fisher(bm, fisher_q)


def all_triangle_list(n_station: int) -> list[tuple[int, int, int]]:
    return list(itertools.combinations(range(n_station), 3))


def balanced_independent_triangles(n_station: int) -> list[tuple[int, int, int]]:
    if n_station != 6:
        raise ValueError("The balanced independent loop set is defined for the six-station benchmark.")
    return list(BALANCED_INDEPENDENT_TRIANGLES)


def loop_labels(triangles: list[tuple[int, int, int]]) -> list[str]:
    return ["-".join(f"S{i + 1}" for i in tri) for tri in triangles]


def loop_matrix_q(bm, triangles: list[tuple[int, int, int]]) -> np.ndarray:
    return np.stack(
        [bm.q_basis.T @ core_remote.edge_vector(bm.edges, tri) for tri in triangles],
        axis=0,
    )


def loop_sigmas_from_q_fisher(
    bm,
    fisher_q: np.ndarray,
    triangles: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    eval_triangles = balanced_independent_triangles(bm.n) if triangles is None else triangles
    cov_q = np.linalg.pinv(0.5 * (fisher_q + fisher_q.T), rcond=1e-12)
    loops = loop_matrix_q(bm, eval_triangles)
    cov_loop = loops @ cov_q @ loops.T
    return np.sqrt(np.maximum(np.diag(0.5 * (cov_loop + cov_loop.T)), 1e-300))


def edge_sigmas_for_triangles(
    bm,
    triangles: list[tuple[int, int, int]] | None = None,
    split: np.ndarray | None = None,
) -> np.ndarray:
    eval_triangles = balanced_independent_triangles(bm.n) if triangles is None else triangles
    split_matrix = bm.uniform_split_matrix() if split is None else split
    edge_fisher = bm.edge_fisher_values(split_matrix)
    edge_cov = np.diag(1.0 / np.maximum(edge_fisher, 1e-300))
    sigmas = []
    for tri in eval_triangles:
        c = core_remote.edge_vector(bm.edges, tri)
        sigmas.append(math.sqrt(max(float(c @ edge_cov @ c), 1e-300)))
    return np.asarray(sigmas, dtype=float)


def direct_fisher_from_triangle_weights(
    bm,
    triangles: list[tuple[int, int, int]],
    unit_scalars: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fisher = np.zeros((bm.q_basis.shape[1], bm.q_basis.shape[1]), dtype=float)
    for tri, unit_scalar, weight in zip(triangles, unit_scalars, weights):
        d = bm.q_basis.T @ core_remote.edge_vector(bm.edges, tri)
        fisher += float(weight) * float(unit_scalar) * np.outer(d, d)
    return 0.5 * (fisher + fisher.T)


def direct_schedule_score(edge_sigma: np.ndarray, sigma: np.ndarray) -> float:
    gains = edge_sigma / np.maximum(sigma, 1e-300)
    log_g = np.log(np.maximum(gains, 1e-300))
    below_one = np.maximum(0.0, -log_g)
    return float(np.mean(log_g) + 0.35 * np.min(log_g) - 0.65 * np.var(log_g) - 30.0 * np.mean(below_one * below_one))


def optimize_direct_schedule(bm, edge_sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Optimize all-triangle direct receiver weights under exact per-station budgets."""
    triangles = all_triangle_list(bm.n)
    incidence = np.zeros((bm.n, len(triangles)), dtype=float)
    for col, tri in enumerate(triangles):
        incidence[list(tri), col] = 1.0
    # EPS_DIRECT_EXTRA is zero in the manuscript run, so direct scalar Fisher is linear
    # in the per-triangle station fraction.  Use the unit-fraction scalar as the slope.
    unit_scalars = np.asarray(
        [variants.alltri.triangle_direct_fisher(bm, tri, (1.0, 1.0, 1.0)) for tri in triangles],
        dtype=float,
    )
    uniform = np.full(len(triangles), 1.0 / math.comb(bm.n - 1, 2), dtype=float)
    _, _, vh = np.linalg.svd(incidence, full_matrices=True)
    rank = int(np.linalg.matrix_rank(incidence))
    null = vh[rank:].T

    def feasible(candidate: np.ndarray) -> np.ndarray:
        return np.asarray(candidate, dtype=float)

    def evaluate(weights: np.ndarray) -> tuple[float, np.ndarray]:
        fisher = direct_fisher_from_triangle_weights(bm, triangles, unit_scalars, weights)
        sigma = root_sigmas(bm, fisher)
        return direct_schedule_score(edge_sigma, sigma), sigma

    best = uniform.copy()
    best_score, best_sigma = evaluate(best)
    rng = np.random.default_rng(20260611)
    for scale in (0.02, 0.05, 0.10, 0.18, 0.30):
        for _ in range(1200):
            candidate = feasible(uniform + null @ rng.normal(scale=scale, size=null.shape[1]))
            if np.min(candidate) < -1.0e-12:
                continue
            score, sigma = evaluate(candidate)
            if score > best_score:
                best_score = score
                best = np.maximum(candidate, 0.0)
                best_sigma = sigma

    for width in (0.08, 0.035, 0.015, 0.006):
        improved = True
        passes = 0
        while improved and passes < 5:
            improved = False
            passes += 1
            for idx in range(null.shape[1]):
                direction = null[:, idx]
                for sign in (-1.0, 1.0):
                    candidate = best + sign * width * direction
                    if np.min(candidate) < -1.0e-12:
                        continue
                    score, sigma = evaluate(candidate)
                    if score > best_score + 1.0e-12:
                        best_score = score
                        best = np.maximum(candidate, 0.0)
                        best_sigma = sigma
                        improved = True

    station_sums = incidence @ best
    gains = edge_sigma / np.maximum(best_sigma, 1e-300)
    order = np.argsort(best)[::-1]
    info = {
        "objective": "maximize mean log root-loop SNR gain with worst-loop and variance regularization",
        "constraints": "w_tri >= 0 and sum_{tri contains station i} w_tri = 1 for every station",
        "score": float(best_score),
        "station_weight_sums": {str(bm.names[i]): float(station_sums[i]) for i in range(bm.n)},
        "max_station_weight_error": float(np.max(np.abs(station_sums - 1.0))),
        "snr_gain_vs_edge": ratio_summary(gains),
        "top_triangle_weights": {
            "-".join(f"S{i + 1}" for i in triangles[idx]): float(best[idx])
            for idx in order[:8]
            if best[idx] > 1.0e-8
        },
        "all_triangle_weights": {
            "-".join(f"S{i + 1}" for i in tri): float(weight)
            for tri, weight in zip(triangles, best)
        },
    }
    return best_sigma, best, info


def balanced_independent_loop_direct_schedule(bm) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Equal-budget direct schedule on a balanced set of ten independent triangle closures."""
    selected = balanced_independent_triangles(bm.n)
    all_triangles = all_triangle_list(bm.n)
    weights = np.zeros(len(all_triangles), dtype=float)
    per_loop_weight = 1.0 / 5.0
    for tri in selected:
        weights[all_triangles.index(tri)] = per_loop_weight
    unit_scalars = np.asarray(
        [variants.alltri.triangle_direct_fisher(bm, tri, (1.0, 1.0, 1.0)) for tri in all_triangles],
        dtype=float,
    )
    fisher = direct_fisher_from_triangle_weights(bm, all_triangles, unit_scalars, weights)
    sigma = root_sigmas(bm, fisher)
    incidence = np.zeros((bm.n, len(all_triangles)), dtype=float)
    for col, tri in enumerate(all_triangles):
        incidence[list(tri), col] = 1.0
    station_sums = incidence @ weights
    loop_matrix = []
    for tri in selected:
        d = bm.q_basis.T @ core_remote.edge_vector(bm.edges, tri)
        loop_matrix.append(d)
    info = {
        "description": "Equal photon budget on ten balanced independent triangle-closure coordinates",
        "per_loop_weight": float(per_loop_weight),
        "selected_loops": ["-".join(f"S{i + 1}" for i in tri) for tri in selected],
        "selected_loop_rank": int(np.linalg.matrix_rank(np.stack(loop_matrix, axis=0), tol=1.0e-10)),
        "station_loop_counts": {
            str(bm.names[i]): int(sum(i in tri for tri in selected))
            for i in range(bm.n)
        },
        "station_weight_sums": {
            str(bm.names[i]): float(station_sums[i])
            for i in range(bm.n)
        },
        "max_station_weight_error": float(np.max(np.abs(station_sums - 1.0))),
        "all_triangle_weights": {
            "-".join(f"S{i + 1}" for i in tri): float(weight)
            for tri, weight in zip(all_triangles, weights)
        },
    }
    return sigma, weights, info


def balanced_loop_gain_summary(
    bm,
    triangles: list[tuple[int, int, int]],
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
    near_sigma: np.ndarray,
    alpha_core: np.ndarray,
) -> dict[str, object]:
    labels = loop_labels(triangles)
    near_vs_edge = edge_sigma / np.maximum(near_sigma, 1e-300)
    direct_vs_edge = edge_sigma / np.maximum(direct_sigma, 1e-300)
    near_vs_direct = direct_sigma / np.maximum(near_sigma, 1e-300)
    log_ratio = np.log(np.maximum(near_vs_direct, 1e-300))
    remote_count = np.asarray([sum(station in core_remote.REMOTE for station in tri) for tri in triangles], dtype=int)
    alpha_values = variants.alpha_core_array(alpha_core, bm)
    return {
        "objective": "match_direct_balanced_10loop_independent_set",
        "loop_set": "balanced_10loop_independent",
        "loops": labels,
        "alpha": float(np.mean(alpha_values)),
        "alpha_core": [float(x) for x in alpha_values],
        "near_snr_gain_vs_edge": ratio_summary(near_vs_edge),
        "direct_target_snr_gain_vs_edge": ratio_summary(direct_vs_edge),
        "near_over_direct_snr": ratio_summary(near_vs_direct),
        "rms_log_near_over_direct_snr": float(np.sqrt(np.mean(log_ratio * log_ratio))),
        "mean_log_near_over_direct_snr": float(np.mean(log_ratio)),
        "var_log_near_over_direct_snr": float(np.var(log_ratio)),
        "n_near_below_edge": int(np.sum(near_vs_edge < 1.0 - 1.0e-9)),
        "n_near_below_direct": int(np.sum(near_vs_direct < 1.0 - 1.0e-9)),
        "core_only_near_over_direct_mean": float(np.mean(near_vs_direct[remote_count == 0])),
        "one_remote_near_over_direct_mean": float(np.mean(near_vs_direct[remote_count == 1])),
        "two_remote_near_over_direct_mean": float(np.mean(near_vs_direct[remote_count == 2])),
        "all_loop_near_snr_gains_vs_edge": {label: float(value) for label, value in zip(labels, near_vs_edge)},
        "all_loop_direct_snr_gains_vs_edge": {label: float(value) for label, value in zip(labels, direct_vs_edge)},
        "all_loop_near_over_direct_snr": {label: float(value) for label, value in zip(labels, near_vs_direct)},
    }


def score_match_balanced_direct(
    *,
    near_sigma: np.ndarray,
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
    variance_lambda: float,
) -> float:
    near_vs_edge = edge_sigma / np.maximum(near_sigma, 1e-300)
    if float(np.min(near_vs_edge)) < 1.0 - 1e-10:
        return -math.inf
    near_vs_direct = direct_sigma / np.maximum(near_sigma, 1e-300)
    log_ratio = np.log(np.maximum(near_vs_direct, 1e-300))
    mse = float(np.mean(log_ratio * log_ratio))
    variance = float(np.var(log_ratio))
    max_abs = float(np.max(np.abs(log_ratio)))
    mean_bias = float(abs(np.mean(log_ratio)))
    min_log = float(np.min(log_ratio))
    max_over = float(max(0.0, np.max(log_ratio)))
    return min_log - 0.20 * max_over * max_over - variance_lambda * variance - 0.05 * mse - 0.02 * mean_bias - 0.02 * max_abs * max_abs


def optimize_balanced10_near_split(
    bm,
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    variant = variants.VARIANTS[0]
    triangles = balanced_independent_triangles(bm.n)
    rng = np.random.default_rng(20260612)
    raw0 = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw0, -np.inf)
    active = [(i, j) for i in core_remote.CORE for j in core_remote.REMOTE]
    active += [(i, j) for i in core_remote.REMOTE for j in range(bm.n) if i != j]
    core_cache: dict[tuple[float, ...], np.ndarray] = {}
    counts = {"core_recomputes": 0, "cached_core_hits": 0, "split_evaluations": 0}

    def core_for_alpha(alpha_core: np.ndarray) -> np.ndarray:
        key = tuple(float(f"{x:.8f}") for x in alpha_core)
        cached = core_cache.get(key)
        if cached is not None:
            counts["cached_core_hits"] += 1
            return cached
        counts["core_recomputes"] += 1
        core_edge = variants.core_direct_edge_fisher_matrix_alpha(bm, alpha_core)
        core_cache[key] = core_edge
        return core_edge

    def evaluate(raw: np.ndarray, raw_alpha: np.ndarray, core_edge: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        counts["split_evaluations"] += 1
        alpha_core = variants.alpha_vector_from_raw(raw_alpha, variant)
        if core_edge is None:
            core_edge = core_for_alpha(alpha_core)
        fisher_q, p = variants.fisher_for_candidate_with_core(bm, core_edge, raw, alpha_core)
        near_sigma = loop_sigmas_from_q_fisher(bm, fisher_q, triangles)
        score = score_match_balanced_direct(
            near_sigma=near_sigma,
            edge_sigma=edge_sigma,
            direct_sigma=direct_sigma,
            variance_lambda=variant.variance_lambda,
        )
        return score, near_sigma, alpha_core, p

    starts: list[tuple[str, np.ndarray, np.ndarray]] = []
    for alpha0 in (0.05, 0.08, 0.12, 0.20, 0.35, 0.50):
        starts.append((f"uniform_alpha_{alpha0:g}", raw0.copy(), variants.raw_vector_from_alpha(alpha0, variant, bm)))
    for previous_payload in (
        CURRENT_PAYLOAD,
        WORKSPACE / "17_remote_star_joint_near_20260611" / "results" / "remote_star_joint_near_summary.json",
        WORKSPACE / "11_near_match_direct_physical_20260611" / "results" / "near_match_direct_split_payload.json",
        WORKSPACE / "10_varlog_lam2_full_rml_20260610" / "results" / "varlog_lam2_split_payload.json",
    ):
        if not previous_payload.exists():
            continue
        try:
            payload = json.loads(previous_payload.read_text())
            previous_p = np.asarray(payload.get("split_matrix", payload.get("summary", {}).get("split_matrix")), dtype=float)
            if previous_p.shape != (bm.n, bm.n):
                continue
            previous_alpha = np.asarray(payload.get("alpha_core", [float(payload.get("alpha", 0.5))] * len(core_remote.CORE)), dtype=float)
            starts.append(
                (
                    f"previous_{previous_payload.parents[1].name}",
                    variants.raw_logits_from_split(previous_p, previous_alpha, bm),
                    variants.raw_vector_from_alpha(previous_alpha, variant, bm),
                )
            )
        except Exception:
            continue

    best_score = -math.inf
    best_sigma: np.ndarray | None = None
    best_alpha: np.ndarray | None = None
    best_p: np.ndarray | None = None
    best_raw: np.ndarray | None = None
    best_raw_alpha: np.ndarray | None = None
    best_start = ""

    def try_candidate(raw: np.ndarray, raw_alpha: np.ndarray, *, core_edge: np.ndarray | None = None, start_name: str = "") -> bool:
        nonlocal best_score, best_sigma, best_alpha, best_p, best_raw, best_raw_alpha, best_start
        score, sigma, alpha_core, p = evaluate(raw, raw_alpha, core_edge)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_sigma = sigma
            best_alpha = alpha_core.copy()
            best_p = p
            best_raw = raw.copy()
            best_raw_alpha = raw_alpha.copy()
            if start_name:
                best_start = start_name
            return True
        return False

    for start_name, raw, raw_alpha in starts:
        try_candidate(raw, raw_alpha, start_name=start_name)

    alpha_centers = (0.035, 0.05, 0.07, 0.10, 0.15, 0.24, 0.38)
    for alpha0 in alpha_centers:
        raw_alpha_center = variants.raw_vector_from_alpha(alpha0, variant, bm)
        for _ in range(6):
            raw_alpha = raw_alpha_center + rng.normal(scale=0.80, size=len(core_remote.CORE))
            alpha_core = variants.alpha_vector_from_raw(raw_alpha, variant)
            core_edge = core_for_alpha(alpha_core)
            for scale in (0.30, 0.65, 1.15, 2.00):
                for _ in range(10):
                    candidate = raw0 + rng.normal(scale=scale, size=(bm.n, bm.n))
                    np.fill_diagonal(candidate, -np.inf)
                    try_candidate(candidate, raw_alpha, core_edge=core_edge, start_name=f"random_alpha_{alpha0:g}_scale_{scale:g}")

    if best_raw is None or best_raw_alpha is None:
        raise RuntimeError("No balanced-10 near split satisfied near/edge >= 1 during initialization.")

    for outer in range(3):
        for width in (0.90, 0.45, 0.22, 0.10, 0.045, 0.020):
            improved = True
            passes = 0
            while improved and passes < 5:
                improved = False
                passes += 1
                for alpha_idx in range(len(core_remote.CORE)):
                    for sign in (-1.0, 1.0):
                        candidate_alpha = best_raw_alpha.copy()
                        candidate_alpha[alpha_idx] += sign * width
                        if try_candidate(best_raw, candidate_alpha, start_name=f"alpha_coord_outer_{outer}"):
                            improved = True
        assert best_alpha is not None
        core_edge = core_for_alpha(best_alpha)
        for width in (1.00, 0.55, 0.28, 0.13, 0.060, 0.026, 0.011):
            improved = True
            passes = 0
            while improved and passes < 8:
                improved = False
                passes += 1
                for i, j in active:
                    for sign in (-1.0, 1.0):
                        candidate = best_raw.copy()
                        candidate[i, j] += sign * width
                        if try_candidate(candidate, best_raw_alpha, core_edge=core_edge, start_name=f"split_coord_outer_{outer}"):
                            improved = True
        core_edge = core_for_alpha(variants.alpha_vector_from_raw(best_raw_alpha, variant))
        for scale in (0.045, 0.090, 0.18):
            for _ in range(90):
                candidate = best_raw + rng.normal(scale=scale, size=(bm.n, bm.n))
                np.fill_diagonal(candidate, -np.inf)
                try_candidate(candidate, best_raw_alpha, core_edge=core_edge, start_name=f"local_random_outer_{outer}")

    assert best_sigma is not None
    assert best_alpha is not None
    assert best_p is not None
    info = balanced_loop_gain_summary(bm, triangles, edge_sigma, direct_sigma, best_sigma, best_alpha)
    info.update(
        {
            "score": float(best_score),
            "objective_formula": (
                "maximize min_l log(SNR_near/SNR_direct_balanced)_l with variance and overshoot penalties; "
                "candidates with any balanced-loop SNR_near/SNR_edge < 1 are infeasible"
            ),
            "match_variance_lambda": float(variant.variance_lambda),
            "alpha_bounds": [float(variant.alpha_min), float(variant.alpha_max)],
            "best_start": best_start,
            "optimization_counts": counts,
            "n_cached_core_blocks": int(len(core_cache)),
        }
    )
    return best_p, best_alpha, best_sigma, info


def ratio_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def make_loop_rows(
    bm,
    sigmas: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    triangles = balanced_independent_triangles(bm.n)
    rows: list[dict[str, object]] = []
    edge = sigmas["edge_uniform"]
    direct = sigmas["direct_balanced_10loop"]
    for idx, tri in enumerate(triangles):
        n_remote = sum(station in core_remote.REMOTE for station in tri)
        row: dict[str, object] = {
            "loop": "-".join(f"S{i + 1}" for i in tri),
            "stations": " | ".join(str(bm.names[i]) for i in tri),
            "loop_class": "core_only" if n_remote == 0 else ("one_remote" if n_remote == 1 else "two_remote"),
        }
        for key, sigma in sigmas.items():
            row[f"rms_{key}_rad"] = float(sigma[idx])
            row[f"snr_gain_{key}_vs_edge"] = float(edge[idx] / max(float(sigma[idx]), 1e-300))
            row[f"snr_ratio_{key}_vs_direct"] = float(direct[idx] / max(float(sigma[idx]), 1e-300))
        rows.append(row)
    return rows


def plot_loop_gains(rows: list[dict[str, object]]) -> None:
    labels = [str(row["loop"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.12
    nuisance_free_gain = math.sqrt(15.0 / 10.0)
    fig, ax = plt.subplots(figsize=(10.4, 3.9), constrained_layout=True)
    for offset, key, label, color in (
        (-3.0 * width, "direct_physical", "direct all-20", "#d00000"),
        (-2.0 * width, "direct_balanced_10loop", "direct balanced 10-loop", "#ff5a5f"),
        (-1.0 * width, "direct_optimized_schedule", "direct optimized", "#9d0208"),
        (0.0 * width, "current_near", "balanced near", "#f77f00"),
        (1.0 * width, "remote_star_scalar", "scalar gamma star", "#2a9d8f"),
        (2.0 * width, "remote_star_independent", "independent gamma star", "#6a4c93"),
        (3.0 * width, "edge_uniform", "edge-first", "#0077b6"),
    ):
        values = [float(row[f"snr_gain_{key}_vs_edge"]) for row in rows]
        ax.bar(x + offset, values, width=width, label=label, color=color)
    ax.axhline(1.0, color="0.15", lw=0.9, ls="--")
    ax.axhline(
        nuisance_free_gain,
        color="0.25",
        lw=1.0,
        ls=":",
        label=r"nuisance-free DOF $\sqrt{15/10}$",
    )
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.set_title("Six-station balanced independent closures: coherent remote-star receiver test", fontsize=11)
    ax.grid(True, axis="y", color="0.88", lw=0.7)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.savefig(FIGURES / "remote_star_joint_loop_gains.png", dpi=240)
    fig.savefig(FIGURES / "remote_star_joint_loop_gains.pdf")
    plt.close(fig)


def match_score(edge_sigma: np.ndarray, direct_sigma: np.ndarray, candidate_sigma: np.ndarray) -> float:
    ratio = direct_sigma / np.maximum(candidate_sigma, 1e-300)
    edge_gain = edge_sigma / np.maximum(candidate_sigma, 1e-300)
    if float(np.min(edge_gain)) < 1.0 - 1.0e-10:
        return -math.inf
    log_ratio = np.log(np.maximum(ratio, 1e-300))
    overshoot = np.maximum(0.0, log_ratio)
    return float(
        -np.mean(log_ratio * log_ratio)
        - 0.25 * np.var(log_ratio)
        - 0.20 * np.mean(overshoot * overshoot)
    )


def gamma_scan_rows(
    bm,
    p: np.ndarray,
    alpha_core: np.ndarray,
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, dict[str, object]]:
    rows: list[dict[str, object]] = []
    best_score = -math.inf
    best_gamma = 0.0
    best_sigma: np.ndarray | None = None
    best_info: dict[str, object] = {}
    for gamma in np.linspace(0.0, 1.0, 21):
        fisher, info = fisher_remote_star_near(
            bm,
            p,
            alpha_core,
            np.full(len(core_remote.REMOTE), float(gamma), dtype=float),
        )
        sigma = root_sigmas(bm, fisher)
        ratio = direct_sigma / np.maximum(sigma, 1e-300)
        edge_gain = edge_sigma / np.maximum(sigma, 1e-300)
        score = match_score(edge_sigma, direct_sigma, sigma)
        row = {
            "gamma": float(gamma),
            "score": float(score),
            "snr_gain_vs_edge_min": float(np.min(edge_gain)),
            "snr_gain_vs_edge_mean": float(np.mean(edge_gain)),
            "snr_ratio_vs_direct_min": float(np.min(ratio)),
            "snr_ratio_vs_direct_mean": float(np.mean(ratio)),
            "snr_ratio_vs_direct_median": float(np.median(ratio)),
            "snr_ratio_vs_direct_max": float(np.max(ratio)),
            "rms_log_ratio_vs_direct": float(np.sqrt(np.mean(np.log(np.maximum(ratio, 1e-300)) ** 2))),
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_gamma = float(gamma)
            best_sigma = sigma
            best_info = info
    assert best_sigma is not None
    return rows, np.full(len(core_remote.REMOTE), best_gamma, dtype=float), best_sigma, best_info


def flatten_gamma(core_gamma: np.ndarray, remote_gamma: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(core_gamma, dtype=float).reshape(-1), np.asarray(remote_gamma, dtype=float).reshape(-1)])


def independent_gamma_search(
    bm,
    p: np.ndarray,
    alpha_core: np.ndarray,
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
    start_gamma: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, dict[str, object]]:
    core_start, remote_start, _ = split_gamma_parameters(start_gamma, bm.n)
    best = flatten_gamma(core_start, remote_start)
    rows: list[dict[str, object]] = []
    eval_id = 0

    def evaluate(candidate: np.ndarray, tag: str) -> tuple[float, np.ndarray, dict[str, object]]:
        nonlocal eval_id
        candidate = np.clip(np.asarray(candidate, dtype=float), 0.0, 1.0)
        fisher, info = fisher_remote_star_near(bm, p, alpha_core, candidate, core_core_handling="nuisance")
        sigma = root_sigmas(bm, fisher)
        ratio = direct_sigma / np.maximum(sigma, 1e-300)
        edge_gain = edge_sigma / np.maximum(sigma, 1e-300)
        score = match_score(edge_sigma, direct_sigma, sigma)
        rows.append(
            {
                "eval": eval_id,
                "tag": tag,
                "score": float(score),
                "snr_gain_vs_edge_min": float(np.min(edge_gain)),
                "snr_gain_vs_edge_mean": float(np.mean(edge_gain)),
                "snr_ratio_vs_direct_min": float(np.min(ratio)),
                "snr_ratio_vs_direct_mean": float(np.mean(ratio)),
                "snr_ratio_vs_direct_median": float(np.median(ratio)),
                "snr_ratio_vs_direct_max": float(np.max(ratio)),
                "rms_log_ratio_vs_direct": float(np.sqrt(np.mean(np.log(np.maximum(ratio, 1e-300)) ** 2))),
                "gamma_vector": ",".join(f"{x:.8g}" for x in candidate),
            }
        )
        eval_id += 1
        return score, sigma, info

    best_score, best_sigma, best_info = evaluate(best, "scalar_start")
    for width in (0.08, 0.035, 0.015):
        improved = True
        passes = 0
        while improved and passes < 2:
            improved = False
            passes += 1
            for idx in range(best.size):
                for sign in (-1.0, 1.0):
                    candidate = best.copy()
                    candidate[idx] = float(np.clip(candidate[idx] + sign * width, 0.0, 1.0))
                    if abs(candidate[idx] - best[idx]) < 1e-14:
                        continue
                    score, sigma, info = evaluate(candidate, f"coord_w{width:g}_i{idx}")
                    if score > best_score + 1e-12:
                        best_score = score
                        best = candidate
                        best_sigma = sigma
                        best_info = info
                        improved = True
    best_info = dict(best_info)
    best_info["independent_gamma_best_score"] = float(best_score)
    best_info["independent_gamma_vector"] = [float(x) for x in best]
    return rows, best, best_sigma, best_info


def write_note(payload: dict[str, object]) -> None:
    lines = [
        "# Balanced 10-Loop Remote-Star Joint Near Receiver Attempt",
        "",
        "This run keeps the six-station case obtained by dropping the original station 1 and relabeling old S2-S7 as S1-S6.",
        "",
        "The plotted loop basis is no longer the default root-loop basis.  It is the balanced independent set `{123, 124, 125, 134, 136, 245, 256, 346, 356, 456}`.  Each station appears in five selected loops, so assigning weight 0.2 to every selected direct-closure receiver uses exactly one unit of station-side photon budget at every station.",
        "",
        "The near split is re-optimized against this balanced 10-loop direct target.  Its objective keeps every selected loop at or above uniform edge-first and then matches `SNR_near/SNR_direct_balanced` as close to one as possible with a variance penalty.",
        "",
        "Current near treats every remote-involved baseline as pairwise edge-first after the compact-core joint receiver.  The new test replaces the three pairwise core-remote beats at each remote station by one local joint receiver on `{core1, core2, core3, remote}`.  This is a physically budget-conserving way to include coherent-sum observables such as `(core1 + core2)` beaten with the remote field.",
        "",
        "A `gamma` parameter controls how much of each core-remote directed split enters the remote-star joint receiver; the residual stays in ordinary pairwise edge-first channels.  The scalar scan uses one common gamma for every such directed split, while the independent search allows separate core-to-remote and remote-to-core gamma values.  Both gamma optimizations use the balanced 10-loop direct target and reject candidates with any selected-loop gain below edge-first.",
        "",
        "For the reported scalar and independent remote-star columns, the local core-core phases inside each remote-star receiver are treated as nuisance parameters and Schur-complemented out before embedding the Fisher block.  This avoids double-counting compact-core phase information already supplied by the compact-core joint receiver.  The unrestricted full-star result is kept only as a diagnostic in the JSON summary.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(payload["summary"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "If the remote-star column improves over current near, the effect is genuine evidence that the edge-first proxy was leaving Fisher information on the table for remote-involved loops.  If it does not, the limiting factor is more likely the station-side budget allocation or the physical split target, not just the lack of coherent addition.",
        "",
    ]
    (NOTES / "remote_star_joint_near_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    variants.configure_six_station_constants()
    bm = variants.configure_six_benchmark()
    eval_triangles = balanced_independent_triangles(bm.n)
    edge_sigma = edge_sigmas_for_triangles(bm, eval_triangles)
    with variants.fig_run.morph.patched_variant(variants.fig_run.GOOD_VARIANT), variants.fig_run.ngc.patched_source(
        variants.fig_run.GOOD_SOURCE
    ):
        direct_fisher, direct_info = variants.physical_direct_alltriangle_fisher(bm)
        direct_balanced_sigma, direct_balanced_weights, direct_balanced_info = balanced_independent_loop_direct_schedule(bm)
        p, alpha_core, near_opt_sigma, near_opt_info = optimize_balanced10_near_split(
            bm,
            edge_sigma,
            direct_balanced_sigma,
        )
        current_fisher = fisher_current_near(bm, p, alpha_core)
        full_star_fisher, full_star_info = fisher_remote_star_near(
            bm,
            p,
            alpha_core,
            np.ones(len(core_remote.REMOTE), dtype=float),
            core_core_handling="full",
        )

    sigmas = {
        "edge_uniform": edge_sigma,
        "direct_physical": root_sigmas(bm, direct_fisher),
        "current_near": root_sigmas(bm, current_fisher),
        "remote_star_full": root_sigmas(bm, full_star_fisher),
        "direct_balanced_10loop": direct_balanced_sigma,
    }
    with variants.fig_run.morph.patched_variant(variants.fig_run.GOOD_VARIANT), variants.fig_run.ngc.patched_source(
        variants.fig_run.GOOD_SOURCE
    ):
        direct_opt_sigma, direct_opt_weights, direct_opt_info = optimize_direct_schedule(bm, sigmas["edge_uniform"])
    sigmas["direct_optimized_schedule"] = direct_opt_sigma
    with variants.fig_run.morph.patched_variant(variants.fig_run.GOOD_VARIANT), variants.fig_run.ngc.patched_source(
        variants.fig_run.GOOD_SOURCE
    ):
        scan_rows, scalar_gamma, scalar_sigma, scalar_star_info = gamma_scan_rows(
            bm,
            p,
            alpha_core,
            sigmas["edge_uniform"],
            sigmas["direct_balanced_10loop"],
        )
        independent_rows, independent_gamma, independent_sigma, independent_star_info = independent_gamma_search(
            bm,
            p,
            alpha_core,
            sigmas["edge_uniform"],
            sigmas["direct_balanced_10loop"],
            scalar_gamma,
        )
    sigmas["remote_star_scalar"] = scalar_sigma
    sigmas["remote_star_independent"] = independent_sigma
    rows = make_loop_rows(bm, sigmas)
    write_csv(RESULTS / "remote_star_joint_loop_gains.csv", rows)
    write_csv(RESULTS / "remote_star_gamma_scan.csv", scan_rows)
    write_csv(RESULTS / "remote_star_independent_gamma_search.csv", independent_rows)
    plot_loop_gains(rows)
    near_payload = {
        "loop_set": "balanced_10loop_independent",
        "loop_set_triangles": loop_labels(eval_triangles),
        "source_start_payload": str(CURRENT_PAYLOAD),
        "alpha": float(np.mean(alpha_core)),
        "alpha_core": [float(x) for x in alpha_core],
        "split_matrix": np.asarray(p, dtype=float).tolist(),
        "summary": near_opt_info,
    }
    (RESULTS / "balanced10_near_split_payload.json").write_text(json.dumps(near_payload, indent=2) + "\n")

    summary: dict[str, object] = {
        "case": bm.case.key,
        "n_station": int(bm.n),
        "n_balanced_loops": int(len(rows)),
        "loop_set": "balanced_10loop_independent",
        "loop_set_triangles": loop_labels(eval_triangles),
        "source_start_payload": str(CURRENT_PAYLOAD),
        "alpha_core": [float(x) for x in alpha_core],
        "station_budget_total_minmax": [
            float(np.min(np.sum(p, axis=1) + variants.alpha_by_station(alpha_core, bm))),
            float(np.max(np.sum(p, axis=1) + variants.alpha_by_station(alpha_core, bm))),
        ],
        "direct_weight_info": direct_info,
        "direct_balanced_10loop_info": direct_balanced_info,
        "direct_balanced_10loop_weights": [float(x) for x in direct_balanced_weights],
        "direct_optimized_schedule_info": direct_opt_info,
        "direct_optimized_schedule_weights": [float(x) for x in direct_opt_weights],
        "balanced10_near_split_info": near_opt_info,
        "balanced10_near_split_payload": str(RESULTS / "balanced10_near_split_payload.json"),
        "full_star_info": full_star_info,
        "scalar_star_info": scalar_star_info,
        "scalar_gamma_by_remote": [float(x) for x in scalar_gamma],
        "independent_star_info": independent_star_info,
        "independent_gamma_vector": [float(x) for x in independent_gamma],
        "snr_gain_vs_edge": {
            key: ratio_summary(np.asarray([float(row[f"snr_gain_{key}_vs_edge"]) for row in rows]))
            for key in (
                "direct_physical",
                "direct_balanced_10loop",
                "direct_optimized_schedule",
                "current_near",
                "remote_star_scalar",
                "remote_star_independent",
                "remote_star_full",
            )
        },
        "snr_ratio_vs_direct": {
            key: ratio_summary(np.asarray([float(row[f"snr_ratio_{key}_vs_direct"]) for row in rows]))
            for key in (
                "direct_balanced_10loop",
                "direct_optimized_schedule",
                "current_near",
                "remote_star_scalar",
                "remote_star_independent",
                "remote_star_full",
            )
        },
        "gamma_scan_csv": str(RESULTS / "remote_star_gamma_scan.csv"),
        "independent_gamma_search_csv": str(RESULTS / "remote_star_independent_gamma_search.csv"),
        "loop_rows_csv": str(RESULTS / "remote_star_joint_loop_gains.csv"),
        "figure_png": str(FIGURES / "remote_star_joint_loop_gains.png"),
    }
    out = {
        "summary": summary,
        "rows": rows,
    }
    (RESULTS / "remote_star_joint_near_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    write_note(out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
