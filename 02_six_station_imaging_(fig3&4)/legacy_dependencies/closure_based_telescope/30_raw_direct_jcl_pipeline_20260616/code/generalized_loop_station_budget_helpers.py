from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import rawje_balanced10_helpers as h


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

OLD_LOOP_GAINS = (
    ROOT.parent
    / "18_balanced_10loop_independent_set_20260611"
    / "results"
    / "remote_star_joint_loop_gains.csv"
)

LOCAL_EDGES = h.base.edge_list(3)


@dataclass(frozen=True)
class PreparedTriangle:
    tri: tuple[int, int, int]
    subset: np.ndarray
    eta_local: np.ndarray
    global_edge_indices: tuple[int, int, int]
    u_local: np.ndarray
    vis_local: np.ndarray


@dataclass(frozen=True)
class PreparedBalanced10:
    bm: h.RawJeBenchmark
    triangles: tuple[tuple[int, int, int], ...]
    incident_loops: tuple[tuple[int, ...], ...]
    total_modes: np.ndarray
    prepared_triangles: tuple[PreparedTriangle, ...]
    edge_sigma: np.ndarray


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ratio_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def schedule_score(gains: np.ndarray) -> float:
    log_g = np.log(np.maximum(np.asarray(gains, dtype=float), 1.0e-300))
    below_one = np.maximum(0.0, -log_g)
    return float(
        np.mean(log_g)
        + 0.35 * np.min(log_g)
        - 0.65 * np.var(log_g)
        - 30.0 * np.mean(below_one * below_one)
    )


def loop_class(tri: tuple[int, int, int]) -> str:
    n_remote = sum(station in h.REMOTE for station in tri)
    if n_remote == 0:
        return "core"
    if n_remote == 1:
        return "one remote"
    return "two remote"


def prepare_balanced10_case() -> PreparedBalanced10:
    bm = h.make_benchmark()
    triangles = tuple(tuple(tri) for tri in h.BALANCED10)
    incident_loops = tuple(
        tuple(loop_idx for loop_idx, tri in enumerate(triangles) if station in tri)
        for station in range(bm.n)
    )
    total_modes = np.asarray([sample.total_modes for sample in bm.samples], dtype=float)
    edge_to_index = {edge: idx for idx, edge in enumerate(bm.edges)}
    prepared: list[PreparedTriangle] = []
    for tri in triangles:
        subset = np.asarray(tri, dtype=int)
        eta_local = np.asarray(bm.eta[subset], dtype=float)
        global_edge_indices = tuple(edge_to_index[(tri[i], tri[j])] for i, j in LOCAL_EDGES)
        u_local = np.stack([np.asarray(sample.u_station[subset], dtype=float) for sample in bm.samples], axis=0)
        vis_local = np.stack(
            [
                np.asarray(
                    [sample.vtrue[edge_to_index[(tri[i], tri[j])]] for i, j in LOCAL_EDGES],
                    dtype=complex,
                )
                for sample in bm.samples
            ],
            axis=0,
        )
        prepared.append(
            PreparedTriangle(
                tri=tri,
                subset=subset,
                eta_local=eta_local,
                global_edge_indices=global_edge_indices,
                u_local=u_local,
                vis_local=vis_local,
            )
        )
    edge_sigma = h.loop_sigmas(bm, h.uniform_edge_fisher(bm))
    return PreparedBalanced10(
        bm=bm,
        triangles=triangles,
        incident_loops=incident_loops,
        total_modes=total_modes,
        prepared_triangles=tuple(prepared),
        edge_sigma=edge_sigma,
    )


def uniform_station_schedule(case: PreparedBalanced10) -> np.ndarray:
    schedule = np.zeros((case.bm.n, len(case.triangles)), dtype=float)
    for station, loop_ids in enumerate(case.incident_loops):
        schedule[station, list(loop_ids)] = 1.0 / float(len(loop_ids))
    return schedule


def random_station_schedule(case: PreparedBalanced10, rng: np.random.Generator, alpha: float) -> np.ndarray:
    schedule = np.zeros((case.bm.n, len(case.triangles)), dtype=float)
    for station, loop_ids in enumerate(case.incident_loops):
        schedule[station, list(loop_ids)] = rng.dirichlet(np.full(len(loop_ids), float(alpha), dtype=float))
    return schedule


def validate_station_schedule(case: PreparedBalanced10, schedule: np.ndarray, tol: float = 1.0e-9) -> None:
    arr = np.asarray(schedule, dtype=float)
    if arr.shape != (case.bm.n, len(case.triangles)):
        raise ValueError(f"schedule shape {arr.shape} does not match {(case.bm.n, len(case.triangles))}")
    if float(np.min(arr)) < -tol:
        raise ValueError("station schedule contains a negative entry")
    for station, loop_ids in enumerate(case.incident_loops):
        inactive = sorted(set(range(len(case.triangles))) - set(loop_ids))
        if inactive and float(np.max(np.abs(arr[station, inactive]))) > 5.0 * tol:
            raise ValueError(f"station S{station + 1} has weight on a loop it does not participate in")
        if abs(float(np.sum(arr[station, list(loop_ids)])) - 1.0) > 5.0 * tol:
            raise ValueError(f"station S{station + 1} row sum is not one")


def symmetric_schedule_from_scalar_weight(case: PreparedBalanced10, weight: float = 0.2) -> np.ndarray:
    schedule = np.zeros((case.bm.n, len(case.triangles)), dtype=float)
    for loop_idx, tri in enumerate(case.triangles):
        for station in tri:
            schedule[station, loop_idx] = float(weight)
    validate_station_schedule(case, schedule)
    return schedule


def corrected_raw_fisher_from_station_schedule(
    case: PreparedBalanced10,
    station_schedule: np.ndarray,
) -> np.ndarray:
    validate_station_schedule(case, station_schedule)
    bm = case.bm
    edge = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
    station_eps = float(h.fig_run.EPS_STATION_RUN)
    direct_extra = float(h.fig_run.EPS_DIRECT_EXTRA_RUN)
    for loop_idx, prepared in enumerate(case.prepared_triangles):
        fractions = np.asarray([station_schedule[station, loop_idx] for station in prepared.tri], dtype=float)
        if float(np.max(fractions)) <= 1.0e-12:
            continue
        eta_local = fractions * prepared.eta_local
        noise_local = fractions * station_eps + direct_extra
        local_edge = np.zeros((len(LOCAL_EDGES), len(LOCAL_EDGES)), dtype=float)
        for sample_idx, total_modes in enumerate(case.total_modes):
            local_edge += float(total_modes) * h.split_sim.raw_edge_phase_fisher_station_u(
                prepared.vis_local[sample_idx],
                eta_local,
                noise_local,
                prepared.u_local[sample_idx],
                LOCAL_EDGES,
            )
        local_edge = 0.5 * (local_edge + local_edge.T)
        for local_i, global_i in enumerate(prepared.global_edge_indices):
            for local_j, global_j in enumerate(prepared.global_edge_indices):
                edge[global_i, global_j] += float(local_edge[local_i, local_j])
    return h.closure_fisher_from_edge(bm, 0.5 * (edge + edge.T))


def gains_from_fisher(case: PreparedBalanced10, fisher_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = h.loop_sigmas(case.bm, fisher_q)
    gains = case.edge_sigma / np.maximum(sigma, 1.0e-300)
    return sigma, gains


def station_schedule_rows(case: PreparedBalanced10, station_schedule: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for station, loop_ids in enumerate(case.incident_loops):
        for loop_idx in loop_ids:
            rows.append(
                {
                    "station": f"S{station + 1}",
                    "loop": h.loop_label(case.triangles[loop_idx]),
                    "fraction": float(station_schedule[station, loop_idx]),
                }
            )
    return rows


def triangle_fraction_summary(case: PreparedBalanced10, station_schedule: np.ndarray) -> dict[str, object]:
    loop_fractions: dict[str, dict[str, float]] = {}
    for loop_idx, tri in enumerate(case.triangles):
        loop_fractions[h.loop_label(tri)] = {
            f"S{station + 1}": float(station_schedule[station, loop_idx])
            for station in tri
        }
    station_rows = {
        f"S{station + 1}": {
            h.loop_label(case.triangles[loop_idx]): float(station_schedule[station, loop_idx])
            for loop_idx in loop_ids
        }
        for station, loop_ids in enumerate(case.incident_loops)
    }
    loop_totals = {
        h.loop_label(case.triangles[loop_idx]): float(np.sum(station_schedule[list(case.triangles[loop_idx]), loop_idx]))
        for loop_idx in range(len(case.triangles))
    }
    return {
        "loop_station_fractions": loop_fractions,
        "station_rows": station_rows,
        "loop_total_incident_fraction": loop_totals,
    }


def load_old_local_opt_gains() -> dict[str, float]:
    rows = list(csv.DictReader(OLD_LOOP_GAINS.open()))
    return {
        row["loop"]: float(row["snr_gain_direct_optimized_schedule_vs_edge"])
        for row in rows
    }


def evaluate_station_schedule(
    case: PreparedBalanced10,
    station_schedule: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    fisher = corrected_raw_fisher_from_station_schedule(case, station_schedule)
    sigma, gains = gains_from_fisher(case, fisher)
    return schedule_score(gains), sigma, gains


def transfer_budget(
    station_schedule: np.ndarray,
    station: int,
    source_loop: int,
    target_loop: int,
    delta: float,
) -> np.ndarray | None:
    if source_loop == target_loop:
        return None
    available = float(station_schedule[station, source_loop])
    if available <= delta + 1.0e-12:
        return None
    candidate = np.asarray(station_schedule, dtype=float).copy()
    candidate[station, source_loop] -= float(delta)
    candidate[station, target_loop] += float(delta)
    return candidate


def optimize_generalized_station_schedule(
    case: PreparedBalanced10,
    *,
    seed: int = 20260616,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    eval_count = 0

    def evaluate(schedule: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        nonlocal eval_count
        eval_count += 1
        return evaluate_station_schedule(case, schedule)

    best = uniform_station_schedule(case)
    best_score, best_sigma, best_gains = evaluate(best)
    best_start = "uniform_0p2_each_incident_loop"

    for alpha in (0.35, 0.70, 1.00, 2.00, 4.00):
        for trial in range(8):
            candidate = random_station_schedule(case, rng, alpha)
            score, sigma, gains = evaluate(candidate)
            if score > best_score + 1.0e-12:
                best = candidate
                best_score = score
                best_sigma = sigma
                best_gains = gains
                best_start = f"dirichlet_alpha_{alpha:g}_trial_{trial}"
    print(
        f"[generalized-rawje] after random starts: score={best_score:.6f}, "
        f"min_gain={float(np.min(best_gains)):.6f}, mean_gain={float(np.mean(best_gains)):.6f}, "
        f"evals={eval_count}"
    )

    for width in (0.16, 0.05, 0.015):
        improved = False
        for station, loop_ids in enumerate(case.incident_loops):
            for source_loop in loop_ids:
                for target_loop in loop_ids:
                    candidate = transfer_budget(best, station, source_loop, target_loop, width)
                    if candidate is None:
                        continue
                    score, sigma, gains = evaluate(candidate)
                    if score > best_score + 1.0e-12:
                        best = candidate
                        best_score = score
                        best_sigma = sigma
                        best_gains = gains
                        improved = True
        print(
            f"[generalized-rawje] after width={width:.3f}: score={best_score:.6f}, "
            f"min_gain={float(np.min(best_gains)):.6f}, mean_gain={float(np.mean(best_gains)):.6f}, "
            f"improved={improved}, evals={eval_count}"
        )

    for concentration in (18.0, 42.0):
        for _ in range(6):
            candidate = np.asarray(best, dtype=float).copy()
            for station, loop_ids in enumerate(case.incident_loops):
                base_row = np.asarray(best[station, list(loop_ids)], dtype=float)
                alpha = np.maximum(concentration * base_row, 0.03) + 0.10
                candidate[station, list(loop_ids)] = rng.dirichlet(alpha)
            score, sigma, gains = evaluate(candidate)
            if score > best_score + 1.0e-12:
                best = candidate
                best_score = score
                best_sigma = sigma
                best_gains = gains
        print(
            f"[generalized-rawje] after concentration={concentration:.1f}: score={best_score:.6f}, "
            f"min_gain={float(np.min(best_gains)):.6f}, mean_gain={float(np.mean(best_gains)):.6f}, "
            f"evals={eval_count}"
        )

    triangle_info = triangle_fraction_summary(case, best)
    info = {
        "objective": "same balanced-10 score as the old direct schedule: mean log gain + 0.35 min log gain - 0.65 var log gain - 30 mean below-one penalty squared",
        "model": "corrected_raw_Je_with_per_loop_per_station_fractions",
        "constraints": "a_{loop,station} >= 0, a_{loop,station} = 0 outside the loop, and sum_{incident loops} a_{loop,station} = 1 for each station",
        "seed": int(seed),
        "n_evaluations": int(eval_count),
        "best_start": str(best_start),
        "score": float(best_score),
        "gain_vs_edge": ratio_summary(best_gains),
        "station_row_sums": {
            f"S{station + 1}": float(np.sum(best[station]))
            for station in range(case.bm.n)
        },
        **triangle_info,
    }
    return best, best_sigma, info


def legacy_raw_gain_rows(case: PreparedBalanced10) -> tuple[np.ndarray, np.ndarray]:
    fisher = h.rawdirect_balanced10_fisher(case.bm, weight=0.2, schur_per_sample=False)
    return gains_from_fisher(case, fisher)


def corrected_symmetric_gain_rows(case: PreparedBalanced10) -> tuple[np.ndarray, np.ndarray]:
    schedule = symmetric_schedule_from_scalar_weight(case, weight=0.2)
    fisher = corrected_raw_fisher_from_station_schedule(case, schedule)
    return gains_from_fisher(case, fisher)


def dump_station_schedule_json(
    path: Path,
    case: PreparedBalanced10,
    station_schedule: np.ndarray,
    info: dict[str, object],
) -> None:
    payload = {
        "loop_set": [h.loop_label(tri) for tri in case.triangles],
        "station_budget_model": "Each station distributes one unit of local photon budget independently over the five balanced-10 loops incident on it.",
        "schedule_matrix": np.asarray(station_schedule, dtype=float).tolist(),
        "schedule_rows": station_schedule_rows(case, station_schedule),
        "summary": info,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
