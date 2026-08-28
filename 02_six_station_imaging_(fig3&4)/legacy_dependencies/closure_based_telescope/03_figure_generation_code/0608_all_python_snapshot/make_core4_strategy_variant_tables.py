from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
BUNDLE = THIS_DIR.parents[1]
CORE_DIR = THIS_DIR.parent / "core"
for _path in (CORE_DIR, THIS_DIR):
    _path_text = str(_path)
    if _path_text in sys.path:
        sys.path.remove(_path_text)
    sys.path.insert(0, _path_text)

import hawaii3_compact_case  # noqa: E402
import plot_prl_broadband_clean as base  # noqa: E402
import run_broad_plume_split_objective_rml as fig_run  # noqa: E402


OUT = BUNDLE / "exploration" / "core4_joint_remote_split" / "strategy_variants"
OUT.mkdir(parents=True, exist_ok=True)

BENCHMARK_FACTOR = 2.0 / 5.0


def configure_active_benchmark_physics() -> None:
    exposure_s = fig_run.BASE_EXPOSURE_S * fig_run.EXPOSURE_SCALE
    fig_run.aug.OBSERVING_DAYS = fig_run.OBSERVING_DAYS
    fig_run.aug.N_TIME_WINDOWS = fig_run.N_TIME_WINDOWS_RUN
    fig_run.aug.EXPOSURE_S = exposure_s
    fig_run.aug.EXPOSURE_GAP_S = fig_run.EXPOSURE_GAP_S_RUN
    fig_run.aug.LAMBDA_MIN_NM = fig_run.LAMBDA_MIN_NM_RUN
    fig_run.aug.LAMBDA_MAX_NM = fig_run.LAMBDA_MAX_NM_RUN
    fig_run.aug.LAMBDA_STEP_NM = fig_run.LAMBDA_STEP_NM_RUN
    fig_run.aug.FIBER_LOSS_DB_PER_KM = fig_run.closure_bm.FIBER_LOSS_DB_PER_KM
    fig_run.aug.FIBER_LENGTH_SCALE = fig_run.closure_bm.FIBER_LENGTH_SCALE
    fig_run.aug.MODE_FALSE_POSITIVE = fig_run.EPS_STATION_RUN
    fig_run.aug.PAIR_FALSE_POSITIVE = fig_run.EPS_PAIR_RUN
    fig_run.aug.BASELINE_FALSE_POSITIVE = fig_run.EPS_PAIR_RUN
    fig_run.wt.OBSERVING_DAYS = fig_run.OBSERVING_DAYS
    fig_run.wt.SNR_BOOST = 1.0


def independent_root_triangles(n: int) -> list[tuple[int, int, int]]:
    return [(0, i, j) for i in range(1, n) for j in range(i + 1, n)]


def closure_kind(tri: tuple[int, int, int]) -> str:
    n_remote = sum(i in fig_run.core4_remote.REMOTE for i in tri)
    if n_remote == 0:
        return "core only"
    if n_remote == 1:
        return "one remote"
    if n_remote == 2:
        return "two remote"
    return "three remote"


def sym_pinv(matrix: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    return np.linalg.pinv(0.5 * (matrix + matrix.T), rcond=rcond)


def rank_aware_snr_for_vector(
    fisher: np.ndarray,
    d: np.ndarray,
    rel_floor: float = 1e-10,
) -> tuple[float, float, bool]:
    fisher = 0.5 * (fisher + fisher.T)
    eig, vec = np.linalg.eigh(fisher)
    max_eig = float(np.max(eig)) if eig.size else 0.0
    tol = max(rel_floor * max(max_eig, 1.0), 1e-300)
    good = eig > tol
    if not np.any(good):
        return 0.0, math.inf, False
    null_component = vec[:, ~good].T @ d
    if float(np.linalg.norm(null_component)) > 1e-8 * max(float(np.linalg.norm(d)), 1.0):
        return 0.0, math.inf, False
    cov = (vec[:, good] / eig[good]) @ vec[:, good].T
    var = float(d @ cov @ d)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0, math.inf, False
    return 1.0 / math.sqrt(var), var, True


def snr_for_vector(fisher: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    snr, var, _estimable = rank_aware_snr_for_vector(fisher, d)
    return snr, var


def rank_summary(fisher: np.ndarray, rel_floor: float = 1e-10) -> dict[str, object]:
    fisher = 0.5 * (fisher + fisher.T)
    eig = np.linalg.eigvalsh(fisher)
    max_eig = float(np.max(eig)) if eig.size else 0.0
    tol = max(rel_floor * max(max_eig, 1.0), 1e-300)
    numerical_rank = int(np.sum(eig > tol))
    full_rank = numerical_rank == fisher.shape[0]
    out: dict[str, object] = {
        "raw_min_eigen": float(np.min(eig)) if eig.size else 0.0,
        "raw_max_eigen": max_eig,
        "rank_tolerance": float(tol),
        "numerical_rank": numerical_rank,
        "null_dim": int(fisher.shape[0] - numerical_rank),
        "full_rank": bool(full_rank),
    }
    out["strict_mean_coord_rms"] = float(strict_mean_coord_rms(fisher, rel_floor=rel_floor)) if full_rank else None
    return out


def strict_mean_coord_rms(fisher: np.ndarray, rel_floor: float = 1e-12) -> float:
    """Mean coordinate RMS with a full-rank requirement.

    The project-wide stable_metrics helper uses a pseudo-inverse, which is
    useful for noisy diagnostics but unsafe as an optimization target here:
    a rank-deficient direct schedule can otherwise look spuriously excellent.
    """
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


def closure_scalar_from_direct_raw(bm, tri: tuple[int, int, int]) -> float:
    d = fig_run.core4_remote.measurement_vector(bm, tri)
    snr, _var = snr_for_vector(bm.direct_raw, d)
    return snr * snr


def direct_fisher_from_loop_weights(bm, weights: dict[tuple[int, int, int], float]) -> np.ndarray:
    fisher = np.zeros((bm.q_basis.shape[1], bm.q_basis.shape[1]), dtype=float)
    for tri, weight in weights.items():
        d = fig_run.core4_remote.measurement_vector(bm, tri)
        fisher += float(weight) * closure_scalar_from_direct_raw(bm, tri) * np.outer(d, d)
    return 0.5 * (fisher + fisher.T)


def project_capped_simplex(values: np.ndarray, total: float, cap: float = 1.0) -> np.ndarray:
    """Project approximately onto {0 <= x <= cap, sum x = total}."""
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
    # Small correction after clipping.
    for _ in range(20):
        deficit = total - float(np.sum(out))
        if abs(deficit) < 1e-12:
            break
        movable = (out < cap - 1e-12) if deficit > 0 else (out > 1e-12)
        if not np.any(movable):
            break
        out[movable] += deficit / float(np.count_nonzero(movable))
        out = np.clip(out, 0.0, cap)
    return out


def optimize_direct_loop_weights_mean_rms(bm, seed: int = 20260605) -> tuple[dict[tuple[int, int, int], float], dict]:
    """Optimize direct closure root-loop scheduling with the same mean-RMS metric.

    This is a coordinate-scheduled direct proxy: each root closure can receive
    a time/resource weight between 0 and 1, and the total weight equals
    (N-1)=6, matching the 2/5 uniform scheduled benchmark over 15 closures.
    """
    triangles = independent_root_triangles(bm.n)
    total_weight = bm.rank_share * len(triangles)
    rng = np.random.default_rng(seed)

    def fisher_for_weights(weight_vec: np.ndarray) -> np.ndarray:
        return direct_fisher_from_loop_weights(bm, dict(zip(triangles, weight_vec)))

    def score(weight_vec: np.ndarray) -> float:
        return -strict_mean_coord_rms(fisher_for_weights(weight_vec))

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
    final_fisher = fisher_for_weights(best_w)
    final_strict_mean = strict_mean_coord_rms(final_fisher)
    return dict(zip(triangles, best_w)), {
        "objective": "mean_coord_rms",
        "score": float(best_score),
        "strict_mean_coord_rms": float(final_strict_mean),
        "min_weight": float(np.min(best_w)),
        "max_weight": float(np.max(best_w)),
        "total_weight": float(total_weight),
        "max_per_closure_weight": 1.0,
        "rank_guard": "reject candidate schedules whose direct Fisher is numerically rank deficient",
    }


def loop_sum_globalized_split(bm, specs: list[dict[str, object]]) -> np.ndarray:
    """Sum independent closure edge-demand patterns and normalize per station.

    Close-only loops are supplied by the close4 direct block and do not enter
    this edge-demand sum.  For remote-involved loops, only remote-related edges
    are included; close-close baselines remain in the close4 closure block.
    """
    demand = np.zeros((bm.n, bm.n), dtype=float)
    for spec in specs:
        tri = tuple(spec["tri"])
        if all(station in fig_run.core4_remote.CORE for station in tri):
            continue
        directed = fig_run.core4_remote.noncore_directed_fractions(
            tri,
            tuple(float(value) for value in spec["split"]),
        )
        for (station, target), fraction in directed.items():
            demand[station, target] += float(fraction)

    split = np.zeros_like(demand)
    for station in range(bm.n):
        budget = fig_run.core4_remote.station_loop_budget(station)
        row_sum = float(np.sum(demand[station]))
        if row_sum <= 0.0:
            continue
        split[station] = budget * demand[station] / row_sum
    return split


def split_rows(bm, split: np.ndarray, variant: str) -> list[dict]:
    rows = []
    for station in range(bm.n):
        for target in range(bm.n):
            value = float(split[station, target])
            if station == target or value <= 0.0:
                continue
            rows.append(
                {
                    "variant": variant,
                    "from": f"S{station + 1}",
                    "station": bm.names[station],
                    "to": f"S{target + 1}",
                    "target": bm.names[target],
                    "fraction": value,
                    "row_budget": fig_run.core4_remote.station_loop_budget(station),
                }
            )
    return rows


def loop_internal_rows(bm, specs: list[dict[str, object]], variant: str) -> list[dict]:
    rows = []
    for spec in specs:
        tri = tuple(spec["tri"])
        if all(station in fig_run.core4_remote.CORE for station in tri):
            continue
        directed = fig_run.core4_remote.noncore_directed_fractions(tri, tuple(float(x) for x in spec["split"]))
        for (station, target), value in sorted(directed.items()):
            rows.append(
                {
                    "variant": variant,
                    "closure": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
                    "from": f"S{station + 1}",
                    "station": bm.names[station],
                    "to": f"S{target + 1}",
                    "target": bm.names[target],
                    "loop_internal_fraction": float(value),
                }
            )
    return rows


def direct_weight_rows(bm, weights: dict[tuple[int, int, int], float], variant: str) -> list[dict]:
    rows = []
    for tri, value in weights.items():
        rows.append(
            {
                "variant": variant,
                "closure": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
                "type": closure_kind(tri),
                "weight": float(value),
            }
        )
    return rows


def closure_rows(bm, variants: dict[str, dict]) -> list[dict]:
    rows = []
    for tri in independent_root_triangles(bm.n):
        d = fig_run.core4_remote.measurement_vector(bm, tri)
        for key, item in variants.items():
            near_snr, near_var, near_estimable = rank_aware_snr_for_vector(item["near_fisher"], d)
            direct_snr, direct_var, direct_estimable = rank_aware_snr_for_vector(item["direct_fisher"], d)
            rows.append(
                {
                    "variant": key,
                    "closure": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
                    "type": closure_kind(tri),
                    "near_estimable": near_estimable,
                    "direct_reference_estimable": direct_estimable,
                    "snr_near": near_snr,
                    "snr_direct_reference": direct_snr,
                    "snr_ratio_near_over_direct": near_snr / direct_snr if direct_snr > 0.0 else math.inf,
                    "rms_near_rad": math.sqrt(near_var) if np.isfinite(near_var) else math.inf,
                    "rms_direct_reference_rad": math.sqrt(direct_var) if np.isfinite(direct_var) else math.inf,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_payload() -> dict:
    fig_run.configure_good_runtime()
    fig_run.apply_sample_stress_runtime()
    case = fig_run.scale_remote_coordinates(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    old_configure = fig_run.closure_bm.configure_physics
    old_loader = fig_run.closure_bm.rml_cases.load_maunakea_plus3_case
    fig_run.closure_bm.configure_physics = configure_active_benchmark_physics
    fig_run.closure_bm.rml_cases.load_maunakea_plus3_case = lambda: case
    try:
        with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT):
            bm = fig_run.closure_bm.AllClosureBenchmark()
            global_split, global_info = fig_run.core4_remote.optimize_split(bm, "mean_rms")
            loop_specs = fig_run.core4_remote.optimize_root_loop_splits(bm)
            loop_sum_split = loop_sum_globalized_split(bm, loop_specs)
            direct_weights, direct_info = optimize_direct_loop_weights_mean_rms(bm)
    finally:
        fig_run.closure_bm.configure_physics = old_configure
        fig_run.closure_bm.rml_cases.load_maunakea_plus3_case = old_loader

    variants = {
        "v1_meanrms_near_and_direct_optimized": {
            "description": (
                "Near: close4 closure plus globally optimized remote-edge split under mean_coord_rms. "
                "Direct: root-closure coordinate schedule optimized under the same mean_coord_rms proxy."
            ),
            "near_fisher": fig_run.core4_remote.fisher_for_split(bm, global_split),
            "direct_fisher": direct_fisher_from_loop_weights(bm, direct_weights),
            "near_split_rows": split_rows(bm, global_split, "v1_meanrms_near_and_direct_optimized"),
            "direct_weight_rows": direct_weight_rows(bm, direct_weights, "v1_meanrms_near_and_direct_optimized"),
            "optimization_info": {
                "near": global_info,
                "direct": direct_info,
            },
        },
        "v2_loop_sum_edge_scheme": {
            "description": (
                "First optimize each independent closure's internal edge split. "
                "Then sum remote-related directed edge demands over the 15 root closures, normalize per station, "
                "and recompute near performance with close4 closure plus that shared edge split. "
                "Direct reference is the scheduled 2/5 full direct benchmark."
            ),
            "near_fisher": fig_run.core4_remote.fisher_for_split(bm, loop_sum_split),
            "direct_fisher": BENCHMARK_FACTOR * bm.direct_raw,
            "near_split_rows": split_rows(bm, loop_sum_split, "v2_loop_sum_edge_scheme"),
            "loop_internal_rows": loop_internal_rows(bm, loop_specs, "v2_loop_sum_edge_scheme"),
            "optimization_info": {
                "loop_internal_objective": "per-closure scalar SNR / harmonic edge Fisher",
                "globalization": "sum directed remote-edge demands from root closure schemes, then normalize each station row to its remote-edge budget",
                "direct_reference": "scheduled 2/5 full direct Fisher",
            },
        },
    }
    closure_table = closure_rows(bm, variants)
    for item in variants.values():
        item["near_metrics"] = fig_run.closure_bm.stable_metrics(item["near_fisher"])
        item["direct_metrics"] = fig_run.closure_bm.stable_metrics(item["direct_fisher"])
        item["near_rank_summary"] = rank_summary(item["near_fisher"])
        item["direct_rank_summary"] = rank_summary(item["direct_fisher"])
        item.pop("near_fisher")
        item.pop("direct_fisher")

    return {
        "run_tag": fig_run.RUN_TAG,
        "case": bm.case.key,
        "station_names": bm.names,
        "resource_model": {
            "observing_days": int(fig_run.OBSERVING_DAYS),
            "samples_per_night": int(fig_run.N_TIME_WINDOWS_RUN),
            "exposure_s": float(fig_run.BASE_EXPOSURE_S * fig_run.EXPOSURE_SCALE),
            "sample_cadence_s": float(fig_run.SAMPLE_CADENCE_S_RUN),
            "lambda_min_nm": float(fig_run.LAMBDA_MIN_NM_RUN),
            "lambda_max_nm": float(fig_run.LAMBDA_MAX_NM_RUN),
            "lambda_step_nm": float(fig_run.LAMBDA_STEP_NM_RUN),
            "benchmark_factor": BENCHMARK_FACTOR,
            "rank_share": float(bm.rank_share),
            "close4_fraction": float(fig_run.core4_remote.CORE_JOINT_FRACTION),
            "close_remote_fraction": float(fig_run.core4_remote.CORE_REMOTE_FRACTION),
            "remote_total_fraction": float(fig_run.core4_remote.REMOTE_TOTAL_FRACTION),
        },
        "variants": variants,
        "closure_rows": closure_table,
    }


def main() -> None:
    payload = make_payload()
    stem = "core4_strategy_variants"
    json_path = OUT / f"{stem}.json"
    closure_csv = OUT / f"{stem}_closure_snr.csv"
    split_csv = OUT / f"{stem}_near_split_rows.csv"
    loop_csv = OUT / f"{stem}_v2_loop_internal_rows.csv"
    direct_csv = OUT / f"{stem}_v1_direct_weights.csv"

    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(closure_csv, payload["closure_rows"])
    split_rows_all = []
    loop_rows_all = []
    direct_rows_all = []
    for item in payload["variants"].values():
        split_rows_all.extend(item.get("near_split_rows", []))
        loop_rows_all.extend(item.get("loop_internal_rows", []))
        direct_rows_all.extend(item.get("direct_weight_rows", []))
    write_csv(split_csv, split_rows_all)
    write_csv(loop_csv, loop_rows_all)
    write_csv(direct_csv, direct_rows_all)
    print(json_path)
    print(closure_csv)
    print(split_csv)
    print(loop_csv)
    print(direct_csv)


if __name__ == "__main__":
    main()
