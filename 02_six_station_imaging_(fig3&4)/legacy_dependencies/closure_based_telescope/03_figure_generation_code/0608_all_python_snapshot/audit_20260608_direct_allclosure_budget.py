from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUNDLE = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = BUNDLE / "code" / "all_python_snapshot"
CORE_DIR = BUNDLE / "code" / "core"
for path in (CORE_DIR, SNAPSHOT_DIR):
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

import core4_joint_remote_split_design as core4  # noqa: E402
import make_all_closure_global_benchmark_note as bm_lib  # noqa: E402


OUT = BUNDLE / "exploration" / "direct_allclosure_benchmark_20260608"
OUT.mkdir(parents=True, exist_ok=True)


def fisher_from_scalar_weights(
    bm: bm_lib.AllClosureBenchmark,
    triangles: list[tuple[int, int, int]],
    scalar_info: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fisher = np.zeros((bm.n_closure, bm.n_closure), dtype=float)
    for tri, scalar, weight in zip(triangles, scalar_info, weights):
        d = core4.measurement_vector(bm, tri)
        fisher += float(weight) * float(scalar) * np.outer(d, d)
    return 0.5 * (fisher + fisher.T)


def root_loop_rms(
    bm: bm_lib.AllClosureBenchmark,
    fisher: np.ndarray,
    triangles: list[tuple[int, int, int]],
    *,
    rel_floor: float = 1e-12,
) -> np.ndarray:
    fisher = 0.5 * (fisher + fisher.T)
    eig = np.linalg.eigvalsh(fisher)
    max_eig = float(np.max(eig)) if eig.size else 0.0
    min_allowed = max(rel_floor * max(max_eig, 1.0), 1e-300)
    if max_eig <= 0.0 or float(np.min(eig)) <= min_allowed:
        return np.full(len(triangles), math.inf, dtype=float)
    cov = np.linalg.inv(fisher)
    values = []
    for tri in triangles:
        d = core4.measurement_vector(bm, tri)
        var = float(d @ cov @ d)
        values.append(math.sqrt(max(var, 0.0)) if np.isfinite(var) else math.inf)
    return np.asarray(values, dtype=float)


def score_weights(
    bm: bm_lib.AllClosureBenchmark,
    triangles: list[tuple[int, int, int]],
    scalar_info: np.ndarray,
    weights: np.ndarray,
    *,
    spread_penalty: float = 2.5,
    min_bonus: float = 0.7,
) -> float:
    fisher = fisher_from_scalar_weights(bm, triangles, scalar_info, weights)
    rms = root_loop_rms(bm, fisher, triangles)
    if not np.all(np.isfinite(rms)) or np.any(rms <= 0.0):
        return -math.inf
    log_snr = -np.log(np.maximum(rms, 1e-300))
    return float(np.mean(log_snr) - spread_penalty * np.std(log_snr) + min_bonus * np.min(log_snr))


def optimize_weights(
    bm: bm_lib.AllClosureBenchmark,
    triangles: list[tuple[int, int, int]],
    scalar_info: np.ndarray,
    *,
    total_weight: float,
    cap: float = 1.0,
    seed: int = 20260608,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(triangles)
    starts = [
        core4.project_capped_simplex(np.full(n, total_weight / n), total_weight, cap),
        core4.project_capped_simplex(np.sqrt(np.maximum(scalar_info, 1e-300)), total_weight, cap),
        core4.project_capped_simplex(1.0 / np.sqrt(np.maximum(scalar_info, 1e-300)), total_weight, cap),
    ]
    for scale in (0.2, 0.6, 1.2, 2.0):
        for _ in range(400):
            starts.append(core4.project_capped_simplex(rng.random(n) ** scale, total_weight, cap))

    best = starts[0]
    best_score = score_weights(bm, triangles, scalar_info, best)
    for candidate in starts[1:]:
        value = score_weights(bm, triangles, scalar_info, candidate)
        if value > best_score:
            best = candidate
            best_score = value

    for width in (0.20, 0.10, 0.05, 0.025, 0.012, 0.006, 0.003):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    movable = min(width, best[j], cap - best[i])
                    if movable <= 1e-14:
                        continue
                    candidate = best.copy()
                    candidate[i] += movable
                    candidate[j] -= movable
                    value = score_weights(bm, triangles, scalar_info, candidate)
                    if value > best_score + 1e-13:
                        best = candidate
                        best_score = value
                        improved = True

    return best, {
        "score": float(best_score),
        "total_weight": float(total_weight),
        "cap": float(cap),
        "min_weight": float(np.min(best)),
        "median_weight": float(np.median(best)),
        "max_weight": float(np.max(best)),
        "std_weight": float(np.std(best)),
    }


def summarize_strategy(
    bm: bm_lib.AllClosureBenchmark,
    triangles: list[tuple[int, int, int]],
    single_loop_info: np.ndarray,
    name: str,
    fisher: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    description: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rms = root_loop_rms(bm, fisher, triangles)
    eff_info = 1.0 / np.maximum(rms, 1e-300) ** 2
    snr_ratio = np.sqrt(eff_info / np.maximum(single_loop_info, 1e-300))
    metrics = bm_lib.stable_metrics(fisher)
    row: dict[str, object] = {
        "strategy": name,
        "description": description,
        "mean_coord_rms": metrics["mean_coord_rms"],
        "median_coord_rms": metrics["median_coord_rms"],
        "max_coord_rms": metrics["max_coord_rms"],
        "root_rms_min": float(np.min(rms)),
        "root_rms_median": float(np.median(rms)),
        "root_rms_mean": float(np.mean(rms)),
        "root_rms_max": float(np.max(rms)),
        "snr_vs_single_min": float(np.min(snr_ratio)),
        "snr_vs_single_median": float(np.median(snr_ratio)),
        "snr_vs_single_mean": float(np.mean(snr_ratio)),
        "snr_vs_single_max": float(np.max(snr_ratio)),
        "snr_vs_single_std": float(np.std(snr_ratio)),
    }
    if weights is not None:
        row.update(
            {
                "total_weight": float(np.sum(weights)),
                "min_weight": float(np.min(weights)),
                "median_weight": float(np.median(weights)),
                "max_weight": float(np.max(weights)),
                "std_weight": float(np.std(weights)),
            }
        )
    loop_rows: list[dict[str, object]] = []
    for tri, rms_value, ratio, eff, single, weight in zip(
        triangles,
        rms,
        snr_ratio,
        eff_info,
        single_loop_info,
        weights if weights is not None else np.full(len(triangles), math.nan),
    ):
        loop_rows.append(
            {
                "strategy": name,
                "closure": f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}",
                "type": "core_only" if all(station in core4.CORE for station in tri) else "remote_involved",
                "weight": float(weight),
                "rms_rad": float(rms_value),
                "effective_scalar_info": float(eff),
                "single_loop_full_budget_info": float(single),
                "snr_vs_single_loop_full_budget": float(ratio),
            }
        )
    return row, loop_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    bm = bm_lib.AllClosureBenchmark()
    triangles = core4.root_independent_triangles(bm.n)
    vectors = [core4.measurement_vector(bm, tri) for tri in triangles]
    single_loop_info = np.asarray(
        [core4.scalar_closure_info_from_cycle_fisher(bm.direct_raw, d) for d in vectors],
        dtype=float,
    )
    n_closure = len(triangles)
    capacity = bm.n - 1.0

    weights_strict_uniform = np.full(n_closure, 1.0 / n_closure)
    weights_capacity_uniform = np.full(n_closure, capacity / n_closure)
    weights_strict_opt, strict_info = optimize_weights(
        bm,
        triangles,
        single_loop_info,
        total_weight=1.0,
        cap=1.0,
        seed=2026060801,
    )
    weights_capacity_opt, capacity_info = optimize_weights(
        bm,
        triangles,
        single_loop_info,
        total_weight=capacity,
        cap=1.0,
        seed=2026060802,
    )

    strategies = [
        (
            "strict_scalar_uniform_total1",
            fisher_from_scalar_weights(bm, triangles, single_loop_info, weights_strict_uniform),
            weights_strict_uniform,
            "Sequential scalar direct closure polling with equal root-loop weights and total scalar budget 1.",
        ),
        (
            "strict_scalar_optimized_total1",
            fisher_from_scalar_weights(bm, triangles, single_loop_info, weights_strict_opt),
            weights_strict_opt,
            "Sequential scalar direct closure polling with optimized root-loop weights and total scalar budget 1.",
        ),
        (
            "capacity_relaxed_uniform_totalNminus1_default",
            fisher_from_scalar_weights(bm, triangles, single_loop_info, weights_capacity_uniform),
            weights_capacity_uniform,
            "Paper-default capacity-relaxed scalar direct schedule: equal root-loop weights summing to N-1.",
        ),
        (
            "capacity_optimized_totalNminus1",
            fisher_from_scalar_weights(bm, triangles, single_loop_info, weights_capacity_opt),
            weights_capacity_opt,
            "Capacity-relaxed scalar direct design with weights optimized under total scalar budget N-1.",
        ),
        (
            "raw_full_qfi_upper_bound_no_split",
            bm.direct_raw,
            None,
            "Raw multiparameter closure-space QFI upper bound; not an implementable all-closure schedule by itself.",
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    loop_rows: list[dict[str, object]] = []
    for name, fisher, weights, description in strategies:
        summary, loops = summarize_strategy(
            bm,
            triangles,
            single_loop_info,
            name,
            fisher,
            weights=weights,
            description=description,
        )
        summary_rows.append(summary)
        loop_rows.extend(loops)

    payload = {
        "noise_model": {
            "eps_station": bm_lib.EPS_STATION,
            "eps_pair": bm_lib.EPS_PAIR,
            "eps_direct_extra": bm_lib.EPS_DIRECT_EXTRA,
        },
        "n_station": bm.n,
        "n_root_closure": n_closure,
        "capacity_scalar_budget_N_minus_1": capacity,
        "strict_optimized_info": strict_info,
        "capacity_optimized_info": capacity_info,
        "paper_default_strategy": "capacity_relaxed_uniform_totalNminus1_default",
        "paper_default_weight_rule": "w_l=(N-1)/C for every independent root closure l",
        "summary": summary_rows,
        "warning": (
            "raw_full_qfi_upper_bound_no_split does not include photon-budget splitting or "
            "multi-closure measurement compatibility. The paper-default benchmark is the "
            "capacity-relaxed scalar schedule with explicit root-closure weights w_l."
        ),
    }

    summary_json = OUT / "direct_allclosure_budget_summary.json"
    loop_csv = OUT / "direct_allclosure_budget_loop_rows.csv"
    summary_csv = OUT / "direct_allclosure_budget_summary.csv"
    summary_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(summary_csv, summary_rows)
    write_csv(loop_csv, loop_rows)

    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    labels = [str(row["strategy"]).replace("_", "\n") for row in summary_rows]
    med = [float(row["snr_vs_single_median"]) for row in summary_rows]
    lo = [float(row["snr_vs_single_min"]) for row in summary_rows]
    hi = [float(row["snr_vs_single_max"]) for row in summary_rows]
    x = np.arange(len(summary_rows), dtype=float)
    ax.bar(x, med, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#7f7f7f"], alpha=0.85)
    ax.vlines(x, lo, hi, color="black", lw=1.0)
    ax.axhline(1.0, color="black", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.7)
    ax.set_ylabel("SNR relative to single-loop full-budget direct")
    ax.set_ylim(0.0, max(1.08, max(hi) * 1.08))
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "direct_allclosure_budget_snr_ratio.png", dpi=240)
    fig.savefig(OUT / "direct_allclosure_budget_snr_ratio.pdf", bbox_inches="tight")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
