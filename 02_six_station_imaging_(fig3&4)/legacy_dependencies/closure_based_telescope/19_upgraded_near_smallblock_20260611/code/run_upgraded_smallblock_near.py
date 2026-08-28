from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE = THIS_DIR.parents[1]
ROOT = WORKSPACE / "19_upgraded_near_smallblock_20260611"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
NOTES = ROOT / "notes"
for folder in (RESULTS, FIGURES, LOGS, NOTES, LOGS / "mplconfig"):
    folder.mkdir(parents=True, exist_ok=True)

SOURCE18 = WORKSPACE / "18_balanced_10loop_independent_set_20260611"


def load_balanced_module():
    path = SOURCE18 / "code" / "run_remote_star_joint_near_benchmark.py"
    spec = importlib.util.spec_from_file_location("balanced10_remote_star", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b18 = load_balanced_module()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def allowed_smallblock_triangles(bm) -> list[tuple[int, int, int]]:
    """Small-block near modules: any 3-station block containing at least one compact-core station."""
    out = []
    for tri in b18.all_triangle_list(bm.n):
        if any(station in b18.core_remote.CORE for station in tri):
            out.append(tri)
    return out


def class_initial_weights(triangles: list[tuple[int, int, int]]) -> np.ndarray:
    """Positive exact-budget starting point for three core plus three remote stations."""
    weights = np.zeros(len(triangles), dtype=float)
    for idx, tri in enumerate(triangles):
        n_remote = sum(station in b18.core_remote.REMOTE for station in tri)
        if n_remote == 0:
            weights[idx] = 0.08
        elif n_remote == 1:
            weights[idx] = 0.28 / 3.0
        elif n_remote == 2:
            weights[idx] = 0.12
        else:
            raise ValueError("Pure remote triangles are not allowed in this model.")
    return weights


def incidence_matrix(bm, triangles: list[tuple[int, int, int]]) -> np.ndarray:
    incidence = np.zeros((bm.n, len(triangles)), dtype=float)
    for col, tri in enumerate(triangles):
        incidence[list(tri), col] = 1.0
    return incidence


def fisher_from_weights(bm, triangles: list[tuple[int, int, int]], unit_scalars: np.ndarray, weights: np.ndarray) -> np.ndarray:
    fisher = np.zeros((bm.q_basis.shape[1], bm.q_basis.shape[1]), dtype=float)
    for tri, unit_scalar, weight in zip(triangles, unit_scalars, weights):
        d = bm.q_basis.T @ b18.core_remote.edge_vector(bm.edges, tri)
        fisher += float(weight) * float(unit_scalar) * np.outer(d, d)
    return 0.5 * (fisher + fisher.T)


def match_directopt_score(edge_sigma: np.ndarray, directopt_sigma: np.ndarray, candidate_sigma: np.ndarray) -> float:
    gain = edge_sigma / np.maximum(candidate_sigma, 1e-300)
    if float(np.min(gain)) < 1.0 - 1e-10:
        return -math.inf
    ratio = directopt_sigma / np.maximum(candidate_sigma, 1e-300)
    log_ratio = np.log(np.maximum(ratio, 1e-300))
    overshoot = np.maximum(0.0, log_ratio)
    return float(
        -np.mean(log_ratio * log_ratio)
        - 0.30 * np.var(log_ratio)
        - 0.20 * np.mean(overshoot * overshoot)
        - 0.03 * np.max(np.abs(log_ratio)) ** 2
    )


def optimize_smallblock_schedule(
    bm,
    edge_sigma: np.ndarray,
    directopt_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    triangles = allowed_smallblock_triangles(bm)
    incidence = incidence_matrix(bm, triangles)
    unit_scalars = np.asarray(
        [b18.variants.alltri.triangle_direct_fisher(bm, tri, (1.0, 1.0, 1.0)) for tri in triangles],
        dtype=float,
    )
    _, _, vh = np.linalg.svd(incidence, full_matrices=True)
    rank = int(np.linalg.matrix_rank(incidence))
    null = vh[rank:].T

    def evaluate(weights: np.ndarray) -> tuple[float, np.ndarray]:
        fisher = fisher_from_weights(bm, triangles, unit_scalars, weights)
        sigma = b18.loop_sigmas_from_q_fisher(bm, fisher)
        return match_directopt_score(edge_sigma, directopt_sigma, sigma), sigma

    best = class_initial_weights(triangles)
    station_sums = incidence @ best
    if float(np.max(np.abs(station_sums - 1.0))) > 1e-10:
        raise RuntimeError(f"Initial small-block weights are not budget-feasible: {station_sums}")
    best_score, best_sigma = evaluate(best)

    # Add the previous full direct optimum, projected into the allowed set and repaired by least squares,
    # as a useful warm start.  It is not always feasible after dropping the pure remote triangle.
    summary18 = json.loads((SOURCE18 / "results" / "remote_star_joint_near_summary.json").read_text())["summary"]
    full_weights = summary18["direct_optimized_schedule_info"]["all_triangle_weights"]
    candidate = np.asarray(
        [float(full_weights.get("-".join(f"S{i + 1}" for i in tri), 0.0)) for tri in triangles],
        dtype=float,
    )
    deficit = 1.0 - incidence @ candidate
    repair = np.linalg.pinv(incidence, rcond=1e-12) @ deficit
    candidate = candidate + repair
    if np.min(candidate) >= -1e-12 and float(np.max(np.abs(incidence @ candidate - 1.0))) < 1e-9:
        score, sigma = evaluate(np.maximum(candidate, 0.0))
        if score > best_score:
            best_score = score
            best = np.maximum(candidate, 0.0)
            best_sigma = sigma

    rng = np.random.default_rng(20260611)
    center = best.copy()
    for scale in (0.015, 0.035, 0.070, 0.12, 0.20, 0.32):
        for _ in range(2500):
            candidate = center + null @ rng.normal(scale=scale, size=null.shape[1])
            if np.min(candidate) < -1e-12:
                continue
            score, sigma = evaluate(candidate)
            if score > best_score:
                best_score = score
                best = np.maximum(candidate, 0.0)
                best_sigma = sigma
                center = best.copy()

    for width in (0.10, 0.045, 0.020, 0.009, 0.004):
        improved = True
        passes = 0
        while improved and passes < 8:
            improved = False
            passes += 1
            for idx in range(null.shape[1]):
                direction = null[:, idx]
                for sign in (-1.0, 1.0):
                    candidate = best + sign * width * direction
                    if np.min(candidate) < -1e-12:
                        continue
                    score, sigma = evaluate(candidate)
                    if score > best_score + 1e-12:
                        best_score = score
                        best = np.maximum(candidate, 0.0)
                        best_sigma = sigma
                        improved = True

    gains = edge_sigma / np.maximum(best_sigma, 1e-300)
    ratio = directopt_sigma / np.maximum(best_sigma, 1e-300)
    station_sums = incidence @ best
    order = np.argsort(best)[::-1]
    info = {
        "model": "upgraded_smallblock_near",
        "description": (
            "Receiver schedule restricted to local three-station blocks that contain at least one compact-core station. "
            "This keeps compact-core multi-end access and remote-involved small blocks, but forbids the pure remote "
            "S4-S5-S6 direct block and forbids a full six-mode joint receiver."
        ),
        "objective": "match full direct-optimized selected-loop SNR under exact station budgets and strict gain>=edge",
        "score": float(best_score),
        "n_allowed_blocks": int(len(triangles)),
        "forbidden_blocks": ["S4-S5-S6"],
        "station_weight_sums": {str(bm.names[i]): float(station_sums[i]) for i in range(bm.n)},
        "max_station_weight_error": float(np.max(np.abs(station_sums - 1.0))),
        "snr_gain_vs_edge": b18.ratio_summary(gains),
        "snr_ratio_vs_direct_optimized": b18.ratio_summary(ratio),
        "top_weights": {
            "-".join(f"S{i + 1}" for i in triangles[idx]): float(best[idx])
            for idx in order[:12]
            if best[idx] > 1e-8
        },
        "all_weights": {
            "-".join(f"S{i + 1}" for i in tri): float(weight)
            for tri, weight in zip(triangles, best)
        },
    }
    return best_sigma, best, info


def make_rows(
    labels: list[str],
    baseline_rows: list[dict[str, str]],
    upgraded_sigma: np.ndarray,
    edge_sigma: np.ndarray,
    directopt_sigma: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, base_row in enumerate(baseline_rows):
        row: dict[str, object] = {
            "loop": labels[idx],
            "loop_class": base_row["loop_class"],
        }
        for key in (
            "edge_uniform",
            "direct_balanced_10loop",
            "direct_optimized_schedule",
            "current_near",
            "remote_star_independent",
        ):
            row[f"snr_gain_{key}_vs_edge"] = float(base_row[f"snr_gain_{key}_vs_edge"])
            row[f"snr_ratio_{key}_vs_direct_optimized"] = float(directopt_sigma[idx] / max(float(base_row[f"rms_{key}_rad"]), 1e-300))
        row["rms_upgraded_smallblock_rad"] = float(upgraded_sigma[idx])
        row["snr_gain_upgraded_smallblock_vs_edge"] = float(edge_sigma[idx] / max(float(upgraded_sigma[idx]), 1e-300))
        row["snr_ratio_upgraded_smallblock_vs_direct_optimized"] = float(directopt_sigma[idx] / max(float(upgraded_sigma[idx]), 1e-300))
        rows.append(row)
    return rows


def plot_rows(rows: list[dict[str, object]]) -> None:
    labels = [str(row["loop"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.14
    fig, ax = plt.subplots(figsize=(10.4, 3.8), constrained_layout=True)
    series = (
        (-2.0 * width, "direct_optimized_schedule", "direct optimized", "#9d0208"),
        (-1.0 * width, "direct_balanced_10loop", "direct balanced 10-loop", "#ff5a5f"),
        (0.0 * width, "remote_star_independent", "remote-star near", "#6a4c93"),
        (1.0 * width, "upgraded_smallblock", "upgraded small-block near", "#2a9d8f"),
        (2.0 * width, "edge_uniform", "edge-first", "#0077b6"),
    )
    for offset, key, label, color in series:
        values = [float(row[f"snr_gain_{key}_vs_edge"]) for row in rows]
        ax.bar(x + offset, values, width=width, label=label, color=color)
    ax.axhline(1.0, color="0.15", lw=0.9, ls="--")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.set_title("Upgraded near: compact-core plus core-anchored remote small blocks", fontsize=11)
    ax.grid(True, axis="y", color="0.88", lw=0.7)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(FIGURES / "upgraded_smallblock_loop_gains.png", dpi=240)
    fig.savefig(FIGURES / "upgraded_smallblock_loop_gains.pdf")
    plt.close(fig)


def write_note(summary: dict[str, object]) -> None:
    lines = [
        "# Upgraded Near Small-Block Receiver",
        "",
        "This is a Fisher-level test of the idea that a near-optimal closure strategy can approach the direct optimum without a full six-mode joint receiver.",
        "",
        "Allowed receiver modules are local three-station blocks containing at least one compact-core station.  Thus the model allows `{core, core, remote}` and `{core, remote, remote}` blocks, plus the compact-core block implicitly through the allowed core-containing schedule, but forbids the pure remote `{S4,S5,S6}` direct block.",
        "",
        "The weights obey exact station-side photon budgets: for every station, the sum of all small-block weights containing that station is one.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    (NOTES / "upgraded_smallblock_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    b18.variants.configure_six_station_constants()
    bm = b18.variants.configure_six_benchmark()
    triangles = b18.balanced_independent_triangles(bm.n)
    labels = b18.loop_labels(triangles)
    edge_sigma = b18.edge_sigmas_for_triangles(bm, triangles)
    baseline_rows = list(csv.DictReader((SOURCE18 / "results" / "remote_star_joint_loop_gains.csv").open()))
    directopt_sigma = np.asarray([float(row["rms_direct_optimized_schedule_rad"]) for row in baseline_rows], dtype=float)

    b18.variants.alltri.bm_lib.EPS_STATION = b18.variants.fig_run.EPS_STATION_RUN
    b18.variants.alltri.bm_lib.EPS_DIRECT_EXTRA = b18.variants.fig_run.EPS_DIRECT_EXTRA_RUN
    old_source = b18.variants.alltri.ngc.NGC4151
    b18.variants.alltri.ngc.NGC4151 = b18.variants.fig_run.GOOD_SOURCE
    try:
        with b18.variants.fig_run.morph.patched_variant(b18.variants.fig_run.GOOD_VARIANT), b18.variants.fig_run.ngc.patched_source(
            b18.variants.fig_run.GOOD_SOURCE
        ):
            upgraded_sigma, upgraded_weights, upgraded_info = optimize_smallblock_schedule(bm, edge_sigma, directopt_sigma)
    finally:
        b18.variants.alltri.ngc.NGC4151 = old_source

    rows = make_rows(labels, baseline_rows, upgraded_sigma, edge_sigma, directopt_sigma)
    write_csv(RESULTS / "upgraded_smallblock_loop_gains.csv", rows)
    plot_rows(rows)

    summary = {
        "source_18_summary": str(SOURCE18 / "results" / "remote_star_joint_near_summary.json"),
        "loop_set": labels,
        "upgraded_smallblock_info": upgraded_info,
        "comparisons": {
            key: b18.ratio_summary(np.asarray([float(row[f"snr_gain_{key}_vs_edge"]) for row in rows], dtype=float))
            for key in (
                "direct_optimized_schedule",
                "direct_balanced_10loop",
                "remote_star_independent",
                "upgraded_smallblock",
            )
        },
        "ratio_to_direct_optimized": {
            key: b18.ratio_summary(np.asarray([float(row[f"snr_ratio_{key}_vs_direct_optimized"]) for row in rows], dtype=float))
            for key in (
                "direct_balanced_10loop",
                "remote_star_independent",
                "upgraded_smallblock",
            )
        },
        "csv": str(RESULTS / "upgraded_smallblock_loop_gains.csv"),
        "figure": str(FIGURES / "upgraded_smallblock_loop_gains.png"),
    }
    (RESULTS / "upgraded_smallblock_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_note(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
