from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rawje_balanced10_helpers as h


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

OLD_SUMMARY = WORKSPACE / "18_balanced_10loop_independent_set_20260611" / "results" / "remote_star_joint_near_summary.json"
OLD_LOOP_GAINS = WORKSPACE / "18_balanced_10loop_independent_set_20260611" / "results" / "remote_star_joint_loop_gains.csv"
STEM = "rawje_opt_samecost_vs_oldopt"


def parse_loop(label: str) -> tuple[int, int, int]:
    return tuple(int(part[1:]) - 1 for part in label.split("-"))


def load_old_weights() -> dict[tuple[int, int, int], float]:
    data = json.loads(OLD_SUMMARY.read_text())
    raw = data["summary"]["direct_optimized_schedule_info"]["all_triangle_weights"]
    return {parse_loop(label): float(value) for label, value in raw.items()}


def load_old_loop_gains() -> dict[str, float]:
    rows = list(csv.DictReader(OLD_LOOP_GAINS.open()))
    return {
        row["loop"]: float(row["snr_gain_direct_optimized_schedule_vs_edge"])
        for row in rows
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def direct_schedule_score(gains: np.ndarray) -> float:
    log_g = np.log(np.maximum(np.asarray(gains, dtype=float), 1.0e-300))
    below_one = np.maximum(0.0, -log_g)
    return float(
        np.mean(log_g)
        + 0.35 * np.min(log_g)
        - 0.65 * np.var(log_g)
        - 30.0 * np.mean(below_one * below_one)
    )


def incidence_matrix(n_station: int, triangles: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    incidence = np.zeros((n_station, len(triangles)), dtype=float)
    for col, tri in enumerate(triangles):
        incidence[list(tri), col] = 1.0
    return incidence


def vec_to_weights(
    triangles: tuple[tuple[int, int, int], ...],
    values: np.ndarray,
) -> dict[tuple[int, int, int], float]:
    return {tri: float(value) for tri, value in zip(triangles, values)}


def weights_to_vec(
    triangles: tuple[tuple[int, int, int], ...],
    weights: dict[tuple[int, int, int], float],
) -> np.ndarray:
    return np.asarray([float(weights.get(tri, 0.0)) for tri in triangles], dtype=float)


def gain_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "score": direct_schedule_score(values),
    }


def optimize_raw_weights(
    bm: h.RawJeBenchmark,
    triangle_edges: dict[tuple[int, int, int], np.ndarray],
    edge_sigma: np.ndarray,
    old_vec: np.ndarray,
    *,
    seed: int = 20260616,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    triangles = h.all_triangles(bm.n)
    incidence = incidence_matrix(bm.n, triangles)
    uniform = np.full(len(triangles), 1.0 / math.comb(bm.n - 1, 2), dtype=float)
    _, _, vh = np.linalg.svd(incidence, full_matrices=True)
    rank = int(np.linalg.matrix_rank(incidence))
    null = vh[rank:].T
    rng = np.random.default_rng(seed)
    eval_count = 0

    def evaluate(values: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal eval_count
        eval_count += 1
        fisher = h.rawdirect_weighted_fisher_from_edges(
            bm,
            triangle_edges,
            vec_to_weights(triangles, values),
        )
        sigma = h.loop_sigmas(bm, fisher)
        gains = edge_sigma / np.maximum(sigma, 1.0e-300)
        return direct_schedule_score(gains), gains

    best = uniform.copy()
    best_score, best_gains = evaluate(best)
    for name, start in (("old_local_opt", old_vec), ("uniform_alltri", uniform)):
        if np.min(start) < -1.0e-12:
            continue
        score, gains = evaluate(np.maximum(start, 0.0))
        if score > best_score:
            best = np.maximum(start, 0.0)
            best_score = score
            best_gains = gains

    for start in (old_vec, uniform, best.copy()):
        for scale in (0.015, 0.035, 0.070, 0.12, 0.20, 0.32):
            for _ in range(1800):
                candidate = start + null @ rng.normal(scale=scale, size=null.shape[1])
                if np.min(candidate) < -1.0e-12:
                    continue
                candidate = np.maximum(candidate, 0.0)
                score, gains = evaluate(candidate)
                if score > best_score + 1.0e-13:
                    best = candidate
                    best_score = score
                    best_gains = gains

    for width in (0.090, 0.045, 0.022, 0.010, 0.0045, 0.0020):
        improved = True
        passes = 0
        while improved and passes < 7:
            improved = False
            passes += 1
            for idx in range(null.shape[1]):
                direction = null[:, idx]
                for sign in (-1.0, 1.0):
                    candidate = best + sign * width * direction
                    if np.min(candidate) < -1.0e-12:
                        continue
                    candidate = np.maximum(candidate, 0.0)
                    score, gains = evaluate(candidate)
                    if score > best_score + 1.0e-13:
                        best = candidate
                        best_score = score
                        best_gains = gains
                        improved = True

    for scale in (0.018, 0.008, 0.003):
        for _ in range(1400):
            candidate = best + null @ rng.normal(scale=scale, size=null.shape[1])
            if np.min(candidate) < -1.0e-12:
                continue
            candidate = np.maximum(candidate, 0.0)
            score, gains = evaluate(candidate)
            if score > best_score + 1.0e-13:
                best = candidate
                best_score = score
                best_gains = gains

    station_sums = incidence @ best
    order = np.argsort(best)[::-1]
    info = {
        "objective": "same as old direct optimized schedule: mean log gain + 0.35 min log gain - 0.65 var log gain - 30 mean below-one penalty squared",
        "score": float(best_score),
        "n_evaluations": int(eval_count),
        "constraints": "w_tau >= 0 and sum_{tau contains station i} w_tau = 1 for each station",
        "station_weight_sums": {f"S{i + 1}": float(station_sums[i]) for i in range(bm.n)},
        "max_station_weight_error": float(np.max(np.abs(station_sums - 1.0))),
        "top_triangle_weights": {
            h.loop_label(triangles[idx]): float(best[idx])
            for idx in order[:10]
            if best[idx] > 1.0e-8
        },
    }
    return best, best_gains, info


def gains_for_weights(
    bm: h.RawJeBenchmark,
    triangle_edges: dict[tuple[int, int, int], np.ndarray],
    edge_sigma: np.ndarray,
    weights: dict[tuple[int, int, int], float],
) -> np.ndarray:
    fisher = h.rawdirect_weighted_fisher_from_edges(bm, triangle_edges, weights)
    return edge_sigma / np.maximum(h.loop_sigmas(bm, fisher), 1.0e-300)


def make_rows(
    old_local_gains: dict[str, float],
    raw_opt_gains: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for idx, tri in enumerate(h.BALANCED10):
        label = h.loop_label(tri)
        rows.append(
            {
                "loop": label,
                "old_local_schur_opt_gain_vs_edge": float(old_local_gains[label]),
                "rawje_samecost_opt_gain_vs_edge": float(raw_opt_gains[idx]),
                "raw_opt_over_old_local_snr": float(raw_opt_gains[idx] / max(old_local_gains[label], 1.0e-300)),
            }
        )
    return rows


def plot_rows(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    labels = [str(row["loop"]) for row in rows]
    old_local = np.asarray([float(row["old_local_schur_opt_gain_vs_edge"]) for row in rows])
    raw_opt = np.asarray([float(row["rawje_samecost_opt_gain_vs_edge"]) for row in rows])
    n_station = 6
    three_port_limit = math.sqrt(n_station / (n_station - 2.0))
    global_n_limit = math.sqrt(n_station / 2.0)
    x = np.arange(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.3, 4.05), constrained_layout=True)
    ax.bar(x - 0.5 * width, old_local, width, color="#4d4d4d", label="old local-Schur optimal")
    ax.bar(x + 0.5 * width, raw_opt, width, color="#9d0208", label=r"raw $J_e$, same-cost optimal")
    ax.axhline(1.0, color="0.25", lw=1.0, ls="--")
    ax.axhline(
        three_port_limit,
        color="#6a4c93",
        lw=1.2,
        ls=":",
        label=r"$\sqrt{N/(N-2)}$",
    )
    ax.axhline(
        global_n_limit,
        color="#005f73",
        lw=1.2,
        ls="-.",
        label=r"$\sqrt{N/2}$",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.set_ylim(0.95, max(1.55, 1.08 * float(max(np.max(old_local), np.max(raw_opt)))))
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    png = FIGURES / f"{STEM}.png"
    pdf = FIGURES / f"{STEM}.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return pdf, png


def main() -> None:
    bm = h.make_benchmark()
    triangles = h.all_triangles(bm.n)
    old_weights = load_old_weights()
    old_vec = weights_to_vec(triangles, old_weights)
    old_local_gains = load_old_loop_gains()
    edge_sigma = h.loop_sigmas(bm, h.uniform_edge_fisher(bm))
    triangle_edges = h.integrated_raw_triangle_edges(bm, triangles)
    raw_best_vec, raw_best_gains, opt_info = optimize_raw_weights(
        bm,
        triangle_edges,
        edge_sigma,
        old_vec,
        seed=20260616,
    )
    raw_best_weights = vec_to_weights(triangles, raw_best_vec)
    rows = make_rows(old_local_gains, raw_best_gains)
    csv_path = RESULTS / f"{STEM}_loop_gains.csv"
    write_csv(csv_path, rows)
    pdf_path, png_path = plot_rows(rows)

    old_local_array = np.asarray([old_local_gains[h.loop_label(tri)] for tri in h.BALANCED10], dtype=float)
    payload = {
        "definition": {
            "raw_direct": "integrated weighted raw triangle edge Fisher matrices, followed by one global station-gauge Schur complement",
            "optimization_objective": opt_info["objective"],
            "constraints": opt_info["constraints"],
            "old_optimal_source": str(OLD_SUMMARY),
            "old_loop_gain_source": str(OLD_LOOP_GAINS),
            "exposure_s": float(h.aug.EXPOSURE_S),
            "n_samples": int(len(bm.samples)),
        },
        "summary": {
            "old_local_schur_opt_gain_vs_edge": gain_summary(old_local_array),
            "rawje_samecost_opt_gain_vs_edge": gain_summary(raw_best_gains),
            "raw_opt_over_old_local_snr": gain_summary(raw_best_gains / np.maximum(old_local_array, 1.0e-300)),
        },
        "optimization": opt_info,
        "rawje_samecost_opt_weights": {
            h.loop_label(tri): float(weight)
            for tri, weight in raw_best_weights.items()
        },
        "outputs": {
            "loop_gains_csv": str(csv_path),
            "figure_pdf": str(pdf_path),
            "figure_png": str(png_path),
        },
    }
    json_path = RESULTS / f"{STEM}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    weights_path = RESULTS / f"{STEM}_weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "weights": payload["rawje_samecost_opt_weights"],
                "optimization": opt_info,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(payload["summary"], indent=2))
    print(csv_path)
    print(json_path)
    print(weights_path)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
