from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


BUNDLE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import make_all_closure_global_benchmark_note as bm_lib  # noqa: E402
import plot_augmented_existing_telescope_closure_networks as aug  # noqa: E402
import plot_augmented_existing_telescope_ngc_sources as ngc  # noqa: E402
import plot_prl_broadband_clean as base  # noqa: E402
from plot_prl_broadband_blr_realnight import project_enu_baselines  # noqa: E402


OUT = BUNDLE / "exploration" / "hybrid_receiver_approximation"
OUT.mkdir(parents=True, exist_ok=True)

SPLIT_CSV = ROOT / "output" / "pdf" / "all_closure_global_multiobjective_epsst0p02_pair0p01_dir0p01_L1_splits.csv"


def load_split_matrices(bm: bm_lib.AllClosureBenchmark) -> dict[str, np.ndarray]:
    """Load previously optimized station-side split matrices.

    Rows are station-side budgets p_{i->j}; each row sums to one before any
    additional core-vs-remote receiver allocation.
    """
    out: dict[str, np.ndarray] = {}
    if not SPLIT_CSV.exists():
        out["uniform"] = bm.uniform_split_matrix()
        return out
    name_to_idx = {name: idx for idx, name in enumerate(bm.names)}
    with SPLIT_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            objective = row["objective"]
            out.setdefault(objective, np.zeros((bm.n, bm.n), dtype=float))
            i = name_to_idx[row["from_station"]]
            j = name_to_idx[row["to_station"]]
            out[objective][i, j] = float(row["fraction"])
    out.setdefault("uniform", bm.uniform_split_matrix())
    return out


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


def closure_fisher_from_edges(fab: float, fbc: float, fac: float) -> float:
    if min(fab, fbc, fac) <= 0.0:
        return 0.0
    return 1.0 / (1.0 / fab + 1.0 / fbc + 1.0 / fac)


def optimize_triangle_edge_split(
    bm: bm_lib.AllClosureBenchmark,
    tri: tuple[int, int, int],
) -> tuple[float, tuple[float, float, float]]:
    """Best per-loop classical edge-first split for a single triangle.

    This is not a simultaneous all-closure split. It answers the modular
    question: if this remote-involved loop is read in its own time slot, how
    close can optimized edge-first get to the direct-closure benchmark?
    """
    a, b, c = tri
    eab = bm.edge_arrays[(a, b)]
    ebc = bm.edge_arrays[(b, c)]
    eac = bm.edge_arrays[(a, c)]

    def score(xa: float, xb: float, xc: float) -> float:
        return closure_fisher_from_edges(
            edge_fisher_from_arrays(eab, xa, xb),
            edge_fisher_from_arrays(ebc, 1.0 - xb, xc),
            edge_fisher_from_arrays(eac, 1.0 - xa, 1.0 - xc),
        )

    floor = bm_lib.SPLIT_FLOOR
    grid = np.linspace(floor, 1.0 - floor, 81)
    best = (0.5, 0.5, 0.5)
    best_score = score(*best)
    seeds = [
        best,
        (floor, floor, 0.5),
        (1.0 - floor, 1.0 - floor, 0.5),
        (0.5, floor, 1.0 - floor),
        (0.5, 1.0 - floor, floor),
    ]
    for seed in seeds:
        xa, xb, xc = seed
        local_score = score(xa, xb, xc)
        for _ in range(12):
            vals = [(score(v, xb, xc), v) for v in grid]
            _, xa = max(vals, key=lambda item: item[0])
            vals = [(score(xa, v, xc), v) for v in grid]
            _, xb = max(vals, key=lambda item: item[0])
            vals = [(score(xa, xb, v), v) for v in grid]
            local_score, xc = max(vals, key=lambda item: item[0])
        if local_score > best_score:
            best_score = local_score
            best = (xa, xb, xc)
    return best_score, best


def subset_direct_closure_fisher(
    bm: bm_lib.AllClosureBenchmark,
    subset: tuple[int, ...],
    tri_global: tuple[int, int, int],
) -> float:
    """Direct closure Fisher for one triangle/subarray with full local budget."""
    subset = tuple(subset)
    local_map = {global_idx: local_idx for local_idx, global_idx in enumerate(subset)}
    tri_local = tuple(local_map[i] for i in tri_global)
    local_edges = base.edge_list(len(subset))
    local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, len(subset)))
    local_stations = bm.stations[list(subset)]
    local_baselines = np.asarray([local_stations[j] - local_stations[i] for i, j in local_edges], dtype=float)
    local_fq = np.zeros((local_q.shape[1], local_q.shape[1]), dtype=float)
    for lam, freq, total_modes in bm.iter_bands():
        u_station = aug.station_u_modes(freq, bm.diameters[list(subset)])
        uu_rows, vv_rows = project_enu_baselines(
            local_baselines,
            bm.hour_angles,
            lam,
            latitude_deg=bm.case.latitude_deg,
            declination_deg=ngc.NGC4151.dec_deg,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            vlocal = base.interp_vis(bm.vgrid, bm.uv_axis, uu, vv)
            local_fq += total_modes * aug.noisy_closure_fisher_station_u(
                vlocal,
                bm.eta[list(subset)],
                bm.direct_noise[list(subset)],
                u_station,
                local_q,
                local_edges,
            )
    c = edge_vector(local_edges, tri_local)
    d = local_q.T @ c
    cov = np.linalg.pinv(local_fq, rcond=1e-12)
    var = float(d @ cov @ d)
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def core4_direct_fisher_embedded(bm: bm_lib.AllClosureBenchmark, alpha: float) -> np.ndarray:
    """Core-four direct closure Fisher embedded into the full 7-station cycle basis.

    The first four stations are the existing Maunakea core in this benchmark.
    A fraction alpha of each core station mode is sent to the core joint
    receiver. Input background is split with the field; the direct receiver
    adds EPS_DIRECT_EXTRA per mode.
    """
    core = tuple(range(4))
    local_edges = base.edge_list(len(core))
    local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, len(core)))
    local_fq = np.zeros((local_q.shape[1], local_q.shape[1]), dtype=float)

    core_stations = bm.stations[list(core)]
    core_baselines = np.asarray([core_stations[j] - core_stations[i] for i, j in local_edges], dtype=float)
    eta = alpha * bm.eta[list(core)]
    # Station-local input false positives are split with the optical field;
    # the receiver's own direct false-positive load is not.
    direct_noise = alpha * bm_lib.EPS_STATION + bm_lib.EPS_DIRECT_EXTRA
    noise = np.full(len(core), direct_noise, dtype=float)

    for lam, freq, total_modes in bm.iter_bands():
        u_station = aug.station_u_modes(freq, bm.diameters[list(core)])
        uu_rows, vv_rows = project_enu_baselines(
            core_baselines,
            bm.hour_angles,
            lam,
            latitude_deg=bm.case.latitude_deg,
            declination_deg=ngc.NGC4151.dec_deg,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            vlocal = base.interp_vis(bm.vgrid, bm.uv_axis, uu, vv)
            local_fq += total_modes * aug.noisy_closure_fisher_station_u(
                vlocal,
                eta,
                noise,
                u_station,
                local_q,
                local_edges,
            )

    # Map global closure coordinates q_full to local core closure coordinates:
    # q_local = A q_full, so F_full += A^T F_local A.
    selector = np.zeros((len(local_edges), len(bm.edges)), dtype=float)
    full_edge_to_idx = {edge: idx for idx, edge in enumerate(bm.edges)}
    for local_idx, edge in enumerate(local_edges):
        selector[local_idx, full_edge_to_idx[edge]] = 1.0
    amap = local_q.T @ selector @ bm.q_basis
    return amap.T @ local_fq @ amap


def fisher_with_core_budget(
    bm: bm_lib.AllClosureBenchmark,
    split: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Hybrid Fisher matrix for core joint receiver plus remote edge-first network."""
    edge_budget = np.ones(bm.n, dtype=float)
    edge_budget[:4] = 1.0 - alpha
    p_eff = split * edge_budget[:, None]
    return bm.edge_closure_fisher(p_eff) + core4_direct_fisher_embedded(bm, alpha)


def root_loop_rows(bm: bm_lib.AllClosureBenchmark, matrices: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows = []
    covs = {key: np.linalg.pinv(0.5 * (mat + mat.T), rcond=1e-12) for key, mat in matrices.items()}
    for tri in [(0, i, j) for i in range(1, bm.n) for j in range(i + 1, bm.n)]:
        c = edge_vector(bm.edges, tri)
        d = bm.q_basis.T @ c
        row: dict[str, float | str] = {
            "loop": f"{tri[0] + 1}-{tri[1] + 1}-{tri[2] + 1}",
            "stations": " | ".join(bm.names[i] for i in tri),
            "type": "core_only" if all(not bm.is_added[i] for i in tri) else "remote_involved",
        }
        for key, cov in covs.items():
            var = float(d @ cov @ d)
            row[f"rms_{key}_rad"] = math.sqrt(max(var, 0.0)) if np.isfinite(var) else math.inf
        ref = float(row["rms_direct_scheduled_rad"])
        for key in matrices:
            row[f"snr_{key}_over_direct_scheduled"] = ref / max(float(row[f"rms_{key}_rad"]), 1e-300)
        rows.append(row)
    return rows


def modular_loop_rows(bm: bm_lib.AllClosureBenchmark, direct_scheduled: np.ndarray) -> list[dict[str, float | str]]:
    """Per-loop modular implementation: core loops use core/triangle direct; remote loops use optimized edge-first."""
    direct_rows = root_loop_rows(bm, {"direct_scheduled": direct_scheduled})
    direct_rms = {row["loop"]: float(row["rms_direct_scheduled_rad"]) for row in direct_rows}
    rows = []
    for tri in [(0, i, j) for i in range(1, bm.n) for j in range(i + 1, bm.n)]:
        loop = f"{tri[0] + 1}-{tri[1] + 1}-{tri[2] + 1}"
        core_only = all(not bm.is_added[i] for i in tri)
        f_local3 = subset_direct_closure_fisher(bm, tri, tri)
        f_edge_opt, split = optimize_triangle_edge_split(bm, tri)
        if core_only:
            f_modular = f_local3
            receiver = "local 3-mode direct"
        else:
            f_modular = f_edge_opt
            receiver = "per-loop optimized edge-first"
        rms_modular = 1.0 / math.sqrt(max(f_modular, 1e-300))
        rows.append(
            {
                "loop": loop,
                "type": "core_only" if core_only else "remote_involved",
                "receiver": receiver,
                "rms_direct_scheduled_rad": direct_rms[loop],
                "rms_modular_rad": rms_modular,
                "snr_modular_over_direct_scheduled": direct_rms[loop] / max(rms_modular, 1e-300),
                "rms_local3_direct_rad": 1.0 / math.sqrt(max(f_local3, 1e-300)),
                "rms_opt_edge_rad": 1.0 / math.sqrt(max(f_edge_opt, 1e-300)),
                "split_xa": split[0],
                "split_xb": split[1],
                "split_xc": split[2],
            }
        )
    return rows


def metric_summary(fisher: np.ndarray) -> dict[str, float]:
    return bm_lib.stable_metrics(fisher)


def choose_best_hybrid(bm: bm_lib.AllClosureBenchmark, splits: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    reference = {
        "edge_uniform": bm.edge_closure_fisher(splits["uniform"]),
        "direct_raw": bm.direct_raw,
        "direct_scheduled": bm.rank_share * bm.direct_raw,
    }
    candidates = []
    alpha_grid = np.linspace(0.0, 0.9, 31)
    for split_name in ("uniform", "logdet", "mean_rms", "max_rms", "trace"):
        if split_name not in splits:
            continue
        split = splits[split_name]
        for alpha in alpha_grid:
            fisher = fisher_with_core_budget(bm, split, float(alpha))
            metrics = metric_summary(fisher)
            ref_metrics = metric_summary(reference["direct_scheduled"])
            rows = root_loop_rows(bm, {"hybrid": fisher, "direct_scheduled": reference["direct_scheduled"]})
            ratios = np.asarray([float(row["snr_hybrid_over_direct_scheduled"]) for row in rows], dtype=float)
            candidates.append(
                {
                    "split": split_name,
                    "alpha_core_joint": float(alpha),
                    "mean_rms": metrics["mean_coord_rms"],
                    "max_rms": metrics["max_coord_rms"],
                    "median_rms": metrics["median_coord_rms"],
                    "mean_rms_gain_vs_direct_scheduled": ref_metrics["mean_coord_rms"] / metrics["mean_coord_rms"],
                    "median_loop_gain_vs_direct_scheduled": float(np.median(ratios)),
                    "min_loop_gain_vs_direct_scheduled": float(np.min(ratios)),
                    "max_loop_gain_vs_direct_scheduled": float(np.max(ratios)),
                }
            )
    # Use mean coordinate RMS as the primary all-closure implementation metric.
    best = min(candidates, key=lambda item: item["mean_rms"])
    return best, candidates


def latex_text(value: str) -> str:
    return value.replace("_", r"\_")


def write_note(payload: dict, rows: list[dict], modular_rows: list[dict], candidates: list[dict]) -> tuple[Path, Path]:
    tex_path = OUT / "hybrid_receiver_approximation_note.tex"
    pdf_path = OUT / "hybrid_receiver_approximation_note.pdf"

    best = payload["best_hybrid"]
    summary_lines = []
    for item in payload["matrix_metrics"]:
        summary_lines.append(
            f"{latex_text(item['strategy'])} & {item['mean_coord_rms']:.3g} & {item['median_coord_rms']:.3g} & "
            f"{item['max_coord_rms']:.3g} & {item['gain_vs_direct_scheduled_mean_rms']:.2f} \\\\"
        )

    row_lines = []
    for row in rows:
        row_lines.append(
            f"{row['loop']} & {latex_text(str(row['type']))} & {float(row['rms_direct_scheduled_rad']):.3g} & "
            f"{float(row['rms_best_hybrid_rad']):.3g} & "
            f"{float(row['snr_best_hybrid_over_direct_scheduled']):.2f} \\\\"
        )

    modular_lines = []
    for row in modular_rows:
        modular_lines.append(
            f"{row['loop']} & {latex_text(str(row['receiver']))} & {float(row['rms_direct_scheduled_rad']):.3g} & "
            f"{float(row['rms_modular_rad']):.3g} & "
            f"{float(row['snr_modular_over_direct_scheduled']):.2f} \\\\"
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.65in]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{Hybrid receiver approximation for the 7-station closure benchmark}}
\author{{Codex diagnostic note}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Question}}
The previous 7-station direct-closure number used
\[
F_{{\rm dir,scheduled}}=\frac{{N-1}}{{C}}F_{{\rm dir,QFI}},
\qquad N=7,\quad C=15,
\]
so the factor is \(6/15={payload['rank_share']:.2f}\).  This is a uniform scheduling proxy for
the full closure-space QFI, not an explicit simultaneous 7-mode POVM.

\section*{{Hybrid construction tested here}}
We test a more implementable receiver.  A fraction \(\alpha\) of the four existing Maunakea core
stations is sent to a core 4-mode closure receiver.  The remaining core light and all remote-station
light are sent to a classical edge-first network with station-side split matrix
\(p_{{i\to j}}\).  The edge network includes the same pair false-positive load used in the manuscript
benchmark.

The best point in the scan is
\[
\alpha={best['alpha_core_joint']:.2f},\qquad p\text{{ objective}}={best['split']}.
\]

\section*{{Noise/resource model}}
\[
\epsilon_i={payload['eps_station']:.2f},\qquad
\epsilon_{{\rm pair}}=\epsilon_{{\rm dir}}={payload['eps_pair']:.2f},
\qquad L_{{\rm fib}}={payload['fiber_loss_db_per_km']:.2f}\ {{\rm dB/km}}.
\]
Fibre loss is pure signal attenuation.  Input false-positive light is split with the optical field;
the direct receiver has an extra unsplit load \(\epsilon_{{\rm dir}}\).

\section*{{Matrix-level comparison}}
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
strategy & mean RMS & median RMS & max RMS & mean-RMS gain vs scheduled direct \\
\midrule
{chr(10).join(summary_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Root-loop comparison}}
\small
\begin{{longtable}}{{llrrr}}
\toprule
loop & type & scheduled direct RMS & hybrid RMS & hybrid/direct SNR \\
\midrule
{chr(10).join(row_lines)}
\bottomrule
\end{{longtable}}

\section*{{Per-loop modular comparison}}
This second table ignores simultaneous conflicts between different remote-involved loops.  It asks
whether each closure, if assigned its own time slot or switch state, can be implemented without a
global 7-mode POVM.

\begin{{longtable}}{{llrrr}}
\toprule
loop & modular receiver & scheduled direct RMS & modular RMS & modular/direct SNR \\
\midrule
{chr(10).join(modular_lines)}
\bottomrule
\end{{longtable}}

\section*{{Interpretation}}
This hybrid is not identical to the raw 7-mode QFI.  It is closer to an engineering receiver:
core-only closure information is obtained by a small joint receiver on the existing four-station
subarray, while remote-involved closures are mostly obtained by optimized classical splitting.
If its RMS is comparable to \(F_{{\rm dir,scheduled}}\), then the earlier scheduled-direct number
should be read less as evidence for a complicated global POVM and more as evidence that closure-space
resource allocation can be approximated by modular core-joint plus remote edge-first receivers.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True)
    return tex_path, pdf_path


def main() -> None:
    bm = bm_lib.AllClosureBenchmark()
    splits = load_split_matrices(bm)
    best, candidates = choose_best_hybrid(bm, splits)

    matrices = {
        "edge_uniform": bm.edge_closure_fisher(splits["uniform"]),
        "direct_scheduled": bm.rank_share * bm.direct_raw,
        "direct_raw_qfi_upper": bm.direct_raw,
        "best_hybrid": fisher_with_core_budget(bm, splits[best["split"]], best["alpha_core_joint"]),
    }
    ref = metric_summary(matrices["direct_scheduled"])
    matrix_metrics = []
    for key, fisher in matrices.items():
        metrics = metric_summary(fisher)
        matrix_metrics.append(
            {
                "strategy": key,
                **metrics,
                "gain_vs_direct_scheduled_mean_rms": ref["mean_coord_rms"] / metrics["mean_coord_rms"],
            }
        )
    rows = root_loop_rows(bm, matrices)
    modular_rows = modular_loop_rows(bm, matrices["direct_scheduled"])

    csv_path = OUT / "hybrid_receiver_root_loop_rows.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    candidates_path = OUT / "hybrid_receiver_alpha_scan.csv"
    with candidates_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)

    modular_csv_path = OUT / "hybrid_receiver_modular_loop_rows.csv"
    with modular_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(modular_rows[0].keys()))
        writer.writeheader()
        writer.writerows(modular_rows)

    payload = {
        "case": bm.case.key,
        "station_names": bm.names,
        "station_is_added": bm.is_added.tolist(),
        "rank_share": bm.rank_share,
        "eps_station": bm_lib.EPS_STATION,
        "eps_pair": bm_lib.EPS_PAIR,
        "eps_direct_extra": bm_lib.EPS_DIRECT_EXTRA,
        "fiber_loss_db_per_km": bm_lib.FIBER_LOSS_DB_PER_KM,
        "fiber_length_scale": bm_lib.FIBER_LENGTH_SCALE,
        "best_hybrid": best,
        "matrix_metrics": matrix_metrics,
        "rows_csv": str(csv_path),
        "modular_rows_csv": str(modular_csv_path),
        "alpha_scan_csv": str(candidates_path),
    }
    json_path = OUT / "hybrid_receiver_approximation_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    tex_path, pdf_path = write_note(payload, rows, modular_rows, candidates)
    print(json.dumps(payload["best_hybrid"], indent=2))
    print(json_path)
    print(csv_path)
    print(candidates_path)
    print(modular_csv_path)
    print(tex_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
