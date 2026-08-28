from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from make_all_closure_global_benchmark_note import (
    AllClosureBenchmark,
    EPS_DIRECT_EXTRA,
    EPS_PAIR,
    EPS_STATION,
    FIBER_LENGTH_SCALE,
    FIBER_LOSS_DB_PER_KM,
    SPLIT_FLOOR,
    gain_summary,
    project_station_splits,
    stable_metrics,
)


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)


def metric_score(metric: dict[str, float], objective: str) -> float:
    if objective == "logdet":
        return metric["logdet_fisher"] / 15.0
    if objective == "mean_rms":
        return -math.log(max(metric["mean_coord_rms"], 1e-300))
    if objective == "max_rms":
        return -math.log(max(metric["max_coord_rms"], 1e-300))
    if objective == "trace":
        return math.log(max(metric["trace_fisher"], 1e-300))
    raise ValueError(objective)


def optimize_split(bm: AllClosureBenchmark, objective: str) -> tuple[np.ndarray, dict[str, float]]:
    seed_by_objective = {
        "logdet": 2026052701,
        "mean_rms": 2026052702,
        "max_rms": 2026052703,
        "trace": 2026052704,
    }
    rng = np.random.default_rng(seed_by_objective.get(objective, 2026052799))
    n = bm.n
    raw0 = np.zeros((n, n), dtype=float)
    np.fill_diagonal(raw0, -np.inf)

    def score(raw: np.ndarray) -> float:
        fisher = bm.edge_closure_fisher(project_station_splits(raw))
        return metric_score(stable_metrics(fisher), objective)

    best_raw = raw0.copy()
    best_score = score(best_raw)

    # Include structured starts: uniform plus random starts at several temperatures.
    starts = [raw0.copy()]
    for scale in (0.5, 1.0, 1.8, 3.0):
        for _ in range(420):
            cand = rng.normal(scale=scale, size=(n, n))
            np.fill_diagonal(cand, -np.inf)
            starts.append(cand)
    for cand in starts:
        val = score(cand)
        if val > best_score:
            best_score = val
            best_raw = cand

    # Coordinate refinement.  This is slow but robust enough for this diagnostic.
    for width in (1.2, 0.55, 0.25, 0.11, 0.05, 0.02):
        improved = True
        passes = 0
        while improved and passes < 4:
            improved = False
            passes += 1
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for sign in (-1.0, 1.0):
                        cand = best_raw.copy()
                        cand[i, j] += sign * width
                        val = score(cand)
                        if val > best_score:
                            best_score = val
                            best_raw = cand
                            improved = True

    p = project_station_splits(best_raw)
    fisher = bm.edge_closure_fisher(p)
    metrics = stable_metrics(fisher)
    return p, {"objective": objective, "score": best_score, "metrics": metrics}


def fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    if abs(x) >= 1e3 or (0 < abs(x) < 1e-2):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def write_outputs() -> tuple[Path, Path, Path, Path, dict]:
    bm = AllClosureBenchmark()
    p_uniform = bm.uniform_split_matrix()

    split_payloads = {"uniform": {"p": p_uniform, "info": {"objective": "uniform"}}}
    for obj in ("logdet", "mean_rms", "max_rms", "trace"):
        p, info = optimize_split(bm, obj)
        split_payloads[obj] = {"p": p, "info": info}

    matrices = {
        "edge_uniform": bm.edge_closure_fisher(split_payloads["uniform"]["p"]),
        "edge_logdet": bm.edge_closure_fisher(split_payloads["logdet"]["p"]),
        "edge_meanrms": bm.edge_closure_fisher(split_payloads["mean_rms"]["p"]),
        "edge_maxrms": bm.edge_closure_fisher(split_payloads["max_rms"]["p"]),
        "edge_trace": bm.edge_closure_fisher(split_payloads["trace"]["p"]),
        "direct_raw_qfi": bm.direct_raw,
        "direct_scheduled_proxy": bm.rank_share * bm.direct_raw,
    }
    metrics = {key: stable_metrics(value) for key, value in matrices.items()}
    gains = {
        f"{key}_vs_edge_uniform": gain_summary(metrics[key], metrics["edge_uniform"])
        for key in matrices
        if key != "edge_uniform"
    }
    loop_rows = bm.keck1_loop_rows(matrices)

    tag = "all_closure_global_multiobjective_epsst0p02_pair0p01_dir0p01_L1"
    csv_path = OUT / f"{tag}_keck1_loop_rms.csv"
    json_path = OUT / f"{tag}.json"
    split_csv = OUT / f"{tag}_splits.csv"
    tex_path = OUT / f"{tag}_note.tex"
    pdf_path = OUT / f"{tag}_note.pdf"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(loop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loop_rows)

    with split_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["objective", "from_station", "to_station", "fraction"])
        for objective, payload in split_payloads.items():
            p = payload["p"]
            for i, ni in enumerate(bm.names):
                for j, nj in enumerate(bm.names):
                    if i != j:
                        writer.writerow([objective, ni, nj, p[i, j]])

    payload = {
        "metadata": {
            "case": bm.case.key,
            "n_station": bm.n,
            "n_baseline": len(bm.edges),
            "n_closure": bm.n_closure,
            "rank_share": bm.rank_share,
            "eps_station": EPS_STATION,
            "eps_pair": EPS_PAIR,
            "eps_direct_extra": EPS_DIRECT_EXTRA,
            "split_floor": SPLIT_FLOOR,
            "fiber_length_scale": FIBER_LENGTH_SCALE,
            "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
            "direct_caveat": "direct_raw_qfi is a matrix QFI upper bound; direct_scheduled_proxy is a heuristic scheduling proxy, not a constructed simultaneous POVM CFI.",
        },
        "metrics": metrics,
        "gains": gains,
        "split_objectives": {key: val["info"] for key, val in split_payloads.items()},
        "loop_rows": loop_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    metric_lines = []
    labels = {
        "edge_uniform": "edge uniform",
        "edge_logdet": "edge opt-logdet",
        "edge_meanrms": "edge opt-meanRMS",
        "edge_maxrms": "edge opt-maxRMS",
        "edge_trace": "edge opt-trace",
        "direct_raw_qfi": "direct raw QFI upper",
        "direct_scheduled_proxy": "direct scheduled proxy",
    }
    for key in labels:
        m = metrics[key]
        metric_lines.append(
            f"{labels[key]} & {m['trace_fisher']:.3e} & {m['geomean_fisher_eigen']:.3e} & "
            f"{m['mean_coord_rms']:.3g} & {m['max_coord_rms']:.3g} \\\\"
        )

    gain_lines = []
    for key in ("edge_logdet", "edge_meanrms", "edge_maxrms", "edge_trace", "direct_raw_qfi", "direct_scheduled_proxy"):
        g = gains[f"{key}_vs_edge_uniform"]
        gain_lines.append(
            f"{labels[key]} / edge uniform & {g['trace_snr_gain']:.2f} & {g['logdet_snr_gain']:.2f} & "
            f"{g['mean_rms_gain']:.2f} & {g['median_rms_gain']:.2f} \\\\"
        )

    loop_lines = []
    for row in loop_rows:
        loop_lines.append(
            f"{row['loop']} & {row['type']} & {row['rms_edge_uniform_rad']:.3g} & "
            f"{row['rms_edge_meanrms_rad']:.3g} & {row['rms_edge_maxrms_rad']:.3g} & "
            f"{row['rms_direct_scheduled_proxy_rad']:.3g} & {row['rms_direct_raw_qfi_rad']:.3g} \\\\"
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.56in]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\begin{{document}}
\title{{All-Closure Global Benchmark: Edge Splitting Objectives and Direct-QFI Caveat}}
\author{{Codex diagnostic note}}
\date{{\today}}
\maketitle

\section*{{What was corrected}}
There are two corrections relative to the previous note.
First, ``edge-optimal'' is not unique for a multiparameter closure problem: optimizing trace, log determinant, mean RMS, or worst RMS gives different split matrices.  A split that improves weak closure directions can reduce the trace because it removes budget from very strong short baselines.
Second, the global direct result is not a constructed simultaneous \(N\)-mode POVM.  The raw direct matrix below is the closure-space QFI upper bound.  The scheduled direct matrix is the heuristic proxy
\[
F_{{\rm dir,sch}}=\frac{{N-1}}{{M}}F_{{\rm dir,QFI}}={bm.rank_share:.2f}F_{{\rm dir,QFI}},
\]
not a strict CFI unless an explicit simultaneous receiver is supplied.

\section*{{Model}}
We use \(N={bm.n}\), \(M={bm.n_closure}\), \(\epsilon_i={EPS_STATION:.2f}\), \(\epsilon_{{ij}}={EPS_PAIR:.2f}\), and \(\epsilon_i^{{\rm dir}}={EPS_DIRECT_EXTRA:.2f}\).  Each edge-first split obeys
\[
\sum_{{j\ne i}}p_{{i\to j}}=1,\qquad p_{{i\to j}}\ge {SPLIT_FLOOR:.2f}.
\]

\section*{{Matrix-level metrics}}
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
strategy & \(\mathrm{{Tr}}F\) & geom. eig. \(F\) & mean RMS & max RMS \\
\midrule
{chr(10).join(metric_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{SNR-like gains relative to uniform edge-first}}
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
comparison & trace gain & logdet gain & mean-RMS gain & median-RMS gain \\
\midrule
{chr(10).join(gain_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Keck-I loop marginal RMS from all-closure covariance}}
\small
\begin{{longtable}}{{llrrrrr}}
\toprule
loop & type & edge uniform & edge meanRMS & edge maxRMS & direct scheduled & direct raw QFI \\
\midrule
\endfirsthead
\toprule
loop & type & edge uniform & edge meanRMS & edge maxRMS & direct scheduled & direct raw QFI \\
\midrule
\endhead
{chr(10).join(loop_lines)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Interpretation}}
Your intuition that split optimization should help is correct, but the amount of improvement depends on the global objective.  For example, a mean-RMS or max-RMS objective substantially improves weak closure directions, while a logdet objective gives only a small balanced improvement and a trace objective can make weak directions worse.  Direct closure still shows a strong matrix-level advantage, but only the raw-QFI column is a formal quantum upper bound; the scheduled column should be read as a conservative engineering proxy rather than a derived 7-mode POVM.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return csv_path, split_csv, tex_path, pdf_path, payload


def main() -> None:
    csv_path, split_csv, tex_path, pdf_path, payload = write_outputs()
    print(csv_path)
    print(split_csv)
    print(tex_path)
    print(pdf_path)
    print(json.dumps(payload["gains"], indent=2))


if __name__ == "__main__":
    main()
