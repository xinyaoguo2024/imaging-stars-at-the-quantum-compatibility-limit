from __future__ import annotations

import csv
import itertools
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hawaii3_compact_case
import make_all_closure_global_benchmark_note as closure_bm
import plot_augmented_existing_telescope_closure_networks as aug
import plot_prl_broadband_blr_optimized as opt
import run_broad_plume_split_objective_rml as broad
from make_all_closure_global_benchmark_note import stable_metrics


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_remote3_keck1_closure_objective_20260527"
OUT.mkdir(parents=True, exist_ok=True)

GOOD_VARIANT = broad.GOOD_VARIANT
GOOD_CFG = broad.GOOD_CFG


def configure_physics() -> None:
    broad.configure_good_runtime()
    closure_bm.SPLIT_FLOOR = 0.02


def edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


def keck1_loop_basis(bm: closure_bm.AllClosureBenchmark) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    loops = []
    for tri in itertools.combinations(range(bm.n), 3):
        if tri[0] != 0:
            continue
        loops.append((tri, bm.q_basis.T @ edge_vector(bm.edges, tri)))
    return loops


def keck1_loop_rms(bm: closure_bm.AllClosureBenchmark, fisher: np.ndarray) -> np.ndarray:
    cov = np.linalg.pinv(fisher, rcond=1e-12)
    rms = []
    for _tri, d in keck1_loop_basis(bm):
        var = float(d @ cov @ d)
        rms.append(math.sqrt(max(var, 0.0)) if np.isfinite(var) else math.inf)
    return np.asarray(rms, dtype=float)


def optimize_keck1_split(
    bm: closure_bm.AllClosureBenchmark,
    objective: str,
) -> tuple[np.ndarray, dict[str, float]]:
    seed_by_objective = {"mean_rms": 2026052711, "max_rms": 2026052712}
    rng = np.random.default_rng(seed_by_objective[objective])
    n = bm.n
    raw0 = np.zeros((n, n), dtype=float)
    np.fill_diagonal(raw0, -np.inf)

    def score(raw: np.ndarray) -> float:
        p = closure_bm.project_station_splits(raw)
        fisher = bm.edge_closure_fisher(p)
        rms = keck1_loop_rms(bm, fisher)
        if objective == "mean_rms":
            return -math.log(max(float(np.mean(rms)), 1e-300))
        if objective == "max_rms":
            return -math.log(max(float(np.max(rms)), 1e-300))
        raise ValueError(objective)

    best_raw = raw0.copy()
    best_score = score(best_raw)

    starts = [raw0.copy()]
    for scale in (0.5, 1.0, 1.8, 3.0):
        for _ in range(520):
            cand = rng.normal(scale=scale, size=(n, n))
            np.fill_diagonal(cand, -np.inf)
            starts.append(cand)
    for cand in starts:
        value = score(cand)
        if value > best_score:
            best_score = value
            best_raw = cand

    for width in (1.2, 0.55, 0.25, 0.11, 0.05, 0.02, 0.009):
        improved = True
        passes = 0
        while improved and passes < 5:
            improved = False
            passes += 1
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for sign in (-1.0, 1.0):
                        cand = best_raw.copy()
                        cand[i, j] += sign * width
                        value = score(cand)
                        if value > best_score:
                            best_score = value
                            best_raw = cand
                            improved = True

    p = closure_bm.project_station_splits(best_raw)
    rms = keck1_loop_rms(bm, bm.edge_closure_fisher(p))
    return p, {
        "objective": objective,
        "score": best_score,
        "keck1_mean_rms": float(np.mean(rms)),
        "keck1_max_rms": float(np.max(rms)),
        "keck1_median_rms": float(np.median(rms)),
    }


def make_benchmark_and_splits() -> tuple[closure_bm.AllClosureBenchmark, dict[str, np.ndarray], dict]:
    configure_physics()
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    old_loader = closure_bm.rml_cases.load_maunakea_plus3_case
    closure_bm.rml_cases.load_maunakea_plus3_case = lambda: case
    try:
        with broad.morph.patched_variant(GOOD_VARIANT):
            bm = closure_bm.AllClosureBenchmark()
            mean_split, mean_info = optimize_keck1_split(bm, "mean_rms")
            max_split, max_info = optimize_keck1_split(bm, "max_rms")
            splits = {
                "edge_uniform": bm.uniform_split_matrix(),
                "edge_meanrms": mean_split,
                "edge_maxrms": max_split,
            }
    finally:
        closure_bm.rml_cases.load_maunakea_plus3_case = old_loader
    return bm, splits, {"edge_meanrms": mean_info, "edge_maxrms": max_info}


def closure_gain_rows(
    bm: closure_bm.AllClosureBenchmark,
    splits: dict[str, np.ndarray],
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    matrices = {
        "edge_uniform": bm.edge_closure_fisher(splits["edge_uniform"]),
        "edge_meanrms": bm.edge_closure_fisher(splits["edge_meanrms"]),
        "edge_maxrms": bm.edge_closure_fisher(splits["edge_maxrms"]),
        "nmode_scheduled": bm.rank_share * bm.direct_raw,
        "nmode_raw_qfi": bm.direct_raw,
    }
    metrics = {key: stable_metrics(value) for key, value in matrices.items()}
    rows = []
    covariances = {key: np.linalg.pinv(value, rcond=1e-12) for key, value in matrices.items()}
    for tri, d in keck1_loop_basis(bm):
        row = {
            "loop": f"{tri[0] + 1}-{tri[1] + 1}-{tri[2] + 1}",
            "stations": " / ".join(bm.names[i] for i in tri),
            "type": "core" if all(not bm.is_added[i] for i in tri) else "remote",
        }
        for key, cov in covariances.items():
            var = float(d @ cov @ d)
            row[f"rms_{key}_rad"] = math.sqrt(max(var, 0.0)) if np.isfinite(var) else math.inf
        ref = row["rms_edge_uniform_rad"]
        for key in matrices:
            row[f"gain_{key}_vs_uniform"] = ref / max(row[f"rms_{key}_rad"], 1e-300)
        rows.append(row)
    return rows, metrics


def plot_rml_results(case, stats, truth, axis_uas, results: list[dict], tag: str, title_suffix: str) -> tuple[Path, Path]:
    axis = axis_uas
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    result_by = {item["strategy"]: item for item in results}
    fig, axes = plt.subplots(2, 6, figsize=(17.2, 6.3), constrained_layout=True)
    panels = [("truth", "Input source", truth, "black")] + [
        (strategy, label, result_by[strategy]["best"]["image"], color)
        for strategy, label, _start, color in broad.STRATEGIES
    ]
    for col, (strategy, label, image, _color) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        if strategy == "truth":
            ax.set_title(label)
        else:
            m = result_by[strategy]["best"]["metrics"]
            r = result_by[strategy]["best"]["residuals"]
            ax.set_title(
                f"{label}\nBLR r={m['blr_corr']:.3f}, all r={m['global_corr']:.3f}\n"
                rf"$\chi_A^2$={r['amp_reduced_chi2']:.2f}, $\chi_\phi^2$={r['phase_reduced_chi2']:.2f}",
                fontsize=7.6,
            )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")

    theta, truth_prof = broad.morph.angular_profile(truth, axis)
    axes[1, 0].plot(np.rad2deg(theta), truth_prof, color="black", lw=2.0, label="input")
    for strategy, label, _start, color in broad.STRATEGIES:
        _, prof = broad.morph.angular_profile(result_by[strategy]["best"]["image"], axis)
        axes[1, 0].plot(np.rad2deg(theta), prof, color=color, lw=1.35, label=label)
    axes[1, 0].set_title("BLR annular profile")
    axes[1, 0].set_xlabel("azimuth angle (deg)")
    axes[1, 0].set_ylabel("mean-normalized brightness")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=6.2)

    truth_disp = opt.normalize_blr_display(truth)
    for col, (strategy, label, _start, _color) in enumerate(broad.STRATEGIES, start=1):
        ax = axes[1, col]
        residual = opt.normalize_blr_display(result_by[strategy]["best"]["image"]) - truth_disp
        vmax = max(0.08, float(np.percentile(np.abs(residual), 99.0)))
        im = ax.imshow(residual, origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label} residual", fontsize=7.6)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    stress = stats["sample_stress_test"]
    fig.suptitle(
        (
            f"{GOOD_VARIANT.label}: Keck-I-anchored closure-objective splits ({title_suffix})\n"
            f"optimized prior p={GOOD_CFG['prior']:g}, TV={GOOD_CFG['tv']:g}, e={GOOD_CFG['entropy']:g}; "
            f"{stress['rows_after']} kept time rows, {stress['effective_exposure_s'] / 60:.1f} min each; "
            f"rank-share={stats['rank_share']:.2f}; raw-QFI is an upper bound"
        ),
        weight="bold",
    )
    png = OUT / f"keck1_closure_objective_rml_{tag}.png"
    pdf = OUT / f"keck1_closure_objective_rml_{tag}.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def run_rml_case(case, splits: dict[str, np.ndarray], *, tag: str, exposure_scale: float, keep_every: int) -> dict:
    broad.EXPOSURE_SCALE = exposure_scale
    broad.KEEP_EVERY_TIME_ROW = keep_every
    broad.RUN_TAG = f"keck1_{tag}"
    broad.OUTPUT_STEM = f"keck1_closure_objective_rml_{tag}"
    broad.configure_good_runtime()
    bands, stats, truth, axis_uas, prior, starts = broad.simulate_good_bands(case, splits)
    broad.morph.configure_runtime(scan=False)
    broad.morph.amp_rml.PRIOR_WEIGHT = GOOD_CFG["prior"]
    broad.morph.amp_rml.TV_WEIGHT = GOOD_CFG["tv"]
    broad.morph.amp_rml.ENTROPY_WEIGHT = GOOD_CFG["entropy"]
    results = []
    for strategy, label, start_name, _color in broad.STRATEGIES:
        print(f"[keck1-rml:{tag}] {strategy}", flush=True)
        results.append(broad.run_strategy(strategy, label, start_name, case, bands, truth, axis_uas, prior, starts))
    rows = broad.result_rows(results, truth, axis_uas)
    title_suffix = "full reference" if keep_every == 1 and exposure_scale == 1.0 else "thinned half-exposure stress test"
    pdf, png = plot_rml_results(case, stats, truth, axis_uas, results, tag, title_suffix)
    csv_path = OUT / f"keck1_closure_objective_rml_{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT / f"keck1_closure_objective_rml_{tag}_summary.json"
    payload = {
        "tag": tag,
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(csv_path),
        "stats": stats,
        "rows": rows,
        "source_variant": GOOD_VARIANT.__dict__,
        "prior": GOOD_CFG,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def short_name(name: str) -> str:
    return (
        name.replace("Remote-", "R")
        .replace("Gemini North", "Gemini")
        .replace("Keck I", "Keck-I")
        .replace("Keck II", "Keck-II")
    )


def split_table_tex(matrix: np.ndarray, names: list[str], title: str) -> str:
    cols = "l" + "r" * len(names)
    header = "from $\\backslash$ to & " + " & ".join(short_name(name) for name in names) + r" \\"
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        rf"\caption*{{{title}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for i, name in enumerate(names):
        values = []
        for j in range(len(names)):
            values.append("--" if i == j else f"{100.0 * matrix[i, j]:.1f}")
        lines.append(short_name(name) + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_tables_and_summary_pdf(
    bm: closure_bm.AllClosureBenchmark,
    splits: dict[str, np.ndarray],
    split_info: dict,
    closure_rows: list[dict],
    metrics: dict[str, dict[str, float]],
    rml_payloads: list[dict],
) -> tuple[Path, Path, Path, Path]:
    split_csv = OUT / "keck1_closure_objective_split_ratios.csv"
    loop_csv = OUT / "keck1_closure_objective_loop_gains.csv"
    json_path = OUT / "keck1_closure_objective_summary.json"
    tex_path = OUT / "keck1_closure_objective_summary.tex"
    pdf_path = OUT / "keck1_closure_objective_summary.pdf"

    with split_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "from_station", "to_station", "fraction"])
        for strategy, matrix in splits.items():
            for i, ni in enumerate(bm.names):
                for j, nj in enumerate(bm.names):
                    if i != j:
                        writer.writerow([strategy, ni, nj, matrix[i, j]])

    fieldnames = list(closure_rows[0].keys())
    with loop_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(closure_rows)

    payload = {
        "case": bm.case.key,
        "hub_km": list(bm.case.hub_km),
        "station_names": bm.names,
        "station_xy_km": bm.stations.tolist(),
        "split_info": split_info,
        "split_ratios": {key: value.tolist() for key, value in splits.items()},
        "metrics": metrics,
        "closure_rows_keck1_only": closure_rows,
        "rml": rml_payloads,
        "note": "Split optimization objectives include only closures with station 1 (Keck I). The loop-gain table also lists only those Keck-I-anchored closures.",
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    labels = {
        "edge_uniform": "edge uniform",
        "edge_meanrms": "Keck-I mean-RMS split",
        "edge_maxrms": "Keck-I max-RMS split",
        "nmode_scheduled": "N-mode scheduled proxy",
        "nmode_raw_qfi": "N-mode raw-QFI bound",
    }
    ref = metrics["edge_uniform"]
    metric_lines = []
    for key, label in labels.items():
        m = metrics[key]
        metric_lines.append(
            f"{label} & {m['mean_coord_rms']:.3e} & {m['max_coord_rms']:.3e} & "
            f"{math.sqrt(m['trace_fisher'] / ref['trace_fisher']):.2f} & "
            f"{ref['mean_coord_rms'] / m['mean_coord_rms']:.2f} \\\\"
        )

    loop_lines = []
    for row in closure_rows:
        loop_lines.append(
            f"{row['loop']} & {short_name(row['stations'])} & {row['type']} & "
            f"{1e3 * row['rms_edge_uniform_rad']:.2f} & "
            f"{row['gain_edge_meanrms_vs_uniform']:.2f} & "
            f"{row['gain_edge_maxrms_vs_uniform']:.2f} & "
            f"{row['gain_nmode_scheduled_vs_uniform']:.2f} & "
            f"{row['gain_nmode_raw_qfi_vs_uniform']:.2f} \\\\"
        )

    split_tables = "\n\n".join(
        split_table_tex(splits[key], bm.names, title)
        for key, title in (
            ("edge_uniform", "Uniform split ratios $100p_{i\\to j}$ (percent)"),
            ("edge_meanrms", "Keck-I mean-RMS optimized split ratios $100p_{i\\to j}$ (percent)"),
            ("edge_maxrms", "Keck-I max-RMS optimized split ratios $100p_{i\\to j}$ (percent)"),
        )
    )

    full_fig = Path(rml_payloads[0]["figure_png"])
    stress_fig = Path(rml_payloads[1]["figure_png"])
    tex = rf"""\documentclass[9pt]{{article}}
\usepackage[margin=0.45in,landscape]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,caption,graphicx,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\begin{{document}}
\title{{Keck-I-Anchored Closure Optimization: Splits, Gains, and RML Images}}
\author{{Codex diagnostic note}}
\date{{\today}}
\maketitle

\section*{{What changed}}
The split optimization objective now includes only the \(15=\binom{{6}}{{2}}\) triangular closures containing station 1 (Keck I).  These Keck-I-anchored loops form an independent closure basis for the seven-station Hawaii+3 compact network, so the optimizer no longer gives repeated weight to linearly dependent triangles.  The gain table below also lists only those Keck-I closures.

\section*{{Global all-closure metrics using the new split matrices}}
The split matrices are optimized on Keck-I closures, but the matrix metrics below are evaluated on the full all-closure Fisher matrix used by the RML simulator.
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
strategy & mean RMS & max RMS & trace-SNR gain & mean-RMS gain \\
\midrule
{chr(10).join(metric_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Keck-I closure SNR gains}}
\scriptsize
\begin{{longtable}}{{lllrrrrr}}
\toprule
loop & stations & type & RMS$_{{\rm uni}}$ [mrad] & mean-RMS & max-RMS & N-mode sched. & raw QFI \\
\midrule
\endfirsthead
\toprule
loop & stations & type & RMS$_{{\rm uni}}$ [mrad] & mean-RMS & max-RMS & N-mode sched. & raw QFI \\
\midrule
\endhead
{chr(10).join(loop_lines)}
\bottomrule
\end{{longtable}}
\normalsize

\clearpage
\section*{{Station split ratios}}
Rows are transmitting stations and columns are receiving stations.  Entries are percentages of each station's photon budget.
{split_tables}

\clearpage
\section*{{RML image reference: full exposure}}
\begin{{center}}
\includegraphics[width=0.98\linewidth]{{{full_fig}}}
\end{{center}}

\clearpage
\section*{{RML image stress test: every other visibility sample, half exposure}}
\begin{{center}}
\includegraphics[width=0.98\linewidth]{{{stress_fig}}}
\end{{center}}

\section*{{Caveat}}
The N-mode raw-QFI column remains an upper bound.  The scheduled column applies the conservative rank-share factor \((N-1)/M={bm.rank_share:.2f}\).  If one wants a strict CFI for a concrete simultaneous seven-mode receiver, that receiver still has to be specified explicitly.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tex_path, pdf_path, split_csv, loop_csv


def main() -> None:
    bm, splits, split_info = make_benchmark_and_splits()
    closure_rows, metrics = closure_gain_rows(bm, splits)
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    full_payload = run_rml_case(case, splits, tag="full_ref", exposure_scale=1.0, keep_every=1)
    stress_payload = run_rml_case(case, splits, tag="thin2_halfexp", exposure_scale=0.5, keep_every=2)
    tex_path, pdf_path, split_csv, loop_csv = write_tables_and_summary_pdf(
        bm,
        splits,
        split_info,
        closure_rows,
        metrics,
        [full_payload, stress_payload],
    )
    print(tex_path)
    print(pdf_path)
    print(split_csv)
    print(loop_csv)
    print(full_payload["figure_pdf"])
    print(stress_payload["figure_pdf"])


if __name__ == "__main__":
    main()
