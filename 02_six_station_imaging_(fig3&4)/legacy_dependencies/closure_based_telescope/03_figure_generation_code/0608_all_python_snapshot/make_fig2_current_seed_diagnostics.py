from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUNDLE = Path(__file__).resolve().parents[2]
ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
PARENT_OUT = BUNDLE / "rml_remote3_broad_plume_split_objective_20260608_photonlimited"
FIG_DIR = BUNDLE / "figures" / "diagnostics"
OUT = BUNDLE / "exploration" / "fig2_current_seed_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def value_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


CURRENT_ENV = {
    "MPLBACKEND": "Agg",
    "PYTHONPATH": f"{BUNDLE / 'code' / 'all_python_snapshot'}:{BUNDLE / 'code' / 'core'}",
    "BROAD_PLUME_OBSERVING_DAYS": os.environ.get("BROAD_PLUME_OBSERVING_DAYS", os.environ.get("OBSERVING_DAYS", "1")),
    "OBSERVING_DAYS": os.environ.get("OBSERVING_DAYS", "1"),
    "FIG2_N_TIME_WINDOWS": os.environ.get("FIG2_N_TIME_WINDOWS", "36"),
    "FIG2_EXPOSURE_S": os.environ.get("FIG2_EXPOSURE_S", os.environ.get("EXPOSURE_S", "0.050")),
    "EXPOSURE_S": os.environ.get("EXPOSURE_S", "0.050"),
    "FIG2_SAMPLE_CADENCE_S": os.environ.get("FIG2_SAMPLE_CADENCE_S", os.environ.get("SAMPLE_CADENCE_S", "900.0")),
    "FIG2_POST_AVERAGE_DRIFT_STD": os.environ.get(
        "FIG2_POST_AVERAGE_DRIFT_STD",
        os.environ.get("POST_AVERAGE_DRIFT_STD", str(math.pi / 5.0)),
    ),
    "POST_AVERAGE_DRIFT_STD": os.environ.get("POST_AVERAGE_DRIFT_STD", str(math.pi / 5.0)),
    "FIG2_LAMBDA_MIN_NM": os.environ.get("FIG2_LAMBDA_MIN_NM", "600.0"),
    "FIG2_LAMBDA_MAX_NM": os.environ.get("FIG2_LAMBDA_MAX_NM", "700.0"),
    "FIG2_LAMBDA_STEP_NM": os.environ.get("FIG2_LAMBDA_STEP_NM", "10.0"),
    "FIG2_PER_BAND_RML": os.environ.get("FIG2_PER_BAND_RML", "1"),
    "FIG2_DISPLAY_LOG_VMIN": os.environ.get("FIG2_DISPLAY_LOG_VMIN", "5e-3"),
    "EPS_STATION": os.environ.get("EPS_STATION", "1e-9"),
    "EPS_PAIR": os.environ.get("EPS_PAIR", "0.0"),
    "EPS_DIRECT_EXTRA": os.environ.get("EPS_DIRECT_EXTRA", "0.0"),
    "HAWAII3_REMOTE_X_SCALE": os.environ.get("HAWAII3_REMOTE_X_SCALE", "1.0"),
    "HAWAII3_REMOTE_Y_SCALE": os.environ.get("HAWAII3_REMOTE_Y_SCALE", "0.85"),
    "FIG2_EXISTING_COUPLING": os.environ.get("FIG2_EXISTING_COUPLING", "0.1"),
    "FIG2_REMOTE_DIAMETER_M": os.environ.get("FIG2_REMOTE_DIAMETER_M", "2.0"),
    "FIG2_SOURCE_VARIANT": os.environ.get("FIG2_SOURCE_VARIANT", "ngc4151_core_disk_blr_halpha"),
    "KEEP_EVERY_TIME_ROW": os.environ.get("KEEP_EVERY_TIME_ROW", "1"),
    "EXPOSURE_SCALE": os.environ.get("EXPOSURE_SCALE", "1.0"),
    "RML_FIT_N_PIX": os.environ.get("RML_FIT_N_PIX", "40"),
    "RML_ADAM_ITER": os.environ.get("RML_ADAM_ITER", "2600"),
    "RML_ADAM_LR": os.environ.get("RML_ADAM_LR", "0.010"),
    "RML_PRIOR_WEIGHT": os.environ.get("RML_PRIOR_WEIGHT", "0.01"),
    "RML_TV_WEIGHT": os.environ.get("RML_TV_WEIGHT", "0.01"),
    "RML_ENTROPY_WEIGHT": os.environ.get("RML_ENTROPY_WEIGHT", "0.005"),
    "RML_CHI2_THRESHOLD": os.environ.get("RML_CHI2_THRESHOLD", "3.0"),
    "RML_CHI2_MAX_PASSES": os.environ.get("RML_CHI2_MAX_PASSES", "4"),
    "RML_PHASE_CHI2_SELECTION": os.environ.get("RML_PHASE_CHI2_SELECTION", "minimize"),
    "RML_PHASE_CHI2_TARGET": os.environ.get("RML_PHASE_CHI2_TARGET", "1.0"),
    "RML_PHASE_CHI2_MIN": os.environ.get("RML_PHASE_CHI2_MIN", "0.90"),
    "RML_PHASE_CHI2_MAX": os.environ.get("RML_PHASE_CHI2_MAX", "1.05"),
    "PAIR_CORE_DIRECT_NOISE": os.environ.get("PAIR_CORE_DIRECT_NOISE", "1"),
}
for key, value in CURRENT_ENV.items():
    os.environ.setdefault(key, value)
os.environ.setdefault("FIG2_RNG_SEED", "20260529")

SNAPSHOT_DIR = BUNDLE / "code" / "all_python_snapshot"
CORE_DIR = BUNDLE / "code" / "core"
for _path in (CORE_DIR, SNAPSHOT_DIR):
    _path_text = str(_path)
    if _path_text in sys.path:
        sys.path.remove(_path_text)
    sys.path.insert(0, _path_text)

import hawaii3_compact_case  # noqa: E402
import plot_prl_broadband_clean as base  # noqa: E402
import run_broad_plume_split_objective_rml as fig_run  # noqa: E402
import test_fig3_split_objective_imaging as split_sim  # noqa: E402
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles  # noqa: E402


SEEDS = [20260528, 20260529, 20260530, 20260601, 20260602]
STRATEGIES = [
    ("all", "all-vis", "#6a4c93"),
    ("edge_uniform", "edge-first", "#0077b6"),
    ("core4_remote_optimized", "near-opt", "#f77f00"),
    ("nmode_joint_scheduled", "cap.", "#d00000"),
]
LOOP_STRATEGIES = [
    ("edge_uniform", "edge-first", "#0077b6"),
    ("edge_nonsplitting", "non-splitting edge-first", "#2a9d8f"),
    ("core4_remote_optimized", "near-opt", "#f77f00"),
    ("direct_local_raw", "true direct", "#d00000"),
]
LOOPS = [
    ((0, 1, 2), "123", "low"),
    ((0, 1, 4), "125", "mid"),
    ((0, 1, 6), "127", "high"),
]


def epsilon_suffix() -> str:
    values = (
        float(CURRENT_ENV["EPS_STATION"]),
        float(CURRENT_ENV["EPS_PAIR"]),
        float(CURRENT_ENV["EPS_DIRECT_EXTRA"]),
    )
    if np.allclose(values, (0.02, 0.01, 0.01), rtol=0.0, atol=1e-15):
        return ""
    return f"_eps{value_tag(values[0])}_pair{value_tag(values[1])}_dir{value_tag(values[2])}"


def diagnostic_suffix() -> str:
    chi2_tag = "minphase" if CURRENT_ENV["RML_PHASE_CHI2_SELECTION"].strip().lower() == "minimize" else "chi2gate"
    return (
        f"_{fig_run.DEFAULT_RUN_TAG}_fit{CURRENT_ENV['RML_FIT_N_PIX']}"
        f"_i{CURRENT_ENV['RML_ADAM_ITER']}_p{value_tag(float(CURRENT_ENV['RML_PRIOR_WEIGHT']))}"
        f"_tv{value_tag(float(CURRENT_ENV['RML_TV_WEIGHT']))}_{chi2_tag}"
    )


OUTPUT_SUFFIX = diagnostic_suffix()


def run_tag(seed: int) -> str:
    chi2_tag = "minphase" if CURRENT_ENV["RML_PHASE_CHI2_SELECTION"].strip().lower() == "minimize" else "chi2gate"
    return (
        f"{fig_run.DEFAULT_RUN_TAG}_seed{seed}_fit{CURRENT_ENV['RML_FIT_N_PIX']}"
        f"_i{CURRENT_ENV['RML_ADAM_ITER']}_p{value_tag(float(CURRENT_ENV['RML_PRIOR_WEIGHT']))}"
        f"_tv{value_tag(float(CURRENT_ENV['RML_TV_WEIGHT']))}_{chi2_tag}"
    )


def output_stem(seed: int) -> str:
    return f"broad_plume_split_objective_nmode_rml_{run_tag(seed)}"


def expected_paths(seed: int) -> dict[str, Path]:
    stem = output_stem(seed)
    return {
        "pdf": PARENT_OUT / f"{stem}.pdf",
        "png": PARENT_OUT / f"{stem}.png",
        "metrics": PARENT_OUT / f"{stem}_metrics.csv",
        "summary": PARENT_OUT / f"{stem}_summary.json",
    }


def ensure_seed_run(seed: int, *, force: bool = False) -> dict[str, Path]:
    paths = expected_paths(seed)
    if not force and all(path.exists() for path in paths.values()):
        print(f"[skip] seed={seed} already available", flush=True)
        return paths

    env = dict(os.environ)
    env.update(CURRENT_ENV)
    env.update({"FIG2_RNG_SEED": str(seed), "RUN_TAG": run_tag(seed)})
    log_path = OUT / f"{run_tag(seed)}.log"
    print(f"[run] seed={seed} -> {run_tag(seed)}", flush=True)
    with log_path.open("w") as log:
        subprocess.run(
            [sys.executable, str(SNAPSHOT_DIR / "run_broad_plume_split_objective_rml.py")],
            cwd=BUNDLE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing expected seed outputs: " + ", ".join(missing))
    return paths


def read_metrics_csv(path: Path) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            converted: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"strategy", "label", "best_start"}:
                    converted[key] = value
                else:
                    converted[key] = float(value)
            rows[str(row["strategy"])] = converted
    return rows


def collect_seed_metrics(*, force: bool = False) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for idx, seed in enumerate(SEEDS, start=1):
        width = 18
        filled = int(round(width * (idx - 1) / len(SEEDS)))
        print(f"[5seed] [{'#' * filled}{'.' * (width - filled)}] seed={seed}", flush=True)
        paths = ensure_seed_run(seed, force=force)
        by_strategy = read_metrics_csv(paths["metrics"])
        row: dict[str, float | str | int] = {
            "seed": seed,
            "tag": run_tag(seed),
            "metrics": str(paths["metrics"]),
            "summary": str(paths["summary"]),
        }
        for strategy, _label, _color in STRATEGIES:
            m = by_strategy[strategy]
            row[f"{strategy}_global_corr"] = float(m["global_corr"])
            row[f"{strategy}_blr_corr"] = float(m["blr_corr"])
            row[f"{strategy}_amp_chi2"] = float(m["amp_chi2"])
            row[f"{strategy}_phase_chi2"] = float(m["phase_chi2"])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_seed_correspondence(rows: list[dict[str, float | str | int]]) -> tuple[Path, Path]:
    rng = np.random.default_rng(12345)
    fig, ax = plt.subplots(figsize=(6.9, 3.75), constrained_layout=True)
    x = np.arange(len(STRATEGIES), dtype=float)
    width = 0.32
    metric_specs = [
        ("global_corr", "all", -0.5 * width, 0.76, ""),
        ("blr_corr", "BLR", 0.5 * width, 0.42, "///"),
    ]
    for metric_key, metric_label, offset, alpha, hatch in metric_specs:
        means = []
        for strategy, _label, _color in STRATEGIES:
            values = np.asarray([float(row[f"{strategy}_{metric_key}"]) for row in rows], dtype=float)
            means.append(float(np.mean(values)))
        bars = ax.bar(
            x + offset,
            means,
            color=[color for _strategy, _label, color in STRATEGIES],
            alpha=alpha,
            width=width,
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch,
            label=metric_label,
        )
        for idx, (strategy, _label, color) in enumerate(STRATEGIES):
            values = np.asarray([float(row[f"{strategy}_{metric_key}"]) for row in rows], dtype=float)
            jitter = rng.uniform(-0.055, 0.055, size=len(values))
            ax.scatter(
                np.full(len(values), x[idx] + offset) + jitter,
                values,
                s=30,
                facecolor="white",
                edgecolor=color,
                linewidth=1.15,
                zorder=5,
            )
    ax.set_xticks(x, [label for _strategy, label, _color in STRATEGIES], rotation=15, ha="right")
    ax.set_ylabel("correspondence r")
    y_max = max(
        float(row[f"{strategy}_{metric_key}"])
        for row in rows
        for strategy, _label, _color in STRATEGIES
        for metric_key in ("global_corr", "blr_corr")
    )
    ax.set_ylim(0.48, min(1.0, y_max + 0.035))
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    png = FIG_DIR / f"fig2_current_5seed_correspondence_bars{OUTPUT_SUFFIX}.png"
    pdf = FIG_DIR / f"fig2_current_5seed_correspondence_bars{OUTPUT_SUFFIX}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_compact_seed_loop_diagnostic(
    seed_rows: list[dict[str, float | str | int]],
    loop_rows: list[dict[str, float | str]],
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 6.2,
            "axes.labelsize": 6.2,
            "axes.titlesize": 6.6,
            "legend.fontsize": 5.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
        }
    )
    fig = plt.figure(figsize=(7.05, 2.05), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.55, 1.0, 1.0, 1.0], wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax_global = ax.twinx()
    x = np.arange(len(STRATEGIES), dtype=float)
    width = 0.31
    metric_data: dict[str, tuple[list[float], list[float]]] = {}
    for metric_key in ("blr_corr", "global_corr"):
        means = []
        stds = []
        for strategy, _label, _color in STRATEGIES:
            values = np.asarray([float(row[f"{strategy}_{metric_key}"]) for row in seed_rows], dtype=float)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)))
        metric_data[metric_key] = (means, stds)

    blr_means, blr_stds = metric_data["blr_corr"]
    global_means, global_stds = metric_data["global_corr"]
    colors = [color for _strategy, _label, color in STRATEGIES]
    ax.bar(
        x - 0.5 * width,
        blr_means,
        yerr=blr_stds,
        color=colors,
        alpha=0.76,
        width=width,
        edgecolor="black",
        linewidth=0.45,
        error_kw={"lw": 0.45, "capsize": 1.3},
        label="BLR",
    )
    ax_global.bar(
        x + 0.5 * width,
        global_means,
        yerr=global_stds,
        color=colors,
        alpha=0.28,
        width=width,
        edgecolor="black",
        linewidth=0.45,
        hatch="//",
        error_kw={"lw": 0.45, "capsize": 1.3},
        label="global",
    )
    rng = np.random.default_rng(20260609)
    for idx, (strategy, _label, color) in enumerate(STRATEGIES):
        blr_values = np.asarray([float(row[f"{strategy}_blr_corr"]) for row in seed_rows], dtype=float)
        global_values = np.asarray([float(row[f"{strategy}_global_corr"]) for row in seed_rows], dtype=float)
        ax.scatter(
            np.full(len(blr_values), x[idx] - 0.5 * width) + rng.uniform(-0.035, 0.035, size=len(blr_values)),
            blr_values,
            s=7,
            facecolor="white",
            edgecolor=color,
            linewidth=0.55,
            zorder=4,
        )
        ax_global.scatter(
            np.full(len(global_values), x[idx] + 0.5 * width) + rng.uniform(-0.035, 0.035, size=len(global_values)),
            global_values,
            s=7,
            marker="s",
            facecolor="white",
            edgecolor=color,
            linewidth=0.55,
            zorder=4,
        )
    ax.set_xticks(x, ["all", "edge", "near", "direct"], rotation=20, ha="right")
    ax.set_ylabel("BLR correspondence")
    ax_global.set_ylabel("global correspondence")
    all_blr_values = np.asarray(
        [
            float(row[f"{strategy}_blr_corr"])
            for row in seed_rows
            for strategy, _label, _color in STRATEGIES
        ],
        dtype=float,
    )
    all_global_values = np.asarray(
        [
            float(row[f"{strategy}_global_corr"])
            for row in seed_rows
            for strategy, _label, _color in STRATEGIES
        ],
        dtype=float,
    )
    ax.set_ylim(max(0.0, float(np.min(all_blr_values)) - 0.045), min(1.0, float(np.max(all_blr_values)) + 0.045))
    ax_global.set_ylim(
        max(0.0, float(np.min(all_global_values)) - 0.018),
        min(1.0, float(np.max(all_global_values)) + 0.018),
    )
    ax.grid(axis="y", color="0.88", lw=0.45)
    ax.set_axisbelow(True)
    ax.set_title("5-seed correspondence")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_global.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left", handlelength=1.0, borderpad=0.1)

    line_specs = [
        ("edge_uniform", "edge", "#0077b6", "o"),
        ("edge_nonsplitting", "non-splitting edge-first", "#2a9d8f", "D"),
        ("core4_remote_optimized", "near", "#f77f00", "s"),
        ("scheduled_direct_proxy", "direct", "#d00000", "^"),
    ]
    rms_values_deg: list[float] = []
    loop_axes = [fig.add_subplot(gs[0, idx]) for idx in range(1, 4)]
    for ax_loop, (_tri, loop, kind) in zip(loop_axes, LOOPS):
        loop_subset = [row for row in loop_rows if row["loop"] == loop]
        by_strategy = {
            strategy: sorted(
                [row for row in loop_subset if row["strategy"] == strategy],
                key=lambda row: float(row["lambda_center_nm"]),
            )
            for strategy, _label, _color in LOOP_STRATEGIES
        }
        direct_rows = by_strategy["direct_local_raw"]
        for strategy, label, color, marker in line_specs:
            if strategy == "scheduled_direct_proxy":
                vals = direct_rows
                lam = np.asarray([float(row["lambda_center_nm"]) for row in vals], dtype=float)
                rms_deg = np.asarray([float(row["scheduled_direct_proxy_rms_rad"]) for row in vals], dtype=float) * 180.0 / math.pi
            else:
                vals = by_strategy[strategy]
                lam = np.asarray([float(row["lambda_center_nm"]) for row in vals], dtype=float)
                rms_deg = np.asarray([float(row["rms_rad"]) for row in vals], dtype=float) * 180.0 / math.pi
            rms_values_deg.extend([float(v) for v in rms_deg if np.isfinite(v) and v > 0])
            ax_loop.plot(
                lam,
                rms_deg,
                lw=0.9,
                color=color,
                marker=marker,
                ms=2.2,
                label=label if loop == "123" else None,
            )
        ax_loop.set_title(f"loop {loop} ({kind})")
        ax_loop.set_yscale("log")
        ax_loop.set_xlim(600, 700)
        ax_loop.set_xticks([605, 655, 695], ["605", "655", "695"])
        ax_loop.grid(True, which="both", axis="y", color="0.88", lw=0.42)
        ax_loop.grid(True, which="major", axis="x", color="0.93", lw=0.35)
        ax_loop.set_axisbelow(True)
    if rms_values_deg:
        ymin = max(min(rms_values_deg) / 1.45, 0.08)
        ymax = max(rms_values_deg) * 1.45
        for ax_loop in loop_axes:
            ax_loop.set_ylim(ymin, ymax)
    loop_axes[0].set_ylabel("effective RMS (deg)")
    for ax_loop in loop_axes[1:]:
        ax_loop.tick_params(labelleft=False)
    for ax_loop in loop_axes:
        ax_loop.set_xlabel("nm")
    handles, labels = loop_axes[0].get_legend_handles_labels()
    loop_axes[0].legend(handles, labels, frameon=False, loc="upper left", handlelength=1.15, borderpad=0.1)

    png = FIG_DIR / f"fig2_compact_seed_loop_diagnostic_singlecol{OUTPUT_SUFFIX}.png"
    pdf = FIG_DIR / f"fig2_compact_seed_loop_diagnostic_singlecol{OUTPUT_SUFFIX}.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def closure_edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


def marginal_rms(fisher: np.ndarray, q_basis: np.ndarray, edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> float:
    fisher = 0.5 * (fisher + fisher.T)
    cov = np.linalg.pinv(fisher, rcond=1e-12)
    d = q_basis.T @ closure_edge_vector(edges, tri)
    var = float(d @ cov @ d)
    return math.sqrt(max(var, 0.0)) if np.isfinite(var) else math.inf


def uniform_edge_scalar_loop_fisher_for_sample(
    split: np.ndarray,
    tri: tuple[int, int, int],
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    edge_to_index: dict[tuple[int, int], int],
) -> float:
    a, b, c = tri
    edge_values = []
    for i, j in ((a, b), (b, c), (a, c)):
        edge_values.append(
            split_sim.edge_pair_fisher_for_sample(
                i,
                j,
                split[i, j],
                split[j, i],
                total_modes=total_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                nu_eff=nu_eff,
                edge_to_index=edge_to_index,
            )
        )
    return split_sim.scalar_closure_fisher_from_edges(*edge_values)


def nonsplitting_edge_scalar_loop_fisher_for_sample(
    tri: tuple[int, int, int],
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    edge_to_index: dict[tuple[int, int], int],
) -> float:
    """Diagnostic no-fanout edge-first reference with f=1 on each displayed loop edge."""
    a, b, c = tri
    edge_values = []
    for i, j in ((a, b), (b, c), (a, c)):
        edge_values.append(
            split_sim.edge_pair_fisher_for_sample(
                i,
                j,
                1.0,
                1.0,
                total_modes=total_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                nu_eff=nu_eff,
                edge_to_index=edge_to_index,
            )
        )
    return split_sim.scalar_closure_fisher_from_edges(*edge_values)


def core4_remote_scalar_loop_fisher_for_sample(
    spec: dict,
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    edge_to_index: dict[tuple[int, int], int],
) -> float:
    tri = tuple(spec["tri"])
    a, b, c = tri
    pairs = [(a, b), (b, c), (a, c)]
    incident = {station: [] for station in tri}
    for i, j in pairs:
        incident[i].append(j)
        incident[j].append(i)

    directed: dict[tuple[int, int], float] = {}
    for value, station in zip(spec["split"], tri):
        total = 1.0 if station not in split_sim.CORE_STATIONS else split_sim.CORE_JOINT_FRACTION
        value = min(max(float(value), 0.0), total)
        first, second = incident[station]
        directed[(station, first)] = value
        directed[(station, second)] = total - value

    edge_values = []
    for i, j in pairs:
        if i in split_sim.CORE_STATIONS and j in split_sim.CORE_STATIONS:
            fi = fj = split_sim.CORE_JOINT_FRACTION
        else:
            fi = directed[(i, j)]
            fj = directed[(j, i)]
        edge_values.append(
            split_sim.edge_pair_fisher_for_sample(
                i,
                j,
                fi,
                fj,
                total_modes=total_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                nu_eff=nu_eff,
                edge_to_index=edge_to_index,
            )
        )
    return split_sim.scalar_closure_fisher_from_edges(*edge_values)


def scalar_fisher_from_cycle_matrix(
    fisher: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
    tri: tuple[int, int, int],
) -> float:
    d = q_basis.T @ split_sim.closure_edge_vector(edges, tri)
    cov = np.linalg.pinv(0.5 * (fisher + fisher.T), rcond=1e-12)
    var = float(d @ cov @ d)
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def per_wavelength_loop_rms() -> list[dict[str, float | str]]:
    fig_run.configure_good_runtime()
    case = fig_run.scale_remote_coordinates(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    splits = fig_run.make_split_matrices(case)
    split_sim.configure()
    fig_run.apply_sample_stress_runtime()

    stations, diameters, _names, _is_added = fig_run.aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n = len(stations)
    edges = base.edge_list(n)
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, n))
    rank_share = min(1.0, (n - 1.0) / q_basis.shape[1])
    fov_rad = 2.0 * fig_run.aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    effective_hub_dist = fig_run.aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-fig_run.aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n, fig_run.EPS_STATION_RUN, dtype=float)
    direct_noise = np.full(n, fig_run.EPS_STATION_RUN + fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    hour_angles = realnight_hour_angles(fig_run.aug.N_TIME_WINDOWS, fig_run.aug.EXPOSURE_S, fig_run.aug.EXPOSURE_GAP_S)

    lam_edges_nm = np.arange(
        fig_run.aug.LAMBDA_MIN_NM,
        fig_run.aug.LAMBDA_MAX_NM + 0.5 * fig_run.aug.LAMBDA_STEP_NM,
        fig_run.aug.LAMBDA_STEP_NM,
    )
    lam_edges_nm[-1] = fig_run.aug.LAMBDA_MAX_NM

    rows: list[dict[str, float | str]] = []
    with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.GOOD_SOURCE):
        for band_idx, (lo_nm, hi_nm) in enumerate(zip(lam_edges_nm[:-1], lam_edges_nm[1:])):
            center_nm = float(math.sqrt(lo_nm * hi_nm))
            lam_m = center_nm * 1e-9
            freq = base.C_LIGHT / lam_m
            freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
            total_modes = fig_run.aug.EXPOSURE_S * fig_run.OBSERVING_DAYS * (freq_hi - freq_lo)
            u_station = fig_run.aug.station_u_modes(freq, diameters)
            band_truth, _axis = base.make_source_at_wavelength_nm(fig_run.aug.N_PIX, fig_run.aug.HALF_WIDTH_UAS, center_nm)
            band_vgrid, band_uv_axis = base.visibility_grid(band_truth, fov_rad)
            uu_rows, vv_rows = project_enu_baselines(
                baselines,
                hour_angles,
                lam_m,
                latitude_deg=case.latitude_deg,
                declination_deg=fig_run.GOOD_SOURCE.dec_deg,
            )

            fishers = {
                loop_label: {strategy: 0.0 for strategy, _label, _color in LOOP_STRATEGIES}
                for _tri, loop_label, _kind in LOOPS
            }
            near_cycle_fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
            direct_raw_cycle_fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
            direct_schedule_cycle_fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
            for uu, vv in zip(uu_rows, vv_rows):
                vtrue = base.interp_vis(band_vgrid, band_uv_axis, uu, vv)
                nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
                near_cycle_fisher += split_sim.core4_remote_loop_fisher_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    direct_noise=direct_noise,
                    nu_eff=nu_eff,
                    q_basis=q_basis,
                    edges=edges,
                )
                direct_raw_cycle_fisher += total_modes * fig_run.aug.noisy_closure_fisher_station_u(
                    vtrue,
                    eta,
                    direct_noise,
                    u_station,
                    q_basis,
                    edges,
                )
                direct_schedule_cycle_fisher += split_sim.direct_root_weighted_fisher_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    direct_noise=direct_noise,
                    q_basis=q_basis,
                    edges=edges,
                )
                for tri, loop_label, _band_label in LOOPS:
                    fishers[loop_label]["edge_uniform"] += uniform_edge_scalar_loop_fisher_for_sample(
                        splits["edge_uniform"],
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )
                    fishers[loop_label]["edge_nonsplitting"] += nonsplitting_edge_scalar_loop_fisher_for_sample(
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )

            for tri, loop_label, band_label in LOOPS:
                fishers[loop_label]["core4_remote_optimized"] = scalar_fisher_from_cycle_matrix(
                    near_cycle_fisher,
                    q_basis,
                    edges,
                    tri,
                )
                fishers[loop_label]["direct_local_raw"] = scalar_fisher_from_cycle_matrix(
                    direct_raw_cycle_fisher,
                    q_basis,
                    edges,
                    tri,
                )
                scheduled_fisher = scalar_fisher_from_cycle_matrix(
                    direct_schedule_cycle_fisher,
                    q_basis,
                    edges,
                    tri,
                )
                for strategy, label, _color in LOOP_STRATEGIES:
                    fisher = float(fishers[loop_label][strategy])
                    rows.append(
                        {
                            "loop": loop_label,
                            "loop_class": band_label,
                            "strategy": strategy,
                            "label": label,
                            "band_index": band_idx,
                            "lambda_lo_nm": float(lo_nm),
                            "lambda_hi_nm": float(hi_nm),
                            "lambda_center_nm": center_nm,
                            "rms_rad": 1.0 / math.sqrt(max(fisher, 1e-300)),
                            "scalar_fisher": fisher,
                            "scheduled_direct_proxy_rms_rad": 1.0 / math.sqrt(max(scheduled_fisher, 1e-300)),
                            "capacity_relaxed_weight_per_closure": float(rank_share),
                            "effective_exposure_s": float(fig_run.aug.EXPOSURE_S),
                            "observing_days": float(fig_run.OBSERVING_DAYS),
                        }
                    )
    return rows


def single_sample_loop_rms() -> list[dict[str, float | str]]:
    """Closure-phase RMS for one exposure sample, without summing Fisher over time rows."""
    fig_run.configure_good_runtime()
    case = fig_run.scale_remote_coordinates(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    splits = fig_run.make_split_matrices(case)
    split_sim.configure()
    fig_run.apply_sample_stress_runtime()

    stations, diameters, _names, _is_added = fig_run.aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n = len(stations)
    edges = base.edge_list(n)
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, n))
    rank_share = min(1.0, (n - 1.0) / q_basis.shape[1])
    fov_rad = 2.0 * fig_run.aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    effective_hub_dist = fig_run.aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-fig_run.aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n, fig_run.EPS_STATION_RUN, dtype=float)
    direct_noise = np.full(n, fig_run.EPS_STATION_RUN + fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    hour_angles = realnight_hour_angles(fig_run.aug.N_TIME_WINDOWS, fig_run.aug.EXPOSURE_S, fig_run.aug.EXPOSURE_GAP_S)

    lam_edges_nm = np.arange(
        fig_run.aug.LAMBDA_MIN_NM,
        fig_run.aug.LAMBDA_MAX_NM + 0.5 * fig_run.aug.LAMBDA_STEP_NM,
        fig_run.aug.LAMBDA_STEP_NM,
    )
    lam_edges_nm[-1] = fig_run.aug.LAMBDA_MAX_NM

    rows: list[dict[str, float | str]] = []
    with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.GOOD_SOURCE):
        for band_idx, (lo_nm, hi_nm) in enumerate(zip(lam_edges_nm[:-1], lam_edges_nm[1:])):
            center_nm = float(math.sqrt(lo_nm * hi_nm))
            lam_m = center_nm * 1e-9
            freq = base.C_LIGHT / lam_m
            freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
            total_modes = fig_run.aug.EXPOSURE_S * fig_run.OBSERVING_DAYS * (freq_hi - freq_lo)
            u_station = fig_run.aug.station_u_modes(freq, diameters)
            band_truth, _axis = base.make_source_at_wavelength_nm(fig_run.aug.N_PIX, fig_run.aug.HALF_WIDTH_UAS, center_nm)
            band_vgrid, band_uv_axis = base.visibility_grid(band_truth, fov_rad)
            uu_rows, vv_rows = project_enu_baselines(
                baselines,
                hour_angles,
                lam_m,
                latitude_deg=case.latitude_deg,
                declination_deg=fig_run.GOOD_SOURCE.dec_deg,
            )
            for time_idx, (hour_angle_h, uu, vv) in enumerate(zip(hour_angles, uu_rows, vv_rows)):
                vtrue = base.interp_vis(band_vgrid, band_uv_axis, uu, vv)
                nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
                near_cycle_fisher = split_sim.core4_remote_loop_fisher_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    direct_noise=direct_noise,
                    nu_eff=nu_eff,
                    q_basis=q_basis,
                    edges=edges,
                )
                direct_raw_cycle_fisher = total_modes * fig_run.aug.noisy_closure_fisher_station_u(
                    vtrue,
                    eta,
                    direct_noise,
                    u_station,
                    q_basis,
                    edges,
                )
                direct_schedule_cycle_fisher = split_sim.direct_root_weighted_fisher_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    direct_noise=direct_noise,
                    q_basis=q_basis,
                    edges=edges,
                )
                for tri, loop_label, band_label in LOOPS:
                    f_edge = uniform_edge_scalar_loop_fisher_for_sample(
                        splits["edge_uniform"],
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )
                    f_edge_nonsplitting = nonsplitting_edge_scalar_loop_fisher_for_sample(
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )
                    f_near = scalar_fisher_from_cycle_matrix(
                        near_cycle_fisher,
                        q_basis,
                        edges,
                        tri,
                    )
                    f_direct = scalar_fisher_from_cycle_matrix(
                        direct_raw_cycle_fisher,
                        q_basis,
                        edges,
                        tri,
                    )
                    f_scheduled = scalar_fisher_from_cycle_matrix(
                        direct_schedule_cycle_fisher,
                        q_basis,
                        edges,
                        tri,
                    )
                    for strategy, label, fisher in (
                        ("edge_uniform", "edge-first", f_edge),
                        ("edge_nonsplitting", "non-splitting edge-first", f_edge_nonsplitting),
                        ("core4_remote_optimized", "near-opt", f_near),
                        ("scheduled_direct_proxy", "cap. direct", f_scheduled),
                        ("direct_local_raw", "true direct", f_direct),
                    ):
                        rms_rad = 1.0 / math.sqrt(max(float(fisher), 1e-300))
                        rows.append(
                            {
                                "loop": loop_label,
                                "loop_class": band_label,
                                "strategy": strategy,
                                "label": label,
                                "band_index": band_idx,
                                "lambda_lo_nm": float(lo_nm),
                                "lambda_hi_nm": float(hi_nm),
                                "lambda_center_nm": center_nm,
                                "time_index": int(time_idx),
                                "hour_angle_h": float(hour_angle_h),
                                "rms_rad": rms_rad,
                                "rms_deg": rms_rad * 180.0 / math.pi,
                                "scalar_fisher": float(fisher),
                                "capacity_relaxed_weight_per_closure": float(rank_share),
                                "effective_exposure_s": float(fig_run.aug.EXPOSURE_S),
                                "observing_days": float(fig_run.OBSERVING_DAYS),
                            }
                        )
    return rows


def summarize_single_sample_loop_rms(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for _tri, loop, _kind in LOOPS:
        for strategy in (
            "edge_uniform",
            "edge_nonsplitting",
            "core4_remote_optimized",
            "scheduled_direct_proxy",
            "direct_local_raw",
        ):
            vals = np.asarray(
                [float(row["rms_deg"]) for row in rows if row["loop"] == loop and row["strategy"] == strategy],
                dtype=float,
            )
            ha_vals = np.asarray(
                [
                    float(row["rms_deg"])
                    for row in rows
                    if row["loop"] == loop and row["strategy"] == strategy and int(row["band_index"]) == 5
                ],
                dtype=float,
            )
            out.append(
                {
                    "loop": loop,
                    "strategy": strategy,
                    "all_bins_time_p10_deg": float(np.percentile(vals, 10.0)),
                    "all_bins_time_median_deg": float(np.median(vals)),
                    "all_bins_time_p90_deg": float(np.percentile(vals, 90.0)),
                    "ha_bin_650_660_p10_deg": float(np.percentile(ha_vals, 10.0)),
                    "ha_bin_650_660_median_deg": float(np.median(ha_vals)),
                    "ha_bin_650_660_p90_deg": float(np.percentile(ha_vals, 90.0)),
                }
            )
    return out


def plot_loop_rms(rows: list[dict[str, float | str]]) -> tuple[Path, Path]:
    by_loop = {loop: [row for row in rows if row["loop"] == loop] for _tri, loop, _kind in LOOPS}
    fig, ax = plt.subplots(figsize=(5.7, 4.25), constrained_layout=True)
    markers = {
        "edge_uniform": "o",
        "edge_nonsplitting": "D",
        "core4_remote_optimized": "s",
        "direct_local_raw": "^",
    }
    loop_span = 0.72
    lam_min = float(fig_run.aug.LAMBDA_MIN_NM)
    lam_max = float(fig_run.aug.LAMBDA_MAX_NM)
    lam_mid = 0.5 * (lam_min + lam_max)
    lam_width = max(lam_max - lam_min, 1.0)
    for loop_idx, (_tri, loop, _kind) in enumerate(LOOPS):
        loop_rows = by_loop[loop]
        for strategy, label, color in LOOP_STRATEGIES:
            vals = [row for row in loop_rows if row["strategy"] == strategy]
            vals.sort(key=lambda row: float(row["lambda_center_nm"]))
            lam = np.asarray([float(row["lambda_center_nm"]) for row in vals])
            rms = np.asarray([float(row["rms_rad"]) for row in vals])
            x_local = loop_idx + loop_span * (lam - lam_mid) / lam_width
            ax.plot(
                x_local,
                rms,
                lw=1.05,
                color=color,
                alpha=0.78,
                label=label if loop_idx == 0 else None,
            )
            ax.scatter(
                x_local,
                rms,
                s=12,
                marker=markers[strategy],
                color=color,
                edgecolor="white",
                linewidth=0.22,
            )
        ax.text(loop_idx - loop_span / 2.0, -0.060, f"{lam_min:g}", ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
        ax.text(loop_idx + loop_span / 2.0, -0.060, f"{lam_max:g}", ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
    for xpos in (0.5, 1.5):
        ax.axvline(xpos, color="0.80", lw=0.8)
    ax.set_xticks(
        np.arange(len(LOOPS), dtype=float),
        [f"{loop}\n{kind} freq." for _tri, loop, kind in LOOPS],
    )
    ax.set_xlim(-0.52, len(LOOPS) - 0.48)
    ax.set_yscale("log")
    ax.set_ylabel("closure phase RMS (rad)")
    ax.set_xlabel("closure loop; small labels mark wavelength endpoints (nm)")
    ax.grid(True, which="both", axis="y", color="0.88", linewidth=0.7)
    ax.grid(True, which="major", axis="x", color="0.92", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    png = FIG_DIR / f"fig2_loop_123_125_127_per_wavelength_rms{OUTPUT_SUFFIX}.png"
    pdf = FIG_DIR / f"fig2_loop_123_125_127_per_wavelength_rms{OUTPUT_SUFFIX}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def loop_ratio_diagnostics(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for _tri, loop, _kind in LOOPS:
        loop_rows = [row for row in rows if row["loop"] == loop]
        by_strategy = {
            strategy: {
                float(row["lambda_center_nm"]): row
                for row in loop_rows
                if row["strategy"] == strategy
            }
            for strategy, _label, _color in LOOP_STRATEGIES
        }
        lambdas = sorted(set.intersection(*(set(values) for values in by_strategy.values())))
        edge = np.asarray([float(by_strategy["edge_uniform"][lam]["rms_rad"]) for lam in lambdas], dtype=float)
        nonsplitting = np.asarray(
            [float(by_strategy["edge_nonsplitting"][lam]["rms_rad"]) for lam in lambdas],
            dtype=float,
        )
        near = np.asarray(
            [float(by_strategy["core4_remote_optimized"][lam]["rms_rad"]) for lam in lambdas],
            dtype=float,
        )
        direct = np.asarray([float(by_strategy["direct_local_raw"][lam]["rms_rad"]) for lam in lambdas], dtype=float)
        scheduled = np.asarray(
            [float(by_strategy["direct_local_raw"][lam]["scheduled_direct_proxy_rms_rad"]) for lam in lambdas],
            dtype=float,
        )
        out.append(
            {
                "loop": loop,
                "near_over_direct_rms_mean": float(np.mean(near / direct)),
                "near_over_direct_rms_std": float(np.std(near / direct, ddof=1)),
                "edge_over_direct_rms_mean": float(np.mean(edge / direct)),
                "edge_over_direct_rms_std": float(np.std(edge / direct, ddof=1)),
                "nonsplitting_edge_over_direct_rms_mean": float(np.mean(nonsplitting / direct)),
                "nonsplitting_edge_over_direct_rms_std": float(np.std(nonsplitting / direct, ddof=1)),
                "edge_over_nonsplitting_edge_rms_mean": float(np.mean(edge / nonsplitting)),
                "edge_over_nonsplitting_edge_rms_std": float(np.std(edge / nonsplitting, ddof=1)),
                "scheduled_proxy_over_direct_rms_mean": float(np.mean(scheduled / direct)),
                "scheduled_proxy_over_direct_rms_std": float(np.std(scheduled / direct, ddof=1)),
                "near_over_scheduled_rms_mean": float(np.mean(near / scheduled)),
                "near_over_scheduled_rms_std": float(np.std(near / scheduled, ddof=1)),
                "edge_over_scheduled_rms_mean": float(np.mean(edge / scheduled)),
                "edge_over_scheduled_rms_std": float(np.std(edge / scheduled, ddof=1)),
                "nonsplitting_edge_over_scheduled_rms_mean": float(np.mean(nonsplitting / scheduled)),
                "nonsplitting_edge_over_scheduled_rms_std": float(np.std(nonsplitting / scheduled, ddof=1)),
                "direct_over_scheduled_rms_mean": float(np.mean(direct / scheduled)),
                "direct_over_scheduled_rms_std": float(np.std(direct / scheduled, ddof=1)),
                "direct_over_near_fisher_mean": float(np.mean((near / direct) ** 2)),
                "direct_over_edge_fisher_mean": float(np.mean((edge / direct) ** 2)),
                "direct_over_nonsplitting_edge_fisher_mean": float(np.mean((nonsplitting / direct) ** 2)),
            }
        )
    return out


def main() -> None:
    force = "--force" in sys.argv
    seed_rows = collect_seed_metrics(force=force)
    seed_csv = OUT / f"fig2_current_5seed_correspondence_metrics{OUTPUT_SUFFIX}.csv"
    seed_json = OUT / f"fig2_current_5seed_correspondence_metrics{OUTPUT_SUFFIX}.json"
    write_csv(seed_csv, seed_rows)
    seed_json.write_text(json.dumps(seed_rows, indent=2) + "\n")
    seed_pdf, seed_png = plot_seed_correspondence(seed_rows)

    loop_rows = per_wavelength_loop_rms()
    loop_csv = OUT / f"fig2_loop_123_125_127_per_wavelength_rms{OUTPUT_SUFFIX}.csv"
    loop_json = OUT / f"fig2_loop_123_125_127_per_wavelength_rms{OUTPUT_SUFFIX}.json"
    write_csv(loop_csv, loop_rows)
    loop_json.write_text(json.dumps(loop_rows, indent=2) + "\n")
    loop_pdf, loop_png = plot_loop_rms(loop_rows)
    ratio_rows = loop_ratio_diagnostics(loop_rows)
    ratio_csv = OUT / f"fig2_loop_123_125_127_compact_ratio_diagnosis{OUTPUT_SUFFIX}.csv"
    ratio_json = OUT / f"fig2_loop_123_125_127_compact_ratio_diagnosis{OUTPUT_SUFFIX}.json"
    write_csv(ratio_csv, ratio_rows)
    ratio_json.write_text(json.dumps(ratio_rows, indent=2) + "\n")
    compact_pdf, compact_png = plot_compact_seed_loop_diagnostic(seed_rows, loop_rows)

    single_rows = single_sample_loop_rms()
    single_csv = OUT / f"fig2_single_sample_loop_closure_phase_rms{OUTPUT_SUFFIX}.csv"
    single_json = OUT / f"fig2_single_sample_loop_closure_phase_rms{OUTPUT_SUFFIX}.json"
    single_summary = OUT / f"fig2_single_sample_loop_closure_phase_rms_summary{OUTPUT_SUFFIX}.json"
    write_csv(single_csv, single_rows)
    single_json.write_text(json.dumps(single_rows, indent=2) + "\n")
    single_summary_rows = summarize_single_sample_loop_rms(single_rows)
    single_summary.write_text(json.dumps(single_summary_rows, indent=2) + "\n")

    payload = {
        "seed_metrics_csv": str(seed_csv),
        "seed_metrics_json": str(seed_json),
        "seed_bar_pdf": str(seed_pdf),
        "seed_bar_png": str(seed_png),
        "loop_rms_csv": str(loop_csv),
        "loop_rms_json": str(loop_json),
        "loop_rms_pdf": str(loop_pdf),
        "loop_rms_png": str(loop_png),
        "loop_ratio_csv": str(ratio_csv),
        "loop_ratio_json": str(ratio_json),
        "single_sample_loop_rms_csv": str(single_csv),
        "single_sample_loop_rms_json": str(single_json),
        "single_sample_loop_rms_summary_json": str(single_summary),
        "compact_diagnostic_pdf": str(compact_pdf),
        "compact_diagnostic_png": str(compact_png),
        "seeds": SEEDS,
        "resource_model": {
            "observing_days": int(fig_run.OBSERVING_DAYS),
            "samples_per_night": int(fig_run.N_TIME_WINDOWS_RUN),
            "exposure_s": float(fig_run.FIG2_EXPOSURE_S),
            "sample_cadence_s": float(fig_run.SAMPLE_CADENCE_S_RUN),
            "fourier_coverage_h": float(fig_run.FOURIER_COVERAGE_H_RUN),
            "eps_station": float(CURRENT_ENV["EPS_STATION"]),
            "eps_pair": float(CURRENT_ENV["EPS_PAIR"]),
            "eps_direct_extra": float(CURRENT_ENV["EPS_DIRECT_EXTRA"]),
            "post_average_drift_std_rad": float(CURRENT_ENV["FIG2_POST_AVERAGE_DRIFT_STD"]),
            "wavelength_bins": (
                f"{int(round((fig_run.LAMBDA_MAX_NM_RUN - fig_run.LAMBDA_MIN_NM_RUN) / fig_run.LAMBDA_STEP_NM_RUN))} "
                f"RML bins from {fig_run.LAMBDA_MIN_NM_RUN:g} to {fig_run.LAMBDA_MAX_NM_RUN:g} nm"
            ),
            "per_band_rml": bool(fig_run.PER_BAND_RML),
            "non_splitting_edge_first": (
                "diagnostic no-fanout edge-first reference with f=1 on the three displayed loop edges; "
                "it isolates station fan-out and is not a simultaneous all-baseline instrument"
            ),
            "phase_chi2_selection": str(CURRENT_ENV["RML_PHASE_CHI2_SELECTION"]),
            "display_log_vmin": float(fig_run.DISPLAY_LOG_VMIN),
            "edge_first": "uniform all-array edge-first split",
            "near_optimal": "strict V1 close4 phase-frame direct+nuisance plus remote loop-wise edge-first readout with optimized schedule weights",
            "optimal_direct": "capacity-relaxed scalar root-closure direct schedule for imaging; raw full-array QFI is retained for single-loop upper-bound diagnostics",
            "scheduled_direct_proxy": "explicit capacity-relaxed scalar schedule with uniform w_l=(N-1)/C used as the Fig. 3(b) normalization baseline",
            "near_optimal_benchmark_note": (
                "The near-optimal implementation is compared with the capacity-relaxed scalar "
                "root-closure direct schedule; the single-loop raw-QFI diagnostic remains a stricter upper bound."
            ),
        },
    }
    summary_path = OUT / f"fig2_current_diagnostic_figures_summary{OUTPUT_SUFFIX}.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
