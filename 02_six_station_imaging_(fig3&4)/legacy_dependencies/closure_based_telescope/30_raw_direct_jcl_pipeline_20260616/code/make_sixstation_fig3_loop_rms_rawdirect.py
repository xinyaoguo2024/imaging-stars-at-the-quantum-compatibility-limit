from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
WORKSPACE = THIS_DIR.parents[1]
AUDIT_CODE = WORKSPACE / "07_codex_scientific_audit_20260610" / "code"
for path in (
    THIS_DIR,
    AUDIT_CODE,
    WORKSPACE / "03_figure_generation_code" / "0608_core_modules",
    WORKSPACE / "03_figure_generation_code" / "0608_all_python_snapshot",
):
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

import core4_joint_remote_split_design as core4_remote  # noqa: E402
import hawaii3_compact_case  # noqa: E402
import make_fig2_fig3_diagnostics_corrected as diag  # noqa: E402
import plot_prl_broadband_clean as base  # noqa: E402
import run_broad_plume_split_objective_rml_sixstation as fig_run  # noqa: E402
import run_gain_objective_variants_corealpha as variants  # noqa: E402
import test_fig3_split_objective_imaging as split_sim  # noqa: E402
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles  # noqa: E402


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
SOURCE18 = WORKSPACE / "18_balanced_10loop_independent_set_20260611"

LOOPS = [
    ((0, 1, 2), "123", "core"),
    ((0, 1, 3), "124", "one remote"),
    ((0, 1, 4), "125", "one remote"),
    ((0, 2, 3), "134", "one remote"),
    ((0, 2, 5), "136", "one remote"),
    ((1, 3, 4), "245", "two remote"),
    ((1, 4, 5), "256", "two remote"),
    ((2, 3, 5), "346", "two remote"),
    ((2, 4, 5), "356", "two remote"),
    ((3, 4, 5), "456", "two remote"),
]
STRATEGIES = [
    ("edge_uniform", "edge-first", "#0077b6", "o"),
    ("paircombine_strict_near", "strict pair-combine near", "#2a9d8f", "s"),
    ("direct_optimized", "direct optimized", "#9d0208", "^"),
]
PLOT_LOOP_LABELS = {"123", "125", "456"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_loop_label(tri: tuple[int, int, int]) -> str:
    return "-".join(f"S{i + 1}" for i in tri)


def load_triangle_weights(path: Path, section: str, key: str) -> dict[tuple[int, int, int], float]:
    data = json.loads(path.read_text())
    weights = data["summary"][section][key] if "summary" in data else data[section][key]
    out: dict[tuple[int, int, int], float] = {}
    for label, value in weights.items():
        tri = tuple(int(part[1:]) - 1 for part in label.split("-"))
        out[tri] = float(value)
    return out


def latest_direct_optimized_weights() -> dict[tuple[int, int, int], float]:
    return load_triangle_weights(
        SOURCE18 / "results" / "remote_star_joint_near_summary.json",
        "direct_optimized_schedule_info",
        "all_triangle_weights",
    )


def scalar_from_cycle(fisher: np.ndarray, q_basis: np.ndarray, edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> float:
    d = q_basis.T @ split_sim.closure_edge_vector(edges, tri)
    cov = np.linalg.pinv(0.5 * (fisher + fisher.T), rcond=1e-12)
    var = float(d @ cov @ d)
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def cycle_from_edge_split(
    split: np.ndarray,
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    edge_fisher = split_sim.edge_fisher_for_sample(
        split,
        total_modes=total_modes,
        u_station=u_station,
        eta=eta,
        station_noise=station_noise,
        nu_eff=nu_eff,
        edges=edges,
    )
    n_station = max(max(edge) for edge in edges) + 1
    fisher = base.closure_fisher_after_gauge_marginalization(
        np.diag(edge_fisher),
        q_basis,
        edges,
        n_station,
    )
    return 0.5 * (fisher + fisher.T)


def core_direct_edge_fisher_for_sample_alpha(
    alpha_core: np.ndarray,
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
) -> np.ndarray:
    """Embed one sample of the close-four joint receiver with per-station alpha_i."""
    local_edges = base.edge_list(len(split_sim.CORE_STATIONS))
    local_vis = np.asarray([vtrue[edge_to_index[edge]] for edge in local_edges], dtype=complex)
    subset = list(split_sim.CORE_STATIONS)
    alpha_core = np.asarray(alpha_core, dtype=float).reshape(len(subset))
    eta_core = alpha_core * eta[subset]
    station_part = np.maximum(direct_noise[subset] - split_sim.EPS_DIRECT_EXTRA, 0.0)
    noise_core = alpha_core * station_part + split_sim.EPS_DIRECT_EXTRA
    local_edge_fisher = total_modes * split_sim.raw_edge_phase_fisher_station_u(
        local_vis,
        eta_core,
        noise_core,
        u_station[subset],
        local_edges,
    )
    out = np.zeros((len(edges), len(edges)), dtype=float)
    for local_i, edge_i in enumerate(local_edges):
        global_i = edge_to_index[edge_i]
        for local_j, edge_j in enumerate(local_edges):
            global_j = edge_to_index[edge_j]
            out[global_i, global_j] += local_edge_fisher[local_i, local_j]
    return 0.5 * (out + out.T)


def core4_remote_global_split_fisher_for_sample_alpha(
    split: np.ndarray,
    alpha_core: np.ndarray,
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    direct_noise: np.ndarray,
    nu_eff: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    edge_fisher = core_direct_edge_fisher_for_sample_alpha(
        alpha_core,
        total_modes=total_modes,
        vtrue=vtrue,
        u_station=u_station,
        eta=eta,
        direct_noise=direct_noise,
        edges=edges,
        edge_to_index=edge_to_index,
    )
    for idx, (i, j) in enumerate(edges):
        if i in split_sim.CORE_STATIONS and j in split_sim.CORE_STATIONS:
            continue
        f_edge = split_sim.edge_pair_fisher_for_sample(
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
        edge_fisher[idx, idx] += f_edge
    n_station = max(max(edge) for edge in edges) + 1
    fisher = base.closure_fisher_after_gauge_marginalization(edge_fisher, q_basis, edges, n_station)
    return 0.5 * (fisher + fisher.T)


def physical_direct_triangle_weights(n_station: int) -> dict[tuple[int, int, int], float]:
    per_triangle = 1.0 / math.comb(n_station - 1, 2)
    return {tuple(tri): float(per_triangle) for tri in itertools.combinations(range(n_station), 3)}


def physical_direct_triangle_fisher_for_sample(
    weights: dict[tuple[int, int, int], float],
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
    for tri, weight in weights.items():
        scalar = split_sim.core_triangle_direct_fisher_for_sample(
            tri,
            total_modes=total_modes,
            vtrue=vtrue,
            u_station=u_station,
            eta=eta,
            direct_noise=direct_noise,
            edges=edges,
            edge_to_index=edge_to_index,
        )
        d = q_basis.T @ split_sim.closure_edge_vector(edges, tri)
        fisher += float(weight) * scalar * np.outer(d, d)
    return 0.5 * (fisher + fisher.T)


def compute_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    fig_run.configure_good_runtime()
    case = fig_run.make_six_station_case()
    split_sim.configure()
    fig_run.apply_sample_stress_runtime()
    direct_weights = latest_direct_optimized_weights()
    paircombine_profile = fig_run.load_paircombine_profile()
    info = {
        "loop_set": [format_loop_label(tri) for tri, _label, _kind in LOOPS],
        "paper_plot_loops": sorted(PLOT_LOOP_LABELS),
        "exposure_s": float(fig_run.aug.EXPOSURE_S),
        "direct_strategy": "direct optimized all-triangle schedule from folder 18",
        "near_strategy": "strict physical pair-combine near profile from folder 27",
        "direct_weights_source": str(SOURCE18 / "results" / "remote_star_joint_near_summary.json"),
        "near_profile_source": str(paircombine_profile["source"]),
        "near_profile_score": float(paircombine_profile["score"]),
        "near_profile_best_tag": str(paircombine_profile["best_tag"]),
    }

    stations, diameters, _names, _is_added = fig_run.aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n = len(stations)
    edges = base.edge_list(n)
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, n))
    edge_uniform_split = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_uniform_split[i, j] = 1.0 / (n - 1.0)
    fov_rad = 2.0 * fig_run.aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    eta = 10.0 ** (-fig_run.aug.FIBER_LOSS_DB_PER_KM * np.linalg.norm(stations - hub, axis=1) / 10.0)
    station_noise = np.full(n, fig_run.EPS_STATION_RUN, dtype=float)
    direct_noise = np.full(n, fig_run.EPS_STATION_RUN + fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    hour_angles = realnight_hour_angles(fig_run.aug.N_TIME_WINDOWS, fig_run.aug.EXPOSURE_S, fig_run.aug.EXPOSURE_GAP_S)
    lam_edges_nm = np.arange(
        fig_run.aug.LAMBDA_MIN_NM,
        fig_run.aug.LAMBDA_MAX_NM + 0.5 * fig_run.aug.LAMBDA_STEP_NM,
        fig_run.aug.LAMBDA_STEP_NM,
    )
    lam_edges_nm[-1] = fig_run.aug.LAMBDA_MAX_NM

    rows: list[dict[str, object]] = []
    with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.GOOD_SOURCE):
        for band_idx, (lo_nm, hi_nm) in enumerate(zip(lam_edges_nm[:-1], lam_edges_nm[1:])):
            center_nm = float(math.sqrt(lo_nm * hi_nm))
            lam_m = center_nm * 1e-9
            freq = base.C_LIGHT / lam_m
            freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
            total_modes = fig_run.aug.EXPOSURE_S * fig_run.OBSERVING_DAYS * (freq_hi - freq_lo)
            u_station = fig_run.aug.station_u_modes(freq, diameters)
            truth, _axis = base.make_source_at_wavelength_nm(fig_run.aug.N_PIX, fig_run.aug.HALF_WIDTH_UAS, center_nm)
            vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
            uu_rows, vv_rows = project_enu_baselines(
                baselines,
                hour_angles,
                lam_m,
                latitude_deg=case.latitude_deg,
                declination_deg=fig_run.GOOD_SOURCE.dec_deg,
            )
            near_cycle = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
            direct_cycle = np.zeros_like(near_cycle)
            edge_cycle = np.zeros_like(near_cycle)
            fishers = {label: {strategy: 0.0 for strategy, _name, _color, _marker in STRATEGIES} for _tri, label, _kind in LOOPS}
            for uu, vv in zip(uu_rows, vv_rows):
                vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
                nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
                near_cycle += fig_run.paircombine_profile_q_fisher_for_sample(
                    paircombine_profile,
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
                direct_cycle += physical_direct_triangle_fisher_for_sample(
                    direct_weights,
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    direct_noise=direct_noise,
                    q_basis=q_basis,
                    edges=edges,
                )
                edge_cycle += cycle_from_edge_split(
                    edge_uniform_split,
                    total_modes=total_modes,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    nu_eff=nu_eff,
                    q_basis=q_basis,
                    edges=edges,
                )
            for tri, loop_label, kind in LOOPS:
                fishers[loop_label]["edge_uniform"] = scalar_from_cycle(edge_cycle, q_basis, edges, tri)
                fishers[loop_label]["paircombine_strict_near"] = scalar_from_cycle(near_cycle, q_basis, edges, tri)
                fishers[loop_label]["direct_optimized"] = scalar_from_cycle(direct_cycle, q_basis, edges, tri)
                for strategy, strategy_label, _color, _marker in STRATEGIES:
                    fisher = float(fishers[loop_label][strategy])
                    rows.append(
                        {
                "variant": "six_station_latest_balanced10_directopt_paircombine",
                            "loop": loop_label,
                            "loop_class": kind,
                            "strategy": strategy,
                            "label": strategy_label,
                            "band_index": band_idx,
                            "lambda_center_nm": center_nm,
                            "rms_rad": 1.0 / math.sqrt(max(fisher, 1e-300)),
                            "scalar_fisher": fisher,
                        }
                    )
    return rows, info


def ratio_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for _tri, loop, _kind in LOOPS:
        loop_rows = [row for row in rows if row["loop"] == loop]
        by_strategy = {
            strategy: {
                float(row["lambda_center_nm"]): float(row["rms_rad"])
                for row in loop_rows
                if row["strategy"] == strategy
            }
            for strategy, _name, _color, _marker in STRATEGIES
        }
        lambdas = sorted(set.intersection(*(set(values) for values in by_strategy.values())))
        direct = np.asarray([by_strategy["direct_optimized"][lam] for lam in lambdas], dtype=float)
        near = np.asarray([by_strategy["paircombine_strict_near"][lam] for lam in lambdas], dtype=float)
        edge = np.asarray([by_strategy["edge_uniform"][lam] for lam in lambdas], dtype=float)
        out.append(
            {
                "loop": loop,
                "near_over_direct_rms_mean": float(np.mean(near / direct)),
                "edge_over_direct_rms_mean": float(np.mean(edge / direct)),
                "edge_over_near_rms_mean": float(np.mean(edge / near)),
            }
        )
    return out


def plot_rows(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    loop_span = 0.74
    lambdas = [float(row["lambda_center_nm"]) for row in rows]
    lam_min = min(lambdas)
    lam_max = max(lambdas)
    lam_mid = 0.5 * (lam_min + lam_max)
    plot_loops = [(tri, loop, kind) for tri, loop, kind in LOOPS if loop in PLOT_LOOP_LABELS]
    for loop_idx, (_tri, loop, kind) in enumerate(plot_loops):
        loop_rows = [row for row in rows if row["loop"] == loop]
        for strategy, label, color, marker in STRATEGIES:
            vals = [row for row in loop_rows if row["strategy"] == strategy]
            vals.sort(key=lambda row: float(row["lambda_center_nm"]))
            lam = np.asarray([float(row["lambda_center_nm"]) for row in vals])
            rms = np.asarray([float(row["rms_rad"]) for row in vals])
            x = loop_idx + loop_span * (lam - lam_mid) / max(lam_max - lam_min, 1.0)
            ax.plot(x, rms, color=color, lw=1.0, alpha=0.78, label=label if loop_idx == 0 else None)
            ax.scatter(x, rms, color=color, marker=marker, s=12, edgecolor="white", linewidth=0.25)
    for xpos in np.arange(0.5, len(plot_loops) - 0.1, 1.0):
        ax.axvline(xpos, color="0.82", lw=0.7)
    ax.set_xticks(np.arange(len(plot_loops)), [f"{loop}\n{kind}" for _tri, loop, kind in plot_loops])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel("closure phase RMS (rad)")
    ax.set_xlabel("closure loop; wavelength runs 600-700 nm within each group")
    ax.grid(True, axis="y", which="both", color="0.88", lw=0.7)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    exposure_tag = fig_run.seconds_tag(float(fig_run.aug.EXPOSURE_S))
    png = FIGURES / f"latest_balanced10_three_loop_rms_{exposure_tag}.png"
    pdf = FIGURES / f"latest_balanced10_three_loop_rms_{exposure_tag}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return pdf, png


def main() -> None:
    rows, info = compute_rows()
    exposure_tag = fig_run.seconds_tag(float(fig_run.aug.EXPOSURE_S))
    write_csv(RESULTS / "latest_balanced10_loop_rms.csv", rows)
    write_csv(RESULTS / f"latest_balanced10_loop_rms_{exposure_tag}.csv", rows)
    (RESULTS / "latest_balanced10_loop_rms.json").write_text(json.dumps(rows, indent=2) + "\n")
    (RESULTS / f"latest_balanced10_loop_rms_{exposure_tag}.json").write_text(json.dumps(rows, indent=2) + "\n")
    ratios = ratio_rows(rows)
    write_csv(RESULTS / "latest_balanced10_loop_ratios.csv", ratios)
    write_csv(RESULTS / f"latest_balanced10_loop_ratios_{exposure_tag}.csv", ratios)
    (RESULTS / "latest_balanced10_loop_ratios.json").write_text(json.dumps(ratios, indent=2) + "\n")
    (RESULTS / f"latest_balanced10_loop_ratios_{exposure_tag}.json").write_text(json.dumps(ratios, indent=2) + "\n")
    pdf, png = plot_rows(rows)
    print(json.dumps({"summary": info, "ratios": ratios, "pdf": str(pdf), "png": str(png)}, indent=2))


if __name__ == "__main__":
    main()
