from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
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

import compute_from_loop_to_array_physical_results as phys  # noqa: E402
import all_triangle_modular_receiver_design as alltri  # noqa: E402
import core4_joint_remote_split_design as core4_remote  # noqa: E402
import make_all_closure_global_benchmark_note as closure_bm  # noqa: E402
import run_broad_plume_split_objective_rml_corrected as fig_run  # noqa: E402


ROOT = WORKSPACE / "18_balanced_10loop_independent_set_20260611"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MANUSCRIPTS = ROOT / "manuscripts"
NOTES = ROOT / "notes"
for folder in (RESULTS, FIGURES, MANUSCRIPTS, NOTES):
    folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    kind: str
    variance_lambda: float = 0.0
    alpha_min: float = 0.02
    alpha_max: float = 0.80
    seed: int = 20260611


VARIANTS = [
    Variant(
        "near_match_direct",
        "near split matched loop-by-loop to physical direct split",
        "match_direct",
        variance_lambda=0.20,
        seed=20260611,
    ),
]

SIX_CORE = tuple(range(3))
SIX_REMOTE = tuple(range(3, 6))


def configure_six_station_constants() -> None:
    """Use old S2-S7 as a six-station case: three compact core stations plus three remotes."""
    core4_remote.CORE = SIX_CORE
    core4_remote.REMOTE = SIX_REMOTE


def make_six_station_case():
    seven = fig_run.scale_remote_coordinates(fig_run.hawaii3_compact_case.make_hawaii3_compact_remote_case())
    telescopes = list(seven.telescopes[1:])
    return fig_run.aug.NetworkCase(
        key="six_station_oldS2_to_oldS7_compact_remote3",
        title="Six-station subset: old S2-S7 relabeled S1-S6",
        latitude_deg=seven.latitude_deg,
        center_latlon=seven.center_latlon,
        telescopes=telescopes,
        hub_km=seven.hub_km,
        optimization_score=seven.optimization_score,
    )


def configure_six_benchmark() -> closure_bm.AllClosureBenchmark:
    """Build the same six-station benchmark used by the Fig.2/Fig.3 scripts."""
    fig_run.configure_good_runtime()
    configure_six_station_constants()
    case = make_six_station_case()

    old_loader = closure_bm.rml_cases.load_maunakea_plus3_case
    old_configure = closure_bm.configure_physics
    old_bm_source = closure_bm.ngc.NGC4151
    old_core_source = core4_remote.ngc.NGC4151

    def configure_closure_benchmark_runtime() -> None:
        old_configure()
        closure_bm.aug.OBSERVING_DAYS = fig_run.OBSERVING_DAYS
        closure_bm.aug.N_TIME_WINDOWS = fig_run.N_TIME_WINDOWS_RUN
        closure_bm.aug.EXPOSURE_S = fig_run.BASE_EXPOSURE_S * fig_run.EXPOSURE_SCALE
        closure_bm.aug.EXPOSURE_GAP_S = fig_run.EXPOSURE_GAP_S_RUN
        closure_bm.aug.LAMBDA_MIN_NM = fig_run.LAMBDA_MIN_NM_RUN
        closure_bm.aug.LAMBDA_MAX_NM = fig_run.LAMBDA_MAX_NM_RUN
        closure_bm.aug.LAMBDA_STEP_NM = fig_run.LAMBDA_STEP_NM_RUN
        closure_bm.aug.POST_AVERAGE_DRIFT_STD = fig_run.POST_AVERAGE_DRIFT_STD_RUN
        closure_bm.aug.FIBER_LENGTH_SCALE = 1.0
        closure_bm.aug.FIBER_LOSS_DB_PER_KM = 0.20
        closure_bm.wt.OBSERVING_DAYS = fig_run.OBSERVING_DAYS
        closure_bm.wt.SNR_BOOST = 1.0

    closure_bm.rml_cases.load_maunakea_plus3_case = lambda: case
    closure_bm.configure_physics = configure_closure_benchmark_runtime
    closure_bm.ngc.NGC4151 = fig_run.GOOD_SOURCE
    core4_remote.ngc.NGC4151 = fig_run.GOOD_SOURCE
    try:
        with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.GOOD_SOURCE):
            return closure_bm.AllClosureBenchmark()
    finally:
        closure_bm.rml_cases.load_maunakea_plus3_case = old_loader
        closure_bm.configure_physics = old_configure
        closure_bm.ngc.NGC4151 = old_bm_source
        core4_remote.ngc.NGC4151 = old_core_source


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_split_payload(path: Path, variant: Variant, alpha: float, p: np.ndarray, info: dict[str, object]) -> None:
    alpha_core = [float(x) for x in info.get("alpha_core", [alpha] * len(core4_remote.CORE))]
    payload = {
        "variant": variant.name,
        "label": variant.label,
        "alpha": float(np.mean(alpha_core)),
        "alpha_core": alpha_core,
        "alpha_by_station": {
            f"S{station + 1}": float(value)
            for station, value in zip(core4_remote.CORE, alpha_core)
        },
        "split_matrix": np.asarray(p, dtype=float).tolist(),
        "summary": info,
        "physical_model": (
            "Each core station i sends an independently optimized fraction alpha_i to the shared "
            "three-core phase-frame receiver and splits the remaining 1-alpha_i among the three "
            "remote stations. Remote stations split one unit of light among the other five pairwise "
            "edge-first channels. The optimization target is the physical all-triangle direct split, "
            "not the raw N-mode QFI."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def root_sigmas_from_fisher(bm, fisher_q: np.ndarray) -> np.ndarray:
    cov = phys.root_cov_from_q_fisher(bm, fisher_q)
    return np.sqrt(np.maximum(np.diag(cov), 1e-300))


def edge_root_sigmas(bm) -> np.ndarray:
    cov = phys.root_cov_edge_first(bm, bm.uniform_split_matrix())
    return np.sqrt(np.maximum(np.diag(cov), 1e-300))


def physical_direct_triangle_weights(n_station: int) -> tuple[dict[tuple[int, int, int], float], dict[str, object]]:
    """Uniform all-triangle direct split with exact station-side photon budgets."""
    per_triangle = 1.0 / math.comb(n_station - 1, 2)
    weights = {tuple(tri): float(per_triangle) for tri in itertools.combinations(range(n_station), 3)}
    station_sums = {
        f"S{i + 1}": float(sum(weight for tri, weight in weights.items() if i in tri))
        for i in range(n_station)
    }
    return weights, {
        "model": "physical_all_triangle_direct_split_station_budget",
        "description": (
            "Each scalar three-mode direct-closure receiver on triangle (a,b,c) consumes the "
            "same fraction w from stations a,b,c. With w=1/C(N-1,2), each station's total "
            "directed fraction over all triangles is exactly one."
        ),
        "n_triangle_settings": int(len(weights)),
        "per_triangle_weight": float(per_triangle),
        "station_budget_constraint": "for every station i, sum_{tri contains i} w_tri <= 1",
        "station_weight_sums": station_sums,
        "max_station_weight_sum": float(max(station_sums.values())),
        "total_triangle_weight_sum": float(sum(weights.values())),
    }


def physical_direct_alltriangle_fisher(bm) -> tuple[np.ndarray, dict[str, object]]:
    """Build the physical direct target in the same q-basis used by the near split."""
    alltri.bm_lib.EPS_STATION = fig_run.EPS_STATION_RUN
    alltri.bm_lib.EPS_DIRECT_EXTRA = fig_run.EPS_DIRECT_EXTRA_RUN
    weights, info = physical_direct_triangle_weights(bm.n)
    fisher = np.zeros((bm.q_basis.shape[1], bm.q_basis.shape[1]), dtype=float)
    scalar_fishers: dict[str, float] = {}
    for tri, weight in weights.items():
        scalar = alltri.triangle_direct_fisher(bm, tri, (weight, weight, weight))
        alltri.add_scalar_measurement(fisher, bm, tri, scalar)
        scalar_fishers[f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}"] = float(scalar)
    fisher = 0.5 * (fisher + fisher.T)
    info = {
        **info,
        "target": "loop-by-loop matching target for the near split",
        "scalar_fisher_min": float(min(scalar_fishers.values())),
        "scalar_fisher_mean": float(np.mean(list(scalar_fishers.values()))),
        "scalar_fisher_max": float(max(scalar_fishers.values())),
        "scalar_fishers": scalar_fishers,
    }
    return fisher, info


def row_softmax(raw_row: np.ndarray, allowed: np.ndarray, total: float, floor: float) -> np.ndarray:
    out = np.zeros(raw_row.shape[0], dtype=float)
    n_allowed = int(np.sum(allowed))
    if n_allowed == 0 or total <= 0.0:
        return out
    active_floor = min(float(floor), 0.25 * total / max(n_allowed, 1))
    remaining = max(total - n_allowed * active_floor, 0.0)
    values = raw_row[allowed]
    weights = np.exp(values - np.max(values))
    weights /= np.sum(weights)
    out[allowed] = active_floor + remaining * weights
    return out


def alpha_core_array(alpha_core: float | np.ndarray | list[float], bm) -> np.ndarray:
    """Return one joint-receiver fraction for each core station."""
    values = np.asarray(alpha_core, dtype=float)
    if values.ndim == 0:
        return np.full(len(core4_remote.CORE), float(values), dtype=float)
    values = values.reshape(-1)
    if values.size == len(core4_remote.CORE):
        return values.astype(float)
    if values.size == bm.n:
        return values[list(core4_remote.CORE)].astype(float)
    raise ValueError(f"Expected scalar, {len(core4_remote.CORE)} core alphas, or {bm.n} station alphas")


def alpha_by_station(alpha_core: float | np.ndarray | list[float], bm) -> np.ndarray:
    out = np.zeros(bm.n, dtype=float)
    out[list(core4_remote.CORE)] = alpha_core_array(alpha_core, bm)
    return out


def alpha_mean(alpha_core: float | np.ndarray | list[float], bm) -> float:
    return float(np.mean(alpha_core_array(alpha_core, bm)))


def project_split(raw: np.ndarray, alpha_core: float | np.ndarray | list[float], bm) -> np.ndarray:
    station_alpha = alpha_by_station(alpha_core, bm)
    p = np.zeros((bm.n, bm.n), dtype=float)
    for i in core4_remote.CORE:
        allowed = np.zeros(bm.n, dtype=bool)
        allowed[list(core4_remote.REMOTE)] = True
        p[i] = row_softmax(raw[i], allowed, 1.0 - station_alpha[i], core4_remote.SPLIT_FLOOR)
    for i in core4_remote.REMOTE:
        allowed = np.ones(bm.n, dtype=bool)
        allowed[i] = False
        p[i] = row_softmax(raw[i], allowed, 1.0, core4_remote.SPLIT_FLOOR)
    return p


def raw_logits_from_split(p: np.ndarray, alpha_core: float | np.ndarray | list[float], bm) -> np.ndarray:
    station_alpha = alpha_by_station(alpha_core, bm)
    raw = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw, -np.inf)
    for i in core4_remote.CORE:
        allowed = np.zeros(bm.n, dtype=bool)
        allowed[list(core4_remote.REMOTE)] = True
        total = 1.0 - station_alpha[i]
        n_allowed = int(np.sum(allowed))
        active_floor = min(float(core4_remote.SPLIT_FLOOR), 0.25 * total / max(n_allowed, 1))
        remaining = max(total - n_allowed * active_floor, 1e-300)
        weights = np.maximum((p[i, allowed] - active_floor) / remaining, 1e-300)
        raw[i, allowed] = np.log(weights)
    for i in core4_remote.REMOTE:
        allowed = np.ones(bm.n, dtype=bool)
        allowed[i] = False
        total = 1.0
        n_allowed = int(np.sum(allowed))
        active_floor = min(float(core4_remote.SPLIT_FLOOR), 0.25 * total / max(n_allowed, 1))
        remaining = max(total - n_allowed * active_floor, 1e-300)
        weights = np.maximum((p[i, allowed] - active_floor) / remaining, 1e-300)
        raw[i, allowed] = np.log(weights)
    return raw


def alpha_from_raw(raw_alpha: float, variant: Variant) -> float:
    span = variant.alpha_max - variant.alpha_min
    return variant.alpha_min + span / (1.0 + math.exp(-raw_alpha))


def raw_from_alpha(alpha: float, variant: Variant) -> float:
    x = (alpha - variant.alpha_min) / (variant.alpha_max - variant.alpha_min)
    x = min(max(x, 1e-9), 1.0 - 1e-9)
    return math.log(x / (1.0 - x))


def alpha_vector_from_raw(raw_alpha: np.ndarray, variant: Variant) -> np.ndarray:
    raw_alpha = np.asarray(raw_alpha, dtype=float).reshape(len(core4_remote.CORE))
    span = variant.alpha_max - variant.alpha_min
    return variant.alpha_min + span / (1.0 + np.exp(-raw_alpha))


def raw_vector_from_alpha(alpha_core: float | np.ndarray | list[float], variant: Variant, bm) -> np.ndarray:
    values = alpha_core_array(alpha_core, bm)
    x = (values - variant.alpha_min) / (variant.alpha_max - variant.alpha_min)
    x = np.clip(x, 1e-9, 1.0 - 1e-9)
    return np.log(x / (1.0 - x))


def core_direct_edge_fisher_matrix_alpha(bm, alpha_core: float | np.ndarray | list[float]) -> np.ndarray:
    """Exact close-four raw phase Fisher for independent station-side joint fractions."""
    fractions = alpha_core_array(alpha_core, bm)
    core4_remote.bm_lib.EPS_STATION = fig_run.EPS_STATION_RUN
    core4_remote.bm_lib.EPS_DIRECT_EXTRA = fig_run.EPS_DIRECT_EXTRA_RUN
    local_edges = core4_remote.base.edge_list(len(core4_remote.CORE))
    global_edge_index = {edge: idx for idx, edge in enumerate(bm.edges)}
    core_stations = bm.stations[list(core4_remote.CORE)]
    core_baselines = np.asarray([core_stations[j] - core_stations[i] for i, j in local_edges], dtype=float)
    eta = fractions * bm.eta[list(core4_remote.CORE)]
    noise = fractions * core4_remote.bm_lib.EPS_STATION + core4_remote.bm_lib.EPS_DIRECT_EXTRA

    edge_fisher = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
    for lam, freq, total_modes in bm.iter_bands():
        vgrid, uv_axis = bm.visibility_grid_for_wavelength(lam * 1e9)
        u_station = core4_remote.aug.station_u_modes(freq, bm.diameters[list(core4_remote.CORE)])
        uu_rows, vv_rows = core4_remote.project_enu_baselines(
            core_baselines,
            bm.hour_angles,
            lam,
            latitude_deg=bm.case.latitude_deg,
            declination_deg=core4_remote.ngc.NGC4151.dec_deg,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            vlocal = core4_remote.base.interp_vis(vgrid, uv_axis, uu, vv)
            local_edge_fisher = total_modes * core4_remote.raw_edge_phase_fisher_station_u(
                vlocal,
                eta,
                noise,
                u_station,
                local_edges,
            )
            for local_i, edge_i in enumerate(local_edges):
                global_i = global_edge_index[edge_i]
                for local_j, edge_j in enumerate(local_edges):
                    global_j = global_edge_index[edge_j]
                    edge_fisher[global_i, global_j] += local_edge_fisher[local_i, local_j]
    return 0.5 * (edge_fisher + edge_fisher.T)


def precompute_core_unit(bm) -> np.ndarray:
    old_source = core4_remote.ngc.NGC4151
    core4_remote.ngc.NGC4151 = fig_run.GOOD_SOURCE
    try:
        with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.GOOD_SOURCE):
            return core4_remote.core_direct_edge_fisher_matrix(bm, fraction=1.0)
    finally:
        core4_remote.ngc.NGC4151 = old_source


def fisher_for_candidate_with_core(
    bm,
    core_edge_fisher: np.ndarray,
    raw: np.ndarray,
    alpha_core: float | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    p = project_split(raw, alpha_core, bm)
    edge_fisher = core_edge_fisher + core4_remote.remote_edge_fisher_matrix_for_split(bm, p)
    fisher_q = core4_remote.base.closure_fisher_after_gauge_marginalization(
        edge_fisher,
        bm.q_basis,
        bm.edges,
        bm.n,
    )
    return fisher_q, p


def fisher_for_candidate(
    bm,
    raw: np.ndarray,
    alpha_core: float | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    core_edge_fisher = core_direct_edge_fisher_matrix_alpha(bm, alpha_core)
    return fisher_for_candidate_with_core(bm, core_edge_fisher, raw, alpha_core)


def gain_summary(
    labels: list[str],
    triangles: list[tuple[int, int, int]],
    gains: np.ndarray,
    alpha_core: float | np.ndarray | list[float],
    bm,
) -> dict[str, object]:
    log_g = np.log(np.maximum(gains, 1e-300))
    core = np.asarray([all(station in core4_remote.CORE for station in tri) for tri in triangles], dtype=bool)
    two_remote = np.asarray([sum(station in core4_remote.REMOTE for station in tri) == 2 for tri in triangles], dtype=bool)
    remote = ~core
    order = np.argsort(gains)
    alpha_values = alpha_core_array(alpha_core, bm)
    return {
        "alpha": float(np.mean(alpha_values)),
        "alpha_core": [float(x) for x in alpha_values],
        "alpha_core_min": float(np.min(alpha_values)),
        "alpha_core_max": float(np.max(alpha_values)),
        "alpha_by_station": {
            f"S{station + 1}": float(value)
            for station, value in zip(core4_remote.CORE, alpha_values)
        },
        "min_snr_gain": float(np.min(gains)),
        "mean_snr_gain": float(np.mean(gains)),
        "median_snr_gain": float(np.median(gains)),
        "max_snr_gain": float(np.max(gains)),
        "mean_fisher_gain": float(np.mean(gains * gains)),
        "geomean_fisher_gain": float(np.exp(np.mean(2.0 * log_g))),
        "var_log_snr_gain": float(np.var(log_g)),
        "std_log_snr_gain": float(np.std(log_g)),
        "n_below_unity": int(np.sum(gains < 1.0 - 1e-9)),
        "core_mean_snr_gain": float(np.mean(gains[core])),
        "remote_involved_mean_snr_gain": float(np.mean(gains[remote])),
        "two_remote_mean_snr_gain": float(np.mean(gains[two_remote])),
        "worst_loop_gains": {labels[idx]: float(gains[idx]) for idx in order[:5]},
        "best_loop_gains": {labels[idx]: float(gains[idx]) for idx in order[-5:][::-1]},
        "all_loop_snr_gains": {label: float(gain) for label, gain in zip(labels, gains)},
    }


def direct_match_summary(
    labels: list[str],
    triangles: list[tuple[int, int, int]],
    *,
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
    near_sigma: np.ndarray,
    alpha_core: float | np.ndarray | list[float],
    bm,
) -> dict[str, object]:
    near_vs_edge = edge_sigma / np.maximum(near_sigma, 1e-300)
    direct_vs_edge = edge_sigma / np.maximum(direct_sigma, 1e-300)
    near_vs_direct = direct_sigma / np.maximum(near_sigma, 1e-300)
    log_ratio = np.log(np.maximum(near_vs_direct, 1e-300))
    order = np.argsort(np.abs(log_ratio))[::-1]
    remote_count = np.asarray([sum(station in core4_remote.REMOTE for station in tri) for tri in triangles], dtype=int)
    return {
        **gain_summary(labels, triangles, near_vs_edge, alpha_core, bm),
        "direct_target_min_snr_gain_vs_edge": float(np.min(direct_vs_edge)),
        "direct_target_mean_snr_gain_vs_edge": float(np.mean(direct_vs_edge)),
        "direct_target_median_snr_gain_vs_edge": float(np.median(direct_vs_edge)),
        "direct_target_max_snr_gain_vs_edge": float(np.max(direct_vs_edge)),
        "near_over_direct_snr_min": float(np.min(near_vs_direct)),
        "near_over_direct_snr_mean": float(np.mean(near_vs_direct)),
        "near_over_direct_snr_median": float(np.median(near_vs_direct)),
        "near_over_direct_snr_max": float(np.max(near_vs_direct)),
        "near_over_direct_fisher_mean": float(np.mean(near_vs_direct * near_vs_direct)),
        "direct_over_near_rms_mean": float(np.mean(near_sigma / np.maximum(direct_sigma, 1e-300))),
        "rms_log_near_over_direct_snr": float(np.sqrt(np.mean(log_ratio * log_ratio))),
        "mean_log_near_over_direct_snr": float(np.mean(log_ratio)),
        "var_log_near_over_direct_snr": float(np.var(log_ratio)),
        "max_abs_log_near_over_direct_snr": float(np.max(np.abs(log_ratio))),
        "n_near_below_direct": int(np.sum(near_vs_direct < 1.0 - 1e-9)),
        "n_near_below_edge": int(np.sum(near_vs_edge < 1.0 - 1e-9)),
        "all_loop_near_over_direct_snr": {label: float(value) for label, value in zip(labels, near_vs_direct)},
        "all_loop_direct_snr_gains_vs_edge": {label: float(value) for label, value in zip(labels, direct_vs_edge)},
        "all_loop_near_snr_gains_vs_edge": {label: float(value) for label, value in zip(labels, near_vs_edge)},
        "core_only_near_over_direct_mean": float(np.mean(near_vs_direct[remote_count == 0])),
        "one_remote_near_over_direct_mean": float(np.mean(near_vs_direct[remote_count == 1])),
        "two_remote_near_over_direct_mean": float(np.mean(near_vs_direct[remote_count == 2])),
        "worst_match_loops_by_abs_log": {
            labels[idx]: {
                "near_over_direct_snr": float(near_vs_direct[idx]),
                "near_gain_vs_edge": float(near_vs_edge[idx]),
                "direct_gain_vs_edge": float(direct_vs_edge[idx]),
            }
            for idx in order[:5]
        },
    }


def score_varlog(gains: np.ndarray, variance_lambda: float) -> float:
    log_g = np.log(np.maximum(gains, 1e-300))
    min_log = float(np.min(log_g))
    if min_log < 0.0:
        return -math.inf
    return float(np.mean(log_g) - variance_lambda * np.var(log_g))


def score_match_direct(
    *,
    near_sigma: np.ndarray,
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
    variance_lambda: float,
) -> float:
    near_vs_edge = edge_sigma / np.maximum(near_sigma, 1e-300)
    if float(np.min(near_vs_edge)) < 1.0 - 1e-10:
        return -math.inf
    near_vs_direct = direct_sigma / np.maximum(near_sigma, 1e-300)
    log_ratio = np.log(np.maximum(near_vs_direct, 1e-300))
    mse = float(np.mean(log_ratio * log_ratio))
    variance = float(np.var(log_ratio))
    max_abs = float(np.max(np.abs(log_ratio)))
    mean_bias = float(abs(np.mean(log_ratio)))
    min_log = float(np.min(log_ratio))
    max_over = float(max(0.0, np.max(log_ratio)))
    return min_log - 0.20 * max_over * max_over - variance_lambda * variance - 0.05 * mse - 0.02 * mean_bias - 0.02 * max_abs * max_abs


def optimize_varlog(bm, core_unit: np.ndarray, edge_sigma: np.ndarray, variant: Variant) -> tuple[np.ndarray, float, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(variant.seed + int(variant.variance_lambda * 1000))
    triangles = core4_remote.root_independent_triangles(bm.n)
    labels = [f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}" for tri in triangles]
    raw0 = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw0, -np.inf)
    active = [(i, j) for i in core4_remote.CORE for j in core4_remote.REMOTE]
    active += [(i, j) for i in core4_remote.REMOTE for j in range(bm.n) if i != j]

    def evaluate(raw: np.ndarray, raw_alpha: float) -> tuple[float, np.ndarray, float, np.ndarray]:
        alpha = alpha_from_raw(raw_alpha, variant)
        fisher_q, p = fisher_for_candidate(bm, core_unit, raw, alpha)
        gains = edge_sigma / root_sigmas_from_fisher(bm, fisher_q)
        return score_varlog(gains, variant.variance_lambda), gains, alpha, p

    balanced_p, _balanced_info = fig_run.optimize_root_loop_gain_split(bm, seed=20260610)
    best_raw = raw_logits_from_split(balanced_p, 0.5, bm)
    best_raw_alpha = raw_from_alpha(0.5, variant)
    best_score, best_gains, best_alpha, best_p = evaluate(best_raw, best_raw_alpha)
    if not np.isfinite(best_score):
        raise RuntimeError("balanced initial split is unexpectedly infeasible")

    for alpha0 in (0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50):
        for scale in (0.35, 0.75, 1.25, 2.0):
            for _ in range(450):
                candidate = raw0 + rng.normal(scale=scale, size=(bm.n, bm.n))
                np.fill_diagonal(candidate, -np.inf)
                raw_alpha = raw_from_alpha(alpha0, variant) + rng.normal(scale=0.75)
                value, gains, alpha, p = evaluate(candidate, raw_alpha)
                if np.isfinite(value) and value > best_score:
                    best_score = value
                    best_raw = candidate
                    best_raw_alpha = raw_alpha
                    best_gains = gains
                    best_alpha = alpha
                    best_p = p

    for width in (0.90, 0.45, 0.22, 0.10, 0.045, 0.020, 0.009):
        improved = True
        passes = 0
        while improved and passes < 10:
            improved = False
            passes += 1
            for sign in (-1.0, 1.0):
                raw_alpha = best_raw_alpha + sign * width
                value, gains, alpha, p = evaluate(best_raw, raw_alpha)
                if np.isfinite(value) and value > best_score + 1e-12:
                    best_score = value
                    best_raw_alpha = raw_alpha
                    best_gains = gains
                    best_alpha = alpha
                    best_p = p
                    improved = True
            for i, j in active:
                for sign in (-1.0, 1.0):
                    candidate = best_raw.copy()
                    candidate[i, j] += sign * width
                    value, gains, alpha, p = evaluate(candidate, best_raw_alpha)
                    if np.isfinite(value) and value > best_score + 1e-12:
                        best_score = value
                        best_raw = candidate
                        best_gains = gains
                        best_alpha = alpha
                        best_p = p
                        improved = True

    info = gain_summary(labels, triangles, best_gains, best_alpha)
    info.update(
        {
            "objective": "strict_min_gain_var_log",
            "score": float(best_score),
            "variance_lambda": float(variant.variance_lambda),
            "objective_formula": "maximize mean(log G_alpha) - lambda * Var(log G_alpha), with candidates having min(G_alpha)<1 treated as infeasible",
            "alpha_bounds": [float(variant.alpha_min), float(variant.alpha_max)],
        }
    )
    return best_p, best_alpha, best_gains, info


def optimize_match_direct(
    bm,
    edge_sigma: np.ndarray,
    variant: Variant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(variant.seed)
    triangles = core4_remote.root_independent_triangles(bm.n)
    labels = [f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}" for tri in triangles]
    direct_fisher, direct_info = physical_direct_alltriangle_fisher(bm)
    direct_sigma = root_sigmas_from_fisher(bm, direct_fisher)
    raw0 = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw0, -np.inf)
    active = [(i, j) for i in core4_remote.CORE for j in core4_remote.REMOTE]
    active += [(i, j) for i in core4_remote.REMOTE for j in range(bm.n) if i != j]

    core_cache: dict[tuple[float, ...], np.ndarray] = {}
    eval_counts = {"core_recomputes": 0, "cached_core_hits": 0, "split_evaluations": 0}

    def core_for_alpha(alpha_core: np.ndarray) -> np.ndarray:
        key = tuple(float(f"{x:.8f}") for x in alpha_core)
        cached = core_cache.get(key)
        if cached is not None:
            eval_counts["cached_core_hits"] += 1
            return cached
        eval_counts["core_recomputes"] += 1
        core_edge = core_direct_edge_fisher_matrix_alpha(bm, alpha_core)
        core_cache[key] = core_edge
        return core_edge

    def evaluate_with_core(
        raw: np.ndarray,
        alpha_core: np.ndarray,
        core_edge: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        eval_counts["split_evaluations"] += 1
        fisher_q, p = fisher_for_candidate_with_core(bm, core_edge, raw, alpha_core)
        near_sigma = root_sigmas_from_fisher(bm, fisher_q)
        score = score_match_direct(
            near_sigma=near_sigma,
            edge_sigma=edge_sigma,
            direct_sigma=direct_sigma,
            variance_lambda=variant.variance_lambda,
        )
        gains = edge_sigma / np.maximum(near_sigma, 1e-300)
        return score, gains, near_sigma, p

    starts: list[tuple[str, np.ndarray, np.ndarray]] = []
    for alpha0 in (0.06, 0.08, 0.12, 0.20, 0.35):
        starts.append((f"uniform_alpha_{alpha0:g}", raw0.copy(), raw_vector_from_alpha(alpha0, variant, bm)))
    try:
        balanced_p, _balanced_info = fig_run.optimize_root_loop_gain_split(bm, seed=20260610)
        starts.append(("balanced_alpha_0p50", raw_logits_from_split(balanced_p, 0.5, bm), raw_vector_from_alpha(0.5, variant, bm)))
    except Exception:
        pass
    for previous_payload in (
        WORKSPACE / "11_near_match_direct_physical_20260611" / "results" / "near_match_direct_split_payload.json",
        WORKSPACE / "10_varlog_lam2_full_rml_20260610" / "results" / "varlog_lam2_split_payload.json",
    ):
        if not previous_payload.exists():
            continue
        payload = json.loads(previous_payload.read_text())
        previous_p = np.asarray(payload["split_matrix"], dtype=float)
        previous_alpha = np.asarray(payload.get("alpha_core", [float(payload["alpha"])] * len(core4_remote.CORE)), dtype=float)
        if previous_p.shape != (bm.n, bm.n) or previous_alpha.size not in (1, len(core4_remote.CORE), bm.n):
            continue
        starts.append(
            (
                f"previous_{previous_payload.parents[1].name}",
                raw_logits_from_split(previous_p, previous_alpha, bm),
                raw_vector_from_alpha(previous_alpha, variant, bm),
            )
        )

    best_score = -math.inf
    best_gains: np.ndarray | None = None
    best_near_sigma: np.ndarray | None = None
    best_alpha_core: np.ndarray | None = None
    best_p: np.ndarray | None = None
    best_raw: np.ndarray | None = None
    best_raw_alpha: np.ndarray | None = None
    best_start = ""

    def try_candidate(
        raw: np.ndarray,
        raw_alpha: np.ndarray,
        *,
        core_edge: np.ndarray | None = None,
        start_name: str = "",
    ) -> bool:
        nonlocal best_score, best_gains, best_near_sigma, best_alpha_core, best_p, best_raw, best_raw_alpha, best_start
        alpha_core = alpha_vector_from_raw(raw_alpha, variant)
        if core_edge is None:
            core_edge = core_for_alpha(alpha_core)
        value, gains, near_sigma, p = evaluate_with_core(raw, alpha_core, core_edge)
        if np.isfinite(value) and value > best_score:
            best_score = value
            best_gains = gains
            best_near_sigma = near_sigma
            best_alpha_core = alpha_core.copy()
            best_p = p
            best_raw = raw.copy()
            best_raw_alpha = raw_alpha.copy()
            if start_name:
                best_start = start_name
            return True
        return False

    for start_name, raw, raw_alpha in starts:
        try_candidate(raw, raw_alpha, start_name=start_name)

    if best_raw is None:
        raise RuntimeError("No feasible near split found with near/edge >= 1 while initializing match-direct objective")

    alpha_centers = (0.035, 0.05, 0.065, 0.08, 0.10, 0.13, 0.18, 0.26, 0.38)
    for alpha0 in alpha_centers:
        raw_alpha_center = raw_vector_from_alpha(alpha0, variant, bm)
        for _ in range(8):
            raw_alpha = raw_alpha_center + rng.normal(scale=0.85, size=len(core4_remote.CORE))
            alpha_core = alpha_vector_from_raw(raw_alpha, variant)
            core_edge = core_for_alpha(alpha_core)
            for scale in (0.25, 0.55, 1.00, 1.70, 2.60):
                for _ in range(12):
                    candidate = raw0 + rng.normal(scale=scale, size=(bm.n, bm.n))
                    np.fill_diagonal(candidate, -np.inf)
                    try_candidate(
                        candidate,
                        raw_alpha,
                        core_edge=core_edge,
                        start_name=f"random_alpha_{alpha0:g}_scale_{scale:g}",
                    )

    for outer in range(3):
        assert best_raw is not None
        assert best_raw_alpha is not None

        for width in (0.90, 0.45, 0.22, 0.10, 0.045, 0.020):
            improved = True
            passes = 0
            while improved and passes < 5:
                improved = False
                passes += 1
                for alpha_idx in range(len(core4_remote.CORE)):
                    for sign in (-1.0, 1.0):
                        candidate_alpha = best_raw_alpha.copy()
                        candidate_alpha[alpha_idx] += sign * width
                        if try_candidate(best_raw, candidate_alpha, start_name=f"alpha_coord_outer_{outer}"):
                            improved = True

        assert best_alpha_core is not None
        core_edge = core_for_alpha(best_alpha_core)
        for width in (1.00, 0.55, 0.28, 0.13, 0.060, 0.026, 0.011):
            improved = True
            passes = 0
            while improved and passes < 8:
                improved = False
                passes += 1
                assert best_raw is not None
                assert best_raw_alpha is not None
                for i, j in active:
                    for sign in (-1.0, 1.0):
                        candidate = best_raw.copy()
                        candidate[i, j] += sign * width
                        if try_candidate(candidate, best_raw_alpha, core_edge=core_edge, start_name=f"split_coord_outer_{outer}"):
                            improved = True

        assert best_raw is not None
        assert best_raw_alpha is not None
        for scale in (0.045, 0.090, 0.18):
            core_edge = core_for_alpha(alpha_vector_from_raw(best_raw_alpha, variant))
            for _ in range(120):
                candidate = raw0 + rng.normal(scale=scale, size=(bm.n, bm.n))
                np.fill_diagonal(candidate, -np.inf)
                candidate += best_raw
                np.fill_diagonal(candidate, -np.inf)
                try_candidate(candidate, best_raw_alpha, core_edge=core_edge, start_name=f"local_random_outer_{outer}")

    assert best_p is not None
    assert best_gains is not None
    assert best_near_sigma is not None
    assert best_alpha_core is not None
    info = direct_match_summary(
        labels,
        triangles,
        edge_sigma=edge_sigma,
        direct_sigma=direct_sigma,
        near_sigma=best_near_sigma,
        alpha_core=best_alpha_core,
        bm=bm,
    )
    info.update(
        {
            "objective": "match_physical_direct_loop_snr_independent_core_alpha",
            "score": float(best_score),
            "objective_formula": (
                "maximize min(log R_l) - 0.20 max(0,max log R_l)^2 - lambda Var(log R_l) - 0.05 mean[(log R_l)^2] - small bias/spread penalties, "
                "where R_l=(SNR_near/SNR_direct)_l=sigma_direct,l/sigma_near,l; candidates with any near/edge<1 are infeasible"
            ),
            "match_variance_lambda": float(variant.variance_lambda),
            "alpha_bounds": [float(variant.alpha_min), float(variant.alpha_max)],
            "alpha_parameterization": "independent station-side alpha_i for core stations S1-S3",
            "best_start": best_start,
            "optimization_counts": eval_counts,
            "n_cached_core_blocks": int(len(core_cache)),
            "physical_direct_target": direct_info,
        }
    )
    return best_p, best_alpha_core, best_gains, info


def evaluate_fixed_split(bm, edge_sigma: np.ndarray, p: np.ndarray, fisher_q: np.ndarray, alpha: float, objective: str) -> tuple[np.ndarray, dict[str, object]]:
    triangles = core4_remote.root_independent_triangles(bm.n)
    labels = [f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}" for tri in triangles]
    gains = edge_sigma / root_sigmas_from_fisher(bm, fisher_q)
    info = gain_summary(labels, triangles, gains, alpha, bm)
    info.update({"objective": objective})
    return gains, info


def compute_variant(bm, edge_sigma: np.ndarray, variant: Variant) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    old_source = core4_remote.ngc.NGC4151
    old_alltri_source = alltri.ngc.NGC4151
    core4_remote.ngc.NGC4151 = fig_run.GOOD_SOURCE
    alltri.ngc.NGC4151 = fig_run.GOOD_SOURCE
    try:
        with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.GOOD_SOURCE):
            if variant.kind == "legacy_mean_rms":
                p, legacy_info = core4_remote.optimize_split(bm, "mean_rms", seed=20260529)
                fisher_q = core4_remote.fisher_for_split(bm, p)
                gains, info = evaluate_fixed_split(bm, edge_sigma, p, fisher_q, 0.5, "legacy_q_coordinate_mean_rms")
                info["legacy_split_info"] = legacy_info
                return p, 0.5, gains, info
            if variant.kind == "balanced_alpha05":
                p, balanced_info = fig_run.optimize_root_loop_gain_split(bm, seed=20260610)
                fisher_q = core4_remote.fisher_for_split(bm, p)
                gains, info = evaluate_fixed_split(bm, edge_sigma, p, fisher_q, 0.5, "balanced_root_gain_alpha05")
                info["balanced_split_info"] = balanced_info
                return p, 0.5, gains, info
            if variant.kind == "varlog":
                raise NotImplementedError("The independent-core-alpha run only defines the match-direct benchmark variant.")
            if variant.kind == "match_direct":
                return optimize_match_direct(bm, edge_sigma, variant)
    finally:
        core4_remote.ngc.NGC4151 = old_source
        alltri.ngc.NGC4151 = old_alltri_source
    raise ValueError(variant.kind)


def loop_rows(labels: list[str], triangles: list[tuple[int, int, int]], variant: Variant, gains: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for label, tri, gain in zip(labels, triangles, gains):
        n_remote = sum(station in core4_remote.REMOTE for station in tri)
        rows.append(
            {
                "variant": variant.name,
                "label": variant.label,
                "loop": label,
                "n_remote": int(n_remote),
                "loop_class": "core_only" if n_remote == 0 else ("one_remote" if n_remote == 1 else "two_remote"),
                "snr_gain_vs_edge": float(gain),
                "fisher_gain_vs_edge": float(gain * gain),
            }
        )
    return rows


def station_rows(bm, variant: Variant, alpha_core: float | np.ndarray | list[float], p: np.ndarray) -> list[dict[str, object]]:
    rows = []
    station_alpha = alpha_by_station(alpha_core, bm)
    for i, name in enumerate(bm.names):
        parts = [f"{bm.names[j]}:{p[i, j]:.6g}" for j in range(bm.n) if i != j and p[i, j] > 0.0]
        rows.append(
            {
                "variant": variant.name,
                "station": name,
                "group": "core" if i in core4_remote.CORE else "remote",
                "core_joint_alpha": float(station_alpha[i]),
                "remote_split_row_sum": float(np.sum(p[i])),
                "total_station_fraction": float(station_alpha[i] + np.sum(p[i])),
                "fractions": "; ".join(parts),
            }
        )
    return rows


def plot_variant(labels: list[str], rows: list[dict[str, object]], variant: Variant) -> None:
    values = np.asarray([float(row["snr_gain_vs_edge"]) for row in rows], dtype=float)
    classes = [str(row["loop_class"]) for row in rows]
    colors = {"core_only": "#d00000", "one_remote": "#f77f00", "two_remote": "#0077b6"}
    fig, ax = plt.subplots(figsize=(8.5, 3.2), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, values, color=[colors[item] for item in classes], width=0.72)
    ax.axhline(1.0, color="0.15", lw=0.9, ls="--")
    ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.set_title(variant.label)
    ax.grid(True, axis="y", color="0.88", lw=0.7)
    handles = [Patch(facecolor=color, label=key.replace("_", " ")) for key, color in colors.items()]
    ax.legend(handles=handles, frameon=False, ncol=3, fontsize=8, loc="upper right")
    fig.savefig(FIGURES / f"{variant.name}_loop_gains.png", dpi=240)
    fig.savefig(FIGURES / f"{variant.name}_loop_gains.pdf")
    plt.close(fig)


def plot_comparison(labels: list[str], all_rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    x = np.arange(len(labels))
    for idx, variant in enumerate(VARIANTS):
        rows = [row for row in all_rows if row["variant"] == variant.name]
        values = np.asarray([float(row["snr_gain_vs_edge"]) for row in rows], dtype=float)
        ax.plot(x, values, marker="o", ms=3.2, lw=1.2, label=variant.name)
    ax.axhline(1.0, color="0.15", lw=0.9, ls="--")
    ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.set_title("Root-loop gain objective variants")
    ax.grid(True, axis="y", color="0.88", lw=0.7)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(FIGURES / "variant_loop_gain_comparison.png", dpi=240)
    fig.savefig(FIGURES / "variant_loop_gain_comparison.pdf")
    plt.close(fig)


def manuscript_text(base_text: str, variant: Variant, info: dict[str, object]) -> str:
    alpha_values = [float(x) for x in info["alpha_core"]]
    alpha_station_text = ", ".join(f"$\\alpha_{idx + 1}={value:.3f}$" for idx, value in enumerate(alpha_values))
    n_root = len(info.get("all_loop_snr_gains", {}))
    replacement = (
        "At the Fisher level, this variant optimizes the compact-core+remote split by matching the "
        "physical all-triangle direct split loop by loop while allowing each close station to "
        "choose its own compact-core joint-receiver fraction.  The optimized fractions are "
        f"{alpha_station_text}.  For the {n_root} root closures, the minimum, "
        f"mean, median, and maximum near/direct SNR ratios are "
        f"${float(info['near_over_direct_snr_min']):.3f}$, ${float(info['near_over_direct_snr_mean']):.3f}$, "
        f"${float(info['near_over_direct_snr_median']):.3f}$, and ${float(info['near_over_direct_snr_max']):.3f}$, "
        f"with RMS $\\log(\\mathrm{{near}}/\\mathrm{{direct}})={float(info['rms_log_near_over_direct_snr']):.3g}$.  "
        f"Relative to uniform simultaneous edge-first, the near split has minimum and mean SNR gains "
        f"${float(info['min_snr_gain']):.3f}$ and ${float(info['mean_snr_gain']):.3f}$; "
        f"{int(info['n_near_below_edge'])} of the 15 root loops fall below the edge-first reference."
    )
    marker = "At the Fisher level, the balanced core4+remote split improves all 15 root closures relative"
    if marker in base_text:
        start = base_text.index(marker)
        end_marker = "For a single $0.05$ s, 10 nm"
        end = base_text.index(end_marker, start)
        return base_text[:start] + replacement + "  " + base_text[end:]
    return base_text + "\n\n% Variant summary\n" + replacement + "\n"


def write_readme(summaries: dict[str, dict[str, object]]) -> None:
    lines = [
        "# Independent-Core-Alpha Near-Match-Direct Objective Summary",
        "",
        "All variants use the same physical station-side constraints.  Each compact-core station sends its own fraction alpha_i to the shared compact-core phase-frame receiver and splits the remaining 1-alpha_i among remote-related pairwise edge-first channels.  Remote stations split one unit of light among their five pairwise channels.",
        "",
        "This version optimizes the near split against the physical all-triangle direct split:",
        "",
        "`R_l = (SNR_near/SNR_direct)_l = sigma_direct,l / sigma_near,l`,",
        "",
        "and minimizes loop-by-loop deviations of `log R_l` while treating any `near/edge < 1` candidate as infeasible.",
        "",
        "| variant | mean alpha | alpha_i | near/direct min | near/direct mean | near/direct max | RMS log ratio | near below edge |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        info = summaries[variant.name]
        alpha_text = ", ".join(f"{float(x):.3f}" for x in info["alpha_core"])
        lines.append(
            f"| {variant.name} | {float(info['alpha']):.3f} | {alpha_text} | {float(info['near_over_direct_snr_min']):.3f} | "
            f"{float(info['near_over_direct_snr_mean']):.3f} | {float(info['near_over_direct_snr_max']):.3f} | "
            f"{float(info['rms_log_near_over_direct_snr']):.3g} | {int(info['n_near_below_edge'])} |"
        )
    (NOTES / "variant_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    bm = configure_six_benchmark()
    triangles = core4_remote.root_independent_triangles(bm.n)
    labels = [f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}" for tri in triangles]
    edge_sigma = edge_root_sigmas(bm)
    base_main = (WORKSPACE / "01_latest_manuscript" / "current_main_closure_first_telescope" / "main.tex").read_text()

    all_loop_rows: list[dict[str, object]] = []
    all_station_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}

    for variant in VARIANTS:
        p, alpha_core, gains, info = compute_variant(bm, edge_sigma, variant)
        summaries[variant.name] = info
        rows = loop_rows(labels, triangles, variant, gains)
        all_loop_rows.extend(rows)
        all_station_rows.extend(station_rows(bm, variant, alpha_core, p))
        write_csv(RESULTS / f"{variant.name}_loop_gains.csv", rows)
        write_csv(RESULTS / f"{variant.name}_station_fractions.csv", station_rows(bm, variant, alpha_core, p))
        (RESULTS / f"{variant.name}_summary.json").write_text(json.dumps(info, indent=2) + "\n")
        write_split_payload(RESULTS / f"{variant.name}_split_payload.json", variant, alpha_mean(alpha_core, bm), p, info)
        plot_variant(labels, rows, variant)
        (MANUSCRIPTS / f"main_{variant.name}.tex").write_text(manuscript_text(base_main, variant, info))

    write_csv(RESULTS / "all_variant_loop_gains.csv", all_loop_rows)
    write_csv(RESULTS / "all_variant_station_fractions.csv", all_station_rows)
    (RESULTS / "all_variant_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    plot_comparison(labels, all_loop_rows)
    write_readme(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
