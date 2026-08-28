from __future__ import annotations

import json
import math
import os
import sys
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


os.environ.setdefault("FIG2_EXPOSURE_S", "0.100")

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE = THIS_DIR.parents[1]
for path in (
    THIS_DIR,
    WORKSPACE / "03_figure_generation_code" / "0608_core_modules",
    WORKSPACE / "03_figure_generation_code" / "0608_all_python_snapshot",
):
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

import run_broad_plume_split_objective_rml_rawdirect as fig_run  # noqa: E402
from plot_prl_broadband_blr_realnight import project_enu_baselines  # noqa: E402


base = fig_run.opt.base
aug = fig_run.aug
split_sim = fig_run.split_sim

CORE = tuple(range(3))
REMOTE = tuple(range(3, 6))
BALANCED10 = tuple(tuple(tri) for tri in fig_run.BALANCED_INDEPENDENT_TRIANGLES)
SPLIT_FLOOR = 1.0e-4


@dataclass(frozen=True)
class Sample:
    total_modes: float
    u_station: np.ndarray
    vtrue: np.ndarray


@dataclass
class RawJeBenchmark:
    case: object
    stations: np.ndarray
    diameters: np.ndarray
    names: list[str]
    hub: np.ndarray
    edges: list[tuple[int, int]]
    baselines: np.ndarray
    q_basis: np.ndarray
    eta: np.ndarray
    samples: list[Sample]

    @property
    def n(self) -> int:
        return len(self.stations)


def sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def loop_label(tri: tuple[int, int, int]) -> str:
    return "-".join(f"S{i + 1}" for i in tri)


def make_benchmark() -> RawJeBenchmark:
    fig_run.configure_good_runtime()
    fig_run.apply_sample_stress_runtime()
    split_sim.configure()
    case = fig_run.make_six_station_case()
    stations, diameters, names, _is_added = aug.station_table_from_case(case)
    stations = np.asarray(stations, dtype=float)
    diameters = np.asarray(diameters, dtype=float)
    hub = np.asarray(case.hub_km, dtype=float)
    edges = base.edge_list(len(stations))
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, len(stations)))
    eta = 10.0 ** (-fig_run.aug.FIBER_LOSS_DB_PER_KM * np.linalg.norm(stations - hub, axis=1) / 10.0)
    hour_angles = fig_run.realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    lam_edges_nm = fig_run.wavelength_bin_edges_nm()
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    samples: list[Sample] = []

    with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(fig_run.GOOD_SOURCE):
        wavelength_source = getattr(base, "make_source_at_wavelength_nm", None)
        fallback_truth, _axis = base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
        fallback_grid, fallback_uv = base.visibility_grid(fallback_truth, fov_rad)
        for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
            center_nm = float(math.sqrt(lo_nm * hi_nm))
            lam_m = center_nm * 1.0e-9
            freq = base.C_LIGHT / lam_m
            freq_lo = base.C_LIGHT / (hi_nm * 1.0e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1.0e-9)
            total_modes = aug.EXPOSURE_S * fig_run.OBSERVING_DAYS * (freq_hi - freq_lo)
            u_station = aug.station_u_modes(freq, diameters)
            if callable(wavelength_source):
                truth, _axis = wavelength_source(aug.N_PIX, aug.HALF_WIDTH_UAS, center_nm)
                vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
            else:
                vgrid, uv_axis = fallback_grid, fallback_uv
            uu_rows, vv_rows = project_enu_baselines(
                baselines,
                hour_angles,
                lam_m,
                latitude_deg=case.latitude_deg,
                declination_deg=fig_run.GOOD_SOURCE.dec_deg,
            )
            for uu, vv in zip(uu_rows, vv_rows):
                samples.append(
                    Sample(
                        total_modes=float(total_modes),
                        u_station=np.asarray(u_station, dtype=float),
                        vtrue=base.interp_vis(vgrid, uv_axis, uu, vv),
                    )
                )

    return RawJeBenchmark(
        case=case,
        stations=stations,
        diameters=diameters,
        names=list(names),
        hub=hub,
        edges=edges,
        baselines=baselines,
        q_basis=q_basis,
        eta=eta,
        samples=samples,
    )


def edge_fisher_values_for_split(bm: RawJeBenchmark, p: np.ndarray) -> np.ndarray:
    out = np.zeros(len(bm.edges), dtype=float)
    station_noise = np.full(bm.n, fig_run.EPS_STATION_RUN, dtype=float)
    for sample in bm.samples:
        nu_eff = np.clip(np.abs(sample.vtrue), 1.0e-4, 0.98)
        ai = bm.eta * sample.u_station + station_noise
        for idx, (i, j) in enumerate(bm.edges):
            if p[i, j] <= 0.0 or p[j, i] <= 0.0:
                continue
            k = (
                sample.total_modes
                * 4.0
                * bm.eta[i]
                * bm.eta[j]
                * sample.u_station[i]
                * sample.u_station[j]
                * nu_eff[idx] ** 2
            )
            denom = p[i, j] * ai[i] + p[j, i] * ai[j] + fig_run.EPS_PAIR_RUN
            out[idx] += float(k * p[i, j] * p[j, i] / max(denom, 1.0e-300))
    return out


def closure_fisher_from_edge(bm: RawJeBenchmark, edge_fisher: np.ndarray) -> np.ndarray:
    return sym(base.closure_fisher_after_gauge_marginalization(edge_fisher, bm.q_basis, bm.edges, bm.n))


def uniform_edge_fisher(bm: RawJeBenchmark) -> np.ndarray:
    p = np.zeros((bm.n, bm.n), dtype=float)
    for i in range(bm.n):
        for j in range(bm.n):
            if i != j:
                p[i, j] = 1.0 / (bm.n - 1.0)
    return closure_fisher_from_edge(bm, np.diag(edge_fisher_values_for_split(bm, p)))


def raw_triangle_edge_matrix(
    bm: RawJeBenchmark,
    tri: tuple[int, int, int],
    weight: float,
    sample: Sample,
) -> np.ndarray:
    local_edges = base.edge_list(3)
    edge_to_index = {edge: idx for idx, edge in enumerate(bm.edges)}
    local_vis = np.asarray(
        [sample.vtrue[edge_to_index[(tri[i], tri[j])]] for i, j in local_edges],
        dtype=complex,
    )
    subset = list(tri)
    direct_noise = np.full(3, fig_run.EPS_STATION_RUN + fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    local_raw = sample.total_modes * split_sim.raw_edge_phase_fisher_station_u(
        local_vis,
        bm.eta[subset],
        direct_noise,
        sample.u_station[subset],
        local_edges,
    )
    out = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
    for local_i, edge_i in enumerate(local_edges):
        global_i = edge_to_index[(tri[edge_i[0]], tri[edge_i[1]])]
        for local_j, edge_j in enumerate(local_edges):
            global_j = edge_to_index[(tri[edge_j[0]], tri[edge_j[1]])]
            out[global_i, global_j] += float(weight) * local_raw[local_i, local_j]
    return sym(out)


def rawdirect_balanced10_fisher(
    bm: RawJeBenchmark,
    *,
    weight: float = 0.2,
    schur_per_sample: bool = False,
) -> np.ndarray:
    if schur_per_sample:
        fisher_q = np.zeros((bm.q_basis.shape[1], bm.q_basis.shape[1]), dtype=float)
        for sample in bm.samples:
            edge = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
            for tri in BALANCED10:
                edge += raw_triangle_edge_matrix(bm, tri, weight, sample)
            fisher_q += closure_fisher_from_edge(bm, edge)
        return sym(fisher_q)

    edge = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
    for sample in bm.samples:
        for tri in BALANCED10:
            edge += raw_triangle_edge_matrix(bm, tri, weight, sample)
    return closure_fisher_from_edge(bm, edge)


def all_triangles(n_station: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(tri) for tri in itertools.combinations(range(n_station), 3))


def integrated_raw_triangle_edges(
    bm: RawJeBenchmark,
    triangles: tuple[tuple[int, int, int], ...] | None = None,
) -> dict[tuple[int, int, int], np.ndarray]:
    eval_triangles = all_triangles(bm.n) if triangles is None else triangles
    out = {
        tri: np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
        for tri in eval_triangles
    }
    for sample in bm.samples:
        for tri in eval_triangles:
            out[tri] += raw_triangle_edge_matrix(bm, tri, 1.0, sample)
    return {tri: sym(edge) for tri, edge in out.items()}


def rawdirect_weighted_fisher_from_edges(
    bm: RawJeBenchmark,
    triangle_edges: dict[tuple[int, int, int], np.ndarray],
    weights: dict[tuple[int, int, int], float],
) -> np.ndarray:
    edge = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
    for tri, weight in weights.items():
        if abs(float(weight)) > 0.0:
            edge += float(weight) * triangle_edges[tuple(tri)]
    return closure_fisher_from_edge(bm, edge)


def row_softmax(raw_row: np.ndarray, allowed: np.ndarray, total: float) -> np.ndarray:
    out = np.zeros(raw_row.shape[0], dtype=float)
    n_allowed = int(np.sum(allowed))
    if total <= 0.0 or n_allowed == 0:
        return out
    active_floor = min(SPLIT_FLOOR, 0.25 * total / max(n_allowed, 1))
    remaining = max(float(total) - n_allowed * active_floor, 0.0)
    vals = raw_row[allowed]
    vals = vals - np.max(vals)
    weights = np.exp(vals)
    weights /= np.sum(weights)
    out[allowed] = active_floor + remaining * weights
    return out


def project_near_split(raw: np.ndarray, alpha_core: np.ndarray, bm: RawJeBenchmark) -> np.ndarray:
    p = np.zeros((bm.n, bm.n), dtype=float)
    for core_idx, station in enumerate(CORE):
        allowed = np.zeros(bm.n, dtype=bool)
        allowed[list(REMOTE)] = True
        p[station] = row_softmax(raw[station], allowed, 1.0 - float(alpha_core[core_idx]))
    for station in REMOTE:
        allowed = np.ones(bm.n, dtype=bool)
        allowed[station] = False
        p[station] = row_softmax(raw[station], allowed, 1.0)
    return p


def alpha_from_raw(raw_alpha: np.ndarray, alpha_min: float = 0.02, alpha_max: float = 0.80) -> np.ndarray:
    raw_alpha = np.asarray(raw_alpha, dtype=float).reshape(len(CORE))
    span = alpha_max - alpha_min
    return alpha_min + span / (1.0 + np.exp(-raw_alpha))


def raw_from_alpha(alpha: np.ndarray, alpha_min: float = 0.02, alpha_max: float = 0.80) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float).reshape(len(CORE))
    x = np.clip((alpha - alpha_min) / (alpha_max - alpha_min), 1.0e-9, 1.0 - 1.0e-9)
    return np.log(x / (1.0 - x))


def raw_from_split(p: np.ndarray, alpha_core: np.ndarray, bm: RawJeBenchmark) -> np.ndarray:
    raw = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw, -np.inf)
    for core_idx, station in enumerate(CORE):
        allowed = np.zeros(bm.n, dtype=bool)
        allowed[list(REMOTE)] = True
        total = max(1.0 - float(alpha_core[core_idx]), 1.0e-12)
        n_allowed = int(np.sum(allowed))
        active_floor = min(SPLIT_FLOOR, 0.25 * total / max(n_allowed, 1))
        remaining = max(total - n_allowed * active_floor, 1.0e-300)
        raw[station, allowed] = np.log(np.maximum((p[station, allowed] - active_floor) / remaining, 1.0e-300))
    for station in REMOTE:
        allowed = np.ones(bm.n, dtype=bool)
        allowed[station] = False
        total = 1.0
        n_allowed = int(np.sum(allowed))
        active_floor = min(SPLIT_FLOOR, 0.25 * total / max(n_allowed, 1))
        remaining = max(total - n_allowed * active_floor, 1.0e-300)
        raw[station, allowed] = np.log(np.maximum((p[station, allowed] - active_floor) / remaining, 1.0e-300))
    return raw


def core_raw_edge_fisher_alpha(bm: RawJeBenchmark, alpha_core: np.ndarray) -> np.ndarray:
    local_edges = base.edge_list(len(CORE))
    edge_to_index = {edge: idx for idx, edge in enumerate(bm.edges)}
    alpha_core = np.asarray(alpha_core, dtype=float).reshape(len(CORE))
    out = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
    for sample in bm.samples:
        local_vis = np.asarray([sample.vtrue[edge_to_index[edge]] for edge in local_edges], dtype=complex)
        noise = alpha_core * fig_run.EPS_STATION_RUN + fig_run.EPS_DIRECT_EXTRA_RUN
        local_raw = sample.total_modes * split_sim.raw_edge_phase_fisher_station_u(
            local_vis,
            alpha_core * bm.eta[list(CORE)],
            noise,
            sample.u_station[list(CORE)],
            local_edges,
        )
        for local_i, edge_i in enumerate(local_edges):
            global_i = edge_to_index[edge_i]
            for local_j, edge_j in enumerate(local_edges):
                global_j = edge_to_index[edge_j]
                out[global_i, global_j] += local_raw[local_i, local_j]
    return sym(out)


def near_core_direct_remote_edge_fisher(
    bm: RawJeBenchmark,
    p: np.ndarray,
    alpha_core: np.ndarray,
    core_cache: dict[tuple[float, ...], np.ndarray] | None = None,
) -> np.ndarray:
    key = tuple(float(f"{x:.8f}") for x in np.asarray(alpha_core, dtype=float).reshape(len(CORE)))
    if core_cache is not None and key in core_cache:
        core_edge = core_cache[key]
    else:
        core_edge = core_raw_edge_fisher_alpha(bm, np.asarray(alpha_core, dtype=float))
        if core_cache is not None:
            core_cache[key] = core_edge
    remote_diag = edge_fisher_values_for_split(bm, p)
    for idx, (i, j) in enumerate(bm.edges):
        if i in CORE and j in CORE:
            remote_diag[idx] = 0.0
    return closure_fisher_from_edge(bm, core_edge + np.diag(remote_diag))


def loop_matrix(bm: RawJeBenchmark, loops: tuple[tuple[int, int, int], ...] = BALANCED10) -> np.ndarray:
    return np.stack([bm.q_basis.T @ split_sim.closure_edge_vector(bm.edges, tri) for tri in loops], axis=0)


def loop_sigmas(bm: RawJeBenchmark, fisher_q: np.ndarray, loops: tuple[tuple[int, int, int], ...] = BALANCED10) -> np.ndarray:
    cov = np.linalg.pinv(sym(fisher_q), rcond=1.0e-12)
    mat = loop_matrix(bm, loops)
    loop_cov = sym(mat @ cov @ mat.T)
    return np.sqrt(np.maximum(np.diag(loop_cov), 1.0e-300))


def gain_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def score_near_candidate(near_sigma: np.ndarray, edge_sigma: np.ndarray, direct_sigma: np.ndarray) -> float:
    near_vs_edge = edge_sigma / np.maximum(near_sigma, 1.0e-300)
    if float(np.min(near_vs_edge)) < 1.0 - 1.0e-10:
        return -math.inf
    near_vs_direct = direct_sigma / np.maximum(near_sigma, 1.0e-300)
    log_match = np.log(np.maximum(near_vs_direct, 1.0e-300))
    over_direct = np.maximum(0.0, log_match)
    return float(
        np.min(log_match)
        - 0.30 * np.var(log_match)
        - 0.05 * np.mean(log_match * log_match)
        - 0.20 * np.max(over_direct) ** 2
    )


def starts_from_payloads(bm: RawJeBenchmark) -> list[tuple[str, np.ndarray, np.ndarray]]:
    starts: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path in (
        WORKSPACE / "18_balanced_10loop_independent_set_20260611" / "results" / "balanced10_near_split_payload.json",
        WORKSPACE / "16_six_station_reduced_from7_20260611" / "results" / "near_match_direct_split_payload.json",
        WORKSPACE / "12_independent_core_alpha_20260611" / "results" / "near_match_direct_split_payload.json",
    ):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            p = np.asarray(payload["split_matrix"], dtype=float)
            alpha = np.asarray(payload.get("alpha_core", [payload.get("alpha", 0.15)] * len(CORE)), dtype=float)
            alpha = alpha.reshape(-1)[: len(CORE)]
            if p.shape == (bm.n, bm.n) and alpha.size == len(CORE):
                starts.append((path.parent.parent.name, raw_from_split(p, alpha, bm), raw_from_alpha(alpha)))
        except Exception:
            continue
    raw0 = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw0, -np.inf)
    for alpha0 in (0.06, 0.10, 0.16, 0.25, 0.40):
        starts.append((f"uniform_alpha_{alpha0:g}", raw0.copy(), raw_from_alpha(np.full(len(CORE), alpha0))))
    return starts


def optimize_near_coreedge(
    bm: RawJeBenchmark,
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
    *,
    seed: int = 20260616,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    raw0 = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw0, -np.inf)
    active = [(i, j) for i in CORE for j in REMOTE]
    active += [(i, j) for i in REMOTE for j in range(bm.n) if i != j]
    core_cache: dict[tuple[float, ...], np.ndarray] = {}
    counts = {"evaluations": 0, "core_recomputes": 0}

    best_score = -math.inf
    best_raw: np.ndarray | None = None
    best_raw_alpha: np.ndarray | None = None
    best_p: np.ndarray | None = None
    best_alpha: np.ndarray | None = None
    best_sigma: np.ndarray | None = None
    best_start = ""

    def evaluate(raw: np.ndarray, raw_alpha: np.ndarray, start_name: str = "") -> bool:
        nonlocal best_score, best_raw, best_raw_alpha, best_p, best_alpha, best_sigma, best_start
        alpha = alpha_from_raw(raw_alpha)
        before = len(core_cache)
        p = project_near_split(raw, alpha, bm)
        fisher = near_core_direct_remote_edge_fisher(bm, p, alpha, core_cache)
        counts["evaluations"] += 1
        counts["core_recomputes"] += int(len(core_cache) > before)
        sigma = loop_sigmas(bm, fisher)
        score = score_near_candidate(sigma, edge_sigma, direct_sigma)
        if np.isfinite(score) and score > best_score + 1.0e-13:
            best_score = score
            best_raw = raw.copy()
            best_raw_alpha = raw_alpha.copy()
            best_p = p
            best_alpha = alpha.copy()
            best_sigma = sigma
            if start_name:
                best_start = start_name
            return True
        return False

    for name, raw, raw_alpha in starts_from_payloads(bm):
        evaluate(raw, raw_alpha, name)

    for alpha0 in (0.035, 0.055, 0.075, 0.10, 0.14, 0.20, 0.30, 0.45):
        center = raw_from_alpha(np.full(len(CORE), alpha0))
        for _ in range(6):
            raw_alpha = center + rng.normal(scale=0.75, size=len(CORE))
            for scale in (0.35, 0.80, 1.45, 2.30):
                for _ in range(10):
                    raw = raw0 + rng.normal(scale=scale, size=(bm.n, bm.n))
                    np.fill_diagonal(raw, -np.inf)
                    evaluate(raw, raw_alpha, f"random_alpha_{alpha0:g}_scale_{scale:g}")

    if best_raw is None or best_raw_alpha is None:
        raise RuntimeError("No near candidate was evaluated successfully.")

    for outer in range(3):
        for width in (0.75, 0.35, 0.16, 0.07, 0.03):
            improved = True
            passes = 0
            while improved and passes < 5:
                improved = False
                passes += 1
                assert best_raw is not None and best_raw_alpha is not None
                for idx in range(len(CORE)):
                    for sign in (-1.0, 1.0):
                        candidate_alpha = best_raw_alpha.copy()
                        candidate_alpha[idx] += sign * width
                        improved = evaluate(best_raw, candidate_alpha, f"alpha_coord_{outer}") or improved
        for width in (0.85, 0.40, 0.18, 0.08, 0.035, 0.015):
            improved = True
            passes = 0
            while improved and passes < 7:
                improved = False
                passes += 1
                assert best_raw is not None and best_raw_alpha is not None
                for i, j in active:
                    for sign in (-1.0, 1.0):
                        candidate = best_raw.copy()
                        candidate[i, j] += sign * width
                        improved = evaluate(candidate, best_raw_alpha, f"split_coord_{outer}") or improved
        assert best_raw is not None and best_raw_alpha is not None
        for scale in (0.035, 0.075, 0.14):
            for _ in range(80):
                candidate = best_raw + rng.normal(scale=scale, size=(bm.n, bm.n))
                np.fill_diagonal(candidate, -np.inf)
                evaluate(candidate, best_raw_alpha, f"local_random_{outer}")

    assert best_p is not None and best_alpha is not None and best_sigma is not None
    near_vs_edge = edge_sigma / np.maximum(best_sigma, 1.0e-300)
    near_vs_direct = direct_sigma / np.maximum(best_sigma, 1.0e-300)
    info = {
        "objective": "maximize near/direct on fixed raw-Je omega_l=0.2 target, rejecting candidates with any selected-loop near/edge below one",
        "score": float(best_score),
        "best_start": best_start,
        "alpha_core": [float(x) for x in best_alpha],
        "station_total_budgets": {
            f"S{i + 1}": float((best_alpha[list(CORE).index(i)] if i in CORE else 0.0) + np.sum(best_p[i]))
            for i in range(bm.n)
        },
        "counts": counts,
        "near_gain_vs_edge": gain_stats(near_vs_edge),
        "near_snr_ratio_vs_rawdirect": gain_stats(near_vs_direct),
        "n_near_below_edge": int(np.sum(near_vs_edge < 1.0 - 1.0e-9)),
        "n_near_below_rawdirect": int(np.sum(near_vs_direct < 1.0 - 1.0e-9)),
    }
    return best_p, best_alpha, best_sigma, info
