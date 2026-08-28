from __future__ import annotations

import itertools
import math
import os
import sys
from dataclasses import dataclass, replace
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

DEFAULT_WAVELENGTH_NM = float(os.environ.get("RANDOM_SCALING_WAVELENGTH_NM", "658.5"))
DEFAULT_DIAMETER_M = float(os.environ.get("RANDOM_SCALING_DIAMETER_M", "2.0"))
DEFAULT_RADIUS_KM = float(os.environ.get("RANDOM_SCALING_RADIUS_KM", "18.0"))
DEFAULT_MIN_SEPARATION_KM = float(os.environ.get("RANDOM_SCALING_MIN_SEPARATION_KM", "1.6"))


@dataclass(frozen=True)
class Sample:
    total_modes: float
    u_station: np.ndarray
    vtrue: np.ndarray


@dataclass
class MonoScalingBenchmark:
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
    wavelength_nm: float
    equivalent_bandwidth_hz: float
    source_scale: float

    @property
    def n(self) -> int:
        return len(self.stations)

    @property
    def n_closure(self) -> int:
        return int(self.q_basis.shape[1])

    @property
    def rank_share(self) -> float:
        return min(1.0, (self.n - 1.0) / max(float(self.n_closure), 1.0))


def sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


def loop_label(tri: tuple[int, int, int]) -> str:
    return "-".join(f"S{i + 1}" for i in tri)


def root_independent_triangles(n_station: int) -> list[tuple[int, int, int]]:
    return [(0, i, j) for i in range(1, n_station) for j in range(i + 1, n_station)]


def ratio_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def measurement_vector(bm: MonoScalingBenchmark, tri: tuple[int, int, int]) -> np.ndarray:
    return bm.q_basis.T @ edge_vector(bm.edges, tri)


def make_random_case(
    n_station: int,
    seed: int,
    *,
    radius_km: float = DEFAULT_RADIUS_KM,
    min_separation_km: float = DEFAULT_MIN_SEPARATION_KM,
    diameter_m: float = DEFAULT_DIAMETER_M,
) -> aug.NetworkCase:
    rng = np.random.default_rng(seed)
    base_case = fig_run.make_six_station_case()
    points: list[np.ndarray] = []
    max_tries = 20000
    tries = 0
    while len(points) < n_station and tries < max_tries:
        tries += 1
        radius = radius_km * math.sqrt(float(rng.uniform(0.03, 1.0)))
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        cand = np.asarray([radius * math.cos(theta), radius * math.sin(theta)], dtype=float)
        if points and min(float(np.linalg.norm(cand - point)) for point in points) < min_separation_km:
            continue
        points.append(cand)
    if len(points) != n_station:
        raise RuntimeError(
            f"Failed to place {n_station} stations with min separation {min_separation_km:g} km after {tries} tries"
        )
    stations = np.asarray(points, dtype=float)
    stations -= np.mean(stations, axis=0, keepdims=True)
    diameters = np.full(n_station, float(diameter_m), dtype=float)
    hub, score = aug.optimize_hub(stations, diameters)
    telescopes = [
        aug.Telescope(
            name=f"R{idx + 1}",
            x_km=float(pos[0]),
            y_km=float(pos[1]),
            diameter_m=float(diameter_m),
            is_added=True,
        )
        for idx, pos in enumerate(stations)
    ]
    return aug.NetworkCase(
        key=f"random{n_station}_seed{seed}",
        title=f"Random {n_station}-station scaling layout (seed {seed})",
        latitude_deg=base_case.latitude_deg,
        center_latlon=base_case.center_latlon,
        telescopes=telescopes,
        hub_km=(float(hub[0]), float(hub[1])),
        optimization_score=float(score),
    )


def photon_equivalent_bandwidth_hz(source, wavelength_nm: float) -> float:
    freq0 = base.C_LIGHT / (float(wavelength_nm) * 1.0e-9)
    u0 = float(base.source_mode_occupation(freq0, diameter_m=1.0))
    if not np.isfinite(u0) or u0 <= 1.0e-300:
        raise ValueError(f"Monochromatic source occupation is invalid at {wavelength_nm:g} nm")
    total = 0.0
    lam_edges_nm = fig_run.wavelength_bin_edges_nm()
    for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
        center_nm = float(math.sqrt(float(lo_nm) * float(hi_nm)))
        freq = base.C_LIGHT / (center_nm * 1.0e-9)
        freq_lo = base.C_LIGHT / (float(hi_nm) * 1.0e-9)
        freq_hi = base.C_LIGHT / (float(lo_nm) * 1.0e-9)
        delta_freq = freq_hi - freq_lo
        total += float(base.source_mode_occupation(freq, diameter_m=1.0)) * float(delta_freq)
    return total / u0


def make_single_frequency_benchmark(
    case: aug.NetworkCase,
    *,
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
    source_scale: float = 1.0,
) -> MonoScalingBenchmark:
    fig_run.configure_good_runtime()
    fig_run.apply_sample_stress_runtime()
    split_sim.configure()

    stations, diameters, names, _is_added = aug.station_table_from_case(case)
    stations = np.asarray(stations, dtype=float)
    diameters = np.asarray(diameters, dtype=float)
    hub = np.asarray(case.hub_km, dtype=float)
    edges = base.edge_list(len(stations))
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, len(stations)))
    eta = 10.0 ** (-fig_run.aug.FIBER_LOSS_DB_PER_KM * np.linalg.norm(stations - hub, axis=1) / 10.0)
    hour_angles = fig_run.realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    source = fig_run.GOOD_SOURCE
    if abs(float(source_scale) - 1.0) >= 1.0e-12:
        source = replace(
            source,
            sed_fnu_mjy=tuple(float(source_scale) * float(value) for value in source.sed_fnu_mjy),
        )
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    wavelength_m = float(wavelength_nm) * 1.0e-9
    freq = base.C_LIGHT / wavelength_m
    samples: list[Sample] = []

    with fig_run.morph.patched_variant(fig_run.GOOD_VARIANT), fig_run.ngc.patched_source(source):
        truth, _axis = base.make_source_at_wavelength_nm(aug.N_PIX, aug.HALF_WIDTH_UAS, float(wavelength_nm))
        vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
        equiv_bandwidth_hz = float(photon_equivalent_bandwidth_hz(source, wavelength_nm))
        total_modes = float(aug.EXPOSURE_S * fig_run.OBSERVING_DAYS * equiv_bandwidth_hz)
        u_station = np.asarray(aug.station_u_modes(freq, diameters), dtype=float)
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            wavelength_m,
            latitude_deg=case.latitude_deg,
            declination_deg=source.dec_deg,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            samples.append(
                Sample(
                    total_modes=total_modes,
                    u_station=u_station,
                    vtrue=base.interp_vis(vgrid, uv_axis, uu, vv),
                )
            )

    return MonoScalingBenchmark(
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
        wavelength_nm=float(wavelength_nm),
        equivalent_bandwidth_hz=float(equiv_bandwidth_hz),
        source_scale=float(source_scale),
    )


def uniform_split_matrix(n_station: int) -> np.ndarray:
    p = np.zeros((n_station, n_station), dtype=float)
    for i in range(n_station):
        for j in range(n_station):
            if i != j:
                p[i, j] = 1.0 / float(n_station - 1)
    return p


def edge_fisher_values_for_split(bm: MonoScalingBenchmark, p: np.ndarray) -> np.ndarray:
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


def closure_fisher_from_edge_diagonal(bm: MonoScalingBenchmark, edge_diag: np.ndarray) -> np.ndarray:
    return sym(base.closure_fisher_after_gauge_marginalization(np.diag(edge_diag), bm.q_basis, bm.edges, bm.n))


def edge_uniform_fisher(bm: MonoScalingBenchmark) -> np.ndarray:
    p = uniform_split_matrix(bm.n)
    return closure_fisher_from_edge_diagonal(bm, edge_fisher_values_for_split(bm, p))


def global_raw_qfi_fisher(bm: MonoScalingBenchmark) -> np.ndarray:
    direct_noise = np.full(bm.n, fig_run.EPS_STATION_RUN + fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    fisher = np.zeros((bm.q_basis.shape[1], bm.q_basis.shape[1]), dtype=float)
    for sample in bm.samples:
        fisher += sample.total_modes * aug.noisy_closure_fisher_station_u(
            sample.vtrue,
            bm.eta,
            direct_noise,
            sample.u_station,
            bm.q_basis,
            bm.edges,
        )
    return sym(fisher)


def physical_direct_triangle_weights(n_station: int) -> dict[tuple[int, int, int], float]:
    triangles = tuple(tuple(tri) for tri in itertools.combinations(range(n_station), 3))
    per_triangle = 1.0 / math.comb(n_station - 1, 2)
    return {tri: float(per_triangle) for tri in triangles}


def triangle_direct_scalar_fisher(
    bm: MonoScalingBenchmark,
    tri: tuple[int, int, int],
    *,
    edge_to_index: dict[tuple[int, int], int] | None = None,
) -> float:
    direct_noise = np.full(bm.n, fig_run.EPS_STATION_RUN + fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    edge_to_index = {edge: idx for idx, edge in enumerate(bm.edges)} if edge_to_index is None else edge_to_index
    total = 0.0
    for sample in bm.samples:
        total += float(
            split_sim.core_triangle_direct_fisher_for_sample(
                tuple(tri),
                total_modes=sample.total_modes,
                vtrue=sample.vtrue,
                u_station=sample.u_station,
                eta=bm.eta,
                direct_noise=direct_noise,
                edges=bm.edges,
                edge_to_index=edge_to_index,
            )
        )
    return total


def physical_alltriangle_direct_fisher(bm: MonoScalingBenchmark) -> np.ndarray:
    weights = physical_direct_triangle_weights(bm.n)
    fisher = np.zeros((bm.q_basis.shape[1], bm.q_basis.shape[1]), dtype=float)
    edge_to_index = {edge: idx for idx, edge in enumerate(bm.edges)}
    for tri, weight in weights.items():
        scalar = triangle_direct_scalar_fisher(bm, tri, edge_to_index=edge_to_index)
        d = measurement_vector(bm, tri)
        fisher += float(weight) * scalar * np.outer(d, d)
    return sym(fisher)


def root_loop_sigmas(bm: MonoScalingBenchmark, fisher_q: np.ndarray) -> np.ndarray:
    cov = np.linalg.pinv(sym(fisher_q), rcond=1.0e-12)
    values = []
    for tri in root_independent_triangles(bm.n):
        d = measurement_vector(bm, tri)
        var = float(d @ cov @ d)
        values.append(math.sqrt(max(var, 1.0e-300)))
    return np.asarray(values, dtype=float)


def root_loop_gain_rows(
    bm: MonoScalingBenchmark,
    edge_fisher_q: np.ndarray,
    scalar_direct_fisher_q: np.ndarray,
    raw_qfi_fisher_q: np.ndarray,
) -> list[dict[str, object]]:
    edge_sigma = root_loop_sigmas(bm, edge_fisher_q)
    scalar_sigma = root_loop_sigmas(bm, scalar_direct_fisher_q)
    raw_sigma = root_loop_sigmas(bm, raw_qfi_fisher_q)
    rows: list[dict[str, object]] = []
    for idx, tri in enumerate(root_independent_triangles(bm.n)):
        rows.append(
            {
                "loop": loop_label(tri),
                "rms_edge_uniform_rad": float(edge_sigma[idx]),
                "rms_direct_alltriangle_rad": float(scalar_sigma[idx]),
                "rms_global_raw_qfi_rad": float(raw_sigma[idx]),
                "snr_gain_direct_alltriangle_vs_edge": float(edge_sigma[idx] / max(scalar_sigma[idx], 1.0e-300)),
                "snr_gain_global_raw_qfi_vs_edge": float(edge_sigma[idx] / max(raw_sigma[idx], 1.0e-300)),
                "snr_ratio_raw_over_scalar": float(scalar_sigma[idx] / max(raw_sigma[idx], 1.0e-300)),
            }
        )
    return rows
