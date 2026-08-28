from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

UAS_TO_RAD = np.deg2rad(1.0 / 3600.0) * 1e-6

N_STATIONS = int(os.environ.get("N_STATIONS", "8"))
SEED = int(os.environ.get("OPT_SEED", "20260512"))
N_ITER = int(os.environ.get("OPT_N_ITER", "6000"))
N_RESTARTS = int(os.environ.get("OPT_N_RESTARTS", "3"))
GRID_N = int(os.environ.get("OPT_GRID_N", "128"))
HALF_WIDTH_UAS = float(os.environ.get("OPT_HALF_WIDTH_UAS", "80.0"))
TARGET_U_GLAMBDA = float(os.environ.get("TARGET_U_GLAMBDA", "50.0"))
TARGET_V_GLAMBDA = float(os.environ.get("TARGET_V_GLAMBDA", "35.0"))
MAX_BASELINE_KM = float(os.environ.get("MAX_BASELINE_KM", "23.0"))
MIN_LONGEST_BASELINE_KM = float(os.environ.get("MIN_LONGEST_BASELINE_KM", "20.0"))
MAX_LONGEST_BASELINE_KM = float(os.environ.get("MAX_LONGEST_BASELINE_KM", "25.0"))
MIN_BASELINE_KM = float(os.environ.get("MIN_BASELINE_KM", "0.55"))
MAX_SHORTEST_BASELINE_KM = float(os.environ.get("MAX_SHORTEST_BASELINE_KM", "1.05"))
MID_BASELINE_LOW_KM = float(os.environ.get("MID_BASELINE_LOW_KM", "2.0"))
MID_BASELINE_HIGH_KM = float(os.environ.get("MID_BASELINE_HIGH_KM", "4.0"))
LOW_UV_GLAMBDA = float(os.environ.get("LOW_UV_GLAMBDA", "15.0"))
TARGET_LOW_UV_FRACTION = float(os.environ.get("TARGET_LOW_UV_FRACTION", "0.28"))
MIN_BASELINES_UNDER_3KM = int(os.environ.get("MIN_BASELINES_UNDER_3KM", "3"))
MIN_BASELINES_UNDER_6KM = int(os.environ.get("MIN_BASELINES_UNDER_6KM", "7"))
MAX_HUB_DISTANCE_KM = float(os.environ.get("MAX_HUB_DISTANCE_KM", "14.0"))
ARRAY_LAT_DEG = float(os.environ.get("ARRAY_LAT_DEG", "35.0"))
SOURCE_DEC_DEG = float(os.environ.get("SOURCE_DEC_DEG", "2.052388"))
N_TIME_WINDOWS = int(os.environ.get("N_TIME_WINDOWS", "36"))
EXPOSURE_S = float(os.environ.get("EXPOSURE_S", "600.0"))
EXPOSURE_GAP_S = float(os.environ.get("EXPOSURE_GAP_S", "300.0"))
TRANSIT_CENTER_HOUR = float(os.environ.get("TRANSIT_CENTER_HOUR", "0.0"))
WAVELENGTHS_NM = tuple(
    float(x.strip())
    for x in os.environ.get("OPT_WAVELENGTHS_NM", "400,500,600,700,800").split(",")
    if x.strip()
)
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "_uniform_psf_v35")
INIT_LAYOUT_FILE = os.environ.get(
    "INIT_LAYOUT_FILE",
    str(OUTDIR / "one08_uv80v35_scale0p6_about_hub_realnight_layout.json"),
)


@dataclass
class Score:
    total: float
    cell_cv: float
    empty_fraction: float
    radial_cv: float
    angular_cv: float
    psf_peak_sidelobe: float
    psf_rms_sidelobe: float
    psf_ellipticity: float
    range_penalty: float
    geometry_penalty: float
    max_abs_u: float
    max_abs_v: float
    min_baseline: float
    max_baseline: float
    horizontal_long: float
    vertical_long: float
    low_uv_fraction: float
    n_baselines_under_3km: float
    n_baselines_under_6km: float


def load_layout(path: str) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text())
    stations = np.array(payload["stations_km"], dtype=float)
    hub = np.array(payload.get("hub_km", [2.0, 0.0]), dtype=float)
    if len(stations) != N_STATIONS:
        raise ValueError(f"Expected {N_STATIONS} stations, found {len(stations)} in {path}")
    return stations, hub


def edge_list(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def realnight_hour_angles(n_sample: int, exposure_s: float, gap_s: float) -> np.ndarray:
    cadence_s = exposure_s + gap_s
    total_elapsed_s = n_sample * cadence_s
    first_start_s = -0.5 * total_elapsed_s
    mid_s = first_start_s + exposure_s / 2.0 + cadence_s * np.arange(n_sample)
    return (TRANSIT_CENTER_HOUR + mid_s / 3600.0) * (np.pi / 12.0)


def projection_coefficients() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients mapping local east/north baselines to projected u/v samples."""
    phi = np.deg2rad(ARRAY_LAT_DEG)
    dec = np.deg2rad(SOURCE_DEC_DEG)
    north_pole = np.array([0.0, np.cos(phi), np.sin(phi)])
    hour_angles = realnight_hour_angles(N_TIME_WINDOWS, EXPOSURE_S, EXPOSURE_GAP_S)

    u_e: list[float] = []
    u_n: list[float] = []
    v_e: list[float] = []
    v_n: list[float] = []
    for wavelength_nm in WAVELENGTHS_NM:
        wavelength_m = wavelength_nm * 1e-9
        for hour_angle in hour_angles:
            source = np.array(
                [
                    -np.cos(dec) * np.sin(hour_angle),
                    np.cos(phi) * np.sin(dec) - np.sin(phi) * np.cos(dec) * np.cos(hour_angle),
                    np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour_angle),
                ]
            )
            source /= np.linalg.norm(source)
            east_on_sky = np.cross(north_pole, source)
            east_on_sky /= np.linalg.norm(east_on_sky)
            north_on_sky = np.cross(source, east_on_sky)
            north_on_sky /= np.linalg.norm(north_on_sky)

            # Baseline coordinates are local east/north in km.
            scale = 1000.0 / wavelength_m / 1e9
            u_e.append(scale * east_on_sky[0])
            u_n.append(scale * east_on_sky[1])
            v_e.append(scale * north_on_sky[0])
            v_n.append(scale * north_on_sky[1])
    return np.array(u_e), np.array(u_n), np.array(v_e), np.array(v_n)


PROJ = projection_coefficients()
EDGES = edge_list(N_STATIONS)


def baselines_from_stations(stations: np.ndarray) -> np.ndarray:
    return np.array([stations[j] - stations[i] for i, j in EDGES])


def uv_samples(stations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baselines = baselines_from_stations(stations)
    bx = baselines[:, 0]
    by = baselines[:, 1]
    u_e, u_n, v_e, v_n = PROJ
    u = bx[:, None] * u_e[None, :] + by[:, None] * u_n[None, :]
    v = bx[:, None] * v_e[None, :] + by[:, None] * v_n[None, :]
    u = u.reshape(-1)
    v = v.reshape(-1)
    return np.concatenate([u, -u]), np.concatenate([v, -v])


def grid_weights(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fov_rad = 2.0 * HALF_WIDTH_UAS * UAS_TO_RAD
    du_g = 1.0 / fov_rad / 1e9
    mid = GRID_N // 2
    ix = np.rint(u / du_g).astype(int) + mid
    iy = np.rint(v / du_g).astype(int) + mid
    valid = (ix >= 0) & (ix < GRID_N) & (iy >= 0) & (iy < GRID_N)
    grid = np.zeros((GRID_N, GRID_N), dtype=float)
    np.add.at(grid, (iy[valid], ix[valid]), 1.0)

    coord = (np.arange(GRID_N) - mid) * du_g
    uu, vv = np.meshgrid(coord, coord)
    support = (uu / TARGET_U_GLAMBDA) ** 2 + (vv / TARGET_V_GLAMBDA) ** 2 <= 1.0
    return grid, support


def radial_density_cv(u: np.ndarray, v: np.ndarray) -> float:
    rho = np.sqrt((u / TARGET_U_GLAMBDA) ** 2 + (v / TARGET_V_GLAMBDA) ** 2)
    inside = rho <= 1.0
    if np.sum(inside) < 10:
        return 5.0
    bins = np.linspace(0.0, 1.0, 13)
    counts, _ = np.histogram(rho[inside], bins=bins)
    areas = np.diff(bins**2)
    density = counts / np.maximum(areas, 1e-12)
    return float(np.std(density) / (np.mean(density) + 1e-12))


def angular_density_cv(u: np.ndarray, v: np.ndarray) -> float:
    rho = np.sqrt((u / TARGET_U_GLAMBDA) ** 2 + (v / TARGET_V_GLAMBDA) ** 2)
    mask = (rho > 0.20) & (rho <= 1.0)
    if np.sum(mask) < 10:
        return 5.0
    theta = np.mod(np.arctan2(v[mask] / TARGET_V_GLAMBDA, u[mask] / TARGET_U_GLAMBDA), 2.0 * np.pi)
    counts, _ = np.histogram(theta, bins=np.linspace(0.0, 2.0 * np.pi, 19))
    return float(np.std(counts) / (np.mean(counts) + 1e-12))


def psf_metrics(grid: np.ndarray) -> tuple[float, float, float]:
    # Uniform cell weighting: occupied cells carry equal weight. This tests the
    # PSF of the unbiased gridded inverse, not natural-density weighting.
    occupied = (grid > 0.0).astype(float)
    if np.sum(occupied) <= 1.0:
        return 1.0, 1.0, 2.0
    beam = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(occupied))).real
    beam /= np.max(np.abs(beam)) + 1e-12

    pix_uas = 2.0 * HALF_WIDTH_UAS / GRID_N
    yy, xx = np.indices(beam.shape)
    rr_uas = np.sqrt((xx - GRID_N // 2) ** 2 + (yy - GRID_N // 2) ** 2) * pix_uas
    side = rr_uas > 7.5
    peak = float(np.max(np.abs(beam[side])))
    rms = float(np.sqrt(np.mean(beam[side] ** 2)))

    core = rr_uas <= 12.0
    weights = np.clip(beam[core], 0.0, None)
    if np.sum(weights) <= 0.0:
        return peak, rms, 2.0
    x = (xx[core] - GRID_N // 2) * pix_uas
    y = (yy[core] - GRID_N // 2) * pix_uas
    x -= np.average(x, weights=weights)
    y -= np.average(y, weights=weights)
    cov_xx = np.average(x * x, weights=weights)
    cov_yy = np.average(y * y, weights=weights)
    cov_xy = np.average(x * y, weights=weights)
    evals = np.linalg.eigvalsh([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    ellipticity = float(np.sqrt(max(evals[-1], 1e-12) / max(evals[0], 1e-12)) - 1.0)
    return peak, rms, ellipticity


def score_layout(stations: np.ndarray, hub: np.ndarray) -> Score:
    baselines = baselines_from_stations(stations)
    lengths = np.linalg.norm(baselines, axis=1)
    min_b = float(np.min(lengths))
    max_b = float(np.max(lengths))
    abs_dx = np.abs(baselines[:, 0])
    abs_dy = np.abs(baselines[:, 1])
    horizontal = lengths * np.exp(-0.5 * (abs_dy / np.maximum(abs_dx, 1e-6) / 0.22) ** 2)
    vertical = lengths * np.exp(-0.5 * (abs_dx / np.maximum(abs_dy, 1e-6) / 0.25) ** 2)
    horizontal_long = float(np.max(horizontal))
    vertical_long = float(np.max(vertical))
    n_under_3 = float(np.sum(lengths <= 3.0))
    n_under_6 = float(np.sum(lengths <= 6.0))

    u, v = uv_samples(stations)
    grid, support = grid_weights(u, v)
    cells = grid[support]
    cell_cv = float(np.std(cells) / (np.mean(cells) + 1e-12))
    empty_fraction = float(np.mean(cells == 0.0))
    radial_cv = radial_density_cv(u, v)
    angular_cv = angular_density_cv(u, v)
    psf_peak, psf_rms, psf_ell = psf_metrics(grid)
    max_abs_u = float(np.max(np.abs(u)))
    max_abs_v = float(np.max(np.abs(v)))
    uv_radius = np.sqrt(u**2 + v**2)
    low_uv_fraction = float(np.mean(uv_radius <= LOW_UV_GLAMBDA))

    range_penalty = (
        np.log(max_abs_u / TARGET_U_GLAMBDA) ** 2
        + 1.4 * np.log(max_abs_v / TARGET_V_GLAMBDA) ** 2
    )
    hub_distances = np.linalg.norm(stations - hub, axis=1)
    geometry_penalty = 0.0
    geometry_penalty += 8.0 * max(0.0, MIN_BASELINE_KM - min_b) ** 2
    geometry_penalty += 2.8 * max(0.0, min_b - MAX_SHORTEST_BASELINE_KM) ** 2
    mid_distance = np.min(
        np.minimum(
            np.abs(lengths - MID_BASELINE_LOW_KM),
            np.abs(lengths - MID_BASELINE_HIGH_KM),
        )
    )
    if not np.any((lengths >= MID_BASELINE_LOW_KM) & (lengths <= MID_BASELINE_HIGH_KM)):
        geometry_penalty += 1.8 * float(mid_distance**2)
    geometry_penalty += 1.15 * max(0.0, MIN_BASELINES_UNDER_3KM - n_under_3) ** 2
    geometry_penalty += 0.42 * max(0.0, MIN_BASELINES_UNDER_6KM - n_under_6) ** 2
    geometry_penalty += 1.25 * max(0.0, MIN_LONGEST_BASELINE_KM - max_b) ** 2
    geometry_penalty += 1.25 * max(0.0, max_b - MAX_LONGEST_BASELINE_KM) ** 2
    geometry_penalty += 0.045 * max(0.0, np.max(hub_distances) - MAX_HUB_DISTANCE_KM) ** 2
    geometry_penalty += 0.030 * max(0.0, 0.65 * MAX_BASELINE_KM - horizontal_long) ** 2
    geometry_penalty += 0.030 * max(0.0, 0.45 * MAX_BASELINE_KM - vertical_long) ** 2

    total = (
        1.60 * cell_cv
        + 2.00 * empty_fraction
        + 0.85 * radial_cv
        + 0.75 * angular_cv
        + 2.80 * psf_peak
        + 8.00 * psf_rms
        + 0.35 * psf_ell
        + 4.20 * max(0.0, TARGET_LOW_UV_FRACTION - low_uv_fraction) ** 2
        + 1.10 * range_penalty
        + geometry_penalty
    )
    return Score(
        total=float(total),
        cell_cv=cell_cv,
        empty_fraction=empty_fraction,
        radial_cv=radial_cv,
        angular_cv=angular_cv,
        psf_peak_sidelobe=psf_peak,
        psf_rms_sidelobe=psf_rms,
        psf_ellipticity=psf_ell,
        range_penalty=float(range_penalty),
        geometry_penalty=float(geometry_penalty),
        max_abs_u=max_abs_u,
        max_abs_v=max_abs_v,
        min_baseline=min_b,
        max_baseline=max_b,
        horizontal_long=horizontal_long,
        vertical_long=vertical_long,
        low_uv_fraction=low_uv_fraction,
        n_baselines_under_3km=n_under_3,
        n_baselines_under_6km=n_under_6,
    )


def canonicalize(stations: np.ndarray, reference_centroid: np.ndarray) -> np.ndarray:
    out = stations.copy()
    out += reference_centroid - np.mean(out, axis=0)
    return out


def optimize(initial: np.ndarray, hub: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, Score, list[float]]:
    reference_centroid = np.mean(initial, axis=0)
    best = initial.copy()
    best_score = score_layout(best, hub)
    current = best.copy()
    current_score = best_score
    history = [best_score.total]

    for restart in range(N_RESTARTS):
        if restart == 0:
            current = initial.copy()
        else:
            current = initial + rng.normal(scale=1.8, size=initial.shape)
            current = canonicalize(current, reference_centroid)
        current_score = score_layout(current, hub)

        for it in range(N_ITER):
            frac = it / max(1, N_ITER - 1)
            temperature = 0.22 * (1.0 - frac) + 0.015
            step = 1.20 * (1.0 - frac) + 0.12
            trial = current.copy()
            if rng.random() < 0.18:
                # A mild global stretch/shear proposal helps escape layouts
                # with all long baselines aligned along one station pair.
                angle = rng.normal(scale=0.08 * (1.0 - frac) + 0.01)
                shear = rng.normal(scale=0.05 * (1.0 - frac) + 0.008)
                stretch = np.diag(1.0 + rng.normal(scale=0.05 * (1.0 - frac) + 0.008, size=2))
                rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                mat = rot @ np.array([[1.0, shear], [0.0, 1.0]]) @ stretch
                centroid = np.mean(trial, axis=0)
                trial = centroid + (trial - centroid) @ mat.T
            else:
                idx = int(rng.integers(0, len(trial)))
                trial[idx] += rng.normal(scale=step, size=2)
            trial = canonicalize(trial, reference_centroid)
            trial_score = score_layout(trial, hub)
            delta = trial_score.total - current_score.total
            if delta < 0.0 or rng.random() < np.exp(-delta / temperature):
                current = trial
                current_score = trial_score
                if current_score.total < best_score.total:
                    best = current.copy()
                    best_score = current_score
            history.append(best_score.total)
    return best, best_score, history


def make_diagnostic(
    initial: np.ndarray,
    optimized: np.ndarray,
    hub: np.ndarray,
    initial_score: Score,
    optimized_score: Score,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), constrained_layout=True)
    layouts = [("initial", initial, initial_score), ("optimized", optimized, optimized_score)]
    for row, (label, stations, score) in enumerate(layouts):
        ax = axes[row, 0]
        ax.scatter(stations[:, 0], stations[:, 1], s=24, color="#005f73")
        ax.scatter([hub[0]], [hub[1]], s=60, marker="*", color="#ca6702", label="hub", zorder=3)
        for i, j in EDGES:
            ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.82", lw=0.45)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{label} topology")
        ax.set_xlabel("east-west (km)")
        ax.set_ylabel("north-south (km)")
        ax.legend(frameon=False, fontsize=7)

        u, v = uv_samples(stations)
        ax = axes[row, 1]
        ax.scatter(u, v, s=0.9, color="#005f73", alpha=0.18)
        theta = np.linspace(0.0, 2.0 * np.pi, 240)
        ax.plot(TARGET_U_GLAMBDA * np.cos(theta), TARGET_V_GLAMBDA * np.sin(theta), color="#bb3e03", lw=1.0)
        ax.plot(LOW_UV_GLAMBDA * np.cos(theta), LOW_UV_GLAMBDA * np.sin(theta), color="#94d2bd", lw=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"uv samples: |u|={score.max_abs_u:.1f}, |v|={score.max_abs_v:.1f} G$\\lambda$\n"
            f"$|k|<15$: {score.low_uv_fraction:.2f}"
        )
        ax.set_xlabel(r"$u$ (G$\lambda$)")
        ax.set_ylabel(r"$v$ (G$\lambda$)")

        grid, _ = grid_weights(u, v)
        occupied = (grid > 0.0).astype(float)
        beam = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(occupied))).real
        beam /= np.max(np.abs(beam)) + 1e-12
        ax = axes[row, 2]
        extent = [-HALF_WIDTH_UAS, HALF_WIDTH_UAS, -HALF_WIDTH_UAS, HALF_WIDTH_UAS]
        im = ax.imshow(np.abs(beam), origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=0.35)
        ax.set_title(
            f"PSF |sidelobe|={score.psf_peak_sidelobe:.2f}, rms={score.psf_rms_sidelobe:.3f}"
        )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    fig.colorbar(im, ax=axes[:, 2], fraction=0.026, pad=0.02, label="normalized |dirty beam|")
    png = OUTDIR / f"optimized_array_topology{OUTPUT_SUFFIX}_diagnostic.png"
    pdf = OUTDIR / f"optimized_array_topology{OUTPUT_SUFFIX}_diagnostic.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def score_to_dict(score: Score) -> dict[str, float]:
    return {key: float(getattr(score, key)) for key in score.__dataclass_fields__}


def main() -> None:
    rng = np.random.default_rng(SEED)
    initial, hub = load_layout(INIT_LAYOUT_FILE)
    initial_score = score_layout(initial, hub)
    optimized, optimized_score, history = optimize(initial, hub, rng)

    layout_path = OUTDIR / f"optimized_array_topology{OUTPUT_SUFFIX}.json"
    payload = {
        "stations_km": optimized.tolist(),
        "hub_km": hub.tolist(),
        "source_layout": INIT_LAYOUT_FILE,
        "optimizer": "simulated_annealing_uv_uniformity_psf",
        "target_u_g_lambda": TARGET_U_GLAMBDA,
        "target_v_g_lambda": TARGET_V_GLAMBDA,
        "low_uv_g_lambda": LOW_UV_GLAMBDA,
        "target_low_uv_fraction": TARGET_LOW_UV_FRACTION,
        "min_baselines_under_3km": MIN_BASELINES_UNDER_3KM,
        "min_baselines_under_6km": MIN_BASELINES_UNDER_6KM,
        "n_stations": N_STATIONS,
        "min_longest_baseline_km": MIN_LONGEST_BASELINE_KM,
        "max_longest_baseline_km": MAX_LONGEST_BASELINE_KM,
        "array_latitude_deg": ARRAY_LAT_DEG,
        "source_declination_deg": SOURCE_DEC_DEG,
        "n_time_windows": N_TIME_WINDOWS,
        "exposure_s": EXPOSURE_S,
        "exposure_gap_s": EXPOSURE_GAP_S,
        "wavelengths_nm": WAVELENGTHS_NM,
        "initial_score": score_to_dict(initial_score),
        "optimized_score": score_to_dict(optimized_score),
        "score_history_tail": [float(x) for x in history[-200:]],
    }
    layout_path.write_text(json.dumps(payload, indent=2) + "\n")
    pdf, png = make_diagnostic(initial, optimized, hub, initial_score, optimized_score)
    print(layout_path)
    print(pdf)
    print(png)
    print(json.dumps({"initial": score_to_dict(initial_score), "optimized": score_to_dict(optimized_score)}, indent=2))


if __name__ == "__main__":
    main()
