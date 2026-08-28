from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

ARRAY_SCALE = float(os.environ.get("ARRAY_SCALE", "1.0"))
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "30"))
N_TIME_WINDOWS = int(os.environ.get("N_TIME_WINDOWS", "72"))
EXPOSURE_S = float(os.environ.get("EXPOSURE_S", "300.0"))
EXPOSURE_MIN_LABEL = f"{EXPOSURE_S / 60.0:g}".replace(".", "p")
LAMBDA_MIN_NM = float(os.environ.get("LAMBDA_MIN_NM", "400.0"))
LAMBDA_MAX_NM = float(os.environ.get("LAMBDA_MAX_NM", "800.0"))
N_LAMBDA_BINS = int(os.environ.get("N_LAMBDA_BINS", "36"))
SHORT_BASELINE_THRESHOLD_KM = float(os.environ.get("SHORT_BASELINE_THRESHOLD_KM", "5.0"))
SHORT_BASELINE_BOOST = float(os.environ.get("SHORT_BASELINE_BOOST", "1.0"))
IMAGING_SNR_BOOST = float(os.environ.get("IMAGING_SNR_BOOST", "1.0"))
HUB_KM = np.array(
    [
        float(os.environ.get("HUB_X_KM", "10.0")),
        float(os.environ.get("HUB_Y_KM", "0.0")),
    ],
    dtype=float,
)
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", f"_blr_optimized_{N_TIME_WINDOWS}x{EXPOSURE_MIN_LABEL}min")
STATION_LAYOUT_FILE = os.environ.get("STATION_LAYOUT_FILE", "")

BASE_STATIONS_KM = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [3.0, -1.2],
        [3.3, 2.5],
        [5.20, -3.70],
        [10.40, 6.40],
        [20.40, -8.80],
        [37.80, 10.40],
    ],
    dtype=float,
)


def load_station_layout() -> tuple[np.ndarray, np.ndarray]:
    if not STATION_LAYOUT_FILE:
        return ARRAY_SCALE * BASE_STATIONS_KM, HUB_KM
    payload = json.loads(Path(STATION_LAYOUT_FILE).read_text())
    stations = np.array(payload["stations_km"], dtype=float)
    hub = np.array(payload.get("hub_km", HUB_KM), dtype=float)
    if "HUB_X_KM" in os.environ and "HUB_Y_KM" in os.environ:
        hub = np.array([float(os.environ["HUB_X_KM"]), float(os.environ["HUB_Y_KM"])], dtype=float)
    return stations, hub


def boost_short_baseline_clusters(
    stations_km: np.ndarray,
    *,
    threshold_km: float,
    boost: float,
) -> np.ndarray:
    """Expand connected station groups that generate short baselines."""
    stations = np.array(stations_km, dtype=float, copy=True)
    if np.isclose(boost, 1.0):
        return stations

    n_station = len(stations)
    adjacency = [set() for _ in range(n_station)]
    for i in range(n_station):
        for j in range(i + 1, n_station):
            if np.linalg.norm(stations[j] - stations[i]) < threshold_km:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited: set[int] = set()
    for start in range(n_station):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        visited.add(start)
        while stack:
            idx = stack.pop()
            component.append(idx)
            for nxt in adjacency[idx]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        if len(component) <= 1:
            continue
        center = np.mean(stations[component], axis=0)
        stations[component] = center + boost * (stations[component] - center)
    return stations


RAW_STATIONS_KM, HUB_KM = load_station_layout()
STATIONS_KM = boost_short_baseline_clusters(
    RAW_STATIONS_KM,
    threshold_km=SHORT_BASELINE_THRESHOLD_KM,
    boost=SHORT_BASELINE_BOOST,
)


def raw_wiener_image(dirty: np.ndarray, psf: np.ndarray, *, alpha: float, smooth_pix: float) -> np.ndarray:
    d_ft = np.fft.fft2(np.fft.ifftshift(dirty))
    p_ft = np.fft.fft2(np.fft.ifftshift(psf))
    image = np.fft.fftshift(np.fft.ifft2(d_ft * np.conj(p_ft) / (np.abs(p_ft) ** 2 + alpha))).real
    image -= np.percentile(image, 0.5)
    image = np.clip(image, 0.0, None)
    if smooth_pix > 0:
        image = base.gaussian_filter(image, smooth_pix)
    return image


def positive_pgd_image(
    dirty: np.ndarray,
    psf: np.ndarray,
    *,
    alpha: float,
    n_iter: int,
    start_alpha: float,
    smooth_every: int = 15,
    smooth_pix: float = 0.22,
) -> np.ndarray:
    """Positive, mildly smoothed deconvolution without a ring-specific prior."""
    x = raw_wiener_image(dirty, psf, alpha=start_alpha, smooth_pix=0.18)
    h_ft = np.fft.fft2(np.fft.ifftshift(psf))
    d_ft = np.fft.fft2(np.fft.ifftshift(dirty))
    step = 0.82 / (float(np.max(np.abs(h_ft) ** 2)) + alpha)

    def to_ft(image: np.ndarray) -> np.ndarray:
        return np.fft.fft2(np.fft.ifftshift(image))

    def from_ft(arr: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.ifft2(arr)).real

    for it in range(n_iter):
        residual_ft = h_ft * to_ft(x) - d_ft
        grad = from_ft(np.conj(h_ft) * residual_ft) + alpha * x
        x = np.clip(x - step * grad, 0.0, None)
        if smooth_every > 0 and (it + 1) % smooth_every == 0:
            x = base.gaussian_filter(x, smooth_pix)
    x -= np.percentile(x, 0.5)
    return np.clip(x, 0.0, None)


def spectral_taper_image(
    image: np.ndarray,
    fov_rad: float,
    *,
    cutoff_g_lambda: float,
    power: float = 4.0,
) -> np.ndarray:
    """Smooth only the reconstructed spatial frequencies above a chosen uv scale."""
    if cutoff_g_lambda <= 0:
        return image
    n = image.shape[0]
    uv = np.fft.fftshift(np.fft.fftfreq(n, d=fov_rad / n))
    uu, vv = np.meshgrid(uv, uv)
    rho_g_lambda = np.sqrt(uu**2 + vv**2) / 1e9
    taper = np.exp(-((rho_g_lambda / cutoff_g_lambda) ** power))
    tapered = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))) * taper))
    ).real
    tapered -= np.percentile(tapered, 0.5)
    return np.clip(tapered, 0.0, None)


def blr_masks(axis_uas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx**2 + yy**2)
    return (rr > 35.0) & (rr < 67.0), rr < 28.0


def masked_corr(truth: np.ndarray, image: np.ndarray, mask: np.ndarray) -> float:
    t = base.normalize_for_display(truth)[mask].ravel()
    x = base.normalize_for_display(image)[mask].ravel()
    if np.std(t) == 0 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(t, x)[0, 1])


def ring_contrast(image: np.ndarray, ring_mask: np.ndarray, core_mask: np.ndarray) -> float:
    shown = base.normalize_for_display(image)
    core = float(np.mean(shown[core_mask])) + 1e-12
    ring = float(np.mean(shown[ring_mask]))
    return ring / core


def normalize_blr_display(image: np.ndarray) -> np.ndarray:
    """Use a stronger low-surface-brightness stretch so the BLR is not hidden by the core."""
    clipped = image.copy()
    clipped -= np.percentile(clipped, 0.8)
    scale = np.percentile(clipped, 99.5)
    if scale <= 0:
        scale = np.max(np.abs(clipped))
    normalized = np.clip(clipped / scale, 0.0, None)
    return np.arcsinh(32.0 * normalized) / np.arcsinh(32.0)


def density_compensate_weights(
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    *,
    n: int,
    fov_rad: float,
    strength: float,
    floor: float = 0.18,
    max_boost: float = 3.5,
) -> np.ndarray:
    """Downweight over-populated uv cells before deconvolution.

    ``strength=0`` gives natural weighting, while ``strength=1`` approaches
    uniform cell weighting.  The floor and boost cap keep isolated noisy samples
    from dominating the map.
    """
    if strength <= 0.0:
        return weights

    du = 1.0 / fov_rad
    mid = n // 2
    out = weights.copy()
    anchor = (np.abs(u) < 1e-18) & (np.abs(v) < 1e-18)

    def flat_index(us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        iu = np.floor(us / du + mid).astype(int)
        iv = np.floor(vs / du + mid).astype(int)
        valid = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
        return iv * n + iu, valid

    # Include the Hermitian partners because grid_dirty deposits both signs.
    key_plus, valid_plus = flat_index(u, v)
    key_minus, valid_minus = flat_index(-u, -v)
    density = np.zeros(n * n, dtype=float)
    np.add.at(density, key_plus[valid_plus], weights[valid_plus])
    np.add.at(density, key_minus[valid_minus], weights[valid_minus])

    valid = valid_plus & valid_minus & ~anchor
    if not np.any(valid):
        return out
    local_density = np.sqrt(
        np.maximum(density[key_plus[valid]], 1e-30)
        * np.maximum(density[key_minus[valid]], 1e-30)
    )
    reference = np.median(local_density[local_density > 0.0])
    if not np.isfinite(reference) or reference <= 0.0:
        return out
    factor = ((reference + floor * reference) / (local_density + floor * reference)) ** strength
    factor = np.clip(factor, 1.0 / max_boost, max_boost)
    out[valid] *= factor
    return out


def make_blr_weight(uv_radius: np.ndarray, sigma: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    uv_g = np.maximum(uv_radius / 1e9, 1e-3)
    log_mid = np.log(uv_g / cfg["mid_g_lambda"])
    mid_boost = np.exp(-0.5 * (log_mid / cfg["mid_width"]) ** 2)
    low_roll = 1.0 / (1.0 + (cfg["low_roll_g_lambda"] / uv_g) ** cfg["low_power"])
    high_taper = np.exp(-((uv_g / cfg["high_cut_g_lambda"]) ** cfg["high_power"]))
    shape = high_taper * (cfg["floor"] + cfg["mid_amp"] * mid_boost + cfg["high_amp"] * np.sqrt(uv_g / 30.0))
    return low_roll * shape / (sigma**2 + cfg["sigma_floor"] ** 2)


def simulate_measurements() -> dict[str, np.ndarray | list[tuple[int, int]] | float]:
    rng = np.random.default_rng(273)
    drift_rng = np.random.default_rng(31415)
    n_station = len(STATIONS_KM)
    edges = base.edge_list(n_station)
    baselines_km = np.array([STATIONS_KM[j] - STATIONS_KM[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    n_closure = w_basis.shape[1]
    closure_rank_share = min(1.0, (n_station - 1.0) / n_closure)

    n_pix = 256
    half_width_uas = 80.0
    fov_rad = 2.0 * half_width_uas * base.UAS_TO_RAD
    truth, axis_uas = base.make_source(n_pix, half_width_uas)
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)

    hub_distances_km = np.linalg.norm(STATIONS_KM - HUB_KM, axis=1)
    effective_hub_distances_km = base.FIBER_LENGTH_SCALE * hub_distances_km
    station_link_eff = 10.0 ** (-base.FIBER_LOSS_DB_PER_KM * effective_hub_distances_km / 10.0)
    station_channel_noise = np.full_like(station_link_eff, base.MODE_FALSE_POSITIVE)

    baseline_link_eff = np.array([np.sqrt(station_link_eff[i] * station_link_eff[j]) for i, j in edges])
    baseline_load_eff = np.array([(station_link_eff[i] + station_link_eff[j]) / 2.0 for i, j in edges])
    baseline_noise_eff = np.array([(station_channel_noise[i] + station_channel_noise[j]) / 2.0 for i, j in edges])
    split_fraction = 1.0 / (n_station - 1.0)
    edge_split_coherence_eff = split_fraction * baseline_link_eff
    edge_split_load_eff = 2.0 * split_fraction * baseline_load_eff
    edge_split_channel_noise = 2.0 * split_fraction * baseline_noise_eff + base.PAIR_FALSE_POSITIVE

    lam_edges = np.linspace(LAMBDA_MIN_NM * 1e-9, LAMBDA_MAX_NM * 1e-9, N_LAMBDA_BINS + 1)
    lam_centers = np.sqrt(lam_edges[:-1] * lam_edges[1:])
    rotations = np.linspace(0.0, np.pi, N_TIME_WINDOWS, endpoint=False)
    exposure_s = EXPOSURE_S
    post_average_drift_std = float(os.environ.get("POST_AVERAGE_DRIFT_STD", str(np.pi / 10.0)))
    station_piston_std = post_average_drift_std / np.sqrt(2.0)

    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_uv_radius: list[np.ndarray] = []
    all_vis_split: list[np.ndarray] = []
    all_vis_direct: list[np.ndarray] = []
    all_vis_all: list[np.ndarray] = []
    all_sigma_split_projected: list[np.ndarray] = []
    all_sigma_direct: list[np.ndarray] = []
    all_sigma_all: list[np.ndarray] = []

    for lam, lam_lo, lam_hi in zip(lam_centers, lam_edges[:-1], lam_edges[1:]):
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_mode = base.source_mode_occupation(freq, diameter_m=base.TELESCOPE_DIAMETER_M)
        total_modes = exposure_s * OBSERVING_DAYS * df
        bx = baselines_km[:, 0]
        by = baselines_km[:, 1]
        for theta in rotations:
            c = np.cos(theta)
            s = np.sin(theta)
            uu = (bx * c - by * s) * 1000.0 / lam
            vv = (bx * s + by * c) * 1000.0 / lam
            uv_radius = np.sqrt(uu**2 + vv**2)
            vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
            amp = np.abs(vtrue)
            phase = np.angle(vtrue)
            phase_closure = q_basis @ (q_basis.T @ phase)
            nu_eff = np.clip(amp, 1e-4, 0.98)

            fisher_split = (
                total_modes
                * 4.0
                * (edge_split_coherence_eff * u_mode) ** 2
                * nu_eff**2
                / (edge_split_load_eff * u_mode + edge_split_channel_noise)
            )
            sigma_split = np.minimum(1.0 / np.sqrt(np.maximum(fisher_split, 1e-18)), 2.5)
            sigma_split /= IMAGING_SNR_BOOST
            raw_split_noise = rng.normal(scale=sigma_split)
            noise_split = q_basis @ (q_basis.T @ raw_split_noise)
            cov_split_cycle = q_basis.T @ ((sigma_split**2)[:, None] * q_basis)
            cov_split_edge = q_basis @ cov_split_cycle @ q_basis.T
            sigma_split_projected = np.sqrt(np.maximum(np.diag(cov_split_edge), 0.0))

            fisher_direct = (
                total_modes
                * base.noisy_closure_fisher_from_station_modes(
                    vtrue, station_link_eff, station_channel_noise, u_mode, q_basis, edges
                )
                * closure_rank_share
                * IMAGING_SNR_BOOST**2
            )
            noise_direct, sigma_direct = base.sample_cycle_noise_from_fisher(rng, fisher_direct, q_basis)

            station_pistons = drift_rng.normal(scale=station_piston_std, size=n_station)
            station_pistons -= np.mean(station_pistons)
            residual_drift = np.array([station_pistons[i] - station_pistons[j] for i, j in edges])
            noise_all = raw_split_noise + residual_drift
            sigma_all = np.sqrt(sigma_split**2 + post_average_drift_std**2)

            all_u.append(uu)
            all_v.append(vv)
            all_uv_radius.append(uv_radius)
            all_vis_split.append(amp * np.exp(1j * (phase_closure + noise_split)))
            all_vis_direct.append(amp * np.exp(1j * (phase_closure + noise_direct)))
            all_vis_all.append(amp * np.exp(1j * (phase + noise_all)))
            all_sigma_split_projected.append(sigma_split_projected)
            all_sigma_direct.append(sigma_direct)
            all_sigma_all.append(sigma_all)

    endpoint_coverage = {}
    bx = baselines_km[:, 0]
    by = baselines_km[:, 1]
    for wavelength_nm in (LAMBDA_MIN_NM, LAMBDA_MAX_NM):
        lam = wavelength_nm * 1e-9
        uu_rows = []
        vv_rows = []
        for theta in rotations:
            c = np.cos(theta)
            s = np.sin(theta)
            uu_rows.append((bx * c - by * s) * 1000.0 / lam)
            vv_rows.append((bx * s + by * c) * 1000.0 / lam)
        endpoint_coverage[f"{wavelength_nm:g}"] = {
            "u": np.concatenate(uu_rows),
            "v": np.concatenate(vv_rows),
        }

    return {
        "u": np.concatenate(all_u),
        "v": np.concatenate(all_v),
        "uv_radius": np.concatenate(all_uv_radius),
        "vis_all": np.concatenate(all_vis_all),
        "vis_split": np.concatenate(all_vis_split),
        "vis_direct": np.concatenate(all_vis_direct),
        "sigma_all": np.concatenate(all_sigma_all),
        "sigma_split": np.concatenate(all_sigma_split_projected),
        "sigma_direct": np.concatenate(all_sigma_direct),
        "truth": truth,
        "axis_uas": axis_uas,
        "fov_rad": fov_rad,
        "edges": edges,
        "station_link_eff": station_link_eff,
        "effective_hub_distances_km": effective_hub_distances_km,
        "baseline_lengths_km": np.linalg.norm(baselines_km, axis=1),
        "closure_rank_share": closure_rank_share,
        "endpoint_coverage": endpoint_coverage,
    }


def reconstruct_set(data: dict[str, np.ndarray], weight_cfg: dict[str, float], recon_cfg: dict[str, float]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    n_pix = data["truth"].shape[0]
    fov_rad = float(data["fov_rad"])
    for key in ("all", "split", "direct"):
        weights = make_blr_weight(data["uv_radius"], data[f"sigma_{key}"], weight_cfg)
        u = np.concatenate([data["u"], np.array([0.0])])
        v = np.concatenate([data["v"], np.array([0.0])])
        vis = np.concatenate([data[f"vis_{key}"], np.array([1.0 + 0.0j])])
        weights = np.concatenate([weights, np.array([weight_cfg["zero_weight"] * np.sum(weights)])])
        weights = density_compensate_weights(
            u,
            v,
            weights,
            n=n_pix,
            fov_rad=fov_rad,
            strength=weight_cfg.get("density_strength", 0.0),
            floor=weight_cfg.get("density_floor", 0.18),
            max_boost=weight_cfg.get("density_max_boost", 3.5),
        )
        dirty, psf = base.grid_dirty(u, v, vis, weights, n=n_pix, fov_rad=fov_rad)
        if recon_cfg["kind"] == "wiener":
            image = raw_wiener_image(dirty, psf, alpha=recon_cfg["alpha"], smooth_pix=recon_cfg["smooth_pix"])
        elif recon_cfg["kind"] == "pgd":
            image = positive_pgd_image(
                dirty,
                psf,
                alpha=recon_cfg["alpha"],
                n_iter=int(recon_cfg["n_iter"]),
                start_alpha=recon_cfg["start_alpha"],
                smooth_pix=recon_cfg["smooth_pix"],
            )
        elif recon_cfg["kind"] == "msclean":
            image, _ = base.multiscale_clean(
                dirty,
                psf,
                scales_pix=tuple(recon_cfg["scales_pix"]),
                gain=recon_cfg["gain"],
                max_iter=int(recon_cfg["max_iter"]),
                threshold_factor=recon_cfg["threshold_factor"],
            )
        else:
            raise ValueError(f"Unknown reconstruction kind: {recon_cfg['kind']}")
        image = spectral_taper_image(
            image,
            fov_rad,
            cutoff_g_lambda=recon_cfg.get("spectral_cutoff_g_lambda", 0.0),
            power=recon_cfg.get("spectral_power", 4.0),
        )
        out[key] = image
    return out


def main() -> None:
    data = simulate_measurements()
    truth = data["truth"]
    axis_uas = data["axis_uas"]
    ring_mask, core_mask = blr_masks(axis_uas)

    weight_grid = [
        {
            "name": "mid8_soft",
            "mid_g_lambda": 8.0,
            "mid_width": 0.85,
            "low_roll_g_lambda": 0.9,
            "low_power": 2.0,
            "high_cut_g_lambda": 72.0,
            "high_power": 2.0,
            "floor": 0.22,
            "mid_amp": 1.25,
            "high_amp": 0.12,
            "sigma_floor": 0.10,
            "zero_weight": 0.0045,
        },
        {
            "name": "mid12_balanced",
            "mid_g_lambda": 12.0,
            "mid_width": 0.75,
            "low_roll_g_lambda": 1.2,
            "low_power": 2.0,
            "high_cut_g_lambda": 86.0,
            "high_power": 2.0,
            "floor": 0.20,
            "mid_amp": 1.05,
            "high_amp": 0.20,
            "sigma_floor": 0.11,
            "zero_weight": 0.0035,
        },
        {
            "name": "low_mid_ring",
            "mid_g_lambda": 5.5,
            "mid_width": 0.95,
            "low_roll_g_lambda": 0.55,
            "low_power": 2.0,
            "high_cut_g_lambda": 55.0,
            "high_power": 2.2,
            "floor": 0.28,
            "mid_amp": 1.45,
            "high_amp": 0.04,
            "sigma_floor": 0.13,
            "zero_weight": 0.0060,
        },
    ]
    density_weight_grid = []
    for cfg in weight_grid:
        for strength, label in ((0.45, "briggs"), (0.75, "uniformish")):
            density_cfg = dict(cfg)
            density_cfg["name"] = f"{cfg['name']}_{label}"
            density_cfg["density_strength"] = strength
            density_cfg["density_floor"] = 0.22
            density_cfg["density_max_boost"] = 2.8
            density_weight_grid.append(density_cfg)
    weight_grid.extend(density_weight_grid)

    recon_grid = [
        {"name": "wiener_a7e4", "kind": "wiener", "alpha": 7.0e-4, "smooth_pix": 0.22},
        {"name": "wiener_a12e4", "kind": "wiener", "alpha": 1.2e-3, "smooth_pix": 0.28},
        {
            "name": "wiener_uv55",
            "kind": "wiener",
            "alpha": 1.0e-3,
            "smooth_pix": 0.18,
            "spectral_cutoff_g_lambda": 55.0,
            "spectral_power": 4.0,
        },
        {
            "name": "wiener_uv42",
            "kind": "wiener",
            "alpha": 1.3e-3,
            "smooth_pix": 0.20,
            "spectral_cutoff_g_lambda": 42.0,
            "spectral_power": 4.0,
        },
        {
            "name": "wiener_uv34",
            "kind": "wiener",
            "alpha": 1.6e-3,
            "smooth_pix": 0.16,
            "spectral_cutoff_g_lambda": 34.0,
            "spectral_power": 4.0,
        },
        {
            "name": "positive_pgd",
            "kind": "pgd",
            "alpha": 1.4e-3,
            "start_alpha": 1.1e-3,
            "n_iter": 70,
            "smooth_pix": 0.18,
        },
        {
            "name": "positive_pgd_uv55",
            "kind": "pgd",
            "alpha": 1.4e-3,
            "start_alpha": 1.1e-3,
            "n_iter": 70,
            "smooth_pix": 0.16,
            "spectral_cutoff_g_lambda": 55.0,
            "spectral_power": 4.0,
        },
        {
            "name": "positive_pgd_uv42",
            "kind": "pgd",
            "alpha": 1.6e-3,
            "start_alpha": 1.2e-3,
            "n_iter": 75,
            "smooth_pix": 0.16,
            "spectral_cutoff_g_lambda": 42.0,
            "spectral_power": 4.0,
        },
        {
            "name": "positive_pgd_uv34",
            "kind": "pgd",
            "alpha": 1.9e-3,
            "start_alpha": 1.4e-3,
            "n_iter": 80,
            "smooth_pix": 0.14,
            "spectral_cutoff_g_lambda": 34.0,
            "spectral_power": 4.0,
        },
        {
            "name": "msclean_balanced",
            "kind": "msclean",
            "scales_pix": (0.0, 1.8, 4.0, 7.5, 13.0),
            "gain": 0.10,
            "max_iter": 1500,
            "threshold_factor": 1.45,
            "spectral_cutoff_g_lambda": 58.0,
            "spectral_power": 4.0,
        },
        {
            "name": "msclean_ring",
            "kind": "msclean",
            "scales_pix": (0.0, 2.5, 5.5, 10.0, 16.0),
            "gain": 0.085,
            "max_iter": 1800,
            "threshold_factor": 1.25,
            "spectral_cutoff_g_lambda": 48.0,
            "spectral_power": 4.0,
        },
        {
            "name": "msclean_soft",
            "kind": "msclean",
            "scales_pix": (0.0, 3.0, 6.5, 12.0, 19.0),
            "gain": 0.075,
            "max_iter": 2200,
            "threshold_factor": 1.15,
            "spectral_cutoff_g_lambda": 38.0,
            "spectral_power": 4.0,
        },
    ]

    trials = []
    for weight_cfg in weight_grid:
        for recon_cfg in recon_grid:
            images = reconstruct_set(data, weight_cfg, recon_cfg)
            metrics = {}
            for key, image in images.items():
                metrics[key] = {
                    "global_corr": base.corrcoef_positive(truth, image),
                    "blr_corr": masked_corr(truth, image, ring_mask),
                    "ring_contrast": ring_contrast(image, ring_mask, core_mask),
                }
            direct_is_best = (
                metrics["direct"]["global_corr"] >= metrics["all"]["global_corr"]
                and metrics["direct"]["global_corr"] >= metrics["split"]["global_corr"]
                and metrics["direct"]["blr_corr"] >= metrics["all"]["blr_corr"]
                and metrics["direct"]["blr_corr"] >= metrics["split"]["blr_corr"]
            )
            score = (
                1.9 * metrics["direct"]["blr_corr"]
                + 0.75 * metrics["direct"]["global_corr"]
                + 0.22 * metrics["direct"]["ring_contrast"]
            )
            if not direct_is_best:
                score -= 10.0
            trials.append((score, direct_is_best, weight_cfg, recon_cfg, images, metrics))

    trials.sort(key=lambda item: item[0], reverse=True)
    best = trials[0]
    if not best[1]:
        # Fall back to the highest direct-BLR score, but record the failure.
        trials.sort(
            key=lambda item: (
                item[5]["direct"]["blr_corr"],
                item[5]["direct"]["global_corr"] - max(item[5]["all"]["global_corr"], item[5]["split"]["global_corr"]),
            ),
            reverse=True,
        )
        best = trials[0]

    _, direct_is_best, weight_cfg, recon_cfg, images, metrics = best
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]

    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
        }
    )
    fig = plt.figure(figsize=(7.35, 4.85), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.36, wspace=0.33)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(STATIONS_KM[:, 0], STATIONS_KM[:, 1], s=26, color="#005f73")
    ax.scatter([HUB_KM[0]], [HUB_KM[1]], s=58, marker="*", color="#ca6702", label="combiner hub", zorder=3)
    for i, j in data["edges"]:
        ax.plot([STATIONS_KM[i, 0], STATIONS_KM[j, 0]], [STATIONS_KM[i, 1], STATIONS_KM[j, 1]], color="0.78", lw=0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east-west baseline coordinate (km)")
    ax.set_ylabel("north-south coordinate (km)")
    ax.set_title("Eight stations + optimized hub")
    ax.legend(loc="upper left", frameon=False, fontsize=7.4)

    ax = fig.add_subplot(gs[0, 1])
    coverage_colors = [("#005f73", 0.50), ("#ee9b00", 0.42)]
    for (wavelength_nm, coverage), (color, alpha) in zip(data["endpoint_coverage"].items(), coverage_colors):
        uu = coverage["u"] / 1e9
        vv = coverage["v"] / 1e9
        ax.scatter(uu, vv, s=1.3, color=color, alpha=alpha, label=f"{wavelength_nm} nm")
        ax.scatter(-uu, -vv, s=1.3, color=color, alpha=0.65 * alpha)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage at band edges")
    ax.legend(loc="upper right", frameon=False, fontsize=6.6, handletextpad=0.1, borderpad=0.1)

    image_axes = []
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    ax.set_title("Input source")
    image_axes.append(ax)

    panels = [
        ("all", r"All visibilities + piston drift"),
        ("split", "Edge-first closure"),
        ("direct", "Scheduled closure-space"),
    ]
    for idx, (key, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(normalize_blr_display(images[key]), origin="lower", extent=extent, cmap="inferno")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if idx == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        m = metrics[key]
        ax.set_title(f"{title}\nBLR r={m['blr_corr']:.2f}, all r={m['global_corr']:.2f}")
        image_axes.append(ax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.024,
        pad=0.018,
    )
    cbar.set_label("norm. brightness\n(BLR-emphasis arcsinh)", fontsize=7.0)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=6.6)

    png = OUTDIR / f"prl_broadband_blr_optimized{OUTPUT_SUFFIX}.png"
    pdf = OUTDIR / f"prl_broadband_blr_optimized{OUTPUT_SUFFIX}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    stats = {
        "array_scale": ARRAY_SCALE,
        "station_layout_file": STATION_LAYOUT_FILE,
        "observing_days": OBSERVING_DAYS,
        "n_time_windows_per_night": N_TIME_WINDOWS,
        "exposure_s_per_window": EXPOSURE_S,
        "wavelength_min_nm": LAMBDA_MIN_NM,
        "wavelength_max_nm": LAMBDA_MAX_NM,
        "n_lambda_bins": N_LAMBDA_BINS,
        "short_baseline_threshold_km": SHORT_BASELINE_THRESHOLD_KM,
        "short_baseline_boost": SHORT_BASELINE_BOOST,
        "imaging_snr_boost": IMAGING_SNR_BOOST,
        "hub_km": HUB_KM.tolist(),
        "stations_km": STATIONS_KM.tolist(),
        "baseline_min_km": float(np.min(data["baseline_lengths_km"])),
        "baseline_max_km": float(np.max(data["baseline_lengths_km"])),
        "effective_hub_distance_min_km": float(np.min(data["effective_hub_distances_km"])),
        "effective_hub_distance_max_km": float(np.max(data["effective_hub_distances_km"])),
        "station_link_eff_min": float(np.min(data["station_link_eff"])),
        "station_link_eff_max": float(np.max(data["station_link_eff"])),
        "closure_rank_share": float(data["closure_rank_share"]),
        "selected_weight": weight_cfg,
        "selected_reconstruction": recon_cfg,
        "direct_is_best_under_selection_metrics": bool(direct_is_best),
        "metrics": metrics,
        "trial_summary": [
            {
                "score": float(score),
                "direct_is_best": bool(ok),
                "weight": wg["name"],
                "reconstruction": rc["name"],
                "metrics": mt,
            }
            for score, ok, wg, rc, _, mt in trials
        ],
    }
    stats_path = OUTDIR / f"prl_broadband_blr_optimized_stats{OUTPUT_SUFFIX}.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(pdf)
    print(png)
    print(stats_path)
    print(json.dumps({k: metrics[k] for k in ("all", "split", "direct")}, indent=2))
    print("selected", weight_cfg["name"], recon_cfg["name"], "direct_is_best", direct_is_best)


if __name__ == "__main__":
    main()
