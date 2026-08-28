from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_prl_broadband_clean as base
import plot_prl_broadband_blr_optimized as opt
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

LAYOUT_FILE = Path(os.environ.get("WEIGHT_LAYOUT_FILE", "output/figures/blue_noise_uv80v50_realnight_layout.json"))
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "_uv_weighting_v50")
ARRAY_LAT_DEG = float(os.environ.get("ARRAY_LAT_DEG", "35.0"))
SOURCE_DEC_DEG = float(os.environ.get("SOURCE_DEC_DEG", "2.052388"))
N_TIME_WINDOWS = int(os.environ.get("N_TIME_WINDOWS", "36"))
EXPOSURE_S = float(os.environ.get("EXPOSURE_S", "600.0"))
EXPOSURE_GAP_S = float(os.environ.get("EXPOSURE_GAP_S", "300.0"))
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "60"))
LAMBDA_MIN_NM = float(os.environ.get("LAMBDA_MIN_NM", "400.0"))
LAMBDA_MAX_NM = float(os.environ.get("LAMBDA_MAX_NM", "800.0"))
N_LAMBDA_BINS = int(os.environ.get("N_LAMBDA_BINS", "36"))
HUB_X_KM = float(os.environ.get("HUB_X_KM", "2.0"))
HUB_Y_KM = float(os.environ.get("HUB_Y_KM", "0.0"))
SIGMA_FLOOR = float(os.environ.get("WEIGHT_SIGMA_FLOOR", "0.11"))
COARSE_UV_BINS = int(os.environ.get("COARSE_UV_BINS", "60"))
COARSE_UV_BIN_MODE = os.environ.get("COARSE_UV_BIN_MODE", "adaptive").lower()
COARSE_UV_RES_FACTOR = float(os.environ.get("COARSE_UV_RES_FACTOR", "1.0"))
COARSE_SMOOTH_CELLS = float(os.environ.get("COARSE_SMOOTH_CELLS", "0.9"))


def load_layout() -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((ROOT / LAYOUT_FILE).read_text())
    stations = np.array(payload["stations_km"], dtype=float)
    hub = np.array([HUB_X_KM, HUB_Y_KM], dtype=float)
    return stations, hub


def cell_density_compensation(
    u: np.ndarray,
    v: np.ndarray,
    base_weights: np.ndarray,
    *,
    n: int,
    fov_rad: float,
    mode: str,
) -> np.ndarray:
    """Return uv-area compensated weights for irregular Fourier samples.

    ``cell_count`` gives each occupied uv cell equal total count weight.
    ``cell_weight`` gives each occupied uv cell equal total inverse-variance
    weight, preserving only relative weights of samples within the same cell.
    Both include Hermitian partners because ``grid_dirty`` deposits both signs.
    """
    if mode == "natural":
        return base_weights.copy()

    du = 1.0 / fov_rad
    mid = n // 2

    def keys(us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        iu = np.floor(us / du + mid).astype(int)
        iv = np.floor(vs / du + mid).astype(int)
        valid = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
        return iv * n + iu, valid

    key_p, valid_p = keys(u, v)
    key_m, valid_m = keys(-u, -v)
    valid = valid_p & valid_m
    out = np.zeros_like(base_weights)
    if not np.any(valid):
        return base_weights.copy()

    if mode == "cell_count":
        density = np.zeros(n * n, dtype=float)
        np.add.at(density, key_p[valid], 1.0)
        np.add.at(density, key_m[valid], 1.0)
        local = 0.5 * (density[key_p[valid]] + density[key_m[valid]])
        out[valid] = base_weights[valid] / np.maximum(local, 1.0)
        return out

    if mode == "cell_weight":
        density = np.zeros(n * n, dtype=float)
        np.add.at(density, key_p[valid], base_weights[valid])
        np.add.at(density, key_m[valid], base_weights[valid])
        local = 0.5 * (density[key_p[valid]] + density[key_m[valid]])
        occupied = density[density > 0.0]
        target = np.median(occupied) if len(occupied) else 1.0
        out[valid] = base_weights[valid] * target / np.maximum(local, 1e-30)
        return out

    if mode == "cell_weight_capped":
        weighted = cell_density_compensation(u, v, base_weights, n=n, fov_rad=fov_rad, mode="cell_weight")
        ratio = weighted / np.maximum(base_weights, 1e-30)
        ratio = np.clip(ratio, 0.25, 4.0)
        return base_weights * ratio

    raise ValueError(f"Unknown weighting mode {mode!r}")


def uv_cell_indices(
    u: np.ndarray,
    v: np.ndarray,
    *,
    n: int,
    fov_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    du = 1.0 / fov_rad
    mid = n // 2
    iu = np.floor(u / du + mid).astype(int)
    iv = np.floor(v / du + mid).astype(int)
    valid = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
    cell_key = iv * n + iu
    cell_u = (iu - mid) * du
    cell_v = (iv - mid) * du
    return cell_key, cell_u, cell_v, valid


def aggregate_to_uv_cells(
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    sigma: np.ndarray,
    *,
    n: int,
    fov_rad: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average irregular samples into Fourier cells before inversion.

    This is the closest discrete analogue of assigning one quadrature weight to
    each occupied uv-area element.  The samples inside a cell are first combined
    into a single visibility estimate; the final Fourier weight of each cell is
    then set to one, independent of how many raw samples landed there.
    """
    key, cell_u, cell_v, valid = uv_cell_indices(u, v, n=n, fov_rad=fov_rad)
    if not np.any(valid):
        raise ValueError("No uv samples fall inside the image Fourier grid")

    key = key[valid]
    cell_u = cell_u[valid]
    cell_v = cell_v[valid]
    vis = vis[valid]
    sigma = sigma[valid]

    unique, inverse = np.unique(key, return_inverse=True)
    n_cell = len(unique)
    sum_weight = np.zeros(n_cell, dtype=float)
    sum_vis = np.zeros(n_cell, dtype=complex)
    count = np.zeros(n_cell, dtype=float)
    first_u = np.zeros(n_cell, dtype=float)
    first_v = np.zeros(n_cell, dtype=float)

    if mode == "cell_mean":
        sample_weight = np.ones_like(sigma)
    elif mode in {"cell_ivar_equal", "cell_ivar_briggs"}:
        sample_weight = 1.0 / (sigma**2 + SIGMA_FLOOR**2)
    else:
        raise ValueError(f"Unknown cell aggregation mode {mode!r}")

    np.add.at(sum_weight, inverse, sample_weight)
    np.add.at(sum_vis, inverse, sample_weight * vis)
    np.add.at(count, inverse, 1.0)
    # Cell centers are identical for all samples in a cell; averaging is robust
    # against any accidental duplicate representation.
    np.add.at(first_u, inverse, cell_u)
    np.add.at(first_v, inverse, cell_v)

    u_cell = first_u / count
    v_cell = first_v / count
    vis_cell = sum_vis / np.maximum(sum_weight, 1e-30)

    if mode in {"cell_mean", "cell_ivar_equal"}:
        cell_weights = np.ones(n_cell, dtype=float)
    else:
        reference = np.median(sum_weight[sum_weight > 0.0])
        # Briggs-like compromise: sparse/noisy cells keep reduced weight, while
        # densely sampled cells saturate instead of dominating by sample count.
        cell_weights = sum_weight / (1.0 + sum_weight / max(reference, 1e-30))
        cell_weights /= max(np.median(cell_weights[cell_weights > 0.0]), 1e-30)
    return u_cell, v_cell, vis_cell, cell_weights


def coarse_uv_bin_counts(u: np.ndarray, v: np.ndarray, fov_rad: float) -> tuple[int, int]:
    """Choose rectangular uv bins from the image Fourier resolution.

    In adaptive mode the target cell width is COARSE_UV_RES_FACTOR/FOV in
    both directions, so Nu and Nv follow the actual peak horizontal and
    vertical uv extent instead of forcing a square 60 x 60 grid.
    """
    if COARSE_UV_BIN_MODE == "fixed":
        return COARSE_UV_BINS, COARSE_UV_BINS
    if COARSE_UV_BIN_MODE != "adaptive":
        raise ValueError("COARSE_UV_BIN_MODE must be 'adaptive' or 'fixed'.")
    du_target = COARSE_UV_RES_FACTOR / fov_rad
    n_u = max(4, int(np.ceil(2.0002 * float(np.max(np.abs(u))) / du_target)))
    n_v = max(4, int(np.ceil(2.0002 * float(np.max(np.abs(v))) / du_target)))
    return n_u, n_v


def aggregate_to_coarse_uv_grid(
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    sigma: np.ndarray,
    *,
    n_bin_u: int,
    n_bin_v: int,
    mode: str,
    smooth_cells: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average samples into a coarse uv grid and optionally smooth visibilities.

    This implements the user's suggested approximation to the Fourier area
    integral: divide the occupied k-space rectangle into a modest grid, estimate
    one visibility per coarse cell, and assign each occupied cell comparable
    Fourier-area weight.  The optional Gaussian smoothing acts on the coarse
    complex visibility grid, not on the final image.
    """
    max_u = float(np.max(np.abs(u)))
    max_v = float(np.max(np.abs(v)))
    u_edges = np.linspace(-1.0001 * max_u, 1.0001 * max_u, n_bin_u + 1)
    v_edges = np.linspace(-1.0001 * max_v, 1.0001 * max_v, n_bin_v + 1)
    iu = np.searchsorted(u_edges, u, side="right") - 1
    iv = np.searchsorted(v_edges, v, side="right") - 1
    valid = (iu >= 0) & (iu < n_bin_u) & (iv >= 0) & (iv < n_bin_v)
    iu = iu[valid]
    iv = iv[valid]
    vis = vis[valid]
    sigma = sigma[valid]

    if mode == "coarse_mean":
        sample_weight = np.ones_like(sigma)
    elif mode in {"coarse_ivar_equal", "coarse_ivar_briggs"}:
        sample_weight = 1.0 / (sigma**2 + SIGMA_FLOOR**2)
    else:
        raise ValueError(f"Unknown coarse mode {mode!r}")

    sum_weight = np.zeros((n_bin_v, n_bin_u), dtype=float)
    sum_vis = np.zeros((n_bin_v, n_bin_u), dtype=complex)
    count = np.zeros((n_bin_v, n_bin_u), dtype=float)
    np.add.at(sum_weight, (iv, iu), sample_weight)
    np.add.at(sum_vis, (iv, iu), sample_weight * vis)
    np.add.at(count, (iv, iu), 1.0)
    occupied = sum_weight > 0.0
    vis_grid = np.zeros_like(sum_vis)
    vis_grid[occupied] = sum_vis[occupied] / sum_weight[occupied]

    if smooth_cells > 0.0:
        smooth_w = base.gaussian_filter(sum_weight, smooth_cells)
        smooth_vis_num = base.gaussian_filter(sum_vis.real, smooth_cells) + 1j * base.gaussian_filter(
            sum_vis.imag, smooth_cells
        )
        smooth_occ = smooth_w > 0.05 * np.max(smooth_w)
        vis_grid = np.zeros_like(sum_vis)
        vis_grid[smooth_occ] = smooth_vis_num[smooth_occ] / np.maximum(smooth_w[smooth_occ], 1e-30)
        occupied = smooth_occ
        sum_weight = smooth_w

    vv_idx, uu_idx = np.nonzero(occupied)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    u_cell = u_centers[uu_idx]
    v_cell = v_centers[vv_idx]
    vis_cell = vis_grid[vv_idx, uu_idx]

    if mode in {"coarse_mean", "coarse_ivar_equal"}:
        cell_weights = np.ones_like(u_cell)
    else:
        raw = sum_weight[vv_idx, uu_idx]
        reference = np.median(raw[raw > 0.0])
        cell_weights = raw / (1.0 + raw / max(reference, 1e-30))
        cell_weights /= max(np.median(cell_weights[cell_weights > 0.0]), 1e-30)

    return u_cell, v_cell, vis_cell, cell_weights


def coarse_density_weights(
    u: np.ndarray,
    v: np.ndarray,
    base_weights: np.ndarray,
    *,
    n_bin_u: int,
    n_bin_v: int,
    mode: str,
    smooth_cells: float,
) -> np.ndarray:
    """Use a coarse uv grid only to estimate area-density weights.

    Unlike ``aggregate_to_coarse_uv_grid``, this keeps every visibility at its
    original coordinate, avoiding phase-slope errors from moving samples to a
    coarse cell center.
    """
    max_u = float(np.max(np.abs(u)))
    max_v = float(np.max(np.abs(v)))
    u_edges = np.linspace(-1.0001 * max_u, 1.0001 * max_u, n_bin_u + 1)
    v_edges = np.linspace(-1.0001 * max_v, 1.0001 * max_v, n_bin_v + 1)
    iu = np.searchsorted(u_edges, u, side="right") - 1
    iv = np.searchsorted(v_edges, v, side="right") - 1
    valid = (iu >= 0) & (iu < n_bin_u) & (iv >= 0) & (iv < n_bin_v)
    density = np.zeros((n_bin_v, n_bin_u), dtype=float)
    if mode == "coarse_density_count":
        sample_density = np.ones_like(base_weights)
    elif mode == "coarse_density_weight":
        sample_density = base_weights
    else:
        raise ValueError(f"Unknown coarse density mode {mode!r}")

    np.add.at(density, (iv[valid], iu[valid]), sample_density[valid])
    if smooth_cells > 0:
        density = base.gaussian_filter(density, smooth_cells)
    occupied = density[density > 0]
    reference = np.median(occupied) if len(occupied) else 1.0
    out = np.zeros_like(base_weights)
    local = density[iv[valid], iu[valid]]
    out[valid] = base_weights[valid] * reference / np.maximum(local, 1e-30)
    return out


def simulate_dataset(*, noisy: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(273)
    stations_km, hub_km = load_layout()
    n_station = len(stations_km)
    edges = base.edge_list(n_station)
    baselines_km = np.array([stations_km[j] - stations_km[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    closure_rank_share = min(1.0, (n_station - 1.0) / w_basis.shape[1])

    n_pix = 256
    half_width_uas = 80.0
    fov_rad = 2.0 * half_width_uas * base.UAS_TO_RAD
    truth, axis_uas = base.make_source(n_pix, half_width_uas)
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)

    hub_distances_km = np.linalg.norm(stations_km - hub_km, axis=1)
    effective_hub_distances_km = base.FIBER_LENGTH_SCALE * hub_distances_km
    station_link_eff = 10.0 ** (-base.FIBER_LOSS_DB_PER_KM * effective_hub_distances_km / 10.0)
    station_channel_noise = np.full_like(station_link_eff, base.MODE_FALSE_POSITIVE)

    lam_edges = np.linspace(LAMBDA_MIN_NM * 1e-9, LAMBDA_MAX_NM * 1e-9, N_LAMBDA_BINS + 1)
    lam_centers = np.sqrt(lam_edges[:-1] * lam_edges[1:])
    hour_angles = realnight_hour_angles(N_TIME_WINDOWS, EXPOSURE_S, EXPOSURE_GAP_S)

    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_vis: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []

    for lam, lam_lo, lam_hi in zip(lam_centers, lam_edges[:-1], lam_edges[1:]):
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_mode = base.mode_occupation_ab(base.SOURCE_AB_MAG, freq, diameter_m=base.TELESCOPE_DIAMETER_M)
        total_modes = EXPOSURE_S * OBSERVING_DAYS * df
        uu_rows, vv_rows = project_enu_baselines(
            baselines_km,
            hour_angles,
            lam,
            latitude_deg=ARRAY_LAT_DEG,
            declination_deg=SOURCE_DEC_DEG,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
            amp = np.abs(vtrue)
            phase = np.angle(vtrue)
            phase_closure = q_basis @ (q_basis.T @ phase)
            vis = amp * np.exp(1j * phase_closure)

            if noisy:
                fisher_direct = (
                    total_modes
                    * base.noisy_closure_fisher_from_station_modes(
                        vtrue, station_link_eff, station_channel_noise, u_mode, q_basis, edges
                    )
                    * closure_rank_share
                )
                noise_direct, sigma_direct = base.sample_cycle_noise_from_fisher(rng, fisher_direct, q_basis)
                vis = amp * np.exp(1j * (phase_closure + noise_direct))
                sigma = sigma_direct
            else:
                sigma = np.ones_like(amp)

            all_u.append(uu)
            all_v.append(vv)
            all_vis.append(vis)
            all_sigma.append(sigma)

    return {
        "u": np.concatenate(all_u),
        "v": np.concatenate(all_v),
        "vis": np.concatenate(all_vis),
        "sigma": np.concatenate(all_sigma),
        "truth": truth,
        "axis_uas": axis_uas,
        "fov_rad": np.array(fov_rad),
        "stations_km": stations_km,
        "hub_km": hub_km,
        "edges": np.array(edges),
        "baseline_lengths_km": np.linalg.norm(baselines_km, axis=1),
        "station_link_eff": station_link_eff,
    }


def reconstruct(data: dict[str, np.ndarray], mode: str, *, noisy: bool) -> np.ndarray:
    n_pix = data["truth"].shape[0]
    fov_rad = float(data["fov_rad"])
    sigma = data["sigma"]
    if noisy:
        base_weights = 1.0 / (sigma**2 + SIGMA_FLOOR**2)
    else:
        base_weights = np.ones_like(sigma)

    if mode in {"natural", "cell_count", "coarse_density_count", "coarse_density_weight"}:
        u = np.concatenate([data["u"], np.array([0.0])])
        v = np.concatenate([data["v"], np.array([0.0])])
        vis = np.concatenate([data["vis"], np.array([1.0 + 0.0j])])
        base_weights = np.concatenate([base_weights, np.array([0.0035 * np.sum(base_weights)])])
        if mode in {"coarse_density_count", "coarse_density_weight"}:
            n_bin_u, n_bin_v = coarse_uv_bin_counts(u, v, fov_rad)
            weights = coarse_density_weights(
                u,
                v,
                base_weights,
                n_bin_u=n_bin_u,
                n_bin_v=n_bin_v,
                mode=mode,
                smooth_cells=COARSE_SMOOTH_CELLS,
            )
        else:
            weights = cell_density_compensation(u, v, base_weights, n=n_pix, fov_rad=fov_rad, mode=mode)
    elif mode in {"cell_mean", "cell_ivar_equal", "cell_ivar_briggs"}:
        u, v, vis, weights = aggregate_to_uv_cells(
            data["u"],
            data["v"],
            data["vis"],
            sigma,
            n=n_pix,
            fov_rad=fov_rad,
            mode=mode,
        )
        weights = np.concatenate([weights, np.array([0.0035 * len(weights)])])
        u = np.concatenate([u, np.array([0.0])])
        v = np.concatenate([v, np.array([0.0])])
        vis = np.concatenate([vis, np.array([1.0 + 0.0j])])
    elif mode in {"coarse_mean", "coarse_ivar_equal", "coarse_ivar_briggs"}:
        n_bin_u, n_bin_v = coarse_uv_bin_counts(data["u"], data["v"], fov_rad)
        u, v, vis, weights = aggregate_to_coarse_uv_grid(
            data["u"],
            data["v"],
            data["vis"],
            sigma,
            n_bin_u=n_bin_u,
            n_bin_v=n_bin_v,
            mode=mode,
            smooth_cells=COARSE_SMOOTH_CELLS,
        )
        weights = np.concatenate([weights, np.array([0.0035 * len(weights)])])
        u = np.concatenate([u, np.array([0.0])])
        v = np.concatenate([v, np.array([0.0])])
        vis = np.concatenate([vis, np.array([1.0 + 0.0j])])
    else:
        raise ValueError(f"Unknown reconstruction weighting mode {mode!r}")

    dirty, psf = base.grid_dirty(u, v, vis, weights, n=n_pix, fov_rad=fov_rad)
    image = opt.raw_wiener_image(dirty, psf, alpha=1.2e-3, smooth_pix=0.18)
    image = opt.spectral_taper_image(image, fov_rad, cutoff_g_lambda=55.0, power=4.0)
    return image


def metrics(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> dict[str, float]:
    ring_mask, core_mask = opt.blr_masks(axis_uas)
    return {
        "global_corr": base.corrcoef_positive(truth, image),
        "blr_corr": opt.masked_corr(truth, image, ring_mask),
        "ring_contrast": opt.ring_contrast(image, ring_mask, core_mask),
    }


def main() -> None:
    modes = [
        "natural",
        "cell_count",
        "coarse_density_count",
        "coarse_density_weight",
        "coarse_ivar_briggs",
    ]
    noiseless = simulate_dataset(noisy=False)
    physical_snr = simulate_dataset(noisy=True)
    n_bin_u, n_bin_v = coarse_uv_bin_counts(
        np.concatenate([noiseless["u"], np.array([0.0])]),
        np.concatenate([noiseless["v"], np.array([0.0])]),
        float(noiseless["fov_rad"]),
    )
    coarse_label = f"{n_bin_u}x{n_bin_v}"
    labels = {
        "natural": "per-sample natural",
        "cell_count": "fine-cell density",
        "coarse_density_count": f"{coarse_label} density count",
        "coarse_density_weight": f"{coarse_label} density weight",
        "coarse_ivar_briggs": f"{coarse_label} averaged vis",
    }
    datasets = [("noiseless", noiseless, False), ("physical SNR, no gain", physical_snr, True)]
    axis_uas = noiseless["axis_uas"]
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    images: dict[tuple[str, str], np.ndarray] = {}
    for row_name, data, is_noisy in datasets:
        all_metrics[row_name] = {}
        for mode in modes:
            image = reconstruct(data, mode, noisy=is_noisy)
            images[(row_name, mode)] = image
            all_metrics[row_name][mode] = metrics(data["truth"], image, data["axis_uas"])

    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.1,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
        }
    )
    fig, axes = plt.subplots(2, len(modes) + 1, figsize=(10.4, 4.3), constrained_layout=True)
    for row, (row_name, data, _) in enumerate(datasets):
        ax = axes[row, 0]
        ax.imshow(opt.normalize_blr_display(data["truth"]), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(f"{row_name}\ninput")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        for col, mode in enumerate(modes, start=1):
            ax = axes[row, col]
            image = images[(row_name, mode)]
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            m = all_metrics[row_name][mode]
            ax.set_title(f"{labels[mode]}\nBLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}")
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col > 1:
                ax.set_yticklabels([])

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=axes,
        fraction=0.018,
        pad=0.012,
    )
    cbar.set_label("norm. brightness\n(BLR arcsinh)", fontsize=7.0)

    png = OUTDIR / f"prl_uv_weighting_diagnostic{OUTPUT_SUFFIX}.png"
    pdf = OUTDIR / f"prl_uv_weighting_diagnostic{OUTPUT_SUFFIX}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    stats = {
        "layout_file": str(LAYOUT_FILE),
        "diagnostic": "Compare per-sample weighting with uv-cell area weighting. The physical-SNR row uses the real array noise model with no extra SNR multiplier.",
        "array_latitude_deg": ARRAY_LAT_DEG,
        "source_declination_deg": SOURCE_DEC_DEG,
        "n_time_windows": N_TIME_WINDOWS,
        "exposure_s": EXPOSURE_S,
        "exposure_gap_s": EXPOSURE_GAP_S,
        "observing_days": OBSERVING_DAYS,
        "weighting_modes": labels,
        "sigma_floor_for_inverse_variance": SIGMA_FLOOR,
        "coarse_uv_bin_mode": COARSE_UV_BIN_MODE,
        "coarse_uv_res_factor": COARSE_UV_RES_FACTOR,
        "coarse_uv_bins_u": n_bin_u,
        "coarse_uv_bins_v": n_bin_v,
        "coarse_uv_gaussian_smooth_cells": COARSE_SMOOTH_CELLS,
        "metrics": all_metrics,
        "baseline_min_km": float(np.min(noiseless["baseline_lengths_km"])),
        "baseline_max_km": float(np.max(noiseless["baseline_lengths_km"])),
        "station_link_eff_min": float(np.min(noiseless["station_link_eff"])),
        "station_link_eff_max": float(np.max(noiseless["station_link_eff"])),
    }
    stats_path = OUTDIR / f"prl_uv_weighting_diagnostic{OUTPUT_SUFFIX}.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(pdf)
    print(png)
    print(stats_path)
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
