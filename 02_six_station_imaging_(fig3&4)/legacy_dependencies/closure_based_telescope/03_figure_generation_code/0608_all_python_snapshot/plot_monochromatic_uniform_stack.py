from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_prl_broadband_clean as base
import plot_prl_broadband_blr_optimized as opt
from plot_prl_broadband_blr_realnight import (
    ARRAY_LAT_DEG,
    EXPOSURE_GAP_S,
    SOURCE_DEC_DEG,
    project_enu_baselines,
    realnight_hour_angles,
)


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_PIX = int(os.environ.get("MONO_N_PIX", "256"))
HALF_WIDTH_UAS = float(os.environ.get("MONO_HALF_WIDTH_UAS", "80.0"))
LAMBDA_MIN_NM = float(os.environ.get("LAMBDA_MIN_NM", "400.0"))
LAMBDA_MAX_NM = float(os.environ.get("LAMBDA_MAX_NM", "800.0"))
LAMBDA_STEP_NM = float(os.environ.get("LAMBDA_STEP_NM", "10.0"))
SUPPORT_MODE = os.environ.get("MONO_SUPPORT_MODE", "ellipse").lower()
CELL_AVERAGE_MODES = tuple(os.environ.get("MONO_CELL_AVERAGE_MODES", "direct,noise").split(","))
SIGMA_FLOOR = float(os.environ.get("MONO_SIGMA_FLOOR", "0.11"))
BAND_STACK_MODE = os.environ.get("MONO_BAND_STACK_MODE", "noise").lower()
NOISELESS = os.environ.get("MONO_NOISELESS", "0").lower() in {"1", "true", "yes", "on"}
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "_mono_uniform_stack")
RNG_SEED = int(os.environ.get("MONO_RNG_SEED", "273"))
DRIFT_RNG_SEED = int(os.environ.get("MONO_DRIFT_RNG_SEED", "31415"))


def _shift_int(arr: np.ndarray, dy: int, dx: int, fill: int = -1) -> np.ndarray:
    out = np.full_like(arr, fill)
    if dy >= 0:
        ys_src = slice(0, arr.shape[0] - dy)
        ys_dst = slice(dy, arr.shape[0])
    else:
        ys_src = slice(-dy, arr.shape[0])
        ys_dst = slice(0, arr.shape[0] + dy)
    if dx >= 0:
        xs_src = slice(0, arr.shape[1] - dx)
        xs_dst = slice(dx, arr.shape[1])
    else:
        xs_src = slice(-dx, arr.shape[1])
        xs_dst = slice(0, arr.shape[1] + dx)
    out[ys_dst, xs_dst] = arr[ys_src, xs_src]
    return out


def nearest_label_map(occupied: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest occupied-cell labels for every supported Fourier cell.

    This is an 8-connected grid-nearest fill.  It intentionally works on cell
    labels rather than visibility values so the same map can be reused for all
    readout strategies and averaging choices at a fixed wavelength.
    """
    yy, xx = np.indices(occupied.shape)
    label_y = np.where(occupied, yy, -1)
    label_x = np.where(occupied, xx, -1)
    filled = occupied.copy()
    target = support & ~filled
    neighbours = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    for _ in range(max(occupied.shape)):
        if not np.any(target):
            break
        prev_filled = filled.copy()
        prev_y = label_y.copy()
        prev_x = label_x.copy()
        for dy, dx in neighbours:
            src = _shift_int(prev_filled.astype(np.int8), dy, dx, fill=0).astype(bool)
            assign = target & src
            if not np.any(assign):
                continue
            label_y[assign] = _shift_int(prev_y, dy, dx)[assign]
            label_x[assign] = _shift_int(prev_x, dy, dx)[assign]
            filled[assign] = True
            target[assign] = False
    fillable = support & (label_y >= 0) & (label_x >= 0)
    return label_y, label_x, fillable


def support_mask_from_occupied(occupied: np.ndarray, *, du: float, mode: str) -> np.ndarray:
    n = occupied.shape[0]
    mid = n // 2
    coords = (np.arange(n) - mid) * du
    uu, vv = np.meshgrid(coords, coords)
    occ = occupied.copy()
    occ[mid, mid] = False
    if not np.any(occ):
        return occupied.copy()
    max_u = float(np.max(np.abs(uu[occ])))
    max_v = float(np.max(np.abs(vv[occ])))
    max_r = float(np.max(np.sqrt(uu[occ] ** 2 + vv[occ] ** 2)))
    if mode == "circle":
        support = uu**2 + vv**2 <= max_r**2
    elif mode == "rectangle":
        support = (np.abs(uu) <= max_u) & (np.abs(vv) <= max_v)
    elif mode == "ellipse":
        support = (uu / max(max_u, du)) ** 2 + (vv / max(max_v, du)) ** 2 <= 1.0
    else:
        raise ValueError(f"Unknown MONO_SUPPORT_MODE={mode!r}; use ellipse, circle, or rectangle.")
    support[mid, mid] = True
    return support


def aggregate_cells(
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    sigma: np.ndarray,
    *,
    n: int,
    fov_rad: float,
    average_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average samples into nearest Fourier cells before uniform-cell imaging."""
    du = 1.0 / fov_rad
    mid = n // 2
    u_full = np.concatenate([u, -u])
    v_full = np.concatenate([v, -v])
    vis_full = np.concatenate([vis, np.conj(vis)])
    sigma_full = np.concatenate([sigma, sigma])
    iu = np.rint(u_full / du + mid).astype(int)
    iv = np.rint(v_full / du + mid).astype(int)
    valid = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
    flat = iv[valid] * n + iu[valid]
    values = vis_full[valid]
    var_samples = sigma_full[valid] ** 2 + SIGMA_FLOOR**2
    if average_mode == "direct":
        count = np.bincount(flat, minlength=n * n).astype(float)
        denom = count
        real = np.bincount(flat, weights=values.real, minlength=n * n)
        imag = np.bincount(flat, weights=values.imag, minlength=n * n)
        var_sum = np.bincount(flat, weights=var_samples, minlength=n * n)
        cell_var = np.full(n * n, np.inf)
        occupied_for_var = count > 0.0
        cell_var[occupied_for_var] = var_sum[occupied_for_var] / count[occupied_for_var] ** 2
    elif average_mode == "noise":
        ivar = 1.0 / var_samples
        denom = np.bincount(flat, weights=ivar, minlength=n * n).astype(float)
        real = np.bincount(flat, weights=ivar * values.real, minlength=n * n)
        imag = np.bincount(flat, weights=ivar * values.imag, minlength=n * n)
        cell_var = np.full(n * n, np.inf)
        occupied_for_var = denom > 0.0
        cell_var[occupied_for_var] = 1.0 / denom[occupied_for_var]
    else:
        raise ValueError(f"Unknown averaging mode {average_mode!r}; use direct or noise.")

    grid = np.zeros(n * n, dtype=complex)
    occupied_flat = denom > 0.0
    grid[occupied_flat] = (real[occupied_flat] + 1j * imag[occupied_flat]) / denom[occupied_flat]
    grid = grid.reshape(n, n)
    occupied = occupied_flat.reshape(n, n)
    cell_var = cell_var.reshape(n, n)
    grid[mid, mid] = 1.0 + 0.0j
    occupied[mid, mid] = True
    cell_var[mid, mid] = np.inf
    return grid, occupied, cell_var


def monochromatic_dirty_image(
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    sigma: np.ndarray,
    *,
    n: int,
    fov_rad: float,
    average_mode: str,
    label_y: np.ndarray,
    label_x: np.ndarray,
    fillable: np.ndarray,
) -> tuple[np.ndarray, float]:
    grid, occupied, cell_var = aggregate_cells(u, v, vis, sigma, n=n, fov_rad=fov_rad, average_mode=average_mode)
    filled_grid = np.zeros_like(grid)
    filled_grid[fillable] = grid[label_y[fillable], label_x[fillable]]
    filled_grid[occupied] = grid[occupied]
    support_grid = np.zeros_like(grid, dtype=float)
    support_grid[fillable] = 1.0
    support_grid[occupied] = 1.0
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(filled_grid))).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(support_grid))).real
    peak = psf[n // 2, n // 2]
    if peak > 0.0:
        image /= peak

    assigned_var = cell_var[label_y[fillable], label_x[fillable]]
    finite = np.isfinite(assigned_var) & (assigned_var > 0.0)
    if np.any(finite):
        band_weight = 1.0 / float(np.mean(assigned_var[finite]))
    else:
        band_weight = 1.0
    return image, band_weight


def simulate_band(
    *,
    lam_lo: float,
    lam_hi: float,
    rng: np.random.Generator,
    drift_rng: np.random.Generator,
    vgrid: np.ndarray,
    uv_axis: np.ndarray,
    baselines_km: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
    station_link_eff: np.ndarray,
    station_channel_noise: np.ndarray,
    edge_split_coherence_eff: np.ndarray,
    edge_split_load_eff: np.ndarray,
    edge_split_channel_noise: np.ndarray,
    closure_rank_share: float,
    hour_angles: np.ndarray,
) -> dict[str, np.ndarray]:
    lam = np.sqrt(lam_lo * lam_hi)
    freq = base.C_LIGHT / lam
    freq_lo = base.C_LIGHT / lam_hi
    freq_hi = base.C_LIGHT / lam_lo
    df = freq_hi - freq_lo
    u_mode = base.source_mode_occupation(freq, diameter_m=base.TELESCOPE_DIAMETER_M)
    total_modes = opt.EXPOSURE_S * opt.OBSERVING_DAYS * df
    post_average_drift_std = float(os.environ.get("POST_AVERAGE_DRIFT_STD", str(np.pi / 10.0)))
    station_piston_std = post_average_drift_std / np.sqrt(2.0)

    uu_rows, vv_rows = project_enu_baselines(
        baselines_km,
        hour_angles,
        lam,
        latitude_deg=ARRAY_LAT_DEG,
        declination_deg=SOURCE_DEC_DEG,
    )

    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    vis_by_strategy = {"all": [], "split": [], "direct": []}
    sigma_by_strategy = {"all": [], "split": [], "direct": []}

    for uu, vv in zip(uu_rows, vv_rows):
        vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
        amp = np.abs(vtrue)
        phase = np.angle(vtrue)
        phase_closure = q_basis @ (q_basis.T @ phase)
        nu_eff = np.clip(amp, 1e-4, 0.98)

        if NOISELESS:
            zero_sigma = np.zeros_like(amp)
            all_u.append(uu)
            all_v.append(vv)
            vis_by_strategy["all"].append(amp * np.exp(1j * phase))
            vis_by_strategy["split"].append(amp * np.exp(1j * phase_closure))
            vis_by_strategy["direct"].append(amp * np.exp(1j * phase_closure))
            sigma_by_strategy["all"].append(zero_sigma)
            sigma_by_strategy["split"].append(zero_sigma)
            sigma_by_strategy["direct"].append(zero_sigma)
            continue

        fisher_split = (
            total_modes
            * 4.0
            * (edge_split_coherence_eff * u_mode) ** 2
            * nu_eff**2
            / (edge_split_load_eff * u_mode + edge_split_channel_noise)
        )
        sigma_split = np.minimum(1.0 / np.sqrt(np.maximum(fisher_split, 1e-18)), 2.5)
        sigma_split /= opt.IMAGING_SNR_BOOST
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
            * opt.IMAGING_SNR_BOOST**2
        )
        noise_direct, sigma_direct = base.sample_cycle_noise_from_fisher(rng, fisher_direct, q_basis)

        station_pistons = drift_rng.normal(scale=station_piston_std, size=len(station_link_eff))
        station_pistons -= np.mean(station_pistons)
        residual_drift = np.array([station_pistons[i] - station_pistons[j] for i, j in edges])
        noise_all = raw_split_noise + residual_drift
        sigma_all = np.sqrt(sigma_split**2 + post_average_drift_std**2)

        all_u.append(uu)
        all_v.append(vv)
        vis_by_strategy["all"].append(amp * np.exp(1j * (phase + noise_all)))
        vis_by_strategy["split"].append(amp * np.exp(1j * (phase_closure + noise_split)))
        vis_by_strategy["direct"].append(amp * np.exp(1j * (phase_closure + noise_direct)))
        sigma_by_strategy["all"].append(sigma_all)
        sigma_by_strategy["split"].append(sigma_split_projected)
        sigma_by_strategy["direct"].append(sigma_direct)

    out: dict[str, np.ndarray] = {
        "u": np.concatenate(all_u),
        "v": np.concatenate(all_v),
    }
    for key in ("all", "split", "direct"):
        out[f"vis_{key}"] = np.concatenate(vis_by_strategy[key])
        out[f"sigma_{key}"] = np.concatenate(sigma_by_strategy[key])
    return out


def normalize_stack(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    image -= np.percentile(image, 0.5)
    return np.clip(image, 0.0, None)


def make_panel_figure(
    stacks: dict[str, dict[str, np.ndarray]],
    truth: np.ndarray,
    axis_uas: np.ndarray,
    stats: dict,
) -> tuple[Path, Path]:
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    mode = CELL_AVERAGE_MODES[0]
    plt.rcParams.update(
        {
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.1,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
        }
    )
    fig = plt.figure(figsize=(7.35, 4.85), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.36, wspace=0.33)
    image_axes = []

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(opt.STATIONS_KM[:, 0], opt.STATIONS_KM[:, 1], s=24, color="#005f73")
    ax.scatter([opt.HUB_KM[0]], [opt.HUB_KM[1]], s=54, marker="*", color="#ca6702", label="hub", zorder=3)
    for i, j in base.edge_list(len(opt.STATIONS_KM)):
        ax.plot(
            [opt.STATIONS_KM[i, 0], opt.STATIONS_KM[j, 0]],
            [opt.STATIONS_KM[i, 1], opt.STATIONS_KM[j, 1]],
            color="0.80",
            lw=0.5,
            zorder=0,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east-west (km)")
    ax.set_ylabel("north-south (km)")
    ax.set_title("Eight stations + hub")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8, handletextpad=0.2)

    ax = fig.add_subplot(gs[0, 1])
    coverage_colors = [("#005f73", 0.52), ("#ee9b00", 0.44)]
    for key, (color, alpha) in zip(("400", "800"), coverage_colors):
        coverage = stats["endpoint_coverage_g_lambda"][key]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax.scatter(uu, vv, s=1.15, color=color, alpha=alpha, label=f"{key} nm")
        ax.scatter(-uu, -vv, s=1.15, color=color, alpha=0.62 * alpha)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, handletextpad=0.1, borderpad=0.1)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_title("Input source")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    image_axes.append(ax)

    panel_titles = {
        "all": "All visibilities + piston drift",
        "split": "Edge-first closure",
        "direct": "Direct closure-space",
    }
    for col, key in enumerate(("all", "split", "direct")):
        image = stacks[mode][key]
        metric = stats["metrics"][mode][key]
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(f"{panel_titles[key]}\nBLR r={metric['blr_corr']:.2f}, all r={metric['global_corr']:.2f}")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
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
    png = OUTDIR / f"prl_mono_uniform_stack{OUTPUT_SUFFIX}.png"
    pdf = OUTDIR / f"prl_mono_uniform_stack{OUTPUT_SUFFIX}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    drift_rng = np.random.default_rng(DRIFT_RNG_SEED)
    fov_rad = 2.0 * HALF_WIDTH_UAS * base.UAS_TO_RAD
    truth, axis_uas = base.make_source(N_PIX, HALF_WIDTH_UAS)
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)

    n_station = len(opt.STATIONS_KM)
    edges = base.edge_list(n_station)
    baselines_km = np.array([opt.STATIONS_KM[j] - opt.STATIONS_KM[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    closure_rank_share = min(1.0, (n_station - 1.0) / w_basis.shape[1])

    hub_distances_km = np.linalg.norm(opt.STATIONS_KM - opt.HUB_KM, axis=1)
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
    hour_angles = realnight_hour_angles(opt.N_TIME_WINDOWS, opt.EXPOSURE_S, EXPOSURE_GAP_S)
    endpoint_coverage: dict[str, dict[str, list[float]]] = {}
    for wavelength_nm in (LAMBDA_MIN_NM, LAMBDA_MAX_NM):
        uu_rows, vv_rows = project_enu_baselines(
            baselines_km,
            hour_angles,
            wavelength_nm * 1e-9,
            latitude_deg=ARRAY_LAT_DEG,
            declination_deg=SOURCE_DEC_DEG,
        )
        endpoint_coverage[f"{wavelength_nm:g}"] = {
            "u": (uu_rows.reshape(-1) / 1e9).tolist(),
            "v": (vv_rows.reshape(-1) / 1e9).tolist(),
        }

    lam_edges_nm = np.arange(LAMBDA_MIN_NM, LAMBDA_MAX_NM + 0.5 * LAMBDA_STEP_NM, LAMBDA_STEP_NM)
    if lam_edges_nm[-1] < LAMBDA_MAX_NM:
        lam_edges_nm = np.append(lam_edges_nm, LAMBDA_MAX_NM)
    lam_edges_nm[-1] = LAMBDA_MAX_NM
    n_band = len(lam_edges_nm) - 1

    stacks = {
        mode: {key: np.zeros((N_PIX, N_PIX), dtype=float) for key in ("all", "split", "direct")}
        for mode in CELL_AVERAGE_MODES
    }
    stack_weights = {
        mode: {key: 0.0 for key in ("all", "split", "direct")}
        for mode in CELL_AVERAGE_MODES
    }
    band_weight_history = {
        mode: {key: [] for key in ("all", "split", "direct")}
        for mode in CELL_AVERAGE_MODES
    }
    support_cell_counts: list[int] = []
    occupied_cell_counts: list[int] = []

    for band_index, (lo_nm, hi_nm) in enumerate(zip(lam_edges_nm[:-1], lam_edges_nm[1:]), start=1):
        band = simulate_band(
            lam_lo=lo_nm * 1e-9,
            lam_hi=hi_nm * 1e-9,
            rng=rng,
            drift_rng=drift_rng,
            vgrid=vgrid,
            uv_axis=uv_axis,
            baselines_km=baselines_km,
            q_basis=q_basis,
            edges=edges,
            station_link_eff=station_link_eff,
            station_channel_noise=station_channel_noise,
            edge_split_coherence_eff=edge_split_coherence_eff,
            edge_split_load_eff=edge_split_load_eff,
            edge_split_channel_noise=edge_split_channel_noise,
            closure_rank_share=closure_rank_share,
            hour_angles=hour_angles,
        )
        reference_grid, occupied, _ = aggregate_cells(
            band["u"],
            band["v"],
            band["vis_direct"],
            band["sigma_direct"],
            n=N_PIX,
            fov_rad=fov_rad,
            average_mode="direct",
        )
        del reference_grid
        du = 1.0 / fov_rad
        support = support_mask_from_occupied(occupied, du=du, mode=SUPPORT_MODE)
        label_y, label_x, fillable = nearest_label_map(occupied, support)
        support_cell_counts.append(int(np.sum(fillable)))
        occupied_cell_counts.append(int(np.sum(occupied)))

        for mode in CELL_AVERAGE_MODES:
            for key in ("all", "split", "direct"):
                image, band_weight = monochromatic_dirty_image(
                    band["u"],
                    band["v"],
                    band[f"vis_{key}"],
                    band[f"sigma_{key}"],
                    n=N_PIX,
                    fov_rad=fov_rad,
                    average_mode=mode,
                    label_y=label_y,
                    label_x=label_x,
                    fillable=fillable,
                )
                if BAND_STACK_MODE == "equal":
                    use_weight = 1.0
                elif BAND_STACK_MODE == "noise":
                    use_weight = band_weight
                else:
                    raise ValueError("MONO_BAND_STACK_MODE must be 'equal' or 'noise'.")
                stacks[mode][key] += use_weight * image
                stack_weights[mode][key] += use_weight
                band_weight_history[mode][key].append(float(use_weight))
        if band_index % max(1, n_band // 10) == 0:
            print(f"processed {band_index}/{n_band} monochromatic bands")

    for mode in CELL_AVERAGE_MODES:
        for key in ("all", "split", "direct"):
            norm = stack_weights[mode][key] if stack_weights[mode][key] > 0.0 else n_band
            stacks[mode][key] = normalize_stack(stacks[mode][key] / norm)

    metrics = {
        mode: {
            key: {
                "global_corr": float(base.corrcoef_positive(truth, stacks[mode][key])),
                "blr_corr": float(opt.masked_corr(truth, stacks[mode][key], opt.blr_masks(axis_uas)[0])),
                "ring_contrast": float(opt.ring_contrast(stacks[mode][key], *opt.blr_masks(axis_uas))),
            }
            for key in ("all", "split", "direct")
        }
        for mode in CELL_AVERAGE_MODES
    }

    stats = {
        "method": "monochromatic uniform-cell dirty-map stack",
        "lambda_min_nm": LAMBDA_MIN_NM,
        "lambda_max_nm": LAMBDA_MAX_NM,
        "lambda_step_nm": LAMBDA_STEP_NM,
        "n_bands": n_band,
        "n_pix": N_PIX,
        "half_width_uas": HALF_WIDTH_UAS,
        "du_g_lambda": 1.0 / fov_rad / 1e9,
        "support_mode": SUPPORT_MODE,
        "cell_average_modes": CELL_AVERAGE_MODES,
        "sigma_floor": SIGMA_FLOOR,
        "band_stack_mode": BAND_STACK_MODE,
        "band_stack_weight_summary": {
            mode: {
                key: {
                    "min": float(np.min(band_weight_history[mode][key])),
                    "median": float(np.median(band_weight_history[mode][key])),
                    "max": float(np.max(band_weight_history[mode][key])),
                }
                for key in ("all", "split", "direct")
            }
            for mode in CELL_AVERAGE_MODES
        },
        "station_layout_file": opt.STATION_LAYOUT_FILE,
        "hub_km": opt.HUB_KM.tolist(),
        "array_latitude_deg": ARRAY_LAT_DEG,
        "source_declination_deg": SOURCE_DEC_DEG,
        "n_time_windows": opt.N_TIME_WINDOWS,
        "exposure_s": opt.EXPOSURE_S,
        "exposure_gap_s": EXPOSURE_GAP_S,
        "observing_days": opt.OBSERVING_DAYS,
        "imaging_snr_boost": opt.IMAGING_SNR_BOOST,
        "noiseless": NOISELESS,
        "baseline_min_km": float(np.min(np.linalg.norm(baselines_km, axis=1))),
        "baseline_max_km": float(np.max(np.linalg.norm(baselines_km, axis=1))),
        "occupied_cells_median": float(np.median(occupied_cell_counts)),
        "support_cells_median": float(np.median(support_cell_counts)),
        "fiber_length_scale": float(base.FIBER_LENGTH_SCALE),
        "fiber_loss_db_per_km": float(base.FIBER_LOSS_DB_PER_KM),
        "mode_false_positive_per_station_mode": float(base.MODE_FALSE_POSITIVE),
        "pair_false_positive_per_pair_combiner": float(base.PAIR_FALSE_POSITIVE),
        "noise_model": "pure fibre attenuation plus independent mode-local false positives",
        "source_spectrum_name": base.SOURCE_SPECTRUM_NAME,
        "source_spectrum_note": base.SOURCE_SPECTRUM_NOTE,
        "source_spectrum_ned_points_400_800nm": base.SOURCE_SED_NED_POINT_COUNT,
        "source_spectrum_lambda_nm": base.SOURCE_SED_LAMBDA_NM.tolist(),
        "source_spectrum_fnu_jy": base.SOURCE_SED_FNU_JY.tolist(),
        "effective_hub_distance_min_km": float(np.min(effective_hub_distances_km)),
        "effective_hub_distance_max_km": float(np.max(effective_hub_distances_km)),
        "station_link_eff_min": float(np.min(station_link_eff)),
        "station_link_eff_max": float(np.max(station_link_eff)),
        "endpoint_coverage_g_lambda": endpoint_coverage,
        "coverage_400nm_half_range_g_lambda": {
            "u": float(np.max(np.abs(endpoint_coverage[f"{LAMBDA_MIN_NM:g}"]["u"]))),
            "v": float(np.max(np.abs(endpoint_coverage[f"{LAMBDA_MIN_NM:g}"]["v"]))),
        },
        "coverage_800nm_half_range_g_lambda": {
            "u": float(np.max(np.abs(endpoint_coverage[f"{LAMBDA_MAX_NM:g}"]["u"]))),
            "v": float(np.max(np.abs(endpoint_coverage[f"{LAMBDA_MAX_NM:g}"]["v"]))),
        },
        "metrics": metrics,
    }
    pdf, png = make_panel_figure(stacks, truth, axis_uas, stats)
    stats_path = OUTDIR / f"prl_mono_uniform_stack_stats{OUTPUT_SUFFIX}.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(pdf)
    print(png)
    print(stats_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
