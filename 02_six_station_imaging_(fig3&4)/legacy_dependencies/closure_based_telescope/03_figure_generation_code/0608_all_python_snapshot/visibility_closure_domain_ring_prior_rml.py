from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import latest_maunakea_closure_snr_clean_rml as latest
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from plot_monochromatic_uniform_stack import aggregate_cells, normalize_stack


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

REAL_LAYOUT = OUTFIG / "maunakea_top4_plus5_ngc4151_layout.json"
OPTIMAL_LAYOUT = OUTFIG / "optimized_array_topology_u50v50_lowuv15_near10_5_ydown1p5_hub_m3_m4.json"

SOURCE = ngc.NGC4151
N_RML = int(os.environ.get("RML_N_PIX", "128"))
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "30"))
SNR_BOOST = float(os.environ.get("AUGMENTED_SNR_BOOST", "1.0"))
FIBER_LOSS_DB_PER_KM = float(os.environ.get("FIBER_LOSS_DB_PER_KM", "0.2"))
MODE_FALSE_POSITIVE = float(os.environ.get("MODE_FALSE_POSITIVE", os.environ.get("STATION_FALSE_POSITIVE", "0.05")))
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", os.environ.get("BASELINE_FALSE_POSITIVE", "0.0")))
PRIOR_WEIGHT = float(os.environ.get("RML_RING_PRIOR_WEIGHT", "0.85"))
PRIOR_FOURIER_WEIGHT = float(os.environ.get("RML_PRIOR_FOURIER_WEIGHT", "0.35"))
TV_WEIGHT = float(os.environ.get("RML_TV_WEIGHT", "0.08"))
N_ITER = int(os.environ.get("RML_N_ITER", "520"))
STEP = float(os.environ.get("RML_STEP", "0.045"))
CLOSURE_N_ITER = int(os.environ.get("CLOSURE_RML_N_ITER", "260"))
CLOSURE_STEP = float(os.environ.get("CLOSURE_RML_STEP", "0.034"))


def load_synthetic_case(path: Path) -> aug.NetworkCase:
    payload = json.loads(path.read_text())
    telescopes = [
        aug.Telescope(f"S{i + 1}", float(x), float(y), 5.0, True)
        for i, (x, y) in enumerate(np.asarray(payload["stations_km"], dtype=float))
    ]
    return aug.NetworkCase(
        key=path.stem,
        title="Synthetic 8-station optimized array",
        latitude_deg=35.0,
        center_latlon=(35.0, 0.0),
        telescopes=telescopes,
        hub_km=tuple(payload.get("hub_km", [0.0, 0.0])),
        optimization_score=0.0,
    )


def output_tag() -> str:
    return (
        f"{SOURCE.key}_{OBSERVING_DAYS}d_snr{SNR_BOOST:g}_loss{FIBER_LOSS_DB_PER_KM:g}"
        f"_modefp{MODE_FALSE_POSITIVE:g}_pairfp{PAIR_FALSE_POSITIVE:g}"
        f"_n{N_RML}_prior{PRIOR_WEIGHT:g}"
    ).replace(".", "p")


def core_ring_prior(axis_uas: np.ndarray, source: ngc.SourceModel) -> np.ndarray:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    core_sigma = max(source.disc_sigma_major_uas, 7.5)
    ring_sigma = max(source.blr_width_uas, 6.0)
    core = np.exp(-0.5 * (rr / core_sigma) ** 2)
    ring = np.exp(-0.5 * ((rr - source.blr_radius_uas) / ring_sigma) ** 2)
    core /= np.sum(core)
    ring /= np.sum(ring)
    prior = 0.42 * core + 0.58 * ring
    prior = np.clip(prior, 0.0, None)
    prior /= np.sum(prior)
    return prior


def rms(x: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))
    return value if np.isfinite(value) and value > 1e-30 else 1.0


def project_simplex_positive(image: np.ndarray, smooth_pix: float = 0.0) -> np.ndarray:
    out = np.clip(image, 0.0, None)
    if smooth_pix > 0.15:
        out = base.gaussian_filter(out, smooth_pix)
        out = np.clip(out, 0.0, None)
    total = float(np.sum(out))
    if total <= 0.0 or not np.isfinite(total):
        out = np.ones_like(image)
        total = float(np.sum(out))
    return out / total


def make_gridded_data(
    bands: list[dict[str, np.ndarray]],
    strategy: str,
    *,
    n: int,
    fov_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_num = np.zeros((n, n), dtype=complex)
    weight_sum = np.zeros((n, n), dtype=float)
    occupied_any = np.zeros((n, n), dtype=bool)
    for band in bands:
        grid, occupied, cell_var = aggregate_cells(
            band["u"],
            band["v"],
            band[f"vis_{strategy}"],
            band[f"sigma_{strategy}"],
            n=n,
            fov_rad=fov_rad,
            average_mode="noise",
        )
        weight = np.zeros_like(cell_var, dtype=float)
        finite = occupied & np.isfinite(cell_var) & (cell_var > 0.0)
        weight[finite] = 1.0 / cell_var[finite]
        data_num[finite] += weight[finite] * grid[finite]
        weight_sum[finite] += weight[finite]
        occupied_any |= finite
    data = np.zeros_like(data_num)
    valid = weight_sum > 0.0
    data[valid] = data_num[valid] / weight_sum[valid]

    positive = weight_sum[valid]
    weight = np.zeros_like(weight_sum)
    if len(positive):
        # Normalize the data term so optimizer hyperparameters are portable
        # between the real and synthetic topologies.
        clipped = np.clip(positive / np.median(positive), 0.02, 50.0)
        weight[valid] = clipped
    mid = n // 2
    weight[mid, mid] = 0.0
    occupied_any[mid, mid] = False
    return data, weight, occupied_any


def dirty_from_grid(data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data * weight))).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(weight))).real
    peak = psf[psf.shape[0] // 2, psf.shape[1] // 2]
    if peak > 1e-30:
        image /= peak
    image -= np.percentile(image, 1.0)
    return project_simplex_positive(image, smooth_pix=0.7)


def fft_vis(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))


def adjoint_image(grid_gradient: np.ndarray) -> np.ndarray:
    n = grid_gradient.shape[0]
    return (n * n) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_gradient))).real


def add_visibility_adjoint_samples(
    grid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coeff: np.ndarray,
    *,
    fov_rad: float,
) -> None:
    n = grid.shape[0]
    du = 1.0 / fov_rad
    mid = n // 2
    for sign, values in ((1.0, coeff), (-1.0, np.conj(coeff))):
        iu = np.rint(sign * u / du + mid).astype(int)
        iv = np.rint(sign * v / du + mid).astype(int)
        valid = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n) & np.isfinite(values)
        flat = iv[valid] * n + iu[valid]
        np.add.at(grid.reshape(-1), flat, values[valid])


def rml_reconstruct(
    data: np.ndarray,
    weight: np.ndarray,
    prior: np.ndarray,
    *,
    mode: str,
    prior_weight: float = PRIOR_WEIGHT,
    tv_weight: float = TV_WEIGHT,
    n_iter: int = N_ITER,
    step: float = STEP,
) -> tuple[np.ndarray, dict[str, float]]:
    x = project_simplex_positive(0.45 * dirty_from_grid(data, weight) + 0.55 * prior, smooth_pix=0.3)
    history = {}
    mask = weight > 0.0
    for iteration in range(n_iter):
        vis = fft_vis(x)
        if mode == "visibility":
            grid_grad = weight * (vis - data)
            data_grad = 2.0 * adjoint_image(grid_grad)
            data_loss = float(np.sum(weight[mask] * np.abs(vis[mask] - data[mask]) ** 2) / max(np.sum(mask), 1))
        elif mode == "phase":
            dphi = np.zeros_like(weight)
            dphi[mask] = np.angle(vis[mask] * np.conj(data[mask]))
            safe_vis = np.where(np.abs(vis) > 1e-4, vis, 1e-4 + 0j)
            # For L = 1/2 sum w dphi^2, dL/dI is the Fourier adjoint of
            # i w dphi / conj(V).  This fits the gauge-fixed closure-projected
            # phases, not the noisy image-domain dirty map.
            grid_grad = np.zeros_like(data)
            grid_grad[mask] = 1j * weight[mask] * dphi[mask] / np.conj(safe_vis[mask])
            data_grad = adjoint_image(grid_grad)
            data_loss = float(np.sum(weight[mask] * dphi[mask] ** 2) / max(np.sum(mask), 1))
        else:
            raise ValueError("mode must be 'visibility' or 'phase'.")

        prior_grad = 2.0 * (x - prior)
        tv_grad = latest.tv_gradient(x)
        grad = data_grad / rms(data_grad) + prior_weight * prior_grad / rms(prior_grad) + tv_weight * tv_grad / rms(tv_grad)
        x = project_simplex_positive(x - step * grad, smooth_pix=0.18 if (iteration + 1) % 30 == 0 else 0.0)
        if iteration in {0, n_iter // 2, n_iter - 1}:
            history[f"loss_{iteration}"] = data_loss
    return normalize_stack(x), history


def visibility_map_with_ring_prior(
    data: np.ndarray,
    weight: np.ndarray,
    prior: np.ndarray,
    *,
    prior_fourier_weight: float = PRIOR_FOURIER_WEIGHT,
) -> tuple[np.ndarray, dict[str, float]]:
    """Closed-form visibility-domain MAP with a Gaussian core+ring prior.

    This minimizes a quadratic approximation
        sum_k w_k |V_k(I)-D_k|^2 + lambda sum_k |V_k(I)-V_k(P)|^2
    in Fourier space, followed by a conservative positivity/flux projection.
    It is intentionally used as the stable visibility-domain RML baseline.
    """
    prior_vis = fft_vis(prior)
    denom = weight + prior_fourier_weight
    vis = np.where(denom > 0.0, (weight * data + prior_fourier_weight * prior_vis) / denom, prior_vis)
    mid = vis.shape[0] // 2
    vis[mid, mid] = 1.0 + 0.0j
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis))).real
    image = project_simplex_positive(image, smooth_pix=0.35)
    # A short projected-gradient polish keeps measured visibilities honest
    # after clipping negative sidelobes.
    for _ in range(80):
        model = fft_vis(image)
        grid_grad = weight * (model - data) + prior_fourier_weight * (model - prior_vis)
        grad = 2.0 * adjoint_image(grid_grad)
        image = project_simplex_positive(image - 0.006 * grad / rms(grad), smooth_pix=0.0)
    model = fft_vis(image)
    data_loss = float(np.sum(weight[weight > 0.0] * np.abs(model[weight > 0.0] - data[weight > 0.0]) ** 2) / max(np.count_nonzero(weight), 1))
    prior_loss = float(np.mean(np.abs(model - prior_vis) ** 2))
    return normalize_stack(image), {"data_loss": data_loss, "prior_loss": prior_loss}


def closure_phase_only_rml(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    strategy: str,
    prior: np.ndarray,
    dirty_start: np.ndarray,
    *,
    fov_rad: float,
) -> tuple[np.ndarray, dict[str, float]]:
    stations, _, _, _ = aug.station_table_from_case(case)
    edges = base.edge_list(len(stations))
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, len(stations)))
    n_edges = len(edges)
    uv_axis = np.fft.fftshift(np.fft.fftfreq(N_RML, d=fov_rad / N_RML))

    sigma_values = np.concatenate([band[f"sigma_{strategy}"] for band in bands])
    sigma_floor = max(float(np.nanmedian(sigma_values)) * 0.35, 0.12)
    weight_norm = 1.0 / np.nanmedian(1.0 / (sigma_values**2 + sigma_floor**2))
    x = project_simplex_positive(0.72 * prior + 0.28 * dirty_start, smooth_pix=0.3)
    history: dict[str, float] = {}
    for iteration in range(CLOSURE_N_ITER):
        vis_grid = fft_vis(x)
        grad_grid = np.zeros_like(vis_grid)
        phase_loss = 0.0
        amp_loss = 0.0
        n_resid = 0
        for band in bands:
            u = band["u"]
            v = band["v"]
            model = base.interp_vis(vis_grid, uv_axis, u, v)
            data = band[f"vis_{strategy}"]
            sigma = band[f"sigma_{strategy}"]
            model_phase = np.angle(model).reshape(-1, n_edges)
            data_phase = np.angle(data).reshape(-1, n_edges)
            sigma_edge = sigma.reshape(-1, n_edges)
            model_q = model_phase @ q_basis
            data_q = data_phase @ q_basis
            closure_resid_q = np.angle(np.exp(1j * (model_q - data_q)))
            coord_sigma2 = (sigma_edge**2 + sigma_floor**2) @ (q_basis**2)
            wq = np.clip(weight_norm / np.maximum(coord_sigma2, 1e-12), 0.02, 25.0)
            phase_coeff = ((wq * closure_resid_q) @ q_basis.T).reshape(-1)
            safe_model = np.where(np.abs(model) > 1e-4, model, 1e-4 + 0j)
            phase_y = 1j * phase_coeff / np.conj(safe_model)

            add_visibility_adjoint_samples(grad_grid, u, v, phase_y, fov_rad=fov_rad)
            phase_loss += float(np.mean(wq * closure_resid_q**2))
            n_resid += 1
        data_grad = adjoint_image(grad_grid)
        prior_grad = 2.0 * (x - prior)
        tv_grad = latest.tv_gradient(x)
        grad = data_grad / rms(data_grad) + PRIOR_WEIGHT * prior_grad / rms(prior_grad) + TV_WEIGHT * tv_grad / rms(tv_grad)
        x = project_simplex_positive(x - CLOSURE_STEP * grad, smooth_pix=0.18 if (iteration + 1) % 35 == 0 else 0.0)
        if iteration in {0, CLOSURE_N_ITER // 2, CLOSURE_N_ITER - 1}:
            history[f"closure_phase_loss_{iteration}"] = phase_loss / max(n_resid, 1)
    return normalize_stack(x), history


def radial_profile_corr(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> float:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    bins = np.linspace(0.0, np.max(np.abs(axis_uas)), 34)
    truth_n = base.normalize_for_display(truth)
    image_n = base.normalize_for_display(image)
    t_prof = []
    x_prof = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rr >= lo) & (rr < hi)
        if np.count_nonzero(mask) < 4:
            continue
        t_prof.append(float(np.mean(truth_n[mask])))
        x_prof.append(float(np.mean(image_n[mask])))
    if len(t_prof) < 3 or np.std(t_prof) == 0.0 or np.std(x_prof) == 0.0:
        return 0.0
    return float(np.corrcoef(t_prof, x_prof)[0, 1])


def metrics_for(image: np.ndarray, truth: np.ndarray, axis_uas: np.ndarray) -> dict[str, float]:
    metric = latest.image_metrics(SOURCE, truth, image, axis_uas)
    metric["radial_corr"] = radial_profile_corr(truth, image, axis_uas)
    return metric


def simulate_case(case: aug.NetworkCase) -> tuple[list[dict[str, np.ndarray]], dict, np.ndarray, np.ndarray]:
    old_loss = aug.FIBER_LOSS_DB_PER_KM
    old_baseline_fp = getattr(aug, "BASELINE_FALSE_POSITIVE", 0.0)
    old_mode_fp = getattr(aug, "MODE_FALSE_POSITIVE", 0.05)
    old_pair_fp = getattr(aug, "PAIR_FALSE_POSITIVE", 0.0)
    old_wt_fp = getattr(wt, "BASELINE_FALSE_POSITIVE", 0.0)
    old_wt_npix = wt.N_PIX
    old_wt_snr = wt.SNR_BOOST
    old_wt_days = wt.OBSERVING_DAYS
    aug.FIBER_LOSS_DB_PER_KM = FIBER_LOSS_DB_PER_KM
    aug.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    aug.MODE_FALSE_POSITIVE = MODE_FALSE_POSITIVE
    aug.PAIR_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    wt.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    wt.N_PIX = N_RML
    wt.SNR_BOOST = SNR_BOOST
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    try:
        with ngc.patched_source(SOURCE):
            return wt.simulate_bands(case)
    finally:
        aug.FIBER_LOSS_DB_PER_KM = old_loss
        aug.BASELINE_FALSE_POSITIVE = old_baseline_fp
        aug.MODE_FALSE_POSITIVE = old_mode_fp
        aug.PAIR_FALSE_POSITIVE = old_pair_fp
        wt.BASELINE_FALSE_POSITIVE = old_wt_fp
        wt.N_PIX = old_wt_npix
        wt.SNR_BOOST = old_wt_snr
        wt.OBSERVING_DAYS = old_wt_days


def reconstruct_case(case: aug.NetworkCase) -> dict:
    bands, stats, truth, axis_uas = simulate_case(case)
    prior = core_ring_prior(axis_uas, SOURCE)
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    images = {"truth": truth, "prior": normalize_stack(prior)}
    metrics = {"prior": metrics_for(images["prior"], truth, axis_uas)}
    histories = {}
    for strategy in ("split", "direct"):
        data, weight, occupied = make_gridded_data(bands, strategy, n=N_RML, fov_rad=fov_rad)
        images[f"{strategy}_dirty"] = normalize_stack(dirty_from_grid(data, weight))
        metrics[f"{strategy}_dirty"] = metrics_for(images[f"{strategy}_dirty"], truth, axis_uas)
        image, history = visibility_map_with_ring_prior(data, weight, prior)
        key = f"{strategy}_visibility_rml"
        images[key] = image
        metrics[key] = metrics_for(image, truth, axis_uas)
        histories[key] = history
        image, history = rml_reconstruct(data, weight, prior, mode="phase")
        key = f"{strategy}_phase_rml"
        images[key] = image
        metrics[key] = metrics_for(image, truth, axis_uas)
        histories[key] = history
        if strategy == "direct":
            image, history = closure_phase_only_rml(
                bands,
                case,
                strategy,
                prior,
                images[f"{strategy}_dirty"],
                fov_rad=fov_rad,
            )
            key = f"{strategy}_closure_phase_rml"
            images[key] = image
            metrics[key] = metrics_for(image, truth, axis_uas)
            histories[key] = history
        stats[f"{strategy}_occupied_cells"] = int(np.count_nonzero(occupied))
    stats.update(
        {
            "source": SOURCE.name,
            "n_rml": N_RML,
            "observing_days": OBSERVING_DAYS,
            "snr_boost": SNR_BOOST,
            "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": MODE_FALSE_POSITIVE,
            "pair_false_positive": PAIR_FALSE_POSITIVE,
            "noise_model": "pure fibre attenuation plus independent mode-local false positives",
            "rml_prior_weight": PRIOR_WEIGHT,
            "rml_prior_fourier_weight": PRIOR_FOURIER_WEIGHT,
            "rml_tv_weight": TV_WEIGHT,
            "rml_iterations": N_ITER,
            "closure_rml_iterations": CLOSURE_N_ITER,
            "closure_rml_objective": "phase-only in independent closure coordinates; visibility amplitudes treated as known calibration inputs, not an amplitude loss",
            "metrics": metrics,
            "histories": histories,
        }
    )
    return {"case": case, "stats": stats, "images": images, "axis_uas": axis_uas, "metrics": metrics}


def draw_layout_uv(ax_layout: plt.Axes, ax_uv: plt.Axes, result: dict) -> None:
    case = result["case"]
    stats = result["stats"]
    stations, _, _, is_added = aug.station_table_from_case(case)
    for added, marker, color, label in (
        (False, "o", "#005f73", "existing"),
        (True, "^", "#ae2012", "new/synthetic"),
    ):
        mask = is_added == added
        if np.any(mask):
            ax_layout.scatter(stations[mask, 0], stations[mask, 1], s=18, marker=marker, color=color, label=label)
    ax_layout.scatter([case.hub_km[0]], [case.hub_km[1]], s=42, marker="*", color="#ca6702", label="hub")
    ax_layout.set_aspect("equal", adjustable="box")
    ax_layout.set_xlabel("east (km)")
    ax_layout.set_ylabel("north (km)")
    ax_layout.set_title("Topology")
    ax_layout.legend(loc="best", frameon=False, fontsize=5.5)

    for wavelength, color in (("400", "#005f73"), ("800", "#ee9b00")):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax_uv.scatter(uu, vv, s=0.65, color=color, alpha=0.45, label=f"{wavelength} nm")
        ax_uv.scatter(-uu, -vv, s=0.65, color=color, alpha=0.28)
    ax_uv.set_aspect("equal", adjustable="box")
    ax_uv.set_xlabel(r"$u$ (G$\lambda$)")
    ax_uv.set_ylabel(r"$v$ (G$\lambda$)")
    ax_uv.set_title("UV coverage")
    ax_uv.legend(loc="upper right", frameon=False, fontsize=5.5)


def make_figure(results: list[dict]) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(11.3, 6.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 7, hspace=0.43, wspace=0.34)
    plt.rcParams.update(
        {
            "font.size": 6.9,
            "axes.labelsize": 6.7,
            "axes.titlesize": 7.3,
            "legend.fontsize": 5.5,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
        }
    )
    image_axes = []
    panels = [
        ("truth", "Input"),
        ("prior", "Core+ring prior"),
        ("direct_dirty", "Direct dirty"),
        ("direct_visibility_rml", "Pseudo-vis MAP"),
        ("direct_closure_phase_rml", "Closure-phase RML"),
    ]
    for row, result in enumerate(results):
        axis_uas = result["axis_uas"]
        extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
        draw_layout_uv(fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1]), result)
        row_label = "Real Maunakea top4+5" if row == 0 else "Synthetic 8-station optimal"
        for col, (key, title) in enumerate(panels, start=2):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(opt.normalize_blr_display(result["images"][key]), origin="lower", extent=extent, cmap="inferno")
            if key == "truth":
                ax.set_title(f"{row_label}\n{title}")
            else:
                metric = result["metrics"][key]
                ax.set_title(f"{title}\nBLR r={metric['blr_corr']:.2f}, rad r={metric['radial_corr']:.2f}")
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 2:
                ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.016,
        pad=0.011,
    )
    cbar.set_label("norm. brightness\n(BLR stretch)", fontsize=6.4)
    fig.suptitle(
        f"Visibility-/closure-phase-domain RML with strong core+ring prior: {SOURCE.name}, "
        f"{OBSERVING_DAYS} d, loss {FIBER_LOSS_DB_PER_KM:g} dB/km, "
        f"mode p_fp {MODE_FALSE_POSITIVE:g}, pair p_fp {PAIR_FALSE_POSITIVE:g}",
        fontsize=10.2,
        weight="bold",
        y=0.993,
    )
    tag = output_tag()
    png = OUTFIG / f"visibility_closure_domain_ring_prior_rml_{tag}.png"
    pdf = OUTFIG / f"visibility_closure_domain_ring_prior_rml_{tag}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    real_case = latest.load_case(REAL_LAYOUT)
    optimal_case = load_synthetic_case(OPTIMAL_LAYOUT)
    results = [reconstruct_case(real_case), reconstruct_case(optimal_case)]
    pdf, png = make_figure(results)
    summary = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "real_layout": str(REAL_LAYOUT),
        "optimal_layout": str(OPTIMAL_LAYOUT),
        "source": SOURCE.name,
        "n_rml": N_RML,
        "observing_days": OBSERVING_DAYS,
        "snr_boost": SNR_BOOST,
        "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
        "mode_false_positive": MODE_FALSE_POSITIVE,
        "pair_false_positive": PAIR_FALSE_POSITIVE,
        "noise_model": "pure fibre attenuation plus independent mode-local false positives",
        "rml_prior_weight": PRIOR_WEIGHT,
        "rml_prior_fourier_weight": PRIOR_FOURIER_WEIGHT,
        "rml_tv_weight": TV_WEIGHT,
        "cases": [
            {
                "case": result["case"].key,
                "title": result["case"].title,
                "stats": {
                    key: value
                    for key, value in result["stats"].items()
                    if key
                    in {
                        "n_station",
                        "n_baseline",
                        "n_closure",
                        "closure_rank_share",
                        "baseline_max_km",
                        "station_link_eff_min",
                        "station_link_eff_max",
                        "coverage_400nm_half_range_g_lambda",
                        "coverage_800nm_half_range_g_lambda",
                        "direct_occupied_cells",
                        "split_occupied_cells",
                    }
                },
                "metrics": result["metrics"],
            }
            for result in results
        ],
    }
    out = OUTFIG / f"visibility_closure_domain_ring_prior_rml_{output_tag()}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out)
    print(png)
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
