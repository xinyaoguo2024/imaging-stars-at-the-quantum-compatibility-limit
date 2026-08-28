from __future__ import annotations

import csv
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


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

REAL_LAYOUT = OUTFIG / "maunakea_top4_plus5_ngc4151_layout.json"
OPTIMAL_LAYOUT = OUTFIG / "optimized_array_topology_u50v50_lowuv15_near10_5_ydown1p5_hub_m3_m4.json"

SOURCE = ngc.NGC4151
N_RML = int(os.environ.get("PHASE_RML_N_PIX", "128"))
N_ITER = int(os.environ.get("PHASE_RML_N_ITER", "110"))
STEP = float(os.environ.get("PHASE_RML_STEP", "0.014"))
TV_WEIGHT = float(os.environ.get("PHASE_RML_TV_WEIGHT", "0.06"))
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "30"))
FIBER_LOSS_DB_PER_KM = float(os.environ.get("FIBER_LOSS_DB_PER_KM", "0.2"))
MODE_FALSE_POSITIVE = float(os.environ.get("MODE_FALSE_POSITIVE", os.environ.get("STATION_FALSE_POSITIVE", "0.05")))
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", os.environ.get("BASELINE_FALSE_POSITIVE", "0.0")))
PRIOR_STRENGTHS = tuple(float(x) for x in os.environ.get("PHASE_RML_PRIORS", "3.0,1.2,0.4,0.0").split(","))
SNR_GAINS = tuple(float(x) for x in os.environ.get("PHASE_RML_SNR_GAINS", "1,3,10,30,100").split(","))
SNR_SCAN_PRIOR = float(os.environ.get("PHASE_RML_SNR_SCAN_PRIOR", "1.2"))
PRIOR_SCAN_STRATEGIES = tuple(os.environ.get("PHASE_RML_PRIOR_STRATEGIES", "direct").split(","))
SNR_SCAN_STRATEGIES = tuple(os.environ.get("PHASE_RML_SNR_STRATEGIES", "all,split,direct").split(","))
SKIP_PRIOR_SCAN = os.environ.get("PHASE_RML_SKIP_PRIOR", "0") == "1"


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


def core_ring_prior(axis_uas: np.ndarray, source: ngc.SourceModel) -> np.ndarray:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    core = np.exp(-0.5 * (rr / max(source.disc_sigma_major_uas, 7.5)) ** 2)
    ring = np.exp(-0.5 * ((rr - source.blr_radius_uas) / max(source.blr_width_uas, 6.0)) ** 2)
    core /= np.sum(core)
    ring /= np.sum(ring)
    prior = 0.42 * core + 0.58 * ring
    prior = np.clip(prior, 0.0, None)
    prior /= np.sum(prior)
    return prior


def project_flux_positive(image: np.ndarray, smooth_pix: float = 0.0) -> np.ndarray:
    out = np.clip(image, 0.0, None)
    if smooth_pix > 0.15:
        out = base.gaussian_filter(out, smooth_pix)
        out = np.clip(out, 0.0, None)
    total = float(np.sum(out))
    if not np.isfinite(total) or total <= 0.0:
        out = np.ones_like(image)
        total = float(np.sum(out))
    return out / total


def normalize_grad(grad: np.ndarray) -> np.ndarray:
    scale = float(np.sqrt(np.mean(np.asarray(grad, dtype=float) ** 2)))
    if not np.isfinite(scale) or scale <= 1e-30:
        return np.zeros_like(grad)
    return grad / scale


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


def quick_phase_dirty(
    bands: list[dict[str, np.ndarray]],
    strategy: str,
    truth: np.ndarray,
) -> np.ndarray:
    old_npix = wt.N_PIX
    wt.N_PIX = truth.shape[0]
    try:
        dirty, _ = latest.stack_dirty_psf(bands, strategy, truth, fill=True)
    finally:
        wt.N_PIX = old_npix
    return project_flux_positive(dirty - np.percentile(dirty, 1.0), smooth_pix=0.7)


def radial_profile_corr(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> float:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    bins = np.linspace(0.0, np.max(np.abs(axis_uas)), 34)
    truth_n = base.normalize_for_display(truth)
    image_n = base.normalize_for_display(image)
    t_prof: list[float] = []
    x_prof: list[float] = []
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


def simulate_case(case: aug.NetworkCase, snr_gain: float) -> tuple[list[dict[str, np.ndarray]], dict, np.ndarray, np.ndarray]:
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
    wt.SNR_BOOST = snr_gain
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


def phase_only_rml(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    strategy: str,
    prior: np.ndarray,
    start: np.ndarray,
    *,
    prior_strength: float,
    fov_rad: float,
) -> tuple[np.ndarray, dict[str, float]]:
    stations, _, _, _ = aug.station_table_from_case(case)
    edges = base.edge_list(len(stations))
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, len(stations)))
    n_edges = len(edges)
    uv_axis = np.fft.fftshift(np.fft.fftfreq(N_RML, d=fov_rad / N_RML))
    sigma_values = np.concatenate([band[f"sigma_{strategy}"] for band in bands])
    sigma_floor = max(float(np.nanmedian(sigma_values)) * 0.35, 0.10)
    inv_var = 1.0 / (sigma_values**2 + sigma_floor**2)
    weight_norm = 1.0 / max(float(np.nanmedian(inv_var)), 1e-30)

    mix = prior_strength / (prior_strength + 1.0) if prior_strength > 0.0 else 0.0
    x = project_flux_positive(mix * prior + (1.0 - mix) * start, smooth_pix=0.3)
    history: dict[str, float] = {}

    for iteration in range(N_ITER):
        vis_grid = fft_vis(x)
        grad_grid = np.zeros_like(vis_grid)
        phase_loss = 0.0
        n_bucket = 0
        for band in bands:
            u = band["u"]
            v = band["v"]
            model = base.interp_vis(vis_grid, uv_axis, u, v)
            data = band[f"vis_{strategy}"]
            sigma = band[f"sigma_{strategy}"].reshape(-1, n_edges)
            model_phase = np.angle(model).reshape(-1, n_edges)
            data_phase = np.angle(data).reshape(-1, n_edges)
            if strategy == "all":
                resid_edge = np.angle(np.exp(1j * (model_phase - data_phase)))
                w_edge = np.clip(weight_norm / (sigma**2 + sigma_floor**2), 0.02, 25.0)
                phase_coeff = (w_edge * resid_edge).reshape(-1)
                phase_loss += float(np.mean(w_edge * resid_edge**2))
            else:
                model_q = model_phase @ q_basis
                data_q = data_phase @ q_basis
                resid_q = np.angle(np.exp(1j * (model_q - data_q)))
                coord_sigma2 = (sigma**2 + sigma_floor**2) @ (q_basis**2)
                wq = np.clip(weight_norm / np.maximum(coord_sigma2, 1e-12), 0.02, 25.0)
                phase_coeff = ((wq * resid_q) @ q_basis.T).reshape(-1)
                phase_loss += float(np.mean(wq * resid_q**2))
            safe_model = np.where(np.abs(model) > 1e-4, model, 1e-4 + 0j)
            phase_y = 1j * phase_coeff / np.conj(safe_model)
            add_visibility_adjoint_samples(grad_grid, u, v, phase_y, fov_rad=fov_rad)
            n_bucket += 1

        data_grad = adjoint_image(grad_grid)
        prior_grad = 2.0 * (x - prior)
        tv_grad = latest.tv_gradient(x)
        grad = normalize_grad(data_grad)
        if prior_strength > 0.0:
            grad += prior_strength * normalize_grad(prior_grad)
        grad += TV_WEIGHT * normalize_grad(tv_grad)
        x = project_flux_positive(x - STEP * grad, smooth_pix=0.16 if (iteration + 1) % 35 == 0 else 0.0)
        if iteration in {0, N_ITER // 2, N_ITER - 1}:
            history[f"phase_loss_{iteration}"] = phase_loss / max(n_bucket, 1)
    return latest.normalize_stack(x), history


def run_recon_grid(
    case: aug.NetworkCase,
    *,
    snr_gain: float,
    prior_strengths: tuple[float, ...],
    strategies: tuple[str, ...],
) -> dict:
    bands, stats, truth, axis_uas = simulate_case(case, snr_gain)
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    prior = core_ring_prior(axis_uas, SOURCE)
    images: dict[tuple[str, float], np.ndarray] = {}
    metrics: dict[tuple[str, float], dict[str, float]] = {}
    histories: dict[tuple[str, float], dict[str, float]] = {}
    starts = {strategy: quick_phase_dirty(bands, strategy, truth) for strategy in strategies}
    for strategy in strategies:
        for prior_strength in prior_strengths:
            image, history = phase_only_rml(
                bands,
                case,
                strategy,
                prior,
                starts[strategy],
                prior_strength=prior_strength,
                fov_rad=fov_rad,
            )
            key = (strategy, prior_strength)
            images[key] = image
            metrics[key] = metrics_for(image, truth, axis_uas)
            histories[key] = history
    stats.update(
        {
            "source": SOURCE.name,
            "observing_days": OBSERVING_DAYS,
            "snr_gain": snr_gain,
            "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": MODE_FALSE_POSITIVE,
            "pair_false_positive": PAIR_FALSE_POSITIVE,
            "noise_model": "pure fibre attenuation plus independent mode-local false positives",
            "phase_rml_grid": N_RML,
            "phase_rml_iterations": N_ITER,
            "phase_rml_step": STEP,
            "phase_rml_tv_weight": TV_WEIGHT,
            "phase_rml_objective": "phase-only; all-vis uses edge phases, closure strategies use independent q^T phase coordinates; amplitudes are treated as known calibration inputs, not fitted residuals",
        }
    )
    return {
        "case": case,
        "bands": bands,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "prior": latest.normalize_stack(prior),
        "starts": starts,
        "images": images,
        "metrics": metrics,
        "histories": histories,
    }


def nice_strategy(strategy: str) -> str:
    return {"all": "All-vis + piston", "split": "Edge-first closure", "direct": "Direct closure"}[strategy]


def plot_prior_scan(results: list[dict]) -> tuple[Path, Path]:
    strategies = PRIOR_SCAN_STRATEGIES
    n_rows = len(results) * len(strategies)
    n_cols = len(PRIOR_STRENGTHS) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.05 * n_cols, 1.82 * n_rows), constrained_layout=True)
    plt.rcParams.update({"font.size": 6.9, "axes.titlesize": 7.2, "axes.labelsize": 6.6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5})
    image_axes = []
    row = 0
    for result in results:
        axis_uas = result["axis_uas"]
        extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
        case_label = "Real topology" if result["case"].key.startswith("maunakea") else "8-station optimal"
        for strategy in strategies:
            panels = [("truth", result["truth"], "Input"), ("prior", result["prior"], "Core+ring prior")]
            for prior_strength in PRIOR_STRENGTHS:
                panels.append((f"{strategy}_{prior_strength:g}", result["images"][(strategy, prior_strength)], rf"$\lambda_p={prior_strength:g}$"))
            for col, (_, image, title) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
                if col < 2:
                    ax.set_title(title)
                else:
                    metric = result["metrics"][(strategy, PRIOR_STRENGTHS[col - 2])]
                    ax.set_title(f"{title}\nBLR={metric['blr_corr']:.2f}, rad={metric['radial_corr']:.2f}")
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"{case_label}\n{nice_strategy(strategy)}", fontsize=7.0)
                image_axes.append(ax)
            row += 1
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap="inferno"), ax=image_axes, fraction=0.012, pad=0.008)
    cbar.set_label("norm. brightness", fontsize=6.4)
    fig.suptitle(
        f"Phase-only RML prior scan, SNR gain 1, loss {FIBER_LOSS_DB_PER_KM:g} dB/km, "
        f"mode p_fp {MODE_FALSE_POSITIVE:g}, pair p_fp {PAIR_FALSE_POSITIVE:g}",
        fontsize=10.0,
        weight="bold",
    )
    tag = (
        f"phase_only_rml_prior_scan_{SOURCE.key}_{OBSERVING_DAYS}d_loss{FIBER_LOSS_DB_PER_KM:g}"
        f"_modefp{MODE_FALSE_POSITIVE:g}_pairfp{PAIR_FALSE_POSITIVE:g}"
    ).replace(".", "p")
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_snr_scan(case_results: dict[str, dict[float, dict]]) -> tuple[Path, Path]:
    strategies = SNR_SCAN_STRATEGIES
    fig, axes = plt.subplots(len(case_results) * len(strategies), len(SNR_GAINS) + 1, figsize=(2.0 * (len(SNR_GAINS) + 1), 1.82 * len(case_results) * len(strategies)), constrained_layout=True)
    plt.rcParams.update({"font.size": 6.9, "axes.titlesize": 7.2, "axes.labelsize": 6.6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5})
    image_axes = []
    row = 0
    for case_label, by_snr in case_results.items():
        first = by_snr[SNR_GAINS[0]]
        axis_uas = first["axis_uas"]
        extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
        for strategy in strategies:
            ax = axes[row, 0]
            ax.imshow(opt.normalize_blr_display(first["truth"]), origin="lower", extent=extent, cmap="inferno")
            ax.set_title("Input")
            ax.set_ylabel(f"{case_label}\n{nice_strategy(strategy)}", fontsize=7.0)
            ax.set_xticks([])
            ax.set_yticks([])
            image_axes.append(ax)
            for col, snr_gain in enumerate(SNR_GAINS, start=1):
                result = by_snr[snr_gain]
                image = result["images"][(strategy, SNR_SCAN_PRIOR)]
                metric = result["metrics"][(strategy, SNR_SCAN_PRIOR)]
                ax = axes[row, col]
                ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
                ax.set_title(f"SNR x{snr_gain:g}\nBLR={metric['blr_corr']:.2f}, rad={metric['radial_corr']:.2f}")
                ax.set_xticks([])
                ax.set_yticks([])
                image_axes.append(ax)
            row += 1
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap="inferno"), ax=image_axes, fraction=0.012, pad=0.008)
    cbar.set_label("norm. brightness", fontsize=6.4)
    fig.suptitle(rf"Phase-only RML SNR scan at $\lambda_p={SNR_SCAN_PRIOR:g}$", fontsize=10.0, weight="bold")
    tag = (
        f"phase_only_rml_snr_scan_{SOURCE.key}_{OBSERVING_DAYS}d_prior{SNR_SCAN_PRIOR:g}"
        f"_loss{FIBER_LOSS_DB_PER_KM:g}_modefp{MODE_FALSE_POSITIVE:g}_pairfp{PAIR_FALSE_POSITIVE:g}"
    ).replace(".", "p")
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_metrics_csv(rows: list[dict[str, float | str]]) -> Path:
    if SKIP_PRIOR_SCAN:
        tag = f"phase_only_rml_prior_snr_scan_metrics_{SOURCE.key}_{OBSERVING_DAYS}d_prior{SNR_SCAN_PRIOR:g}"
        path = OUTFIG / f"{tag.replace('.', 'p')}.csv"
    else:
        path = OUTFIG / f"phase_only_rml_prior_snr_scan_metrics_{SOURCE.key}_{OBSERVING_DAYS}d.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    cases = [
        latest.load_case(REAL_LAYOUT),
        load_synthetic_case(OPTIMAL_LAYOUT),
    ]
    strategies = PRIOR_SCAN_STRATEGIES
    metric_rows: list[dict[str, float | str]] = []

    prior_pdf = prior_png = None
    if not SKIP_PRIOR_SCAN:
        prior_results = []
        for case in cases:
            print(f"[prior scan] case={case.key}, strategies={strategies}, priors={PRIOR_STRENGTHS}", flush=True)
            result = run_recon_grid(case, snr_gain=1.0, prior_strengths=PRIOR_STRENGTHS, strategies=strategies)
            prior_results.append(result)
            for (strategy, prior_strength), metric in result["metrics"].items():
                metric_rows.append(
                    {
                        "scan": "prior",
                        "case": case.key,
                        "global_snr_gain": 1.0,
                        "strategy": strategy,
                        "prior_strength": prior_strength,
                        **metric,
                    }
                )
        prior_pdf, prior_png = plot_prior_scan(prior_results)

    snr_results: dict[str, dict[float, dict]] = {}
    for case in cases:
        case_label = "Real topology" if case.key.startswith("maunakea") else "8-station optimal"
        snr_results[case_label] = {}
        for snr_gain in SNR_GAINS:
            print(f"[snr scan] case={case.key}, global_snr_gain={snr_gain:g}, strategies={SNR_SCAN_STRATEGIES}", flush=True)
            result = run_recon_grid(case, snr_gain=snr_gain, prior_strengths=(SNR_SCAN_PRIOR,), strategies=SNR_SCAN_STRATEGIES)
            snr_results[case_label][snr_gain] = result
            for (strategy, prior_strength), metric in result["metrics"].items():
                metric_rows.append(
                    {
                        "scan": "snr",
                        "case": case.key,
                        "global_snr_gain": snr_gain,
                        "strategy": strategy,
                        "prior_strength": prior_strength,
                        **metric,
                    }
                )
    snr_pdf, snr_png = plot_snr_scan(snr_results)
    csv_path = write_metrics_csv(metric_rows)

    summary = {
        "source": SOURCE.name,
        "objective": "phase-only RML; amplitudes known/calibrated but not fitted as residuals",
        "observing_days": OBSERVING_DAYS,
        "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
        "mode_false_positive": MODE_FALSE_POSITIVE,
        "pair_false_positive": PAIR_FALSE_POSITIVE,
        "noise_model": "pure fibre attenuation plus independent mode-local false positives",
        "prior_strengths": list(PRIOR_STRENGTHS),
        "global_snr_gains": list(SNR_GAINS),
        "global_snr_gain_definition": "One multiplicative gain applied at data simulation to all measurement Fisher/noise channels. The residual all-vis piston drift is kept as a separate systematic floor.",
        "snr_scan_prior": SNR_SCAN_PRIOR,
        "skip_prior_scan": SKIP_PRIOR_SCAN,
        "figures": {
            "prior_scan_png": str(prior_png) if prior_png is not None else None,
            "prior_scan_pdf": str(prior_pdf) if prior_pdf is not None else None,
            "snr_scan_png": str(snr_png),
            "snr_scan_pdf": str(snr_pdf),
            "metrics_csv": str(csv_path),
        },
    }
    if SKIP_PRIOR_SCAN:
        summary_tag = f"phase_only_rml_prior_snr_scan_summary_{SOURCE.key}_{OBSERVING_DAYS}d_prior{SNR_SCAN_PRIOR:g}"
        out = OUTFIG / f"{summary_tag.replace('.', 'p')}.json"
    else:
        out = OUTFIG / f"phase_only_rml_prior_snr_scan_summary_{SOURCE.key}_{OBSERVING_DAYS}d.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out)
    if prior_png is not None:
        print(prior_png)
    print(snr_png)
    print(csv_path)


if __name__ == "__main__":
    main()
