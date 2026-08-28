from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as case_lib
import latest_maunakea_closure_snr_clean_rml as latest
import parametric_closure_rml_crescent_core as prm
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

SOURCE = ngc.NGC4151
N_RML = int(os.environ.get("WEAK_RML_N_PIX", "96"))
N_ITER = int(os.environ.get("WEAK_RML_N_ITER", "220"))
STEP = float(os.environ.get("WEAK_RML_STEP", "0.016"))
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "30"))
N_TIME_WINDOWS = int(os.environ.get("N_TIME_WINDOWS", "36"))
EXPOSURE_S = float(os.environ.get("EXPOSURE_S", "600.0"))
EXPOSURE_GAP_S = float(os.environ.get("EXPOSURE_GAP_S", "150.0"))
SNR_BOOST = float(os.environ.get("AUGMENTED_SNR_BOOST", "1.0"))
FIBER_LOSS_DB_PER_KM = float(os.environ.get("FIBER_LOSS_DB_PER_KM", "0.20"))
MODE_FALSE_POSITIVE = float(os.environ.get("MODE_FALSE_POSITIVE", "0.05"))
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", "0.0"))

# Weak EHT-style regularization.  These are deliberately not ring priors.
BROAD_PRIOR_FWHM_UAS = float(os.environ.get("WEAK_RML_PRIOR_FWHM_UAS", "105.0"))
COMMON_AMP_WEIGHT = float(os.environ.get("WEAK_RML_COMMON_AMP_WEIGHT", "0.025"))
MEM_WEIGHT = float(os.environ.get("WEAK_RML_MEM_WEIGHT", "0.008"))
TV_WEIGHT = float(os.environ.get("WEAK_RML_TV_WEIGHT", "0.028"))
TSV_WEIGHT = float(os.environ.get("WEAK_RML_TSV_WEIGHT", "0.012"))
L2_PRIOR_WEIGHT = float(os.environ.get("WEAK_RML_L2_PRIOR_WEIGHT", "0.004"))
AMP_REL_SIGMA = float(os.environ.get("WEAK_RML_AMP_REL_SIGMA", "0.05"))
AMP_ABS_FLOOR = float(os.environ.get("WEAK_RML_AMP_ABS_FLOOR", "0.012"))
PHASE_FLOOR_RAD = float(os.environ.get("WEAK_RML_PHASE_FLOOR_RAD", "0.04"))
WEIGHT_CLIP = tuple(float(x) for x in os.environ.get("WEAK_RML_WEIGHT_CLIP", "0.02,40").split(","))
STRATEGIES = ("all", "split", "direct")


def configure_simulation() -> None:
    aug.OBSERVING_DAYS = OBSERVING_DAYS
    aug.N_TIME_WINDOWS = N_TIME_WINDOWS
    aug.EXPOSURE_S = EXPOSURE_S
    aug.EXPOSURE_GAP_S = EXPOSURE_GAP_S
    aug.FIBER_LOSS_DB_PER_KM = FIBER_LOSS_DB_PER_KM
    aug.FIBER_LENGTH_SCALE = prm.FIBER_LENGTH_SCALE
    aug.MODE_FALSE_POSITIVE = MODE_FALSE_POSITIVE
    aug.PAIR_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    aug.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    wt.N_PIX = N_RML
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    wt.SNR_BOOST = SNR_BOOST
    wt.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE


def make_hawaii_remote_case(remote_count: int) -> aug.NetworkCase:
    full = case_lib.load_maunakea_case()
    core = [tel for tel in full.telescopes if not tel.is_added]
    added = [tel for tel in full.telescopes if tel.is_added]
    if remote_count == 3:
        selected = added[-3:]
    elif remote_count == 4:
        farthest = max(
            added,
            key=lambda tel: np.linalg.norm(np.array([tel.x_km, tel.y_km]) - np.array(full.hub_km, dtype=float)),
        )
        selected = [tel for tel in added if tel.name != farthest.name]
    else:
        raise ValueError("remote_count must be 3 or 4")
    return aug.NetworkCase(
        key=f"hawaii_top4_remote{remote_count}_ngc4151",
        title=f"Maunakea top-four core + {remote_count} remote 5 m stations",
        latitude_deg=full.latitude_deg,
        center_latlon=full.center_latlon,
        telescopes=core + selected,
        hub_km=full.hub_km,
        optimization_score=full.optimization_score,
    )


def make_cases() -> list[aug.NetworkCase]:
    return [
        case_lib.load_optimal8_case(),
        make_hawaii_remote_case(3),
        make_hawaii_remote_case(4),
    ]


def simulate_case(case: aug.NetworkCase):
    configure_simulation()
    with ngc.patched_source(SOURCE):
        return wt.simulate_bands(case)


def project_flux_positive(image: np.ndarray, smooth_pix: float = 0.0) -> np.ndarray:
    out = np.clip(image, 0.0, None)
    if smooth_pix > 0.05:
        out = np.clip(base.gaussian_filter(out, smooth_pix), 0.0, None)
    total = float(np.sum(out))
    if not np.isfinite(total) or total <= 1e-300:
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


def broad_gaussian_prior(axis_uas: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    sigma = BROAD_PRIOR_FWHM_UAS / 2.355
    prior = np.exp(-0.5 * (xx * xx + yy * yy) / max(sigma * sigma, 1e-12))
    return project_flux_positive(prior)


def bilinear_adjoint_samples(
    grid: np.ndarray,
    uv_axis: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coeff: np.ndarray,
) -> None:
    n = grid.shape[0]
    du = uv_axis[1] - uv_axis[0]

    def scatter(us: np.ndarray, vs: np.ndarray, cs: np.ndarray) -> None:
        fu = (us - uv_axis[0]) / du
        fv = (vs - uv_axis[0]) / du
        iu = np.floor(fu).astype(int)
        iv = np.floor(fv).astype(int)
        valid = (iu >= 0) & (iu < n - 1) & (iv >= 0) & (iv < n - 1) & np.isfinite(cs)
        if not np.any(valid):
            return
        iu = iu[valid]
        iv = iv[valid]
        tu = fu[valid] - iu
        tv = fv[valid] - iv
        cv = cs[valid]
        flat = grid.reshape(-1)
        for dx, dy, weight in (
            (0, 0, (1.0 - tu) * (1.0 - tv)),
            (1, 0, tu * (1.0 - tv)),
            (0, 1, (1.0 - tu) * tv),
            (1, 1, tu * tv),
        ):
            np.add.at(flat, (iv + dy) * n + (iu + dx), weight * cv)

    scatter(u, v, coeff)
    scatter(-u, -v, np.conj(coeff))


def tsv_gradient(image: np.ndarray) -> np.ndarray:
    dx = np.roll(image, -1, axis=1) - image
    dy = np.roll(image, -1, axis=0) - image
    return (
        2.0 * image
        - np.roll(image, 1, axis=1)
        - np.roll(image, -1, axis=1)
        + 2.0 * image
        - np.roll(image, 1, axis=0)
        - np.roll(image, -1, axis=0)
        + 0.0 * (dx + dy)
    )


def radial_profile_corr(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> float:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    bins = np.linspace(0.0, np.max(np.abs(axis_uas)), 34)
    truth_n = base.normalize_for_display(truth)
    image_n = base.normalize_for_display(image)
    t_prof: list[float] = []
    i_prof: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rr >= lo) & (rr < hi)
        if np.count_nonzero(mask) < 4:
            continue
        t_prof.append(float(np.mean(truth_n[mask])))
        i_prof.append(float(np.mean(image_n[mask])))
    if len(t_prof) < 3 or np.std(t_prof) <= 0 or np.std(i_prof) <= 0:
        return 0.0
    return float(np.corrcoef(t_prof, i_prof)[0, 1])


def metrics_for(image: np.ndarray, truth: np.ndarray, axis_uas: np.ndarray) -> dict[str, float]:
    metric = latest.image_metrics(SOURCE, truth, image, axis_uas)
    metric["radial_corr"] = radial_profile_corr(truth, image, axis_uas)
    return metric


class WeakPriorData:
    def __init__(self, bands: list[dict[str, np.ndarray]], case: aug.NetworkCase, axis_uas: np.ndarray):
        stations, _, _, _ = aug.station_table_from_case(case)
        self.edges = base.edge_list(len(stations))
        self.n_edges = len(self.edges)
        self.w_basis = base.root_cycle_basis(self.edges, len(stations))
        self.fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
        self.uv_axis = np.fft.fftshift(np.fft.fftfreq(len(axis_uas), d=self.fov_rad / len(axis_uas)))
        self.bands = bands

    @staticmethod
    def normalized_weights(var: np.ndarray) -> np.ndarray:
        w = 1.0 / np.maximum(var, 1e-12)
        finite = np.isfinite(w) & (w > 0.0)
        if np.any(finite):
            w = w / max(float(np.nanmedian(w[finite])), 1e-30)
        return np.clip(w, WEIGHT_CLIP[0], WEIGHT_CLIP[1])

    def objective(self, image: np.ndarray, strategy: str, prior: np.ndarray) -> tuple[float, dict[str, float]]:
        vis_grid = fft_vis(image)
        data_loss = 0.0
        amp_loss = 0.0
        n_bucket = 0
        for band in self.bands:
            model = base.interp_vis(vis_grid, self.uv_axis, band["u"], band["v"])
            amp_data = np.asarray(band["amp"], dtype=float)
            amp_model = np.abs(model)
            amp_sigma = AMP_REL_SIGMA * np.maximum(amp_data, AMP_ABS_FLOOR)
            amp_w = self.normalized_weights(amp_sigma**2)
            amp_loss += float(np.mean(amp_w * (amp_model - amp_data) ** 2))
            if strategy == "all":
                data = band["vis_all"]
                sigma = np.maximum(np.asarray(band["sigma_all"], dtype=float), PHASE_FLOOR_RAD)
                var = amp_sigma**2 + (np.maximum(amp_data, AMP_ABS_FLOOR) * sigma) ** 2
                w = self.normalized_weights(var)
                data_loss += float(np.mean(w * np.abs(model - data) ** 2))
            else:
                model_phase = np.angle(model).reshape(-1, self.n_edges)
                data_phase = np.angle(band[f"vis_{strategy}"]).reshape(-1, self.n_edges)
                sigma = np.maximum(np.asarray(band[f"sigma_{strategy}"]).reshape(-1, self.n_edges), PHASE_FLOOR_RAD)
                model_loop = np.angle(np.exp(1j * (model_phase @ self.w_basis)))
                data_loop = np.angle(np.exp(1j * (data_phase @ self.w_basis)))
                residual = np.exp(1j * model_loop) - np.exp(1j * data_loop)
                sigma_loop2 = (sigma**2) @ (self.w_basis**2)
                w = self.normalized_weights(sigma_loop2)
                data_loss += float(np.mean(w * np.abs(residual) ** 2))
            n_bucket += 1
        data_loss /= max(n_bucket, 1)
        amp_loss /= max(n_bucket, 1)
        mem = float(np.sum(image * np.log((image + 1e-12) / (prior + 1e-12))))
        tv = float(np.mean(np.sqrt((np.roll(image, -1, 1) - image) ** 2 + (np.roll(image, -1, 0) - image) ** 2 + 1e-8)))
        tsv = float(np.mean((np.roll(image, -1, 1) - image) ** 2 + (np.roll(image, -1, 0) - image) ** 2))
        l2 = float(np.mean((image - prior) ** 2))
        common_amp = 0.0 if strategy == "all" else COMMON_AMP_WEIGHT * amp_loss
        total = data_loss + common_amp + MEM_WEIGHT * mem + TV_WEIGHT * tv + TSV_WEIGHT * tsv + L2_PRIOR_WEIGHT * l2
        return total, {
            "data_loss": data_loss,
            "amp_loss": amp_loss,
            "mem": mem,
            "tv": tv,
            "tsv": tsv,
            "l2_prior": l2,
            "common_amp_weight": 0.0 if strategy == "all" else COMMON_AMP_WEIGHT,
        }

    def gradient(self, image: np.ndarray, strategy: str, prior: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        vis_grid = fft_vis(image)
        data_grad_grid = np.zeros_like(vis_grid)
        amp_grad_grid = np.zeros_like(vis_grid)
        losses = {"data_loss": 0.0, "amp_loss": 0.0}
        n_bucket = 0
        eps = 1e-6
        for band in self.bands:
            u = band["u"]
            v = band["v"]
            model = base.interp_vis(vis_grid, self.uv_axis, u, v)
            amp_data = np.asarray(band["amp"], dtype=float)
            amp_model = np.abs(model)
            amp_sigma = AMP_REL_SIGMA * np.maximum(amp_data, AMP_ABS_FLOOR)
            amp_w = self.normalized_weights(amp_sigma**2)
            amp_resid = amp_model - amp_data
            amp_coeff = amp_w * amp_resid * model / np.maximum(amp_model, eps)
            bilinear_adjoint_samples(amp_grad_grid, self.uv_axis, u, v, amp_coeff)
            losses["amp_loss"] += float(np.mean(amp_w * amp_resid**2))

            if strategy == "all":
                data = band["vis_all"]
                sigma = np.maximum(np.asarray(band["sigma_all"], dtype=float), PHASE_FLOOR_RAD)
                var = amp_sigma**2 + (np.maximum(amp_data, AMP_ABS_FLOOR) * sigma) ** 2
                w = self.normalized_weights(var)
                residual = model - data
                bilinear_adjoint_samples(data_grad_grid, self.uv_axis, u, v, w * residual)
                losses["data_loss"] += float(np.mean(w * np.abs(residual) ** 2))
            else:
                model_phase = np.angle(model).reshape(-1, self.n_edges)
                data_phase = np.angle(band[f"vis_{strategy}"]).reshape(-1, self.n_edges)
                sigma = np.maximum(np.asarray(band[f"sigma_{strategy}"]).reshape(-1, self.n_edges), PHASE_FLOOR_RAD)
                model_loop = model_phase @ self.w_basis
                data_loop = data_phase @ self.w_basis
                delta = np.angle(np.exp(1j * (model_loop - data_loop)))
                sigma_loop2 = (sigma**2) @ (self.w_basis**2)
                w = self.normalized_weights(sigma_loop2)
                loop_coeff = 2.0 * w * np.sin(delta)
                edge_coeff = loop_coeff @ self.w_basis.T
                safe_model = np.where(np.abs(model) > eps, model, eps + 0j)
                phase_coeff = 1j * edge_coeff.reshape(-1) / np.conj(safe_model)
                bilinear_adjoint_samples(data_grad_grid, self.uv_axis, u, v, phase_coeff)
                losses["data_loss"] += float(np.mean(w * np.abs(np.exp(1j * model_loop) - np.exp(1j * data_loop)) ** 2))
            n_bucket += 1
        data_grad = adjoint_image(data_grad_grid)
        amp_grad = adjoint_image(amp_grad_grid)
        data_part = normalize_grad(data_grad)
        if strategy != "all" and COMMON_AMP_WEIGHT > 0.0:
            data_part += COMMON_AMP_WEIGHT * normalize_grad(amp_grad)

        mem_grad = np.log((image + 1e-12) / (prior + 1e-12))
        tv_grad = latest.tv_gradient(image)
        tsv_grad = tsv_gradient(image)
        l2_grad = 2.0 * (image - prior)
        reg_grad = (
            MEM_WEIGHT * normalize_grad(mem_grad)
            + TV_WEIGHT * normalize_grad(tv_grad)
            + TSV_WEIGHT * normalize_grad(tsv_grad)
            + L2_PRIOR_WEIGHT * normalize_grad(l2_grad)
        )
        for key in losses:
            losses[key] /= max(n_bucket, 1)
        return data_part + reg_grad, losses


def quick_dirty_start(bands: list[dict[str, np.ndarray]], strategy: str, truth: np.ndarray) -> np.ndarray:
    old_npix = wt.N_PIX
    wt.N_PIX = truth.shape[0]
    try:
        dirty, _ = latest.stack_dirty_psf(bands, strategy, truth, fill=False)
    finally:
        wt.N_PIX = old_npix
    return project_flux_positive(dirty - np.percentile(dirty, 1.0), smooth_pix=0.5)


def weak_prior_rml(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    strategy: str,
    prior: np.ndarray,
    truth: np.ndarray,
    axis_uas: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    data = WeakPriorData(bands, case, axis_uas)
    start = quick_dirty_start(bands, strategy, truth)
    image = project_flux_positive(0.72 * prior + 0.28 * start, smooth_pix=0.2)
    history: dict[str, float] = {}
    for iteration in range(N_ITER):
        grad, losses = data.gradient(image, strategy, prior)
        candidates = (grad, -grad)
        current, _ = data.objective(image, strategy, prior)
        best_image = image
        best_value = current
        for direction in candidates:
            step = STEP
            for _ in range(7):
                trial = project_flux_positive(image - step * direction, smooth_pix=0.0)
                value, _parts = data.objective(trial, strategy, prior)
                if np.isfinite(value) and value < best_value:
                    best_value = value
                    best_image = trial
                    break
                step *= 0.5
        image = best_image
        if (iteration + 1) % 55 == 0:
            image = project_flux_positive(base.gaussian_filter(image, 0.10), smooth_pix=0.0)
        if iteration in {0, N_ITER // 2, N_ITER - 1}:
            value, parts = data.objective(image, strategy, prior)
            history[f"objective_{iteration}"] = value
            for key, item in parts.items():
                history[f"{key}_{iteration}"] = item
            history[f"grad_data_loss_{iteration}"] = losses["data_loss"]
            history[f"grad_amp_loss_{iteration}"] = losses["amp_loss"]
    return project_flux_positive(image), history


def run_case(case: aug.NetworkCase) -> dict:
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = simulate_case(case)
    prior = broad_gaussian_prior(axis_uas)
    images = {"truth": truth, "broad_gaussian_prior": prior}
    metrics = {"broad_gaussian_prior": metrics_for(prior, truth, axis_uas)}
    histories = {}
    for strategy in STRATEGIES:
        print(f"[weak-rml] {case.key} strategy={strategy}", flush=True)
        image, history = weak_prior_rml(bands, case, strategy, prior, truth, axis_uas)
        images[strategy] = image
        metrics[strategy] = metrics_for(image, truth, axis_uas)
        histories[strategy] = history
    stats.update(
        {
            "method": "EHT-style weak-prior pixel RML using complex visibility or complex closure phasors",
            "image_family": "positive pixel image on a fixed FOV; no ring/crescent parameterization",
            "prior": "broad circular Gaussian MEM/soft-mask prior plus weak TV/TSV/L2 regularization",
            "prior_fwhm_uas": BROAD_PRIOR_FWHM_UAS,
            "n_rml": N_RML,
            "n_iter": N_ITER,
            "step": STEP,
            "observing_days": OBSERVING_DAYS,
            "n_time_windows": N_TIME_WINDOWS,
            "exposure_s": EXPOSURE_S,
            "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": MODE_FALSE_POSITIVE,
            "pair_false_positive": PAIR_FALSE_POSITIVE,
            "snr_boost": SNR_BOOST,
            "common_amp_weight": COMMON_AMP_WEIGHT,
            "mem_weight": MEM_WEIGHT,
            "tv_weight": TV_WEIGHT,
            "tsv_weight": TSV_WEIGHT,
            "l2_prior_weight": L2_PRIOR_WEIGHT,
            "metrics": metrics,
            "histories": histories,
        }
    )
    return {"case": case, "stats": stats, "truth": truth, "axis_uas": axis_uas, "images": images, "metrics": metrics}


def plot_results(results: list[dict], tag: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(results), 4, figsize=(8.6, 2.25 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axes = axes[None, :]
    case_labels = {
        "optimal8_ngc4151_hub_m2_m5": "Optimal 8",
        "hawaii_top4_remote3_ngc4151": "Hawaii+3",
        "hawaii_top4_remote4_ngc4151": "Hawaii+4",
    }
    cols = [("truth", "Input"), ("all", "All vis. + drift"), ("split", "Edge-first closure"), ("direct", "Direct closure")]
    image_axes = []
    for row, result in enumerate(results):
        axis_uas = result["axis_uas"]
        extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
        for col, (key, title) in enumerate(cols):
            ax = axes[row, col]
            image = result["images"][key]
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            if key == "truth":
                ax.set_title(title)
            else:
                metric = result["metrics"][key]
                ax.set_title(f"{title}\nBLR={metric['blr_corr']:.2f}, rad={metric['radial_corr']:.2f}")
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 0:
                ax.set_ylabel(f"{case_labels.get(result['case'].key, result['case'].key)}\n" + r"$\Delta\delta$ ($\mu$as)")
            else:
                ax.set_yticklabels([])
            image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.018,
        pad=0.012,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.6)
    fig.suptitle(
        (
            "EHT-style weak-prior pixel RML: broad Gaussian/TV/TSV only; "
            f"{SOURCE.name}, {OBSERVING_DAYS} d"
        ),
        fontsize=9.4,
        weight="bold",
    )
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_outputs(results: list[dict], tag: str, pdf: Path, png: Path) -> tuple[Path, Path]:
    rows = []
    for result in results:
        for key, metric in result["metrics"].items():
            rows.append({"case": result["case"].key, "image": key, **metric})
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(csv_path),
        "results": [result["stats"] for result in results],
    }
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def main() -> None:
    results = [run_case(case) for case in make_cases()]
    tag = (
        f"eht_style_weak_prior_closure_rml_{SOURCE.key}_{OBSERVING_DAYS}d_"
        f"snr{SNR_BOOST:g}_loss{FIBER_LOSS_DB_PER_KM:g}_fp{MODE_FALSE_POSITIVE:g}_n{N_RML}"
    ).replace(".", "p")
    pdf, png = plot_results(results, tag)
    csv_path, json_path = write_outputs(results, tag, pdf, png)
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(result["case"].key)
        for key in STRATEGIES:
            print(" ", key, result["metrics"][key])


if __name__ == "__main__":
    main()
