from __future__ import annotations

import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
import run_hawaii3_hawaii4_amp_closure_rml as hrun
import test_midbaseline_amp_closure_rml as mid


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

STRATEGY = os.environ.get("RML_VALIDATION_STRATEGY", "direct")
CASE_NAMES = [name.strip() for name in os.environ.get("RML_VALIDATION_CASES", "hawaii4").split(",") if name.strip()]
OPTIMIZER = os.environ.get("RML_VALIDATION_OPTIMIZER", "physical")
PHYSICAL_ITER = int(os.environ.get("RML_PHYSICAL_ITER", str(amp_rml.N_ITER)))
PHYSICAL_STEP = float(os.environ.get("RML_PHYSICAL_STEP", "0.010"))
ADAM_ITER = int(os.environ.get("RML_ADAM_ITER", "400"))
ADAM_LR = float(os.environ.get("RML_ADAM_LR", "0.035"))
# Early stopping should be tied to the statistically self-consistent
# truth-image floor.  A value near 2 was useful for quick exploratory runs, but
# it can falsely declare convergence before the physical likelihood is fitted.
ADAM_TARGET_AMP_CHI2 = float(os.environ.get("RML_ADAM_TARGET_AMP_CHI2", "1.05"))
ADAM_TARGET_PHASE_CHI2 = float(os.environ.get("RML_ADAM_TARGET_PHASE_CHI2", "1.05"))
ADAM_MIN_PHASE_CHI2 = float(os.environ.get("RML_PHASE_CHI2_MIN", "0.90"))
FIT_N_PIX = int(os.environ.get("RML_FIT_N_PIX", str(amp_rml.N_RML)))
DISPLAY_SMOOTH_PIX = float(os.environ.get("RML_DISPLAY_SMOOTH_PIX", "1.0"))


def chi2_target_score_values(values: dict[str, float]) -> float:
    amp_chi2 = max(float(values["amp_chi2"]), 1e-300)
    phase_chi2 = max(float(values["phase_chi2"]), 1e-300)
    phase_target = max(float(os.environ.get("RML_PHASE_CHI2_TARGET", "1.0")), 1e-300)
    return float(abs(np.log(phase_chi2 / phase_target)) + 0.30 * abs(np.log(amp_chi2)))


REGULARIZER_CONFIGS = [
    {
        "label": "current",
        "prior": amp_rml.PRIOR_WEIGHT,
        "tv": amp_rml.TV_WEIGHT,
        "entropy": amp_rml.ENTROPY_WEIGHT,
        "step": amp_rml.STEP,
    },
    {
        "label": "weak_smooth",
        "prior": 0.055,
        "tv": 0.025,
        "entropy": 0.005,
        "step": amp_rml.STEP,
    },
]


def make_case(name: str) -> aug.NetworkCase:
    if name == "hawaii3":
        return hrun.make_hawaii_remote_case(3)
    if name == "hawaii4":
        return hrun.make_hawaii_remote_case(4)
    if name == "mid4":
        return mid.make_mid_case(4, np.array([1.6, 3.0, 5.0, 6.0]), key="hawaii_mid4_r1p6_3_5_6")
    if name == "optimal8":
        return amp_rml.load_optimal8_case()
    raise ValueError(f"Unknown validation case: {name}")


@contextmanager
def temporary_regularizers(config: dict[str, float | str]):
    old = {
        "prior": amp_rml.PRIOR_WEIGHT,
        "tv": amp_rml.TV_WEIGHT,
        "entropy": amp_rml.ENTROPY_WEIGHT,
        "step": amp_rml.STEP,
    }
    amp_rml.PRIOR_WEIGHT = float(config["prior"])
    amp_rml.TV_WEIGHT = float(config["tv"])
    amp_rml.ENTROPY_WEIGHT = float(config["entropy"])
    amp_rml.STEP = float(config["step"])
    try:
        yield
    finally:
        amp_rml.PRIOR_WEIGHT = old["prior"]
        amp_rml.TV_WEIGHT = old["tv"]
        amp_rml.ENTROPY_WEIGHT = old["entropy"]
        amp_rml.STEP = old["step"]


def metric_rows_prefix(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def rebin_image_average(image: np.ndarray, n_out: int) -> np.ndarray:
    n_in = image.shape[0]
    if n_in == n_out:
        return image.copy()
    edges = np.linspace(0, n_in, n_out + 1)
    out = np.zeros((n_out, n_out), dtype=float)
    for iy in range(n_out):
        y0 = int(np.floor(edges[iy]))
        y1 = int(np.ceil(edges[iy + 1]))
        for ix in range(n_out):
            x0 = int(np.floor(edges[ix]))
            x1 = int(np.ceil(edges[ix + 1]))
            block = image[max(y0, 0) : min(y1, n_in), max(x0, 0) : min(x1, n_in)]
            out[iy, ix] = float(np.mean(block)) if block.size else 0.0
    return amp_rml.project_flux_positive(out)


def upsample_image_nearest(image: np.ndarray, n_out: int, *, smooth_pix: float = DISPLAY_SMOOTH_PIX) -> np.ndarray:
    n_in = image.shape[0]
    if n_in == n_out:
        out = image.copy()
    else:
        y_idx = np.minimum((np.arange(n_out) * n_in / n_out).astype(int), n_in - 1)
        x_idx = np.minimum((np.arange(n_out) * n_in / n_out).astype(int), n_in - 1)
        out = image[np.ix_(y_idx, x_idx)]
    if smooth_pix > 0.15:
        # Convert smoothing from fit-pixel units to display-pixel units.
        out = base.gaussian_filter(out, smooth_pix * n_out / max(n_in, 1))
    return amp_rml.project_flux_positive(out)


def build_starts(bands: list[dict[str, np.ndarray]], truth: np.ndarray, prior: np.ndarray) -> dict[str, np.ndarray]:
    fit_truth = rebin_image_average(truth, FIT_N_PIX)
    fit_prior = rebin_image_average(prior, FIT_N_PIX)
    starts = {
        "direct_dirty": rebin_image_average(amp_rml.quick_dirty_start(bands, "direct", truth), FIT_N_PIX),
        "split_dirty": rebin_image_average(amp_rml.quick_dirty_start(bands, "split", truth), FIT_N_PIX),
        "all_dirty": rebin_image_average(amp_rml.quick_dirty_start(bands, "all", truth), FIT_N_PIX),
        "truth_coarse": fit_truth,
        "prior": amp_rml.project_flux_positive(fit_prior, smooth_pix=0.0),
    }
    if STRATEGY != "direct":
        starts[f"{STRATEGY}_dirty"] = rebin_image_average(amp_rml.quick_dirty_start(bands, STRATEGY, truth), FIT_N_PIX)
    return starts


def closure_basis(case: aug.NetworkCase) -> tuple[list[tuple[int, int]], np.ndarray]:
    stations, _, _, _ = aug.station_table_from_case(case)
    edges = base.edge_list(len(stations))
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, len(stations)))
    return edges, q_basis


def full_covariance_phase_terms(
    residual_q: np.ndarray,
    covariance_q: np.ndarray,
    phase_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten closure-coordinate residuals and return precision-weighted residuals."""
    residual = np.asarray(residual_q, dtype=float)
    n_sample, n_coord = residual.shape
    cov = np.asarray(covariance_q, dtype=float).reshape(n_sample, n_coord, n_coord)
    cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
    cov = cov + (float(phase_floor) ** 2) * np.eye(n_coord)[None, :, :]
    evals, evecs = np.linalg.eigh(cov)
    safe = np.maximum(evals, 1.0e-24)
    rotated = np.einsum("nij,ni->nj", evecs, residual)
    whitened = rotated / np.sqrt(safe)
    precision_residual = np.einsum("nij,nj->ni", evecs, rotated / safe)
    return whitened, precision_residual


def residual_diagnostics(
    image: np.ndarray,
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    strategy: str,
    axis_uas: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    edges, q_basis = closure_basis(case)
    n_edges = len(edges)
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    n_fit = image.shape[0]
    uv_axis = np.fft.fftshift(np.fft.fftfreq(n_fit, d=fov_rad / n_fit))
    vis_grid = amp_rml.fft_vis(image)
    phase_floor = amp_rml.PHASE_FLOOR_RAD

    amp_z_values: list[np.ndarray] = []
    phase_z_values: list[np.ndarray] = []
    for band in bands:
        model = base.interp_vis(vis_grid, uv_axis, band["u"], band["v"])
        amp_model = np.abs(model)
        amp_data = np.asarray(band["amp"], dtype=float)
        amp_sigma = amp_rml.amplitude_sigma(amp_data, band.get("amp_sigma"))
        amp_z_values.append((amp_model - amp_data) / np.maximum(amp_sigma, 1e-12))

        data = band[f"vis_{strategy}"]
        sigma = band[f"sigma_{strategy}"].reshape(-1, n_edges)
        model_phase = np.angle(model).reshape(-1, n_edges)
        data_phase = np.angle(data).reshape(-1, n_edges)
        if strategy == "all":
            residual = np.angle(np.exp(1j * (model_phase - data_phase)))
            phase_sigma = np.sqrt(sigma**2 + phase_floor**2)
        else:
            model_q = model_phase @ q_basis
            data_q = data_phase @ q_basis
            residual = np.angle(np.exp(1j * (model_q - data_q)))
            sigma_q_cov = band.get(f"sigmaqcov_{strategy}")
            sigma_q = band.get(f"sigmaq_{strategy}")
            if sigma_q_cov is not None:
                phase_z, _precision_residual = full_covariance_phase_terms(
                    residual,
                    sigma_q_cov,
                    phase_floor,
                )
                phase_z_values.append(phase_z.reshape(-1))
                continue
            elif sigma_q is not None:
                phase_sigma = np.sqrt(np.asarray(sigma_q, dtype=float).reshape(-1, q_basis.shape[1]) ** 2 + phase_floor**2)
            else:
                phase_sigma = np.sqrt((sigma**2 + phase_floor**2) @ (q_basis**2))
        phase_z_values.append((residual / np.maximum(phase_sigma, 1e-12)).reshape(-1))

    amp_z = np.concatenate(amp_z_values)
    phase_z = np.concatenate(phase_z_values)
    finite_amp = amp_z[np.isfinite(amp_z)]
    finite_phase = phase_z[np.isfinite(phase_z)]
    diagnostics = {
        "amp_reduced_chi2": float(np.mean(finite_amp**2)),
        "amp_z_rms": float(np.sqrt(np.mean(finite_amp**2))),
        "amp_z_p50_abs": float(np.percentile(np.abs(finite_amp), 50.0)),
        "amp_z_p90_abs": float(np.percentile(np.abs(finite_amp), 90.0)),
        "phase_reduced_chi2": float(np.mean(finite_phase**2)),
        "phase_z_rms": float(np.sqrt(np.mean(finite_phase**2))),
        "phase_z_p50_abs": float(np.percentile(np.abs(finite_phase), 50.0)),
        "phase_z_p90_abs": float(np.percentile(np.abs(finite_phase), 90.0)),
        "phase_floor_rad": float(phase_floor),
        "n_amp_samples": int(finite_amp.size),
        "n_phase_samples": int(finite_phase.size),
    }
    return diagnostics, {"amp_z": finite_amp, "phase_z": finite_phase}


def physical_objective_and_grads(
    image: np.ndarray,
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    strategy: str,
    axis_uas: np.ndarray,
    prior: np.ndarray,
    config: dict[str, float | str],
) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
    edges, q_basis = closure_basis(case)
    n_edges = len(edges)
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    n_fit = image.shape[0]
    uv_axis = np.fft.fftshift(np.fft.fftfreq(n_fit, d=fov_rad / n_fit))
    vis_grid = amp_rml.fft_vis(image)
    phase_floor = amp_rml.PHASE_FLOOR_RAD

    amp_grad_grid = np.zeros_like(vis_grid)
    phase_grad_grid = np.zeros_like(vis_grid)
    amp_chi2 = 0.0
    phase_chi2 = 0.0
    n_band = 0
    eps = 1e-9
    for band in bands:
        model = base.interp_vis(vis_grid, uv_axis, band["u"], band["v"])
        amp_model = np.abs(model)
        amp_data = np.asarray(band["amp"], dtype=float)
        amp_sigma = amp_rml.amplitude_sigma(amp_data, band.get("amp_sigma"))
        amp_z = (amp_model - amp_data) / np.maximum(amp_sigma, 1e-12)
        amp_chi2 += float(np.mean(amp_z**2))
        # Scatter the physical chi-square gradient.  The overall sign is tested
        # by the line search, so the important part is using the true sigma.
        amp_coeff = (amp_z / np.maximum(amp_sigma, 1e-12)) * model / np.maximum(amp_model, eps)
        amp_coeff /= max(model.size, 1)
        amp_rml.bilinear_adjoint_samples(amp_grad_grid, uv_axis, band["u"], band["v"], amp_coeff)

        data = band[f"vis_{strategy}"]
        sigma = band[f"sigma_{strategy}"].reshape(-1, n_edges)
        model_phase = np.angle(model).reshape(-1, n_edges)
        data_phase = np.angle(data).reshape(-1, n_edges)
        if strategy == "all":
            residual = np.angle(np.exp(1j * (model_phase - data_phase)))
            phase_sigma = np.sqrt(sigma**2 + phase_floor**2)
            phase_z = residual / np.maximum(phase_sigma, 1e-12)
            phase_coeff = (phase_z / np.maximum(phase_sigma, 1e-12)).reshape(-1)
        else:
            model_q = model_phase @ q_basis
            data_q = data_phase @ q_basis
            residual_q = np.angle(np.exp(1j * (model_q - data_q)))
            sigma_q_cov = band.get(f"sigmaqcov_{strategy}")
            sigma_q = band.get(f"sigmaq_{strategy}")
            if sigma_q_cov is not None:
                phase_z, phase_coeff_q = full_covariance_phase_terms(
                    residual_q,
                    sigma_q_cov,
                    phase_floor,
                )
            elif sigma_q is not None:
                phase_sigma_q = np.sqrt(np.asarray(sigma_q, dtype=float).reshape(-1, q_basis.shape[1]) ** 2 + phase_floor**2)
                phase_z = residual_q / np.maximum(phase_sigma_q, 1e-12)
                phase_coeff_q = phase_z / np.maximum(phase_sigma_q, 1e-12)
            else:
                phase_sigma_q = np.sqrt((sigma**2 + phase_floor**2) @ (q_basis**2))
                phase_z = residual_q / np.maximum(phase_sigma_q, 1e-12)
                phase_coeff_q = phase_z / np.maximum(phase_sigma_q, 1e-12)
            phase_coeff = (phase_coeff_q @ q_basis.T).reshape(-1)
        phase_chi2 += float(np.mean(phase_z**2))
        safe_model = np.where(np.abs(model) > 1e-4, model, 1e-4 + 0j)
        phase_sample_grad = 1j * phase_coeff / np.conj(safe_model)
        phase_sample_grad /= max(model.size, 1)
        amp_rml.bilinear_adjoint_samples(phase_grad_grid, uv_axis, band["u"], band["v"], phase_sample_grad)
        n_band += 1

    amp_chi2 /= max(n_band, 1)
    phase_chi2 /= max(n_band, 1)
    amp_grad = amp_rml.adjoint_image(amp_grad_grid)
    phase_grad = amp_rml.adjoint_image(phase_grad_grid)
    prior_grad = 2.0 * (image - prior)
    entropy_grad = np.log((image + 1e-12) / (prior + 1e-12))
    tv_grad = amp_rml.latest.tv_gradient(image)
    reg_grad = (
        float(config["prior"]) * amp_rml.normalize_grad(prior_grad)
        + float(config["entropy"]) * amp_rml.normalize_grad(entropy_grad)
        + float(config["tv"]) * amp_rml.normalize_grad(tv_grad)
    )
    reg_value = (
        float(config["prior"]) * float(np.mean((image - prior) ** 2))
        + float(config["entropy"]) * float(np.mean(image * np.log((image + 1e-12) / (prior + 1e-12))))
        + float(config["tv"]) * float(np.mean(np.abs(np.gradient(image, axis=0))) + np.mean(np.abs(np.gradient(image, axis=1))))
    )
    total = amp_rml.AMP_GRAD_WEIGHT * amp_chi2 + amp_rml.PHASE_GRAD_WEIGHT * phase_chi2 + reg_value
    values = {
        "objective": float(total),
        "amp_chi2": float(amp_chi2),
        "phase_chi2": float(phase_chi2),
        "reg": float(reg_value),
    }
    grads = {
        "amp": amp_grad,
        "phase": phase_grad,
        "reg": reg_grad,
    }
    return float(total), values, grads


def physical_amp_closure_rml(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    strategy: str,
    prior: np.ndarray,
    start: np.ndarray,
    axis_uas: np.ndarray,
    config: dict[str, float | str],
) -> tuple[np.ndarray, dict[str, float]]:
    x = amp_rml.project_flux_positive(0.65 * start + 0.35 * prior, smooth_pix=0.20)
    history: dict[str, float] = {}
    step0 = PHYSICAL_STEP
    for iteration in range(PHYSICAL_ITER):
        current_obj, values, grads = physical_objective_and_grads(x, bands, case, strategy, axis_uas, prior, config)
        amp_dir = amp_rml.normalize_grad(grads["amp"])
        phase_dir = amp_rml.normalize_grad(grads["phase"])
        reg_dir = amp_rml.normalize_grad(grads["reg"])
        # Joint, two-observable optimization.  We never optimize the amplitude
        # or closure phase in isolation here: each trial direction contains both
        # data gradients and is accepted only if the single physical objective
        # chi2_amp + chi2_CP + regularizers decreases.  The sign combinations
        # are a numerical safeguard against FFT/Wirtinger convention mistakes in
        # the hand-written adjoint.
        amp_scale = amp_rml.AMP_GRAD_WEIGHT * max(1.0, np.sqrt(values["amp_chi2"]))
        phase_scale = amp_rml.PHASE_GRAD_WEIGHT * max(1.0, np.sqrt(values["phase_chi2"]))
        directions = []
        for amp_sign in (-1.0, 1.0):
            for phase_sign in (-1.0, 1.0):
                label = f"{amp_sign:+.0f}amp{phase_sign:+.0f}cp"
                direction = amp_rml.normalize_grad(
                    amp_sign * amp_scale * amp_dir
                    + phase_sign * phase_scale * phase_dir
                    + reg_dir
                )
                directions.append((label, direction))
        best_x = x
        best_obj = current_obj
        best_values = values
        best_label = "none"
        step = step0
        for _ in range(9):
            improved = False
            for label, direction in directions:
                if not np.any(np.isfinite(direction)):
                    continue
                trial = amp_rml.project_flux_positive(x - step * direction, smooth_pix=0.0)
                trial_obj, trial_values, _ = physical_objective_and_grads(
                    trial, bands, case, strategy, axis_uas, prior, config
                )
                if np.isfinite(trial_obj) and trial_obj < best_obj:
                    best_x = trial
                    best_obj = trial_obj
                    best_values = trial_values
                    best_label = f"{label}@{step:.2g}"
                    improved = True
            if improved:
                break
            step *= 0.5
        x = best_x
        if iteration in {0, PHYSICAL_ITER // 4, PHYSICAL_ITER // 2, 3 * PHYSICAL_ITER // 4, PHYSICAL_ITER - 1}:
            history[f"objective_{iteration}"] = best_obj
            history[f"amp_chi2_{iteration}"] = best_values["amp_chi2"]
            history[f"phase_chi2_{iteration}"] = best_values["phase_chi2"]
            history[f"move_{iteration}"] = best_label
    return amp_rml.project_flux_positive(x, smooth_pix=0.0), history


def adam_amp_closure_rml(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    strategy: str,
    prior: np.ndarray,
    start: np.ndarray,
    axis_uas: np.ndarray,
    config: dict[str, float | str],
) -> tuple[np.ndarray, dict[str, float]]:
    """Softmax-parameterized RML solver for strict likelihood fits.

    The projected line-search solver is robust for visual reconstructions but
    can stall when the amplitude likelihood is assigned a small absolute sigma.
    Here positivity and total flux are enforced by ``image = softmax(y)``, so
    Adam can descend the same Fisher-weighted objective without projection
    artifacts.
    """
    x0 = amp_rml.project_flux_positive(0.65 * start + 0.35 * prior, smooth_pix=0.0)
    y = np.log(np.maximum(x0, 1e-12))
    y -= float(np.mean(y))
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    history: dict[str, float] = {}
    best_score = np.inf
    best_x = x0
    best_values: dict[str, float] | None = None
    checkpoint_images: list[dict[str, object]] = []
    checkpoints = {1, 2, 5, 10, 20, 50, 100, 200, ADAM_ITER}
    checkpoints.update({max(1, int(round(frac * ADAM_ITER))) for frac in (0.35, 0.50, 0.65, 0.80)})
    for iteration in range(1, ADAM_ITER + 1):
        yy = y - float(np.max(y))
        x = np.exp(yy)
        x /= float(np.sum(x))
        _, values, grads = physical_objective_and_grads(x, bands, case, strategy, axis_uas, prior, config)
        grad_x = amp_rml.AMP_GRAD_WEIGHT * grads["amp"] + amp_rml.PHASE_GRAD_WEIGHT * grads["phase"] + grads["reg"]
        grad_y = x * (grad_x - float(np.sum(grad_x * x)))
        clip = float(np.percentile(np.abs(grad_y), 99.5))
        if np.isfinite(clip) and clip > 0.0:
            grad_y = np.clip(grad_y, -20.0 * clip, 20.0 * clip)
        m = beta1 * m + (1.0 - beta1) * grad_y
        v = beta2 * v + (1.0 - beta2) * (grad_y * grad_y)
        mhat = m / (1.0 - beta1**iteration)
        vhat = v / (1.0 - beta2**iteration)
        y -= ADAM_LR * mhat / (np.sqrt(vhat) + eps)
        y -= float(np.mean(y))

        score = chi2_target_score_values(values)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_x = x.copy()
            best_values = dict(values)
        if iteration in checkpoints:
            history[f"objective_{iteration}"] = float(values["objective"])
            history[f"amp_chi2_{iteration}"] = float(values["amp_chi2"])
            history[f"phase_chi2_{iteration}"] = float(values["phase_chi2"])
            checkpoint_images.append(
                {
                    "iteration": int(iteration),
                    "image": x.copy(),
                    "values": dict(values),
                    "score": float(score),
                }
            )
        if (
            values["amp_chi2"] < ADAM_TARGET_AMP_CHI2
            and values["phase_chi2"] < ADAM_TARGET_PHASE_CHI2
            and values["phase_chi2"] >= ADAM_MIN_PHASE_CHI2
            and iteration >= 50
        ):
            history[f"early_stop_{iteration}"] = 1.0
            break
    if best_values is not None:
        history["best_amp_chi2"] = float(best_values["amp_chi2"])
        history["best_phase_chi2"] = float(best_values["phase_chi2"])
    history["_checkpoint_images"] = checkpoint_images
    return amp_rml.project_flux_positive(best_x, smooth_pix=0.0), history


def run_single_reconstruction(
    *,
    case: aug.NetworkCase,
    bands: list[dict[str, np.ndarray]],
    truth: np.ndarray,
    axis_uas: np.ndarray,
    prior: np.ndarray,
    start_name: str,
    start: np.ndarray,
    config: dict[str, float | str],
    split_label: str,
) -> dict:
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    if OPTIMIZER == "physical":
        image, history = physical_amp_closure_rml(
            bands,
            case,
            STRATEGY,
            prior,
            start,
            axis_uas,
            config,
        )
    elif OPTIMIZER == "adam":
        image, history = adam_amp_closure_rml(
            bands,
            case,
            STRATEGY,
            prior,
            start,
            axis_uas,
            config,
        )
    else:
        with temporary_regularizers(config):
            image, history = amp_rml.amplitude_closure_rml(
                bands,
                case,
                STRATEGY,
                prior,
                start,
                fov_rad=fov_rad,
            )
    display_image = upsample_image_nearest(image, len(axis_uas))
    prior_for_objective = upsample_image_nearest(prior, len(axis_uas), smooth_pix=0.0)
    objective, amp_obj, phase_obj = hrun.data_objective(display_image, bands, case, STRATEGY, axis_uas, prior_for_objective)
    metrics = amp_rml.metrics_for(display_image, truth, axis_uas)
    residuals, residual_arrays = residual_diagnostics(image, bands, case, STRATEGY, axis_uas)
    validation_score = residual_selection_score(residuals, objective)
    return {
        "case": case.key,
        "split": split_label,
        "strategy": STRATEGY,
        "optimizer": OPTIMIZER,
        "config": str(config["label"]),
        "start": start_name,
        "objective": float(objective),
        "amp_objective": float(amp_obj),
        "phase_objective": float(phase_obj),
        "image": display_image,
        "fit_image": image,
        "history": history,
        "metrics": metrics,
        "residuals": residuals,
        "residual_arrays": residual_arrays,
        "validation_score": validation_score,
    }


def residual_selection_score(residuals: dict[str, float], internal_objective: float) -> float:
    """Choose reconstructions by data consistency, not by image cosmetics.

    The gradient-descent objective normalizes some weights internally and can
    occasionally prefer a solution with visually plausible closure phases but a
    terrible calibrated-amplitude fit.  For validation we therefore rank images
    by physical reduced-chi2 diagnostics.  Closure-phase overfitting is penalized
    symmetrically: phase reduced chi2 well below unity is no longer treated as
    automatically better.
    """
    amp_chi2 = max(float(residuals["amp_reduced_chi2"]), 1e-300)
    phase_chi2 = max(float(residuals["phase_reduced_chi2"]), 1e-300)
    phase_target = max(float(os.environ.get("RML_PHASE_CHI2_TARGET", "1.0")), 1e-300)
    amp_score = abs(np.log(amp_chi2))
    phase_score = abs(np.log(phase_chi2 / phase_target))
    return float(phase_score + 0.30 * amp_score + 0.001 * internal_objective)


def run_ensemble(
    case: aug.NetworkCase,
    bands: list[dict[str, np.ndarray]],
    truth: np.ndarray,
    axis_uas: np.ndarray,
    *,
    split_label: str,
    configs: list[dict[str, float | str]],
    start_limit: int | None = None,
) -> list[dict]:
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = rebin_image_average(prior_full, FIT_N_PIX)
    starts = build_starts(bands, truth, prior)
    if start_limit is not None:
        ordered = list(starts.items())
        preferred = [item for item in ordered if item[0] in {"direct_dirty", "prior"}]
        starts = dict((preferred + ordered)[:start_limit])
    results = []
    for config in configs:
        for start_name, start in starts.items():
            print(
                f"[rml] case={case.key} split={split_label} config={config['label']} start={start_name}",
                flush=True,
            )
            results.append(
                run_single_reconstruction(
                    case=case,
                    bands=bands,
                    truth=truth,
                    axis_uas=axis_uas,
                    prior=prior,
                    start_name=start_name,
                    start=start,
                    config=config,
                    split_label=split_label,
                )
            )
    return results


def copy_band_with_rows(band: dict[str, np.ndarray], row_mask: np.ndarray, n_edges: int) -> dict[str, np.ndarray]:
    rows = len(band["u"]) // n_edges
    out: dict[str, np.ndarray] = {}
    for key, value in band.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size == rows * n_edges:
            out[key] = arr.reshape(rows, n_edges)[row_mask].reshape(-1).copy()
        else:
            out[key] = arr.copy()
    return out


def time_jackknife_bands(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    parity: int,
) -> list[dict[str, np.ndarray]]:
    edges, _ = closure_basis(case)
    n_edges = len(edges)
    subset = []
    for band in bands:
        rows = len(band["u"]) // n_edges
        mask = (np.arange(rows) % 2) == parity
        subset.append(copy_band_with_rows(band, mask, n_edges))
    return subset


def wavelength_jackknife_bands(bands: list[dict[str, np.ndarray]], parity: int) -> list[dict[str, np.ndarray]]:
    return [band for index, band in enumerate(bands) if (index % 2) == parity]


def ensemble_summary(results: list[dict]) -> dict[str, float]:
    images = np.stack([result["image"] for result in results], axis=0)
    mean_image = amp_rml.project_flux_positive(np.mean(images, axis=0))
    std_image = np.std(images, axis=0)
    return {
        "pixel_ensemble_std_mean": float(np.mean(std_image)),
        "pixel_ensemble_std_p90": float(np.percentile(std_image, 90.0)),
        "n_ensemble": int(len(results)),
        "mean_image": mean_image,
        "std_image": std_image,
    }


def image_difference_metric(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = opt.normalize_blr_display(image_a).reshape(-1)
    b = opt.normalize_blr_display(image_b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plot_case_summary(
    case: aug.NetworkCase,
    truth: np.ndarray,
    axis_uas: np.ndarray,
    full_results: list[dict],
    jackknife_results: dict[str, dict],
    tag: str,
) -> tuple[Path, Path]:
    best = min(full_results, key=lambda result: result["validation_score"])
    ensemble = ensemble_summary(full_results)
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]

    fig, axes = plt.subplots(2, 4, figsize=(10.2, 5.2), constrained_layout=True)
    image_axes = []
    panels = [
        ("Truth", truth),
        (
            f"Best full RML\n{best['config']}/{best['start']}\n"
            f"BLR={best['metrics']['blr_corr']:.2f}, global={best['metrics']['global_corr']:.2f}",
            best["image"],
        ),
        ("Ensemble mean", ensemble["mean_image"]),
        ("Ensemble std", ensemble["std_image"] / max(float(np.max(ensemble["std_image"])), 1e-30)),
        (
            f"lambda even\nBLR={jackknife_results['lambda_even']['metrics']['blr_corr']:.2f}",
            jackknife_results["lambda_even"]["image"],
        ),
        (
            f"lambda odd\nBLR={jackknife_results['lambda_odd']['metrics']['blr_corr']:.2f}",
            jackknife_results["lambda_odd"]["image"],
        ),
        (
            f"time even\nBLR={jackknife_results['time_even']['metrics']['blr_corr']:.2f}",
            jackknife_results["time_even"]["image"],
        ),
        (
            f"time odd\nBLR={jackknife_results['time_odd']['metrics']['blr_corr']:.2f}",
            jackknife_results["time_odd"]["image"],
        ),
    ]
    for ax, (title, image) in zip(axes.reshape(-1), panels):
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(title, fontsize=8.0)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.020,
        pad=0.012,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=7)
    fig.suptitle(
        f"RML validation pipeline: {case.key}, strategy={STRATEGY}, {amp_rml.SOURCE.name}",
        fontsize=10.5,
        weight="bold",
    )
    png = OUTFIG / f"{tag}_images.png"
    pdf = OUTFIG / f"{tag}_images.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_residuals(best: dict, tag: str) -> tuple[Path, Path]:
    amp_z = np.clip(best["residual_arrays"]["amp_z"], -8.0, 8.0)
    phase_z = np.clip(best["residual_arrays"]["phase_z"], -8.0, 8.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True)
    axes[0].hist(amp_z, bins=80, density=True, color="#1f77b4", alpha=0.80)
    axes[0].set_title(f"Amplitude residuals\nchi2={best['residuals']['amp_reduced_chi2']:.2g}")
    axes[0].set_xlabel(r"$(|V|_{\rm model}-|V|_{\rm data})/\sigma_{|V|}$")
    axes[0].set_ylabel("density")
    axes[1].hist(phase_z, bins=80, density=True, color="#d62728", alpha=0.80)
    axes[1].set_title(f"Closure-phase residuals\nchi2={best['residuals']['phase_reduced_chi2']:.2g}")
    axes[1].set_xlabel(r"$\Delta\psi/\sigma_{\psi}$")
    axes[1].set_ylabel("density")
    fig.suptitle(f"Residual diagnostics: {best['case']}, {best['config']}/{best['start']}", fontsize=10)
    png = OUTFIG / f"{tag}_residuals.png"
    pdf = OUTFIG / f"{tag}_residuals.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def serializable_result(result: dict) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "case": result["case"],
        "split": result["split"],
        "strategy": result["strategy"],
        "config": result["config"],
        "optimizer": result["optimizer"],
        "start": result["start"],
        "objective": result["objective"],
        "amp_objective": result["amp_objective"],
        "phase_objective": result["phase_objective"],
        "validation_score": result["validation_score"],
    }
    row.update(metric_rows_prefix("metric", result["metrics"]))
    row.update(metric_rows_prefix("resid", result["residuals"]))
    return row


def write_case_outputs(
    case: aug.NetworkCase,
    stats: dict,
    truth: np.ndarray,
    axis_uas: np.ndarray,
    full_results: list[dict],
    jackknife_results: dict[str, dict],
    image_paths: tuple[Path, Path],
    residual_paths: tuple[Path, Path],
    tag: str,
) -> tuple[Path, Path]:
    rows = [serializable_result(result) for result in full_results]
    rows.extend(serializable_result(result) for result in jackknife_results.values())
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = min(full_results, key=lambda result: result["validation_score"])
    jackknife_distances = {
        split: image_difference_metric(best["image"], result["image"])
        for split, result in jackknife_results.items()
    }
    payload = {
        "case": case.key,
        "strategy": STRATEGY,
        "optimizer": OPTIMIZER,
        "source": amp_rml.SOURCE.name,
        "simulation_stats": stats,
        "environment": {
            "N_RML": amp_rml.N_RML,
            "N_ITER": amp_rml.N_ITER,
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "exposure_s": amp_rml.EXPOSURE_S,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
            "pair_false_positive": amp_rml.PAIR_FALSE_POSITIVE,
            "snr_boost": amp_rml.SNR_BOOST,
            "amp_sigma_mode": amp_rml.AMP_SIGMA_MODE,
            "amp_sigma_abs": amp_rml.AMP_SIGMA_ABS,
            "amp_rel_sigma": amp_rml.AMP_REL_SIGMA,
            "amp_abs_floor": amp_rml.AMP_ABS_FLOOR,
            "phase_floor_rad": amp_rml.PHASE_FLOOR_RAD,
            "physical_iter": PHYSICAL_ITER,
            "physical_step": PHYSICAL_STEP,
            "adam_iter": ADAM_ITER,
            "adam_lr": ADAM_LR,
            "adam_target_amp_chi2": ADAM_TARGET_AMP_CHI2,
            "adam_target_phase_chi2": ADAM_TARGET_PHASE_CHI2,
            "fit_n_pix": FIT_N_PIX,
            "display_smooth_pix_fit_units": DISPLAY_SMOOTH_PIX,
        },
        "regularizer_configs": REGULARIZER_CONFIGS,
        "best_full_result": serializable_result(best),
        "jackknife_image_rmse_vs_best": jackknife_distances,
        "figures": {
            "images_pdf": str(image_paths[0]),
            "images_png": str(image_paths[1]),
            "residuals_pdf": str(residual_paths[0]),
            "residuals_png": str(residual_paths[1]),
            "metrics_csv": str(csv_path),
        },
    }
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    npz_path = OUTFIG / f"{tag}_best_images.npz"
    np.savez_compressed(
        npz_path,
        truth=np.asarray(truth, dtype=float),
        best_display_image=np.asarray(best["image"], dtype=float),
        best_fit_image=np.asarray(best["fit_image"], dtype=float),
        axis_uas=np.asarray(axis_uas, dtype=float),
        case_key=str(case.key),
        best_start=str(best["start"]),
        best_config=str(best["config"]),
        strategy=str(best["strategy"]),
        optimizer=str(best["optimizer"]),
        fit_n_pix=int(FIT_N_PIX),
        display_smooth_pix=float(DISPLAY_SMOOTH_PIX),
        adam_iter=int(ADAM_ITER),
        adam_lr=float(ADAM_LR),
        adam_target_amp_chi2=float(ADAM_TARGET_AMP_CHI2),
        adam_target_phase_chi2=float(ADAM_TARGET_PHASE_CHI2),
        amp_sigma_mode=str(amp_rml.AMP_SIGMA_MODE),
        amp_sigma_abs=float(amp_rml.AMP_SIGMA_ABS),
        amp_rel_sigma=float(amp_rml.AMP_REL_SIGMA),
        amp_abs_floor=float(amp_rml.AMP_ABS_FLOOR),
        phase_floor_rad=float(amp_rml.PHASE_FLOOR_RAD),
    )
    payload["figures"]["best_images_npz"] = str(npz_path)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return csv_path, json_path


def validate_case(case: aug.NetworkCase) -> tuple[Path, Path]:
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    full_results = run_ensemble(
        case,
        bands,
        truth,
        axis_uas,
        split_label="full",
        configs=REGULARIZER_CONFIGS,
    )
    best = min(full_results, key=lambda result: result["validation_score"])
    best_config = next(config for config in REGULARIZER_CONFIGS if config["label"] == best["config"])

    jackknife_bands = {
        "lambda_even": wavelength_jackknife_bands(bands, 0),
        "lambda_odd": wavelength_jackknife_bands(bands, 1),
        "time_even": time_jackknife_bands(bands, case, 0),
        "time_odd": time_jackknife_bands(bands, case, 1),
    }
    jackknife_results = {}
    for split_label, split_bands in jackknife_bands.items():
        split_results = run_ensemble(
            case,
            split_bands,
            truth,
            axis_uas,
            split_label=split_label,
            configs=[best_config],
            start_limit=2,
        )
        jackknife_results[split_label] = min(split_results, key=lambda result: result["validation_score"])

    tag = (
        f"rml_validation_{case.key}_{amp_rml.SOURCE.key}_{STRATEGY}_"
        f"{OPTIMIZER}_{amp_rml.sigma_tag()}_"
        f"{amp_rml.OBSERVING_DAYS}d_fitn{FIT_N_PIX}_shown{amp_rml.N_RML}_"
        f"iter{PHYSICAL_ITER if OPTIMIZER == 'physical' else amp_rml.N_ITER}"
    ).replace(".", "p")
    image_paths = plot_case_summary(case, truth, axis_uas, full_results, jackknife_results, tag)
    residual_paths = plot_residuals(best, tag)
    csv_path, json_path = write_case_outputs(
        case,
        stats,
        truth,
        axis_uas,
        full_results,
        jackknife_results,
        image_paths,
        residual_paths,
        tag,
    )
    print(image_paths[1])
    print(residual_paths[1])
    print(csv_path)
    print(json_path)
    return csv_path, json_path


def main() -> None:
    for case_name in CASE_NAMES:
        validate_case(make_case(case_name))


if __name__ == "__main__":
    main()
