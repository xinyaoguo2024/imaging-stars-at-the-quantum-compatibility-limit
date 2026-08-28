from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as case_lib
import latest_maunakea_closure_snr_clean_rml as latest
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

SOURCE = ngc.NGC4151

N_MODEL = int(os.environ.get("PARAM_RML_N_PIX", "80"))
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "30"))
N_TIME_WINDOWS = int(os.environ.get("N_TIME_WINDOWS", "36"))
EXPOSURE_S = float(os.environ.get("EXPOSURE_S", "600.0"))
EXPOSURE_GAP_S = float(os.environ.get("EXPOSURE_GAP_S", "150.0"))
FIBER_LOSS_DB_PER_KM = float(os.environ.get("FIBER_LOSS_DB_PER_KM", "0.20"))
FIBER_LENGTH_SCALE = float(os.environ.get("FIBER_LENGTH_SCALE", "0.75"))
MODE_FALSE_POSITIVE = float(os.environ.get("MODE_FALSE_POSITIVE", "0.05"))
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", "0.0"))
SNR_BOOST = float(os.environ.get("AUGMENTED_SNR_BOOST", "1.0"))

AMP_REL_SIGMA = float(os.environ.get("PARAM_RML_AMP_REL_SIGMA", "0.04"))
AMP_ABS_FLOOR = float(os.environ.get("PARAM_RML_AMP_ABS_FLOOR", "0.012"))
PHASE_FLOOR_RAD = float(os.environ.get("PARAM_RML_PHASE_FLOOR_RAD", "0.025"))
AMP_WEIGHT = float(os.environ.get("PARAM_RML_AMP_WEIGHT", "1.0"))
PHASE_WEIGHT = float(os.environ.get("PARAM_RML_PHASE_WEIGHT", "1.0"))
COMMON_AMP_WEIGHT = float(os.environ.get("PARAM_RML_COMMON_AMP_WEIGHT", "0.0"))

N_RANDOM = int(os.environ.get("PARAM_RML_N_RANDOM", "180"))
N_LOCAL = int(os.environ.get("PARAM_RML_N_LOCAL", "360"))
N_RESTARTS = int(os.environ.get("PARAM_RML_N_RESTARTS", "7"))
RNG_SEED = int(os.environ.get("PARAM_RML_RNG_SEED", "20260519"))

STRATEGIES = ("all", "split", "direct")


PARAM_NAMES = (
    "f_crescent",
    "radius_uas",
    "width_uas",
    "asymmetry",
    "asym_pa_rad",
    "core_x_uas",
    "core_y_uas",
    "core_sigma_major_uas",
    "core_axis_ratio",
    "core_pa_rad",
)

BOUNDS = np.array(
    [
        (0.22, 0.78),
        (32.0, 76.0),
        (4.0, 20.0),
        (0.00, 0.80),
        (-math.pi, math.pi),
        (-24.0, 24.0),
        (-24.0, 24.0),
        (2.6, 18.0),
        (0.28, 1.00),
        (-math.pi, math.pi),
    ],
    dtype=float,
)


def configure_simulation() -> None:
    """Use the same physical noise convention as the current PRL simulations."""
    aug.OBSERVING_DAYS = OBSERVING_DAYS
    aug.N_TIME_WINDOWS = N_TIME_WINDOWS
    aug.EXPOSURE_S = EXPOSURE_S
    aug.EXPOSURE_GAP_S = EXPOSURE_GAP_S
    aug.FIBER_LOSS_DB_PER_KM = FIBER_LOSS_DB_PER_KM
    aug.FIBER_LENGTH_SCALE = FIBER_LENGTH_SCALE
    aug.MODE_FALSE_POSITIVE = MODE_FALSE_POSITIVE
    aug.PAIR_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    aug.BASELINE_FALSE_POSITIVE = PAIR_FALSE_POSITIVE
    wt.N_PIX = N_MODEL
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
        raise ValueError("remote_count must be 3 or 4 for this benchmark.")
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


def unit_to_params(unit: np.ndarray) -> np.ndarray:
    unit = np.asarray(unit, dtype=float).copy()
    unit = np.mod(unit, 1.0)
    return BOUNDS[:, 0] + unit * (BOUNDS[:, 1] - BOUNDS[:, 0])


def params_to_unit(params: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(params) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0]), 0.0, 1.0)


def canonical_unit_guesses() -> list[np.ndarray]:
    guesses = [
        np.array([0.58, 60.0, 10.0, 0.28, math.radians(35.0), 7.0, -3.0, 6.5, 0.62, math.radians(-25.0)]),
        np.array([0.48, 58.0, 12.0, 0.20, math.radians(20.0), 0.0, 0.0, 8.0, 0.65, math.radians(-20.0)]),
        np.array([0.68, 62.0, 8.0, 0.38, math.radians(45.0), 12.0, 4.0, 5.0, 0.45, math.radians(-30.0)]),
    ]
    return [params_to_unit(g) for g in guesses]


def crescent_core_image(params: np.ndarray, axis_uas: np.ndarray) -> np.ndarray:
    (
        f_crescent,
        radius,
        width,
        asymmetry,
        asym_pa,
        core_x,
        core_y,
        core_sigma_major,
        core_axis_ratio,
        core_pa,
    ) = params
    xg, yg = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xg * xg + yg * yg)
    theta = np.arctan2(yg, xg)

    crescent = np.exp(-0.5 * ((rr - radius) / max(width, 1e-3)) ** 2)
    crescent *= np.clip(1.0 + asymmetry * np.cos(theta - asym_pa), 0.03, None)

    dx = xg - core_x
    dy = yg - core_y
    xp = dx * np.cos(core_pa) + dy * np.sin(core_pa)
    yp = -dx * np.sin(core_pa) + dy * np.cos(core_pa)
    sigma_minor = max(core_sigma_major * core_axis_ratio, 1e-3)
    core = np.exp(-0.5 * ((xp / core_sigma_major) ** 2 + (yp / sigma_minor) ** 2))

    crescent /= max(float(np.sum(crescent)), 1e-300)
    core /= max(float(np.sum(core)), 1e-300)
    image = f_crescent * crescent + (1.0 - f_crescent) * core
    image = np.clip(image, 0.0, None)
    image /= max(float(np.sum(image)), 1e-300)
    return image


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


def image_metrics(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> dict[str, float]:
    metrics = latest.image_metrics(SOURCE, truth, image, axis_uas)
    metrics["radial_corr"] = radial_profile_corr(truth, image, axis_uas)
    return metrics


def simulate_case(case: aug.NetworkCase) -> tuple[list[dict[str, np.ndarray]], dict, np.ndarray, np.ndarray]:
    configure_simulation()
    with ngc.patched_source(SOURCE):
        return wt.simulate_bands(case)


class RMLData:
    def __init__(self, bands: list[dict[str, np.ndarray]], case: aug.NetworkCase, axis_uas: np.ndarray):
        stations, _, _, _ = aug.station_table_from_case(case)
        self.edges = base.edge_list(len(stations))
        self.n_edges = len(self.edges)
        # Use integer root-loop closure phases for the likelihood.  Projecting
        # wrapped edge phases onto a non-integer orthonormal basis can create
        # artificial jumps; root loops implement arg(prod V_ij) directly.
        self.w_basis = base.root_cycle_basis(self.edges, len(stations))
        self.fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
        self.uv_axis = np.fft.fftshift(np.fft.fftfreq(len(axis_uas), d=self.fov_rad / len(axis_uas)))
        self.u = np.concatenate([band["u"] for band in bands])
        self.v = np.concatenate([band["v"] for band in bands])
        self.amp = np.concatenate([band["amp"] for band in bands])
        self.amp_sigma = AMP_REL_SIGMA * np.maximum(self.amp, AMP_ABS_FLOOR)
        self.vis_data = {
            strategy: np.concatenate([band[f"vis_{strategy}"] for band in bands])
            for strategy in STRATEGIES
        }
        self.phase_data = {}
        self.phase_sigma = {}
        for strategy in STRATEGIES:
            phase = []
            sigma = []
            for band in bands:
                phase.append(np.angle(band[f"vis_{strategy}"]).reshape(-1, self.n_edges))
                sigma.append(np.asarray(band[f"sigma_{strategy}"]).reshape(-1, self.n_edges))
            self.phase_data[strategy] = np.vstack(phase)
            self.phase_sigma[strategy] = np.maximum(np.vstack(sigma), PHASE_FLOOR_RAD)

    def model_vis(self, params: np.ndarray, axis_uas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = crescent_core_image(params, axis_uas)
        vis_grid = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
        return image, base.interp_vis(vis_grid, self.uv_axis, self.u, self.v)

    def objective(
        self,
        unit_params: np.ndarray,
        axis_uas: np.ndarray,
        strategy: str,
        *,
        phase_scale: float = 1.0,
        amp_only: bool = False,
    ) -> tuple[float, dict[str, float]]:
        params = unit_to_params(unit_params)
        _image, model = self.model_vis(params, axis_uas)
        amp_resid = (np.abs(model) - self.amp) / self.amp_sigma
        amp_loss = float(np.mean(amp_resid * amp_resid))
        if amp_only:
            return AMP_WEIGHT * amp_loss, {"amp_chi2": amp_loss, "complex_chi2": 0.0}

        model_phase = np.angle(model).reshape(-1, self.n_edges)
        if strategy == "all":
            # Raw visibility imaging uses the complex visibility itself.  The
            # variance combines the common calibrated-amplitude uncertainty and
            # the phase uncertainty from the baseline readout plus piston drift.
            sigma_edge = self.phase_sigma[strategy].reshape(-1)
            complex_var = self.amp_sigma**2 + (np.maximum(self.amp, AMP_ABS_FLOOR) * sigma_edge) ** 2
            complex_residual = model - self.vis_data[strategy]
            complex_loss = float(np.mean(np.abs(complex_residual) ** 2 / np.maximum(complex_var, 1e-12)))
        else:
            # Closure-protected strategies use the complex normalized closure
            # phasor C=exp(i psi_cl), not a separate closure amplitude.  This is
            # the likelihood counterpart of directly reading closure phase.
            data_phase = self.phase_data[strategy]
            sigma = self.phase_sigma[strategy]
            model_loop = np.angle(np.exp(1j * (model_phase @ self.w_basis)))
            data_loop = np.angle(np.exp(1j * (data_phase @ self.w_basis)))
            model_closure = np.exp(1j * model_loop)
            data_closure = np.exp(1j * data_loop)
            residual_loop = model_closure - data_closure
            sigma_loop2 = (sigma**2) @ (self.w_basis**2)
            complex_loss = float(np.mean(np.abs(residual_loop) ** 2 / np.maximum(sigma_loop2, PHASE_FLOOR_RAD**2)))

        # Mild compactness prior: prevent the optimizer from using unphysical,
        # very broad cores to absorb amplitude residuals outside the BLR scale.
        compact_penalty = 0.004 * float((params[7] / 18.0) ** 2 + (params[2] / 20.0) ** 2)
        common_amp = COMMON_AMP_WEIGHT * amp_loss if strategy != "all" else 0.0
        total = common_amp + PHASE_WEIGHT * phase_scale * complex_loss + compact_penalty
        return total, {
            "amp_chi2": amp_loss,
            "complex_chi2": complex_loss,
            "common_amp_weight": COMMON_AMP_WEIGHT if strategy != "all" else 0.0,
            "compact_penalty": compact_penalty,
        }


def propose(rng: np.random.Generator, center: np.ndarray, step: np.ndarray) -> np.ndarray:
    out = center + rng.normal(scale=step, size=center.shape)
    # Periodic angular parameters.
    for idx in (4, 9):
        out[idx] = np.mod(out[idx], 1.0)
    return np.clip(out, 0.0, 1.0)


def optimize_unit(
    data: RMLData,
    axis_uas: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
    amp_only: bool = False,
    phase_scale: float = 1.0,
) -> tuple[np.ndarray, float, dict[str, float]]:
    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    guesses = canonical_unit_guesses()
    if start is not None:
        guesses.insert(0, start)
    for guess in guesses:
        value, parts = data.objective(guess, axis_uas, strategy, phase_scale=phase_scale, amp_only=amp_only)
        candidates.append((value, guess.copy(), parts))
    for _ in range(N_RANDOM):
        unit = rng.uniform(0.0, 1.0, size=len(PARAM_NAMES))
        value, parts = data.objective(unit, axis_uas, strategy, phase_scale=phase_scale, amp_only=amp_only)
        candidates.append((value, unit, parts))
    candidates.sort(key=lambda item: item[0])
    active = [(value, unit.copy(), parts) for value, unit, parts in candidates[:N_RESTARTS]]

    step = np.array([0.08, 0.07, 0.08, 0.10, 0.12, 0.08, 0.08, 0.08, 0.08, 0.12])
    best_value, best_unit, best_parts = active[0]
    for outer in range(5):
        new_active = []
        for value, unit, parts in active:
            local_value = value
            local_unit = unit.copy()
            local_parts = parts
            for _ in range(max(N_LOCAL // 5, 1)):
                trial = propose(rng, local_unit, step)
                trial_value, trial_parts = data.objective(
                    trial,
                    axis_uas,
                    strategy,
                    phase_scale=phase_scale,
                    amp_only=amp_only,
                )
                if trial_value < local_value:
                    local_value = trial_value
                    local_unit = trial
                    local_parts = trial_parts
            new_active.append((local_value, local_unit, local_parts))
        new_active.append((best_value, best_unit.copy(), best_parts))
        new_active.sort(key=lambda item: item[0])
        best_value, best_unit, best_parts = new_active[0]
        active = new_active[:N_RESTARTS]
        # Add a few children around the current best so that different
        # strategies can escape an amplitude-only local optimum.
        for _ in range(max(2, N_RESTARTS // 2)):
            child = propose(rng, best_unit, 0.65 * step)
            child_value, child_parts = data.objective(
                child,
                axis_uas,
                strategy,
                phase_scale=phase_scale,
                amp_only=amp_only,
            )
            active.append((child_value, child, child_parts))
        active.sort(key=lambda item: item[0])
        active = active[:N_RESTARTS]
        step *= 0.60
    return best_unit, best_value, best_parts


def fit_case(case: aug.NetworkCase, phase_scale: float = 1.0) -> dict:
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = simulate_case(case)
    data = RMLData(bands, case, axis_uas)
    rng = np.random.default_rng(RNG_SEED + abs(hash(case.key)) % 100_000)

    print(f"[fit-amp] {case.key}", flush=True)
    amp_unit, amp_value, amp_parts = optimize_unit(data, axis_uas, "split", rng, amp_only=True)

    images = {"truth": truth, "amp_only": crescent_core_image(unit_to_params(amp_unit), axis_uas)}
    metrics = {"amp_only": image_metrics(truth, images["amp_only"], axis_uas)}
    fits = {
        "amp_only": {
            "objective": amp_value,
            "parts": amp_parts,
            "params": dict(zip(PARAM_NAMES, unit_to_params(amp_unit), strict=True)),
        }
    }
    for strategy in STRATEGIES:
        print(f"[fit] {case.key} strategy={strategy}", flush=True)
        unit, value, parts = optimize_unit(
            data,
            axis_uas,
            strategy,
            rng,
            start=amp_unit,
            phase_scale=phase_scale,
        )
        image = crescent_core_image(unit_to_params(unit), axis_uas)
        images[strategy] = image
        metrics[strategy] = image_metrics(truth, image, axis_uas)
        fits[strategy] = {
            "objective": value,
            "parts": parts,
            "params": dict(zip(PARAM_NAMES, unit_to_params(unit), strict=True)),
        }

    stats.update(
        {
            "method": "parametric RML with reliable amplitudes and strategy-dependent phase/closure likelihood",
            "model_family": "crescent/ring-like disk plus elliptical Gaussian core",
            "source": SOURCE.name,
            "n_model": N_MODEL,
            "observing_days": OBSERVING_DAYS,
            "n_time_windows": N_TIME_WINDOWS,
            "exposure_s": EXPOSURE_S,
            "exposure_gap_s": EXPOSURE_GAP_S,
            "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
            "fiber_length_scale": FIBER_LENGTH_SCALE,
            "mode_false_positive": MODE_FALSE_POSITIVE,
            "pair_false_positive": PAIR_FALSE_POSITIVE,
            "snr_boost": SNR_BOOST,
            "amp_rel_sigma": AMP_REL_SIGMA,
            "amp_abs_floor": AMP_ABS_FLOOR,
            "phase_floor_rad": PHASE_FLOOR_RAD,
            "phase_scale": phase_scale,
            "common_amp_weight": COMMON_AMP_WEIGHT,
            "metrics": metrics,
            "fits": fits,
        }
    )
    return {"case": case, "stats": stats, "truth": truth, "axis_uas": axis_uas, "images": images, "metrics": metrics}


def plot_results(results: list[dict], tag: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(results), 4, figsize=(8.6, 2.30 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axes = axes[None, :]
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
        }
    )
    cols = [
        ("truth", "Input"),
        ("all", "All vis. + drift"),
        ("split", "Edge-first closure"),
        ("direct", "Direct closure"),
    ]
    row_labels = {
        "optimal8_ngc4151_hub_m2_m5": "Optimal 8",
        "hawaii_top4_remote3_ngc4151": "Hawaii+3",
        "hawaii_top4_remote4_ngc4151": "Hawaii+4",
    }
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
                m = result["metrics"][key]
                ax.set_title(f"{title}\nBLR={m['blr_corr']:.2f}, rad={m['radial_corr']:.2f}")
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 0:
                ax.set_ylabel(f"{row_labels.get(result['case'].key, result['case'].key)}\n" + r"$\Delta\delta$ ($\mu$as)")
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
            "Parametric RML: reliable amplitudes + phase/closure likelihood; "
            f"{SOURCE.name}, {OBSERVING_DAYS} d, SNR boost {SNR_BOOST:g}"
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
            rows.append({"case": result["case"].key, "strategy": key, **metric})
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
    phase_scale = float(os.environ.get("PARAM_RML_PHASE_SCALE", "1.0"))
    results = [fit_case(case, phase_scale=phase_scale) for case in make_cases()]
    tag = (
        f"parametric_closure_rml_crescent_core_{SOURCE.key}_{OBSERVING_DAYS}d_"
        f"snr{SNR_BOOST:g}_phase{phase_scale:g}_loss{FIBER_LOSS_DB_PER_KM:g}_"
        f"fp{MODE_FALSE_POSITIVE:g}_n{N_MODEL}"
    ).replace(".", "p")
    pdf, png = plot_results(results, tag)
    csv_path, json_path = write_outputs(results, tag, pdf, png)
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(result["case"].key)
        for strategy in STRATEGIES:
            print(" ", strategy, result["metrics"][strategy], result["stats"]["fits"][strategy]["parts"])


if __name__ == "__main__":
    main()
