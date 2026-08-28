from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import plot_prl_broadband_blr_optimized as opt
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    config = {
        "label": "current",
        "prior": amp_rml.PRIOR_WEIGHT,
        "tv": amp_rml.TV_WEIGHT,
        "entropy": amp_rml.ENTROPY_WEIGHT,
        "step": amp_rml.STEP,
    }

    runs = []
    for start_name in ("direct_dirty", "split_dirty", "prior"):
        print(f"[rml] {case.key} start={start_name}", flush=True)
        runs.append(
            val.run_single_reconstruction(
                case=case,
                bands=bands,
                truth=truth,
                axis_uas=axis_uas,
                prior=prior,
                start_name=start_name,
                start=starts[start_name],
                config=config,
                split_label="full_current",
            )
        )

    best = min(runs, key=lambda item: item["validation_score"])
    tag = (
        f"{case.key}_direct_closure_joint_rml_best_"
        f"fit{val.FIT_N_PIX}_shown{amp_rml.N_RML}_"
        f"{amp_rml.sigma_tag()}_{val.OPTIMIZER}"
        f"i{val.ADAM_ITER}lr{val.ADAM_LR:g}_"
        f"aw{amp_rml.AMP_GRAD_WEIGHT:g}pw{amp_rml.PHASE_GRAD_WEIGHT:g}_"
        f"{amp_rml.OBSERVING_DAYS}d"
    ).replace(".", "p")
    npz_path = OUTFIG / f"{tag}_best_images.npz"
    np.savez_compressed(
        npz_path,
        truth=np.asarray(truth, dtype=float),
        best_fit_image=np.asarray(best["fit_image"], dtype=float),
        best_display_image=np.asarray(best["image"], dtype=float),
        axis_uas=np.asarray(axis_uas, dtype=float),
        case_key=str(case.key),
        best_start=str(best["start"]),
        best_config=str(best["config"]),
        strategy=str(best["strategy"]),
        optimizer=str(best["optimizer"]),
        fit_n_pix=int(val.FIT_N_PIX),
        shown_n_pix=int(amp_rml.N_RML),
        adam_iter=int(val.ADAM_ITER),
        adam_lr=float(val.ADAM_LR),
        adam_target_amp_chi2=float(val.ADAM_TARGET_AMP_CHI2),
        adam_target_phase_chi2=float(val.ADAM_TARGET_PHASE_CHI2),
        amp_grad_weight=float(amp_rml.AMP_GRAD_WEIGHT),
        phase_grad_weight=float(amp_rml.PHASE_GRAD_WEIGHT),
        amp_sigma_mode=str(amp_rml.AMP_SIGMA_MODE),
        amp_sigma_abs=float(amp_rml.AMP_SIGMA_ABS),
        amp_rel_sigma=float(amp_rml.AMP_REL_SIGMA),
        amp_abs_floor=float(amp_rml.AMP_ABS_FLOOR),
        phase_floor_rad=float(amp_rml.PHASE_FLOOR_RAD),
        observing_days=int(amp_rml.OBSERVING_DAYS),
        n_time_windows=int(amp_rml.N_TIME_WINDOWS),
        exposure_s=float(amp_rml.EXPOSURE_S),
        fiber_loss_db_per_km=float(amp_rml.FIBER_LOSS_DB_PER_KM),
        mode_false_positive=float(amp_rml.MODE_FALSE_POSITIVE),
        pair_false_positive=float(amp_rml.PAIR_FALSE_POSITIVE),
    )

    summary = {
        "case": case.key,
        "best_start": best["start"],
        "metrics": best["metrics"],
        "residuals": best["residuals"],
        "validation_score": best["validation_score"],
        "npz": str(npz_path),
        "config": {
            "fit_n_pix": val.FIT_N_PIX,
            "shown_n_pix": amp_rml.N_RML,
            "amp_sigma_mode": amp_rml.AMP_SIGMA_MODE,
            "amp_sigma_abs": amp_rml.AMP_SIGMA_ABS,
            "amp_rel_sigma": amp_rml.AMP_REL_SIGMA,
            "phase_floor_rad": amp_rml.PHASE_FLOOR_RAD,
            "optimizer": val.OPTIMIZER,
            "adam_iter": val.ADAM_ITER,
            "adam_lr": val.ADAM_LR,
            "adam_target_amp_chi2": val.ADAM_TARGET_AMP_CHI2,
            "adam_target_phase_chi2": val.ADAM_TARGET_PHASE_CHI2,
            "amp_grad_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_grad_weight": amp_rml.PHASE_GRAD_WEIGHT,
            "physical_iter": val.PHYSICAL_ITER,
            "physical_step": val.PHYSICAL_STEP,
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "exposure_s": amp_rml.EXPOSURE_S,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
        },
        "all_starts": [
            {
                "start": run["start"],
                "validation_score": run["validation_score"],
                "metrics": run["metrics"],
                "residuals": run["residuals"],
            }
            for run in runs
        ],
    }
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), constrained_layout=True)
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    panels = [
        (truth, "Input source"),
        (best["image"], f"Best RML\nstart={best['start']}"),
        (np.abs(best["image"] - truth), "Absolute image residual"),
    ]
    image_axes = []
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        image_axes.append(ax)
    axes[0].set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle(
        (
            f"Hawaii+3 direct-closure RML, {amp_rml.OBSERVING_DAYS} d, "
            f"{amp_rml.sigma_tag()}, phase floor={amp_rml.PHASE_FLOOR_RAD:g} rad\n"
            f"amp chi2={best['residuals']['amp_reduced_chi2']:.2f}, "
            f"CP chi2={best['residuals']['phase_reduced_chi2']:.2f}"
        ),
        fontsize=9.5,
        weight="bold",
    )
    png_path = OUTFIG / f"{tag}_best_image.png"
    pdf_path = OUTFIG / f"{tag}_best_image.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    summary["best_image_png"] = str(png_path)
    summary["best_image_pdf"] = str(pdf_path)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(npz_path)
    print(png_path)
    print(pdf_path)
    print(json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
