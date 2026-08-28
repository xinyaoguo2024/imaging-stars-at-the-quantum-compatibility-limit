from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import plot_prl_broadband_blr_optimized as opt
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

Y_SCALE = float(os.environ.get("HAWAII3_Y_SCALE", "1.0"))


STRATEGIES = [
    ("all", "All visibilities\nwith piston drift", "all_dirty"),
    ("split", "Edge-first\nclosure", "split_dirty"),
    ("direct", "Direct\nclosure", "direct_dirty"),
]


def run_strategy(
    strategy: str,
    label: str,
    start_name: str,
    case,
    bands,
    truth,
    axis_uas,
    prior,
    starts,
) -> dict:
    old_strategy = val.STRATEGY
    old_optimizer = val.OPTIMIZER
    val.STRATEGY = strategy
    val.OPTIMIZER = "adam"
    config = {
        "label": "adam_datafit",
        "prior": amp_rml.PRIOR_WEIGHT,
        "tv": amp_rml.TV_WEIGHT,
        "entropy": amp_rml.ENTROPY_WEIGHT,
        "step": amp_rml.STEP,
    }
    candidates = []
    for candidate_start in (start_name, "prior"):
        print(f"[rml] strategy={strategy} start={candidate_start}", flush=True)
        candidates.append(
            val.run_single_reconstruction(
                case=case,
                bands=bands,
                truth=truth,
                axis_uas=axis_uas,
                prior=prior,
                start_name=candidate_start,
                start=starts[candidate_start],
                config=config,
                split_label="strategy_comparison",
            )
        )
    val.STRATEGY = old_strategy
    val.OPTIMIZER = old_optimizer
    best = min(candidates, key=lambda item: item["validation_score"])
    return {
        "strategy": strategy,
        "label": label,
        "best": best,
        "candidates": candidates,
    }


def serializable_result(item: dict) -> dict:
    return {
        "strategy": item["strategy"],
        "start": item["start"],
        "validation_score": float(item["validation_score"]),
        "metrics": {key: float(value) for key, value in item["metrics"].items()},
        "residuals": {key: float(value) for key, value in item["residuals"].items()},
    }


def drift_tag(value: float) -> str:
    return f"drift{value / np.pi:g}pi"


def value_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def scale_case_y(case):
    """Return a case with station north coordinates scaled by Y_SCALE.

    The hub is intentionally left fixed because the requested geometry change
    is a station-topology change; link lengths are then recomputed to the same
    central receiver location.
    """
    if abs(Y_SCALE - 1.0) < 1e-12:
        return case
    telescopes = [
        amp_rml.aug.Telescope(
            tel.name,
            float(tel.x_km),
            float(Y_SCALE * tel.y_km),
            float(tel.diameter_m),
            bool(tel.is_added),
        )
        for tel in case.telescopes
    ]
    return amp_rml.aug.NetworkCase(
        key=f"{case.key}_yscale{value_tag(Y_SCALE)}",
        title=f"{case.title}, y coordinates x {Y_SCALE:g}",
        latitude_deg=case.latitude_deg,
        center_latlon=case.center_latlon,
        telescopes=telescopes,
        hub_km=case.hub_km,
        optimization_score=case.optimization_score,
    )


def plot_prl_six_panel(case, stats, truth, axis_uas, results, tag: str) -> tuple[Path, Path]:
    stations, diameters, _names, is_added = amp_rml.aug.station_table_from_case(case)
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.1,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
        }
    )
    fig = plt.figure(figsize=(7.55, 4.88), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.40, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    for added, marker, color, label in (
        (False, "o", "#005f73", "existing"),
        (True, "^", "#ae2012", "new 5 m"),
    ):
        mask = is_added == added
        if np.any(mask):
            ax.scatter(
                stations[mask, 0],
                stations[mask, 1],
                s=30 if added else 25,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
                zorder=3,
            )
    ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=55, marker="*", color="#ca6702", label="hub", zorder=4)
    for i, j in amp_rml.base.edge_list(len(stations)):
        ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.83", lw=0.42, zorder=0)
    label_offsets = {
        0: (-0.95, 0.45),
        1: (0.15, 0.72),
        2: (-1.35, -0.35),
        3: (0.55, -0.78),
        4: (0.18, 0.34),
        5: (0.18, 0.34),
        6: (0.18, 0.34),
    }
    for i, (x, y) in enumerate(stations):
        dx, dy = label_offsets.get(i, (0.2, 0.2))
        ax.text(x + dx, y + dy, f"S{i + 1}\n{diameters[i]:g}m", fontsize=5.35)
    x_pad = max(0.9, 0.08 * (float(np.max(stations[:, 0])) - float(np.min(stations[:, 0]))))
    y_pad = max(0.9, 0.10 * (float(np.max(stations[:, 1])) - float(np.min(stations[:, 1]))))
    ax.set_xlim(float(np.min(stations[:, 0])) - x_pad, float(np.max(stations[:, 0])) + x_pad)
    ax.set_ylim(float(np.min(stations[:, 1])) - y_pad, float(np.max(stations[:, 1])) + 1.55 * y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title("Maunakea top4 + remote3")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.15, borderpad=0.1)

    ax = fig.add_subplot(gs[0, 1])
    for wavelength, color, alpha in (("400", "#005f73", 0.50), ("800", "#ee9b00", 0.42)):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax.scatter(uu, vv, s=1.2, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax.scatter(-uu, -vv, s=1.2, color=color, alpha=0.62 * alpha)
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 256)
    for theta_uas, ls in ((60.0, ":"), (30.0, "--"), (10.0, "-.")):
        radius_g_lambda = 1.0 / (theta_uas * amp_rml.base.UAS_TO_RAD) / 1.0e9
        ax.plot(
            radius_g_lambda * np.cos(theta_circle),
            radius_g_lambda * np.sin(theta_circle),
            ls=ls,
            lw=0.55,
            color="0.35",
            alpha=0.70,
            label=rf"{theta_uas:g} $\mu$as",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.1, borderpad=0.1)

    image_axes = []
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_title("Input NGC 4151\nnon-spotted BLR")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    image_axes.append(ax)

    result_by_strategy = {result["strategy"]: result for result in results}
    labels = {
        "all": "All visibilities + drift",
        "split": "Edge-first closure",
        "direct": "Direct closure-space",
    }
    for col, strategy in enumerate(("all", "split", "direct")):
        result = result_by_strategy[strategy]
        ax = fig.add_subplot(gs[1, col])
        image = result["best"]["image"]
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        metric = result["best"]["metrics"]
        residual = result["best"]["residuals"]
        ax.set_title(
            f"{labels[strategy]}\nBLR r={metric['blr_corr']:.2f}, all r={metric['global_corr']:.2f}",
            fontsize=7.8,
        )
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
    cbar.set_label("norm. brightness\n(BLR-emphasis arcsinh)", fontsize=6.8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    fig.suptitle(
        f"RML imaging benchmark: {case.title}",
        fontsize=9.6,
        weight="bold",
        y=0.995,
    )
    tagged_png = OUTFIG / f"{tag}_prl_6panel.png"
    tagged_pdf = OUTFIG / f"{tag}_prl_6panel.pdf"
    if abs(Y_SCALE - 1.0) < 1e-12:
        stable_stem = "prl_fig3_hawaii4_remote3_ngc4151_rml_6panel"
    else:
        stable_stem = f"prl_fig3_hawaii4_remote3_ngc4151_y{value_tag(Y_SCALE)}_rml_6panel"
    stable_png = OUTFIG / f"{stable_stem}.png"
    stable_pdf = OUTFIG / f"{stable_stem}.pdf"
    fig.savefig(tagged_png, dpi=260, bbox_inches="tight")
    fig.savefig(tagged_pdf, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(tagged_png, stable_png)
    shutil.copyfile(tagged_pdf, stable_pdf)
    return stable_pdf, stable_png


def main() -> None:
    case = scale_case_y(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)

    results = [
        run_strategy(strategy, label, start_name, case, bands, truth, axis_uas, prior, starts)
        for strategy, label, start_name in STRATEGIES
    ]

    tag = (
        f"{case.key}_rml_strategy_comparison_"
        f"{amp_rml.ngc.SOURCE_MORPHOLOGY}_"
        f"fit{val.FIT_N_PIX}_shown{amp_rml.N_RML}_"
        f"{amp_rml.sigma_tag()}_{drift_tag(float(stats.get('post_average_drift_std_rad', 0.0)))}_adam"
        f"i{val.ADAM_ITER}lr{val.ADAM_LR:g}_"
        f"aw{amp_rml.AMP_GRAD_WEIGHT:g}pw{amp_rml.PHASE_GRAD_WEIGHT:g}_"
        f"{amp_rml.OBSERVING_DAYS}d"
    ).replace(".", "p")

    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.2), constrained_layout=True)
    panels = [("truth", "Input source", truth)] + [
        (result["strategy"], result["label"], result["best"]["image"]) for result in results
    ]
    for col, (key, title, image) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        if key == "truth":
            ax.set_title(title, fontsize=9.0)
        else:
            m = next(item for item in results if item["strategy"] == key)["best"]["metrics"]
            r = next(item for item in results if item["strategy"] == key)["best"]["residuals"]
            ax.set_title(
                (
                    f"{title}\n"
                    f"BLR={m['blr_corr']:.2f}, all={m['global_corr']:.2f}\n"
                    f"$\\chi^2_A$={r['amp_reduced_chi2']:.3g}, "
                    f"$\\chi^2_\\phi$={r['phase_reduced_chi2']:.2f}"
                ),
                fontsize=8.2,
            )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        else:
            ax.set_yticklabels([])

    truth_norm = opt.normalize_blr_display(truth)
    for col, result in enumerate(results, start=1):
        ax = axes[1, col]
        image_norm = opt.normalize_blr_display(result["best"]["image"])
        residual = image_norm - truth_norm
        vmax = max(0.12, float(np.percentile(np.abs(residual), 99.0)))
        im = ax.imshow(residual, origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{result['label']} - input", fontsize=8.2)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_yticklabels([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.95,
        (
            "Same simulation and RML solver:\n"
            f"{amp_rml.OBSERVING_DAYS} day, {amp_rml.sigma_tag()}, "
            f"phase floor={amp_rml.PHASE_FLOOR_RAD:g} rad\n"
            f"Adam {val.ADAM_ITER} steps, lr={val.ADAM_LR:g}, "
            f"phase weight={amp_rml.PHASE_GRAD_WEIGHT:g}\n\n"
            "Bottom row shows displayed-image residuals relative to the input source."
        ),
        va="top",
        ha="left",
        fontsize=8.0,
    )
    fig.suptitle(
        "Hawaii+3 RML comparison: all visibilities vs edge-first closure vs direct closure",
        fontsize=11.0,
        weight="bold",
    )
    png_path = OUTFIG / f"{tag}.png"
    pdf_path = OUTFIG / f"{tag}.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    prl_pdf, prl_png = plot_prl_six_panel(case, stats, truth, axis_uas, results, tag)

    rows = []
    for result in results:
        best = serializable_result(result["best"])
        rows.append(
            {
                "strategy": result["strategy"],
                "label": result["label"].replace("\n", " "),
                "best_start": best["start"],
                "validation_score": best["validation_score"],
                **{f"metric_{key}": value for key, value in best["metrics"].items()},
                **{f"resid_{key}": value for key, value in best["residuals"].items()},
            }
        )
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "case": case.key,
        "hawaii3_y_scale": Y_SCALE,
        "source": amp_rml.SOURCE.name,
        "simulation_stats": stats,
        "environment": {
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "exposure_s": amp_rml.EXPOSURE_S,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
            "pair_false_positive": amp_rml.PAIR_FALSE_POSITIVE,
            "post_average_drift_std_rad": stats.get("post_average_drift_std_rad"),
            "amp_sigma_mode": amp_rml.AMP_SIGMA_MODE,
            "amp_sigma_abs": amp_rml.AMP_SIGMA_ABS,
            "phase_floor_rad": amp_rml.PHASE_FLOOR_RAD,
            "fit_n_pix": val.FIT_N_PIX,
            "shown_n_pix": amp_rml.N_RML,
            "optimizer": "adam",
            "adam_iter": val.ADAM_ITER,
            "adam_lr": val.ADAM_LR,
            "adam_target_amp_chi2": val.ADAM_TARGET_AMP_CHI2,
            "adam_target_phase_chi2": val.ADAM_TARGET_PHASE_CHI2,
            "amp_grad_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_grad_weight": amp_rml.PHASE_GRAD_WEIGHT,
        },
        "results": [
            {
                "strategy": result["strategy"],
                "label": result["label"],
                "best": serializable_result(result["best"]),
                "candidates": [serializable_result(item) for item in result["candidates"]],
            }
            for result in results
        ],
        "figures": {
            "comparison_png": str(png_path),
            "comparison_pdf": str(pdf_path),
            "prl_6panel_png": str(prl_png),
            "prl_6panel_pdf": str(prl_pdf),
            "metrics_csv": str(csv_path),
        },
    }
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(json_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
