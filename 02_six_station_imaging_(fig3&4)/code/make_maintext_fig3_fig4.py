#!/usr/bin/env python3
"""Generate the current main-text Figs. 3 and 4 from archived results.

The four-panel figure uses the representative 100-ms, seed-20260536 display
cache.  The statistics panel uses the paired twelve-seed ensemble generated
from the same 2%-efficiency benchmark.  No RML reconstruction or receiver
calibration is recomputed here, and every input path is local to this module.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated_outputs"
OUT.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_STEM = (
    "results_12seed_raw/100ms/seed_20260536/rml_outputs/"
    "broad_plume_split_objective_nmode_rml_paired_povm_100ms_seed20260536_"
    "ns3_eff002_perband_localclosure_nogauge_nocoreprior_chi0p85w075to100"
)
DISPLAY_CACHE = ROOT / f"{REPRESENTATIVE_STEM}_selected_display_cache.npz"
REPRESENTATIVE_SUMMARY = ROOT / f"{REPRESENTATIVE_STEM}_summary.json"
STATISTICS_JSON = ROOT / "plotting_data" / "paired_12seed_100ms_summary.json"
SEED_METRICS_CSV = ROOT / "plotting_data" / "paired_12seed_100ms_seed_metrics.csv"

# Exact layout used in the current manuscript figure.
STATIONS_KM = np.asarray(
    [
        [-0.233697412234395, 0.17351968302884233],
        [-0.42037081717666885, 0.056037795231751446],
        [0.308923898971121, -0.13327267932961595],
        [-1.8415198336534673, 0.6632204741890173],
        [3.863129857898074, 0.8818018564192034],
        [-3.6343232752108077, -6.99853550068413],
    ],
    dtype=float,
)
DIAMETERS_M = np.asarray([10.0, 8.2, 8.1, 6.0, 6.0, 6.0])
IS_REMOTE = np.asarray([False, False, False, True, True, True])
HUB_KM = np.asarray([-0.1796718400995452, 0.06880134886697599])

CUTOFF = 0.015
# Include the display-cutoff isophote so that the outer BLR morphology is tied
# directly to the lowest labelled brightness level on the colorbar.
CONTOUR_LEVELS = (0.015, 0.03, 0.10, 0.30)
UAS_TO_RAD = np.deg2rad(1.0 / 3.6e9)

COLORS = {
    "edge_uniform": "#1687b8",
    "optimal_singlecopy": "#159c9c",
    "collective_ns3": "#b51219",
}
STRATEGIES = ("edge_uniform", "optimal_singlecopy", "collective_ns3")
STRATEGY_LABELS = {
    "edge_uniform": "Uniform\nedge-first",
    "optimal_singlecopy": "Optimal\nsingle-copy",
    "collective_ns3": "Collective\n($n_s=3$)",
}


def normalize(image: np.ndarray) -> np.ndarray:
    # Match the archived display normalization exactly: remove the
    # 0.8-percentile numerical floor and use the 99.7-percentile brightness as
    # the display scale.  ``CUTOFF`` then defines the plotted support mask.
    image = np.asarray(image, dtype=float).copy()
    image -= np.percentile(image, 0.8)
    scale = float(np.percentile(image, 99.7))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(np.abs(image)))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(image)
    return np.clip(image / scale, 0.0, 1.0)


def masked_display(image: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_less(normalize(image), CUTOFF)


def display_colormap():
    """High-contrast multicolour map with a black sub-cutoff background."""
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    return cmap


def draw_topology(ax: plt.Axes) -> None:
    for i, j in itertools.combinations(range(len(STATIONS_KM)), 2):
        ax.plot(
            [STATIONS_KM[i, 0], STATIONS_KM[j, 0]],
            [STATIONS_KM[i, 1], STATIONS_KM[j, 1]],
            color="0.84",
            lw=0.35,
            zorder=0,
        )
    ax.scatter(
        STATIONS_KM[~IS_REMOTE, 0],
        STATIONS_KM[~IS_REMOTE, 1],
        s=15,
        marker="o",
        color="#005f73",
        edgecolor="white",
        linewidth=0.25,
        label="existing",
        zorder=3,
    )
    ax.scatter(
        STATIONS_KM[IS_REMOTE, 0],
        STATIONS_KM[IS_REMOTE, 1],
        s=21,
        marker="^",
        color="#ae2012",
        edgecolor="white",
        linewidth=0.25,
        label="remote 6 m",
        zorder=3,
    )
    ax.scatter(*HUB_KM, s=34, marker="*", color="#ca6702", label="hub", zorder=4)

    core = STATIONS_KM[~IS_REMOTE]
    core_x0, core_x1 = float(core[:, 0].min() - 0.18), float(core[:, 0].max() + 0.18)
    core_y0, core_y1 = float(core[:, 1].min() - 0.18), float(core[:, 1].max() + 0.18)
    ax.add_patch(
        Rectangle(
            (core_x0, core_y0),
            core_x1 - core_x0,
            core_y1 - core_y0,
            facecolor="none",
            edgecolor="0.2",
            lw=0.5,
            ls="--",
        )
    )
    # The aperture is already encoded in the legend; compact station-only
    # labels keep the remote-site annotations clear of baselines and markers.
    label_positions = {
        3: (-2.42, 0.18, "left"),
        4: (3.18, 0.18, "left"),
        5: (-3.45, -6.55, "left"),
    }
    for i in np.flatnonzero(IS_REMOTE):
        x_text, y_text, horizontal_alignment = label_positions[i]
        ax.text(
            x_text,
            y_text,
            f"S{i + 1}",
            fontsize=5.2,
            ha=horizontal_alignment,
            va="center",
        )

    inset = ax.inset_axes([0.55, 0.51, 0.41, 0.35])
    for i, j in itertools.combinations(np.flatnonzero(~IS_REMOTE), 2):
        inset.plot(
            [STATIONS_KM[i, 0], STATIONS_KM[j, 0]],
            [STATIONS_KM[i, 1], STATIONS_KM[j, 1]],
            color="0.72",
            lw=0.42,
        )
    inset.scatter(
        core[:, 0], core[:, 1], s=12, color="#005f73", edgecolor="white", linewidth=0.2, zorder=3
    )
    inset.scatter(*HUB_KM, s=22, marker="*", color="#ca6702", zorder=4)
    inset_offsets = {0: (-0.035, 0.080), 1: (-0.145, 0.020), 2: (0.038, -0.022)}
    for i in range(3):
        dx, dy = inset_offsets[i]
        inset.text(STATIONS_KM[i, 0] + dx, STATIONS_KM[i, 1] + dy, f"S{i + 1}", fontsize=4.4)
    # Restore the physical scale marker used in the original zoomed-in core
    # panel.  Coordinates are in kilometres, so the bar length is exact.
    scale_x0 = core_x0 + 0.080
    scale_x1 = scale_x0 + 0.20
    scale_y = core_y0 + 0.070
    tick_half = 0.018
    inset.plot([scale_x0, scale_x1], [scale_y, scale_y], color="0.12", lw=0.75, zorder=6)
    inset.plot(
        [scale_x0, scale_x0],
        [scale_y - tick_half, scale_y + tick_half],
        color="0.12",
        lw=0.75,
        zorder=6,
    )
    inset.plot(
        [scale_x1, scale_x1],
        [scale_y - tick_half, scale_y + tick_half],
        color="0.12",
        lw=0.75,
        zorder=6,
    )
    inset.text(
        0.5 * (scale_x0 + scale_x1),
        scale_y + 0.028,
        "0.2 km",
        ha="center",
        va="bottom",
        fontsize=4.2,
        color="0.12",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.10},
        zorder=7,
    )
    inset.set(xlim=(core_x0, core_x1), ylim=(core_y0, core_y1), xticks=[], yticks=[])
    inset.set_aspect("equal")
    for spine in inset.spines.values():
        spine.set_linewidth(0.45)

    y_min, y_max = float(STATIONS_KM[:, 1].min() - 0.8), float(STATIONS_KM[:, 1].max() + 0.8)
    x_min = float(STATIONS_KM[:, 0].min() - 0.65)
    ax.set_xlim(x_min, x_min + (y_max - y_min))
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.yaxis.set_label_coords(-0.16, 0.50)
    ax.set_title("Array topology", pad=3)
    ax.legend(loc="lower right", frameon=False, handlelength=1.0, handletextpad=0.15, borderpad=0.1)


def draw_uv(ax: plt.Axes, stats: dict) -> None:
    endpoint_items = (("600", "#005f73", 0.55), ("700", "#ee9b00", 0.48))
    for wavelength, color, alpha in endpoint_items:
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        u, v = np.asarray(coverage["u"]), np.asarray(coverage["v"])
        ax.scatter(u, v, s=0.9, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax.scatter(-u, -v, s=0.9, color=color, alpha=0.65 * alpha)
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    for theta_uas, ls in ((60.0, ":"), (30.0, "--"), (10.0, "-.")):
        radius = 1.0 / (theta_uas * UAS_TO_RAD) / 1.0e9
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), ls=ls, lw=0.45, color="0.35", label=rf"{theta_uas:g} $\mu$as")
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    # Tighten only the numerical y-tick labels against their tick marks.  The
    # axis frame, tick locations, and the v-axis title remain fixed.
    ax.tick_params(axis="y", pad=0.8)
    # Keep the vertical label tied to this axis rather than positioning it in
    # figure coordinates; a small label pad places it directly beside the
    # tick labels without letting it drift into the topology panel.
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.set_title("Fourier coverage", pad=3)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        handlelength=1.2,
        handletextpad=0.15,
        borderpad=0.1,
    )


def add_truth_contours(ax: plt.Axes, truth: np.ndarray, extent: list[float]) -> None:
    normalized_truth = normalize(truth)
    ax.contour(
        normalized_truth,
        levels=CONTOUR_LEVELS,
        origin="lower",
        extent=extent,
        colors="black",
        linewidths=(1.00, 1.10, 1.30, 1.50),
        linestyles=((0, (1.0, 1.2)), ":", "--", "-"),
        alpha=0.70,
        zorder=4,
    )
    ax.contour(
        normalized_truth,
        levels=CONTOUR_LEVELS,
        origin="lower",
        extent=extent,
        colors="white",
        linewidths=(0.52, 0.60, 0.75, 0.90),
        linestyles=((0, (1.0, 1.2)), ":", "--", "-"),
        alpha=1.0,
        zorder=5,
    )


def make_four_panel() -> tuple[Path, Path]:
    cache = np.load(DISPLAY_CACHE)
    truth = np.asarray(cache["truth"])
    collective = np.asarray(cache["promoted_singlecopy_image"])
    axis = np.asarray(cache["axis_uas"])
    extent = [float(axis[0]), float(axis[-1]), float(axis[0]), float(axis[-1])]
    summary = json.loads(REPRESENTATIVE_SUMMARY.read_text())
    stats = summary["stats"]

    style = {
        "font.family": "sans-serif",
        "font.size": 6.6,
        "axes.labelsize": 6.7,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 5.9,
        "ytick.labelsize": 5.9,
        "legend.fontsize": 5.1,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        # Use the same canvas and row coordinates as the vertically stacked
        # statistics figure.  This makes both single-column figures align when
        # they are placed in adjacent manuscript columns.
        fig = plt.figure(figsize=(3.38, 4.00))
        axes = np.asarray(
            [
                [fig.add_axes([0.14, 0.650, 0.37, 0.310]), fig.add_axes([0.59, 0.650, 0.37, 0.310])],
                [fig.add_axes([0.14, 0.160, 0.37, 0.310]), fig.add_axes([0.59, 0.160, 0.37, 0.310])],
            ]
        )
        draw_topology(axes[0, 0])
        draw_uv(axes[0, 1], stats)

        cmap = display_colormap()
        norm = mcolors.LogNorm(vmin=CUTOFF, vmax=1.0)
        for ax, image, title in (
            (axes[1, 0], truth, "Input source"),
            (
                axes[1, 1],
                collective,
                "Collective ($n_s=3$)",
            ),
        ):
            ax.imshow(masked_display(image), origin="lower", extent=extent, cmap=cmap, norm=norm)
            add_truth_contours(ax, truth, extent)
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            ax.set_box_aspect(1)
            ax.set_title(title, pad=3)
        axes[1, 0].set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        axes[1, 0].yaxis.set_label_coords(-0.16, 0.50)
        axes[1, 1].set_yticklabels([])

        cax = fig.add_axes([0.31, 0.030, 0.61, 0.015])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="horizontal",
            extend="min",
        )
        cbar.set_ticks([CUTOFF, 0.03, 0.1, 0.3, 1.0])
        cbar.set_ticklabels([f"{CUTOFF:g}", "0.03", "0.1", "0.3", "1"])
        cbar.ax.tick_params(labelsize=5.4, width=0.45, length=1.8, pad=1.0)
        fig.text(0.290, 0.0375, "norm. brightness", ha="right", va="center", fontsize=6.0)

        pdf = OUT / "fig3_fourpanel_singlecolumn.pdf"
        png = OUT / "fig3_fourpanel_singlecolumn.png"
        fig.savefig(pdf)
        fig.savefig(png, dpi=500)
        plt.close(fig)
    return pdf, png


def load_seed_values() -> dict[str, dict[str, np.ndarray]]:
    rows = list(csv.DictReader(SEED_METRICS_CSV.open()))
    result: dict[str, dict[str, np.ndarray]] = {}
    for strategy in STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy]
        result[strategy] = {
            "blr_corr": np.asarray([float(row["blr_corr"]) for row in selected]),
            "global_corr": np.asarray([float(row["global_corr"]) for row in selected]),
        }
    return result


def correlation_ylim(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(arrays)
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 0.02)
    return max(0.0, lo - 0.06 * span), min(1.0, hi + 0.34 * span)


def make_vertical_statistics() -> tuple[Path, Path]:
    payload = json.loads(STATISTICS_JSON.read_text())
    stats = payload["statistics"]
    loop_stats = payload["loop_gain_statistics"]
    values = load_seed_values()

    style = {
        "font.family": "sans-serif",
        "font.size": 7.1,
        "axes.labelsize": 7.3,
        "axes.titlesize": 7.6,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.1,
        "axes.linewidth": 0.60,
        "xtick.major.width": 0.50,
        "ytick.major.width": 0.50,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        # Match the four-panel figure's canvas and row positions exactly.
        fig = plt.figure(figsize=(3.38, 4.00))
        ax = fig.add_axes([0.17, 0.650, 0.68, 0.310])
        ax_right = ax.twinx()
        x = np.arange(len(STRATEGIES), dtype=float)
        width = 0.34
        rng = np.random.default_rng(31415)
        for target, offset, metric, hatch, alpha in (
            (ax, -width / 2, "blr_corr", "", 0.96),
            (ax_right, width / 2, "global_corr", "//", 0.47),
        ):
            means = [float(stats[metric][key]["mean"]) for key in STRATEGIES]
            sems = [float(stats[metric][key]["sem"]) for key in STRATEGIES]
            target.bar(
                x + offset,
                means,
                width=width,
                color=[COLORS[key] for key in STRATEGIES],
                alpha=alpha,
                edgecolor="0.18",
                linewidth=0.5,
                hatch=hatch,
                zorder=2,
            )
            for i, key in enumerate(STRATEGIES):
                seed_values = values[key][metric]
                target.scatter(
                    np.full(seed_values.size, x[i] + offset) + rng.normal(0.0, 0.012, seed_values.size),
                    seed_values,
                    s=9,
                    color=COLORS[key],
                    edgecolor="white",
                    linewidth=0.25,
                    alpha=0.85,
                    zorder=3,
                )
            # Draw the mean uncertainty last so that neither the bars nor the
            # paired-seed markers can obscure the SEM caps and stems.
            target.errorbar(
                x + offset,
                means,
                yerr=sems,
                fmt="none",
                ecolor="0.10",
                elinewidth=0.85,
                capsize=2.0,
                capthick=0.85,
                zorder=5,
            )
        ax.set_xticks(x, [STRATEGY_LABELS[key] for key in STRATEGIES])
        ax.set_xlim(-0.48, len(STRATEGIES) - 0.52)
        ax.set_ylim(*correlation_ylim([values[key]["blr_corr"] for key in STRATEGIES]))
        ax_right.set_ylim(*correlation_ylim([values[key]["global_corr"] for key in STRATEGIES]))
        ax.set_ylabel("BLR correlation")
        ax_right.set_ylabel("all-pixel correlation")
        ax.set_title("Image fidelity", pad=4)
        ax.grid(axis="y", color="0.88", lw=0.45)
        ax.set_axisbelow(True)
        ax.legend(
            handles=[
                Patch(facecolor="0.45", edgecolor="0.18", label=r"BLR mean $\pm$ SEM"),
                Patch(facecolor="0.72", edgecolor="0.18", hatch="//", alpha=0.55, label=r"all-pixel mean $\pm$ SEM"),
            ],
            loc="upper center",
            ncol=2,
            frameon=False,
            handlelength=1.0,
            columnspacing=0.8,
            borderaxespad=0.15,
        )

        ax = fig.add_axes([0.17, 0.160, 0.78, 0.310])
        loop_records = loop_stats["optimal_singlecopy"]["per_loop"]
        labels = [record["loop"] for record in loop_records]
        x_loop = np.arange(len(labels), dtype=float)
        gain_width = 0.36
        gain_specs = (
            ("optimal_singlecopy", -gain_width / 2, "Optimal single-copy", COLORS["optimal_singlecopy"]),
            ("collective_ns3", gain_width / 2, r"Collective ($n_s=3$)", COLORS["collective_ns3"]),
        )
        upper_gain = 1.0
        for key, offset, label, color in gain_specs:
            records = loop_stats[key]["per_loop"]
            centers = np.asarray([float(item["geometric_mean"]) for item in records])
            lower = centers - np.asarray([float(item["quantile_05"]) for item in records])
            upper = np.asarray([float(item["quantile_95"]) for item in records]) - centers
            upper_gain = max(upper_gain, float(np.max(centers + upper)))
            ax.bar(
                x_loop + offset,
                centers,
                width=gain_width,
                yerr=np.vstack([lower, upper]),
                capsize=1.7,
                color=color,
                edgecolor="0.18",
                linewidth=0.5,
                label=label,
                zorder=2,
            )
        ax.axhline(1.0, color="0.25", lw=0.75, ls="--")
        ax.set_xticks(x_loop, labels, rotation=32, ha="right")
        ax.set_xlabel("loop label", labelpad=3.0)
        ax.set_ylabel("closure phase SNR gain")
        ax.set_title("Parameter-estimation precision", pad=4)
        ax.grid(axis="y", color="0.88", lw=0.45)
        ax.set_axisbelow(True)
        ax.set_ylim(0.85, max(3.0, 1.13 * upper_gain))
        ax.legend(frameon=False, loc="upper center", ncol=2, columnspacing=0.8)

        pdf = OUT / "fig4_statistics_singlecolumn.pdf"
        png = OUT / "fig4_statistics_singlecolumn.png"
        fig.savefig(pdf)
        fig.savefig(png, dpi=500)
        plt.close(fig)
    return pdf, png


def main() -> None:
    four_panel = make_four_panel()
    statistics = make_vertical_statistics()
    audit = {
        "source_display_cache": str(DISPLAY_CACHE.resolve()),
        "source_representative_summary": str(REPRESENTATIVE_SUMMARY.resolve()),
        "source_statistics": str(STATISTICS_JSON.resolve()),
        "source_seed_metrics": str(SEED_METRICS_CSV.resolve()),
        "cutoff": CUTOFF,
        "contour_levels": list(CONTOUR_LEVELS),
        "gain_definition": "closure-phase SNR gain = square root of the reciprocal nuisance-marginalized variance ratio",
        "asymptotic_series_in_statistics_figure": False,
        "outputs": [str(path.resolve()) for path in (*four_panel, *statistics)],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for path in (*four_panel, *statistics):
        print(path)


if __name__ == "__main__":
    main()
