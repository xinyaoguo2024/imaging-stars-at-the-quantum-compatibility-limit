#!/usr/bin/env python3
"""Rebuild Fig. 2 with direct sequential encoding into n_s registers.

The figure keeps the original landscape layout, acquisition-time axis, joint
processing block, and red dashed effective-POVM boundary.  Successive temporal
modes are routed directly into the n_s parallel registers; no intermediate
continuous-memory or partition stage is shown.
"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


OUTDIR = Path(__file__).resolve().parents[1] / "generated_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "font.size": 10.5,
        "font.weight": "normal",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
    }
)


INK = "#15191f"
GRAY = "#697386"
LIGHT_GRAY = "#d7dce3"
BLUE = "#0b61c5"
BLUE_FILL = "#e9f1fb"
RED = "#cf3030"
BOX_PAD = 0.004
DASHED_BOX_PAD = 0.006


def rounded_box(ax, x, y, w, h, *, fc="white", ec=INK, lw=1.4, radius=0.008, z=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def arrow(ax, p0, p1, *, color=INK, lw=1.25, ms=9, z=3):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def line(ax, p0, p1, *, color=INK, lw=1.1, z=2):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=z)


def mid_arrow(ax, p0, p1, *, color=INK, lw=1.2, ms=8, at=0.52, z=1):
    """Draw a connector whose arrowhead is geometrically centered on it.

    A conventional ``FancyArrowPatch`` places the *tip* at the requested
    coordinate, which makes the triangular head look displaced to the left of
    the midpoint.  Here the connector is drawn as a line and a centered marker,
    so the visible triangle itself is centered at fraction ``at``.
    """
    line(ax, p0, p1, color=color, lw=lw, z=z)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    mx = p0[0] + at * dx
    my = p0[1] + at * dy
    if abs(dx) >= abs(dy):
        marker = ">" if dx >= 0 else "<"
    else:
        marker = "^" if dy >= 0 else "v"
    ax.plot(
        [mx],
        [my],
        marker=marker,
        markersize=ms,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0,
        linestyle="none",
        zorder=z + 2,
    )


def dashed_round_box(ax, x, y, w, h):
    # FancyBboxPatch dash corners render more cleanly than a hand-built path.
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={DASHED_BOX_PAD},rounding_size=0.014",
        facecolor="none",
        edgecolor=RED,
        linewidth=1.65,
        linestyle=(0, (5.2, 3.2)),
        zorder=0,
    )
    ax.add_patch(p)


def bracket(ax, x, y0, y1, *, color=GRAY, lw=1.2):
    verts = [(x - 0.010, y1), (x, y1), (x, y0), (x - 0.010, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
    ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, ec=color, lw=lw, zorder=2))


def main():
    # Preserve the original landscape aspect ratio while using a compact
    # source canvas.  At 0.98 APS-column width, 17--18 pt source text renders
    # at approximately 8 pt in the manuscript.
    fig, ax = plt.subplots(figsize=(7.15, 4.01))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.050, top=0.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header.
    ax.text(
        0.040,
        0.958,
        "quantum mode / superposition",
        ha="left",
        va="center",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    # Dashed effective-measurement envelope, as in the original figure.
    # The visible right edge of the dashed boundary is constrained below to
    # bisect the final theta-output arrow exactly.
    red_x = 0.070
    red_right_visible = 0.925
    red_w = red_right_visible - red_x - DASHED_BOX_PAD
    dashed_round_box(ax, red_x, 0.245, red_w, 0.575)
    ax.text(
        0.480,
        0.830,
        "effective joint POVM",
        ha="center",
        va="center",
        color=RED,
        fontsize=17.5,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
        zorder=5,
    )

    # Laboratory time axis.  Only the acquisition interval is marked.
    arrow(ax, (0.035, 0.095), (0.960, 0.095), lw=1.25, ms=9)
    arrow(ax, (0.035, 0.095), (0.035, 0.930), lw=1.25, ms=9)
    ax.text(0.964, 0.050, "time", ha="right", va="center", fontsize=17, weight="bold")

    # One box represents the complete stream of n temporal copies.
    src_x, src_y, src_w, src_h = 0.075, 0.130, 0.285, 0.083
    rounded_box(ax, src_x, src_y, src_w, src_h, fc="white", ec=INK, lw=1.35)
    ax.text(
        src_x + src_w / 2,
        src_y + src_h / 2,
        r"$\rho^{\otimes n}$",
        ha="center",
        va="center",
        fontsize=18,
    )

    # Each temporal mode is written instantaneously (vertical connector) and
    # then retained until the common readout time (horizontal dashed segment).
    source_top = src_y + src_h + BOX_PAD
    source_points = [0.120, 0.220, 0.320]
    # Leave enough open connector length before the processing block for the
    # arrowhead to sit visibly at the midpoint rather than against either box.
    mem_x, mem_w, mem_h = 0.390, 0.085, 0.074
    write_w, write_h = mem_w, mem_h
    # Align the time-axis ticks with the vertical borders of the sequential
    # write registers.  The intermediate spacing is indicated only by an
    # ellipsis; an explicit Delta-t label is unnecessary in this schematic.
    register_left_edges = [x0 - write_w / 2 for x0 in source_points]
    register_right_edge = source_points[-1] + write_w / 2
    for x in (*register_left_edges, register_right_edge):
        line(ax, (x, 0.083), (x, 0.108), lw=1.0)
    ax.text(register_left_edges[0], 0.055, r"$0$", ha="center", va="center", fontsize=16.5)
    ax.text(
        0.5 * (register_left_edges[1] + register_left_edges[2]),
        0.055,
        r"$\cdots$",
        ha="center",
        va="center",
        fontsize=16.5,
    )
    ax.text(register_right_edge, 0.055, r"$T$", ha="center", va="center", fontsize=16.5)
    rail_y = [0.640, 0.525, 0.390]
    write_labels = [r"$a_1$", r"$a_2$", r"$a_{n_s}$"]
    stored_labels = [r"$a_{1}^{\prime}$", r"$a_{2}^{\prime}$", r"$a_{n_s}^{\prime}$"]
    ax.text(
        0.335,
        0.745,
        r"$n_s$ registers",
        ha="center",
        va="center",
        fontsize=17.0,
        weight="semibold",
        color=INK,
    )
    for y, a_label in zip(rail_y, stored_labels):
        rounded_box(ax, mem_x, y - mem_h / 2, mem_w, mem_h, fc="white", ec=BLUE, lw=1.55)
        ax.text(mem_x + mem_w / 2, y, a_label, ha="center", va="center", fontsize=17.0, color=BLUE)
    for x0, y, a_label in zip(source_points, rail_y, write_labels):
        write_left = x0 - write_w / 2
        write_right = x0 + write_w / 2
        rounded_box(
            ax,
            write_left,
            y - write_h / 2,
            write_w,
            write_h,
            fc="white",
            ec=BLUE,
            lw=1.55,
        )
        ax.text(x0, y, a_label, ha="center", va="center", fontsize=17.0, color=BLUE)
        mid_arrow(
            ax,
            (x0, source_top),
            (x0, y - write_h / 2 - BOX_PAD),
            color=BLUE,
            lw=1.15,
            ms=6.5,
            at=0.50,
            z=1,
        )
        ax.plot(
            [write_right + BOX_PAD, mem_x - BOX_PAD],
            [y, y],
            color=BLUE,
            lw=1.15,
            linestyle=(0, (4.0, 3.0)),
            zorder=1,
        )
    ellipsis_y = 0.5 * (rail_y[1] + rail_y[2])
    ax.text(mem_x + mem_w / 2, ellipsis_y, r"$\vdots$", ha="center", va="center", fontsize=19, color=GRAY)

    # Stage 2: joint processing across the n_s memory groups.
    # Keep the processing block only slightly taller than the outer register
    # boxes, with exactly the same protrusion above and below them.
    proc_x, proc_w = 0.520, 0.140
    proc_margin = 0.035
    proc_y = rail_y[-1] - mem_h / 2 - proc_margin
    proc_top = rail_y[0] + mem_h / 2 + proc_margin
    proc_h = proc_top - proc_y
    rounded_box(ax, proc_x, proc_y, proc_w, proc_h, fc=BLUE_FILL, ec=BLUE, lw=1.65, radius=0.010)
    ax.text(
        proc_x + proc_w / 2,
        proc_y + proc_h / 2,
        "joint\nquantum\nprocessing",
        ha="center",
        va="center",
        fontsize=15.0,
        weight="semibold",
        color=BLUE,
        linespacing=1.05,
    )
    for y in rail_y:
        mid_arrow(
            ax,
            (mem_x + mem_w + BOX_PAD, y),
            (proc_x - BOX_PAD, y),
            color=BLUE,
            lw=1.25,
            ms=6.5,
            at=0.50,
            z=1,
        )

    # Stage 3: identical single-copy POVMs on the n_s output rails.
    # A slightly narrower POVM box leaves equal visual breathing room for the
    # centered connector arrow without moving the outcome labels.
    read_x, read_w, read_h = 0.705, 0.085, 0.066
    read_center = read_x + read_w / 2
    ax.text(0.735, 0.745, r"$n_s$ single-copy POVMs", ha="center", va="center", fontsize=16.0, weight="semibold", color=INK)
    outcome_labels = [r"$\mathbf{x}_1$", r"$\mathbf{x}_2$", r"$\mathbf{x}_{n_s}$"]
    for y, outcome in zip(rail_y, outcome_labels):
        mid_arrow(
            ax,
            (proc_x + proc_w + BOX_PAD, y),
            (read_x - BOX_PAD, y),
            color=INK,
            lw=1.15,
            ms=6.5,
            at=0.50,
            z=1,
        )
        rounded_box(ax, read_x, y - read_h / 2, read_w, read_h, fc="white", ec=INK, lw=1.45)
        ax.text(read_center, y, r"$\Pi$", ha="center", va="center", fontsize=18)
        arrow(ax, (read_x + read_w + BOX_PAD, y), (0.835, y), color=INK, lw=1.15, ms=7, z=1)
        ax.text(0.845, y, outcome, ha="left", va="center", fontsize=16.5)
    ax.text(read_center, ellipsis_y, r"$\vdots$", ha="center", va="center", fontsize=19, color=GRAY)

    # Classical combination of the complete outcome records.
    bracket_x = 0.905
    theta_arrow_end = 2.0 * red_right_visible - bracket_x
    bracket(ax, bracket_x, rail_y[-1], rail_y[0], color=GRAY, lw=1.25)
    theta_y = 0.5 * (rail_y[0] + rail_y[-1])
    arrow(ax, (bracket_x, theta_y), (theta_arrow_end, theta_y), color=GRAY, lw=1.25, ms=8)
    ax.text(theta_arrow_end + 0.007, theta_y, r"$\widehat{\theta}$", ha="left", va="center", fontsize=18)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUTDIR / f"fig2_implementation_schematic.{ext}", dpi=320, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
