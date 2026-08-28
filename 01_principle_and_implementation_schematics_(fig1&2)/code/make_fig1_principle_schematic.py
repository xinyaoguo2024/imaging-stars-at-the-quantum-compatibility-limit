#!/usr/bin/env python3
"""One-column Section-II schematic with publication-scale typography.

The figure is intentionally conceptual.  The upper row uses the VCZ theorem
to link one selected physical baseline to one complex-visibility sample in
the Fourier plane, which is Fourier paired with the source image.  The lower
row makes the order of coherent
processing and measurement-induced collapse explicit.
"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as MplPath


OUTDIR = Path(__file__).resolve().parents[1] / "generated_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "STIXGeneral",
        # The PDF is intended to be reduced to one APS column. A nominal
        # 17--19 pt source size becomes approximately 8--9 pt after scaling.
        "font.size": 17.5,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
    }
)


INK = "#161b22"
MUTED = "#697386"
LIGHT = "#d9dee7"
PALE = "#f4f6f9"
BLUE = "#1267c4"
BLUE_DARK = "#084d98"
BLUE_FILL = "#e8f1fb"
CYAN = "#2aa5b8"
RED = "#cf3b3b"
RED_FILL = "#faecec"
GOLD = "#e59b19"
GOLD_FILL = "#fff3cf"
PURPLE = "#6b4aa5"

# Shared lower-panel geometry.  Panels (c) and (d) intentionally use exactly
# the same time axis, channel positions, receiver centers, and vertical-arrow
# endpoints so that their only visual difference is separable versus joint
# readout.
READOUT_XS = [0.22, 0.50, 0.78]
READOUT_TIMELINE_Y = 0.765
READOUT_DETECTOR_Y = 0.505
READOUT_DETECTOR_SCALE = 1.26
READOUT_DETECTOR_HALF_HEIGHT = 0.5 * 0.105 * READOUT_DETECTOR_SCALE
# ``rounded_box`` uses ``pad=0.006``; include it so arrows terminate at the
# visible outer edge rather than entering the receiver housing.
READOUT_DETECTOR_BOX_PAD = 0.006
READOUT_DETECTOR_TOP = READOUT_DETECTOR_Y + READOUT_DETECTOR_HALF_HEIGHT + READOUT_DETECTOR_BOX_PAD
READOUT_DETECTOR_BOTTOM = READOUT_DETECTOR_Y - READOUT_DETECTOR_HALF_HEIGHT - READOUT_DETECTOR_BOX_PAD
READOUT_OUTPUT_TOP = 0.335


def arrow(ax, p0, p1, *, color=INK, lw=1.3, ms=9, style="-|>", z=4, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def rounded_box(ax, x, y, w, h, *, fc="white", ec=INK, lw=1.25, radius=0.025, z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel_label(ax, letter, title):
    ax.text(
        0.015,
        0.965,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18.5,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.125,
        0.965,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18.5,
        weight="bold",
        color=INK,
    )


def draw_star(ax, x, y, r=0.055):
    # Vector-only luminous disk.
    for frac, alpha in [(1.9, 0.08), (1.55, 0.12), (1.22, 0.22), (1.0, 1.0)]:
        ax.add_patch(Circle((x, y), r * frac, fc=GOLD, ec="none", alpha=alpha, zorder=2))
    ax.add_patch(Circle((x, y), r, fc=GOLD_FILL, ec=GOLD, lw=1.3, zorder=3))
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = []
    for n, a in enumerate(angles):
        rr = r * (1.45 if n % 2 == 0 else 0.76)
        pts.append((x + rr * np.cos(a), y + rr * np.sin(a)))
    ax.add_patch(Polygon(pts, closed=True, fc=GOLD, ec=GOLD, lw=0.6, zorder=4))
    ax.add_patch(Circle((x, y), r * 0.54, fc="#fff8d9", ec="none", zorder=5))


def _rotate_points(points, center, angle_deg):
    """Rotate a small local drawing without changing the data transform."""
    p = np.asarray(points, dtype=float)
    c = np.asarray(center, dtype=float)
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (p - c) @ rot.T + c


def draw_telescope(ax, x, y, *, scale=1.0, selected=False, label=None):
    """Side-view optical telescope inspired by standard network schematics.

    ``(x,y)`` is the location of the receiver pedestal in the array plane.  A
    filled parabolic dish, focal feed, support mast, and local receiver box are
    drawn as one compact station icon.  All dishes point toward the common
    far-field source at the upper right.
    """
    dish = BLUE if selected else "#4777bf"
    edge = BLUE_DARK if selected else "#315f9e"
    lw = 1.35 if selected else 1.0
    pivot = np.array([x, y + 0.066 * scale])
    angle = -19.0
    width = 0.092 * scale
    depth = 0.038 * scale

    local = np.array(
        [
            pivot + (-width / 2, 0),
            pivot + (0, -depth),
            pivot + (width / 2, 0),
            pivot + (-width / 2, 0),
        ]
    )
    bowl = _rotate_points(local, pivot, angle)
    path = MplPath(
        bowl,
        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.CLOSEPOLY],
    )
    ax.add_patch(PathPatch(path, fc=dish, ec=edge, lw=lw, zorder=6))
    ax.plot(bowl[[0, 2], 0], bowl[[0, 2], 1], color="#244d87", lw=0.85, zorder=7)

    feed = _rotate_points([pivot + (0, 0.058 * scale)], pivot, angle)[0]
    for rim in (bowl[0], bowl[2]):
        ax.plot([rim[0], feed[0]], [rim[1], feed[1]], color=edge, lw=0.55, alpha=0.85, zorder=7)
    ax.add_patch(Circle(feed, 0.0065 * scale, fc=GOLD_FILL, ec=GOLD, lw=0.75, zorder=8))

    # Mechanical support and the compact local receiver node.
    bowl_low = _rotate_points([pivot + (0, -0.020 * scale)], pivot, angle)[0]
    ax.plot([bowl_low[0], x], [bowl_low[1], y + 0.012 * scale], color=edge, lw=1.25, zorder=5)
    rounded_box(
        ax,
        x - 0.028 * scale,
        y - 0.022 * scale,
        0.056 * scale,
        0.038 * scale,
        fc="#303640",
        ec=edge,
        lw=0.9,
        radius=0.006,
        z=5,
    )
    ax.add_patch(Circle((x, y - 0.003 * scale), 0.006 * scale, fc="#e95370", ec="#ffb2c0", lw=0.45, zorder=7))
    ax.plot([x - 0.033 * scale, x + 0.033 * scale], [y - 0.025 * scale, y - 0.025 * scale], color=MUTED, lw=0.75, zorder=4)
    if label:
        ax.text(x - 0.046 * scale, y + 0.116 * scale, label, ha="center", va="center", fontsize=17.0, color=edge, weight="bold" if selected else None, zorder=9)


def source_morphology(ax, cx, cy, scale=1.0):
    # An asymmetric but compact source, represented entirely by vector patches.
    ax.add_patch(Ellipse((cx, cy), 0.20 * scale, 0.135 * scale, angle=-18, fc="#5d3b96", ec="none", alpha=0.22, zorder=1))
    ax.add_patch(Ellipse((cx + 0.010 * scale, cy - 0.003 * scale), 0.145 * scale, 0.085 * scale, angle=-18, fc="#d94e72", ec="none", alpha=0.55, zorder=2))
    ax.add_patch(Ellipse((cx - 0.015 * scale, cy + 0.005 * scale), 0.064 * scale, 0.050 * scale, angle=-18, fc=GOLD, ec="none", alpha=0.95, zorder=3))
    ax.add_patch(Circle((cx - 0.018 * scale, cy + 0.006 * scale), 0.014 * scale, fc="#fff6b0", ec="none", zorder=4))


def draw_top_array(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", r"$N$-station array")

    # The array is deliberately two-dimensional: the pale ellipse is the
    # projected ground plane and the telescope locations are irregular rather
    # than a linear chain.
    ax.add_patch(Ellipse((0.49, 0.335), 0.86, 0.43, angle=-2, fc="#f6f8fb", ec=LIGHT, lw=0.9, zorder=0))
    sites = {
        "1": (0.15, 0.36),
        r"$i$": (0.31, 0.19),
        "2": (0.37, 0.45),
        r"$k$": (0.57, 0.25),
        r"$j$": (0.72, 0.43),
        r"$N$": (0.84, 0.22),
    }
    network_edges = [
        ("1", r"$i$"), ("1", "2"), (r"$i$", "2"), (r"$i$", r"$k$"),
        ("2", r"$k$"), ("2", r"$j$"), (r"$k$", r"$j$"),
        (r"$k$", r"$N$"), (r"$j$", r"$N$"),
    ]
    for a, b in network_edges:
        xa, ya = sites[a]
        xb, yb = sites[b]
        ax.plot([xa, xb], [ya, yb], color="#8eb7df", lw=0.75, ls=(0, (3, 2.4)), alpha=0.68, zorder=1)

    # A distant source defines one propagation direction.  The short ray
    # segments incident on every dish are parallel, while the two orange
    # strokes are wavefronts perpendicular to that direction.
    draw_star(ax, 0.88, 0.84, r=0.036)
    ax.text(0.88, 0.915, "source", ha="center", va="center", fontsize=17.0, color=MUTED)
    ray_vec = np.array([-0.105, -0.145])
    for x, y in sites.values():
        end = np.array([x + 0.015, y + 0.105])
        start = end - ray_vec
        arrow(ax, tuple(start), tuple(end), color=GOLD, lw=0.72, ms=5.4, z=3)

    # Main source-to-array direction and its associated plane wavefronts.
    arrow(ax, (0.845, 0.795), (0.705, 0.605), color=GOLD, lw=1.25, ms=7.0, z=3)
    wf_centers = [np.array([0.79, 0.72]), np.array([0.70, 0.60])]
    tangent = np.array([0.155, -0.112])
    for center in wf_centers:
        p0, p1 = center - tangent / 2, center + tangent / 2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GOLD, lw=1.1, zorder=2)
    ax.text(0.625, 0.770, "wavefront", ha="center", va="center", fontsize=17.0, color=GOLD)

    for label, (x, y) in sites.items():
        sel = label in (r"$i$", r"$j$")
        draw_telescope(ax, x, y, scale=1.22, selected=sel, label=label)

    # Highlight the directed baseline vector b_ij = r_j - r_i.  Its arrow
    # therefore starts at station i and terminates at station j.
    p_i = np.array(sites[r"$i$"])
    p_j = np.array(sites[r"$j$"])
    d = p_j - p_i
    d /= np.linalg.norm(d)
    p0 = p_i + 0.045 * d
    p1 = p_j - 0.045 * d
    arrow(ax, tuple(p0), tuple(p1), color=RED, lw=1.85, ms=11, style="-|>", z=8)
    angle = np.degrees(np.arctan2(d[1], d[0]))
    normal = np.array([-d[1], d[0]])
    b_pos = p0 + 0.30 * (p1 - p0) + 0.055 * normal
    ax.text(b_pos[0], b_pos[1], r"$\mathbf{b}_{ij}$", ha="center", va="center", fontsize=17.5, color=RED, weight="bold", rotation=angle, rotation_mode="anchor", zorder=9)


def draw_uv_track(ax, center=(0.31, 0.50), scale=1.0):
    cx, cy = center
    ax.plot([cx - 0.245 * scale, cx + 0.245 * scale], [cy, cy], color=MUTED, lw=0.9)
    ax.plot([cx, cx], [cy - 0.235 * scale, cy + 0.235 * scale], color=MUTED, lw=0.9)
    ax.text(cx + 0.212 * scale, cy - 0.054, r"$u$", ha="center", va="top", fontsize=17.0, color=MUTED)
    ax.text(cx - 0.025, cy + 0.245 * scale, r"$v$", ha="right", va="center", fontsize=17.0, color=MUTED)

    # Several conjugate Earth-rotation tracks.
    angles = np.linspace(-1.15, 1.15, 13)
    tracks = [(0.18, 0.075, 0.30), (0.25, 0.10, -0.45), (0.32, 0.13, 0.75)]
    all_pts = []
    for a, b, rot in tracks:
        c, s = np.cos(rot), np.sin(rot)
        xs = a * np.cos(angles)
        ys = b * np.sin(angles)
        xr = c * xs - s * ys
        yr = s * xs + c * ys
        for sign in (1, -1):
            ax.scatter(cx + sign * xr, cy + sign * yr, s=10, fc=CYAN, ec="white", lw=0.25, alpha=0.80, zorder=3)
            all_pts.extend(zip(cx + sign * xr, cy + sign * yr))
    return all_pts


def draw_top_vcz(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b", "Fourier sampling")

    # Keep the panel title fixed while lowering the complete VCZ construction
    # to give the heading and coordinate annotation more breathing room.
    pts = draw_uv_track(ax, center=(0.28, 0.425), scale=0.95)
    # Select an upper-right high-frequency point.  Its labels are placed in
    # the open band above the UV tracks rather than on top of sampled points.
    sel = max(pts, key=lambda p: 0.8 * p[0] + p[1])
    ax.add_patch(Circle(sel, 0.018, fc=RED_FILL, ec=RED, lw=1.6, zorder=6))
    ax.text(
        sel[0],
        0.770,
        r"$(u_{ij},v_{ij})=\mathbf{b}_{ij}/\lambda$",
        ha="center",
        va="center",
        fontsize=16.5,
        color=RED,
        weight="bold",
        zorder=7,
    )
    # The sampled Fourier plane and source brightness form a Fourier pair.
    arrow(ax, (0.52, 0.425), (0.66, 0.425), color=INK, lw=1.35, ms=9, style="<->")
    ax.text(0.59, 0.480, "FT", ha="center", va="bottom", fontsize=18.0, color=INK, weight="bold")

    rounded_box(ax, 0.68, 0.170, 0.285, 0.505, fc=PALE, ec=LIGHT, lw=1.0, radius=0.018, z=0)
    source_morphology(ax, 0.82, 0.435, scale=1.15)
    ax.text(0.82, 0.630, r"$I(\theta)$", ha="center", va="center", fontsize=18.0, color=INK, weight="bold")


def collapse_marker(ax, x, y, r=0.021):
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = []
    for n, a in enumerate(angles):
        rr = r * (1.0 if n % 2 == 0 else 0.48)
        pts.append((x + rr * np.cos(a), y + rr * np.sin(a)))
    ax.add_patch(Polygon(pts, closed=True, fc=RED_FILL, ec=RED, lw=1.0, zorder=5))


def photon_pulse(ax, x, y, *, scale=1.0, color=GOLD):
    """Draw a localized single-photon wave packet moving to the right."""
    xx = np.linspace(x - 0.055 * scale, x - 0.012 * scale, 32)
    yy = y + 0.009 * scale * np.sin(np.linspace(0, 4 * np.pi, xx.size))
    ax.plot(xx, yy, color=color, lw=1.15, zorder=5)
    ax.add_patch(Circle((x, y), 0.014 * scale, fc="#fff4a8", ec=color, lw=1.15, zorder=6))
    arrow(ax, (x + 0.014 * scale, y), (x + 0.045 * scale, y), color=color, lw=0.95, ms=6, z=5)


def physical_detector(ax, x, y, *, clicked=True, scale=1.0):
    """Compact photon-counter icon with a colored photosensitive window."""
    w, h = 0.090 * scale, 0.105 * scale
    rounded_box(ax, x - w / 2, y - h / 2, w, h, fc="#303640", ec=INK, lw=1.0, radius=0.012, z=3)
    ax.add_patch(Circle((x, y + 0.012 * scale), 0.021 * scale, fc="#5b82d0", ec="#ef5573", lw=0.85, zorder=4))
    ax.plot([x - 0.018 * scale, x + 0.018 * scale], [y - 0.026 * scale] * 2, color=LIGHT, lw=1.0, zorder=4)
    if clicked:
        collapse_marker(ax, x + 0.050 * scale, y + 0.040 * scale, r=0.017 * scale)


def memory_cell(ax, x, y, *, scale=1.0):
    """A pictorial optical/atomic memory cell, not a circuit symbol."""
    ax.add_patch(Ellipse((x, y), 0.095 * scale, 0.115 * scale, fc=BLUE_FILL, ec=BLUE, lw=1.15, zorder=3))
    ax.add_patch(Arc((x, y), 0.070 * scale, 0.084 * scale, theta1=35, theta2=325, ec=CYAN, lw=1.15, zorder=4))
    ax.add_patch(Circle((x, y), 0.014 * scale, fc=GOLD_FILL, ec=GOLD, lw=0.9, zorder=5))


def draw_multiport(ax, x, y, w, h):
    """Three-port coherent mixer with explicit crossing optical paths."""
    rounded_box(ax, x, y, w, h, fc=BLUE_FILL, ec=BLUE, lw=1.35, radius=0.018, z=2)
    levels = [y + 0.73 * h, y + 0.50 * h, y + 0.27 * h]
    for yy in levels:
        ax.plot([x + 0.03 * w, x + 0.22 * w], [yy, yy], color=BLUE, lw=1.05, zorder=4)
        ax.plot([x + 0.78 * w, x + 0.97 * w], [yy, yy], color=BLUE, lw=1.05, zorder=4)
    # Coupled alternatives inside the physical multiport.
    ax.plot([x + 0.22 * w, x + 0.50 * w, x + 0.78 * w], [levels[0], levels[2], levels[1]], color=CYAN, lw=1.35, zorder=4)
    ax.plot([x + 0.22 * w, x + 0.50 * w, x + 0.78 * w], [levels[1], levels[0], levels[2]], color=PURPLE, lw=1.35, zorder=4)
    ax.plot([x + 0.22 * w, x + 0.50 * w, x + 0.78 * w], [levels[2], levels[1], levels[0]], color=BLUE, lw=1.35, zorder=4)
    return levels


def draw_vertical_multiport(ax, x, y, w, h):
    """Three-port coherent mixer arranged from top inputs to bottom outputs."""
    rounded_box(ax, x, y, w, h, fc=BLUE_FILL, ec=BLUE, lw=1.35, radius=0.018, z=2)
    ports = [x + 0.22 * w, x + 0.50 * w, x + 0.78 * w]
    for xx in ports:
        ax.plot([xx, xx], [y + 0.80 * h, y + 0.98 * h], color=BLUE, lw=1.05, zorder=4)
        ax.plot([xx, xx], [y + 0.02 * h, y + 0.20 * h], color=BLUE, lw=1.05, zorder=4)
    # Three indistinguishable routing alternatives cross inside the multiport.
    top = y + 0.80 * h
    bottom = y + 0.20 * h
    middle = 0.5 * (top + bottom)
    ax.plot([ports[0], ports[1], ports[2]], [top, middle, bottom], color=CYAN, lw=1.35, zorder=4)
    ax.plot([ports[1], ports[2], ports[0]], [top, middle, bottom], color=PURPLE, lw=1.35, zorder=4)
    ax.plot([ports[2], ports[0], ports[1]], [top, middle, bottom], color=BLUE, lw=1.35, zorder=4)
    return ports


def draw_single_copy(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "c", "Single-copy")

    # A pulse train is sampled immediately, so every detection retains its
    # arrival-time label before any information can be shared across copies.
    arrow(ax, (0.075, READOUT_TIMELINE_Y), (0.91, READOUT_TIMELINE_Y), color=INK, lw=1.05, ms=7)
    xs = READOUT_XS
    tags = [r"$t_1$", r"$t_2$", r"$t_{n_s}$"]
    channel_colors = [CYAN, PURPLE, BLUE]
    for x, tag, color in zip(xs, tags, channel_colors):
        photon_pulse(ax, x, READOUT_TIMELINE_Y, scale=0.90, color=color)
        ax.text(x, 0.825, tag, ha="center", va="center", fontsize=17.0, color=color)
        arrow(ax, (x, READOUT_TIMELINE_Y - 0.030), (x, READOUT_DETECTOR_TOP), color=color, lw=1.0, ms=7, z=2)
        physical_detector(ax, x, READOUT_DETECTOR_Y, clicked=True, scale=READOUT_DETECTOR_SCALE)
        arrow(ax, (x, READOUT_DETECTOR_BOTTOM), (x, READOUT_OUTPUT_TOP), color=color, lw=0.95, ms=6, z=2)
        rounded_box(ax, x - 0.075, 0.255, 0.150, 0.080, fc=PALE, ec=MUTED, lw=0.85, radius=0.012)
        ax.text(x, 0.295, tag.replace("t", "x"), ha="center", va="center", fontsize=17.0, color=INK)
    ax.text(0.64, 0.825, r"$\cdots$", ha="center", va="center", fontsize=17.0, color=MUTED)


def draw_collective(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "d", "Collective")

    # Match panel (c) geometrically.  The distinction is that the receiver
    # modules are coherently linked before the common classical record is
    # formed, rather than producing three separately labelled outcomes.
    arrow(ax, (0.075, READOUT_TIMELINE_Y), (0.91, READOUT_TIMELINE_Y), color=INK, lw=1.05, ms=7)
    xs = READOUT_XS
    tags = [r"$t_1$", r"$t_2$", r"$t_{n_s}$"]
    channel_colors = [CYAN, PURPLE, BLUE]
    detector_y = READOUT_DETECTOR_Y
    for x, tag, color in zip(xs, tags, channel_colors):
        photon_pulse(ax, x, READOUT_TIMELINE_Y, scale=0.90, color=color)
        ax.text(x, 0.825, tag, ha="center", va="center", fontsize=17.0, color=color)
        arrow(ax, (x, READOUT_TIMELINE_Y - 0.030), (x, READOUT_DETECTOR_TOP), color=color, lw=1.0, ms=7, z=2)
        physical_detector(ax, x, detector_y, clicked=False, scale=READOUT_DETECTOR_SCALE)
    ax.text(0.64, 0.825, r"$\cdots$", ha="center", va="center", fontsize=17.0, color=MUTED)

    # A horizontally stretched lemniscate (infinity sign) between adjacent
    # receivers evokes a pi-like coherent bond.  The housings remain distinct,
    # while the closed crossing path marks a nonseparable joint receiver.
    bond_color = "#7b61b5"
    for xa, xb in zip(xs[:-1], xs[1:]):
        left_edge = xa + 0.061
        right_edge = xb - 0.061
        mid = 0.5 * (left_edge + right_edge)
        half_width = 0.5 * (right_edge - left_edge)
        tt = np.linspace(0, 2 * np.pi, 240)
        xx = mid + half_width * np.sin(tt)
        yy = detector_y + 0.027 * np.sin(2 * tt)
        ax.plot(xx, yy, color=bond_color, lw=1.35, alpha=0.98, zorder=6)

    # Every receiver contributes to the same joint outcome; these three links
    # therefore share one colour rather than retaining their temporal colours.
    joint_color = BLUE_DARK
    joint_y, joint_h = 0.245, 0.090
    for x in xs:
        arrow(ax, (x, READOUT_DETECTOR_BOTTOM), (x, READOUT_OUTPUT_TOP), color=joint_color, lw=1.0, ms=6, z=2)
    rounded_box(ax, 0.200, joint_y, 0.600, joint_h, fc=PALE, ec=joint_color, lw=0.95, radius=0.013, z=3)
    ax.text(0.50, joint_y + 0.50 * joint_h, r"$x(1,2,\ldots,n_s)$", ha="center", va="center", fontsize=16.5, color=INK, zorder=5)


def main():
    fig = plt.figure(figsize=(7.15, 6.05), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.025,
        right=0.985,
        bottom=0.035,
        top=0.985,
        wspace=0.075,
        hspace=0.10,
        height_ratios=[1.02, 0.98],
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    draw_top_array(axes[0])
    draw_top_vcz(axes[1])
    draw_single_copy(axes[2])
    draw_collective(axes[3])

    # Quiet separators preserve the two-row reading order without heavy boxes.
    fig.add_artist(plt.Line2D([0.035, 0.975], [0.505, 0.505], transform=fig.transFigure, color=LIGHT, lw=0.8))
    # Leave a deliberate opening in the upper divider for the physical-array
    # to Fourier-plane mapping supplied by the VCZ theorem.
    fig.add_artist(plt.Line2D([0.505, 0.505], [0.045, 0.705], transform=fig.transFigure, color=LIGHT, lw=0.7))
    fig.add_artist(plt.Line2D([0.505, 0.505], [0.825, 0.965], transform=fig.transFigure, color=LIGHT, lw=0.7))
    fig.add_artist(
        FancyArrowPatch(
            (0.470, 0.755),
            (0.540, 0.755),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.2,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            zorder=20,
        )
    )
    fig.text(0.505, 0.782, "VCZ", ha="center", va="bottom", fontsize=17.5, color=INK, weight="bold")

    stem = OUTDIR / "fig1_principle_schematic"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
