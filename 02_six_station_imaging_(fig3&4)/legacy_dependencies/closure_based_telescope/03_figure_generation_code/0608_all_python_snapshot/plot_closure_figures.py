from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


PALETTE = {
    "science": "#1f77b4",
    "control": "#d95f02",
    "aux": "#7570b3",
    "plant": "#2ca02c",
    "fill_science": "#eef5fc",
    "fill_control": "#fff2e9",
    "fill_aux": "#f2effb",
    "fill_plant": "#eef8ee",
}


def add_box(ax, x, y, w, h, text, *, fc="white", ec="0.25", lw=1.3, fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(
    ax,
    p1,
    p2,
    *,
    text=None,
    color="0.25",
    lw=1.4,
    rad=0.0,
    ls="-",
    tdx=0.0,
    tdy=0.0,
    fontsize=9,
):
    patch = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="->",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=ls,
    )
    ax.add_patch(patch)
    if text:
        mx = (p1[0] + p2[0]) / 2 + tdx
        my = (p1[1] + p2[1]) / 2 + tdy
        ax.text(mx, my, text, color=color, ha="center", va="center", fontsize=fontsize)


def style_axis(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)


def build_adaptive_figure():
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.2), constrained_layout=True)

    # Three-station panel
    ax = axes[0]
    style_axis(ax, "Three-station adaptive closure servo")
    add_box(
        ax,
        0.6,
        5.55,
        1.95,
        1.15,
        "Input field\nthree stations",
        fc=PALETTE["fill_plant"],
        ec=PALETTE["plant"],
        fontsize=11,
    )
    add_box(
        ax,
        3.3,
        5.45,
        2.7,
        1.35,
        "Closure receiver\n$L_{\\mathrm{cl}}(\\bar\\phi)$",
        fc=PALETTE["fill_science"],
        ec=PALETTE["science"],
        fontsize=11,
    )
    add_box(
        ax,
        6.6,
        5.55,
        2.2,
        1.15,
        "Science output\n$\\hat\\Phi_{\\mathrm{cl}}$",
        fc=PALETTE["fill_science"],
        ec=PALETTE["science"],
        fontsize=11,
    )
    add_box(
        ax,
        0.7,
        1.55,
        2.3,
        1.55,
        "Auxiliary pairwise\nfringe tracker",
        fc=PALETTE["fill_aux"],
        ec=PALETTE["aux"],
        fontsize=10.5,
    )
    add_box(
        ax,
        3.1,
        1.55,
        2.55,
        1.55,
        "Fast gauge controller\ntrack station pistons",
        fc=PALETTE["fill_control"],
        ec=PALETTE["control"],
        fontsize=10.1,
    )
    add_box(
        ax,
        6.45,
        1.35,
        2.45,
        1.95,
        "Slow closure loop\ndither + lock near\nsteepest slope",
        fc=PALETTE["fill_control"],
        ec=PALETTE["control"],
        fontsize=10.2,
    )
    add_arrow(ax, (2.55, 6.12), (3.3, 6.12), color=PALETTE["plant"])
    add_arrow(ax, (6.0, 6.12), (6.6, 6.12), color=PALETTE["science"])
    add_arrow(
        ax,
        (2.1, 5.55),
        (1.75, 3.0),
        color=PALETTE["aux"],
        rad=0.04,
        text="tap-off",
        tdx=-0.15,
        tdy=0.15,
        fontsize=10,
    )
    add_arrow(
        ax,
        (2.9, 2.3),
        (3.35, 2.3),
        color=PALETTE["aux"],
    )
    add_arrow(
        ax,
        (5.65, 2.32),
        (4.8, 5.45),
        color=PALETTE["control"],
        rad=0.16,
        text="phase corrections",
        tdx=-0.55,
        tdy=0.22,
        fontsize=9.5,
    )
    add_arrow(
        ax,
        (7.7, 5.55),
        (7.7, 3.3),
        color=PALETTE["science"],
        text="science samples",
        tdx=0.78,
        tdy=0.0,
        fontsize=9.5,
    )
    add_arrow(
        ax,
        (6.65, 2.3),
        (6.02, 2.3),
        color=PALETTE["control"],
    )
    add_arrow(ax, (7.88, 3.3), (7.88, 5.55), color=PALETTE["control"], lw=1.2, ls="--")
    ax.text(
        5.0,
        7.38,
        "Fast loop suppresses cut-space motion; slow loop keeps the science channel near the local optimum.",
        ha="center",
        va="center",
        fontsize=9.2,
    )

    # N-station panel
    ax = axes[1]
    style_axis(ax, "$N$-station closure-subspace adaptive receiver")
    add_box(
        ax,
        0.55,
        5.45,
        2.1,
        1.35,
        "Complete $N$-station\narray",
        fc=PALETTE["fill_plant"],
        ec=PALETTE["plant"],
        fontsize=11,
    )
    add_box(
        ax,
        2.9,
        5.25,
        3.2,
        1.75,
        "Programmable receiver\n$L(\\alpha; q_{\\mathrm{ref}})$ or\nscheduled $L_\\mu$",
        fc=PALETTE["fill_science"],
        ec=PALETTE["science"],
        fontsize=10.5,
    )
    add_box(
        ax,
        7.25,
        5.45,
        2.05,
        1.35,
        "Protected outputs\n$\\hat q$ / scanned closures",
        fc=PALETTE["fill_science"],
        ec=PALETTE["science"],
        fontsize=10.4,
    )
    add_box(
        ax,
        0.75,
        1.45,
        2.2,
        1.65,
        "Gauge-lock layer\nestimate nuisance $\\psi$",
        fc=PALETTE["fill_aux"],
        ec=PALETTE["aux"],
        fontsize=10.1,
    )
    add_box(
        ax,
        3.05,
        1.25,
        2.95,
        2.05,
        "Re-linearization engine\nupdate $R(q_{\\mathrm{ref}})$,\n$Y_\\mu$, and $L_\\mu$",
        fc=PALETTE["fill_control"],
        ec=PALETTE["control"],
        fontsize=9.9,
    )
    add_box(
        ax,
        6.95,
        1.45,
        2.35,
        1.65,
        "Measurement scheduler\nchoose $\\alpha$ or\nscan basis directions",
        fc=PALETTE["fill_control"],
        ec=PALETTE["control"],
        fontsize=9.9,
    )
    add_arrow(ax, (2.65, 6.12), (2.9, 6.12), color=PALETTE["plant"])
    add_arrow(ax, (6.1, 6.12), (7.25, 6.12), color=PALETTE["science"])
    add_arrow(
        ax,
        (1.85, 5.45),
        (1.85, 3.05),
        color=PALETTE["aux"],
        text="auxiliary taps",
        tdx=0.75,
        tdy=0.0,
        fontsize=9.5,
    )
    add_arrow(
        ax,
        (2.9, 2.25),
        (3.15, 2.25),
        color=PALETTE["aux"],
    )
    add_arrow(
        ax,
        (8.2, 5.45),
        (8.2, 3.1),
        color=PALETTE["science"],
        text="samples",
        tdx=0.5,
        tdy=0.0,
        fontsize=9.4,
    )
    add_arrow(
        ax,
        (5.1, 3.25),
        (5.1, 5.25),
        color=PALETTE["control"],
        text="updated SLD basis",
        tdx=0.85,
        tdy=0.02,
        fontsize=9.4,
    )
    add_arrow(
        ax,
        (7.2, 2.25),
        (6.0, 2.25),
        color=PALETTE["control"],
    )
    add_arrow(
        ax,
        (6.95, 3.0),
        (6.0, 5.85),
        color=PALETTE["control"],
        rad=-0.1,
        text="scheduled POVM",
        tdx=0.18,
        tdy=0.55,
        fontsize=9.2,
    )
    ax.text(
        5.0,
        7.38,
        "Cycle-space observables are updated locally while nuisance station phases stay out of the science basis.",
        ha="center",
        va="center",
        fontsize=9.2,
    )

    fig.suptitle("Adaptive maintenance of closure-protected optimal measurements", fontsize=15, y=1.02)
    png = OUTDIR / "adaptive_closure_servo.png"
    pdf = OUTDIR / "adaptive_closure_servo.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def panel_header(ax, x, y, text):
    ax.text(x, y, text, fontsize=12, fontweight="bold", ha="left", va="top")


def draw_wire(ax, y, x0=0.8, x1=9.2, label=None, color="0.25", lw=1.5):
    ax.plot([x0, x1], [y, y], color=color, lw=lw)
    if label is not None:
        ax.text(x0 - 0.18, y, label, ha="right", va="center", fontsize=10.5)


def draw_gate(ax, x, y, text, w=0.72, h=0.5, fc="white", ec="0.25", fontsize=9.2):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def draw_bs(ax, x, y1, y2, label, color=None):
    color = color or PALETTE["science"]
    ax.plot([x - 0.28, x + 0.28], [y1, y2], color=color, lw=1.7)
    ax.plot([x - 0.28, x + 0.28], [y2, y1], color=color, lw=1.7)
    ax.text(x, (y1 + y2) / 2 + 0.42, label, color=color, ha="center", va="bottom", fontsize=9.2)


def draw_pd(ax, x, y, label="PD", color=None):
    color = color or PALETTE["control"]
    w, h = 0.56, 0.42
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=9.0, color=color)


def draw_measure(ax, x, y, color=None):
    color = color or PALETTE["control"]
    w, h = 0.52, 0.42
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x, y, "M", ha="center", va="center", fontsize=9.2, color=color)


def draw_two_wire_gate(ax, x, y1, y2, text, color=None):
    color = color or PALETTE["science"]
    ax.plot([x, x], [y2, y1], color=color, lw=1.2)
    draw_gate(ax, x, (y1 + y2) / 2, text, w=0.95, h=0.55, fc="white", ec=color, fontsize=8.8)


def build_implementation_figure():
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.8), constrained_layout=True)

    for ax in axes.ravel():
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7.6)
        ax.axis("off")

    # 3-station optical
    ax = axes[0, 0]
    panel_header(ax, 0.2, 7.35, "(a) Three-station optical implementation")
    ys = [6.0, 4.8, 3.6]
    labels = [r"$a_1$", r"$a_2$", r"$a_3$"]
    for y, label in zip(ys, labels):
        draw_wire(ax, y, label=label)
    draw_gate(ax, 1.6, ys[0], r"$PS(\phi_1)$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"])
    draw_gate(ax, 1.6, ys[1], r"$PS(\phi_2)$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"])
    draw_gate(ax, 1.6, ys[2], r"$PS(\phi_3)$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"])
    draw_bs(ax, 3.05, ys[0], ys[1], r"$BS_{12}$")
    draw_gate(ax, 4.05, ys[0], r"$PS$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"], fontsize=8.8)
    draw_gate(ax, 4.05, ys[1], r"$PS$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"], fontsize=8.8)
    draw_bs(ax, 5.1, ys[1], ys[2], r"$BS_{23}$")
    draw_gate(ax, 6.0, ys[1], r"$PS$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"], fontsize=8.8)
    draw_bs(ax, 6.9, ys[0], ys[1], r"$BS_{12}$")
    draw_pd(ax, 8.55, ys[0], r"$PD_+$")
    draw_pd(ax, 8.55, ys[1], r"$PD_0$")
    draw_pd(ax, 8.55, ys[2], r"$PD_-$")
    ax.text(4.9, 6.85, r"$U_{\mathrm{cl}}^\dagger = BS_{12}\,PS\,BS_{23}\,PS\,BS_{12}$", color=PALETTE["science"], ha="center", fontsize=10.0)
    ax.text(8.95, 6.42, r"$+$", color=PALETTE["control"], fontsize=11)
    ax.text(8.95, 5.22, r"$0$", color=PALETTE["control"], fontsize=11)
    ax.text(8.95, 4.02, r"$-$", color=PALETTE["control"], fontsize=11)
    ax.text(5.0, 1.55, r"Estimator: $\,\hat s_{\mathrm{cl}}\propto n_+ - n_-\,$ (or eigenvalue-weighted counts)", ha="center", fontsize=9.7)
    ax.text(5.0, 0.72, "Symmetric case: the mesh reduces to a tritter / DFT combiner.", ha="center", fontsize=9.4)

    # 3-station circuit
    ax = axes[0, 1]
    panel_header(ax, 0.2, 7.35, "(b) Three-station circuit view")
    ys = [6.0, 4.8, 3.6]
    labels = [r"$|100\rangle$", r"$|010\rangle$", r"$|001\rangle$"]
    for y, label in zip(ys, labels):
        draw_wire(ax, y, label=label)
    draw_gate(ax, 1.8, ys[0], r"$P_1$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"])
    draw_gate(ax, 1.8, ys[1], r"$P_2$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"])
    draw_gate(ax, 1.8, ys[2], r"$P_3$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"])
    draw_two_wire_gate(ax, 3.2, ys[0], ys[1], r"$G_{12}$", color=PALETTE["science"])
    draw_two_wire_gate(ax, 4.8, ys[1], ys[2], r"$G_{23}$", color=PALETTE["science"])
    draw_two_wire_gate(ax, 6.4, ys[0], ys[1], r"$G_{12}$", color=PALETTE["science"])
    draw_measure(ax, 8.25, ys[0])
    draw_measure(ax, 8.25, ys[1])
    draw_measure(ax, 8.25, ys[2])
    ax.text(4.75, 6.85, r"$U_{\mathrm{cl}}^\dagger = G_{12}(\theta_3,\chi_3)\,G_{23}(\theta_2,\chi_2)\,G_{12}(\theta_1,\chi_1)\,P$", color=PALETTE["science"], ha="center", fontsize=9.8)
    ax.text(5.0, 1.55, r"Computational-basis outcomes are remapped to $\{+\Lambda,0,-\Lambda\}$.", ha="center", fontsize=9.6)
    ax.text(5.0, 0.72, "Equivalent qutrit / unary-register circuit for the same projective measurement.", ha="center", fontsize=9.4)

    # N-station optical
    ax = axes[1, 0]
    panel_header(ax, 0.2, 7.35, "(c) $N$-station optical implementation")
    ys = [6.3, 5.1, 3.9, 2.7]
    labels = [r"$a_1$", r"$a_2$", r"$\vdots$", r"$a_N$"]
    for y, label in zip(ys, labels):
        draw_wire(ax, y, label=label)
    mesh = FancyBboxPatch((2.5, 2.25), 4.5, 4.8, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=1.3, edgecolor=PALETTE["science"], facecolor=PALETTE["fill_science"])
    ax.add_patch(mesh)
    ax.text(4.75, 4.95, r"Universal mesh $U_{\alpha}^{\dagger}$", ha="center", fontsize=11.0)
    ax.text(4.75, 4.25, r"sequence of $BS_{ij}(\theta,\chi)$ and $PS_i(\varphi)$", ha="center", fontsize=9.6, color=PALETTE["science"])
    ax.text(4.75, 3.45, r"or scheduled $\{U_{\mu}^{\dagger}\}$ for multiple closures", ha="center", fontsize=9.4)
    for x in [3.2, 4.4, 5.6]:
        draw_bs(ax, x, ys[0], ys[1], r"$BS$")
    draw_gate(ax, 3.2, ys[2], r"$PS$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"], fontsize=8.8)
    draw_gate(ax, 5.6, ys[3], r"$PS$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"], fontsize=8.8)
    draw_pd(ax, 8.45, ys[0], r"$PD_1$")
    draw_pd(ax, 8.45, ys[1], r"$PD_2$")
    draw_pd(ax, 8.45, ys[2], r"$\cdots$")
    draw_pd(ax, 8.45, ys[3], r"$PD_N$")
    ax.text(5.0, 1.55, r"Program the mesh to diagonalize $L(\alpha)$; if SLDs do not commute, schedule several settings blockwise.", ha="center", fontsize=9.4)
    ax.text(5.0, 0.72, "Standard linear optics: programmable BS/PS mesh followed by a detector array.", ha="center", fontsize=9.4)

    # N-station circuit
    ax = axes[1, 1]
    panel_header(ax, 0.2, 7.35, "(d) $N$-station circuit view")
    ys = [6.3, 5.1, 3.9, 2.7]
    labels = [r"$|1\rangle$", r"$|2\rangle$", r"$\vdots$", r"$|N\rangle$"]
    for y, label in zip(ys, labels):
        draw_wire(ax, y, label=label)
    for y in [ys[0], ys[1], ys[3]]:
        draw_gate(ax, 1.7, y, r"$P$", fc=PALETTE["fill_aux"], ec=PALETTE["aux"], fontsize=9.0)
    draw_two_wire_gate(ax, 3.15, ys[0], ys[1], r"$G_{12}$", color=PALETTE["science"])
    draw_two_wire_gate(ax, 4.55, ys[1], ys[2], r"$G_{23}$", color=PALETTE["science"])
    draw_two_wire_gate(ax, 5.95, ys[2], ys[3], r"$G_{N-1,N}$", color=PALETTE["science"])
    draw_gate(ax, 7.15, ys[0], r"$\alpha$", fc=PALETTE["fill_control"], ec=PALETTE["control"], fontsize=9.0)
    draw_gate(ax, 7.15, ys[1], r"$/\mu$", fc=PALETTE["fill_control"], ec=PALETTE["control"], fontsize=9.0)
    for y in ys:
        draw_measure(ax, 8.55, y)
    ax.text(4.6, 6.95, r"Gate schedule for $U_{\alpha}^{\dagger}$ or $U_{\mu}^{\dagger}$", ha="center", fontsize=10.0, color=PALETTE["science"])
    ax.text(5.0, 1.55, r"Apply the chosen Givens-rotation mesh, then measure the register and classically weight the outcomes.", ha="center", fontsize=9.4)
    ax.text(5.0, 0.72, "Standard circuit view of the same local SLD projective measurement.", ha="center", fontsize=9.4)

    fig.suptitle("Physical realizations of the optimal closure measurement", fontsize=15, y=1.01)
    png = OUTDIR / "optimal_measurement_implementations.png"
    pdf = OUTDIR / "optimal_measurement_implementations.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    build_adaptive_figure()
    build_implementation_figure()


if __name__ == "__main__":
    main()
