import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


def box(ax, xy, w, h, text, fc="#f7f3ea", ec="#2f2a24", fs=9, lw=1.2, r=0.025):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#1f1b16",
        linespacing=1.18,
    )
    return patch


def arrow(ax, start, end, text=None, fs=8, color="#2f2a24", rad=0.0, lw=1.1):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, text, ha="center", va="bottom", fontsize=fs, color=color)
    return patch


def draw_panel_title(ax, label, title):
    ax.text(0.02, 0.965, label, fontsize=13, fontweight="bold", va="top", ha="left")
    ax.text(0.095, 0.965, title, fontsize=12, fontweight="bold", va="top", ha="left")


def draw_teleport_panel(ax):
    draw_panel_title(ax, "(a)", "Entanglement-assisted station-to-hub mode transfer")
    ax.text(
        0.05,
        0.08,
        "Per time-frequency bin, each station mode is mapped to a hub mode.\n"
        "Classical heralds select successful bins; known Pauli/phase frames are corrected at the hub.",
        fontsize=8.1,
        ha="left",
        va="bottom",
        color="#4c453c",
        linespacing=1.15,
    )
    box(
        ax,
        (0.34, 0.82),
        0.34,
        0.08,
        "pre-shared Bell resource in each\n"
        "time-frequency bin",
        fc="#fff2cc",
        fs=8.0,
    )
    ys = [0.73, 0.53, 0.33]
    for k, y in enumerate(ys, start=1):
        box(ax, (0.045, y - 0.04), 0.12, 0.08, f"station {k}\nmode $a_{k}$", fc="#d9edf7", fs=8.5)
        box(ax, (0.22, y - 0.045), 0.13, 0.09, "Bell\nmeasurement", fc="#fde9d9", fs=8.2)
        box(ax, (0.63, y - 0.04), 0.11, 0.08, f"hub\nmode $b_{k}$", fc="#e2f0d9", fs=8.5)
        arrow(ax, (0.165, y), (0.22, y), "$a_i$", fs=7.5)
        arrow(ax, (0.35, y), (0.63, y), "herald + frame", fs=7.3)
        ax.plot([0.41, 0.56], [y - 0.075, y - 0.075], color="#947a2f", lw=1.3)
        ax.plot([0.285, 0.41], [y - 0.045, y - 0.075], color="#947a2f", lw=0.9)
        ax.plot([0.685, 0.56], [y - 0.040, y - 0.075], color="#947a2f", lw=0.9)
        ax.text(0.485, y - 0.105, f"$|\\Phi^+\\rangle_{{A_{k}B_{k}}}$", ha="center", va="top", fontsize=7.8)


def draw_three_panel(ax):
    draw_panel_title(ax, "(b)", "Three-station SLD unitary")
    ys = [0.72, 0.55, 0.38]
    labels = ["$b_1$", "$b_2$", "$b_3$"]
    for y, lab in zip(ys, labels):
        ax.text(0.06, y, lab, ha="right", va="center", fontsize=10)
        ax.plot([0.075, 0.29], [y, y], color="#2f2a24", lw=1.3)
    box(
        ax,
        (0.30, 0.31),
        0.22,
        0.48,
        "$U_{\\rm cl}^{\\dagger}$\n"
        "tritter for\n"
        "$\\nu_{12}=\\nu_{23}=\\nu_{31}$\n"
        "or 3 Givens\n"
        "rotations",
        fc="#efe4f7",
        fs=8.2,
    )
    out_labels = ["$D_+$", "$D_0$", "$D_-$"]
    for y, lab in zip(ys, out_labels):
        ax.plot([0.52, 0.70], [y, y], color="#2f2a24", lw=1.3)
        circ = Circle((0.735, y), 0.035, facecolor="#f9d7d7", edgecolor="#2f2a24", lw=1.1)
        ax.add_patch(circ)
        ax.text(0.735, y, lab, ha="center", va="center", fontsize=8.2)
    box(
        ax,
        (0.12, 0.08),
        0.72,
        0.13,
        "One click in eigenmode $r$ returns score $\\lambda_r$.\n"
        "For the symmetric triangle this is the chiral count contrast $n_+ - n_-$.",
        fc="#f7f3ea",
        fs=8.4,
    )


def draw_eight_panel(ax):
    draw_panel_title(ax, "(c)", "Eight-station closure-subspace receiver")
    y0, dy = 0.79, 0.075
    for i in range(8):
        y = y0 - i * dy
        ax.text(0.055, y, f"$b_{i+1}$", ha="right", va="center", fontsize=8.5)
        ax.plot([0.07, 0.25], [y, y], color="#2f2a24", lw=1.0)
    box(
        ax,
        (0.26, 0.22),
        0.27,
        0.62,
        "$8\\times8$ programmable\n"
        "Clements/Reck mesh\n"
        "$U_{\\alpha}^{\\dagger}$\n\n"
        "28 tunable MZIs\n"
        "+ phase shifters",
        fc="#e7f0fb",
        fs=8.6,
    )
    for i in range(8):
        y = y0 - i * dy
        ax.plot([0.53, 0.69], [y, y], color="#2f2a24", lw=1.0)
        circ = Circle((0.72, y), 0.023, facecolor="#f9d7d7", edgecolor="#2f2a24", lw=0.9)
        ax.add_patch(circ)
    box(
        ax,
        (0.76, 0.37),
        0.18,
        0.30,
        "detectors\n"
        "$D_1,\\ldots,D_8$\n\n"
        "score\n"
        "$\\sum_r \\lambda_r n_r$",
        fc="#fbe5d6",
        fs=8.1,
    )
    box(
        ax,
        (0.12, 0.05),
        0.78,
        0.12,
        "Choose a protected scalar $q_\\alpha=\\alpha^Tq$.\n"
        "For a complete 8-station graph: 28 edges, 21 independent closure modes.",
        fc="#f7f3ea",
        fs=8.3,
    )


def draw_circuit_panel(ax):
    draw_panel_title(ax, "(d)", "Equivalent single-excitation circuit")
    y_top = 0.78
    ys = [y_top - i * 0.12 for i in range(5)]
    line_labels = ["$|1\\rangle$", "$|2\\rangle$", "$|3\\rangle$", "$\\cdots$", "$|8\\rangle$"]
    for y, lab in zip(ys, line_labels):
        ax.text(0.07, y, lab, ha="right", va="center", fontsize=9)
        ax.plot([0.09, 0.88], [y, y], color="#2f2a24", lw=1.0)
    box(ax, (0.13, 0.28), 0.13, 0.58, "phase\nframe\n$P_f$", fc="#fff2cc", fs=8.6)
    box(ax, (0.31, 0.28), 0.13, 0.58, "$G_{ij}$\n(two-level\nrotation)", fc="#e2f0d9", fs=8.2)
    box(ax, (0.49, 0.28), 0.13, 0.58, "$G_{kl}$\n$\\cdots$", fc="#e2f0d9", fs=8.4)
    box(ax, (0.67, 0.28), 0.13, 0.58, "measure\nmode\nindex $r$", fc="#f9d7d7", fs=8.4)
    box(
        ax,
        (0.12, 0.07),
        0.76,
        0.11,
        "The circuit is just the compiled optical unitary in the one-photon basis.\n"
        "Classical post-processing weights each outcome by the SLD eigenvalue $\\lambda_r$.",
        fc="#f7f3ea",
        fs=8.0,
    )


fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.4))
for ax in axs.flat:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

draw_teleport_panel(axs[0])
draw_three_panel(axs[1])

fig.patch.set_facecolor("white")
plt.tight_layout(pad=1.6)
for ext in ("pdf", "png"):
    fig.savefig(f"output/figures/supplement_receiver_construction.{ext}", dpi=220, bbox_inches="tight")
