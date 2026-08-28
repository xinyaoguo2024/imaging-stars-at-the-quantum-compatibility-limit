import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


N = 8


def gate_sequence():
    """Reck-style adjacent Givens sequence with 7+6+...+1 = 28 gates."""
    seq = []
    for col in range(1, N):
        for i in range(N, col, -1):
            seq.append((col, i - 1, i))
    return seq


SEQ = gate_sequence()


def setup_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def rounded_box(ax, xy, w, h, text, fc, ec="#2f2a24", fs=8.0, lw=1.0):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.007,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def draw_title(ax, label, title):
    ax.text(0.02, 0.96, label, ha="left", va="top", fontsize=13, fontweight="bold")
    ax.text(0.08, 0.96, title, ha="left", va="top", fontsize=12, fontweight="bold")


def mode_positions():
    top = 0.82
    bottom = 0.20
    return [top - i * (top - bottom) / (N - 1) for i in range(N)]


def draw_optical_reck(ax):
    draw_title(ax, "(a)", "Explicit eight-mode optical Reck mesh")
    ys = mode_positions()
    x0, x1 = 0.095, 0.93
    for k, y in enumerate(ys, start=1):
        ax.text(0.055, y, f"$b_{k}$", ha="right", va="center", fontsize=9.5)
        ax.plot([x0, x1], [y, y], color="#2f2a24", lw=1.05)

    xs = [0.13 + m * 0.0265 for m in range(len(SEQ))]
    stage_centers = {}
    for m, (stage, i, j) in enumerate(SEQ):
        x = xs[m]
        y_top, y_bot = ys[i - 1], ys[j - 1]
        y_mid = (y_top + y_bot) / 2
        h = abs(y_top - y_bot) * 0.78
        w = 0.019
        fc = "#fff2cc" if stage % 2 else "#e2f0d9"
        rounded_box(ax, (x - w / 2, y_mid - h / 2), w, h, "", fc=fc, lw=0.85)
        ax.plot([x, x], [y_bot, y_top], color="#947a2f", lw=0.8)
        ax.text(x, y_mid, f"$G_{{{i}{j}}}$", ha="center", va="center", fontsize=4.8, rotation=90)
        stage_centers.setdefault(stage, []).append(x)

    for stage, vals in stage_centers.items():
        ax.text(sum(vals) / len(vals), 0.135, f"$j={stage}$", ha="center", va="top", fontsize=7.3)

    rounded_box(
        ax,
        (0.20, 0.02),
        0.60,
        0.075,
        "Reck triangular sequence applied left-to-right as drawn.\n"
        r"Stage $j$ contains $8-j$ adjacent MZIs; total gate count $7+6+\cdots+1=28$.",
        fc="#f7f3ea",
        fs=8.2,
    )


def draw_quantum_circuit(ax):
    draw_title(ax, "(b)", "Equivalent single-excitation quantum circuit")
    ys = mode_positions()
    x0, x1 = 0.095, 0.90
    for k, y in enumerate(ys, start=1):
        ax.text(0.055, y, f"$|{k}\\rangle$", ha="right", va="center", fontsize=9.5)
        ax.plot([x0, x1], [y, y], color="#2f2a24", lw=1.0)

    xs = [0.13 + m * 0.0265 for m in range(len(SEQ))]
    for m, (stage, i, j) in enumerate(SEQ):
        x = xs[m]
        y_top, y_bot = ys[i - 1], ys[j - 1]
        ax.plot([x, x], [y_bot, y_top], color="#2f2a24", lw=0.8)
        for y in (y_top, y_bot):
            circ = Circle((x, y), 0.0085, facecolor="#ffffff", edgecolor="#2f2a24", lw=0.8)
            ax.add_patch(circ)
        y_mid = (y_top + y_bot) / 2
        fc = "#fff2cc" if stage % 2 else "#e2f0d9"
        rounded_box(ax, (x - 0.013, y_mid - 0.024), 0.026, 0.048, "$G$", fc=fc, fs=5.5, lw=0.75)

    rounded_box(ax, (0.91, 0.17), 0.065, 0.70, "measure\nmode\nindex $r$", fc="#f9d7d7", fs=8.0)
    for y in ys:
        ax.plot([x1, 0.91], [y, y], color="#2f2a24", lw=0.9)

    rounded_box(
        ax,
        (0.19, 0.02),
        0.62,
        0.075,
        "Unary photonic circuit: each two-level gate is the same MZI as in panel (a).\n"
        r"Outcome $r$ is assigned the SLD eigenvalue $\lambda_r$ and accumulated as the score.",
        fc="#f7f3ea",
        fs=8.0,
    )


fig, axs = plt.subplots(2, 1, figsize=(13.6, 8.1))
for ax in axs:
    setup_axis(ax)

draw_optical_reck(axs[0])
draw_quantum_circuit(axs[1])

fig.patch.set_facecolor("white")
plt.tight_layout(pad=1.4)
for ext in ("pdf", "png"):
    fig.savefig(f"output/figures/supplement_8station_reck_circuit.{ext}", dpi=240, bbox_inches="tight")
