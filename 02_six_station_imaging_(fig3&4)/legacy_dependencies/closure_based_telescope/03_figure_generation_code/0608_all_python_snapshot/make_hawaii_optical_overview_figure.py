from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

R_EARTH_KM = 6371.0

CLUSTER_CENTER = {
    "Maunakea": (19.8250, -155.4720),
    "Haleakala": (20.7074, -156.2570),
    "Mauna Loa": (19.5360, -155.5760),
}

HAWAII_CENTER = (
    sum(lat for lat, lon in CLUSTER_CENTER.values()) / 3.0,
    sum(lon for lat, lon in CLUSTER_CENTER.values()) / 3.0,
)

VISIBLE_400_800 = [
    ("Keck I", 19.8259465, -155.4747185, "10 m", "Maunakea"),
    ("Keck II", 19.8265605, -155.4742341, "10 m", "Maunakea"),
    ("Subaru", 19.82550396, -155.47601866, "8.2 m", "Maunakea"),
    ("Gemini North", 19.82380145, -155.46904675, "8.1 m", "Maunakea"),
    ("CFHT", 19.82525180, -155.46887572, "3.6 m", "Maunakea"),
    ("DKIST", 20.70670, -156.25630, "4.0 m solar", "Haleakala"),
    ("Pan-STARRS", 20.70730, -156.25590, "2 x 1.8 m", "Haleakala"),
    ("Faulkes North", 20.70695, -156.25735, "2.0 m", "Haleakala"),
    ("AEOS", 20.70820, -156.25720, "3.67 m", "Haleakala"),
    ("ATLAS-HKO", 20.70780, -156.25820, "0.5 m", "Haleakala"),
    ("ATLAS-MLO", 19.53600, -155.57600, "0.5 m", "Mauna Loa"),
]

COLORS = {"Maunakea": "#2ca02c", "Haleakala": "#ff7f0e", "Mauna Loa": "#9467bd"}


def xy_km(lat: float, lon: float, center: tuple[float, float]) -> tuple[float, float]:
    lat0, lon0 = center
    x = R_EARTH_KM * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
    y = R_EARTH_KM * math.radians(lat - lat0)
    return x, y


def annotate(ax, x: float, y: float, text: str, offset: tuple[int, int], fs: float = 7.0) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fs,
        bbox={"facecolor": "white", "edgecolor": "0.70", "alpha": 0.90, "pad": 1.3},
        zorder=5,
    )


def setup(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11.5, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.58, linewidth=0.65)
    ax.set_aspect("equal", adjustable="box")


def draw_overview(ax) -> None:
    xs, ys = [], []
    cluster_notes = {
        "Maunakea": "Maunakea\n400-800 nm capable:\nKeck, Subaru,\nGemini N, CFHT",
        "Haleakala": "Haleakala\nDKIST, Pan-STARRS,\nFaulkes, AEOS, ATLAS",
        "Mauna Loa": "Mauna Loa\nATLAS-MLO 0.5 m",
    }
    offsets = {"Maunakea": (12, -8), "Haleakala": (-150, 18), "Mauna Loa": (12, -36)}
    for cluster, (lat, lon) in CLUSTER_CENTER.items():
        x, y = xy_km(lat, lon, HAWAII_CENTER)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=105, color=COLORS[cluster], edgecolor="black", zorder=4)
        annotate(ax, x, y, cluster_notes[cluster], offsets[cluster], fs=7.1)
    ax.scatter(0, 0, marker="+", color="crimson", s=180, linewidths=2.0, zorder=3)
    setup(ax, "Hawaii full-region 400-800 nm overview", "x east of Hawaii center (km)", "y north of Hawaii center (km)")
    ax.set_xlim(min(xs) - 34, max(xs) + 36)
    ax.set_ylim(min(ys) - 34, max(ys) + 36)
    ax.text(
        0.02,
        0.025,
        f"center: {HAWAII_CENTER[0]:.3f} deg N, {abs(HAWAII_CENTER[1]):.3f} deg W",
        transform=ax.transAxes,
        fontsize=7.6,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.83},
    )


def draw_cluster_zoom(ax, cluster: str, title: str) -> None:
    rows = [row for row in VISIBLE_400_800 if row[4] == cluster]
    center = CLUSTER_CENTER[cluster]
    offsets = {
        "Keck I": (-78, 18),
        "Keck II": (16, 20),
        "Subaru": (-88, -30),
        "Gemini North": (16, -16),
        "CFHT": (18, 20),
        "DKIST": (-100, -18),
        "Pan-STARRS": (18, 18),
        "Faulkes North": (-108, 26),
        "AEOS": (18, 24),
        "ATLAS-HKO": (18, -36),
        "ATLAS-MLO": (18, 12),
    }
    xs, ys = [], []
    for name, lat, lon, aperture, _ in rows:
        x, y = xy_km(lat, lon, center)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=80, color=COLORS[cluster], edgecolor="black", zorder=4)
        annotate(ax, x, y, f"{name}\n{aperture}", offsets[name], fs=6.9)
    setup(ax, title, f"x east of {cluster} center (km)", f"y north of {cluster} center (km)")
    pad = 0.65 if len(rows) > 1 else 5.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15.8, 9.2), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.08, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_over = fig.add_subplot(gs[:, 0])
    ax_mk = fig.add_subplot(gs[0, 1:])
    ax_hal = fig.add_subplot(gs[1, 1])
    ax_ml = fig.add_subplot(gs[1, 2])
    draw_overview(ax_over)
    draw_cluster_zoom(ax_mk, "Maunakea", "Maunakea 400-800 nm-capable zoom")
    draw_cluster_zoom(ax_hal, "Haleakala", "Haleakala optical zoom")
    draw_cluster_zoom(ax_ml, "Mauna Loa", "Mauna Loa auxiliary optical site")
    fig.suptitle("Hawaii 400-800 nm-capable optical telescope distribution", fontsize=16, weight="bold", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "IR-dominated Maunakea facilities (NASA IRTF and UKIRT) and mm/sub-mm/radio facilities are excluded.",
        ha="center",
        fontsize=9,
    )
    out_pdf = OUT / "hawaii_optical_overview_zoom.pdf"
    out_png = OUT / "hawaii_optical_overview_zoom.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()
