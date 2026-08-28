from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

R_EARTH_KM = 6371.0


CHILE_CENTER = (-27.32302158, -70.05071717)
CHILE_OPTICAL = [
    ("Armazones / ELT", -24.589444, -70.191667, "ELT 39 m\nunder construction", "Chile"),
    ("Paranal / VLT", -24.627222, -70.404167, "VLT 4 x 8.2 m\nVISTA 4.1 m, VST 2.6 m", "Chile"),
    ("La Silla", -29.261056, -70.731750, "3.6 m, NTT 3.58 m\nMPG/ESO 2.2 m", "Chile"),
    ("LCO / Magellan", -29.015000, -70.691700, "Magellan 2 x 6.5 m\ndu Pont 2.5 m, Swope 1 m", "Chile"),
    ("LCO / GMT", -29.048333, -70.683611, "GMT 24.5-25.4 m class\nunder construction", "Chile"),
    ("Cerro Tololo / CTIO", -30.169000, -70.804000, "Blanco 4 m\nSMARTS 1.5/1.3/1.0/0.9 m", "Chile"),
    ("Cerro Pachon", -30.240750, -70.736693, "Gemini S 8.1 m\nSOAR 4.1 m", "Chile"),
    ("Rubin / El Penon", -30.244633, -70.749417, "Rubin 8.4 m primary\n6.7 m effective", "Chile"),
]


HAWAII_CLUSTER_CENTER = {
    "Maunakea": (19.8250, -155.4720),
    "Haleakala": (20.7074, -156.2570),
    "Mauna Loa": (19.5360, -155.5760),
}

HAWAII_CENTER = (
    sum(v[0] for v in HAWAII_CLUSTER_CENTER.values()) / 3,
    sum(v[1] for v in HAWAII_CLUSTER_CENTER.values()) / 3,
)

HAWAII_OPTICAL = [
    ("Keck I", 19.8259465, -155.4747185, "10 m", "Maunakea"),
    ("Keck II", 19.8265605, -155.4742341, "10 m", "Maunakea"),
    ("Subaru", 19.82550396, -155.47601866, "8.2 m", "Maunakea"),
    ("Gemini North", 19.82380145, -155.46904675, "8.1 m", "Maunakea"),
    ("CFHT", 19.82525180, -155.46887572, "3.6 m", "Maunakea"),
    ("NASA IRTF", 19.82621832, -155.47199879, "3.0 m IR", "Maunakea"),
    ("UKIRT", 19.82243148, -155.47032675, "3.8 m IR", "Maunakea"),
    ("UH 2.2 m", 19.82299107, -155.46943354, "2.2 m", "Maunakea"),
    ("UH 0.6 m", 19.82161430, -155.47096274, "0.6 m", "Maunakea"),
    ("DKIST", 20.70670, -156.25630, "4.0 m solar", "Haleakala"),
    ("Pan-STARRS", 20.70730, -156.25590, "2 x 1.8 m", "Haleakala"),
    ("Faulkes North", 20.70695, -156.25735, "2.0 m", "Haleakala"),
    ("AEOS", 20.70820, -156.25720, "3.67 m", "Haleakala"),
    ("ATLAS-HKO", 20.70780, -156.25820, "0.5 m", "Haleakala"),
    ("ATLAS-MLO", 19.53600, -155.57600, "0.5 m", "Mauna Loa"),
]


def xy_km(lat: float, lon: float, center: tuple[float, float]) -> tuple[float, float]:
    lat0, lon0 = center
    x = R_EARTH_KM * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
    y = R_EARTH_KM * math.radians(lat - lat0)
    return x, y


def annotate(ax, x: float, y: float, text: str, offset: tuple[int, int], fontsize: float = 6.9) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "0.70", "alpha": 0.88, "pad": 1.4},
        zorder=5,
    )


def setup_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10.5, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.65, alpha=0.55)
    ax.set_aspect("equal", adjustable="box")


def draw_chile_overview(ax) -> None:
    xs, ys = [], []
    for name, lat, lon, aperture, _ in CHILE_OPTICAL:
        x, y = xy_km(lat, lon, CHILE_CENTER)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=62, color="#1f77b4", edgecolor="black", zorder=4)
    cluster_labels = [
        ("Paranal / Armazones", -24.61, -70.30, "ELT 39 m; VLT 4 x 8.2 m", (12, 10)),
        ("La Silla / LCO", -29.12, -70.70, "3.6-25 m class", (12, -4)),
        ("CTIO / Pachon / Rubin", -30.22, -70.76, "4-8 m class", (12, -16)),
    ]
    for label, lat, lon, aperture, offset in cluster_labels:
        x, y = xy_km(lat, lon, CHILE_CENTER)
        annotate(ax, x, y, f"{label}\n{aperture}", offset, fontsize=6.6)
    ax.scatter(0, 0, marker="+", color="crimson", s=150, linewidths=1.8, zorder=3)
    setup_axis(ax, "Chile overview", "x east of Chile center (km)", "y north of Chile center (km)")
    ax.set_xlim(min(xs) - 75, max(xs) + 75)
    ax.set_ylim(min(ys) - 55, max(ys) + 55)
    ax.text(
        0.02,
        0.025,
        f"center: {abs(CHILE_CENTER[0]):.3f} deg S, {abs(CHILE_CENTER[1]):.3f} deg W",
        transform=ax.transAxes,
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.82},
    )


def draw_chile_zoom(ax, names: set[str], title: str) -> None:
    rows = [row for row in CHILE_OPTICAL if row[0] in names]
    center = (
        sum(row[1] for row in rows) / len(rows),
        sum(row[2] for row in rows) / len(rows),
    )
    offsets = {
        "Armazones / ELT": (8, 8),
        "Paranal / VLT": (-130, -28),
        "La Silla": (8, -38),
        "LCO / Magellan": (8, 18),
        "LCO / GMT": (8, -35),
        "Cerro Tololo / CTIO": (-132, -48),
        "Cerro Pachon": (8, 20),
        "Rubin / El Penon": (8, -45),
    }
    xs, ys = [], []
    for name, lat, lon, aperture, _ in rows:
        x, y = xy_km(lat, lon, center)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=75, color="#1f77b4", edgecolor="black", zorder=4)
        annotate(ax, x, y, f"{name}\n{aperture}", offsets[name], fontsize=6.7)
    setup_axis(ax, title, "local x east (km)", "local y north (km)")
    ax.set_xlim(min(xs) - 34, max(xs) + 42)
    ax.set_ylim(min(ys) - 34, max(ys) + 34)


def draw_hawaii_overview(ax) -> None:
    colors = {"Maunakea": "#2ca02c", "Haleakala": "#ff7f0e", "Mauna Loa": "#9467bd"}
    xs, ys = [], []
    for cluster, (lat, lon) in HAWAII_CLUSTER_CENTER.items():
        x, y = xy_km(lat, lon, HAWAII_CENTER)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=72, color=colors[cluster], edgecolor="black", zorder=4)
        label = {
            "Maunakea": "Maunakea\nKeck, Subaru, Gemini N,\nCFHT, IRTF, UKIRT, UH",
            "Haleakala": "Haleakala\nDKIST, Pan-STARRS,\nFaulkes, AEOS, ATLAS",
            "Mauna Loa": "Mauna Loa\nATLAS-MLO 0.5 m",
        }[cluster]
        offset = {"Maunakea": (12, 10), "Haleakala": (-145, 12), "Mauna Loa": (12, -34)}[cluster]
        annotate(ax, x, y, label, offset, fontsize=6.6)
    ax.scatter(0, 0, marker="+", color="crimson", s=150, linewidths=1.8, zorder=3)
    setup_axis(ax, "Hawaii optical / near-IR clusters", "x east of Hawaii center (km)", "y north of Hawaii center (km)")
    ax.set_xlim(min(xs) - 28, max(xs) + 30)
    ax.set_ylim(min(ys) - 30, max(ys) + 30)
    ax.text(
        0.02,
        0.02,
        f"center: {HAWAII_CENTER[0]:.3f} deg N, {abs(HAWAII_CENTER[1]):.3f} deg W",
        transform=ax.transAxes,
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.8},
    )


def draw_hawaii_zoom(ax, cluster: str, title: str) -> None:
    rows = [row for row in HAWAII_OPTICAL if row[4] == cluster]
    center = HAWAII_CLUSTER_CENTER[cluster]
    colors = {"Maunakea": "#2ca02c", "Haleakala": "#ff7f0e", "Mauna Loa": "#9467bd"}
    offsets = {
        "Keck I": (-78, 18),
        "Keck II": (16, 20),
        "Subaru": (-88, -30),
        "Gemini North": (16, -14),
        "CFHT": (18, 20),
        "NASA IRTF": (18, -40),
        "UKIRT": (16, -14),
        "UH 2.2 m": (18, 23),
        "UH 0.6 m": (-78, -34),
        "DKIST": (-96, -18),
        "Pan-STARRS": (18, 18),
        "Faulkes North": (-102, 26),
        "AEOS": (18, 24),
        "ATLAS-HKO": (18, -36),
    }
    xs, ys = [], []
    for name, lat, lon, aperture, _ in rows:
        x, y = xy_km(lat, lon, center)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=68, color=colors[cluster], edgecolor="black", zorder=4)
        annotate(ax, x, y, f"{name}\n{aperture}", offsets[name], fontsize=6.4)
    setup_axis(ax, title, f"x east of {cluster} center (km)", f"y north of {cluster} center (km)")
    pad = 0.65
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16.2, 10.8), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.02, 1.18], height_ratios=[1.0, 1.0])
    ax_chile_overview = fig.add_subplot(gs[0, 0])
    ax_chile_north = fig.add_subplot(gs[0, 1])
    ax_hawaii = fig.add_subplot(gs[0, 2])
    ax_chile_south = fig.add_subplot(gs[1, 0])
    ax_mk = fig.add_subplot(gs[1, 1])
    ax_hal = fig.add_subplot(gs[1, 2])
    draw_chile_overview(ax_chile_overview)
    draw_chile_zoom(ax_chile_north, {"Armazones / ELT", "Paranal / VLT"}, "Chile northern zoom")
    draw_hawaii_overview(ax_hawaii)
    draw_chile_zoom(
        ax_chile_south,
        {"La Silla", "LCO / Magellan", "LCO / GMT", "Cerro Tololo / CTIO", "Cerro Pachon", "Rubin / El Penon"},
        "Chile central/southern zoom",
    )
    draw_hawaii_zoom(ax_mk, "Maunakea", "Maunakea optical/NIR zoom")
    draw_hawaii_zoom(ax_hal, "Haleakala", "Haleakala optical zoom")
    fig.suptitle(
        "Optical / near-infrared telescope distribution in Chile and Hawaii",
        fontsize=16,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Only optical / near-IR facilities from the current Chile and Hawaii summaries are shown; mm/sub-mm/radio facilities are excluded.",
        ha="center",
        fontsize=9,
    )
    out_pdf = OUT / "optical_telescope_distribution_chile_hawaii.pdf"
    out_png = OUT / "optical_telescope_distribution_chile_hawaii.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()
