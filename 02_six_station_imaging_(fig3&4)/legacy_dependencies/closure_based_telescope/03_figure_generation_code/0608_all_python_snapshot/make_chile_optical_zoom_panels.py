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


# Representative 400-800 nm-capable optical site centers.  Dense summit clusters are
# expanded where we have meaningful telescope-level coordinates or separations.
SITES = [
    ("Armazones / ELT", -24.589444, -70.191667, "ELT 39 m\nunder construction"),
    ("Paranal / VLT", -24.627222, -70.404167, "VLT 4 x 8.2 m\nVLTI ATs 4 x 1.8 m\nVST 2.6 m"),
    ("La Silla", -29.261056, -70.731750, "ESO 3.6 m\nNTT 3.58 m\nMPG/ESO 2.2 m"),
    ("LCO / Magellan", -29.015000, -70.691700, "Magellan 2 x 6.5 m\ndu Pont 2.5 m\nSwope 1 m"),
    ("LCO / GMT", -29.048333, -70.683611, "GMT 24.5-25.4 m class\nunder construction"),
    ("CTIO / Blanco", -30.169000, -70.804000, "Blanco 4 m\nSMARTS 0.9-1.5 m"),
    ("Gemini South", -30.240750, -70.736693, "8.1 m"),
    ("SOAR", -30.238000, -70.733600, "4.1 m"),
    ("Rubin / El Penon", -30.244633, -70.749417, "8.4/6.7 m"),
]


def xy_km(lat: float, lon: float, center: tuple[float, float]) -> tuple[float, float]:
    lat0, lon0 = center
    x = R_EARTH_KM * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
    y = R_EARTH_KM * math.radians(lat - lat0)
    return x, y


def local_center(names: list[str]) -> tuple[float, float]:
    rows = [row for row in SITES if row[0] in names]
    return sum(row[1] for row in rows) / len(rows), sum(row[2] for row in rows) / len(rows)


def annotate(ax, x: float, y: float, text: str, offset: tuple[int, int], fs: float = 7.1) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fs,
        bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.90, "pad": 1.35},
        zorder=5,
    )


def setup(ax, title: str, xlabel: str = "local x east (km)", ylabel: str = "local y north (km)") -> None:
    ax.set_title(title, fontsize=11.2, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.65, alpha=0.58)
    ax.set_aspect("equal", adjustable="box")


def draw_overview(ax) -> None:
    xs, ys = [], []
    for name, lat, lon, label in SITES:
        x, y = xy_km(lat, lon, CHILE_CENTER)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=54, color="#1f77b4", edgecolor="black", zorder=4)
    overview_labels = [
        ("Paranal-Armazones", -24.61, -70.30, "22 km pair\nELT + VLT", (10, 8)),
        ("La Silla-LCO", -29.13, -70.71, "28 km regional pair\n3-25 m class", (10, -10)),
        ("LCO local", -29.03, -70.69, "Magellan + GMT\n3.8 km", (10, 16)),
        ("Coquimbo south", -30.22, -70.75, "CTIO + Pachon + Rubin\n10 km scale", (10, -28)),
    ]
    for title, lat, lon, note, offset in overview_labels:
        x, y = xy_km(lat, lon, CHILE_CENTER)
        annotate(ax, x, y, f"{title}\n{note}", offset, fs=6.8)
    ax.scatter(0, 0, marker="+", color="crimson", s=150, linewidths=1.8)
    setup(ax, "Chile 400-800 nm-capable overview", "x east of Chile center (km)", "y north of Chile center (km)")
    ax.set_xlim(min(xs) - 85, max(xs) + 95)
    ax.set_ylim(min(ys) - 58, max(ys) + 58)
    ax.text(
        0.02,
        0.025,
        f"center: {abs(CHILE_CENTER[0]):.3f} deg S, {abs(CHILE_CENTER[1]):.3f} deg W",
        transform=ax.transAxes,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.83},
    )


def draw_zoom(
    ax,
    names: list[str],
    title: str,
    offsets: dict[str, tuple[int, int]],
    pad_x: float,
    pad_y: float,
    *,
    fixed_xlim: tuple[float, float] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
    center: tuple[float, float] | None = None,
) -> None:
    if center is None:
        center = local_center(names)
    rows = [row for row in SITES if row[0] in names]
    xs, ys = [], []
    for name, lat, lon, label in rows:
        x, y = xy_km(lat, lon, center)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=76, color="#1f77b4", edgecolor="black", zorder=4)
        annotate(ax, x, y, f"{name}\n{label}", offsets.get(name, (10, 10)), fs=6.9)
    setup(ax, title)
    if fixed_xlim is None:
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    else:
        ax.set_xlim(*fixed_xlim)
    if fixed_ylim is None:
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    else:
        ax.set_ylim(*fixed_ylim)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15.8, 10.8), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.05, 1.08], height_ratios=[1.0, 1.0])
    ax_overview = fig.add_subplot(gs[:, 0])
    ax_north = fig.add_subplot(gs[0, 1])
    ax_lco_regional = fig.add_subplot(gs[0, 2])
    ax_lco_local = fig.add_subplot(gs[1, 1])
    ax_coquimbo = fig.add_subplot(gs[1, 2])

    draw_overview(ax_overview)
    draw_zoom(
        ax_north,
        ["Armazones / ELT", "Paranal / VLT"],
        "Zoom 1: Paranal-Armazones",
        {
            "Armazones / ELT": (12, 14),
            "Paranal / VLT": (-138, -42),
        },
        pad_x=34,
        pad_y=26,
    )
    draw_zoom(
        ax_lco_regional,
        ["La Silla", "LCO / Magellan", "LCO / GMT"],
        "Zoom 2: La Silla-Las Campanas",
        {
            "La Silla": (12, -54),
            "LCO / Magellan": (12, 18),
            "LCO / GMT": (12, -38),
        },
        pad_x=30,
        pad_y=42,
    )
    draw_zoom(
        ax_lco_local,
        ["LCO / Magellan", "LCO / GMT"],
        "Zoom 3: Las Campanas local",
        {
            "LCO / Magellan": (12, 18),
            "LCO / GMT": (12, -50),
        },
        pad_x=5.2,
        pad_y=5.0,
    )
    # A strict 10 km square can include CTIO and Rubin/Pachon if centered
    # between the two ridges; their separation is close to the square diagonal.
    ctio_rubin_center = (
        (-30.169000 + -30.244633) / 2.0,
        (-70.804000 + -70.749417) / 2.0,
    )
    draw_zoom(
        ax_coquimbo,
        ["CTIO / Blanco", "Gemini South", "SOAR", "Rubin / El Penon"],
        "Zoom 4: CTIO-Pachon-Rubin (10 km x 10 km)",
        {
            "CTIO / Blanco": (10, -42),
            "Gemini South": (-118, 42),
            "SOAR": (-98, 4),
            "Rubin / El Penon": (-116, -36),
        },
        pad_x=5,
        pad_y=5,
        fixed_xlim=(-5.0, 5.0),
        fixed_ylim=(-5.0, 5.0),
        center=ctio_rubin_center,
    )
    fig.suptitle("Chile 400-800 nm-capable optical telescope dense regions", fontsize=16, weight="bold", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Each label gives representative 400-800 nm-capable aperture(s).  Coordinates are local tangent-plane distances in km; VISTA is omitted as IR-dominated.",
        ha="center",
        fontsize=9,
    )
    out_pdf = OUT / "chile_optical_zoom_panels.pdf"
    out_png = OUT / "chile_optical_zoom_panels.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()
