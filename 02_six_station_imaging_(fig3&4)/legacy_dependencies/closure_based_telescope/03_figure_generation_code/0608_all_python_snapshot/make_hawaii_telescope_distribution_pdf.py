from __future__ import annotations

import csv
import math
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT_PDF = ROOT / "output" / "pdf"
OUT_FIG = ROOT / "output" / "figures"
OUT_DATA = ROOT / "output" / "data"
for directory in (OUT_PDF, OUT_FIG, OUT_DATA):
    directory.mkdir(parents=True, exist_ok=True)


# Coordinates are representative telescope or facility positions.  Most
# Maunakea coordinates follow the IRTF coordinate table; compact array centers
# and Haleakala facilities are rounded to the precision needed for km-scale
# layout work.
TELESCOPES = [
    {
        "facility": "Keck I",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.8259465,
        "lon": -155.4747185,
        "alt_m": 4145,
        "aperture": "10 m",
        "status": "active",
        "notes": "W. M. Keck Observatory optical/IR telescope",
    },
    {
        "facility": "Keck II",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.8265605,
        "lon": -155.4742341,
        "alt_m": 4145,
        "aperture": "10 m",
        "status": "active",
        "notes": "W. M. Keck Observatory optical/IR telescope",
    },
    {
        "facility": "Subaru",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82550396,
        "lon": -155.47601866,
        "alt_m": 4139,
        "aperture": "8.2 m",
        "status": "active",
        "notes": "NAOJ optical/IR telescope",
    },
    {
        "facility": "Gemini North",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82380145,
        "lon": -155.46904675,
        "alt_m": 4213,
        "aperture": "8.1 m",
        "status": "active",
        "notes": "International Gemini Observatory optical/IR telescope",
    },
    {
        "facility": "CFHT",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82525180,
        "lon": -155.46887572,
        "alt_m": 4204,
        "aperture": "3.6 m",
        "status": "active",
        "notes": "Canada-France-Hawaii Telescope",
    },
    {
        "facility": "NASA IRTF",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82621832,
        "lon": -155.47199879,
        "alt_m": 4168,
        "aperture": "3.0 m",
        "status": "active",
        "notes": "NASA Infrared Telescope Facility",
    },
    {
        "facility": "UKIRT",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82243148,
        "lon": -155.47032675,
        "alt_m": 4194,
        "aperture": "3.8 m",
        "status": "active",
        "notes": "UK Infrared Telescope; operational status subject to UH decommissioning plans",
    },
    {
        "facility": "UH 2.2 m",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82299107,
        "lon": -155.46943354,
        "alt_m": 4200,
        "aperture": "2.2 m",
        "status": "active",
        "notes": "University of Hawaii telescope",
    },
    {
        "facility": "UH 0.6 m",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82161430,
        "lon": -155.47096274,
        "alt_m": 4210,
        "aperture": "0.6 m",
        "status": "small / local",
        "notes": "University of Hawaii small telescope",
    },
    {
        "facility": "JCMT",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82280761,
        "lon": -155.47700109,
        "alt_m": 4092,
        "aperture": "15 m",
        "status": "active",
        "notes": "James Clerk Maxwell Telescope, sub-mm single dish",
    },
    {
        "facility": "SMA",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.8242,
        "lon": -155.4780,
        "alt_m": 4080,
        "aperture": "8 x 6 m",
        "status": "active",
        "notes": "Submillimeter Array; representative array-center coordinate",
    },
    {
        "facility": "VLBA Maunakea",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.80138,
        "lon": -155.45550,
        "alt_m": 3720,
        "aperture": "25 m",
        "status": "active",
        "notes": "NRAO Very Long Baseline Array antenna",
    },
    {
        "facility": "CSO",
        "cluster": "Maunakea",
        "island": "Hawai'i",
        "lat": 19.82425764,
        "lon": -155.47753210,
        "alt_m": 4080,
        "aperture": "10.4 m",
        "status": "decommissioned",
        "notes": "Caltech Submillimeter Observatory; included for site context",
    },
    {
        "facility": "DKIST",
        "cluster": "Haleakala",
        "island": "Maui",
        "lat": 20.70670,
        "lon": -156.25630,
        "alt_m": 3050,
        "aperture": "4.0 m",
        "status": "active",
        "notes": "Daniel K. Inouye Solar Telescope",
    },
    {
        "facility": "Pan-STARRS",
        "cluster": "Haleakala",
        "island": "Maui",
        "lat": 20.70730,
        "lon": -156.25590,
        "alt_m": 3050,
        "aperture": "2 x 1.8 m",
        "status": "active",
        "notes": "PS1 and PS2 wide-field survey telescopes",
    },
    {
        "facility": "Faulkes North",
        "cluster": "Haleakala",
        "island": "Maui",
        "lat": 20.70695,
        "lon": -156.25735,
        "alt_m": 3050,
        "aperture": "2.0 m",
        "status": "active",
        "notes": "Las Cumbres Observatory / Faulkes Telescope North",
    },
    {
        "facility": "AEOS",
        "cluster": "Haleakala",
        "island": "Maui",
        "lat": 20.70820,
        "lon": -156.25720,
        "alt_m": 3050,
        "aperture": "3.67 m",
        "status": "space surveillance",
        "notes": "Air Force Maui Optical and Supercomputing site telescope",
    },
    {
        "facility": "ATLAS-HKO",
        "cluster": "Haleakala",
        "island": "Maui",
        "lat": 20.70780,
        "lon": -156.25820,
        "alt_m": 3050,
        "aperture": "0.5 m",
        "status": "active",
        "notes": "Asteroid Terrestrial-impact Last Alert System, Haleakala unit",
    },
    {
        "facility": "ATLAS-MLO",
        "cluster": "Mauna Loa",
        "island": "Hawai'i",
        "lat": 19.53600,
        "lon": -155.57600,
        "alt_m": 3397,
        "aperture": "0.5 m",
        "status": "active",
        "notes": "ATLAS unit at Mauna Loa Observatory; representative coordinate",
    },
]


SOURCES = [
    ("IRTF Maunakea telescope coordinates", "https://irtfweb.ifa.hawaii.edu/observing/telescopeCoordinates.php"),
    ("W. M. Keck Observatory", "https://www.keckobservatory.org/"),
    ("Subaru Telescope", "https://subarutelescope.org/en/about/telescope/"),
    ("Gemini North site", "https://www.gemini.edu/observing/telescopes-and-sites/sites"),
    ("Canada-France-Hawaii Telescope", "https://www.cfht.hawaii.edu/en/about/"),
    ("NASA IRTF", "https://irtfweb.ifa.hawaii.edu/"),
    ("UKIRT", "https://about.ifa.hawaii.edu/ukirt/"),
    ("James Clerk Maxwell Telescope", "https://www.eaobservatory.org/jcmt/"),
    ("Submillimeter Array", "https://lweb.cfa.harvard.edu/sma/"),
    ("NRAO VLBA", "https://science.nrao.edu/facilities/vlba"),
    ("DKIST / NSO", "https://nso.edu/telescopes/dki-solar-telescope/"),
    ("Pan-STARRS", "https://panstarrs.stsci.edu/"),
    ("UH Haleakala Observatory", "https://www.ifa.hawaii.edu/haleakalanew/"),
    ("ATLAS", "https://atlas.fallingstar.com/"),
    ("Air Force Maui Optical and Supercomputing site", "https://www.amostech.com/"),
]


R_EARTH_KM = 6371.0


def cluster_center(cluster: str) -> tuple[float, float]:
    rows = [row for row in TELESCOPES if row["cluster"] == cluster and row["status"] != "decommissioned"]
    return sum(row["lat"] for row in rows) / len(rows), sum(row["lon"] for row in rows) / len(rows)


CENTER_ANCHORS = [cluster_center("Maunakea"), cluster_center("Haleakala"), cluster_center("Mauna Loa")]
LAT0 = sum(lat for lat, _ in CENTER_ANCHORS) / len(CENTER_ANCHORS)
LON0 = sum(lon for _, lon in CENTER_ANCHORS) / len(CENTER_ANCHORS)


COLORS = {
    "Maunakea": "#1f77b4",
    "Haleakala": "#2ca02c",
    "Mauna Loa": "#ff7f0e",
}


def xy_km(lat: float, lon: float, lat0: float = LAT0, lon0: float = LON0) -> tuple[float, float]:
    x = R_EARTH_KM * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
    y = R_EARTH_KM * math.radians(lat - lat0)
    return x, y


def distance_km(a: dict, b: dict) -> float:
    lat1, lon1 = map(math.radians, (a["lat"], a["lon"]))
    lat2, lon2 = map(math.radians, (b["lat"], b["lon"]))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(h))


def escape_latex(value: object) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def enrich_rows() -> list[dict]:
    rows = []
    for row in TELESCOPES:
        new = dict(row)
        new["x_km"], new["y_km"] = xy_km(row["lat"], row["lon"])
        rows.append(new)
    return rows


def save_overview(rows: list[dict]) -> tuple[Path, Path]:
    pdf = OUT_FIG / "hawaii_telescope_overview_map.pdf"
    png = OUT_FIG / "hawaii_telescope_overview_map.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    for row in rows:
        marker = "x" if row["status"] == "decommissioned" else "o"
        ax.scatter(
            row["x_km"],
            row["y_km"],
            s=78,
            color=COLORS[row["cluster"]],
            edgecolor="black" if marker == "o" else None,
            marker=marker,
            zorder=3,
        )
    cluster_positions = {}
    for cluster in COLORS:
        active = [row for row in rows if row["cluster"] == cluster and row["status"] != "decommissioned"]
        cluster_positions[cluster] = (
            sum(row["x_km"] for row in active) / len(active),
            sum(row["y_km"] for row in active) / len(active),
        )
    for cluster, (x, y) in cluster_positions.items():
        ax.annotate(cluster, (x, y), xytext=(12, 12), textcoords="offset points", fontsize=10, weight="bold")
    ax.scatter(0, 0, marker="+", s=220, color="crimson", linewidths=2.0)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="black", label=cluster, markersize=8)
        for cluster, color in COLORS.items()
    ]
    handles.append(plt.Line2D([0], [0], marker="+", color="crimson", label="projection center", markersize=10, linestyle="None"))
    ax.legend(handles=handles, loc="best", frameon=True)
    ax.set_xlabel("x east of center (km)")
    ax.set_ylabel("y north of center (km)")
    ax.set_title("Major Hawaii telescope facilities in a local Cartesian frame")
    ax.grid(True, alpha=0.55, linewidth=0.65)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(row["x_km"] for row in rows) - 20, max(row["x_km"] for row in rows) + 20)
    ax.set_ylim(min(row["y_km"] for row in rows) - 20, max(row["y_km"] for row in rows) + 20)
    fig.tight_layout()
    fig.savefig(pdf)
    fig.savefig(png, dpi=240)
    plt.close(fig)
    return pdf, png


def save_zoom(rows: list[dict], cluster: str, title: str, stem: str) -> tuple[Path, Path]:
    pdf = OUT_FIG / f"{stem}.pdf"
    png = OUT_FIG / f"{stem}.png"
    chosen = [row for row in rows if row["cluster"] == cluster]
    latc = sum(row["lat"] for row in chosen) / len(chosen)
    lonc = sum(row["lon"] for row in chosen) / len(chosen)
    local = []
    for row in chosen:
        x, y = xy_km(row["lat"], row["lon"], latc, lonc)
        item = dict(row)
        item["zx_km"] = x
        item["zy_km"] = y
        local.append(item)
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    offsets = {
        "Keck I": (-95, 18),
        "Keck II": (18, 22),
        "Subaru": (-125, -28),
        "Gemini North": (18, -16),
        "CFHT": (18, 24),
        "NASA IRTF": (18, -42),
        "UKIRT": (18, -14),
        "UH 2.2 m": (18, 25),
        "UH 0.6 m": (-88, -33),
        "JCMT": (-95, 28),
        "SMA": (-40, -58),
        "VLBA Maunakea": (20, -10),
        "CSO": (35, 22),
        "DKIST": (-115, -20),
        "Pan-STARRS": (20, 18),
        "Faulkes North": (-128, 28),
        "AEOS": (18, 26),
        "ATLAS-HKO": (18, -38),
        "ATLAS-MLO": (20, 10),
    }
    for row in local:
        marker = "x" if row["status"] == "decommissioned" else "o"
        ax.scatter(
            row["zx_km"],
            row["zy_km"],
            s=88,
            color=COLORS[row["cluster"]],
            edgecolor="black" if marker == "o" else None,
            marker=marker,
            zorder=3,
        )
        dx, dy = offsets.get(row["facility"], (10, 10))
        label = f"{row['facility']}\n{row['aperture']}"
        if row["status"] in {"decommissioned", "space surveillance", "small / local"}:
            label += f"\n{row['status']}"
        ax.annotate(
            label,
            (row["zx_km"], row["zy_km"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.6,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.86, "pad": 1.3},
            zorder=4,
        )
    pad = 0.65 if cluster == "Maunakea" else 0.55
    ax.set_xlim(min(row["zx_km"] for row in local) - pad, max(row["zx_km"] for row in local) + pad)
    ax.set_ylim(min(row["zy_km"] for row in local) - pad, max(row["zy_km"] for row in local) + pad)
    ax.set_xlabel(f"x east of {cluster} zoom center (km)")
    ax.set_ylabel(f"y north of {cluster} zoom center (km)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.55, linewidth=0.65)
    fig.tight_layout()
    fig.savefig(pdf)
    fig.savefig(png, dpi=260)
    plt.close(fig)
    return pdf, png


def save_mauna_loa_zoom(rows: list[dict]) -> tuple[Path, Path]:
    pdf = OUT_FIG / "hawaii_telescope_zoom_mauna_loa_apertures.pdf"
    png = OUT_FIG / "hawaii_telescope_zoom_mauna_loa_apertures.png"
    chosen = [row for row in rows if row["cluster"] == "Mauna Loa"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for row in chosen:
        ax.scatter(row["x_km"], row["y_km"], s=90, color=COLORS[row["cluster"]], edgecolor="black")
        ax.annotate(f"{row['facility']}\n{row['aperture']}", (row["x_km"], row["y_km"]), xytext=(14, 10), textcoords="offset points", fontsize=8, bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.86})
    ax.set_xlim(chosen[0]["x_km"] - 6, chosen[0]["x_km"] + 6)
    ax.set_ylim(chosen[0]["y_km"] - 6, chosen[0]["y_km"] + 6)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.55)
    ax.set_xlabel("x east of center (km)")
    ax.set_ylabel("y north of center (km)")
    ax.set_title("Mauna Loa auxiliary site")
    fig.tight_layout()
    fig.savefig(pdf)
    fig.savefig(png, dpi=240)
    plt.close(fig)
    return pdf, png


def build_pdf() -> None:
    rows = enrich_rows()
    csv_path = OUT_DATA / "hawaii_telescope_sites_cartesian.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "facility",
                "cluster",
                "island",
                "lat",
                "lon",
                "alt_m",
                "x_km",
                "y_km",
                "aperture",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    overview_pdf, overview_png = save_overview(rows)
    maunakea_pdf, maunakea_png = save_zoom(
        rows,
        "Maunakea",
        "Maunakea summit zoom with apertures",
        "hawaii_telescope_zoom_maunakea_apertures",
    )
    haleakala_pdf, haleakala_png = save_zoom(
        rows,
        "Haleakala",
        "Haleakala summit zoom with apertures",
        "hawaii_telescope_zoom_haleakala_apertures",
    )
    mauna_loa_pdf, mauna_loa_png = save_mauna_loa_zoom(rows)

    table_rows = "\n".join(
        rf"{escape_latex(row['facility'])} & {escape_latex(row['cluster'])} & {escape_latex(row['island'])} & "
        rf"{row['lat']:.6f} & {row['lon']:.6f} & {row['x_km']:.1f} & {row['y_km']:.1f} & "
        rf"{escape_latex(row['aperture'])} & {escape_latex(row['status'])} & {escape_latex(row['notes'])} \\"
        for row in rows
    )
    source_rows = "\n".join(
        rf"\item {escape_latex(label)}: \url{{{url}}}" for label, url in SOURCES
    )
    by_facility = {row["facility"]: row for row in rows}
    distance_pairs = [
        ("Keck I", "Keck II"),
        ("Keck I", "Subaru"),
        ("Keck I", "Gemini North"),
        ("Keck I", "VLBA Maunakea"),
        ("DKIST", "Pan-STARRS"),
        ("DKIST", "AEOS"),
        ("Keck I", "DKIST"),
        ("Keck I", "ATLAS-MLO"),
        ("DKIST", "ATLAS-MLO"),
    ]
    distance_rows = "\n".join(
        rf"{escape_latex(a)} -- {escape_latex(b)} & {distance_km(by_facility[a], by_facility[b]):.2f} \\"
        for a, b in distance_pairs
    )

    tex = dedent(
        rf"""
        \documentclass[10pt]{{article}}
        \usepackage[a4paper,margin=1.25cm]{{geometry}}
        \usepackage{{fontspec}}
        \usepackage{{xeCJK}}
        \setmainfont{{Helvetica Neue}}
        \setsansfont{{Helvetica Neue}}
        \setCJKmainfont{{PingFang SC}}
        \setCJKmonofont{{PingFang SC}}
        \usepackage{{booktabs}}
        \usepackage{{longtable}}
        \usepackage{{tabularx}}
        \usepackage{{array}}
        \usepackage{{graphicx}}
        \usepackage{{xcolor}}
        \usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{{hyperref}}
        \usepackage{{caption}}
        \captionsetup{{font=small,labelfont=bf}}
        \setlength{{\parindent}}{{0pt}}
        \setlength{{\parskip}}{{0.38em}}
        \emergencystretch=2em
        \renewcommand{{\arraystretch}}{{1.08}}
        \newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}

        \begin{{document}}
        \begin{{center}}
        {{\LARGE \bfseries Hawaii Major Telescope Distribution}}\\[0.2em]
        {{\large Apertures, local coordinates, and summit-cluster zooms}}\\[0.4em]
        {{\small Generated 2026-05-14.  Major professional facilities and selected small survey telescopes; not a complete inventory of every auxiliary instrument.}}
        \end{{center}}

        \section*{{Coordinate convention / 坐标约定}}
        The global Cartesian frame is centered on the unweighted mean of three cluster centers:
        Maunakea, Haleakala, and Mauna Loa.  This avoids over-weighting Maunakea simply because it
        hosts many individual telescopes.  The projection center is
        \[
        \phi_0={LAT0:.6f}^\circ,\qquad \lambda_0={LON0:.6f}^\circ,
        \]
        i.e. center latitude ${abs(LAT0):.4f}^\circ$ N and center longitude ${abs(LON0):.4f}^\circ$ W.
        Coordinates are in km with $x$ positive east and $y$ positive north:
        \[
        x=R_\oplus\cos\phi_0\,(\lambda-\lambda_0),\qquad y=R_\oplus(\phi-\phi_0),
        \]
        with angular differences in radians and $R_\oplus=6371$ km.  Zoom panels for Maunakea and
        Haleakala use their own local zoom centers only for readability; Table~1 always reports the
        global $x,y$ coordinates.

        \begin{{figure}}[h]
        \centering
        \includegraphics[width=0.78\linewidth]{{{overview_pdf}}}
        \caption{{Island-scale distribution of major Hawaii telescope facilities.}}
        \end{{figure}}

        \section*{{Zoomed summit views with apertures}}
        \begin{{figure}}[h]
        \centering
        \includegraphics[width=0.49\linewidth]{{{maunakea_pdf}}}\hfill
        \includegraphics[width=0.49\linewidth]{{{haleakala_pdf}}}
        \caption{{Zoomed summit clusters.  Labels include representative apertures; ``x'' marks a decommissioned facility.}}
        \end{{figure}}

        \section*{{Summary}}
        Hawaii's major telescope distribution is dominated by two high-altitude summit clusters.
        Maunakea hosts the large night-time optical/IR and sub-mm facilities: Keck I/II, Subaru,
        Gemini North, CFHT, IRTF, UKIRT, JCMT, SMA, and the Maunakea VLBA antenna.  Haleakala hosts
        the solar and survey cluster, including DKIST, Pan-STARRS, Faulkes North, ATLAS-HKO, and the
        AEOS space-surveillance telescope.  A smaller Mauna Loa site is included for ATLAS-MLO.
        Proposed or politically unresolved facilities, such as TMT, are not plotted as active
        telescopes.

        \section*{{Table 1. Facility positions and apertures}}
        \begingroup
        \scriptsize
        \setlength{{\tabcolsep}}{{2.1pt}}
        \begin{{longtable}}{{p{{1.9cm}}p{{1.45cm}}p{{1.0cm}}rrrrp{{1.35cm}}p{{1.5cm}}p{{4.15cm}}}}
        \toprule
        Facility & Cluster & Island & Lat. & Lon. & $x$ & $y$ & Aperture & Status & Notes \\
        & & & deg & deg & km & km & & & \\
        \midrule
        \endfirsthead
        \toprule
        Facility & Cluster & Island & Lat. & Lon. & $x$ & $y$ & Aperture & Status & Notes \\
        & & & deg & deg & km & km & & & \\
        \midrule
        \endhead
        {table_rows}
        \bottomrule
        \end{{longtable}}
        \endgroup
        \normalsize

        \section*{{Useful separations}}
        \small
        \begin{{tabularx}}{{0.78\linewidth}}{{Yr}}
        \toprule
        Pair & Great-circle separation (km) \\
        \midrule
        {distance_rows}
        \bottomrule
        \end{{tabularx}}
        \normalsize

        \section*{{Caveats}}
        The Maunakea coordinates for individual optical/IR telescopes are precise enough for
        engineering-scale layout work, but compact arrays such as SMA are represented by an array
        center rather than every antenna pad.  Some statuses change on policy timescales; this note
        marks CSO as decommissioned and does not treat TMT as an active facility.  Military/space
        surveillance facilities are included only where they are large optical telescopes relevant to
        aperture geography.

        \section*{{Sources}}
        \small
        \begin{{itemize}}
        {source_rows}
        \end{{itemize}}
        \normalsize
        \end{{document}}
        """
    ).strip() + "\n"

    tex_path = OUT_PDF / "hawaii_telescope_distribution.tex"
    tex_path.write_text(tex)
    print(csv_path)
    print(overview_pdf)
    print(overview_png)
    print(maunakea_pdf)
    print(maunakea_png)
    print(haleakala_pdf)
    print(haleakala_png)
    print(mauna_loa_pdf)
    print(mauna_loa_png)
    print(tex_path)


if __name__ == "__main__":
    build_pdf()
