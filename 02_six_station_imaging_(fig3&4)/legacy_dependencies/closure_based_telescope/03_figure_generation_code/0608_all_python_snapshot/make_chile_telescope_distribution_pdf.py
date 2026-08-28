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


# Site coordinates are representative site centers.  Co-located instruments
# share a site row unless their separation is relevant at the km level.
SITES = [
    {
        "site": "ALMA AOS / Llano de Chajnantor",
        "region": "Antofagasta",
        "lat": -23.0290,
        "lon": -67.7550,
        "alt_m": 5000,
        "facilities": "ALMA: 54 x 12 m and 12 x 7 m antennas; baselines up to about 16 km",
    },
    {
        "site": "APEX / Chajnantor",
        "region": "Antofagasta",
        "lat": -23.0057778,
        "lon": -67.7591667,
        "alt_m": 5105,
        "facilities": "APEX: 12 m sub-mm telescope",
    },
    {
        "site": "Cerro Armazones",
        "region": "Antofagasta",
        "lat": -24.589444,
        "lon": -70.191667,
        "alt_m": 3046,
        "facilities": "ESO ELT: 39 m optical/IR telescope, under construction",
    },
    {
        "site": "Cerro Paranal",
        "region": "Antofagasta",
        "lat": -24.627222,
        "lon": -70.404167,
        "alt_m": 2635,
        "facilities": "VLT/VLTI: 4 x 8.2 m UTs, 4 x 1.8 m ATs; VISTA 4.1 m; VST 2.6 m",
    },
    {
        "site": "La Silla",
        "region": "Coquimbo",
        "lat": -29.261056,
        "lon": -70.731750,
        "alt_m": 2400,
        "facilities": "ESO 3.6 m, NTT 3.58 m, MPG/ESO 2.2 m, and smaller survey telescopes",
    },
    {
        "site": "Las Campanas / Magellan ridge",
        "region": "Atacama",
        "lat": -29.015000,
        "lon": -70.691700,
        "alt_m": 2380,
        "facilities": "Magellan Baade/Clay: 2 x 6.5 m; du Pont 2.5 m; Swope 1.0 m",
    },
    {
        "site": "Las Campanas / GMT site",
        "region": "Atacama",
        "lat": -29.048333,
        "lon": -70.683611,
        "alt_m": 2516,
        "facilities": "Giant Magellan Telescope: seven 8.4 m segments; 24.5-25.4 m class",
    },
    {
        "site": "Cerro Tololo",
        "region": "Coquimbo",
        "lat": -30.169000,
        "lon": -70.804000,
        "alt_m": 2200,
        "facilities": "CTIO: Blanco 4 m, SMARTS 1.5/1.3/1.0/0.9 m, smaller telescopes",
    },
    {
        "site": "Cerro Pachon",
        "region": "Coquimbo",
        "lat": -30.240750,
        "lon": -70.736693,
        "alt_m": 2720,
        "facilities": "Gemini South 8.1 m and SOAR 4.1 m, separated by less than about 1 km",
    },
    {
        "site": "El Penon / Rubin Observatory",
        "region": "Coquimbo",
        "lat": -30.244633,
        "lon": -70.749417,
        "alt_m": 2647,
        "facilities": "Vera C. Rubin Observatory: 8.4 m primary, about 6.7 m effective aperture",
    },
]


TELESCOPES = [
    ("ALMA", "Llano de Chajnantor", "54 x 12 m + 12 x 7 m", "mm/sub-mm array; maximum baselines about 16 km"),
    ("APEX", "Chajnantor", "12 m", "single-dish sub-mm telescope"),
    ("ESO ELT", "Cerro Armazones", "39 m", "optical/IR; under construction"),
    ("VLT UTs", "Cerro Paranal", "4 x 8.2 m", "optical/IR; VLTI-capable unit telescopes"),
    ("VLTI ATs", "Cerro Paranal", "4 x 1.8 m", "relocatable auxiliary telescopes"),
    ("VISTA", "Cerro Paranal", "4.1 m", "wide-field IR survey telescope"),
    ("VST", "Cerro Paranal", "2.6 m", "wide-field optical survey telescope"),
    ("ESO 3.6 m", "La Silla", "3.6 m", "HARPS host telescope"),
    ("NTT", "La Silla", "3.58 m", "New Technology Telescope"),
    ("MPG/ESO 2.2 m", "La Silla", "2.2 m", "wide-field optical imaging and spectroscopy"),
    ("Magellan Baade/Clay", "Las Campanas", "2 x 6.5 m", "twin optical/IR telescopes"),
    ("du Pont", "Las Campanas", "2.5 m", "optical telescope"),
    ("Giant Magellan Telescope", "Las Campanas", "24.5-25.4 m class", "seven 8.4 m primary segments; under construction"),
    ("Blanco", "Cerro Tololo", "4 m", "DECam host telescope"),
    ("SMARTS", "Cerro Tololo", "1.5/1.3/1.0/0.9 m", "small-telescope network"),
    ("Gemini South", "Cerro Pachon", "8.1 m", "optical/IR telescope"),
    ("SOAR", "Cerro Pachon", "4.1 m", "optical/IR telescope"),
    ("Vera C. Rubin Observatory", "El Penon", "8.4 m primary; 6.7 m effective", "wide-field survey telescope"),
]


SOURCES = [
    ("ESO Paranal / VLT", "https://www.eso.org/public/teles-instr/paranal-observatory/vlt/"),
    ("ESO VISTA", "https://www.eso.org/public/teles-instr/paranal-observatory/surveytelescopes/vista/"),
    ("ESO VST", "https://www.eso.org/public/teles-instr/paranal-observatory/surveytelescopes/vst/"),
    ("ESO ELT", "https://elt.eso.org/"),
    ("ESO La Silla", "https://www.eso.org/public/teles-instr/las-silla/"),
    ("ALMA overview", "https://www.eso.org/public/teles-instr/alma/"),
    ("APEX overview", "https://www.apex-telescope.org/ns/apex/"),
    ("NOIRLab / CTIO", "https://noirlab.edu/public/programs/ctio/"),
    ("Gemini South site information", "https://www.gemini.edu/observing/telescopes-and-sites/sites"),
    ("SOAR Telescope", "https://noirlab.edu/public/programs/ctio/soar-telescope/"),
    ("Rubin Observatory numbers", "https://rubinobservatory.org/explore/how-rubin-works/numbers"),
    ("Las Campanas Observatory", "https://obs.carnegiescience.edu/our-facilities/las-campanas-observatory"),
    ("Giant Magellan Telescope", "https://giantmagellan.org/"),
]


R_EARTH_KM = 6371.0
LAT0 = sum(site["lat"] for site in SITES) / len(SITES)
LON0 = sum(site["lon"] for site in SITES) / len(SITES)

SITE_SHORT_LABELS = {
    "ALMA AOS / Llano de Chajnantor": "ALMA AOS",
    "APEX / Chajnantor": "APEX",
    "Cerro Armazones": "Armazones / ELT",
    "Cerro Paranal": "Paranal / VLT",
    "La Silla": "La Silla",
    "Las Campanas / Magellan ridge": "LCO / Magellan",
    "Las Campanas / GMT site": "LCO / GMT",
    "Cerro Tololo": "Cerro Tololo",
    "Cerro Pachon": "Cerro Pachon",
    "El Penon / Rubin Observatory": "Rubin / El Penon",
}

SITE_APERTURE_LABELS = {
    "ALMA AOS / Llano de Chajnantor": "ALMA AOS\n54 x 12 m + 12 x 7 m",
    "APEX / Chajnantor": "APEX\n12 m",
    "Cerro Armazones": "Armazones / ELT\n39 m",
    "Cerro Paranal": "Paranal / VLT\n4 x 8.2 m; ATs 1.8 m",
    "La Silla": "La Silla\n3.6 m; NTT 3.58 m; 2.2 m",
    "Las Campanas / Magellan ridge": "LCO / Magellan\n2 x 6.5 m",
    "Las Campanas / GMT site": "LCO / GMT\n24.5-25.4 m class",
    "Cerro Tololo": "Cerro Tololo\nBlanco 4 m",
    "Cerro Pachon": "Cerro Pachon\nGemini S 8.1 m; SOAR 4.1 m",
    "El Penon / Rubin Observatory": "Rubin / El Penon\n8.4 m primary; 6.7 m eff.",
}

SITE_LABEL_OFFSETS = {
    "ALMA AOS / Llano de Chajnantor": (8, -24),
    "APEX / Chajnantor": (8, 9),
    "Cerro Armazones": (10, 12),
    "Cerro Paranal": (-68, -8),
    "La Silla": (8, -18),
    "Las Campanas / Magellan ridge": (8, 13),
    "Las Campanas / GMT site": (8, -10),
    "Cerro Tololo": (8, -17),
    "Cerro Pachon": (8, 12),
    "El Penon / Rubin Observatory": (8, -33),
}

ZOOM_LABEL_OFFSETS = {
    "ALMA AOS / Llano de Chajnantor": (12, -40),
    "APEX / Chajnantor": (12, 12),
    "Cerro Armazones": (10, 15),
    "Cerro Paranal": (-108, -30),
    "Las Campanas / Magellan ridge": (12, 15),
    "Las Campanas / GMT site": (12, -36),
    "La Silla": (12, -35),
    "Cerro Tololo": (-108, -34),
    "Cerro Pachon": (12, 16),
    "El Penon / Rubin Observatory": (12, -48),
}


def site_xy_km(lat: float, lon: float) -> tuple[float, float]:
    x = R_EARTH_KM * math.cos(math.radians(LAT0)) * math.radians(lon - LON0)
    y = R_EARTH_KM * math.radians(lat - LAT0)
    return x, y


def km_distance(a: dict, b: dict) -> float:
    lat1, lon1 = map(math.radians, (a["lat"], a["lon"]))
    lat2, lon2 = map(math.radians, (b["lat"], b["lon"]))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(h))


def escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
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
    return "".join(replacements.get(ch, ch) for ch in text)


def annotate_site(ax, row: dict, *, with_aperture: bool = False) -> None:
    label = SITE_APERTURE_LABELS[row["site"]] if with_aperture else SITE_SHORT_LABELS[row["site"]]
    offsets = ZOOM_LABEL_OFFSETS if with_aperture else SITE_LABEL_OFFSETS
    dx, dy = offsets[row["site"]]
    ax.annotate(
        label,
        (row["x_km"], row["y_km"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8 if with_aperture else 8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.84, "pad": 1.5} if with_aperture else None,
        zorder=4,
    )


def save_zoom_map(rows: list[dict], title: str, stem: str, colors: dict[str, str]) -> tuple[Path, Path]:
    fig_pdf = OUT_FIG / f"{stem}.pdf"
    fig_png = OUT_FIG / f"{stem}.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for row in rows:
        ax.scatter(
            row["x_km"],
            row["y_km"],
            s=95,
            color=colors[row["region"]],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        annotate_site(ax, row, with_aperture=True)
    xmin = min(row["x_km"] for row in rows) - 55
    xmax = max(row["x_km"] for row in rows) + 55
    ymin = min(row["y_km"] for row in rows) - 55
    ymax = max(row["y_km"] for row in rows) + 55
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x east of center (km)")
    ax.set_ylabel("y north of center (km)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.65, alpha=0.55)
    fig.tight_layout()
    fig.savefig(fig_pdf)
    fig.savefig(fig_png, dpi=240)
    plt.close(fig)
    return fig_pdf, fig_png


def make_outputs() -> None:
    enriched = []
    for site in SITES:
        x, y = site_xy_km(site["lat"], site["lon"])
        row = dict(site)
        row["x_km"] = x
        row["y_km"] = y
        enriched.append(row)

    csv_path = OUT_DATA / "chile_observatory_sites_cartesian.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["site", "region", "lat", "lon", "alt_m", "x_km", "y_km", "facilities"],
        )
        writer.writeheader()
        for row in enriched:
            writer.writerow(row)

    fig_pdf = OUT_FIG / "chile_observatory_cartesian_map.pdf"
    fig_png = OUT_FIG / "chile_observatory_cartesian_map.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 8.2))
    colors = {
        "Antofagasta": "#1f77b4",
        "Atacama": "#ff7f0e",
        "Coquimbo": "#2ca02c",
    }
    for row in enriched:
        ax.scatter(row["x_km"], row["y_km"], s=90, color=colors[row["region"]], edgecolor="black", zorder=3)
        annotate_site(ax, row, with_aperture=False)
    ax.scatter(0, 0, marker="+", s=220, color="crimson", linewidths=2.0, label="projection center")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="black", label=region, markersize=8)
        for region, color in colors.items()
    ]
    handles.append(plt.Line2D([0], [0], marker="+", color="crimson", label="projection center", markersize=10, linestyle="None"))
    ax.legend(handles=handles, loc="lower right", frameon=True)
    ax.set_xlabel("x east of center (km)")
    ax.set_ylabel("y north of center (km)")
    ax.set_title("Major Chilean observatory sites in a local Cartesian frame")
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.axvline(0, color="0.45", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-120, 260)
    ax.set_ylim(-360, 525)
    fig.tight_layout()
    fig.savefig(fig_pdf)
    fig.savefig(fig_png, dpi=220)
    plt.close(fig)

    atacama_zoom_rows = [
        row
        for row in enriched
        if row["region"] in {"Antofagasta", "Atacama"}
    ]
    coquimbo_zoom_rows = [
        row
        for row in enriched
        if row["region"] == "Coquimbo"
    ]
    atacama_zoom_pdf, atacama_zoom_png = save_zoom_map(
        atacama_zoom_rows,
        "Atacama / northern Chile zoom with representative apertures",
        "chile_observatory_zoom_atacama_apertures",
        colors,
    )
    coquimbo_zoom_pdf, coquimbo_zoom_png = save_zoom_map(
        coquimbo_zoom_rows,
        "Coquimbo zoom with representative apertures",
        "chile_observatory_zoom_coquimbo_apertures",
        colors,
    )

    key_pairs = [
        ("ALMA AOS / Llano de Chajnantor", "APEX / Chajnantor"),
        ("Cerro Paranal", "Cerro Armazones"),
        ("Las Campanas / Magellan ridge", "Las Campanas / GMT site"),
        ("Cerro Tololo", "Cerro Pachon"),
        ("Cerro Pachon", "El Penon / Rubin Observatory"),
        ("La Silla", "Las Campanas / Magellan ridge"),
        ("Cerro Paranal", "ALMA AOS / Llano de Chajnantor"),
        ("Cerro Paranal", "La Silla"),
        ("La Silla", "Cerro Tololo"),
    ]
    by_name = {row["site"]: row for row in SITES}
    distance_rows = [
        (a, b, km_distance(by_name[a], by_name[b]))
        for a, b in key_pairs
    ]

    site_rows_tex = "\n".join(
        rf"{escape_latex(row['site'])} & {escape_latex(row['region'])} & "
        rf"{row['lat']:.5f} & {row['lon']:.5f} & {row['x_km']:.1f} & {row['y_km']:.1f} & "
        rf"{row['alt_m']} & {escape_latex(row['facilities'])} \\"
        for row in enriched
    )
    telescope_rows_tex = "\n".join(
        rf"{escape_latex(name)} & {escape_latex(site)} & {escape_latex(aperture)} & {escape_latex(notes)} \\"
        for name, site, aperture, notes in TELESCOPES
    )
    distance_rows_tex = "\n".join(
        rf"{escape_latex(a)} -- {escape_latex(b)} & {distance:.1f} \\"
        for a, b, distance in distance_rows
    )
    source_rows_tex = "\n".join(
        rf"\item {escape_latex(label)}: \url{{{url}}}"
        for label, url in SOURCES
    )

    fig_pdf_tex = str(fig_pdf)
    atacama_zoom_pdf_tex = str(atacama_zoom_pdf)
    coquimbo_zoom_pdf_tex = str(coquimbo_zoom_pdf)

    tex = dedent(
        rf"""
        \documentclass[10pt]{{article}}
        \usepackage[a4paper,margin=1.35cm]{{geometry}}
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
        \setlength{{\parskip}}{{0.42em}}
        \emergencystretch=2em
        \renewcommand{{\arraystretch}}{{1.12}}
        \newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}
        \newcolumntype{{R}}{{>{{\raggedleft\arraybackslash}}p{{1.1cm}}}}

        \begin{{document}}
        \begin{{center}}
        {{\LARGE \bfseries Chile Major Telescope Distribution}}\\[0.2em]
        {{\large Apertures, site coordinates, and a local Cartesian frame}}\\[0.4em]
        {{\small Generated 2026-05-14.  Selected major professional facilities; not an exhaustive inventory of all small telescopes.}}
        \end{{center}}

        \section*{{Coordinate convention / 坐标约定}}
        We use a local Cartesian frame centered on the unweighted mean of the ten site centers listed in Table~1.
        The projection center is
        \[
        \phi_0={LAT0:.6f}^\circ,\qquad \lambda_0={LON0:.6f}^\circ,
        \]
        i.e. center latitude $27.3230^\circ$ S and center longitude $70.0507^\circ$ W.
        Coordinates are reported in km with $x$ positive east and $y$ positive north:
        \[
        x=R_\oplus\cos\phi_0\,(\lambda-\lambda_0),\qquad
        y=R_\oplus(\phi-\phi_0),
        \]
        where angular differences are in radians and $R_\oplus=6371$ km.  This plane is meant for
        layout intuition and first-pass baseline estimates; for sub-meter metrology one should use
        WGS84/ECEF coordinates for the individual telescope pads.

        \begin{{figure}}[h]
        \centering
        \includegraphics[width=0.82\linewidth]{{{fig_pdf_tex}}}
        \caption{{Major Chilean observatory sites in the Cartesian frame used in this note.}}
        \end{{figure}}

        \section*{{Zoomed views with apertures}}
        The two zoom panels below label each site by its representative collecting aperture.  Here
        ``Atacama'' is used in the practical astronomical sense of the northern desert site cluster,
        so it includes both the Antofagasta sites and the Las Campanas sites in the administrative
        Atacama Region.

        \begin{{figure}}[h]
        \centering
        \includegraphics[width=0.49\linewidth]{{{atacama_zoom_pdf_tex}}}\hfill
        \includegraphics[width=0.49\linewidth]{{{coquimbo_zoom_pdf_tex}}}
        \caption{{Zoomed Cartesian maps for the northern Atacama-desert cluster and the Coquimbo cluster.
        Each station label includes the representative aperture or array collecting elements.}}
        \end{{figure}}

        \section*{{Summary}}
        Chile's major sites form three useful geographic groups.  The northern Atacama group contains
        Chajnantor/APEX/ALMA and the Paranal-Armazones pair; the latter are separated by only about
        22 km.  The central-southern optical group contains La Silla, Las Campanas, Cerro Tololo,
        Cerro Pachon, and Rubin/El Penon; the Tololo-Pachon separation is about 10 km, while
        La Silla to Las Campanas is about 28 km.  The largest optical/IR apertures are the ELT
        on Armazones, the GMT site at Las Campanas, the VLT UTs at Paranal, Gemini South and Rubin
        on the Pachon ridge system, and the twin Magellan telescopes at Las Campanas.

        \section*{{Table 1. Site positions and representative facilities}}
        \begingroup
        \scriptsize
        \setlength{{\tabcolsep}}{{2.1pt}}
        \begin{{longtable}}{{p{{2.45cm}}p{{1.50cm}}rrrrp{{0.68cm}}p{{4.95cm}}}}
        \toprule
        Site & Region & Lat. & Lon. & $x$ & $y$ & Alt. & Representative facilities \\
        & & deg & deg & km & km & m & \\
        \midrule
        \endfirsthead
        \toprule
        Site & Region & Lat. & Lon. & $x$ & $y$ & Alt. & Representative facilities \\
        & & deg & deg & km & km & m & \\
        \midrule
        \endhead
        {site_rows_tex}
        \bottomrule
        \end{{longtable}}
        \endgroup
        \normalsize

        \section*{{Table 2. Aperture inventory used in this summary}}
        \small
        \setlength{{\tabcolsep}}{{3.5pt}}
        \begin{{longtable}}{{p{{3.4cm}}p{{3.0cm}}p{{2.8cm}}p{{6.5cm}}}}
        \toprule
        Facility & Site & Aperture / collecting elements & Notes \\
        \midrule
        \endfirsthead
        \toprule
        Facility & Site & Aperture / collecting elements & Notes \\
        \midrule
        \endhead
        {telescope_rows_tex}
        \bottomrule
        \end{{longtable}}
        \normalsize

        \section*{{Useful separations}}
        \small
        \begin{{tabularx}}{{0.78\linewidth}}{{Yr}}
        \toprule
        Site pair & Great-circle separation (km) \\
        \midrule
        {distance_rows_tex}
        \bottomrule
        \end{{tabularx}}
        \normalsize

        \section*{{Caveats}}
        Coordinates are rounded site centers, not engineering survey coordinates for every pier, pad,
        or antenna.  ALMA is an extended array and its listed coordinate is a representative Array
        Operations Site / Chajnantor center; actual antenna pads span up to about 16 km.  Cerro
        Pachon, SOAR, Gemini South, and Rubin are all close enough that a telescope-level layout
        should use exact pad coordinates rather than the simplified site centers above.

        \section*{{Sources}}
        \small
        Official observatory pages were used for apertures, site descriptions, and status.  Coordinate
        values were cross-checked against observatory coordinate pages where available and rounded for
        this engineering-summary table.
        \begin{{itemize}}
        {source_rows_tex}
        \end{{itemize}}
        \normalsize

        \end{{document}}
        """
    ).strip() + "\n"

    tex_path = OUT_PDF / "chile_telescope_distribution.tex"
    tex_path.write_text(tex)
    print(csv_path)
    print(fig_pdf)
    print(fig_png)
    print(tex_path)


if __name__ == "__main__":
    make_outputs()
