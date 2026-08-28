from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
from make_chile_optical_zoom_panels import SITES as CHILE_SITES
from make_hawaii_optical_overview_figure import CLUSTER_CENTER, VISIBLE_400_800


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)


def maunakea_plus5_case(rng: np.random.Generator) -> aug.NetworkCase:
    center = CLUSTER_CENTER["Maunakea"]
    apertures = {
        "Keck I": 10.0,
        "Keck II": 10.0,
        "Subaru": 8.2,
        "Gemini North": 8.1,
        "CFHT": 3.6,
    }
    existing = []
    for name, lat, lon, _, cluster in VISIBLE_400_800:
        if cluster != "Maunakea" or name not in apertures:
            continue
        x, y = aug.xy_km(lat, lon, center)
        existing.append(aug.Telescope(name, x, y, apertures[name], False))
    case = aug.optimize_added_telescopes(
        existing,
        n_added=5,
        center=center,
        latitude_deg=center[0],
        radius_range=(5.0, 10.0),
        max_target_g_lambda=42.0,
        rng=rng,
        n_trials=4200,
    )
    return aug.NetworkCase(
        key="maunakea_plus5",
        title="Maunakea optical core + five 5 m outstations",
        latitude_deg=center[0],
        center_latlon=center,
        telescopes=case.telescopes,
        hub_km=case.hub_km,
        optimization_score=case.optimization_score,
    )


def ctio_plus_case(rng: np.random.Generator, n_added: int) -> aug.NetworkCase:
    wanted = {
        "CTIO / Blanco": 4.0,
        "Gemini South": 8.1,
        "SOAR": 4.1,
        "Rubin / El Penon": 6.7,
    }
    rows = [row for row in CHILE_SITES if row[0] in wanted]
    center = (
        sum(row[1] for row in rows) / len(rows),
        sum(row[2] for row in rows) / len(rows),
    )
    existing = []
    for name, lat, lon, _ in rows:
        x, y = aug.xy_km(lat, lon, center)
        existing.append(aug.Telescope(name.replace(" / El Penon", ""), x, y, wanted[name], False))
    case = aug.optimize_added_telescopes(
        existing,
        n_added=n_added,
        center=center,
        latitude_deg=center[0],
        radius_range=(5.0, 10.0),
        max_target_g_lambda=42.0,
        rng=rng,
        n_trials=3600 if n_added == 3 else 4600,
    )
    return aug.NetworkCase(
        key=f"ctio_plus{n_added}",
        title=f"CTIO/Pachon/Rubin core + {n_added} five-meter outstations",
        latitude_deg=center[0],
        center_latlon=center,
        telescopes=case.telescopes,
        hub_km=case.hub_km,
        optimization_score=case.optimization_score,
    )


def compact_stats(stats: dict) -> dict:
    return {
        "title": stats["title"],
        "n_station": stats["n_station"],
        "n_baseline": stats["n_baseline"],
        "n_closure": stats["n_closure"],
        "closure_rank_share": stats["closure_rank_share"],
        "baseline_min_km": stats["baseline_min_km"],
        "baseline_median_km": stats["baseline_median_km"],
        "baseline_max_km": stats["baseline_max_km"],
        "station_link_eff_min": stats["station_link_eff_min"],
        "station_link_eff_max": stats["station_link_eff_max"],
        "coverage_400nm_half_range_g_lambda": stats["coverage_400nm_half_range_g_lambda"],
        "coverage_800nm_half_range_g_lambda": stats["coverage_800nm_half_range_g_lambda"],
        "metrics": stats["metrics"],
        "hub_km": stats["hub_km"],
        "figure_pdf": stats["figure_pdf"],
        "figure_png": stats["figure_png"],
    }


def main() -> None:
    rng = np.random.default_rng(aug.RNG_SEED + 9001)
    cases = [
        maunakea_plus5_case(rng),
        ctio_plus_case(rng, 3),
        ctio_plus_case(rng, 4),
    ]
    summary = {}
    for case in cases:
        print(f"simulating {case.key}")
        stats, image_pack, truth, axis_uas = aug.simulate_case(case)
        pdf, png = aug.plot_case(case, stats, image_pack, truth, axis_uas)
        stats["figure_pdf"] = str(pdf)
        stats["figure_png"] = str(png)
        stats_path = OUTFIG / f"augmented_existing_telescope_{case.key}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        summary[case.key] = compact_stats(stats)
        print(pdf)
        print(png)
        print(stats_path)
        print(json.dumps(compact_stats(stats), indent=2))
    summary_path = OUTFIG / "augmented_existing_telescope_more_outstations_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
