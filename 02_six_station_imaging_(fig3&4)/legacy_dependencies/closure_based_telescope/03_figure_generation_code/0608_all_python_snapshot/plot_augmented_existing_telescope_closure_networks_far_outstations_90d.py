from __future__ import annotations

import json
from pathlib import Path

import plot_augmented_existing_telescope_closure_networks as aug


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)


def case_from_stats(path: Path, key_suffix: str = "_90d") -> aug.NetworkCase:
    stats = json.loads(path.read_text())
    telescopes = [
        aug.Telescope(
            station["name"],
            station["x_km"],
            station["y_km"],
            station["diameter_m"],
            station["is_added"],
        )
        for station in stats["stations"]
    ]
    return aug.NetworkCase(
        key=f"{stats['case']}{key_suffix}",
        title=f"{stats['title']} (90-day integration)",
        latitude_deg=stats["latitude_deg"],
        center_latlon=tuple(stats["center_latlon"]),
        telescopes=telescopes,
        hub_km=tuple(stats["hub_km"]),
        optimization_score=stats["optimization_score"],
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
    aug.OBSERVING_DAYS = 90
    input_stats = [
        OUTFIG / "augmented_existing_telescope_maunakea_plus5_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus3_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus4_far_stats.json",
    ]
    summary = {}
    for stats_path in input_stats:
        case = case_from_stats(stats_path)
        print(f"simulating {case.key}")
        stats, image_pack, truth, axis_uas = aug.simulate_case(case)
        pdf, png = aug.plot_case(case, stats, image_pack, truth, axis_uas)
        stats["figure_pdf"] = str(pdf)
        stats["figure_png"] = str(png)
        out_stats = OUTFIG / f"augmented_existing_telescope_{case.key}_stats.json"
        out_stats.write_text(json.dumps(stats, indent=2) + "\n")
        summary[case.key] = compact_stats(stats)
        print(pdf)
        print(png)
        print(out_stats)
        print(json.dumps(compact_stats(stats), indent=2))
    summary_path = OUTFIG / "augmented_existing_telescope_far_outstations_90d_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
