from __future__ import annotations

import json
from pathlib import Path

import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_existing_telescope_ngc_sources_noiseaware_p1 as noiseaware


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"


def case_from_layout(path: Path) -> aug.NetworkCase:
    payload = json.loads(path.read_text())
    telescopes = [
        aug.Telescope(
            station["name"],
            float(station["x_km"]),
            float(station["y_km"]),
            float(station["diameter_m"]),
            bool(station["is_added"]),
        )
        for station in payload["stations"]
    ]
    return aug.NetworkCase(
        key="maunakea_plus8_ngc4151_opt",
        title="Maunakea optical core + eight optimized 5 m outstations",
        latitude_deg=19.8250,
        center_latlon=(19.8250, -155.4720),
        telescopes=telescopes,
        hub_km=tuple(payload["hub_km"]),
        optimization_score=float(payload["metrics"]["score"]),
    )


def main() -> None:
    layout = OUTFIG / "maunakea_ngc4151_optimized_plus8_layout.json"
    case = case_from_layout(layout)
    stats = noiseaware.run_case(case, ngc.NGC4151, q_values=[0.25, 0.50, 0.75])
    out = OUTFIG / "maunakea_ngc4151_optimized_plus8_noiseaware_p1_summary.json"
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(stats["figure_pdf"])
    print(stats["figure_png"])
    print(out)
    print(json.dumps(stats["metrics_by_q"], indent=2))


if __name__ == "__main__":
    main()
