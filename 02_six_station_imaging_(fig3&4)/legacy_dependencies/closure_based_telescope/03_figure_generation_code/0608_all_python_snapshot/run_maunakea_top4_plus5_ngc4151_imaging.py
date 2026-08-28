from __future__ import annotations

import json
import os
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
        key=payload.get("case_key", path.stem.replace("_layout", "")),
        title=payload.get("case_title", "Maunakea top-four core + five 5 m outstations"),
        latitude_deg=19.8250,
        center_latlon=(19.8250, -155.4720),
        telescopes=telescopes,
        hub_km=tuple(payload["hub_km"]),
        optimization_score=float(payload["metrics"]["score"]),
    )


def main() -> None:
    noiseaware.wt.SNR_BOOST = float(os.environ.get("AUGMENTED_SNR_BOOST", "1.0"))
    layout = Path(os.environ.get("MAUNAKEA_LAYOUT", str(OUTFIG / "maunakea_top4_plus5_ngc4151_layout.json")))
    case = case_from_layout(layout)
    suffix = os.environ.get("CASE_KEY_SUFFIX", "")
    if suffix:
        case = aug.NetworkCase(
            key=f"{case.key}{suffix}",
            title=case.title,
            latitude_deg=case.latitude_deg,
            center_latlon=case.center_latlon,
            telescopes=case.telescopes,
            hub_km=case.hub_km,
            optimization_score=case.optimization_score,
        )
    q_values = [float(q) for q in os.environ.get("Q_VALUES", "0").split(",")]
    stats = noiseaware.run_case(case, ngc.NGC4151, q_values=q_values)
    out = OUTFIG / f"{case.key}_uniform_p1_summary.json"
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(stats["figure_pdf"])
    print(stats["figure_png"])
    print(out)
    print(json.dumps(stats["metrics_by_q"], indent=2))


if __name__ == "__main__":
    main()
