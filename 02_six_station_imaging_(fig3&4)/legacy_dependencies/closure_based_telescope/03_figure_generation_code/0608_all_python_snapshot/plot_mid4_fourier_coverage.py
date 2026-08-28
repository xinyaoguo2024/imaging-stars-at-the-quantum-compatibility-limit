from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
SUMMARY = OUTFIG / "mid4_lowmax_r1p6_3_5_6_amp_closure_rml_ngc4151_30d_ampw3_n80_summary.json"


def case_from_summary(item: dict) -> aug.NetworkCase:
    telescopes = [
        aug.Telescope(
            station["name"],
            float(station["x_km"]),
            float(station["y_km"]),
            float(station["diameter_m"]),
            bool(station["is_added"]),
        )
        for station in item["stations"]
    ]
    return aug.NetworkCase(
        key=item["case"],
        title=item["case"].replace("_", " "),
        latitude_deg=19.8250,
        center_latlon=(19.8250, -155.4720),
        telescopes=telescopes,
        hub_km=tuple(item["hub_km"]),
        optimization_score=0.0,
    )


def coverage_points(case: aug.NetworkCase, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stations, _, _, _ = aug.station_table_from_case(case)
    edges = base.edge_list(len(stations))
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    hour = aug.realnight_hour_angles(36, 600.0, 150.0)
    uu, vv = aug.project_enu_baselines(
        baselines,
        hour,
        wavelength_nm * 1e-9,
        latitude_deg=case.latitude_deg,
        declination_deg=ngc.NGC4151.dec_deg,
    )
    u = uu.reshape(-1) / 1e9
    v = vv.reshape(-1) / 1e9
    # Include Hermitian conjugate points because image reconstruction uses real sky brightness.
    u_full = np.concatenate([u, -u])
    v_full = np.concatenate([v, -v])
    q_full = np.sqrt(u_full**2 + v_full**2)
    return u_full, v_full, q_full


def plot_ring_zero_circles(ax: plt.Axes, max_radius: float) -> None:
    zeros = [1.32, 3.02, 4.73, 6.45]
    theta = np.linspace(0, 2 * np.pi, 256)
    for q in zeros:
        if q <= max_radius:
            ax.plot(q * np.cos(theta), q * np.sin(theta), "--", lw=0.75, color="0.55", alpha=0.75)
            ax.text(q / np.sqrt(2), q / np.sqrt(2), f"{q:g}", fontsize=5.3, color="0.35")


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    cases = [case_from_summary(item) for item in payload["results"]]
    metrics = {item["case"]: item["metric"] for item in payload["results"]}
    q_hists = {item["case"]: item["q_hist"] for item in payload["results"]}

    fig, axes = plt.subplots(len(cases), 4, figsize=(11.8, 2.65 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = axes[None, :]
    for row, case in enumerate(cases):
        stations, _, names, is_added = aug.station_table_from_case(case)
        metric = metrics[case.key]

        ax = axes[row, 0]
        ax.scatter(stations[~is_added, 0], stations[~is_added, 1], s=42, color="#1f77b4", label="existing")
        ax.scatter(stations[is_added, 0], stations[is_added, 1], s=52, marker="^", color="#d62728", label="remote")
        ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=70, marker="*", color="#ffb000", label="hub")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{case.key}\nBLR={metric['blr_corr']:.2f}, global={metric['global_corr']:.2f}")
        ax.set_xlabel("x east (km)")
        ax.set_ylabel("y north (km)")
        if row == 0:
            ax.legend(fontsize=6, loc="best")

        u400, v400, q400 = coverage_points(case, 400.0)
        u800, v800, q800 = coverage_points(case, 800.0)
        for col, zoom in ((1, None), (2, 12.0)):
            ax = axes[row, col]
            ax.scatter(u400, v400, s=3, alpha=0.38, color="#005f73", label="400 nm")
            ax.scatter(u800, v800, s=3, alpha=0.38, color="#ee9b00", label="800 nm")
            limit = zoom if zoom is not None else max(8.0, 1.05 * max(np.max(np.abs(u400)), np.max(np.abs(v400))))
            plot_ring_zero_circles(ax, limit)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.20, lw=0.5)
            ax.set_xlabel(r"$u$ (G$\lambda$)")
            ax.set_ylabel(r"$v$ (G$\lambda$)")
            ax.set_title("Fourier coverage" if zoom is None else r"zoom: $|u|,|v|<12$ G$\lambda$")
            if row == 0 and col == 1:
                ax.legend(fontsize=6, markerscale=2.5, loc="upper right")

        ax = axes[row, 3]
        labels = list(q_hists[case.key].keys())
        vals = list(q_hists[case.key].values())
        ax.bar(np.arange(len(vals)), vals, color="#669bbc")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=5.5)
        ax.set_ylabel("samples")
        ax.set_title("radial counts, 400-800 nm")
        for target in (1.32, 3.02, 4.73, 6.45):
            ax.axvline(np.interp(target, [0, 60], [0, len(vals) - 1]), color="0.75", lw=0.4, alpha=0.4)

    fig.suptitle(
        "Fourier coverage for the mid-baseline Hawaii+4 tests; dashed rings mark thin-BLR-ring visibility zeros",
        fontsize=10.2,
        weight="bold",
    )
    tag = "mid4_lowmax_r1p6_3_5_6_fourier_coverage"
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
