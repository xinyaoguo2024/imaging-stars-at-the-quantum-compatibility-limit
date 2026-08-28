from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import make_fig2_current_seed_diagnostics as diag


BUNDLE = Path(__file__).resolve().parents[2]
FIG_DIR = BUNDLE / "figures" / "diagnostics"
OUT = BUNDLE / "exploration" / "fig2_current_seed_diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def edge_fullcycle_fisher_for_sample(
    split: np.ndarray,
    tri: tuple[int, int, int],
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
    n_station: int,
) -> float:
    """Uniform edge-first GLS Fisher for one closure using the full cycle space."""
    edge_values = np.zeros(len(edges), dtype=float)
    for idx, (i, j) in enumerate(edges):
        edge_values[idx] = diag.split_sim.edge_pair_fisher_for_sample(
            i,
            j,
            split[i, j],
            split[j, i],
            total_modes=total_modes,
            u_station=u_station,
            eta=eta,
            station_noise=station_noise,
            nu_eff=nu_eff,
            edge_to_index=edge_to_index,
        )
    cycle_fisher = diag.base.closure_fisher_after_gauge_marginalization(
        np.diag(edge_values),
        q_basis,
        edges,
        n_station,
    )
    return diag.scalar_fisher_from_cycle_matrix(cycle_fisher, q_basis, edges, tri)


def per_wavelength_rows() -> list[dict[str, float | str]]:
    diag.fig_run.configure_good_runtime()
    case = diag.fig_run.scale_remote_coordinates(diag.hawaii3_compact_case.make_hawaii3_compact_remote_case())
    splits = diag.fig_run.make_split_matrices(case)
    diag.split_sim.configure()
    diag.fig_run.apply_sample_stress_runtime()

    stations, diameters, _names, _is_added = diag.fig_run.aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n = len(stations)
    edges = diag.base.edge_list(n)
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    q_basis = diag.base.orthonormal_cycle_basis(diag.base.root_cycle_basis(edges, n))
    fov_rad = 2.0 * diag.fig_run.aug.HALF_WIDTH_UAS * diag.base.UAS_TO_RAD
    effective_hub_dist = diag.fig_run.aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-diag.fig_run.aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n, diag.fig_run.EPS_STATION_RUN, dtype=float)
    direct_noise = np.full(n, diag.fig_run.EPS_STATION_RUN + diag.fig_run.EPS_DIRECT_EXTRA_RUN, dtype=float)
    hour_angles = diag.realnight_hour_angles(
        diag.fig_run.aug.N_TIME_WINDOWS,
        diag.fig_run.aug.EXPOSURE_S,
        diag.fig_run.aug.EXPOSURE_GAP_S,
    )

    lam_edges_nm = np.arange(
        diag.fig_run.aug.LAMBDA_MIN_NM,
        diag.fig_run.aug.LAMBDA_MAX_NM + 0.5 * diag.fig_run.aug.LAMBDA_STEP_NM,
        diag.fig_run.aug.LAMBDA_STEP_NM,
    )
    lam_edges_nm[-1] = diag.fig_run.aug.LAMBDA_MAX_NM

    rows: list[dict[str, float | str]] = []
    with diag.fig_run.morph.patched_variant(diag.fig_run.GOOD_VARIANT), diag.fig_run.ngc.patched_source(
        diag.fig_run.GOOD_SOURCE
    ):
        for band_idx, (lo_nm, hi_nm) in enumerate(zip(lam_edges_nm[:-1], lam_edges_nm[1:])):
            center_nm = float(math.sqrt(lo_nm * hi_nm))
            lam_m = center_nm * 1e-9
            freq = diag.base.C_LIGHT / lam_m
            freq_lo = diag.base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = diag.base.C_LIGHT / (lo_nm * 1e-9)
            total_modes = diag.fig_run.aug.EXPOSURE_S * diag.fig_run.OBSERVING_DAYS * (freq_hi - freq_lo)
            u_station = diag.fig_run.aug.station_u_modes(freq, diameters)
            band_truth, _axis = diag.base.make_source_at_wavelength_nm(
                diag.fig_run.aug.N_PIX,
                diag.fig_run.aug.HALF_WIDTH_UAS,
                center_nm,
            )
            band_vgrid, band_uv_axis = diag.base.visibility_grid(band_truth, fov_rad)
            uu_rows, vv_rows = diag.project_enu_baselines(
                baselines,
                hour_angles,
                lam_m,
                latitude_deg=case.latitude_deg,
                declination_deg=diag.fig_run.GOOD_SOURCE.dec_deg,
            )

            fishers = {
                loop_label: {
                    "edge_local": 0.0,
                    "edge_fullcycle_gls": 0.0,
                    "edge_nonsplitting": 0.0,
                    "near": 0.0,
                    "direct_sched": 0.0,
                }
                for _tri, loop_label, _kind in diag.LOOPS
            }
            near_cycle_fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
            direct_schedule_cycle_fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)

            for uu, vv in zip(uu_rows, vv_rows):
                vtrue = diag.base.interp_vis(band_vgrid, band_uv_axis, uu, vv)
                nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
                near_cycle_fisher += diag.split_sim.core4_remote_loop_fisher_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    direct_noise=direct_noise,
                    nu_eff=nu_eff,
                    q_basis=q_basis,
                    edges=edges,
                )
                direct_schedule_cycle_fisher += diag.split_sim.direct_root_weighted_fisher_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    direct_noise=direct_noise,
                    q_basis=q_basis,
                    edges=edges,
                )
                for tri, loop_label, _band_label in diag.LOOPS:
                    fishers[loop_label]["edge_local"] += diag.uniform_edge_scalar_loop_fisher_for_sample(
                        splits["edge_uniform"],
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )
                    fishers[loop_label]["edge_fullcycle_gls"] += edge_fullcycle_fisher_for_sample(
                        splits["edge_uniform"],
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        q_basis=q_basis,
                        edges=edges,
                        edge_to_index=edge_to_index,
                        n_station=n,
                    )
                    fishers[loop_label]["edge_nonsplitting"] += diag.nonsplitting_edge_scalar_loop_fisher_for_sample(
                        tri,
                        total_modes=total_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        edge_to_index=edge_to_index,
                    )

            for tri, loop_label, band_label in diag.LOOPS:
                fishers[loop_label]["near"] = diag.scalar_fisher_from_cycle_matrix(
                    near_cycle_fisher,
                    q_basis,
                    edges,
                    tri,
                )
                fishers[loop_label]["direct_sched"] = diag.scalar_fisher_from_cycle_matrix(
                    direct_schedule_cycle_fisher,
                    q_basis,
                    edges,
                    tri,
                )
                for strategy, fisher in fishers[loop_label].items():
                    rows.append(
                        {
                            "loop": loop_label,
                            "loop_class": band_label,
                            "strategy": strategy,
                            "band_index": band_idx,
                            "lambda_lo_nm": float(lo_nm),
                            "lambda_hi_nm": float(hi_nm),
                            "lambda_center_nm": center_nm,
                            "rms_rad": 1.0 / math.sqrt(max(float(fisher), 1e-300)),
                            "scalar_fisher": float(fisher),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_ratios(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for _tri, loop, _kind in diag.LOOPS:
        local = np.asarray(
            [float(r["rms_rad"]) for r in rows if r["loop"] == loop and r["strategy"] == "edge_local"],
            dtype=float,
        )
        full = np.asarray(
            [float(r["rms_rad"]) for r in rows if r["loop"] == loop and r["strategy"] == "edge_fullcycle_gls"],
            dtype=float,
        )
        sched = np.asarray(
            [float(r["rms_rad"]) for r in rows if r["loop"] == loop and r["strategy"] == "direct_sched"],
            dtype=float,
        )
        out.append(
            {
                "loop": loop,
                "edge_fullcycle_over_edge_local_rms_mean": float(np.mean(full / local)),
                "edge_fullcycle_over_edge_local_rms_max_abs_minus_one": float(np.max(np.abs(full / local - 1.0))),
                "edge_local_over_scheduled_direct_rms_mean": float(np.mean(local / sched)),
                "edge_fullcycle_over_scheduled_direct_rms_mean": float(np.mean(full / sched)),
                "note": "full-cycle GLS uses all uniform edge measurements and therefore includes equivalent closure combinations such as Phi124+Phi234-Phi134",
            }
        )
    return out


def plot_rows(rows: list[dict[str, float | str]]) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 6.4,
            "axes.labelsize": 6.4,
            "axes.titlesize": 6.8,
            "legend.fontsize": 5.4,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
        }
    )
    specs = [
        ("edge_local", "edge local", "#0077b6", "o", "-"),
        ("edge_fullcycle_gls", "edge full-cycle GLS", "#7b2cbf", "x", "--"),
        ("edge_nonsplitting", "non-split edge", "#2a9d8f", "D", "-"),
        ("near", "near", "#f77f00", "s", "-"),
        ("direct_sched", "direct", "#d00000", "^", "-"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(5.35, 1.85), sharey=True, constrained_layout=True)
    rms_values = []
    for ax, (_tri, loop, kind) in zip(axes, diag.LOOPS):
        loop_rows = [row for row in rows if row["loop"] == loop]
        for strategy, label, color, marker, linestyle in specs:
            vals = sorted(
                [row for row in loop_rows if row["strategy"] == strategy],
                key=lambda row: float(row["lambda_center_nm"]),
            )
            lam = np.asarray([float(row["lambda_center_nm"]) for row in vals], dtype=float)
            rms_deg = np.asarray([float(row["rms_rad"]) for row in vals], dtype=float) * 180.0 / math.pi
            rms_values.extend([float(v) for v in rms_deg if np.isfinite(v) and v > 0.0])
            ax.plot(
                lam,
                rms_deg,
                lw=0.85,
                ls=linestyle,
                color=color,
                marker=marker,
                ms=2.2,
                label=label if loop == "123" else None,
            )
        ax.set_title(f"loop {loop} ({kind})")
        ax.set_yscale("log")
        ax.set_xlim(600, 700)
        ax.set_xticks([605, 655, 695], ["605", "655", "695"])
        ax.set_xlabel("nm")
        ax.grid(True, which="both", axis="y", color="0.88", lw=0.42)
        ax.grid(True, which="major", axis="x", color="0.93", lw=0.35)
        ax.set_axisbelow(True)
    if rms_values:
        ymin = max(min(rms_values) / 1.45, 0.08)
        ymax = max(rms_values) * 1.45
        for ax in axes:
            ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel("effective RMS (deg)")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.2, borderpad=0.1)
    pdf = FIG_DIR / f"fig3b_loop_rms_edge_fullcycle_gls{diag.OUTPUT_SUFFIX}.pdf"
    png = FIG_DIR / f"fig3b_loop_rms_edge_fullcycle_gls{diag.OUTPUT_SUFFIX}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    rows = per_wavelength_rows()
    csv_path = OUT / f"fig3b_loop_rms_edge_fullcycle_gls{diag.OUTPUT_SUFFIX}.csv"
    json_path = OUT / f"fig3b_loop_rms_edge_fullcycle_gls{diag.OUTPUT_SUFFIX}.json"
    summary_path = OUT / f"fig3b_loop_rms_edge_fullcycle_gls_summary{diag.OUTPUT_SUFFIX}.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    summary = summarize_ratios(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    pdf, png = plot_rows(rows)
    payload = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "rows_csv": str(csv_path),
        "rows_json": str(json_path),
        "summary_json": str(summary_path),
        "summary": summary,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
