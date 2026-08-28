from __future__ import annotations

import csv
import json
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_clean as base
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)


def wrapped_phase_diff(model: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.angle(model) - np.angle(truth))))


def radial_bin_stats(q_glambda: np.ndarray, values: np.ndarray, bins: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    finite = np.isfinite(values)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = finite & (q_glambda >= lo) & (q_glambda < hi)
        if np.count_nonzero(mask) == 0:
            continue
        vv = values[mask]
        rows.append(
            {
                "q_lo_glambda": float(lo),
                "q_hi_glambda": float(hi),
                "n": int(vv.size),
                "mean": float(np.mean(vv)),
                "median": float(np.median(vv)),
                "rms": float(np.sqrt(np.mean(vv**2))),
                "p90_abs": float(np.percentile(np.abs(vv), 90.0)),
            }
        )
    return rows


def main() -> None:
    explicit_npz = os.environ.get("HAWAII3_RML_NPZ", "")
    if explicit_npz:
        npz_path = Path(explicit_npz)
    else:
        candidates = sorted(
            OUTFIG.glob("*hawaii_top4_remote3_compact_r1p5_3_6*best_images.npz"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        npz_path = candidates[0] if candidates else None
    if npz_path is None or not npz_path.exists() or not npz_path.is_file():
        raise SystemExit(
            "No saved Hawaii+3 RML image array was found. "
            "I will not rerun RML in this residual script. "
            "Please rerun the validation/imaging pipeline once with the updated saver, "
            "then pass HAWAII3_RML_NPZ=/path/to/*best_images.npz."
        )

    saved = np.load(npz_path, allow_pickle=False)
    fit_image = np.asarray(saved["best_fit_image"], dtype=float)
    display_image = (
        np.asarray(saved["best_display_image"], dtype=float)
        if "best_display_image" in saved.files
        else val.upsample_image_nearest(fit_image, amp_rml.N_RML)
    )
    best_start = str(saved["best_start"]) if "best_start" in saved.files else "saved"

    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    best_residuals, _ = val.residual_diagnostics(fit_image, bands, case, "direct", axis_uas)
    best_metrics = amp_rml.metrics_for(val.upsample_image_nearest(fit_image, len(axis_uas)), truth, axis_uas)

    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    truth_grid, truth_uv_axis = base.visibility_grid(truth, fov_rad)
    display_grid, display_uv_axis = base.visibility_grid(display_image, fov_rad)
    model_grid = amp_rml.fft_vis(fit_image)
    model_uv_axis = np.fft.fftshift(np.fft.fftfreq(fit_image.shape[0], d=fov_rad / fit_image.shape[0]))
    fit_nyquist_glambda = float(np.max(np.abs(model_uv_axis)) / 1e9)

    stations, _, names, _ = aug.station_table_from_case(case)
    edges = base.edge_list(len(stations))
    n_edges = len(edges)
    lam_edges_nm = np.arange(
        aug.LAMBDA_MIN_NM,
        aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM,
        aug.LAMBDA_STEP_NM,
    )
    lam_edges_nm[-1] = aug.LAMBDA_MAX_NM

    rows: list[dict[str, float | int | str]] = []
    all_u = []
    all_v = []
    all_q = []
    all_amp_z = []
    all_amp_frac = []
    all_phase = []
    all_outside = []

    for band_index, band in enumerate(bands):
        lo_nm = float(lam_edges_nm[band_index])
        hi_nm = float(lam_edges_nm[band_index + 1])
        lam_nm = float(np.sqrt(lo_nm * hi_nm))
        u = np.asarray(band["u"], dtype=float)
        v = np.asarray(band["v"], dtype=float)
        truth_vis = base.interp_vis(truth_grid, truth_uv_axis, u, v)
        model_vis = base.interp_vis(model_grid, model_uv_axis, u, v)
        amp_truth = np.asarray(band.get("amp_true", np.abs(truth_vis)), dtype=float)
        amp_data = np.asarray(band["amp"], dtype=float)
        amp_model = np.abs(model_vis)
        amp_sigma = amp_rml.amplitude_sigma(amp_data, band.get("amp_sigma"))
        amp_delta = amp_model - amp_data
        amp_delta_truth = amp_model - amp_truth
        amp_z = amp_delta / np.maximum(amp_sigma, 1e-12)
        amp_frac = amp_delta_truth / np.maximum(amp_truth, amp_rml.AMP_ABS_FLOOR)
        phase_delta = wrapped_phase_diff(model_vis, truth_vis)
        q_glambda = np.sqrt(u * u + v * v) / 1e9
        outside = (
            (u < model_uv_axis[0])
            | (u > model_uv_axis[-1])
            | (v < model_uv_axis[0])
            | (v > model_uv_axis[-1])
        )

        n_rows = len(u) // n_edges
        edge_ids = np.tile(np.arange(n_edges), n_rows)
        time_ids = np.repeat(np.arange(n_rows), n_edges)
        for idx in range(len(u)):
            edge = edges[int(edge_ids[idx])]
            rows.append(
                {
                    "band_index": band_index,
                    "lambda_nm": lam_nm,
                    "time_index": int(time_ids[idx]),
                    "edge_index": int(edge_ids[idx]),
                    "station_i": names[edge[0]],
                    "station_j": names[edge[1]],
                    "u_glambda": float(u[idx] / 1e9),
                    "v_glambda": float(v[idx] / 1e9),
                    "q_glambda": float(q_glambda[idx]),
                    "outside_fit_fft_grid": int(outside[idx]),
                    "amp_truth": float(amp_truth[idx]),
                    "amp_data": float(amp_data[idx]),
                    "amp_sigma": float(amp_sigma[idx]),
                    "amp_rml": float(amp_model[idx]),
                    "delta_amp_data": float(amp_delta[idx]),
                    "delta_amp_truth": float(amp_delta_truth[idx]),
                    "frac_delta_amp": float(amp_frac[idx]),
                    "amp_z": float(amp_z[idx]),
                    "phase_truth_rad": float(np.angle(truth_vis[idx])),
                    "phase_rml_rad": float(np.angle(model_vis[idx])),
                    "delta_phase_rad": float(phase_delta[idx]),
                }
            )
        all_u.append(u / 1e9)
        all_v.append(v / 1e9)
        all_q.append(q_glambda)
        all_amp_z.append(amp_z)
        all_amp_frac.append(amp_frac)
        all_phase.append(phase_delta)
        all_outside.append(outside.astype(float))

    u_all = np.concatenate(all_u)
    v_all = np.concatenate(all_v)
    q_all = np.concatenate(all_q)
    amp_z_all = np.concatenate(all_amp_z)
    amp_frac_all = np.concatenate(all_amp_frac)
    phase_all = np.concatenate(all_phase)
    outside_all = np.concatenate(all_outside).astype(bool)

    npz_tag = npz_path.stem.replace("_best_images", "")
    safe_key = f"{npz_tag}_visibility_residuals".replace(".", "p")
    csv_path = OUTFIG / f"{safe_key}_samples.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bins = np.linspace(0.0, max(1.0, float(np.percentile(q_all, 99.5))), 18)
    amp_z_bins = radial_bin_stats(q_all, amp_z_all, bins)
    amp_frac_bins = radial_bin_stats(q_all, amp_frac_all, bins)
    phase_bins = radial_bin_stats(q_all, phase_all, bins)
    radial_path = OUTFIG / f"{safe_key}_radial_bins.csv"
    with radial_path.open("w", newline="") as f:
        fieldnames = [
            "quantity",
            "q_lo_glambda",
            "q_hi_glambda",
            "n",
            "mean",
            "median",
            "rms",
            "p90_abs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for quantity, table in (
            ("amp_z", amp_z_bins),
            ("frac_delta_amp", amp_frac_bins),
            ("delta_phase_rad", phase_bins),
        ):
            for row in table:
                writer.writerow({"quantity": quantity, **row})

    # Full Fourier-grid diagnostic: compare the final displayed RML image with
    # the original truth on every FFT cell, not only on sampled baselines.
    uu_grid, vv_grid = np.meshgrid(display_uv_axis, display_uv_axis)
    q_grid = np.sqrt(uu_grid * uu_grid + vv_grid * vv_grid) / 1e9
    amp_truth_grid = np.abs(truth_grid)
    amp_display_grid = np.abs(display_grid)
    sample_amp_sigma = np.concatenate(
        [
            amp_rml.amplitude_sigma(np.asarray(band["amp"], dtype=float), band.get("amp_sigma"))
            for band in bands
        ]
    )
    amp_sigma_grid = np.full_like(amp_truth_grid, float(np.nanmedian(sample_amp_sigma)))
    full_amp_z_grid = (amp_display_grid - amp_truth_grid) / np.maximum(amp_sigma_grid, 1e-12)
    full_amp_frac_grid = (amp_display_grid - amp_truth_grid) / np.maximum(amp_truth_grid, amp_rml.AMP_ABS_FLOOR)
    full_phase_grid = wrapped_phase_diff(display_grid, truth_grid)
    fit_support_grid = (np.abs(uu_grid) <= np.max(np.abs(model_uv_axis))) & (
        np.abs(vv_grid) <= np.max(np.abs(model_uv_axis))
    )

    covered_grid = np.zeros_like(amp_truth_grid, dtype=bool)
    du_full = display_uv_axis[1] - display_uv_axis[0]

    def mark_coverage(us: np.ndarray, vs: np.ndarray) -> None:
        iu = np.rint((us - display_uv_axis[0]) / du_full).astype(int)
        iv = np.rint((vs - display_uv_axis[0]) / du_full).astype(int)
        valid = (iu >= 0) & (iu < covered_grid.shape[1]) & (iv >= 0) & (iv < covered_grid.shape[0])
        covered_grid[iv[valid], iu[valid]] = True

    for band in bands:
        u = np.asarray(band["u"], dtype=float)
        v = np.asarray(band["v"], dtype=float)
        mark_coverage(u, v)
        mark_coverage(-u, -v)

    full_rows = []
    for iy in range(display_grid.shape[0]):
        for ix in range(display_grid.shape[1]):
            full_rows.append(
                {
                    "iy": iy,
                    "ix": ix,
                    "u_glambda": float(uu_grid[iy, ix] / 1e9),
                    "v_glambda": float(vv_grid[iy, ix] / 1e9),
                    "q_glambda": float(q_grid[iy, ix]),
                    "covered_by_measured_sample": int(covered_grid[iy, ix]),
                    "inside_fit_grid_nyquist_box": int(fit_support_grid[iy, ix]),
                    "amp_truth": float(amp_truth_grid[iy, ix]),
                    "amp_rml_display": float(amp_display_grid[iy, ix]),
                    "delta_amp": float(amp_display_grid[iy, ix] - amp_truth_grid[iy, ix]),
                    "frac_delta_amp": float(full_amp_frac_grid[iy, ix]),
                    "amp_z": float(full_amp_z_grid[iy, ix]),
                    "phase_truth_rad": float(np.angle(truth_grid[iy, ix])),
                    "phase_rml_rad": float(np.angle(display_grid[iy, ix])),
                    "delta_phase_rad": float(full_phase_grid[iy, ix]),
                }
            )
    full_grid_csv = OUTFIG / f"{safe_key}_full_kgrid.csv"
    with full_grid_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        writer.writeheader()
        writer.writerows(full_rows)

    full_radial_path = OUTFIG / f"{safe_key}_full_kgrid_radial_bins.csv"
    full_bins = np.linspace(0.0, float(np.max(q_grid)), 18)
    full_groups = {
        "all": np.ones(q_grid.size, dtype=bool),
        "covered": covered_grid.reshape(-1),
        "uncovered": ~covered_grid.reshape(-1),
        "inside_fit_nyquist": fit_support_grid.reshape(-1),
        "outside_fit_nyquist": ~fit_support_grid.reshape(-1),
    }
    with full_radial_path.open("w", newline="") as f:
        fieldnames = [
            "quantity",
            "region",
            "q_lo_glambda",
            "q_hi_glambda",
            "n",
            "mean",
            "median",
            "rms",
            "p90_abs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        q_flat = q_grid.reshape(-1)
        for quantity, values in (
            ("amp_z", full_amp_z_grid.reshape(-1)),
            ("frac_delta_amp", full_amp_frac_grid.reshape(-1)),
            ("delta_phase_rad", full_phase_grid.reshape(-1)),
        ):
            for region, mask in full_groups.items():
                table = radial_bin_stats(q_flat[mask], values.reshape(-1)[mask], full_bins)
                for row in table:
                    writer.writerow({"quantity": quantity, "region": region, **row})

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), constrained_layout=True)
    sample = np.arange(len(q_all))
    if len(sample) > 80000:
        rng = np.random.default_rng(20260520)
        sample = rng.choice(sample, size=80000, replace=False)

    sc0 = axes[0, 0].scatter(
        u_all[sample],
        v_all[sample],
        c=np.clip(amp_z_all[sample], -5.0, 5.0),
        s=2.0,
        cmap="coolwarm",
        vmin=-5.0,
        vmax=5.0,
        rasterized=True,
    )
    axes[0, 0].set_title(r"Amplitude residual $(|V|_{\rm RML}-|V|_{\rm truth})/\sigma_{|V|}$")
    axes[0, 0].set_xlabel(r"$u$ (G$\lambda$)")
    axes[0, 0].set_ylabel(r"$v$ (G$\lambda$)")
    axes[0, 0].axvline(fit_nyquist_glambda, color="k", lw=0.7, ls="--")
    axes[0, 0].axvline(-fit_nyquist_glambda, color="k", lw=0.7, ls="--")
    axes[0, 0].axhline(fit_nyquist_glambda, color="k", lw=0.7, ls="--")
    axes[0, 0].axhline(-fit_nyquist_glambda, color="k", lw=0.7, ls="--")
    fig.colorbar(sc0, ax=axes[0, 0], label="clipped amp z")

    sc1 = axes[0, 1].scatter(
        u_all[sample],
        v_all[sample],
        c=np.clip(phase_all[sample], -np.pi, np.pi),
        s=2.0,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        rasterized=True,
    )
    axes[0, 1].set_title(r"Per-baseline phase residual $\arg V_{\rm RML}-\arg V_{\rm truth}$")
    axes[0, 1].set_xlabel(r"$u$ (G$\lambda$)")
    axes[0, 1].set_ylabel(r"$v$ (G$\lambda$)")
    axes[0, 1].axvline(fit_nyquist_glambda, color="k", lw=0.7, ls="--")
    axes[0, 1].axvline(-fit_nyquist_glambda, color="k", lw=0.7, ls="--")
    axes[0, 1].axhline(fit_nyquist_glambda, color="k", lw=0.7, ls="--")
    axes[0, 1].axhline(-fit_nyquist_glambda, color="k", lw=0.7, ls="--")
    fig.colorbar(sc1, ax=axes[0, 1], label="rad")

    axes[0, 2].hist(np.clip(amp_z_all, -10.0, 10.0), bins=90, density=True, alpha=0.8, label="amp z")
    axes[0, 2].hist(np.clip(phase_all, -np.pi, np.pi), bins=90, density=True, alpha=0.5, label="phase rad")
    axes[0, 2].set_title("Residual histograms")
    axes[0, 2].set_xlabel("residual")
    axes[0, 2].set_ylabel("density")
    axes[0, 2].legend(fontsize=8)

    def plot_radial(ax, table, label, color):
        x = np.array([(row["q_lo_glambda"] + row["q_hi_glambda"]) / 2.0 for row in table])
        y = np.array([row["rms"] for row in table])
        p90 = np.array([row["p90_abs"] for row in table])
        ax.plot(x, y, marker="o", ms=3, label=f"{label} rms", color=color)
        ax.plot(x, p90, marker="s", ms=2.5, ls="--", label=f"{label} p90", color=color, alpha=0.75)

    plot_radial(axes[1, 0], amp_z_bins, "amp z", "#1f77b4")
    axes[1, 0].axvline(fit_nyquist_glambda, color="k", lw=0.8, ls="--", label="fit-grid Nyquist")
    axes[1, 0].set_title("Radial amplitude residual")
    axes[1, 0].set_xlabel(r"$|k|$ (G$\lambda$)")
    axes[1, 0].set_ylabel("residual")
    axes[1, 0].legend(fontsize=7)

    plot_radial(axes[1, 1], phase_bins, "phase", "#d62728")
    axes[1, 1].axvline(fit_nyquist_glambda, color="k", lw=0.8, ls="--", label="fit-grid Nyquist")
    axes[1, 1].set_title("Radial per-baseline phase residual")
    axes[1, 1].set_xlabel(r"$|k|$ (G$\lambda$)")
    axes[1, 1].set_ylabel("rad")
    axes[1, 1].legend(fontsize=7)

    axes[1, 2].scatter(q_all, np.clip(amp_frac_all, -2.0, 2.0), s=1.5, alpha=0.18, rasterized=True)
    axes[1, 2].axvline(fit_nyquist_glambda, color="k", lw=0.8, ls="--", label="fit-grid Nyquist")
    axes[1, 2].set_title(r"Fractional amplitude residual")
    axes[1, 2].set_xlabel(r"$|k|$ (G$\lambda$)")
    axes[1, 2].set_ylabel(r"$(|V|_{\rm RML}-|V|_{\rm truth})/\max(|V|_{\rm truth}, floor)$")
    axes[1, 2].legend(fontsize=7)

    for ax in axes[:1, :2].reshape(-1):
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        (
            f"Hawaii+3 visibility-domain residuals; best start={best_start}; "
            f"fit grid={fit_image.shape[0]}x{fit_image.shape[1]}; "
            f"amp chi2={best_residuals['amp_reduced_chi2']:.2f}, "
            f"CP chi2={best_residuals['phase_reduced_chi2']:.2f}"
        ),
        fontsize=10.5,
        weight="bold",
    )
    png = OUTFIG / f"{safe_key}.png"
    pdf = OUTFIG / f"{safe_key}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 3, figsize=(12.4, 7.0), constrained_layout=True)
    extent = [
        display_uv_axis[0] / 1e9,
        display_uv_axis[-1] / 1e9,
        display_uv_axis[0] / 1e9,
        display_uv_axis[-1] / 1e9,
    ]
    im0 = axes2[0, 0].imshow(
        np.clip(full_amp_z_grid, -10.0, 10.0),
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-10.0,
        vmax=10.0,
    )
    axes2[0, 0].contour(
        uu_grid / 1e9,
        vv_grid / 1e9,
        covered_grid.astype(float),
        levels=[0.5],
        colors="k",
        linewidths=0.45,
    )
    axes2[0, 0].set_title("Full-grid amplitude residual z\ncontour = measured-cell support")
    fig2.colorbar(im0, ax=axes2[0, 0], label="clipped amp z")

    im1 = axes2[0, 1].imshow(
        full_phase_grid,
        origin="lower",
        extent=extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes2[0, 1].contour(
        uu_grid / 1e9,
        vv_grid / 1e9,
        covered_grid.astype(float),
        levels=[0.5],
        colors="k",
        linewidths=0.45,
    )
    axes2[0, 1].set_title("Full-grid per-baseline phase residual\nfrom displayed RML image")
    fig2.colorbar(im1, ax=axes2[0, 1], label="rad")

    im2 = axes2[0, 2].imshow(
        covered_grid.astype(float) + 0.35 * fit_support_grid.astype(float),
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes2[0, 2].set_title("Fourier grid regions\nbright=covered, background=fit support")
    fig2.colorbar(im2, ax=axes2[0, 2], label="mask value")

    q_flat = q_grid.reshape(-1)
    amp_flat = full_amp_z_grid.reshape(-1)
    phase_flat = full_phase_grid.reshape(-1)

    def radial_curve(values: np.ndarray, mask: np.ndarray, bins_for_curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for lo, hi in zip(bins_for_curve[:-1], bins_for_curve[1:]):
            mm = mask & (q_flat >= lo) & (q_flat < hi) & np.isfinite(values)
            if np.count_nonzero(mm) < 2:
                continue
            xs.append(0.5 * (lo + hi))
            ys.append(float(np.sqrt(np.mean(values[mm] ** 2))))
        return np.asarray(xs), np.asarray(ys)

    for region, mask, color in (
        ("all", np.ones_like(q_flat, dtype=bool), "k"),
        ("covered", covered_grid.reshape(-1), "#1f77b4"),
        ("uncovered", ~covered_grid.reshape(-1), "#ff7f0e"),
        ("outside fit Nyquist", ~fit_support_grid.reshape(-1), "#d62728"),
    ):
        x, y = radial_curve(amp_flat, mask, full_bins)
        axes2[1, 0].plot(x, y, marker="o", ms=3, label=region, color=color)
    axes2[1, 0].axvline(fit_nyquist_glambda, color="gray", ls="--", lw=0.9)
    axes2[1, 0].set_title("Full-grid radial amplitude residual")
    axes2[1, 0].set_xlabel(r"$|k|$ (G$\lambda$)")
    axes2[1, 0].set_ylabel("RMS amp z")
    axes2[1, 0].legend(fontsize=7)

    for region, mask, color in (
        ("all", np.ones_like(q_flat, dtype=bool), "k"),
        ("covered", covered_grid.reshape(-1), "#1f77b4"),
        ("uncovered", ~covered_grid.reshape(-1), "#ff7f0e"),
        ("outside fit Nyquist", ~fit_support_grid.reshape(-1), "#d62728"),
    ):
        x, y = radial_curve(phase_flat, mask, full_bins)
        axes2[1, 1].plot(x, y, marker="o", ms=3, label=region, color=color)
    axes2[1, 1].axvline(fit_nyquist_glambda, color="gray", ls="--", lw=0.9)
    axes2[1, 1].set_title("Full-grid radial phase residual")
    axes2[1, 1].set_xlabel(r"$|k|$ (G$\lambda$)")
    axes2[1, 1].set_ylabel("RMS rad")
    axes2[1, 1].legend(fontsize=7)

    axes2[1, 2].hist(
        np.clip(amp_flat[covered_grid.reshape(-1)], -10.0, 10.0),
        bins=80,
        density=True,
        alpha=0.65,
        label="covered amp z",
    )
    axes2[1, 2].hist(
        np.clip(amp_flat[(~covered_grid).reshape(-1)], -10.0, 10.0),
        bins=80,
        density=True,
        alpha=0.65,
        label="uncovered amp z",
    )
    axes2[1, 2].set_title("Covered vs uncovered full-grid residuals")
    axes2[1, 2].set_xlabel("clipped amp z")
    axes2[1, 2].set_ylabel("density")
    axes2[1, 2].legend(fontsize=7)
    for col in range(3):
        axes2[0, col].set_xlabel(r"$u$ (G$\lambda$)")
        axes2[0, col].set_ylabel(r"$v$ (G$\lambda$)")
    for ax in (axes2[0, 0], axes2[0, 1]):
        ax.axvline(fit_nyquist_glambda, color="gray", lw=0.8, ls="--")
        ax.axvline(-fit_nyquist_glambda, color="gray", lw=0.8, ls="--")
        ax.axhline(fit_nyquist_glambda, color="gray", lw=0.8, ls="--")
        ax.axhline(-fit_nyquist_glambda, color="gray", lw=0.8, ls="--")
    fig2.suptitle(
        (
            f"Full k-space residuals: displayed RML vs truth; {case.key}; "
            f"fit Nyquist={fit_nyquist_glambda:.1f} Gλ"
        ),
        fontsize=10.5,
        weight="bold",
    )
    full_png = OUTFIG / f"{safe_key}_full_kgrid.png"
    full_pdf = OUTFIG / f"{safe_key}_full_kgrid.pdf"
    fig2.savefig(full_png, dpi=260, bbox_inches="tight")
    fig2.savefig(full_pdf, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "case": case.key,
        "source": amp_rml.SOURCE.name,
        "loaded_npz": str(npz_path),
        "best_start": best_start,
        "metrics": best_metrics,
        "residuals": best_residuals,
        "fit_grid": int(fit_image.shape[0]),
        "display_grid": int(amp_rml.N_RML),
        "fit_nyquist_glambda": fit_nyquist_glambda,
        "fraction_samples_outside_fit_fft_grid": float(np.mean(outside_all)),
        "amp_sigma_mode": amp_rml.AMP_SIGMA_MODE,
        "amp_sigma_abs": amp_rml.AMP_SIGMA_ABS,
        "amp_rel_sigma": amp_rml.AMP_REL_SIGMA,
        "amp_abs_floor": amp_rml.AMP_ABS_FLOOR,
        "phase_floor_rad": amp_rml.PHASE_FLOOR_RAD,
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        "full_kgrid_png": str(full_png),
        "full_kgrid_pdf": str(full_pdf),
        "samples_csv": str(csv_path),
        "radial_bins_csv": str(radial_path),
        "full_kgrid_csv": str(full_grid_csv),
        "full_kgrid_radial_bins_csv": str(full_radial_path),
    }
    json_path = OUTFIG / f"{safe_key}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(png)
    print(pdf)
    print(csv_path)
    print(radial_path)
    print(json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
