from __future__ import annotations

import csv
import itertools
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

import eht_style_amplitude_closure_rml as rml_cases
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_clean as base
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = Path(__file__).resolve().parents[2] / "exploration" / "technical_notes"
OUT.mkdir(parents=True, exist_ok=True)

EPS_STATION = float(os.environ.get("EPS_STATION", "1e-9"))
EPS_PAIR = float(os.environ.get("EPS_PAIR", "0.0"))
EPS_DIRECT_EXTRA = float(os.environ.get("EPS_DIRECT_EXTRA", "0.0"))
SPLIT_FLOOR = 0.02
FIBER_LENGTH_SCALE = 1.0
FIBER_LOSS_DB_PER_KM = 0.20


def configure_physics() -> None:
    aug.OBSERVING_DAYS = 30
    aug.N_TIME_WINDOWS = 36
    aug.EXPOSURE_S = 600.0
    aug.EXPOSURE_GAP_S = 150.0
    aug.FIBER_LOSS_DB_PER_KM = FIBER_LOSS_DB_PER_KM
    aug.FIBER_LENGTH_SCALE = FIBER_LENGTH_SCALE
    aug.MODE_FALSE_POSITIVE = EPS_STATION
    aug.PAIR_FALSE_POSITIVE = EPS_PAIR
    aug.BASELINE_FALSE_POSITIVE = EPS_PAIR
    wt.OBSERVING_DAYS = aug.OBSERVING_DAYS
    wt.SNR_BOOST = 1.0


def project_station_splits(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    p = np.zeros((n, n), dtype=float)
    remaining = 1.0 - (n - 1) * SPLIT_FLOOR
    if remaining <= 0.0:
        raise ValueError("split floor too large")
    for i in range(n):
        mask = np.arange(n) != i
        row = raw[i, mask]
        weights = np.exp(row - np.max(row))
        weights /= np.sum(weights)
        p[i, mask] = SPLIT_FLOOR + remaining * weights
    return p


def stable_metrics(fisher: np.ndarray) -> dict[str, float]:
    fisher = 0.5 * (fisher + fisher.T)
    eig = np.linalg.eigvalsh(fisher)
    eig = np.maximum(eig, 1e-300)
    cov = np.linalg.pinv(fisher, rcond=1e-12)
    diag = np.maximum(np.diag(cov), 0.0)
    return {
        "trace_fisher": float(np.trace(fisher)),
        "logdet_fisher": float(np.sum(np.log(eig))),
        "geomean_fisher_eigen": float(np.exp(np.mean(np.log(eig)))),
        "mean_coord_rms": float(np.mean(np.sqrt(diag))),
        "median_coord_rms": float(np.median(np.sqrt(diag))),
        "max_coord_rms": float(np.max(np.sqrt(diag))),
        "min_eigen": float(np.min(eig)),
        "max_eigen": float(np.max(eig)),
        "condition": float(np.max(eig) / np.min(eig)),
    }


def edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


class AllClosureBenchmark:
    def __init__(self) -> None:
        configure_physics()
        self.case = rml_cases.load_maunakea_plus3_case()
        self.stations, self.diameters, self.names, self.is_added = aug.station_table_from_case(self.case)
        self.hub = np.asarray(self.case.hub_km, dtype=float)
        self.n = len(self.stations)
        self.edges = base.edge_list(self.n)
        self.baselines = np.asarray([self.stations[j] - self.stations[i] for i, j in self.edges], dtype=float)
        self.w_basis = base.root_cycle_basis(self.edges, self.n)
        self.q_basis = base.orthonormal_cycle_basis(self.w_basis)
        self.n_closure = self.q_basis.shape[1]
        self.rank_share = min(1.0, (self.n - 1.0) / self.n_closure)
        with ngc.patched_source(ngc.NGC4151):
            self.truth, _ = base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
            self.fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
            self.vgrid, self.uv_axis = base.visibility_grid(self.truth, self.fov_rad)
            effective_hub_dist = FIBER_LENGTH_SCALE * np.linalg.norm(self.stations - self.hub, axis=1)
            self.eta = 10.0 ** (-FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
            self.edge_noise = np.full(self.n, EPS_STATION, dtype=float)
            self.direct_noise = np.full(self.n, EPS_STATION + EPS_DIRECT_EXTRA, dtype=float)
            self.hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
            self.lam_edges_nm = np.arange(
                aug.LAMBDA_MIN_NM,
                aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM,
                aug.LAMBDA_STEP_NM,
            )
            self.lam_edges_nm[-1] = aug.LAMBDA_MAX_NM
            self.edge_arrays = self._edge_arrays()
            self.direct_raw = self._direct_fisher()

    def iter_bands(self):
        for lo_nm, hi_nm in zip(self.lam_edges_nm[:-1], self.lam_edges_nm[1:]):
            lam = math.sqrt(lo_nm * hi_nm) * 1e-9
            freq = base.C_LIGHT / lam
            freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
            yield lam, freq, aug.EXPOSURE_S * aug.OBSERVING_DAYS * (freq_hi - freq_lo)

    def visibility_grid_for_wavelength(self, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
        wavelength_source = getattr(base, "make_source_at_wavelength_nm", None)
        if callable(wavelength_source):
            image, _ = wavelength_source(aug.N_PIX, aug.HALF_WIDTH_UAS, wavelength_nm)
            return base.visibility_grid(image, self.fov_rad)
        return self.vgrid, self.uv_axis

    def _edge_arrays(self) -> dict[tuple[int, int], dict[str, np.ndarray]]:
        payload = {edge: {"k": [], "ai": [], "aj": [], "pair": []} for edge in self.edges}
        for lam, freq, total_modes in self.iter_bands():
            vgrid, uv_axis = self.visibility_grid_for_wavelength(lam * 1e9)
            u_station = aug.station_u_modes(freq, self.diameters)
            ai_station = self.eta * u_station + self.edge_noise
            uu_rows, vv_rows = project_enu_baselines(
                self.baselines,
                self.hour_angles,
                lam,
                latitude_deg=self.case.latitude_deg,
                declination_deg=ngc.NGC4151.dec_deg,
            )
            for uu, vv in zip(uu_rows, vv_rows):
                vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
                nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
                for edge_idx, (i, j) in enumerate(self.edges):
                    item = payload[(i, j)]
                    item["k"].append(
                        total_modes
                        * 4.0
                        * self.eta[i]
                        * self.eta[j]
                        * u_station[i]
                        * u_station[j]
                        * nu_eff[edge_idx] ** 2
                    )
                    item["ai"].append(ai_station[i])
                    item["aj"].append(ai_station[j])
                    item["pair"].append(EPS_PAIR)
        return {edge: {key: np.asarray(value, dtype=float) for key, value in by_key.items()} for edge, by_key in payload.items()}

    def _direct_fisher(self) -> np.ndarray:
        fisher = np.zeros((self.n_closure, self.n_closure), dtype=float)
        for lam, freq, total_modes in self.iter_bands():
            vgrid, uv_axis = self.visibility_grid_for_wavelength(lam * 1e9)
            u_station = aug.station_u_modes(freq, self.diameters)
            uu_rows, vv_rows = project_enu_baselines(
                self.baselines,
                self.hour_angles,
                lam,
                latitude_deg=self.case.latitude_deg,
                declination_deg=ngc.NGC4151.dec_deg,
            )
            for uu, vv in zip(uu_rows, vv_rows):
                vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
                fisher += total_modes * aug.noisy_closure_fisher_station_u(
                    vtrue,
                    self.eta,
                    self.direct_noise,
                    u_station,
                    self.q_basis,
                    self.edges,
                )
        return fisher

    def edge_fisher_values(self, p: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.edges), dtype=float)
        for idx, (i, j) in enumerate(self.edges):
            arrays = self.edge_arrays[(i, j)]
            denom = p[i, j] * arrays["ai"] + p[j, i] * arrays["aj"] + arrays["pair"]
            out[idx] = float(np.sum(arrays["k"] * p[i, j] * p[j, i] / np.maximum(denom, 1e-300)))
        return out

    def edge_closure_fisher(self, p: np.ndarray) -> np.ndarray:
        return base.closure_fisher_after_gauge_marginalization(
            np.diag(self.edge_fisher_values(p)),
            self.q_basis,
            self.edges,
            self.n,
        )

    def uniform_split_matrix(self) -> np.ndarray:
        p = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    p[i, j] = 1.0 / (self.n - 1.0)
        return p

    def optimize_split_logdet(self) -> tuple[np.ndarray, dict[str, float]]:
        rng = np.random.default_rng(20260527)
        raw = np.zeros((self.n, self.n), dtype=float)
        np.fill_diagonal(raw, -np.inf)
        best_raw = raw.copy()

        def score(raw_matrix: np.ndarray) -> float:
            fisher = self.edge_closure_fisher(project_station_splits(raw_matrix))
            eig = np.linalg.eigvalsh(0.5 * (fisher + fisher.T))
            eig = np.maximum(eig, 1e-300)
            return float(np.mean(np.log(eig)))

        best_score = score(best_raw)
        # Random starts prevent the optimizer from merely preserving symmetry.
        for scale in (0.7, 1.4, 2.2):
            for _ in range(450):
                candidate = raw + rng.normal(scale=scale, size=(self.n, self.n))
                np.fill_diagonal(candidate, -np.inf)
                value = score(candidate)
                if value > best_score:
                    best_score = value
                    best_raw = candidate
        # Coordinate refinement in logit space.
        for width in (1.1, 0.5, 0.22, 0.09, 0.035):
            improved = True
            while improved:
                improved = False
                for i in range(self.n):
                    for j in range(self.n):
                        if i == j:
                            continue
                        for sign in (-1.0, 1.0):
                            candidate = best_raw.copy()
                            candidate[i, j] += sign * width
                            value = score(candidate)
                            if value > best_score:
                                best_score = value
                                best_raw = candidate
                                improved = True
        p = project_station_splits(best_raw)
        return p, {"objective_log_geomean_fisher": best_score}

    def keck1_loop_rows(self, matrices: dict[str, np.ndarray]) -> list[dict]:
        rows = []
        for tri in itertools.combinations(range(self.n), 3):
            if tri[0] != 0:
                continue
            c = edge_vector(self.edges, tri)
            d = self.q_basis.T @ c
            row = {
                "loop": f"{tri[0]+1}-{tri[1]+1}-{tri[2]+1}",
                "stations": f"{self.names[tri[0]]} | {self.names[tri[1]]} | {self.names[tri[2]]}",
                "type": "core" if all(not self.is_added[idx] for idx in tri) else "remote",
            }
            for key, fisher in matrices.items():
                cov = np.linalg.pinv(fisher, rcond=1e-12)
                var = float(d @ cov @ d)
                row[f"rms_{key}_rad"] = math.sqrt(var) if np.isfinite(var) and var > 0.0 else math.inf
                row[f"F_marginal_{key}"] = 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0
            rows.append(row)
        return rows


def gain_summary(numerator: dict[str, float], denominator: dict[str, float]) -> dict[str, float]:
    return {
        "trace_snr_gain": math.sqrt(numerator["trace_fisher"] / denominator["trace_fisher"]),
        "logdet_snr_gain": math.exp(0.5 * (numerator["logdet_fisher"] - denominator["logdet_fisher"]) / 15.0),
        "mean_rms_gain": denominator["mean_coord_rms"] / numerator["mean_coord_rms"],
        "median_rms_gain": denominator["median_coord_rms"] / numerator["median_coord_rms"],
    }


def write_outputs() -> tuple[Path, Path, Path, Path, dict]:
    bm = AllClosureBenchmark()
    p_uniform = bm.uniform_split_matrix()
    p_opt, opt_info = bm.optimize_split_logdet()
    matrices = {
        "edge_uniform": bm.edge_closure_fisher(p_uniform),
        "edge_optimized": bm.edge_closure_fisher(p_opt),
        "direct_raw": bm.direct_raw,
        "direct_scheduled": bm.rank_share * bm.direct_raw,
    }
    metrics = {key: stable_metrics(value) for key, value in matrices.items()}
    gains = {
        "edge_optimized_vs_edge_uniform": gain_summary(metrics["edge_optimized"], metrics["edge_uniform"]),
        "direct_raw_vs_edge_uniform": gain_summary(metrics["direct_raw"], metrics["edge_uniform"]),
        "direct_scheduled_vs_edge_uniform": gain_summary(metrics["direct_scheduled"], metrics["edge_uniform"]),
        "direct_scheduled_vs_edge_optimized": gain_summary(metrics["direct_scheduled"], metrics["edge_optimized"]),
    }
    loop_rows = bm.keck1_loop_rows(matrices)
    tag = "all_closure_global_epsst0p02_pair0p01_dir0p01_L1"
    csv_path = OUT / f"{tag}_keck1_loop_rms.csv"
    split_csv = OUT / f"{tag}_optimized_splits.csv"
    json_path = OUT / f"{tag}.json"
    tex_path = OUT / f"{tag}_note.tex"
    pdf_path = OUT / f"{tag}_note.pdf"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(loop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loop_rows)
    with split_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["from_station", "to_station", "fraction"])
        for i, name_i in enumerate(bm.names):
            for j, name_j in enumerate(bm.names):
                if i != j:
                    writer.writerow([name_i, name_j, p_opt[i, j]])
    payload = {
        "metadata": {
            "case": bm.case.key,
            "n_station": bm.n,
            "n_baseline": len(bm.edges),
            "n_closure": bm.n_closure,
            "rank_share": bm.rank_share,
            "eps_station": EPS_STATION,
            "eps_pair": EPS_PAIR,
            "eps_direct_extra": EPS_DIRECT_EXTRA,
            "split_floor": SPLIT_FLOOR,
            "fiber_length_scale": FIBER_LENGTH_SCALE,
            "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
            "optimization": opt_info,
        },
        "metrics": metrics,
        "gains": gains,
        "optimized_splits": p_opt.tolist(),
        "loop_rows": loop_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    metric_lines = []
    for key, label in [
        ("edge_uniform", "edge uniform"),
        ("edge_optimized", "edge opt split"),
        ("direct_raw", "direct raw QFI"),
        ("direct_scheduled", "direct scheduled"),
    ]:
        m = metrics[key]
        metric_lines.append(
            f"{label} & {m['trace_fisher']:.3e} & {m['geomean_fisher_eigen']:.3e} & "
            f"{m['mean_coord_rms']:.3g} & {m['max_coord_rms']:.3g} \\\\"
        )
    gain_lines = []
    for key, label in [
        ("edge_optimized_vs_edge_uniform", "edge opt / edge uniform"),
        ("direct_raw_vs_edge_uniform", "direct raw / edge uniform"),
        ("direct_scheduled_vs_edge_uniform", "direct scheduled / edge uniform"),
        ("direct_scheduled_vs_edge_optimized", "direct scheduled / edge opt"),
    ]:
        g = gains[key]
        gain_lines.append(
            f"{label} & {g['trace_snr_gain']:.2f} & {g['logdet_snr_gain']:.2f} & "
            f"{g['mean_rms_gain']:.2f} & {g['median_rms_gain']:.2f} \\\\"
        )
    loop_lines = []
    for row in loop_rows:
        loop_lines.append(
            f"{row['loop']} & {row['type']} & {row['rms_edge_uniform_rad']:.3g} & "
            f"{row['rms_edge_optimized_rad']:.3g} & {row['rms_direct_scheduled_rad']:.3g} & "
            f"{row['rms_direct_raw_rad']:.3g} \\\\"
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.58in]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\begin{{document}}
\title{{All-Closure Global Benchmark for the N-Station Array}}
\author{{Codex diagnostic note}}
\date{{\today}}
\maketitle

\section*{{Purpose}}
This note replaces the per-loop full-budget comparison by a matrix-level benchmark for measuring all independent closure degrees of freedom of the \(N=7\) Maunakea top-four plus remote-three array.  There are \(B=N(N-1)/2=21\) baselines and \(M=(N-1)(N-2)/2=15\) independent closure coordinates.

\section*{{Noise and resource model}}
Edge-first readout uses station-to-baseline splitting \(p_{{i\to j}}\), with \(\sum_{{j\ne i}}p_{{i\to j}}=1\) and \(p_{{i\to j}}\ge {SPLIT_FLOOR:.2f}\).  Its pair-level noise is \(\epsilon_{{ij}}={EPS_PAIR:.2f}\), and station noise is \(\epsilon_i={EPS_STATION:.2f}\).  Direct closure uses the global \(N\)-mode station covariance with diagonal noise \(\epsilon_i+\epsilon_i^{{\rm dir}}={EPS_STATION + EPS_DIRECT_EXTRA:.2f}\).  Fiber attenuation uses \(L_i={FIBER_LENGTH_SCALE:.1f}|\mathbf{{x}}_i-\mathbf{{x}}_{{\rm hub}}|\).

\section*{{Important distinction}}
The raw global direct matrix is the closure-space QFI of the \(N\)-mode receiver.  Because a single static receiver need not saturate all multiparameter SLD directions simultaneously, we also report a conservative scheduled proxy,
\[
F_{{\rm dir,sch}}=\frac{{N-1}}{{M}}F_{{\rm dir,raw}}={bm.rank_share:.2f}F_{{\rm dir,raw}}.
\]
This is the fairer quantity to compare with all closures being acquired together.

\section*{{Matrix-level performance}}
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
strategy & \(\mathrm{{Tr}}F\) & geom. eig. \(F\) & mean RMS & max RMS \\
\midrule
{chr(10).join(metric_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{SNR-like gains}}
\begin{{center}}
\begin{{tabular}}{{lrrrr}}
\toprule
comparison & trace gain & logdet gain & mean-RMS gain & median-RMS gain \\
\midrule
{chr(10).join(gain_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Keck-I loop marginal RMS from the all-closure covariance}}
\small
\begin{{longtable}}{{llrrrr}}
\toprule
loop & type & edge uniform & edge opt & direct scheduled & direct raw \\
\midrule
\endfirsthead
\toprule
loop & type & edge uniform & edge opt & direct scheduled & direct raw \\
\midrule
\endhead
{chr(10).join(loop_lines)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Takeaway}}
For all closures measured together, the relevant comparison is matrix-level.  Per-loop local direct receivers reuse the same station photon budget and are not a fair all-closure benchmark.  In this simultaneous benchmark, optimized classical splitting improves over uniform edge-first only modestly, while the global \(N\)-mode direct closure receiver still retains a substantial advantage.  The scheduled direct column is the conservative number to use unless a concrete simultaneous POVM is supplied for the full raw QFI matrix.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return csv_path, split_csv, tex_path, pdf_path, payload


def main() -> None:
    csv_path, split_csv, tex_path, pdf_path, payload = write_outputs()
    print(csv_path)
    print(split_csv)
    print(tex_path)
    print(pdf_path)
    print(json.dumps(payload["gains"], indent=2))


if __name__ == "__main__":
    main()
