from __future__ import annotations

import csv
import itertools
import json
import math
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
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

EPS_STATION = 0.02
EPS_PAIR = 0.01
EPS_DIRECT_EXTRA = 0.01
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


def edge_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    out = np.zeros(len(edges), dtype=float)
    out[edge_to_index[(a, b)]] = 1.0
    out[edge_to_index[(b, c)]] = 1.0
    out[edge_to_index[(a, c)]] = -1.0
    return out


def fisher_for_closure_marginal(fq: np.ndarray, q_basis: np.ndarray, c_edge: np.ndarray) -> float:
    d = q_basis.T @ c_edge
    cov = np.linalg.pinv(fq, rcond=1e-12)
    var = float(d @ cov @ d)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0
    return 1.0 / var


def edge_fisher_from_arrays(arrays: dict[str, np.ndarray], fi: float, fj: float) -> float:
    if fi <= 0.0 or fj <= 0.0:
        return 0.0
    denom = fi * arrays["ai"] + fj * arrays["aj"] + arrays["pair"]
    return float(np.sum(arrays["k"] * fi * fj / np.maximum(denom, 1e-300)))


def closure_fisher_from_edges(fab: float, fbc: float, fac: float) -> float:
    if min(fab, fbc, fac) <= 0.0:
        return 0.0
    return 1.0 / (1.0 / fab + 1.0 / fbc + 1.0 / fac)


def optimize_triangle_split(
    edge_arrays: dict[tuple[int, int], dict[str, np.ndarray]],
    tri: tuple[int, int, int],
) -> tuple[float, tuple[float, float, float]]:
    a, b, c = tri
    eab = edge_arrays[(a, b)]
    ebc = edge_arrays[(b, c)]
    eac = edge_arrays[(a, c)]

    def score(xa: float, xb: float, xc: float) -> float:
        return closure_fisher_from_edges(
            edge_fisher_from_arrays(eab, xa, xb),
            edge_fisher_from_arrays(ebc, 1.0 - xb, xc),
            edge_fisher_from_arrays(eac, 1.0 - xa, 1.0 - xc),
        )

    grid = np.linspace(SPLIT_FLOOR, 1.0 - SPLIT_FLOOR, 97)
    seeds = [
        (0.5, 0.5, 0.5),
        (SPLIT_FLOOR, SPLIT_FLOOR, 0.5),
        (1.0 - SPLIT_FLOOR, 1.0 - SPLIT_FLOOR, 0.5),
        (0.5, SPLIT_FLOOR, 1.0 - SPLIT_FLOOR),
        (0.5, 1.0 - SPLIT_FLOOR, SPLIT_FLOOR),
        (0.2, 0.8, 0.5),
        (0.8, 0.2, 0.5),
    ]
    best = seeds[0]
    best_score = score(*best)
    for seed in seeds:
        xa, xb, xc = seed
        local_score = score(xa, xb, xc)
        for _ in range(10):
            vals = [(score(v, xb, xc), v) for v in grid]
            local_score, xa = max(vals, key=lambda item: item[0])
            vals = [(score(xa, v, xc), v) for v in grid]
            local_score, xb = max(vals, key=lambda item: item[0])
            vals = [(score(xa, xb, v), v) for v in grid]
            local_score, xc = max(vals, key=lambda item: item[0])
        if local_score > best_score:
            best_score = local_score
            best = (xa, xb, xc)

    def to_y(x: float) -> float:
        z = (x - SPLIT_FLOOR) / (1.0 - 2.0 * SPLIT_FLOOR)
        z = min(max(z, 1e-12), 1.0 - 1e-12)
        return math.log(z / (1.0 - z))

    def from_y(y: np.ndarray) -> tuple[float, float, float]:
        vals = []
        for item in y:
            if item >= 0:
                z = 1.0 / (1.0 + math.exp(-float(item)))
            else:
                expy = math.exp(float(item))
                z = expy / (1.0 + expy)
            vals.append(SPLIT_FLOOR + (1.0 - 2.0 * SPLIT_FLOOR) * z)
        return tuple(vals)  # type: ignore[return-value]

    def objective(y: np.ndarray) -> float:
        return -score(*from_y(y))

    simplex = [np.asarray([to_y(v) for v in best], dtype=float)]
    for idx in range(3):
        y = simplex[0].copy()
        y[idx] += 0.45
        simplex.append(y)
    simplex = np.asarray(simplex)
    vals = np.asarray([objective(y) for y in simplex])
    for _ in range(600):
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if np.std(vals) < 1e-13 * max(1.0, abs(vals[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_reflected = objective(reflected)
        if vals[0] <= f_reflected < vals[-2]:
            simplex[-1] = reflected
            vals[-1] = f_reflected
            continue
        if f_reflected < vals[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            f_expanded = objective(expanded)
            simplex[-1] = expanded if f_expanded < f_reflected else reflected
            vals[-1] = min(f_expanded, f_reflected)
            continue
        contracted = centroid + 0.5 * (simplex[-1] - centroid)
        f_contracted = objective(contracted)
        if f_contracted < vals[-1]:
            simplex[-1] = contracted
            vals[-1] = f_contracted
            continue
        for idx in range(1, 4):
            simplex[idx] = simplex[0] + 0.5 * (simplex[idx] - simplex[0])
            vals[idx] = objective(simplex[idx])
    order = np.argsort(vals)
    refined = from_y(simplex[order[0]])
    refined_score = score(*refined)
    if refined_score > best_score:
        return refined_score, refined
    return best_score, best


class Benchmark:
    def __init__(self) -> None:
        configure_physics()
        self.case = rml_cases.load_maunakea_plus3_case()
        self.stations, self.diameters, self.names, self.is_added = aug.station_table_from_case(self.case)
        self.hub = np.asarray(self.case.hub_km, dtype=float)
        self.n_station = len(self.stations)
        self.edges = base.edge_list(self.n_station)
        self.baselines = np.asarray([self.stations[j] - self.stations[i] for i, j in self.edges], dtype=float)
        with ngc.patched_source(ngc.NGC4151):
            self.truth, _ = base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
        self.fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
        self.vgrid, self.uv_axis = base.visibility_grid(self.truth, self.fov_rad)
        effective_hub_dist = FIBER_LENGTH_SCALE * np.linalg.norm(self.stations - self.hub, axis=1)
        self.eta = 10.0 ** (-FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
        self.edge_noise = np.full(self.n_station, EPS_STATION, dtype=float)
        self.direct_noise = np.full(self.n_station, EPS_STATION + EPS_DIRECT_EXTRA, dtype=float)
        self.hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
        self.lam_edges_nm = np.arange(
            aug.LAMBDA_MIN_NM,
            aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM,
            aug.LAMBDA_STEP_NM,
        )
        self.lam_edges_nm[-1] = aug.LAMBDA_MAX_NM
        self.edge_arrays = self._edge_arrays()
        self.global_split = 1.0 / (self.n_station - 1.0)
        self.core4 = (0, 1, 2, 3)
        self.core4_fq = self.subset_fisher(self.core4)

    def iter_bands(self):
        for lo_nm, hi_nm in zip(self.lam_edges_nm[:-1], self.lam_edges_nm[1:]):
            lam = math.sqrt(lo_nm * hi_nm) * 1e-9
            freq = base.C_LIGHT / lam
            freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
            df = freq_hi - freq_lo
            total_modes = aug.EXPOSURE_S * aug.OBSERVING_DAYS * df
            yield lam, freq, total_modes

    def _edge_arrays(self) -> dict[tuple[int, int], dict[str, np.ndarray]]:
        payload = {edge: {"k": [], "ai": [], "aj": [], "pair": []} for edge in self.edges}
        for lam, freq, total_modes in self.iter_bands():
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
                vtrue = base.interp_vis(self.vgrid, self.uv_axis, uu, vv)
                nu_eff = np.clip(np.abs(vtrue), 1e-4, 0.98)
                for edge_index, (i, j) in enumerate(self.edges):
                    item = payload[(i, j)]
                    item["k"].append(
                        total_modes
                        * 4.0
                        * self.eta[i]
                        * self.eta[j]
                        * u_station[i]
                        * u_station[j]
                        * nu_eff[edge_index] ** 2
                    )
                    item["ai"].append(ai_station[i])
                    item["aj"].append(ai_station[j])
                    item["pair"].append(EPS_PAIR)
        return {
            edge: {key: np.asarray(value, dtype=float) for key, value in by_key.items()}
            for edge, by_key in payload.items()
        }

    def subset_fisher(self, subset: tuple[int, ...]) -> np.ndarray:
        subset = tuple(subset)
        m = len(subset)
        local_edges = base.edge_list(m)
        local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, m))
        fq = np.zeros((local_q.shape[1], local_q.shape[1]), dtype=float)
        local_stations = self.stations[list(subset)]
        local_baselines = np.asarray([local_stations[b] - local_stations[a] for a, b in local_edges])
        for lam, freq, total_modes in self.iter_bands():
            u_local = aug.station_u_modes(freq, self.diameters[list(subset)])
            uu_rows, vv_rows = project_enu_baselines(
                local_baselines,
                self.hour_angles,
                lam,
                latitude_deg=self.case.latitude_deg,
                declination_deg=ngc.NGC4151.dec_deg,
            )
            for uu, vv in zip(uu_rows, vv_rows):
                vlocal = base.interp_vis(self.vgrid, self.uv_axis, uu, vv)
                fq += total_modes * aug.noisy_closure_fisher_station_u(
                    vlocal,
                    self.eta[list(subset)],
                    self.direct_noise[list(subset)],
                    u_local,
                    local_q,
                    local_edges,
                )
        return fq

    def subset_closure_fisher(self, subset: tuple[int, ...], fq: np.ndarray, tri_global: tuple[int, int, int]) -> float:
        local_map = {global_i: local_i for local_i, global_i in enumerate(subset)}
        tri_local = tuple(local_map[i] for i in tri_global)
        local_edges = base.edge_list(len(subset))
        local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, len(subset)))
        c = edge_vector(local_edges, tri_local)
        return fisher_for_closure_marginal(fq, local_q, c)

    def edge_uniform(self, tri: tuple[int, int, int]) -> float:
        a, b, c = tri
        return closure_fisher_from_edges(
            edge_fisher_from_arrays(self.edge_arrays[(a, b)], self.global_split, self.global_split),
            edge_fisher_from_arrays(self.edge_arrays[(b, c)], self.global_split, self.global_split),
            edge_fisher_from_arrays(self.edge_arrays[(a, c)], self.global_split, self.global_split),
        )

    def edge_optimized(self, tri: tuple[int, int, int]) -> tuple[float, tuple[float, float, float]]:
        return optimize_triangle_split(self.edge_arrays, tri)


def make_rows() -> tuple[list[dict], dict]:
    bm = Benchmark()
    rows = []
    for tri in itertools.combinations(range(bm.n_station), 3):
        if tri[0] != 0:
            continue
        is_core = all(not bm.is_added[idx] for idx in tri)
        f_edge = bm.edge_uniform(tri)
        f_direct = bm.subset_closure_fisher(tri, bm.subset_fisher(tri), tri)
        f_opt_edge, split = bm.edge_optimized(tri)
        f_core4 = math.nan
        if is_core:
            f_core4 = bm.subset_closure_fisher(bm.core4, bm.core4_fq, tri)
        if is_core:
            f_approx = max(f_direct, f_core4)
            approx_label = "direct/core-joint"
        else:
            f_approx = f_opt_edge
            approx_label = "optimized split edge"
        a, b, c = tri
        xa, xb, xc = split
        rows.append(
            {
                "loop": f"{a+1}-{b+1}-{c+1}",
                "stations": f"{bm.names[a]} | {bm.names[b]} | {bm.names[c]}",
                "type": "core" if is_core else "remote",
                "approx": approx_label,
                "F_edge": f_edge,
                "rms_edge_rad": 1.0 / math.sqrt(max(f_edge, 1e-300)),
                "F_direct": f_direct,
                "rms_direct_rad": 1.0 / math.sqrt(max(f_direct, 1e-300)),
                "F_approx": f_approx,
                "rms_approx_rad": 1.0 / math.sqrt(max(f_approx, 1e-300)),
                "G_direct_vs_edge": math.sqrt(f_direct / f_edge) if f_edge > 0 else math.nan,
                "G_approx_vs_edge": math.sqrt(f_approx / f_edge) if f_edge > 0 else math.nan,
                "G_approx_vs_direct": math.sqrt(f_approx / f_direct) if f_direct > 0 else math.nan,
                "split_a_to_ab": xa,
                "split_a_to_ac": 1.0 - xa,
                "split_b_to_ab": xb,
                "split_b_to_bc": 1.0 - xb,
                "split_c_to_bc": xc,
                "split_c_to_ac": 1.0 - xc,
            }
        )
    metadata = {
        "case": bm.case.key,
        "station_names": bm.names,
        "station_eta": bm.eta.tolist(),
        "hub_km": list(bm.case.hub_km),
        "global_uniform_station_split": bm.global_split,
        "eps_station": EPS_STATION,
        "eps_pair": EPS_PAIR,
        "eps_direct_extra": EPS_DIRECT_EXTRA,
        "split_floor": SPLIT_FLOOR,
        "fiber_length_scale": FIBER_LENGTH_SCALE,
        "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
        "observing_days": aug.OBSERVING_DAYS,
        "n_time_windows": aug.N_TIME_WINDOWS,
        "exposure_s": aug.EXPOSURE_S,
    }
    return rows, metadata


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) >= 1e3 or (0 < abs(value) < 1e-2):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def write_outputs(rows: list[dict], metadata: dict) -> tuple[Path, Path, Path, Path]:
    tag = "keck1_noise_epsst0p02_pair0p01_dir0p01_L1"
    csv_path = OUT / f"{tag}.csv"
    json_path = OUT / f"{tag}.json"
    tex_path = OUT / f"{tag}_note.tex"
    pdf_path = OUT / f"{tag}_note.pdf"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")

    core = [r for r in rows if r["type"] == "core"]
    remote = [r for r in rows if r["type"] == "remote"]

    def median_gain(key: str, subset: list[dict]) -> float:
        return float(np.median([row[key] for row in subset]))

    table_lines = []
    for r in rows:
        table_lines.append(
            " & ".join(
                [
                    r["loop"],
                    r["type"],
                    fmt(r["F_edge"], 3),
                    fmt(r["rms_edge_rad"], 3),
                    fmt(r["F_direct"], 3),
                    fmt(r["rms_direct_rad"], 3),
                    fmt(r["F_approx"], 3),
                    fmt(r["rms_approx_rad"], 3),
                    fmt(r["G_direct_vs_edge"], 2),
                    fmt(r["G_approx_vs_edge"], 2),
                    fmt(r["G_approx_vs_direct"], 2),
                ]
            )
            + r" \\"
        )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.62in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{amsmath}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\begin{{document}}
\title{{Keck-I Closure Benchmark with Pair and Direct Receiver Noise}}
\author{{Codex diagnostic note}}
\date{{\today}}
\maketitle

\section*{{Noise model}}
This note recomputes the Keck-I triangular closures in the Maunakea top-four plus three remote-station layout.  The detector model is
\begin{{align}}
 B^{{\rm dir}}_{{ii}} &= \eta_i u_i+\epsilon_i+\epsilon_i^{{\rm dir}},&
 B^{{\rm dir}}_{{ij}} &= \sqrt{{\eta_i\eta_j u_i u_j}}\,V_{{ij}},
\end{{align}}
for the direct closure receiver, while the edge-first baseline Fisher uses
\begin{{equation}}
F_{{ij}}=
N_{{\rm mode}}\,
\frac{{4 f_i f_j \eta_i\eta_j u_i u_j |V_{{ij}}|^2}}
{{f_i(\eta_i u_i+\epsilon_i)+f_j(\eta_j u_j+\epsilon_j)+\epsilon_{{ij}}}}.
\end{{equation}}
The numerical values are
\[
\epsilon_i={EPS_STATION:.3f},\qquad
\epsilon_{{ij}}={EPS_PAIR:.3f},\qquad
\epsilon_i^{{\rm dir}}={EPS_DIRECT_EXTRA:.3f}.
\]
Fiber attenuation is treated as pure signal loss with
\[
\eta_i=10^{{-\alpha L_i/10}},\qquad
\alpha={FIBER_LOSS_DB_PER_KM:.2f}\,{{\rm dB/km}},\qquad
L_i={FIBER_LENGTH_SCALE:.1f}\,|\mathbf{{x}}_i-\mathbf{{x}}_{{\rm hub}}|.
\]

\section*{{Strategy definitions}}
\begin{{itemize}}
\item \textbf{{Edge-first}}: all baselines are read simultaneously with uniform station splitting, \(f_i=1/(N-1)={metadata['global_uniform_station_split']:.4f}\).  This is the baseline-by-baseline strategy.
\item \textbf{{Direct closure}}: each triangle is measured by its local three-mode SLD receiver with the additional direct receiver noise \(\epsilon_i^{{\rm dir}}\).
\item \textbf{{Approximate hybrid}}: core-only triangles use direct/core-joint readout; triangles involving remote stations use optimized edge-first splitting with each station-side split constrained to be at least {SPLIT_FLOOR:.2f}.
\end{{itemize}}

\section*{{Summary}}
For the three core-only closures, the approximate strategy is effectively identical to direct closure.  For the twelve remote-involving closures, optimized edge-first splitting recovers a large fraction of direct closure while keeping the receiver simpler.
\[
\mathrm{{median}}\left(\frac{{\mathrm{{SNR}}_{{\mathrm{{dir}}}}}}{{\mathrm{{SNR}}_{{\mathrm{{edge}}}}}}\right)_{{\rm all}}
= {median_gain('G_direct_vs_edge', rows):.2f},\qquad
\mathrm{{median}}\left(\frac{{\mathrm{{SNR}}_{{\mathrm{{approx}}}}}}{{\mathrm{{SNR}}_{{\mathrm{{edge}}}}}}\right)_{{\rm all}}
= {median_gain('G_approx_vs_edge', rows):.2f}.
\]
For remote-involving loops only,
\[
\mathrm{{median}}\left(\frac{{\mathrm{{SNR}}_{{\mathrm{{approx}}}}}}{{\mathrm{{SNR}}_{{\mathrm{{dir}}}}}}\right)_{{\rm remote}}
= {median_gain('G_approx_vs_direct', remote):.2f}.
\]

\section*{{Per-closure Fisher and RMS}}
\small
\begin{{longtable}}{{@{{}}llrrrrrrrrr@{{}}}}
\toprule
Loop & type & \(F_e\) & \(\sigma_e\) & \(F_d\) & \(\sigma_d\) & \(F_a\) & \(\sigma_a\) & \(G_d/e\) & \(G_a/e\) & \(G_a/d\) \\
\midrule
\endfirsthead
\toprule
Loop & type & \(F_e\) & \(\sigma_e\) & \(F_d\) & \(\sigma_d\) & \(F_a\) & \(\sigma_a\) & \(G_d/e\) & \(G_a/e\) & \(G_a/d\) \\
\midrule
\endhead
{chr(10).join(table_lines)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Interpretation}}
\(F_e\) is the uniform-split edge-first Fisher, \(F_d\) is the local direct-closure Fisher, and \(F_a\) is the approximate hybrid Fisher.  RMS values are \(1/\sqrt{{F}}\) in radians.  The positive \(\epsilon_{{ij}}\) makes the edge-first benchmark more realistic by adding pair-combiner/readout noise; the positive \(\epsilon_i^{{\rm dir}}\) simultaneously assigns a direct-receiver penalty, so the comparison is not one-sided in favor of direct closure.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return csv_path, json_path, tex_path, pdf_path


def main() -> None:
    rows, metadata = make_rows()
    csv_path, json_path, tex_path, pdf_path = write_outputs(rows, metadata)
    print(csv_path)
    print(json_path)
    print(tex_path)
    print(pdf_path)
    print(
        json.dumps(
            {
                "median_G_direct_vs_edge": float(np.median([r["G_direct_vs_edge"] for r in rows])),
                "median_G_approx_vs_edge": float(np.median([r["G_approx_vs_edge"] for r in rows])),
                "median_G_approx_vs_direct_remote": float(
                    np.median([r["G_approx_vs_direct"] for r in rows if r["type"] == "remote"])
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
