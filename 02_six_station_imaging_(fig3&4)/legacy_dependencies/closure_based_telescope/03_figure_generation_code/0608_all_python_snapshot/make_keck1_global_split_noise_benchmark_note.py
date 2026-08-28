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
    return 1.0 / var if np.isfinite(var) and var > 0.0 else 0.0


def edge_fisher_from_arrays(arrays: dict[str, np.ndarray], fi: float, fj: float) -> float:
    if fi <= 0.0 or fj <= 0.0:
        return 0.0
    denom = fi * arrays["ai"] + fj * arrays["aj"] + arrays["pair"]
    return float(np.sum(arrays["k"] * fi * fj / np.maximum(denom, 1e-300)))


def closure_fisher_from_edges(fab: float, fbc: float, fac: float) -> float:
    if min(fab, fbc, fac) <= 0.0:
        return 0.0
    return 1.0 / (1.0 / fab + 1.0 / fbc + 1.0 / fac)


def project_station_splits(raw: np.ndarray) -> np.ndarray:
    """Map unconstrained positive weights to directed split matrix.

    Each row i sums to one over j != i and has p_{i->j} >= SPLIT_FLOOR.
    """
    n = raw.shape[0]
    p = np.zeros((n, n), dtype=float)
    remaining = 1.0 - (n - 1) * SPLIT_FLOOR
    if remaining <= 0:
        raise ValueError("split floor is too large")
    for i in range(n):
        weights = np.exp(raw[i, np.arange(n) != i] - np.max(raw[i, np.arange(n) != i]))
        weights /= np.sum(weights)
        p[i, np.arange(n) != i] = SPLIT_FLOOR + remaining * weights
    return p


class Benchmark:
    def __init__(self) -> None:
        configure_physics()
        self.case = rml_cases.load_maunakea_plus3_case()
        self.stations, self.diameters, self.names, self.is_added = aug.station_table_from_case(self.case)
        self.hub = np.asarray(self.case.hub_km, dtype=float)
        self.n = len(self.stations)
        self.edges = base.edge_list(self.n)
        self.baselines = np.asarray([self.stations[j] - self.stations[i] for i, j in self.edges], dtype=float)
        with ngc.patched_source(ngc.NGC4151):
            self.truth, _ = base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
        self.fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
        self.vgrid, self.uv_axis = base.visibility_grid(self.truth, self.fov_rad)
        effective_hub_dist = FIBER_LENGTH_SCALE * np.linalg.norm(self.stations - self.hub, axis=1)
        self.eta = 10.0 ** (-FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
        self.edge_noise = np.full(self.n, EPS_STATION, dtype=float)
        self.direct_noise = np.full(self.n, EPS_STATION + EPS_DIRECT_EXTRA, dtype=float)
        self.hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
        self.lam_edges_nm = np.arange(aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM, aug.LAMBDA_STEP_NM)
        self.lam_edges_nm[-1] = aug.LAMBDA_MAX_NM
        self.edge_arrays = self._edge_arrays()
        self.global_uniform_split = 1.0 / (self.n - 1)
        self.core4 = (0, 1, 2, 3)
        self.core4_fq = self.subset_fisher(self.core4)
        self.keck1_loops = [tri for tri in itertools.combinations(range(self.n), 3) if tri[0] == 0]
        self.remote_loops = [tri for tri in self.keck1_loops if any(self.is_added[idx] for idx in tri)]

    def iter_bands(self):
        for lo_nm, hi_nm in zip(self.lam_edges_nm[:-1], self.lam_edges_nm[1:]):
            lam = math.sqrt(lo_nm * hi_nm) * 1e-9
            freq = base.C_LIGHT / lam
            freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
            freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
            yield lam, freq, aug.EXPOSURE_S * aug.OBSERVING_DAYS * (freq_hi - freq_lo)

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
        return {e: {k: np.asarray(v, dtype=float) for k, v in d.items()} for e, d in payload.items()}

    def subset_fisher(self, subset: tuple[int, ...]) -> np.ndarray:
        subset = tuple(subset)
        m = len(subset)
        local_edges = base.edge_list(m)
        q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, m))
        fq = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
        local_stations = self.stations[list(subset)]
        local_baselines = np.asarray([local_stations[j] - local_stations[i] for i, j in local_edges])
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
                    q_basis,
                    local_edges,
                )
        return fq

    def subset_closure_fisher(self, subset: tuple[int, ...], fq: np.ndarray, tri_global: tuple[int, int, int]) -> float:
        local_map = {g: i for i, g in enumerate(subset)}
        tri_local = tuple(local_map[i] for i in tri_global)
        local_edges = base.edge_list(len(subset))
        q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, len(subset)))
        return fisher_for_closure_marginal(fq, q_basis, edge_vector(local_edges, tri_local))

    def edge_fisher(self, i: int, j: int, p: np.ndarray) -> float:
        a, b = (i, j) if i < j else (j, i)
        return edge_fisher_from_arrays(self.edge_arrays[(a, b)], p[i, j], p[j, i])

    def closure_fisher_from_split(self, tri: tuple[int, int, int], p: np.ndarray) -> float:
        a, b, c = tri
        return closure_fisher_from_edges(
            self.edge_fisher(a, b, p),
            self.edge_fisher(b, c, p),
            self.edge_fisher(a, c, p),
        )

    def edge_uniform(self, tri: tuple[int, int, int]) -> float:
        p = np.full((self.n, self.n), 0.0)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    p[i, j] = self.global_uniform_split
        return self.closure_fisher_from_split(tri, p)

    def direct_fisher(self, tri: tuple[int, int, int]) -> float:
        return self.subset_closure_fisher(tri, self.subset_fisher(tri), tri)

    def optimize_global_split(self) -> tuple[np.ndarray, dict[str, float]]:
        """Optimize one global directed split matrix for all remote Keck-I loops."""
        n = self.n
        raw = np.zeros((n, n), dtype=float)
        for i in range(n):
            raw[i, i] = -np.inf
        # Warm start: emphasize remote-involving edges for Keck-I loops.
        for tri in self.remote_loops:
            for i, j in itertools.permutations(tri, 2):
                raw[i, j] += 1.0

        def objective_from_p(p: np.ndarray) -> float:
            # Maximize the geometric mean of approximate/direct ratios for
            # remote-involving loops.  This avoids sacrificing a weak loop.
            logs = []
            for tri in self.remote_loops:
                f_split = self.closure_fisher_from_split(tri, p)
                f_dir = self.direct_cache[tri]
                logs.append(math.log(max(f_split / max(f_dir, 1e-300), 1e-300)))
            return float(np.mean(logs))

        def objective(raw_matrix: np.ndarray) -> float:
            return objective_from_p(project_station_splits(raw_matrix))

        self.direct_cache = {tri: self.direct_fisher(tri) for tri in self.keck1_loops}
        rng = np.random.default_rng(20260527)
        best_raw = raw.copy()
        best_score = objective(best_raw)
        # Random restarts with station-row softmax variables.
        for scale in (0.6, 1.2, 2.0):
            for _ in range(350):
                candidate = raw + rng.normal(scale=scale, size=(n, n))
                np.fill_diagonal(candidate, -np.inf)
                score = objective(candidate)
                if score > best_score:
                    best_score = score
                    best_raw = candidate

        # Coordinate refinement over directed logits.
        for width in (1.2, 0.55, 0.25, 0.10, 0.04):
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        for sign in (-1.0, 1.0):
                            candidate = best_raw.copy()
                            candidate[i, j] += sign * width
                            score = objective(candidate)
                            if score > best_score:
                                best_score = score
                                best_raw = candidate
                                improved = True
        p = project_station_splits(best_raw)
        metrics = {
            "objective_log_geomean_ratio": best_score,
            "geomean_approx_over_direct": math.exp(0.5 * best_score),
        }
        return p, metrics


def make_rows() -> tuple[list[dict], dict, np.ndarray]:
    bm = Benchmark()
    p_opt, opt_metrics = bm.optimize_global_split()
    p_uniform = np.zeros((bm.n, bm.n), dtype=float)
    for i in range(bm.n):
        for j in range(bm.n):
            if i != j:
                p_uniform[i, j] = bm.global_uniform_split
    rows = []
    for tri in bm.keck1_loops:
        is_core = all(not bm.is_added[idx] for idx in tri)
        f_edge = bm.closure_fisher_from_split(tri, p_uniform)
        f_direct = bm.direct_cache[tri]
        f_global_split = bm.closure_fisher_from_split(tri, p_opt)
        f_core4 = math.nan
        if is_core:
            f_core4 = bm.subset_closure_fisher(bm.core4, bm.core4_fq, tri)
        if is_core:
            f_approx = max(f_direct, f_core4)
            label = "direct/core-joint"
        else:
            f_approx = f_global_split
            label = "global optimized split"
        a, b, c = tri
        rows.append(
            {
                "loop": f"{a+1}-{b+1}-{c+1}",
                "stations": f"{bm.names[a]} | {bm.names[b]} | {bm.names[c]}",
                "type": "core" if is_core else "remote",
                "approx": label,
                "F_edge_uniform": f_edge,
                "rms_edge_uniform_rad": 1.0 / math.sqrt(max(f_edge, 1e-300)),
                "F_direct": f_direct,
                "rms_direct_rad": 1.0 / math.sqrt(max(f_direct, 1e-300)),
                "F_approx": f_approx,
                "rms_approx_rad": 1.0 / math.sqrt(max(f_approx, 1e-300)),
                "G_direct_vs_edge": math.sqrt(f_direct / f_edge) if f_edge > 0 else math.nan,
                "G_approx_vs_edge": math.sqrt(f_approx / f_edge) if f_edge > 0 else math.nan,
                "G_approx_vs_direct": math.sqrt(f_approx / f_direct) if f_direct > 0 else math.nan,
            }
        )
    metadata = {
        "case": bm.case.key,
        "station_names": bm.names,
        "station_eta": bm.eta.tolist(),
        "hub_km": list(bm.case.hub_km),
        "eps_station": EPS_STATION,
        "eps_pair": EPS_PAIR,
        "eps_direct_extra": EPS_DIRECT_EXTRA,
        "split_floor": SPLIT_FLOOR,
        "fiber_length_scale": FIBER_LENGTH_SCALE,
        "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
        "global_uniform_station_split": bm.global_uniform_split,
        "optimization_metrics": opt_metrics,
    }
    return rows, metadata, p_opt


def fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    if abs(x) >= 1e3 or (0 < abs(x) < 1e-2):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def write_outputs(rows: list[dict], metadata: dict, p: np.ndarray) -> tuple[Path, Path, Path, Path]:
    tag = "keck1_global_split_epsst0p02_pair0p01_dir0p01_L1"
    csv_path = OUT / f"{tag}.csv"
    split_csv_path = OUT / f"{tag}_directed_splits.csv"
    json_path = OUT / f"{tag}.json"
    tex_path = OUT / f"{tag}_note.tex"
    pdf_path = OUT / f"{tag}_note.pdf"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with split_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["from_station", "to_station", "fraction"])
        names = metadata["station_names"]
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i != j:
                    writer.writerow([name_i, name_j, p[i, j]])
    json_path.write_text(json.dumps({"metadata": metadata, "rows": rows, "directed_splits": p.tolist()}, indent=2) + "\n")

    remote = [r for r in rows if r["type"] == "remote"]
    lines = []
    for r in rows:
        lines.append(
            " & ".join(
                [
                    r["loop"],
                    r["type"],
                    fmt(r["F_edge_uniform"]),
                    fmt(r["rms_edge_uniform_rad"]),
                    fmt(r["F_direct"]),
                    fmt(r["rms_direct_rad"]),
                    fmt(r["F_approx"]),
                    fmt(r["rms_approx_rad"]),
                    fmt(r["G_direct_vs_edge"], 2),
                    fmt(r["G_approx_vs_edge"], 2),
                    fmt(r["G_approx_vs_direct"], 2),
                ]
            )
            + r" \\"
        )
    split_lines = []
    names = metadata["station_names"]
    for i, name_i in enumerate(names):
        if i == 0 or "new 5 m" in name_i:
            entries = []
            for j, name_j in enumerate(names):
                if i != j:
                    entries.append(f"{j+1}:{p[i,j]:.3f}")
            split_lines.append(f"{i+1} {name_i} & " + ", ".join(entries) + r" \\")

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.58in]{{geometry}}
\usepackage{{booktabs,longtable,amsmath,array,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\begin{{document}}
\title{{Keck-I Closure Benchmark with Globally Constrained Station Splitting}}
\author{{Codex diagnostic note}}
\date{{\today}}
\maketitle

\section*{{Correction to the previous local-split table}}
The previous approximate table optimized the split ratios separately for each triangle.  That is not a valid simultaneous array readout: each station must distribute one photon budget among all six other stations.  Here the approximate remote strategy uses directed fractions \(p_{{i\to j}}\) satisfying
\[
\sum_{{j\ne i}}p_{{i\to j}}=1,\qquad p_{{i\to j}}\ge {SPLIT_FLOOR:.2f}.
\]
The baseline Fisher uses \(p_{{i\to j}}\) and \(p_{{j\to i}}\).  The noise parameters are
\[
\epsilon_i={EPS_STATION:.2f},\qquad \epsilon_{{ij}}={EPS_PAIR:.2f},\qquad \epsilon_i^{{\rm dir}}={EPS_DIRECT_EXTRA:.2f}.
\]
Fiber attenuation uses \(L_i={FIBER_LENGTH_SCALE:.1f}|\mathbf{{x}}_i-\mathbf{{x}}_{{\rm hub}}|\) and \(\alpha={FIBER_LOSS_DB_PER_KM:.2f}\,\mathrm{{dB/km}}\).

\section*{{Global split optimization}}
The directed split matrix is optimized once, using the twelve Keck-I closures that involve at least one remote station.  The objective is the geometric mean of \(F_{{\rm split}}/F_{{\rm direct}}\) over those remote loops, so the optimizer cannot spend the whole budget on only one loop.
\[
\mathrm{{median}}\left(\frac{{\mathrm{{SNR}}_{{\rm dir}}}}{{\mathrm{{SNR}}_{{\rm edge}}}}\right)={np.median([r['G_direct_vs_edge'] for r in rows]):.2f},\quad
\mathrm{{median}}\left(\frac{{\mathrm{{SNR}}_{{\rm approx}}}}{{\mathrm{{SNR}}_{{\rm edge}}}}\right)={np.median([r['G_approx_vs_edge'] for r in rows]):.2f},\quad
\mathrm{{median}}\left(\frac{{\mathrm{{SNR}}_{{\rm approx}}}}{{\mathrm{{SNR}}_{{\rm dir}}}}\right)_{{\rm remote}}={np.median([r['G_approx_vs_direct'] for r in remote]):.2f}.
\]

\section*{{Selected directed station splits}}
\small
\begin{{longtable}}{{@{{}}p{{0.30\linewidth}}p{{0.62\linewidth}}@{{}}}}
\toprule
Station & fractions to station index \(j\) \\
\midrule
{chr(10).join(split_lines)}
\bottomrule
\end{{longtable}}
\normalsize

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
{chr(10).join(lines)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Takeaway}}
Once the station-level photon budget is enforced globally, the optimized classical split still beats uniform edge-first, but it is farther from direct closure than the previous per-triangle table suggested.  This is the physically relevant benchmark for simultaneous readout of the Keck-I closure basis.

\end{{document}}
"""
    tex_path.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return csv_path, split_csv_path, tex_path, pdf_path


def main() -> None:
    rows, metadata, p = make_rows()
    csv_path, split_csv_path, tex_path, pdf_path = write_outputs(rows, metadata, p)
    remote = [r for r in rows if r["type"] == "remote"]
    print(csv_path)
    print(split_csv_path)
    print(tex_path)
    print(pdf_path)
    print(
        json.dumps(
            {
                "median_G_direct_vs_edge": float(np.median([r["G_direct_vs_edge"] for r in rows])),
                "median_G_approx_vs_edge": float(np.median([r["G_approx_vs_edge"] for r in rows])),
                "median_G_approx_vs_direct_remote": float(np.median([r["G_approx_vs_direct"] for r in remote])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
