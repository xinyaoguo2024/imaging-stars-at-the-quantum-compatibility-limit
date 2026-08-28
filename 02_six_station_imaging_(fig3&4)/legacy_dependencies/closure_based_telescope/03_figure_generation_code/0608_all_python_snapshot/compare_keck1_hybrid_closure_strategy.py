from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

import eht_style_amplitude_closure_rml as rml_cases
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_clean as base
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
SPLIT_FLOOR = 0.02


def configure_fig3_physics() -> None:
    aug.OBSERVING_DAYS = 30
    aug.N_TIME_WINDOWS = 36
    aug.EXPOSURE_S = 600.0
    aug.EXPOSURE_GAP_S = 150.0
    aug.FIBER_LOSS_DB_PER_KM = 0.20
    aug.FIBER_LENGTH_SCALE = 1.0
    aug.MODE_FALSE_POSITIVE = 0.05
    aug.PAIR_FALSE_POSITIVE = 0.0
    aug.BASELINE_FALSE_POSITIVE = 0.0
    wt.OBSERVING_DAYS = aug.OBSERVING_DAYS
    wt.SNR_BOOST = 1.0


def edge_vector(
    edges: list[tuple[int, int]],
    tri: tuple[int, int, int],
    *,
    local_labels: bool = False,
) -> np.ndarray:
    """Closure vector for C=phi_ab+phi_bc-phi_ac with a<b<c."""
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


def fisher_for_closure_scalar_path(fq: np.ndarray, q_basis: np.ndarray, c_edge: np.ndarray) -> float:
    # Physical closure C=c.phi, so the edge tangent for unit C is c/(c.c).
    denom = float(c_edge @ c_edge)
    tangent_q = q_basis.T @ (c_edge / denom)
    return float(tangent_q @ fq @ tangent_q)


def edge_fisher_from_arrays(arrays: dict[str, np.ndarray], fi: float, fj: float) -> float:
    if fi <= 0.0 or fj <= 0.0:
        return 0.0
    denom = fi * arrays["ai"] + fj * arrays["aj"] + arrays["pair"]
    return float(np.sum(arrays["k"] * fi * fj / np.maximum(denom, 1e-300)))


def closure_fisher_from_three_edges(fab: float, fbc: float, fac: float) -> float:
    if min(fab, fbc, fac) <= 0.0:
        return 0.0
    return 1.0 / (1.0 / fab + 1.0 / fbc + 1.0 / fac)


def optimize_triangle_split(
    edge_arrays: dict[tuple[int, int], dict[str, np.ndarray]],
    tri: tuple[int, int, int],
) -> tuple[float, tuple[float, float, float]]:
    """Optimize classical station-side split for one triangle.

    For tri=(a,b,c), xa is station a's fraction to edge ab; 1-xa goes
    to ac.  xb is b->ab; 1-xb is b->bc.  xc is c->bc; 1-xc is c->ac.
    """
    a, b, c = tri
    eab = edge_arrays[(a, b)]
    ebc = edge_arrays[(b, c)]
    eac = edge_arrays[(a, c)]

    def score(xa: float, xb: float, xc: float) -> float:
        fab = edge_fisher_from_arrays(eab, xa, xb)
        fbc = edge_fisher_from_arrays(ebc, 1.0 - xb, xc)
        fac = edge_fisher_from_arrays(eac, 1.0 - xa, 1.0 - xc)
        return closure_fisher_from_three_edges(fab, fbc, fac)

    # Dense enough for stable fractions, still cheap for 15 Keck-I loops.
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
        for _ in range(10):
            vals = [(score(v, xb, xc), v) for v in grid]
            _, xa = max(vals, key=lambda item: item[0])
            vals = [(score(xa, v, xc), v) for v in grid]
            _, xb = max(vals, key=lambda item: item[0])
            vals = [(score(xa, xb, v), v) for v in grid]
            local_score, xc = max(vals, key=lambda item: item[0])
        if local_score > best_score:
            best_score = local_score
            best = (xa, xb, xc)

    # Refine the grid result with a small dependency-free Nelder--Mead search.
    # This matters when the optimum is close to the implementation floor.
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
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                vals[-1] = f_expanded
            else:
                simplex[-1] = reflected
                vals[-1] = f_reflected
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


class ClosureCalculator:
    def __init__(self) -> None:
        configure_fig3_physics()
        self.case = rml_cases.load_maunakea_plus3_case()
        self.stations, self.diameters, self.names, self.is_added = aug.station_table_from_case(self.case)
        self.hub = np.asarray(self.case.hub_km, dtype=float)
        self.n_station = len(self.stations)
        self.edges = base.edge_list(self.n_station)
        self.baselines = np.asarray([self.stations[j] - self.stations[i] for i, j in self.edges], dtype=float)
        self.w_basis = base.root_cycle_basis(self.edges, self.n_station)
        self.q_basis = base.orthonormal_cycle_basis(self.w_basis)
        self.n_closure = self.q_basis.shape[1]
        self.global_split = 1.0 / (self.n_station - 1.0)
        self.rank_share = min(1.0, (self.n_station - 1.0) / self.n_closure)
        with ngc.patched_source(ngc.NGC4151):
            self.truth, _ = base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
        self.fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
        self.vgrid, self.uv_axis = base.visibility_grid(self.truth, self.fov_rad)
        effective_hub_dist = aug.FIBER_LENGTH_SCALE * np.linalg.norm(self.stations - self.hub, axis=1)
        self.eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
        self.noise = np.full(self.n_station, aug.MODE_FALSE_POSITIVE, dtype=float)
        self.hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
        self.lam_edges_nm = np.arange(
            aug.LAMBDA_MIN_NM,
            aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM,
            aug.LAMBDA_STEP_NM,
        )
        self.lam_edges_nm[-1] = aug.LAMBDA_MAX_NM
        self.edge_arrays = self._edge_arrays()
        self.full_fq = self.subset_fisher(tuple(range(self.n_station)))
        self.core4_subset = (0, 1, 2, 3)
        self.core4_fq = self.subset_fisher(self.core4_subset)

    def _iter_bands(self):
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
        for lam, freq, total_modes in self._iter_bands():
            u_station = aug.station_u_modes(freq, self.diameters)
            ai_station = self.eta * u_station + self.noise
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
                    item["pair"].append(aug.PAIR_FALSE_POSITIVE)
        return {
            edge: {key: np.asarray(values, dtype=float) for key, values in values_by_key.items()}
            for edge, values_by_key in payload.items()
        }

    def subset_fisher(self, subset: tuple[int, ...]) -> np.ndarray:
        subset = tuple(subset)
        m = len(subset)
        local_edges = base.edge_list(m)
        local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, m))
        local_fq = np.zeros((local_q.shape[1], local_q.shape[1]), dtype=float)
        local_stations = self.stations[list(subset)]
        local_baselines = np.asarray(
            [local_stations[b] - local_stations[a] for a, b in local_edges],
            dtype=float,
        )
        for lam, freq, total_modes in self._iter_bands():
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
                local_fq += total_modes * aug.noisy_closure_fisher_station_u(
                    vlocal,
                    self.eta[list(subset)],
                    self.noise[list(subset)],
                    u_local,
                    local_q,
                    local_edges,
                )
        return local_fq

    def subset_closure_fisher(
        self,
        subset: tuple[int, ...],
        fq: np.ndarray,
        tri_global: tuple[int, int, int],
        *,
        scalar_path: bool = False,
    ) -> float:
        local_map = {global_i: local_i for local_i, global_i in enumerate(subset)}
        tri_local = tuple(local_map[i] for i in tri_global)
        local_edges = base.edge_list(len(subset))
        local_q = base.orthonormal_cycle_basis(base.root_cycle_basis(local_edges, len(subset)))
        c = edge_vector(local_edges, tri_local)
        if scalar_path:
            return fisher_for_closure_scalar_path(fq, local_q, c)
        return fisher_for_closure_marginal(fq, local_q, c)

    def edge_first_uniform_global(self, tri: tuple[int, int, int]) -> float:
        a, b, c = tri
        return closure_fisher_from_three_edges(
            edge_fisher_from_arrays(self.edge_arrays[(a, b)], self.global_split, self.global_split),
            edge_fisher_from_arrays(self.edge_arrays[(b, c)], self.global_split, self.global_split),
            edge_fisher_from_arrays(self.edge_arrays[(a, c)], self.global_split, self.global_split),
        )

    def edge_first_all_photon(self, tri: tuple[int, int, int]) -> float:
        a, b, c = tri
        return closure_fisher_from_three_edges(
            edge_fisher_from_arrays(self.edge_arrays[(a, b)], 1.0, 1.0),
            edge_fisher_from_arrays(self.edge_arrays[(b, c)], 1.0, 1.0),
            edge_fisher_from_arrays(self.edge_arrays[(a, c)], 1.0, 1.0),
        )

    def fulln_fishers(self, tri: tuple[int, int, int]) -> tuple[float, float]:
        c = edge_vector(self.edges, tri)
        return (
            fisher_for_closure_marginal(self.full_fq, self.q_basis, c),
            fisher_for_closure_scalar_path(self.full_fq, self.q_basis, c),
        )

    def all_photon_edge_snrs(self, tri: tuple[int, int, int]) -> tuple[float, float, float]:
        a, b, c = tri
        return (
            math.sqrt(edge_fisher_from_arrays(self.edge_arrays[(a, b)], 1.0, 1.0)),
            math.sqrt(edge_fisher_from_arrays(self.edge_arrays[(b, c)], 1.0, 1.0)),
            math.sqrt(edge_fisher_from_arrays(self.edge_arrays[(a, c)], 1.0, 1.0)),
        )


def main() -> None:
    calc = ClosureCalculator()
    rows = []
    for tri in itertools.combinations(range(calc.n_station), 3):
        if tri[0] != 0:
            continue
        a, b, c = tri
        is_core_only = all(not calc.is_added[idx] for idx in tri)
        f_full_multi, f_full_scalar = calc.fulln_fishers(tri)
        f_full_multi_sched = calc.rank_share * f_full_multi

        f3 = calc.subset_closure_fisher(tri, calc.subset_fisher(tri), tri)
        f_edge_uniform = calc.edge_first_uniform_global(tri)
        f_edge_opt, split = optimize_triangle_split(calc.edge_arrays, tri)
        f_edge_all = calc.edge_first_all_photon(tri)
        f4_core_joint = math.nan
        f4_core_scalar = math.nan
        if is_core_only:
            f4_core_joint = calc.subset_closure_fisher(calc.core4_subset, calc.core4_fq, tri, scalar_path=False)
            f4_core_scalar = calc.subset_closure_fisher(calc.core4_subset, calc.core4_fq, tri, scalar_path=True)

        if is_core_only:
            # Proposed core readout: use the better of repeated 3-mode loop SLD
            # and a joint 4-mode receiver on the existing core.
            f_hybrid = max(f3, f4_core_joint)
            label = "core max(3-mode, 4-mode joint)"
        else:
            f_hybrid = f_edge_opt
            label = "remote optimized edge-first"

        edge_snrs = calc.all_photon_edge_snrs(tri)
        xa, xb, xc = split
        rows.append(
            {
                "loop": f"{a+1}-{b+1}-{c+1}",
                "stations": f"{calc.names[a]} | {calc.names[b]} | {calc.names[c]}",
                "type": "core_only" if is_core_only else "remote_involved",
                "proposed_receiver": label,
                "edge_snr_allphot_ab": edge_snrs[0],
                "edge_snr_allphot_bc": edge_snrs[1],
                "edge_snr_allphot_ac": edge_snrs[2],
                "edge_snr_min_over_max": min(edge_snrs) / max(edge_snrs),
                "global_uniform_station_split": calc.global_split,
                "triangle_split_floor": SPLIT_FLOOR,
                "opt_station_a_to_ab": xa,
                "opt_station_a_to_ac": 1.0 - xa,
                "opt_station_b_to_ab": xb,
                "opt_station_b_to_bc": 1.0 - xb,
                "opt_station_c_to_bc": xc,
                "opt_station_c_to_ac": 1.0 - xc,
                "F_full7_multi": f_full_multi,
                "F_full7_multi_scheduled": f_full_multi_sched,
                "F_full7_scalar_upper": f_full_scalar,
                "F_local3_direct": f3,
                "F_core4_joint": f4_core_joint,
                "F_core4_scalar_upper": f4_core_scalar,
                "F_edge_uniform_global_split": f_edge_uniform,
                "F_edge_optimized_split": f_edge_opt,
                "F_edge_all_photon_no_split": f_edge_all,
                "F_proposed_hybrid": f_hybrid,
                "rms_proposed_rad": 1.0 / math.sqrt(max(f_hybrid, 1e-300)),
                "rms_local3_rad": 1.0 / math.sqrt(max(f3, 1e-300)),
                "rms_full7_multi_rad": 1.0 / math.sqrt(max(f_full_multi, 1e-300)),
                "snr_hybrid_over_local3": math.sqrt(f_hybrid / f3) if f3 > 0 else math.nan,
                "snr_hybrid_over_full7_multi": math.sqrt(f_hybrid / f_full_multi) if f_full_multi > 0 else math.nan,
                "snr_hybrid_over_full7_scheduled": math.sqrt(f_hybrid / f_full_multi_sched)
                if f_full_multi_sched > 0
                else math.nan,
                "snr_hybrid_over_full7_scalar_upper": math.sqrt(f_hybrid / f_full_scalar)
                if f_full_scalar > 0
                else math.nan,
                "snr_hybrid_over_uniform_edge": math.sqrt(f_hybrid / f_edge_uniform) if f_edge_uniform > 0 else math.nan,
                "snr_edgeopt_over_local3": math.sqrt(f_edge_opt / f3) if f3 > 0 else math.nan,
            }
        )

    length_tag = f"Lscale{aug.FIBER_LENGTH_SCALE:g}".replace(".", "p")
    floor_tag = f"splitfloor{SPLIT_FLOOR:g}".replace(".", "p")
    csv_path = OUTDIR / f"keck1_hybrid_closure_strategy_snr_{length_tag}_{floor_tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def pct(key: str, subset: list[dict] | None = None) -> dict[str, float]:
        data = rows if subset is None else subset
        arr = np.asarray([float(row[key]) for row in data if np.isfinite(float(row[key]))], dtype=float)
        return {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    core_rows = [row for row in rows if row["type"] == "core_only"]
    remote_rows = [row for row in rows if row["type"] == "remote_involved"]
    summary = {
        "case": calc.case.key,
        "source": "NGC 4151",
        "included_loops": "triangular closures containing Keck I only",
        "n_rows": len(rows),
        "n_core_only": len(core_rows),
        "n_remote_involved": len(remote_rows),
        "station_names": calc.names,
        "station_is_added": calc.is_added.tolist(),
        "station_eta": calc.eta.tolist(),
        "hub_km": list(calc.case.hub_km),
        "global_uniform_station_split": calc.global_split,
        "triangle_split_floor": SPLIT_FLOOR,
        "full7_closure_rank_share": calc.rank_share,
        "physics": {
            "observing_days": aug.OBSERVING_DAYS,
            "n_time_windows": aug.N_TIME_WINDOWS,
            "exposure_s": aug.EXPOSURE_S,
            "fiber_loss_db_per_km": aug.FIBER_LOSS_DB_PER_KM,
            "fiber_length_scale": aug.FIBER_LENGTH_SCALE,
            "mode_false_positive": aug.MODE_FALSE_POSITIVE,
            "pair_false_positive": aug.PAIR_FALSE_POSITIVE,
        },
        "ratio_summary": {
            "all_hybrid_over_local3": pct("snr_hybrid_over_local3"),
            "all_hybrid_over_full7_multi": pct("snr_hybrid_over_full7_multi"),
            "all_hybrid_over_full7_scheduled": pct("snr_hybrid_over_full7_scheduled"),
            "all_hybrid_over_full7_scalar_upper": pct("snr_hybrid_over_full7_scalar_upper"),
            "all_hybrid_over_uniform_edge": pct("snr_hybrid_over_uniform_edge"),
            "remote_edgeopt_over_local3": pct("snr_edgeopt_over_local3", remote_rows),
            "core_hybrid_over_full7_multi": pct("snr_hybrid_over_full7_multi", core_rows),
            "remote_hybrid_over_full7_multi": pct("snr_hybrid_over_full7_multi", remote_rows),
        },
        "csv": str(csv_path),
    }
    json_path = OUTDIR / f"keck1_hybrid_closure_strategy_snr_summary_{length_tag}_{floor_tag}.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(csv_path)
    print(json_path)
    print(json.dumps(summary["ratio_summary"], indent=2))
    print("\nrepresentative rows")
    for loop in ["1-2-3", "1-2-5", "1-2-7", "1-5-7"]:
        for row in rows:
            if row["loop"] == loop:
                print(
                    loop,
                    row["type"],
                    "hybrid/local3",
                    f"{row['snr_hybrid_over_local3']:.3f}",
                    "hybrid/full7multi",
                    f"{row['snr_hybrid_over_full7_multi']:.3f}",
                    "hybrid/uniformEdge",
                    f"{row['snr_hybrid_over_uniform_edge']:.3f}",
                    "rms",
                    f"{row['rms_proposed_rad']:.4g}",
                )


if __name__ == "__main__":
    main()
