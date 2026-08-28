from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


LOCAL_EDGES = [(0, 1), (0, 2), (1, 2)]


@dataclass(frozen=True)
class PairCombineModule:
    loop: tuple[int, int, int]
    pair: tuple[int, int]
    third: int
    label: str

    @property
    def stations(self) -> tuple[int, int, int]:
        return (self.pair[0], self.pair[1], self.third)


def module_label(pair: tuple[int, int], third: int) -> str:
    return f"S{pair[0] + 1}+S{pair[1] + 1}|S{third + 1}"


def balanced_paircombine_modules(b18, bm) -> list[PairCombineModule]:
    out: list[PairCombineModule] = []
    for loop in b18.balanced_independent_triangles(bm.n):
        a, b, c = loop
        for pair, third in (((a, b), c), ((a, c), b), ((b, c), a)):
            out.append(PairCombineModule(loop=loop, pair=pair, third=third, label=module_label(pair, third)))
    return out


def local_edge_signs(stations: tuple[int, int, int]) -> np.ndarray:
    signs = []
    for i, j in LOCAL_EDGES:
        signs.append(1.0 if stations[i] < stations[j] else -1.0)
    return np.asarray(signs, dtype=float)


def embed_local_fisher(
    out: np.ndarray,
    local_fisher: np.ndarray,
    stations: tuple[int, int, int],
    global_index: dict[tuple[int, int], int],
) -> None:
    for li, (ai, bi) in enumerate(LOCAL_EDGES):
        gi = global_index[tuple(sorted((stations[ai], stations[bi])))]
        for lj, (aj, bj) in enumerate(LOCAL_EDGES):
            gj = global_index[tuple(sorted((stations[aj], stations[bj])))]
            out[gi, gj] += local_fisher[li, lj]


def paircombine_local_fisher(
    b18,
    bm,
    module: PairCombineModule,
    fractions: np.ndarray,
    beta: float,
    delta: float,
) -> np.ndarray:
    if float(np.min(fractions)) <= 0.0:
        return np.zeros((3, 3), dtype=float)

    stations = module.stations
    local_baselines = np.asarray(
        [bm.stations[stations[j]] - bm.stations[stations[i]] for i, j in LOCAL_EDGES],
        dtype=float,
    )
    signs = local_edge_signs(stations)
    c = math.cos(beta)
    s = math.sin(beta)
    phase = complex(math.cos(delta), math.sin(delta))
    transform = np.asarray([[c, phase * s, 0.0], [0.0, 0.0, 1.0]], dtype=complex)

    fisher = np.zeros((3, 3), dtype=float)
    for lam, freq, total_modes in bm.iter_bands():
        vgrid, uv_axis = bm.visibility_grid_for_wavelength(lam * 1e9)
        u_station = b18.core_remote.aug.station_u_modes(freq, bm.diameters[list(stations)])
        signal = bm.eta[list(stations)] * u_station
        load = signal + b18.variants.fig_run.EPS_STATION_RUN
        diag = fractions * load
        source = fractions * signal
        uu_rows, vv_rows = b18.project_enu_baselines(
            local_baselines,
            bm.hour_angles,
            lam,
            latitude_deg=bm.case.latitude_deg,
            declination_deg=b18.variants.fig_run.GOOD_SOURCE.dec_deg,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            v_local = b18.base.interp_vis(vgrid, uv_axis, uu, vv)
            bmat = np.diag(diag).astype(complex)
            coherences: dict[tuple[int, int], complex] = {}
            for edge_idx, (li, lj) in enumerate(LOCAL_EDGES):
                vis = complex(v_local[edge_idx])
                coh = math.sqrt(max(float(source[li] * source[lj]), 0.0)) * vis
                coherences[(li, lj)] = coh
                bmat[li, lj] = coh
                bmat[lj, li] = np.conj(coh)

            derivs = []
            for edge_idx, (li, lj) in enumerate(LOCAL_EDGES):
                coh = coherences[(li, lj)]
                deriv = np.zeros((3, 3), dtype=complex)
                deriv[li, lj] = 1j * coh * signs[edge_idx]
                deriv[lj, li] = -1j * np.conj(coh) * signs[edge_idx]
                derivs.append(transform @ deriv @ transform.conj().T)

            b2 = transform @ bmat @ transform.conj().T
            fisher += total_modes * b18.base.qfi_from_bmat_derivatives(b2, derivs, eig_floor=1e-12)

    return 0.5 * (fisher + fisher.T)


def paircombine_edge_fisher(
    b18,
    bm,
    modules: list[PairCombineModule],
    q: np.ndarray,
    beta: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    global_index = {edge: idx for idx, edge in enumerate(bm.edges)}
    out = np.zeros((len(bm.edges), len(bm.edges)), dtype=float)
    for idx, module in enumerate(modules):
        fractions = np.asarray(q[idx], dtype=float)
        if float(np.max(fractions)) <= 1.0e-12:
            continue
        local = paircombine_local_fisher(
            b18,
            bm,
            module,
            fractions,
            float(beta[idx]),
            float(delta[idx]),
        )
        embed_local_fisher(out, local, module.stations, global_index)
    return 0.5 * (out + out.T)
