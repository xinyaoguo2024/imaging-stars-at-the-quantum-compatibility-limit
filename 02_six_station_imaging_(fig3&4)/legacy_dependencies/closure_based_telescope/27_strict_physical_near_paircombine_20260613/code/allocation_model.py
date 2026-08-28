from __future__ import annotations

import math

import numpy as np

from paircombine_receiver import PairCombineModule


def softmax_budget(raw_values: np.ndarray, total: float) -> np.ndarray:
    raw_values = np.asarray(raw_values, dtype=float)
    if raw_values.size == 0 or total <= 0.0:
        return np.zeros(raw_values.size, dtype=float)
    finite = np.isfinite(raw_values)
    if not np.any(finite):
        return np.full(raw_values.size, total / raw_values.size, dtype=float)
    values = raw_values.copy()
    values[~finite] = -1.0e100
    weights = np.exp(values - np.max(values))
    weights /= np.sum(weights)
    return total * weights


def q_positions_for_station(modules: list[PairCombineModule], station: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for midx, module in enumerate(modules):
        for slot, module_station in enumerate(module.stations):
            if module_station == station:
                out.append((midx, slot))
    return out


def allowed_pair_sinks(b18, bm, station: int) -> list[int]:
    if station in b18.core_remote.CORE:
        return list(b18.core_remote.REMOTE)
    return [j for j in range(bm.n) if j != station]


def project_allocation(
    b18,
    bm,
    modules: list[PairCombineModule],
    raw_p: np.ndarray,
    raw_q: np.ndarray,
    alpha_core: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros((bm.n, bm.n), dtype=float)
    q = np.zeros((len(modules), 3), dtype=float)
    station_alpha = b18.variants.alpha_by_station(alpha_core, bm)

    for station in range(bm.n):
        pair_targets = allowed_pair_sinks(b18, bm, station)
        q_positions = q_positions_for_station(modules, station)
        raw_values: list[float] = []
        tags: list[tuple[str, int, int]] = []
        for target in pair_targets:
            raw_values.append(float(raw_p[station, target]))
            tags.append(("p", station, target))
        for midx, slot in q_positions:
            raw_values.append(float(raw_q[midx, slot]))
            tags.append(("q", midx, slot))

        total = 1.0 - float(station_alpha[station])
        values = softmax_budget(np.asarray(raw_values, dtype=float), total)
        for value, tag in zip(values, tags):
            if tag[0] == "p":
                p[tag[1], tag[2]] = float(value)
            else:
                q[tag[1], tag[2]] = float(value)
    return p, q


def raw_from_saved_split(
    b18,
    bm,
    modules: list[PairCombineModule],
    p_saved: np.ndarray,
    *,
    q_raw_value: float = -18.0,
) -> tuple[np.ndarray, np.ndarray]:
    raw_p = np.full((bm.n, bm.n), -np.inf, dtype=float)
    raw_q = np.full((len(modules), 3), q_raw_value, dtype=float)
    for station in range(bm.n):
        targets = allowed_pair_sinks(b18, bm, station)
        values = np.maximum(np.asarray([p_saved[station, target] for target in targets], dtype=float), 1.0e-300)
        values = values / max(float(np.sum(values)), 1.0e-300)
        for target, value in zip(targets, values):
            raw_p[station, target] = math.log(float(value))
    return raw_p, raw_q


def beta_from_raw(raw_beta: np.ndarray) -> np.ndarray:
    raw_beta = np.asarray(raw_beta, dtype=float)
    return 0.5 * math.pi / (1.0 + np.exp(-raw_beta))


def raw_from_beta(beta: float, n_module: int) -> np.ndarray:
    x = np.clip(float(beta) / (0.5 * math.pi), 1.0e-9, 1.0 - 1.0e-9)
    return np.full(n_module, math.log(x / (1.0 - x)), dtype=float)


def station_budget_totals_from_modules(
    b18,
    bm,
    modules: list[PairCombineModule],
    p: np.ndarray,
    q: np.ndarray,
    alpha_core: np.ndarray,
) -> np.ndarray:
    totals = np.sum(p, axis=1) + b18.variants.alpha_by_station(alpha_core, bm)
    for midx, module in enumerate(modules):
        for slot, station in enumerate(module.stations):
            totals[station] += q[midx, slot]
    return totals
