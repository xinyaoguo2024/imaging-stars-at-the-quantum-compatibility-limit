from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allocation_model import (
    allowed_pair_sinks,
    beta_from_raw,
    project_allocation,
    raw_from_beta,
    raw_from_saved_split,
    station_budget_totals_from_modules,
)
from common_io import (
    FIGURES,
    LOGS,
    NOTES,
    RESULTS,
    SOURCE16,
    good_runtime,
    load_b18,
    load_baselines,
    load_saved_split_payload,
    load_saved_summary18,
    ratio_summary,
    reproduce_saved_columns,
    write_csv,
)
from paircombine_receiver import balanced_paircombine_modules, paircombine_edge_fisher


def wrap_phase(delta: np.ndarray) -> np.ndarray:
    return (np.asarray(delta, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


class PairCombineOptimizer:
    def __init__(
        self,
        b18,
        bm,
        *,
        seed: int,
        hard_edge_floor: bool,
        objective: str,
        target_sigma: np.ndarray,
        checkpoint_path: Path,
        log,
    ):
        self.b18 = b18
        self.bm = bm
        self.rng = np.random.default_rng(seed)
        self.hard_edge_floor = hard_edge_floor
        self.objective = objective
        self.target_sigma = np.asarray(target_sigma, dtype=float)
        self.checkpoint_path = checkpoint_path
        self.log = log
        self.variant = b18.variants.VARIANTS[0]
        self.modules = balanced_paircombine_modules(b18, bm)
        self.triangles = b18.balanced_independent_triangles(bm.n)
        self.labels = b18.loop_labels(self.triangles)
        self.edge_sigma = b18.edge_sigmas_for_triangles(bm, self.triangles)
        self.active_p = [(i, j) for i in range(bm.n) for j in allowed_pair_sinks(b18, bm, i)]
        self.gamma_size = 2 * len(b18.core_remote.CORE) * len(b18.core_remote.REMOTE)
        self.core_cache: dict[tuple[float, ...], np.ndarray] = {}
        self.star_cache: dict[tuple[float, ...], np.ndarray] = {}
        self.eval_count = 0

        self.best_score = -math.inf
        self.best: dict[str, np.ndarray] = {}
        self.best_sigma: np.ndarray | None = None
        self.best_gains: np.ndarray | None = None
        self.best_tag = ""

    def core_for_alpha(self, alpha: np.ndarray) -> np.ndarray:
        key = tuple(float(f"{x:.8f}") for x in alpha)
        cached = self.core_cache.get(key)
        if cached is not None:
            return cached
        core = self.b18.variants.core_direct_edge_fisher_matrix_alpha(self.bm, alpha)
        self.core_cache[key] = core
        return core

    def star_for_split_gamma(self, p: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        key = tuple(float(f"{x:.8f}") for x in np.concatenate([np.ravel(p), np.ravel(gamma)]))
        cached = self.star_cache.get(key)
        if cached is not None:
            return cached
        star, _ = self.b18.remote_star_joint_edge_fisher_matrix(
            self.bm,
            p,
            gamma,
            core_core_handling="nuisance",
        )
        self.star_cache[key] = star
        return star

    def unpack(self, state: dict[str, np.ndarray]):
        alpha = self.b18.variants.alpha_vector_from_raw(state["raw_alpha"], self.variant)
        p, q = project_allocation(self.b18, self.bm, self.modules, state["raw_p"], state["raw_q"], alpha)
        beta = beta_from_raw(state["raw_beta"])
        delta = wrap_phase(state["delta"])
        gamma = np.clip(state["gamma"], 0.0, 1.0)
        return p, q, alpha, gamma, beta, delta

    def sigma_for_state(self, state: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        p, q, alpha, gamma, beta, delta = self.unpack(state)
        core_edge = self.core_for_alpha(alpha)
        star_edge = self.star_for_split_gamma(p, gamma)
        pc_edge = paircombine_edge_fisher(self.b18, self.bm, self.modules, q, beta, delta)
        fisher_q = self.b18.closure_fisher_from_edge_matrix(self.bm, core_edge + star_edge + pc_edge)
        sigma = self.b18.root_sigmas(self.bm, fisher_q)
        unpacked = {"p": p, "q": q, "alpha": alpha, "gamma": gamma, "beta": beta, "delta": delta}
        return sigma, unpacked

    def score_sigma(self, sigma: np.ndarray) -> tuple[float, np.ndarray]:
        gains = self.edge_sigma / np.maximum(sigma, 1.0e-300)
        ratio_objectives = {"ratio-only", "worst-ratio"}
        if self.hard_edge_floor and self.objective not in ratio_objectives and float(np.min(gains)) < 1.0 - 1.0e-10:
            return -math.inf, gains
        if self.objective == "direct-score":
            return self.b18.direct_schedule_score(self.edge_sigma, sigma), gains
        if self.objective == "match-direct":
            near_over_direct = self.target_sigma / np.maximum(sigma, 1.0e-300)
            log_r = np.log(np.maximum(near_over_direct, 1.0e-300))
            overshoot = np.maximum(log_r, 0.0)
            undershoot = np.maximum(-log_r, 0.0)
            score = (
                -float(np.mean(log_r * log_r))
                -0.80 * float(np.var(log_r))
                -1.20 * float(np.mean(overshoot * overshoot))
                -0.80 * float(np.mean(undershoot * undershoot))
                -0.15 * float(np.max(np.abs(log_r)) ** 2)
            )
            return score, gains
        if self.objective == "ratio-only":
            near_over_direct = self.target_sigma / np.maximum(sigma, 1.0e-300)
            log_r = np.log(np.maximum(near_over_direct, 1.0e-300))
            return -float(np.mean(log_r * log_r)), gains
        if self.objective == "worst-ratio":
            near_over_direct = self.target_sigma / np.maximum(sigma, 1.0e-300)
            log_r = np.log(np.maximum(near_over_direct, 1.0e-300))
            max_abs = float(np.max(np.abs(log_r)))
            rms = float(np.sqrt(np.mean(log_r * log_r)))
            var = float(np.var(log_r))
            bias = float(abs(np.mean(log_r)))
            score = -(max_abs + 0.20 * rms + 0.05 * math.sqrt(var) + 0.02 * bias)
            return score, gains
        raise ValueError(f"Unknown objective {self.objective!r}")

    def write_checkpoint(self) -> None:
        if self.best_sigma is None or self.best_gains is None or not self.best:
            return
        payload = {
            "score": float(self.best_score),
            "best_tag": self.best_tag,
            "objective": self.objective,
            "eval_count": int(self.eval_count),
            "state": {
                key: np.asarray(self.best[key]).tolist()
                for key in ("raw_p", "raw_q", "raw_alpha", "gamma", "raw_beta", "delta")
            },
            "unpacked": {
                key: np.asarray(self.best[key]).tolist()
                for key in ("p", "q", "alpha", "beta")
            },
            "sigma": np.asarray(self.best_sigma).tolist(),
            "gains": np.asarray(self.best_gains).tolist(),
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2) + "\n")

    def try_state(self, state: dict[str, np.ndarray], tag: str) -> bool:
        self.eval_count += 1
        sigma, unpacked = self.sigma_for_state(state)
        score, gains = self.score_sigma(sigma)
        if np.isfinite(score) and score > self.best_score + 1.0e-13:
            self.best_score = float(score)
            self.best = {key: np.asarray(value).copy() for key, value in {**state, **unpacked}.items()}
            self.best_sigma = sigma
            self.best_gains = gains
            self.best_tag = tag
            self.log(
                f"new best {score:.8g} from {tag}: "
                f"gain min/mean/max={np.min(gains):.4f}/{np.mean(gains):.4f}/{np.max(gains):.4f}"
            )
            self.write_checkpoint()
            return True
        return False

    def saved_state(self, *, gamma: np.ndarray, q_raw_value: float = -30.0) -> dict[str, np.ndarray]:
        payload = load_saved_split_payload()
        p_saved = np.asarray(payload["split_matrix"], dtype=float)
        alpha = np.asarray(payload["alpha_core"], dtype=float)
        raw_p, raw_q = raw_from_saved_split(self.b18, self.bm, self.modules, p_saved, q_raw_value=q_raw_value)
        return {
            "raw_p": raw_p,
            "raw_q": raw_q,
            "raw_alpha": self.b18.variants.raw_vector_from_alpha(alpha, self.variant, self.bm),
            "gamma": np.asarray(gamma, dtype=float).reshape(self.gamma_size),
            "raw_beta": raw_from_beta(math.pi / 4.0, len(self.modules)),
            "delta": np.zeros(len(self.modules), dtype=float),
        }

    def checkpoint_state(self, path: Path) -> dict[str, np.ndarray] | None:
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        raw = payload.get("state", {})
        keys = ("raw_p", "raw_q", "raw_alpha", "gamma", "raw_beta", "delta")
        if not all(key in raw for key in keys):
            return None
        return {key: np.asarray(raw[key], dtype=float) for key in keys}

    def initialize(self) -> None:
        checkpoint_candidates = [self.checkpoint_path]
        ratio_checkpoint = RESULTS / "checkpoint_ratio-only.json"
        if self.objective == "worst-ratio" and ratio_checkpoint != self.checkpoint_path:
            checkpoint_candidates.append(ratio_checkpoint)
        for path in checkpoint_candidates:
            state = self.checkpoint_state(path)
            if state is not None:
                self.try_state(state, f"checkpoint:{path.name}")

        summary = load_saved_summary18()
        gamma_ind = np.asarray(summary["independent_gamma_vector"], dtype=float)
        gamma_zero = np.zeros_like(gamma_ind)
        for tag, gamma in (("saved_current_near_gamma0", gamma_zero), ("saved_independent_gamma", gamma_ind)):
            self.try_state(self.saved_state(gamma=gamma), tag)

        source16 = SOURCE16 / "results" / "near_match_direct_split_payload.json"
        if source16.exists():
            payload = json.loads(source16.read_text())
            p_saved = np.asarray(payload["split_matrix"], dtype=float)
            alpha = np.asarray(payload.get("alpha_core", [float(payload.get("alpha", 0.15))] * 3), dtype=float)
            raw_p, raw_q = raw_from_saved_split(self.b18, self.bm, self.modules, p_saved, q_raw_value=-30.0)
            state = {
                "raw_p": raw_p,
                "raw_q": raw_q,
                "raw_alpha": self.b18.variants.raw_vector_from_alpha(alpha, self.variant, self.bm),
                "gamma": gamma_ind.copy(),
                "raw_beta": raw_from_beta(math.pi / 4.0, len(self.modules)),
                "delta": np.zeros(len(self.modules), dtype=float),
            }
            self.try_state(state, "folder16_split_gamma_independent")

        if not self.best:
            raise RuntimeError("No feasible starting state found.")

    def paircombine_activation_probes(self, n_probe: int) -> None:
        if n_probe <= 0:
            return
        base_state = {key: np.asarray(self.best[key]).copy() for key in ("raw_p", "raw_q", "raw_alpha", "gamma", "raw_beta", "delta")}
        for midx in range(min(n_probe, len(self.modules))):
            for raw_level in (-4.0, -2.0, 0.0):
                state = {key: value.copy() for key, value in base_state.items()}
                state["raw_q"][midx, :] = raw_level
                state["raw_beta"][midx] = 0.0
                for delta in (0.0, math.pi):
                    state["delta"][midx] = delta
                    self.try_state(state, f"activate_{self.modules[midx].label}_q{raw_level:g}_d{delta:.1f}")

    def coordinate_refine(self, passes: int) -> None:
        for outer in range(passes):
            for width in (0.35, 0.14, 0.055):
                improved = True
                while improved:
                    improved = False
                    for key, size in (("raw_alpha", len(self.best["raw_alpha"])), ("gamma", self.gamma_size)):
                        for idx in range(size):
                            for sign in (-1.0, 1.0):
                                state = {k: np.asarray(v).copy() for k, v in self.best.items() if k in ("raw_p", "raw_q", "raw_alpha", "gamma", "raw_beta", "delta")}
                                state[key][idx] += sign * width
                                if key == "gamma":
                                    state[key][idx] = float(np.clip(state[key][idx], 0.0, 1.0))
                                improved |= self.try_state(state, f"{key}_o{outer}_w{width:g}_i{idx}")

                    for i, j in self.active_p:
                        for sign in (-1.0, 1.0):
                            state = {k: np.asarray(v).copy() for k, v in self.best.items() if k in ("raw_p", "raw_q", "raw_alpha", "gamma", "raw_beta", "delta")}
                            state["raw_p"][i, j] += sign * width
                            improved |= self.try_state(state, f"raw_p_o{outer}_w{width:g}_{i}_{j}")

                    for midx in range(len(self.modules)):
                        for field in ("raw_q", "raw_beta", "delta"):
                            slots = range(3) if field == "raw_q" else range(1)
                            for slot in slots:
                                for sign in (-1.0, 1.0):
                                    state = {k: np.asarray(v).copy() for k, v in self.best.items() if k in ("raw_p", "raw_q", "raw_alpha", "gamma", "raw_beta", "delta")}
                                    if field == "raw_q":
                                        state[field][midx, slot] += sign * 2.0 * width
                                    else:
                                        state[field][midx] += sign * width
                                    improved |= self.try_state(state, f"{field}_o{outer}_w{width:g}_m{midx}_s{slot}")
                self.log(f"finished refine outer={outer} width={width:g}, best={self.best_score:.8g}")

    def summary(self) -> dict[str, object]:
        if self.best_sigma is None or self.best_gains is None:
            raise RuntimeError("No optimization result")
        totals = station_budget_totals_from_modules(
            self.b18,
            self.bm,
            self.modules,
            self.best["p"],
            self.best["q"],
            self.best["alpha"],
        )
        q_by_module = np.sum(self.best["q"], axis=1)
        top = np.argsort(q_by_module)[::-1][:10]
        return {
            "score": float(self.best_score),
            "best_tag": self.best_tag,
            "objective": self.objective,
            "eval_count": int(self.eval_count),
            "core_cache_size": int(len(self.core_cache)),
            "star_cache_size": int(len(self.star_cache)),
            "alpha_core": [float(x) for x in self.best["alpha"]],
            "station_budget_total": {str(self.bm.names[i]): float(totals[i]) for i in range(self.bm.n)},
            "station_budget_max_abs_error": float(np.max(np.abs(totals - 1.0))),
            "gain_vs_edge": ratio_summary(self.best_gains),
            "top_paircombine_modules": {
                self.modules[idx].label: {
                    "total_q": float(q_by_module[idx]),
                    "fractions": [float(x) for x in self.best["q"][idx]],
                    "beta": float(self.best["beta"][idx]),
                    "delta": float(self.best["delta"][idx]),
                }
                for idx in top
                if q_by_module[idx] > 1.0e-8
            },
        }


def save_outputs(opt: PairCombineOptimizer, baselines: dict[str, object], reproduction: dict[str, float], elapsed: float) -> None:
    suffix = opt.objective.replace("-", "_")
    loop_csv = RESULTS / f"paircombine_strict_near_loop_gains_{suffix}.csv"
    module_csv = RESULTS / f"paircombine_modules_{suffix}.csv"
    summary_json = RESULTS / f"paircombine_strict_near_summary_{suffix}.json"
    diagnostic_png = FIGURES / f"paircombine_strict_near_loop_gain_diagnostic_{suffix}.png"
    diagnostic_pdf = FIGURES / f"paircombine_strict_near_loop_gain_diagnostic_{suffix}.pdf"
    labels = list(baselines["labels"])
    strict_gain = np.asarray(opt.best_gains, dtype=float)
    strict_sigma = np.asarray(opt.best_sigma, dtype=float)
    direct_gain = np.asarray(baselines["direct_gain"], dtype=float)
    direct_sigma = np.asarray(baselines["direct_sigma"], dtype=float)
    rows = []
    for idx, label in enumerate(labels):
        tri = opt.triangles[idx]
        n_remote = sum(station in opt.b18.core_remote.REMOTE for station in tri)
        rows.append(
            {
                "loop": label,
                "loop_class": "core_only" if n_remote == 0 else ("one_remote" if n_remote == 1 else "two_remote"),
                "gain_direct_optimized_vs_edge": float(direct_gain[idx]),
                "gain_old_remote_star_independent_vs_edge": float(baselines["old_star_gain"][idx]),
                "gain_paircombine_strict_near_vs_edge": float(strict_gain[idx]),
                "ratio_paircombine_near_to_direct_snr": float(direct_sigma[idx] / max(strict_sigma[idx], 1.0e-300)),
                "log_ratio_paircombine_near_to_direct": float(np.log(max(direct_sigma[idx] / max(strict_sigma[idx], 1.0e-300), 1.0e-300))),
                "rms_paircombine_strict_near_rad": float(strict_sigma[idx]),
            }
        )
    write_csv(loop_csv, rows)
    write_csv(RESULTS / "paircombine_strict_near_loop_gains.csv", rows)

    module_rows = []
    q = opt.best["q"]
    for idx, module in enumerate(opt.modules):
        module_rows.append(
            {
                "module": module.label,
                "loop": "-".join(f"S{i + 1}" for i in module.loop),
                "stations": ",".join(f"S{i + 1}" for i in module.stations),
                "q0": float(q[idx, 0]),
                "q1": float(q[idx, 1]),
                "q2": float(q[idx, 2]),
                "q_total": float(np.sum(q[idx])),
                "beta": float(opt.best["beta"][idx]),
                "delta": float(opt.best["delta"][idx]),
            }
        )
    write_csv(module_csv, module_rows)
    write_csv(RESULTS / "paircombine_modules.csv", module_rows)

    summary = {
        "model": "strict_physical_near_with_paircombine",
        "receiver": "station-side p/alpha/gamma plus pair-combine modules (two stations coherently combined, then beaten with the third station)",
        "objective": opt.objective,
        "objective_detail": (
            "direct-score maximizes the direct optimized schedule score on gain vs edge; "
            "match-direct penalizes per-loop log(SNR_near/SNR_direct_optimized) deviations, including overshoot; "
            "ratio-only uses only -mean(log(SNR_near/SNR_direct_optimized)^2); "
            "worst-ratio is dominated by max_l |log(SNR_near,l/SNR_direct,l)|, with small RMS/variance tie-breakers"
        ),
        "reproduction_check": reproduction,
        "elapsed_s": float(elapsed),
        "optimized": opt.summary(),
        "direct_gain_vs_edge": ratio_summary(direct_gain),
        "old_remote_star_independent_gain_vs_edge": ratio_summary(np.asarray(baselines["old_star_gain"], dtype=float)),
        "paircombine_strict_near_gain_vs_edge": ratio_summary(strict_gain),
        "paircombine_near_to_direct_snr": ratio_summary(direct_sigma / np.maximum(strict_sigma, 1.0e-300)),
        "files": {
            "loop_gains_csv": str(loop_csv),
            "module_csv": str(module_csv),
            "summary_json": str(summary_json),
            "diagnostic_png": str(diagnostic_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS / "paircombine_strict_near_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.7, 4.3), constrained_layout=True)
    ax.bar(x - 0.5 * width, direct_gain, width, color="#9e2119", label="direct optimized")
    ax.bar(x + 0.5 * width, strict_gain, width, color="#145a9e", label="pair-combine strict near")
    ax.axhline(math.sqrt(1.5), color="0.25", lw=1.2, ls="--", label=r"$\sqrt{1.5}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(1.0, 1.4)
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.set_title("Strict physical near with pair-combine taps")
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(diagnostic_png, dpi=260)
    fig.savefig(diagnostic_pdf)
    fig.savefig(FIGURES / "paircombine_strict_near_loop_gain_diagnostic.png", dpi=260)
    fig.savefig(FIGURES / "paircombine_strict_near_loop_gain_diagnostic.pdf")
    plt.close(fig)

    note = [
        "# Strict Physical Near With Pair-Combine Taps",
        "",
        "This run optimizes independent station-side split variables.  The variables are not tied across stations.",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    (NOTES / "paircombine_strict_near_note.md").write_text("\n".join(note) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--activation-probes", type=int, default=30)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--objective", choices=("direct-score", "match-direct", "ratio-only", "worst-ratio"), default="match-direct")
    parser.add_argument("--allow-below-edge", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"[{time.time() - t0:8.2f}s] {message}"
        log_lines.append(line)
        print(line, flush=True)

    b18, bm = load_b18()
    baselines = load_baselines()
    with good_runtime(b18):
        reproduction = reproduce_saved_columns(b18, bm, baselines)
        log(f"reproduced folder18 columns: {reproduction}")
        opt = PairCombineOptimizer(
            b18,
            bm,
            seed=args.seed,
            hard_edge_floor=not args.allow_below_edge,
            objective=args.objective,
            target_sigma=np.asarray(baselines["direct_sigma"], dtype=float),
            checkpoint_path=RESULTS / f"checkpoint_{args.objective}.json",
            log=log,
        )
        opt.initialize()
        opt.paircombine_activation_probes(args.activation_probes)
        opt.coordinate_refine(args.passes)

    elapsed = time.time() - t0
    save_outputs(opt, baselines, reproduction, elapsed)
    (LOGS / "run_optimize_paircombine.log").write_text("\n".join(log_lines) + "\n")
    print(json.dumps(opt.summary(), indent=2))


if __name__ == "__main__":
    main()
