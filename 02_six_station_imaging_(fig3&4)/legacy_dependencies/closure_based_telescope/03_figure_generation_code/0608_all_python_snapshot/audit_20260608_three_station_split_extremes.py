from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_eight_station_cfi_qfi_note_tables_gauge_marginalized as cfi


BUNDLE = Path(__file__).resolve().parents[2]
OUT = BUNDLE / "exploration" / "three_station_split_extremes_20260608"
OUT.mkdir(parents=True, exist_ok=True)


T_VALUES = np.logspace(-4.0, -1.0, 49)
NU_REFS = (0.05, 0.30, 0.80)
FLOORS = (1.0e-8, 2.0e-2)


MODELS = {
    "clean_equal_load": {
        "s_edge": (1.0, 1.0, 1.0),
        "s_direct": (1.0, 1.0, 1.0),
        "pair_background": 0.0,
        "description": "equal station loads, no pair background, no direct excess load",
    },
    "paper_noise_normalized": {
        "s_edge": (1.0, 1.0, 1.0),
        "s_direct": (1.5, 1.5, 1.5),
        "pair_background": 0.5,
        "description": "current Fig. 1 normalization: station load 1, pair background 0.5, direct load 1.5",
    },
}


FAMILIES = {
    "hierarchical": "chi2=t, chi3=t^2",
    "two_weak_equal": "chi2=t, chi3=t",
}


def edge_pair_fisher(g: float, si: float, sj: float, fi: float, fj: float, pair_background: float) -> float:
    if fi <= 0.0 or fj <= 0.0 or g <= 0.0:
        return 0.0
    load = fi * si + fj * sj + pair_background
    return 4.0 * fi * fj * g * g / max(load, 1e-300)


def edge_closure_fisher(
    g12: float,
    g23: float,
    g31: float,
    s_edge: tuple[float, float, float],
    pair_background: float,
    split: tuple[float, float, float] | np.ndarray,
) -> tuple[float, tuple[float, float, float]]:
    """Closure Fisher for one edge-first triangle.

    The split variables are (x1, x2, x3):
    x1: station 1 fraction to edge 12; 1-x1 goes to edge 31.
    x2: station 2 fraction to edge 12; 1-x2 goes to edge 23.
    x3: station 3 fraction to edge 23; 1-x3 goes to edge 31.
    """
    x1, x2, x3 = [float(v) for v in split]
    f12 = edge_pair_fisher(g12, s_edge[0], s_edge[1], x1, x2, pair_background)
    f23 = edge_pair_fisher(g23, s_edge[1], s_edge[2], 1.0 - x2, x3, pair_background)
    f31 = edge_pair_fisher(g31, s_edge[2], s_edge[0], 1.0 - x3, 1.0 - x1, pair_background)
    f_edges = (max(f12, 0.0), max(f23, 0.0), max(f31, 0.0))
    if min(f_edges) <= 0.0:
        return 0.0, f_edges
    return 1.0 / (1.0 / f12 + 1.0 / f23 + 1.0 / f31), f_edges


def optimize_edge_split(
    g12: float,
    g23: float,
    g31: float,
    s_edge: tuple[float, float, float],
    pair_background: float,
    floor: float,
    *,
    seed: int,
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    lo = float(floor)
    hi = 1.0 - lo
    if not (0.0 < lo < hi < 1.0):
        raise ValueError(f"bad split floor: {floor}")

    def score(x: np.ndarray | tuple[float, float, float]) -> float:
        return edge_closure_fisher(g12, g23, g31, s_edge, pair_background, x)[0]

    starts = [
        (0.5, 0.5, 0.5),
        (lo, lo, 0.5),
        (hi, hi, 0.5),
        (0.5, lo, hi),
        (0.5, hi, lo),
        (lo, 0.5, hi),
        (hi, 0.5, lo),
        (0.1, 0.9, 0.5),
        (0.9, 0.1, 0.5),
    ]
    rng = np.random.default_rng(seed)
    starts.extend(tuple(lo + (hi - lo) * rng.random(3)) for _ in range(96))

    best = np.asarray(max(starts, key=score), dtype=float)
    best_score = score(best)

    for width in (0.22, 0.09, 0.035, 0.014, 0.0055, 0.0022, 8.0e-4, 3.0e-4, 1.0e-4, 3.0e-5):
        improved = True
        while improved:
            improved = False
            for idx in range(3):
                for sign in (-1.0, 1.0):
                    cand = best.copy()
                    cand[idx] = min(hi, max(lo, cand[idx] + sign * width))
                    cand_score = score(cand)
                    if cand_score > best_score * (1.0 + 1.0e-14):
                        best = cand
                        best_score = cand_score
                        improved = True

    def to_y(x: np.ndarray) -> np.ndarray:
        z = np.clip((x - lo) / (hi - lo), 1.0e-15, 1.0 - 1.0e-15)
        return np.log(z / (1.0 - z))

    def from_y(y: np.ndarray) -> np.ndarray:
        y = np.clip(y, -60.0, 60.0)
        z = 1.0 / (1.0 + np.exp(-y))
        return lo + (hi - lo) * z

    def objective(y: np.ndarray) -> float:
        return -score(from_y(y))

    simplex = [to_y(best)]
    for idx in range(3):
        y = simplex[0].copy()
        y[idx] += 0.65
        simplex.append(y)
    simplex = np.asarray(simplex)
    values = np.asarray([objective(y) for y in simplex])

    for _ in range(900):
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]
        if np.std(values) < 1.0e-14 * max(1.0, abs(float(values[0]))):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_reflected = objective(reflected)
        if values[0] <= f_reflected < values[-2]:
            simplex[-1] = reflected
            values[-1] = f_reflected
            continue
        if f_reflected < values[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            f_expanded = objective(expanded)
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                values[-1] = f_expanded
            else:
                simplex[-1] = reflected
                values[-1] = f_reflected
            continue
        contracted = centroid + 0.5 * (simplex[-1] - centroid)
        f_contracted = objective(contracted)
        if f_contracted < values[-1]:
            simplex[-1] = contracted
            values[-1] = f_contracted
            continue
        for idx in range(1, 4):
            simplex[idx] = simplex[0] + 0.5 * (simplex[idx] - simplex[0])
            values[idx] = objective(simplex[idx])

    refined = from_y(simplex[int(np.argmin(values))])
    refined_score, refined_edges = edge_closure_fisher(g12, g23, g31, s_edge, pair_background, refined)
    if refined_score > best_score:
        best = refined
        best_score = refined_score
        edge_values = refined_edges
    else:
        _best_score, edge_values = edge_closure_fisher(g12, g23, g31, s_edge, pair_background, best)
    return best_score, tuple(float(v) for v in best), tuple(float(v) for v in edge_values)


def chi_pair(family: str, t: float) -> tuple[float, float]:
    if family == "hierarchical":
        return float(t), float(t * t)
    if family == "two_weak_equal":
        return float(t), float(t)
    raise ValueError(family)


def compute_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for model_name, model in MODELS.items():
        s_edge = model["s_edge"]
        s_direct = model["s_direct"]
        pair_background = float(model["pair_background"])
        for nu_ref in NU_REFS:
            for family in FAMILIES:
                for floor in FLOORS:
                    for t in T_VALUES:
                        chi2, chi3 = chi_pair(family, float(t))
                        g12, g23, g31 = float(nu_ref), float(nu_ref * chi2), float(nu_ref * chi3)
                        f_direct = cfi.triangle_direct_fisher(g12, g23, g31, *s_direct)
                        f_equal, equal_edges = edge_closure_fisher(
                            g12,
                            g23,
                            g31,
                            s_edge,
                            pair_background,
                            (0.5, 0.5, 0.5),
                        )
                        f_opt, split, opt_edges = optimize_edge_split(
                            g12,
                            g23,
                            g31,
                            s_edge,
                            pair_background,
                            floor,
                            seed=20260608 + int(round(1.0e6 * t)) + int(1000 * nu_ref) + (0 if family == "hierarchical" else 19),
                        )
                        rows.append(
                            {
                                "model": model_name,
                                "model_description": str(model["description"]),
                                "family": family,
                                "family_definition": FAMILIES[family],
                                "nu_ref": float(nu_ref),
                                "split_floor": float(floor),
                                "t": float(t),
                                "chi1": 1.0,
                                "chi2": chi2,
                                "chi3": chi3,
                                "g12": g12,
                                "g23": g23,
                                "g31": g31,
                                "f_direct_qfi": float(f_direct),
                                "f_edge_equal": float(f_equal),
                                "f_edge_opt": float(f_opt),
                                "edge_opt_over_direct_snr": math.sqrt(max(f_opt, 0.0) / max(f_direct, 1.0e-300)),
                                "direct_over_edge_opt_snr": math.sqrt(max(f_direct, 0.0) / max(f_opt, 1.0e-300)),
                                "edge_equal_over_direct_snr": math.sqrt(max(f_equal, 0.0) / max(f_direct, 1.0e-300)),
                                "direct_over_edge_equal_snr": math.sqrt(max(f_direct, 0.0) / max(f_equal, 1.0e-300)),
                                "x1_to_12": split[0],
                                "x2_to_12": split[1],
                                "x3_to_23": split[2],
                                "edge_equal_f12": equal_edges[0],
                                "edge_equal_f23": equal_edges[1],
                                "edge_equal_f31": equal_edges[2],
                                "edge_opt_f12": opt_edges[0],
                                "edge_opt_f23": opt_edges[1],
                                "edge_opt_f31": opt_edges[2],
                            }
                        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> dict[str, object]:
    summary_rows = []
    for model in MODELS:
        for family in FAMILIES:
            for nu_ref in NU_REFS:
                for floor in FLOORS:
                    subset = [
                        row
                        for row in rows
                        if row["model"] == model
                        and row["family"] == family
                        and abs(float(row["nu_ref"]) - nu_ref) < 1.0e-12
                        and abs(float(row["split_floor"]) - floor) < 1.0e-18
                    ]
                    edge_over = np.asarray([float(row["edge_opt_over_direct_snr"]) for row in subset])
                    direct_over = np.asarray([float(row["direct_over_edge_opt_snr"]) for row in subset])
                    asym = min(subset, key=lambda row: float(row["t"]))
                    summary_rows.append(
                        {
                            "model": model,
                            "family": family,
                            "nu_ref": nu_ref,
                            "split_floor": floor,
                            "edge_opt_over_direct_snr_min": float(np.min(edge_over)),
                            "edge_opt_over_direct_snr_max": float(np.max(edge_over)),
                            "direct_over_edge_opt_snr_min": float(np.min(direct_over)),
                            "direct_over_edge_opt_snr_max": float(np.max(direct_over)),
                            "asymptotic_t_min": float(asym["t"]),
                            "asymptotic_chi2": float(asym["chi2"]),
                            "asymptotic_chi3": float(asym["chi3"]),
                            "asymptotic_edge_opt_over_direct_snr": float(asym["edge_opt_over_direct_snr"]),
                            "asymptotic_direct_over_edge_opt_snr": float(asym["direct_over_edge_opt_snr"]),
                            "asymptotic_split_x1_to_12": float(asym["x1_to_12"]),
                            "asymptotic_split_x2_to_12": float(asym["x2_to_12"]),
                            "asymptotic_split_x3_to_23": float(asym["x3_to_23"]),
                        }
                    )
    return {
        "definitions": {
            "chi_normalization": "chi1 is normalized to 1; hierarchical means chi2=t, chi3=t^2; two_weak_equal means chi2=chi3=t.",
            "split_variables": "x1 sends station 1 to edge 12, 1-x1 to edge 31; x2 sends station 2 to edge 12, 1-x2 to edge 23; x3 sends station 3 to edge 23, 1-x3 to edge 31.",
            "saturation_metric": "edge_opt_over_direct_snr=1 means optimized edge-first saturates the direct three-mode QFI for this noise model. Values above 1 indicate the two channels are not the same physical noisy measurement problem.",
        },
        "models": MODELS,
        "families": FAMILIES,
        "nu_refs": NU_REFS,
        "floors": FLOORS,
        "summary_rows": summary_rows,
    }


def plot_rows(rows: list[dict[str, float | str]], path_pdf: Path, path_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6), sharex=True, sharey=False, constrained_layout=True)
    colors = {0.05: "#0077b6", 0.30: "#d00000", 0.80: "#f77f00"}
    linestyles = {1.0e-8: "-", 2.0e-2: "--"}
    for row_idx, model in enumerate(MODELS):
        for col_idx, family in enumerate(FAMILIES):
            ax = axes[row_idx, col_idx]
            for nu_ref in NU_REFS:
                for floor in FLOORS:
                    subset = [
                        row
                        for row in rows
                        if row["model"] == model
                        and row["family"] == family
                        and abs(float(row["nu_ref"]) - nu_ref) < 1.0e-12
                        and abs(float(row["split_floor"]) - floor) < 1.0e-18
                    ]
                    subset.sort(key=lambda row: float(row["t"]))
                    t = np.asarray([float(row["t"]) for row in subset])
                    ratio = np.asarray([float(row["edge_opt_over_direct_snr"]) for row in subset])
                    floor_label = r"$p_{\min}=0$" if floor < 1.0e-6 else r"$p_{\min}=0.02$"
                    ax.plot(
                        t,
                        ratio,
                        color=colors[nu_ref],
                        lw=1.15 if floor < 1.0e-6 else 0.95,
                        ls=linestyles[floor],
                        label=rf"$\nu_1={nu_ref:g}$, {floor_label}",
                    )
            ax.axhline(1.0, color="0.2", lw=0.75, ls=":")
            ax.set_xscale("log")
            ax.set_title(f"{model.replace('_', ' ')}\n{FAMILIES[family]}")
            ax.grid(color="0.88", lw=0.5)
            if col_idx == 0:
                ax.set_ylabel(r"SNR(edge opt) / SNR(direct QFI)")
            if row_idx == 1:
                ax.set_xlabel(r"$t$")
            ax.set_ylim(0.52, 1.14)
    axes[0, 1].legend(frameon=False, loc="lower right", ncol=1)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_note(path: Path, summary: dict[str, object]) -> None:
    rows = summary["summary_rows"]

    def find(model: str, family: str, nu_ref: float, floor: float) -> dict[str, object]:
        for row in rows:  # type: ignore[assignment]
            if (
                row["model"] == model
                and row["family"] == family
                and abs(float(row["nu_ref"]) - nu_ref) < 1.0e-12
                and abs(float(row["split_floor"]) - floor) < 1.0e-18
            ):
                return row
        raise KeyError((model, family, nu_ref, floor))

    lines = [
        "# Three-station optimized edge-first extreme-asymmetry audit - 2026-06-08",
        "",
        "Definitions:",
        "",
        "- `chi1` is normalized to 1.",
        "- `hierarchical`: `chi2=t`, `chi3=t^2`, representing `chi1 >> chi2 >> chi3`.",
        "- `two_weak_equal`: `chi2=chi3=t`, representing `chi1 >> chi2 ~ chi3`.",
        "- `edge_opt_over_direct_snr=1` means optimized edge-first saturates the three-mode direct QFI for the same noise model.",
        "- `p_min=0` is implemented as `1e-8`; `p_min=0.02` matches the common split floor in the modular receiver code.",
        "",
        "Main conclusion:",
        "",
        "- In the clean equal-load model, optimized edge-first really does asymptotically saturate direct QFI in the strictly hierarchical limit `chi2=t, chi3=t^2`.",
        "- In the `chi2~chi3` two-weak-edge limit, optimized edge-first does not saturate; the remaining direct/edge SNR gap is finite.",
        "- In the current paper-noise normalization, optimized edge-first can slightly exceed the noisy direct benchmark in the strictly hierarchical limit. This is not a violation of QFI; it means the compared noisy channels are different (`s_direct=1.5`, edge pair background `0.5`).",
        "",
        "Asymptotic values at `t=1e-4`, `nu_ref=0.3`:",
        "",
        "| model | family | floor | edge/direct SNR | direct/edge SNR | split `(x1,x2,x3)` |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model in ("clean_equal_load", "paper_noise_normalized"):
        for family in ("hierarchical", "two_weak_equal"):
            for floor in FLOORS:
                row = find(model, family, 0.30, floor)
                lines.append(
                    "| {model} | {family} | {floor:.0e} | {edge:.4f} | {direct:.4f} | ({x1:.3g}, {x2:.3g}, {x3:.3g}) |".format(
                        model=model,
                        family=family,
                        floor=float(floor),
                        edge=float(row["asymptotic_edge_opt_over_direct_snr"]),
                        direct=float(row["asymptotic_direct_over_edge_opt_snr"]),
                        x1=float(row["asymptotic_split_x1_to_12"]),
                        x2=float(row["asymptotic_split_x2_to_12"]),
                        x3=float(row["asymptotic_split_x3_to_23"]),
                    )
                )
    lines.extend(
        [
            "",
            "Interpretation of split variables:",
            "",
            "- `x1`: station 1 to strong edge 12; `1-x1` to edge 31.",
            "- `x2`: station 2 to strong edge 12; `1-x2` to edge 23.",
            "- `x3`: station 3 to edge 23; `1-x3` to edge 31.",
            "",
            "For the clean hierarchical case, the optimizer drives the strong-edge split fractions down with the weak ratios, so the weak edge receives nearly all of the relevant station budgets. For the two-weak-edge case, station 3 must feed two comparable weak edges, and the joint three-mode SLD keeps a finite advantage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = compute_rows()
    csv_path = OUT / "three_station_split_extreme_scan.csv"
    summary_path = OUT / "three_station_split_extreme_summary.json"
    pdf_path = OUT / "three_station_split_extreme_snr_ratio.pdf"
    png_path = OUT / "three_station_split_extreme_snr_ratio.png"
    note_path = OUT / "three_station_split_extreme_note.md"
    write_csv(csv_path, rows)
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_rows(rows, pdf_path, png_path)
    write_note(note_path, summary)
    print(csv_path)
    print(summary_path)
    print(pdf_path)
    print(png_path)
    print(note_path)
    for row in summary["summary_rows"]:  # type: ignore[index]
        if row["nu_ref"] == 0.3 and row["split_floor"] in FLOORS and row["family"] in ("hierarchical", "two_weak_equal"):
            print(
                "{model} {family} floor={floor:g} edge/direct@tmin={edge:.4f} direct/edge@tmin={direct:.4f}".format(
                    model=row["model"],
                    family=row["family"],
                    floor=float(row["split_floor"]),
                    edge=float(row["asymptotic_edge_opt_over_direct_snr"]),
                    direct=float(row["asymptotic_direct_over_edge_opt_snr"]),
                )
            )


if __name__ == "__main__":
    main()
