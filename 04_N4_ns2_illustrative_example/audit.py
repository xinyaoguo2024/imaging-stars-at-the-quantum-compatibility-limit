#!/usr/bin/env python3
"""Validate the complete N=4 finite-copy receiver archive."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def close(value: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(value - expected) > tolerance:
        raise RuntimeError(f"numeric mismatch: {value} vs {expected}")


def load_summary(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def main() -> None:
    pvm = load_summary("n4_ns2_phase_pvm_complexwp_summary.json")
    povm2 = load_summary("n4_ns2_phase_povm_q2_complexwp_summary.json")
    povm3 = load_summary("n4_ns3_phase_povm_q2_complexwp_summary.json")
    holevo = load_summary("n4_complexwp_holevo_summary.json")

    close(pvm["joint_A_risk"], 84.4204)
    close(pvm["A_risk_gain"], 1.70575)
    close(pvm["mean_nuisance_aware_edge_gain"], 1.70829)
    close(povm2["joint_A_risk"], 80.2666)
    close(povm2["A_risk_gain"], 1.79402)
    close(povm3["joint_A_risk"], 72.7033)
    close(povm3["A_risk_gain"], 1.98065)
    optimum = holevo["unit_weight_holevo_optimum"]
    close(optimum["unsmoothed_holevo_risk"], 47.8922)
    close(optimum["A_risk_gain_over_repetitive"], 3.00675)

    with np.load(DATA / "n4_ns2_phase_pvm_complexwp.npz") as saved:
        probabilities = saved["probabilities"]
        derivatives = saved["derivatives"]
        if probabilities.shape != (16,) or derivatives.shape != (16, 6):
            raise RuntimeError("incomplete 16-outcome PVM statistics")
        close(float(probabilities.sum()), 1.0, 1e-10)
        close(float(probabilities[0]), 0.07217635, 1e-7)
        close(float(probabilities[10]), 0.04278028, 1e-7)
        if saved["unitary_symmetric"].shape != (10, 10):
            raise RuntimeError("missing 10x10 U_+")
        if saved["unitary_antisymmetric"].shape != (6, 6):
            raise RuntimeError("missing 6x6 U_-")
        expected_plus = -0.4633 - 0.0841j
        expected_minus = -0.4033 + 0.0708j
        if abs(saved["unitary_symmetric"][0, 0] - expected_plus) > 8e-5:
            raise RuntimeError("Appendix symmetric projection vector mismatch")
        if abs(saved["unitary_antisymmetric"][0, 0] - expected_minus) > 8e-5:
            raise RuntimeError("Appendix antisymmetric projection vector mismatch")

    with np.load(DATA / "n4_ns2_phase_povm_q2_complexwp.npz") as saved:
        if saved["probabilities"].shape != (32,) or saved["derivatives"].shape != (32, 6):
            raise RuntimeError("incomplete ns=2 POVM statistics")
        close(float(saved["probabilities"].sum()), 1.0, 1e-10)
    with np.load(DATA / "n4_ns3_phase_povm_q2_complexwp.npz") as saved:
        if saved["probabilities"].shape != (88,) or saved["derivatives"].shape != (88, 6):
            raise RuntimeError("incomplete ns=3 POVM statistics")
        close(float(saved["probabilities"].sum()), 1.0, 1e-10)

    csv_path = DATA / "n4_ns2_pvm_outcomes.csv"
    if not csv_path.is_file():
        raise FileNotFoundError("run export_n4_outcome_table.py first")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16:
        raise RuntimeError("the exported outcome table does not contain 16 rows")
    close(float(rows[0]["probability"]), 0.07217635, 1e-7)
    close(float(rows[10]["probability"]), 0.04278028, 1e-7)
    score_columns = [key for key in rows[0] if key.startswith("score_edge_")]
    if len(score_columns) != 6:
        raise RuntimeError("the outcome table does not expose all six score components")

    for name in (
        "n4_ns2_pvm_outcome_atlas.pdf",
        "n4_finite_receiver_edge_gains.pdf",
        "n4_ns2_collective_receiver.pdf",
    ):
        generated = ROOT / "generated_outputs" / name
        if generated.is_file() and generated.stat().st_size < 10_000:
            raise RuntimeError(f"suspiciously small generated figure: {name}")
    print("PASS 16-outcome PVM: probabilities, scores, U_+, U_-, and Appendix ports")
    print("PASS finite-copy comparison: ns=2 PVM/POVM, ns=3 POVM, Holevo limit")


if __name__ == "__main__":
    main()

