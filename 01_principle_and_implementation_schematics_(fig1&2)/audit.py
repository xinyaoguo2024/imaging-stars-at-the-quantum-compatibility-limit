#!/usr/bin/env python3
"""Check the two regenerated schematics against the manuscript PNGs."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAIRS = (
    (
        ROOT / "generated_outputs" / "fig1_principle_schematic.png",
        ROOT / "reference_outputs" / "fig1_principle_schematic.png",
    ),
    (
        ROOT / "generated_outputs" / "fig2_implementation_schematic.png",
        ROOT / "reference_outputs" / "fig2_implementation_schematic.png",
    ),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for generated, reference in PAIRS:
        if not generated.is_file() or not reference.is_file():
            raise FileNotFoundError(f"missing comparison pair: {generated}, {reference}")
        if digest(generated) != digest(reference):
            raise RuntimeError(f"PNG mismatch: {generated.name}")
        print(f"PASS exact PNG: {generated.name}")


if __name__ == "__main__":
    main()
