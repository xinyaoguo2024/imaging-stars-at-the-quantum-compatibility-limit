#!/usr/bin/env python3
"""Diagnose and gauge-register independently reconstructed spectral images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_promoted_povm_rml as fig2  # noqa: E402


def normalized(image: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(image, dtype=float), 0.0)
    total = float(np.sum(out))
    if total <= 0.0:
        raise ValueError("image has no positive flux")
    return out / total


def translate_without_wrap(image: np.ndarray, shift_yx: tuple[float, float]) -> np.ndarray:
    image = normalized(image)
    shift_y, shift_x = shift_yx
    grid = np.arange(image.shape[0], dtype=float)
    source_x = grid - float(shift_x)
    shifted_x = np.stack(
        [np.interp(source_x, grid, row, left=0.0, right=0.0) for row in image]
    )
    source_y = grid - float(shift_y)
    shifted = np.stack(
        [
            np.interp(source_y, grid, shifted_x[:, column], left=0.0, right=0.0)
            for column in range(shifted_x.shape[1])
        ],
        axis=1,
    )
    return normalized(shifted)


def gauge_location(
    image: np.ndarray,
    axis_uas: np.ndarray,
    method: str,
    core_radius_uas: float,
) -> tuple[float, float]:
    image = normalized(image)
    yy, xx = np.meshgrid(axis_uas, axis_uas, indexing="ij")
    core = xx * xx + yy * yy <= core_radius_uas * core_radius_uas
    if method == "core_peak":
        masked = np.where(core, image, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(masked)), image.shape)
        return float(axis_uas[iy]), float(axis_uas[ix])
    if method == "core_centroid":
        weight = image * core
    elif method == "centroid":
        weight = image
    else:
        raise ValueError(f"unknown gauge method {method!r}")
    weight = weight / np.sum(weight)
    return float(np.sum(weight * yy)), float(np.sum(weight * xx))


def register_bands(
    images: np.ndarray,
    axis_uas: np.ndarray,
    method: str,
    core_radius_uas: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_uas = float(axis_uas[1] - axis_uas[0])
    registered = []
    locations = []
    for image in images:
        y_uas, x_uas = gauge_location(image, axis_uas, method, core_radius_uas)
        locations.append((y_uas, x_uas))
        registered.append(
            translate_without_wrap(image, (-y_uas / pixel_uas, -x_uas / pixel_uas))
        )
    return np.stack(registered), np.asarray(locations, dtype=float)


def weighted_stack(images: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return normalized(np.tensordot(weights, images, axes=(0, 0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--strategy", default="promoted_singlecopy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    cache_path = Path(summary["selected_display_cache_npz"])
    with np.load(cache_path, allow_pickle=False) as cache:
        truth = np.asarray(cache["truth"], dtype=float)
        axis = np.asarray(cache["axis_uas"], dtype=float)
        bands = np.asarray(cache[f"{args.strategy}_band_images"], dtype=float)
    weights = np.asarray(summary["wavelength_bin_photon_weights"], dtype=float)

    stacks = {"unregistered": weighted_stack(bands, weights)}
    locations: dict[str, np.ndarray] = {}
    registered_sets: dict[str, np.ndarray] = {}
    for method in ("centroid", "core_centroid", "core_peak"):
        registered, method_locations = register_bands(bands, axis, method)
        registered_sets[method] = registered
        locations[method] = method_locations
        stacks[method] = weighted_stack(registered, weights)

    metrics = {
        name: {
            key: float(value)
            for key, value in fig2.morph.amp_rml.metrics_for(image, truth, axis).items()
        }
        for name, image in stacks.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        truth=truth,
        axis_uas=axis,
        band_images=bands,
        weights=weights,
        **{f"stack_{key}": value for key, value in stacks.items()},
        **{f"locations_{key}": value for key, value in locations.items()},
    )

    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axes = plt.subplots(2, 5, figsize=(10.0, 4.25), constrained_layout=True)
    for index, (ax, image) in enumerate(zip(axes.ravel(), bands)):
        ax.imshow(fig2.log_display(image), origin="lower", extent=extent, cmap="inferno")
        y_peak, x_peak = locations["core_peak"][index]
        ax.plot(x_peak, y_peak, marker="+", color="cyan", ms=7, mew=1.0)
        ax.set_xlim(-42, 42)
        ax.set_ylim(-42, 42)
        ax.set_title(f"{605 + 10 * index} nm\npeak=({x_peak:.1f},{y_peak:.1f})")
        ax.tick_params(labelsize=7)
    montage = args.output.with_name(args.output.name + "_bands.png")
    fig.savefig(montage, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.4), constrained_layout=True)
    display = [("truth", truth)] + list(stacks.items())
    for ax, (name, image) in zip(axes, display):
        ax.imshow(fig2.log_display(image), origin="lower", extent=extent, cmap="inferno")
        ax.set_xlim(-65, 65)
        ax.set_ylim(-65, 65)
        if name == "truth":
            ax.set_title("truth")
        else:
            item = metrics[name]
            ax.set_title(
                f"{name.replace('_', ' ')}\n"
                f"BLR={item['blr_corr']:.3f}, all={item['global_corr']:.3f}"
            )
        ax.tick_params(labelsize=7)
    comparison = args.output.with_name(args.output.name + "_comparison.png")
    fig.savefig(comparison, dpi=220)
    plt.close(fig)

    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "summary": str(args.summary.resolve()),
                "cache": str(cache_path.resolve()),
                "strategy": args.strategy,
                "metrics": metrics,
                "locations_yx_uas": {
                    key: value.tolist() for key, value in locations.items()
                },
                "outputs": {
                    "npz": str(npz_path.resolve()),
                    "band_montage": str(montage.resolve()),
                    "comparison": str(comparison.resolve()),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(json_path)
    print(montage)
    print(comparison)


if __name__ == "__main__":
    main()
