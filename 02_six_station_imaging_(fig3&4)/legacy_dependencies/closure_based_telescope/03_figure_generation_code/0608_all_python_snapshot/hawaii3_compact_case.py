from __future__ import annotations

import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import plot_augmented_existing_telescope_closure_networks as aug


def make_hawaii3_compact_remote_case() -> aug.NetworkCase:
    """Maunakea top-four core plus three compact 5 m remote stations.

    The three remote stations keep the angular directions of the previously
    used far-out Hawaii+3 stations, but their radii are reset to 2, 4, and
    9 km from the Maunakea coordinate origin.  This preserves a comparable
    topology while moving the Fourier coverage to lower spatial frequencies.
    """
    full = amp_rml.load_maunakea_case()
    core = [tel for tel in full.telescopes if not tel.is_added]
    added = [tel for tel in full.telescopes if tel.is_added]
    far_template = added[-3:]
    target_radii_km = np.array([2.0, 4.0, 9.0], dtype=float)
    remotes: list[aug.Telescope] = []
    for radius, template in zip(target_radii_km, far_template):
        direction = np.array([template.x_km, template.y_km], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError(f"Remote template {template.name} has zero radius")
        xy = radius * direction / norm
        remotes.append(
            aug.Telescope(
                f"new 5 m r={radius:g}km compact",
                float(xy[0]),
                float(xy[1]),
                5.0,
                True,
            )
        )
    return aug.NetworkCase(
        key="hawaii_top4_remote3_compact_r2_4_9_ngc4151",
        title="Maunakea top-four core + compact 2/4/9 km 5 m outstations",
        latitude_deg=full.latitude_deg,
        center_latlon=full.center_latlon,
        telescopes=core + remotes,
        hub_km=full.hub_km,
        optimization_score=full.optimization_score,
    )
