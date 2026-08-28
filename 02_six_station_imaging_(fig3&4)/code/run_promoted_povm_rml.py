from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
RUN_ROOT = THIS_DIR.parent
DEPENDENCY_ROOT = RUN_ROOT / "legacy_dependencies"
os.environ.setdefault(
    "VLBI_HOMODYNE_ROOT",
    str(DEPENDENCY_ROOT / "Homodyne_telescope_official"),
)
# The historical RML/source-model modules are read-only dependencies.  All new
# results stay under RUN_ROOT; the legacy workspace is never modified.
WORKSPACE = Path(
    os.environ.get(
        "VLBI_LEGACY_WORKSPACE",
        str(DEPENDENCY_ROOT / "closure_based_telescope"),
    )
)
CORE_DIR = WORKSPACE / "03_figure_generation_code" / "0608_core_modules"
SNAPSHOT_DIR = WORKSPACE / "03_figure_generation_code" / "0608_all_python_snapshot"
BUNDLE = WORKSPACE
COMMON_DIR = DEPENDENCY_ROOT / "promoted_povm_common"
# Insert low-priority paths first.  The local audited modules win, followed by
# the mutually compatible legacy snapshot before older core-module variants.
for _path in (COMMON_DIR, CORE_DIR, SNAPSHOT_DIR, THIS_DIR):
    _path_text = str(_path)
    if _path_text in sys.path:
        sys.path.remove(_path_text)
    sys.path.insert(0, _path_text)

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import hawaii3_compact_case
import make_all_closure_global_benchmark_note as closure_bm
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles
import run_hawaii3_rml_strategy_comparison as strategy_run
import run_rml_validation_pipeline as val
import test_fig3_split_objective_imaging as split_sim
import core4_joint_remote_split_design as core4_remote
import local_receiver_design as receiver_design


ROOT = Path(os.environ["VLBI_HOMODYNE_ROOT"])
MORPH_DIR = ROOT / "rml_remote3_source_morphology_optimization_20260525"
if str(MORPH_DIR) not in sys.path:
    sys.path.insert(0, str(MORPH_DIR))
import optimize_remote3_source_morphology_priors as morph  # noqa: E402


DEFAULT_OUTPUT_ROOT = RUN_ROOT / "results"
CORRECTED_ROOT = Path(os.environ.get("FIG2_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
OUT = CORRECTED_ROOT / "rml_outputs"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE18 = BUNDLE / "18_balanced_10loop_independent_set_20260611"
SOURCE19 = BUNDLE / "19_upgraded_near_smallblock_20260611"
SOURCE27 = BUNDLE / "27_strict_physical_near_paircombine_20260613"
SOURCE30 = BUNDLE / "30_raw_direct_jcl_pipeline_20260616"
GENERALIZED_DIRECT_SCHEDULE = Path(
    os.environ.get(
        "GENERALIZED_DIRECT_STATION_SCHEDULE",
        str(SOURCE30 / "results" / "balanced10_generalized_station_budget_rawje_optimized_station_schedule.json"),
    )
)
PAIRCOMBINE_CHECKPOINT = Path(
    os.environ.get(
        "PAIRCOMBINE_PROFILE_CHECKPOINT",
        str(SOURCE27 / "results" / "checkpoint_worst-ratio.json"),
    )
)

SIX_CORE = tuple(range(3))
SIX_REMOTE = tuple(range(3, 6))


def configure_six_station_constants() -> None:
    """Use old S2-S7 as a six-station case: three compact core stations plus three remotes."""
    split_sim.CORE_STATIONS = SIX_CORE
    split_sim.REMOTE_STATIONS = SIX_REMOTE
    core4_remote.CORE = SIX_CORE
    core4_remote.REMOTE = SIX_REMOTE


def make_six_station_case() -> aug.NetworkCase:
    seven = scale_remote_coordinates(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    telescopes = list(seven.telescopes[1:])
    return aug.NetworkCase(
        key="six_station_oldS2_to_oldS7_compact_remote3",
        title="Six-station subset: old S2-S7 relabeled S1-S6",
        latitude_deg=seven.latitude_deg,
        center_latlon=seven.center_latlon,
        telescopes=telescopes,
        hub_km=seven.hub_km,
        optimization_score=seven.optimization_score,
    )


def percent_tag(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def value_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def occupancy_tag(value: float) -> str:
    return percent_tag(value) if value >= 1.0e-4 else value_tag(value)


def seconds_tag(value: float) -> str:
    if value < 1.0:
        ms = value * 1000.0
        if abs(ms - round(ms)) < 1.0e-9:
            return f"{int(round(ms))}ms"
        return f"{value_tag(ms)}ms"
    text = value_tag(value)
    return f"{text}s"


def cadence_tag(value: float) -> str:
    if abs(value % 60.0) < 1.0e-9:
        return f"{value_tag(value / 60.0)}min"
    return seconds_tag(value)


FIG2_EXPOSURE_S = float(os.environ.get("FIG2_EXPOSURE_S", os.environ.get("EXPOSURE_S", "0.050")))
BASE_EXPOSURE_S = FIG2_EXPOSURE_S
EXPOSURE_SCALE = float(os.environ.get("EXPOSURE_SCALE", "1.0"))
KEEP_EVERY_TIME_ROW = int(os.environ.get("KEEP_EVERY_TIME_ROW", "1"))
OBSERVING_DAYS = int(os.environ.get("BROAD_PLUME_OBSERVING_DAYS", os.environ.get("OBSERVING_DAYS", "1")))
N_TIME_WINDOWS_RUN = int(os.environ.get("FIG2_N_TIME_WINDOWS", os.environ.get("N_TIME_WINDOWS", "36")))
SAMPLE_CADENCE_S_RUN = float(
    os.environ.get(
        "FIG2_SAMPLE_CADENCE_S",
        os.environ.get("SAMPLE_CADENCE_S", "900.0"),
    )
)
EXPOSURE_GAP_S_RUN = float(
    os.environ.get(
        "FIG2_EXPOSURE_GAP_S",
        os.environ.get("EXPOSURE_GAP_S", str(max(SAMPLE_CADENCE_S_RUN - FIG2_EXPOSURE_S, 0.0))),
    )
)
SAMPLE_CADENCE_S_RUN = FIG2_EXPOSURE_S + EXPOSURE_GAP_S_RUN
FOURIER_COVERAGE_H_RUN = max(N_TIME_WINDOWS_RUN - 1, 0) * SAMPLE_CADENCE_S_RUN / 3600.0
LAMBDA_MIN_NM_RUN = float(os.environ.get("FIG2_LAMBDA_MIN_NM", os.environ.get("LAMBDA_MIN_NM", "600.0")))
LAMBDA_MAX_NM_RUN = float(os.environ.get("FIG2_LAMBDA_MAX_NM", os.environ.get("LAMBDA_MAX_NM", "700.0")))
LAMBDA_STEP_NM_RUN = float(os.environ.get("FIG2_LAMBDA_STEP_NM", os.environ.get("LAMBDA_STEP_NM", "10.0")))
EPS_STATION_RUN = float(os.environ.get("EPS_STATION", "1e-9"))
EPS_PAIR_RUN = float(os.environ.get("EPS_PAIR", "0.0"))
EPS_DIRECT_EXTRA_RUN = float(os.environ.get("EPS_DIRECT_EXTRA", "0.0"))
POST_AVERAGE_DRIFT_STD_RUN = float(
    os.environ.get(
        "FIG2_POST_AVERAGE_DRIFT_STD",
        os.environ.get("POST_AVERAGE_DRIFT_STD", str(np.pi / 5.0)),
    )
)
REMOTE_X_SCALE = float(os.environ.get("HAWAII3_REMOTE_X_SCALE", "1.0"))
REMOTE_Y_SCALE = float(os.environ.get("HAWAII3_REMOTE_Y_SCALE", "0.85"))
EXISTING_COUPLED_AREA_FRACTION = float(os.environ.get("FIG2_EXISTING_COUPLING", "0.1"))
REMOTE_DIAMETER_M = float(os.environ.get("FIG2_REMOTE_DIAMETER_M", "6.0"))
PHOTON_COLLECTION_EFFICIENCY = float(
    os.environ.get("FIG2_COLLECTION_EFFICIENCY", "0.02")
)
if not (0.0 < PHOTON_COLLECTION_EFFICIENCY <= 1.0):
    raise ValueError("FIG2_COLLECTION_EFFICIENCY must lie in (0,1]")
SOURCE_VARIANT_KEY = os.environ.get("FIG2_SOURCE_VARIANT", "ngc4151_core_disk_blr_halpha").strip()
FIG2_RNG_SEED = int(os.environ.get("FIG2_RNG_SEED", os.environ.get("RNG_SEED", "20260529")))
RML_FIT_N_PIX_RUN = int(os.environ.get("RML_FIT_N_PIX", "40"))
RML_ADAM_ITER_RUN = int(os.environ.get("RML_ADAM_ITER", "2600"))
RML_ADAM_LR_RUN = float(os.environ.get("RML_ADAM_LR", "0.010"))
RML_PRIOR_WEIGHT_RUN = float(os.environ.get("RML_PRIOR_WEIGHT", "0.01"))
RML_TV_WEIGHT_RUN = float(os.environ.get("RML_TV_WEIGHT", "0.01"))
RML_ENTROPY_WEIGHT_RUN = float(os.environ.get("RML_ENTROPY_WEIGHT", "0.005"))
RML_CHI2_THRESHOLD = float(os.environ.get("RML_CHI2_THRESHOLD", "3.0"))
RML_CHI2_MAX_PASSES = int(os.environ.get("RML_CHI2_MAX_PASSES", "4"))
RML_PHASE_CHI2_SELECTION = os.environ.get("RML_PHASE_CHI2_SELECTION", "target").strip().lower()
RML_PHASE_CHI2_TARGET = float(os.environ.get("RML_PHASE_CHI2_TARGET", "1.0"))
RML_PHASE_CHI2_MIN = float(os.environ.get("RML_PHASE_CHI2_MIN", "0.90"))
RML_PHASE_CHI2_MAX = float(os.environ.get("RML_PHASE_CHI2_MAX", "1.15"))
RML_AMP_CHI2_TARGET = float(os.environ.get("RML_AMP_CHI2_TARGET", "1.0"))
RML_AMP_CHI2_MIN = float(os.environ.get("RML_AMP_CHI2_MIN", "0.90"))
RML_AMP_CHI2_MAX = float(os.environ.get("RML_AMP_CHI2_MAX", "1.15"))
PER_BAND_RML = os.environ.get("FIG2_PER_BAND_RML", "1").strip().lower() not in {"0", "false", "no"}
SKIP_PLOT = os.environ.get("FIG2_SKIP_PLOT", "0").strip().lower() in {"1", "true", "yes"}
EXTERNAL_BAND_START_CACHE = os.environ.get("FIG2_EXTERNAL_BAND_START_CACHE", "").strip()
EXTERNAL_BAND_START_KEY = os.environ.get(
    "FIG2_EXTERNAL_BAND_START_KEY", "promoted_singlecopy_band_fit_images"
).strip()
PAIRED_LOOP_NOISE = os.environ.get("FIG2_PAIRED_LOOP_NOISE", "1").strip().lower() not in {"0", "false", "no"}
AMPLITUDE_BRANCH_FRACTION = float(os.environ.get("FIG2_AMPLITUDE_BRANCH_FRACTION", "0.5"))
PHASE_BRANCH_FRACTION = float(os.environ.get("FIG2_PHASE_BRANCH_FRACTION", "0.5"))
NEW_DIRECT_MODEL = os.environ.get("FIG2_NEW_DIRECT_MODEL", "raw_triangle_global_gauge").strip().lower()
COHERENT_BLOCK_SIZE = int(os.environ.get("FIG2_COHERENT_BLOCK_SIZE", "10"))
PHASE_POVM_OBJECTIVE = "closure_qfi_whitened_maximin_score_dictionary_v1"
PROMOTION_MODEL = os.environ.get(
    "FIG2_PROMOTION_MODEL", "coherent_score_operator_qlan_surrogate"
).strip()
FINITE_NS_QLAN_FISHER_FACTOR = float(
    os.environ.get("FIG2_FINITE_NS_QLAN_FISHER_FACTOR", "1.0")
)
if not (0.0 < FINITE_NS_QLAN_FISHER_FACTOR <= 1.0):
    raise ValueError("FIG2_FINITE_NS_QLAN_FISHER_FACTOR must lie in (0,1]")
if "per_band" in PROMOTION_MODEL:
    FINITE_BLOCK_DESCRIPTION = (
        "Each wavelength band uses its own finite-copy factor calibrated at "
        "that band's near-transit working point; the factor is shared only "
        "across epochs within the same band."
    )
elif "global_655nm" in PROMOTION_MODEL:
    FINITE_BLOCK_DESCRIPTION = (
        "The finite-block covariance is read from the receiver cache and uses one "
        "global finite-copy factor calibrated at a representative 655-nm working point."
    )
else:
    FINITE_BLOCK_DESCRIPTION = (
        "The QLAN Fisher matrix is multiplied by the configured universal finite-block "
        "saturation factor."
    )
PROMOTED_DISPLAY_LABEL = os.environ.get(
    "FIG2_PROMOTED_DISPLAY_LABEL",
    f"Optimal POVM\n$n_s={COHERENT_BLOCK_SIZE}$ collective",
)
RECEIVER_CACHE_PATH = Path(
    os.environ.get(
        "FIG2_RECEIVER_CACHE",
        str(RUN_ROOT / "results" / "receiver_cache_mixed_closure_amp_200ms_ns10_v1.npz"),
    )
)
RECEIVER_CACHE_METADATA_PATH = RECEIVER_CACHE_PATH.with_suffix(".json")
PRECOMPUTE_ONLY = os.environ.get("FIG2_PRECOMPUTE_ONLY", "0").strip().lower() in {"1", "true", "yes"}
RAWJE_STATION_SCHEDULE_MODE = os.environ.get(
    "FIG2_RAWJE_SCHEDULE",
    os.environ.get("RAWJE_STATION_SCHEDULE_MODE", "optimized"),
).strip().lower()
DISPLAY_LOG_VMIN = float(os.environ.get("FIG2_DISPLAY_LOG_VMIN", "5e-3"))
DISPLAY_ASINH_Q = float(os.environ.get("FIG2_DISPLAY_ASINH_Q", "0.035"))
NGC4151_REDSHIFT = 0.00332
H_ALPHA_REST_NM = 656.3
H_BETA_REST_NM = 486.1

GOOD_VARIANT = next((variant for variant in morph.VARIANTS if variant.key == SOURCE_VARIANT_KEY), None)
if GOOD_VARIANT is None:
    known = ", ".join(variant.key for variant in morph.VARIANTS)
    raise ValueError(f"Unknown FIG2_SOURCE_VARIANT={SOURCE_VARIANT_KEY!r}; known variants: {known}")

GOOD_SOURCE = replace(
    ngc.NGC4151,
    sed_lambda_nm=(
        600.0,
        620.0,
        635.0,
        640.0,
        645.0,
        650.0,
        654.0,
        656.0,
        658.5,
        661.0,
        663.0,
        666.0,
        670.0,
        675.0,
        680.0,
        690.0,
        700.0,
    ),
    sed_fnu_mjy=(
        67.5,
        67.3,
        67.2,
        67.8,
        76.1,
        119.8,
        188.5,
        218.9,
        234.6,
        218.3,
        187.6,
        134.4,
        86.7,
        68.8,
        66.8,
        66.6,
        66.5,
    ),
    sed_reference=(
        "NGC 4151 nuclear 600-700 nm spectrum: the smooth continuum is kept near "
        "F_nu about 65-70 mJy, and the broad H-alpha excess is represented by a "
        "FWHM about 6000 km/s line-width bump assigned only to the BLR component; "
        "host/plume/halo light is not included."
    ),
    note="Final-candidate nuclear core+disk+BLR benchmark; H-beta lies outside the 600-700 nm band and H-alpha line flux is BLR-only.",
)


def remote_scale_tag() -> str:
    parts = []
    if abs(REMOTE_X_SCALE - 1.0) >= 1.0e-12:
        parts.append(f"x{value_tag(REMOTE_X_SCALE)}")
    if abs(REMOTE_Y_SCALE - 1.0) >= 1.0e-12:
        parts.append(f"y{value_tag(REMOTE_Y_SCALE)}")
    return "" if not parts else "_" + "_".join(parts)


REMOTE_SCALE_TAG = remote_scale_tag()
SOURCE_TAG = "coreblr" if SOURCE_VARIANT_KEY == "ngc4151_core_disk_blr_halpha" else value_tag(SOURCE_VARIANT_KEY)
TIME_TAG = f"{seconds_tag(FIG2_EXPOSURE_S)}_cad{cadence_tag(SAMPLE_CADENCE_S_RUN)}"
BAND_TAG = f"{value_tag(LAMBDA_MIN_NM_RUN)}to{value_tag(LAMBDA_MAX_NM_RUN)}nm"
BINNING_TAG = f"per{value_tag(LAMBDA_STEP_NM_RUN)}nm" if PER_BAND_RML else "jointband"
APERTURE_TAG = f"corec{value_tag(EXISTING_COUPLED_AREA_FRACTION)}_rem{value_tag(REMOTE_DIAMETER_M)}m"
if abs(POST_AVERAGE_DRIFT_STD_RUN - np.pi / 5.0) < 1.0e-12:
    DRIFT_TAG = "driftpi5"
elif abs(POST_AVERAGE_DRIFT_STD_RUN - np.pi / 10.0) < 1.0e-12:
    DRIFT_TAG = "driftpi10"
else:
    DRIFT_TAG = f"drift{value_tag(POST_AVERAGE_DRIFT_STD_RUN)}"
DEFAULT_RUN_TAG = (
    f"{OBSERVING_DAYS}night{REMOTE_SCALE_TAG}_{TIME_TAG}_{BAND_TAG}_{BINNING_TAG}_{SOURCE_TAG}_{APERTURE_TAG}"
    f"_eps{occupancy_tag(EPS_STATION_RUN)}_pair{occupancy_tag(EPS_PAIR_RUN)}"
    f"_dir{occupancy_tag(EPS_DIRECT_EXTRA_RUN)}_{DRIFT_TAG}_balanced10_directopt_paircombine"
)
RUN_TAG = os.environ.get(
    "RUN_TAG",
    DEFAULT_RUN_TAG
    if KEEP_EVERY_TIME_ROW == 1 and abs(EXPOSURE_SCALE - 1.0) < 1.0e-12
    else f"{DEFAULT_RUN_TAG}_thin{KEEP_EVERY_TIME_ROW}_x{percent_tag(EXPOSURE_SCALE)}",
)
OUTPUT_STEM = f"broad_plume_split_objective_nmode_rml_{RUN_TAG}"
GOOD_CFG = {
    "key": f"p{value_tag(RML_PRIOR_WEIGHT_RUN)}_tv{value_tag(RML_TV_WEIGHT_RUN)}_e{value_tag(RML_ENTROPY_WEIGHT_RUN)}",
    "prior": RML_PRIOR_WEIGHT_RUN,
    "tv": RML_TV_WEIGHT_RUN,
    "entropy": RML_ENTROPY_WEIGHT_RUN,
}
SPLIT_DIAGNOSTICS: dict[str, object] = {}

BALANCED_INDEPENDENT_TRIANGLES = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 5),
    (1, 3, 4),
    (1, 4, 5),
    (2, 3, 5),
    (2, 4, 5),
    (3, 4, 5),
]


def loop_label(tri: tuple[int, int, int]) -> str:
    return "-".join(f"S{i + 1}" for i in tri)


def latest_loop_basis(edges: list[tuple[int, int]]) -> np.ndarray:
    return np.stack(
        [split_sim.closure_edge_vector(edges, tri) for tri in BALANCED_INDEPENDENT_TRIANGLES],
        axis=1,
    )


def latest_closure_basis(case: aug.NetworkCase) -> tuple[list[tuple[int, int]], np.ndarray]:
    stations, _, _, _ = aug.station_table_from_case(case)
    edges = opt.base.edge_list(len(stations))
    return edges, latest_loop_basis(edges)


def detected_station_mode_occupations(
    freq_hz: float, diameters_m: np.ndarray
) -> np.ndarray:
    """Source occupation after the explicitly configured collection loss."""
    return np.asarray(
        [
            opt.base.source_mode_occupation(
                freq_hz,
                diameter_m=float(diameter_m),
                capture_efficiency=PHOTON_COLLECTION_EFFICIENCY,
            )
            for diameter_m in np.asarray(diameters_m, dtype=float)
        ],
        dtype=float,
    )


def loop_from_base_transform(base_q: np.ndarray, selected_basis: np.ndarray) -> np.ndarray:
    return selected_basis.T @ base_q


def selected_loop_noise_from_q_fisher(
    rng: np.random.Generator,
    fisher_q: np.ndarray,
    base_q: np.ndarray,
    selected_basis: np.ndarray,
    selected_to_edge: np.ndarray,
    *,
    max_std: float = 2.5,
    standard_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(0.5 * (fisher_q + fisher_q.T))
    safe = np.maximum(evals, 1.0 / max_std**2)
    cov_q = (evecs / safe) @ evecs.T
    transform = loop_from_base_transform(base_q, selected_basis)
    cov_loop = 0.5 * (transform @ cov_q @ transform.T + (transform @ cov_q @ transform.T).T)
    eval_loop, evec_loop = np.linalg.eigh(cov_loop)
    eval_loop = np.maximum(eval_loop, 0.0)
    if standard_normals is None:
        z = rng.normal(size=len(eval_loop))
    else:
        z = np.asarray(standard_normals, dtype=float)
        if z.shape != eval_loop.shape:
            raise ValueError(f"standard_normals shape {z.shape} does not match loop covariance shape {eval_loop.shape}")
    loop_noise = evec_loop @ (np.sqrt(eval_loop) * z)
    edge_noise = selected_to_edge @ loop_noise
    edge_cov = selected_to_edge @ cov_loop @ selected_to_edge.T
    loop_sigma = np.sqrt(np.maximum(np.diag(cov_loop), 0.0))
    edge_sigma = np.sqrt(np.maximum(np.diag(edge_cov), 0.0))
    return edge_noise, edge_sigma, loop_sigma


def q_fisher_from_edge_split_for_sample(
    split: np.ndarray,
    *,
    total_modes: float,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    base_q: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    edge_fisher = split_sim.edge_fisher_for_sample(
        split,
        total_modes=total_modes,
        u_station=u_station,
        eta=eta,
        station_noise=station_noise,
        nu_eff=nu_eff,
        edges=edges,
    )
    n_station = max(max(edge) for edge in edges) + 1
    return opt.base.closure_fisher_after_gauge_marginalization(
        np.diag(edge_fisher),
        base_q,
        edges,
        n_station,
    )


def triangle_weighted_q_fisher_for_sample(
    weights: dict[tuple[int, int, int], float],
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    base_q: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    fisher = np.zeros((base_q.shape[1], base_q.shape[1]), dtype=float)
    for tri, weight in weights.items():
        scalar = split_sim.core_triangle_direct_fisher_for_sample(
            tuple(tri),
            total_modes=total_modes,
            vtrue=vtrue,
            u_station=u_station,
            eta=eta,
            direct_noise=direct_noise,
            edges=edges,
            edge_to_index=edge_to_index,
        )
        d = base_q.T @ split_sim.closure_edge_vector(edges, tuple(tri))
        fisher += float(weight) * scalar * np.outer(d, d)
    return 0.5 * (fisher + fisher.T)


def load_triangle_weights(path: Path, section: str, key: str) -> dict[tuple[int, int, int], float]:
    data = json.loads(path.read_text())
    weights = data["summary"][section][key] if "summary" in data else data[section][key]
    out: dict[tuple[int, int, int], float] = {}
    for label, value in weights.items():
        tri = tuple(int(part[1:]) - 1 for part in label.split("-"))
        out[tri] = float(value)
    return out


def latest_direct_optimized_weights() -> dict[tuple[int, int, int], float]:
    return load_triangle_weights(
        SOURCE18 / "results" / "remote_star_joint_near_summary.json",
        "direct_optimized_schedule_info",
        "all_triangle_weights",
    )


_GENERALIZED_DIRECT_SCHEDULE_CACHE: tuple[np.ndarray, dict[str, object]] | None = None


def validate_station_loop_schedule(schedule: np.ndarray, *, source: str) -> None:
    expected_shape = (6, len(BALANCED_INDEPENDENT_TRIANGLES))
    if schedule.shape != expected_shape:
        raise ValueError(f"{source} has shape {schedule.shape}, expected {expected_shape}")
    for station in range(schedule.shape[0]):
        incident = [idx for idx, tri in enumerate(BALANCED_INDEPENDENT_TRIANGLES) if station in tri]
        inactive = sorted(set(range(schedule.shape[1])) - set(incident))
        if inactive and float(np.max(np.abs(schedule[station, inactive]))) > 1.0e-9:
            raise ValueError(f"Station S{station + 1} has generalized direct weight on a non-incident loop")
        if abs(float(np.sum(schedule[station, incident])) - 1.0) > 5.0e-9:
            raise ValueError(f"Station S{station + 1} generalized direct fractions do not sum to one")


def uniform_balanced10_station_schedule() -> tuple[np.ndarray, dict[str, object]]:
    schedule = np.zeros((6, len(BALANCED_INDEPENDENT_TRIANGLES)), dtype=float)
    for station in range(schedule.shape[0]):
        incident = [idx for idx, tri in enumerate(BALANCED_INDEPENDENT_TRIANGLES) if station in tri]
        for loop_idx in incident:
            schedule[station, loop_idx] = 1.0 / float(len(incident))
    validate_station_loop_schedule(schedule, source="generated uniform balanced-10 station schedule")
    loop_station_fractions = {
        loop_label(tri): {f"S{station + 1}": float(schedule[station, loop_idx]) for station in tri}
        for loop_idx, tri in enumerate(BALANCED_INDEPENDENT_TRIANGLES)
    }
    station_rows = {
        f"S{station + 1}": {
            loop_label(BALANCED_INDEPENDENT_TRIANGLES[loop_idx]): float(schedule[station, loop_idx])
            for loop_idx, tri in enumerate(BALANCED_INDEPENDENT_TRIANGLES)
            if station in tri
        }
        for station in range(schedule.shape[0])
    }
    summary = {
        "nonzero_fraction": 0.2,
        "loop_station_fractions": loop_station_fractions,
        "station_rows": station_rows,
    }
    info = {
        "source": "generated:uniform_balanced10_station_incident_loops",
        "schedule_mode": "uniform",
        "station_budget_model": "uniform_station_budget_over_incident_balanced10_loops",
        "summary": summary,
        "loop_station_fractions": loop_station_fractions,
        "station_rows": station_rows,
    }
    return schedule, info


def load_generalized_direct_station_schedule() -> tuple[np.ndarray, dict[str, object]]:
    """Load the corrected per-station loop splitter used for the new direct column."""
    global _GENERALIZED_DIRECT_SCHEDULE_CACHE
    if _GENERALIZED_DIRECT_SCHEDULE_CACHE is not None:
        return _GENERALIZED_DIRECT_SCHEDULE_CACHE
    mode = RAWJE_STATION_SCHEDULE_MODE
    if mode in {"uniform", "symmetric", "uniform_omega02", "uniform_station"}:
        _GENERALIZED_DIRECT_SCHEDULE_CACHE = uniform_balanced10_station_schedule()
        return _GENERALIZED_DIRECT_SCHEDULE_CACHE
    if mode not in {"optimized", "opt", "generalized_opt", "generalized"}:
        raise ValueError(
            "Unknown FIG2_RAWJE_SCHEDULE="
            f"{RAWJE_STATION_SCHEDULE_MODE!r}; expected optimized or uniform"
        )
    payload = json.loads(GENERALIZED_DIRECT_SCHEDULE.read_text())
    schedule = np.asarray(payload["schedule_matrix"], dtype=float)
    validate_station_loop_schedule(schedule, source="GENERALIZED_DIRECT_STATION_SCHEDULE")
    loop_set = [str(item) for item in payload.get("loop_set", [])]
    expected_loops = [loop_label(tri) for tri in BALANCED_INDEPENDENT_TRIANGLES]
    if loop_set and loop_set != expected_loops:
        raise ValueError(
            "GENERALIZED_DIRECT_STATION_SCHEDULE loop order does not match "
            f"the balanced-10 basis: {loop_set} != {expected_loops}"
        )
    summary = payload.get("summary", {})
    info = {
        "source": str(GENERALIZED_DIRECT_SCHEDULE),
        "schedule_mode": "optimized",
        "station_budget_model": payload.get("station_budget_model"),
        "summary": summary,
        "loop_station_fractions": summary.get("loop_station_fractions", {}),
        "station_rows": summary.get("station_rows", {}),
    }
    _GENERALIZED_DIRECT_SCHEDULE_CACHE = (schedule, info)
    return _GENERALIZED_DIRECT_SCHEDULE_CACHE


def corrected_raw_direct_q_fisher_for_sample(
    station_schedule: np.ndarray,
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    base_q: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Build the new raw-Je direct Fisher from asymmetric station-loop fractions."""
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    local_edges = opt.base.edge_list(3)
    edge_fisher = np.zeros((len(edges), len(edges)), dtype=float)
    direct_extra = float(split_sim.EPS_DIRECT_EXTRA)
    for loop_idx, tri in enumerate(BALANCED_INDEPENDENT_TRIANGLES):
        fractions = np.asarray([station_schedule[station, loop_idx] for station in tri], dtype=float)
        if float(np.max(fractions)) <= 1.0e-12:
            continue
        subset = np.asarray(tri, dtype=int)
        local_vis = np.asarray(
            [vtrue[edge_to_index[(tri[i], tri[j])]] for i, j in local_edges],
            dtype=complex,
        )
        eta_local = fractions * eta[subset]
        station_part = np.maximum(direct_noise[subset] - direct_extra, 0.0)
        noise_local = fractions * station_part + direct_extra
        local_fisher = total_modes * split_sim.raw_edge_phase_fisher_station_u(
            local_vis,
            eta_local,
            noise_local,
            u_station[subset],
            local_edges,
        )
        local_fisher = 0.5 * (local_fisher + local_fisher.T)
        for local_i, edge_i in enumerate(local_edges):
            global_i = edge_to_index[(tri[edge_i[0]], tri[edge_i[1]])]
            for local_j, edge_j in enumerate(local_edges):
                global_j = edge_to_index[(tri[edge_j[0]], tri[edge_j[1]])]
                edge_fisher[global_i, global_j] += float(local_fisher[local_i, local_j])
    n_station = max(max(edge) for edge in edges) + 1
    fisher = opt.base.closure_fisher_after_gauge_marginalization(
        0.5 * (edge_fisher + edge_fisher.T),
        base_q,
        edges,
        n_station,
    )
    return 0.5 * (fisher + fisher.T)


def sld_jd_from_bmat_derivatives(
    bmat: np.ndarray,
    derivatives: list[np.ndarray],
    *,
    eig_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return SLD Fisher J and mean commutator matrix D for an unnormalized B.

    The returned real matrices are
    J_ab = Re Tr(B L_a L_b) and D_ab = Im Tr(B L_a L_b), where L_a solves
    dB_a = (B L_a + L_a B)/2.  Keeping D is what distinguishes the complex
    working-point Holevo covariance from the SLD/QFI covariance.
    """
    bmat = 0.5 * (bmat + bmat.conj().T)
    evals, evecs = np.linalg.eigh(bmat)
    if np.min(evals) < eig_floor:
        evals = np.maximum(evals, eig_floor)
        bmat = (evecs * evals) @ evecs.conj().T
        evals, evecs = np.linalg.eigh(0.5 * (bmat + bmat.conj().T))

    denom = evals[:, None] + evals[None, :]
    slds = []
    for deriv in derivatives:
        deriv_p = evecs.conj().T @ deriv @ evecs
        sld_p = np.zeros_like(deriv_p, dtype=complex)
        mask = denom > eig_floor
        sld_p[mask] = 2.0 * deriv_p[mask] / denom[mask]
        slds.append(sld_p)

    n_param = len(derivatives)
    gram = np.zeros((n_param, n_param), dtype=complex)
    for a, la in enumerate(slds):
        for b, lb in enumerate(slds):
            gram[a, b] = np.sum(evals[:, None] * la * lb.T)
    fisher = 0.5 * (gram.real + gram.real.T)
    comm = 0.5 * (gram.imag - gram.imag.T)
    return fisher, comm


def edge_sld_jd_station_u(
    visibilities: np.ndarray,
    station_efficiencies: np.ndarray,
    station_noise: np.ndarray,
    station_u: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    eig_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    n_station = len(station_efficiencies)
    bmat = np.diag(station_efficiencies * station_u + station_noise).astype(complex)
    source_coherences = np.zeros(len(edges), dtype=complex)
    for edge_index, (i, j) in enumerate(edges):
        coherence = (
            math.sqrt(station_efficiencies[i] * station_efficiencies[j] * station_u[i] * station_u[j])
            * visibilities[edge_index]
        )
        source_coherences[edge_index] = coherence
        bmat[i, j] = coherence
        bmat[j, i] = np.conj(coherence)

    edge_derivatives = []
    for edge_index, (i, j) in enumerate(edges):
        deriv = np.zeros((n_station, n_station), dtype=complex)
        deriv[i, j] = 1j * source_coherences[edge_index]
        deriv[j, i] = -1j * np.conj(source_coherences[edge_index])
        edge_derivatives.append(deriv)
    return sld_jd_from_bmat_derivatives(bmat, edge_derivatives, eig_floor=eig_floor)


def edge_amplitude_sld_station_u(
    visibilities: np.ndarray,
    station_efficiencies: np.ndarray,
    station_noise: np.ndarray,
    station_u: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    eig_floor: float = 1e-12,
) -> np.ndarray:
    """SLD Fisher matrix for simultaneous edge-amplitude estimation.

    The parameters are the normalized visibility magnitudes |g_ij| at the
    current complex working point; the visibility phases are held fixed.
    """
    n_station = len(station_efficiencies)
    bmat = np.diag(station_efficiencies * station_u + station_noise).astype(complex)
    source_coherences = np.zeros(len(edges), dtype=complex)
    source_scales = np.zeros(len(edges), dtype=float)
    phase_factors = np.ones(len(edges), dtype=complex)
    for edge_index, (i, j) in enumerate(edges):
        source_scales[edge_index] = math.sqrt(
            station_efficiencies[i]
            * station_efficiencies[j]
            * station_u[i]
            * station_u[j]
        )
        visibility = visibilities[edge_index]
        amp = abs(visibility)
        if amp > 1.0e-14:
            phase_factors[edge_index] = visibility / amp
        coherence = source_scales[edge_index] * visibility
        source_coherences[edge_index] = coherence
        bmat[i, j] = coherence
        bmat[j, i] = np.conj(coherence)

    edge_derivatives = []
    for edge_index, (i, j) in enumerate(edges):
        deriv = np.zeros((n_station, n_station), dtype=complex)
        value = source_scales[edge_index] * phase_factors[edge_index]
        deriv[i, j] = value
        deriv[j, i] = np.conj(value)
        edge_derivatives.append(deriv)
    fisher, _comm = sld_jd_from_bmat_derivatives(bmat, edge_derivatives, eig_floor=eig_floor)
    return fisher


def amplitude_sigma_from_edge_fisher(edge_fisher: np.ndarray) -> np.ndarray:
    """Return marginal CRB standard deviations for edge amplitudes."""
    fisher = 0.5 * (np.asarray(edge_fisher, dtype=float) + np.asarray(edge_fisher, dtype=float).T)
    cov = np.linalg.pinv(fisher, rcond=1.0e-12)
    cov = 0.5 * (cov + cov.T)
    return np.sqrt(np.maximum(np.diag(cov), 1.0e-300))


def triangle_weighted_amplitude_sigma_for_sample(
    weights: dict[tuple[int, int, int], float],
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Amplitude CRB for a station-budgeted library of three-port modules."""
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    local_edges = opt.base.edge_list(3)
    edge_fisher = np.zeros((len(edges), len(edges)), dtype=float)
    for tri, weight in weights.items():
        if weight <= 0.0:
            continue
        tri = tuple(tri)
        subset = np.asarray(tri, dtype=int)
        local_vis = np.asarray(
            [vtrue[edge_to_index[(tri[i], tri[j])]] for i, j in local_edges],
            dtype=complex,
        )
        local_fisher = (
            float(total_modes)
            * float(weight)
            * edge_amplitude_sld_station_u(
                local_vis,
                eta[subset],
                direct_noise[subset],
                u_station[subset],
                local_edges,
            )
        )
        for local_i, edge_i in enumerate(local_edges):
            global_i = edge_to_index[(tri[edge_i[0]], tri[edge_i[1]])]
            for local_j, edge_j in enumerate(local_edges):
                global_j = edge_to_index[(tri[edge_j[0]], tri[edge_j[1]])]
                edge_fisher[global_i, global_j] += float(local_fisher[local_i, local_j])
    return amplitude_sigma_from_edge_fisher(edge_fisher)


def station_scheduled_triangle_amplitude_sigma_for_sample(
    station_schedule: np.ndarray,
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Amplitude CRB for station-dependent routed three-port modules."""
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    local_edges = opt.base.edge_list(3)
    edge_fisher = np.zeros((len(edges), len(edges)), dtype=float)
    direct_extra = float(split_sim.EPS_DIRECT_EXTRA)
    for loop_idx, tri in enumerate(BALANCED_INDEPENDENT_TRIANGLES):
        fractions = np.asarray([station_schedule[station, loop_idx] for station in tri], dtype=float)
        if float(np.max(fractions)) <= 1.0e-12:
            continue
        subset = np.asarray(tri, dtype=int)
        local_vis = np.asarray(
            [vtrue[edge_to_index[(tri[i], tri[j])]] for i, j in local_edges],
            dtype=complex,
        )
        eta_local = fractions * eta[subset]
        station_part = np.maximum(direct_noise[subset] - direct_extra, 0.0)
        noise_local = fractions * station_part + direct_extra
        local_fisher = float(total_modes) * edge_amplitude_sld_station_u(
            local_vis,
            eta_local,
            noise_local,
            u_station[subset],
            local_edges,
        )
        for local_i, edge_i in enumerate(local_edges):
            global_i = edge_to_index[(tri[edge_i[0]], tri[edge_i[1]])]
            for local_j, edge_j in enumerate(local_edges):
                global_j = edge_to_index[(tri[edge_j[0]], tri[edge_j[1]])]
                edge_fisher[global_i, global_j] += float(local_fisher[local_i, local_j])
    return amplitude_sigma_from_edge_fisher(edge_fisher)


def global_nport_amplitude_sigma_for_sample(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Amplitude CRB for a full-array coherent N-port readout."""
    edge_fisher = float(total_modes) * edge_amplitude_sld_station_u(
        vtrue,
        eta,
        direct_noise,
        u_station,
        edges,
    )
    return amplitude_sigma_from_edge_fisher(edge_fisher)


def strategy_amplitude_sigma_for_sample(
    strategy: str,
    *,
    total_modes: float,
    vtrue: np.ndarray,
    uniform_split: np.ndarray,
    direct_weights: dict[tuple[int, int, int], float],
    station_schedule: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    direct_noise: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Common full-edge amplitude branch used by all three receivers.

    The amplitude tangent space is the complete E-dimensional edge space.  A
    shared branch keeps the comparison focused on the phase receiver.
    """
    if strategy in {"edge_uniform", "optimal_singlecopy", "promoted_singlecopy"}:
        return split_sim.amplitude_sigma_for_sample(
            uniform_split,
            total_modes=total_modes,
            u_station=u_station,
            eta=eta,
            station_noise=station_noise,
            edges=edges,
        )
    raise ValueError(f"Unknown amplitude strategy {strategy!r}")


def abs_i_antisymmetric(mat: np.ndarray) -> np.ndarray:
    """Matrix absolute value of iA for a real antisymmetric A."""
    herm = 1j * (0.5 * (mat - mat.T))
    evals, evecs = np.linalg.eigh(herm)
    out = (evecs * np.abs(evals)) @ evecs.conj().T
    return 0.5 * (out.real + out.real.T)


def complex_holevo_closure_cov_station_u(
    *,
    vtrue: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    u_station: np.ndarray,
    base_q: np.ndarray,
    edges: list[tuple[int, int]],
    eig_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical Holevo covariance for closure coordinates at complex vtrue.

    This is the full N-port benchmark used for the HCRB imaging panels.  It
    first computes the edge-phase SLD Gram matrix at the actual complex
    visibility working point, forms nuisance-efficient closure SLDs by removing
    station-piston cut directions, and then adds the canonical Holevo
    compatibility term |i J^{-1} D J^{-1}| for unit weight in the closure basis.
    """
    n_station = max(max(edge) for edge in edges) + 1
    edge_j, edge_d = edge_sld_jd_station_u(
        vtrue,
        eta,
        direct_noise,
        u_station,
        edges,
        eig_floor=eig_floor,
    )
    cut = opt.base.edge_cut_basis(edges, n_station)
    transform = np.concatenate([base_q, cut], axis=1)
    jz = transform.T @ edge_j @ transform
    dz = transform.T @ edge_d @ transform
    n_q = base_q.shape[1]
    jqq = jz[:n_q, :n_q]
    jqg = jz[:n_q, n_q:]
    jgg = jz[n_q:, n_q:]
    m = jqg @ np.linalg.pinv(jgg, rcond=1e-12)
    efficient = np.concatenate([np.eye(n_q), -m], axis=1)
    j_eff = 0.5 * (efficient @ jz @ efficient.T + (efficient @ jz @ efficient.T).T)
    d_eff = efficient @ dz @ efficient.T
    d_eff = 0.5 * (d_eff - d_eff.T)
    evals, evecs = np.linalg.eigh(j_eff)
    evals = np.maximum(evals, 0.0)
    j_eff = 0.5 * ((evecs * evals) @ evecs.T + ((evecs * evals) @ evecs.T).T)
    cov_sld = np.linalg.pinv(j_eff, rcond=1e-12)
    im_z = cov_sld @ d_eff @ cov_sld
    cov_holevo = cov_sld + abs_i_antisymmetric(im_z)
    cov_holevo = 0.5 * (cov_holevo + cov_holevo.T)
    eval_h, evec_h = np.linalg.eigh(cov_holevo)
    eval_h = np.maximum(eval_h, 0.0)
    cov_holevo = 0.5 * ((evec_h * eval_h) @ evec_h.T + ((evec_h * eval_h) @ evec_h.T).T)
    return cov_holevo, cov_sld, d_eff


def global_raw_direct_q_fisher_for_sample(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    base_q: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Full-array coherent HCRB receiver at the actual complex working point."""
    cov_holevo, _cov_sld, _d_eff = complex_holevo_closure_cov_station_u(
        vtrue=vtrue,
        eta=eta,
        direct_noise=direct_noise,
        u_station=u_station,
        base_q=base_q,
        edges=edges,
    )
    cov_holevo = cov_holevo / max(float(total_modes), 1.0e-300)
    fisher_like = np.linalg.pinv(cov_holevo, rcond=1e-12)
    fisher_like = 0.5 * (fisher_like + fisher_like.T)
    evals, evecs = np.linalg.eigh(fisher_like)
    evals = np.maximum(evals, 0.0)
    return 0.5 * ((evecs * evals) @ evecs.T + ((evecs * evals) @ evecs.T).T)


def latest_upgraded_smallblock_weights() -> dict[tuple[int, int, int], float]:
    return load_triangle_weights(
        SOURCE19 / "results" / "upgraded_smallblock_summary.json",
        "upgraded_smallblock_info",
        "all_weights",
    )


LOCAL_PAIRCOMBINE_EDGES = [(0, 1), (0, 2), (1, 2)]


def load_paircombine_profile() -> dict[str, np.ndarray | object]:
    payload = json.loads(PAIRCOMBINE_CHECKPOINT.read_text())
    unpacked = payload["unpacked"]
    state = payload["state"]
    return {
        "source": str(PAIRCOMBINE_CHECKPOINT),
        "score": float(payload["score"]),
        "best_tag": str(payload["best_tag"]),
        "p": np.asarray(unpacked["p"], dtype=float),
        "q": np.asarray(unpacked["q"], dtype=float),
        "alpha": np.asarray(unpacked["alpha"], dtype=float),
        "beta": np.asarray(unpacked["beta"], dtype=float),
        "delta": np.asarray(state["delta"], dtype=float),
        "gamma": np.asarray(state["gamma"], dtype=float),
        "gains": np.asarray(payload["gains"], dtype=float),
    }


def paircombine_modules() -> list[dict[str, object]]:
    modules: list[dict[str, object]] = []
    for loop in BALANCED_INDEPENDENT_TRIANGLES:
        a, b, c = loop
        for pair, third in (((a, b), c), ((a, c), b), ((b, c), a)):
            modules.append(
                {
                    "loop": tuple(loop),
                    "pair": tuple(pair),
                    "third": int(third),
                    "stations": (int(pair[0]), int(pair[1]), int(third)),
                    "label": f"S{pair[0] + 1}+S{pair[1] + 1}|S{third + 1}",
                }
            )
    return modules


def local_visibilities_from_global(
    vtrue: np.ndarray,
    local_edges: list[tuple[int, int]],
    stations: list[int] | tuple[int, ...],
    edge_to_index: dict[tuple[int, int], int],
) -> np.ndarray:
    values = []
    for li, lj in local_edges:
        a = int(stations[li])
        b = int(stations[lj])
        edge = tuple(sorted((a, b)))
        vis = complex(vtrue[edge_to_index[edge]])
        if a > b:
            vis = np.conj(vis)
        values.append(vis)
    return np.asarray(values, dtype=complex)


def embed_local_edge_fisher_sample(
    out: np.ndarray,
    local_fisher: np.ndarray,
    subset: list[int] | tuple[int, ...],
    local_edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
) -> None:
    for local_i, (ai, bi) in enumerate(local_edges):
        edge_i = tuple(sorted((int(subset[ai]), int(subset[bi]))))
        gi = edge_to_index[edge_i]
        for local_j, (aj, bj) in enumerate(local_edges):
            edge_j = tuple(sorted((int(subset[aj]), int(subset[bj]))))
            gj = edge_to_index[edge_j]
            out[gi, gj] += local_fisher[local_i, local_j]


def marginalize_local_core_core_edges_sample(local_fisher: np.ndarray, local_edges: list[tuple[int, int]]) -> np.ndarray:
    remote_local = max(max(edge) for edge in local_edges)
    desired = [idx for idx, edge in enumerate(local_edges) if remote_local in edge]
    nuisance = [idx for idx, edge in enumerate(local_edges) if remote_local not in edge]
    out = np.zeros_like(local_fisher)
    fdd = local_fisher[np.ix_(desired, desired)]
    if nuisance:
        fdn = local_fisher[np.ix_(desired, nuisance)]
        fnn = local_fisher[np.ix_(nuisance, nuisance)]
        efficient = fdd - fdn @ np.linalg.pinv(fnn, rcond=1e-12) @ fdn.T
    else:
        efficient = fdd
    for a, ia in enumerate(desired):
        for b, ib in enumerate(desired):
            out[ia, ib] = efficient[a, b]
    return 0.5 * (out + out.T)


def split_gamma_for_six(gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    expected = 2 * len(SIX_CORE) * len(SIX_REMOTE)
    if gamma.size != expected:
        raise ValueError(f"Pair-combine profile gamma has {gamma.size} values, expected {expected}")
    n_core = len(SIX_CORE)
    n_remote = len(SIX_REMOTE)
    core_gamma = gamma[: n_core * n_remote].reshape(n_core, n_remote)
    remote_gamma = gamma[n_core * n_remote :].reshape(n_remote, n_core)
    return np.clip(core_gamma, 0.0, 1.0), np.clip(remote_gamma, 0.0, 1.0)


def remote_star_edge_fisher_for_sample_paircombine_profile(
    profile: dict[str, np.ndarray | object],
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    nu_eff: np.ndarray,
    edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
) -> np.ndarray:
    p = np.asarray(profile["p"], dtype=float)
    gamma = np.asarray(profile["gamma"], dtype=float)
    core_gamma, remote_gamma = split_gamma_for_six(gamma)
    out = np.zeros((len(edges), len(edges)), dtype=float)
    for remote_index, remote in enumerate(SIX_REMOTE):
        core_g = core_gamma[:, remote_index]
        remote_g = remote_gamma[remote_index, :]
        subset = list(SIX_CORE) + [remote]
        local_edges = opt.base.edge_list(len(subset))
        core_available = np.asarray([float(p[core, remote]) for core in SIX_CORE], dtype=float)
        remote_available = np.asarray([float(p[remote, core]) for core in SIX_CORE], dtype=float)
        star_core = core_g * core_available
        remote_star = float(np.sum(remote_g * remote_available))
        star_fractions = np.concatenate([star_core, [remote_star]])
        if remote_star > 0.0 and float(np.max(star_core)) > 0.0:
            local_vis = local_visibilities_from_global(vtrue, local_edges, subset, edge_to_index)
            eta_local = star_fractions * eta[subset]
            noise_local = star_fractions * station_noise[subset] + EPS_DIRECT_EXTRA_RUN
            local_fisher = total_modes * split_sim.raw_edge_phase_fisher_station_u(
                local_vis,
                eta_local,
                noise_local,
                u_station[subset],
                local_edges,
            )
            local_fisher = marginalize_local_core_core_edges_sample(local_fisher, local_edges)
            embed_local_edge_fisher_sample(out, local_fisher, subset, local_edges, edge_to_index)

        for core_idx, core in enumerate(SIX_CORE):
            core_residual = (1.0 - core_g[core_idx]) * p[core, remote]
            remote_residual = (1.0 - remote_g[core_idx]) * p[remote, core]
            if core_residual > 0.0 and remote_residual > 0.0:
                edge = tuple(sorted((core, remote)))
                out[edge_to_index[edge], edge_to_index[edge]] += split_sim.edge_pair_fisher_for_sample(
                    edge[0],
                    edge[1],
                    core_residual if edge[0] == core else remote_residual,
                    remote_residual if edge[1] == remote else core_residual,
                    total_modes=total_modes,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    nu_eff=nu_eff,
                    edge_to_index=edge_to_index,
                )

    for i, j in itertools.combinations(SIX_REMOTE, 2):
        out[edge_to_index[(i, j)], edge_to_index[(i, j)]] += split_sim.edge_pair_fisher_for_sample(
            i,
            j,
            float(p[i, j]),
            float(p[j, i]),
            total_modes=total_modes,
            u_station=u_station,
            eta=eta,
            station_noise=station_noise,
            nu_eff=nu_eff,
            edge_to_index=edge_to_index,
        )
    return 0.5 * (out + out.T)


def paircombine_edge_fisher_for_sample(
    profile: dict[str, np.ndarray | object],
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
) -> np.ndarray:
    q = np.asarray(profile["q"], dtype=float)
    beta = np.asarray(profile["beta"], dtype=float)
    delta = np.asarray(profile["delta"], dtype=float)
    modules = paircombine_modules()
    out = np.zeros((len(edges), len(edges)), dtype=float)
    for idx, module in enumerate(modules):
        fractions = np.asarray(q[idx], dtype=float)
        if float(np.max(fractions)) <= 1.0e-12:
            continue
        stations = tuple(module["stations"])
        c = math.cos(float(beta[idx]))
        s = math.sin(float(beta[idx]))
        phase = complex(math.cos(float(delta[idx])), math.sin(float(delta[idx])))
        transform = np.asarray([[c, phase * s, 0.0], [0.0, 0.0, 1.0]], dtype=complex)
        load = eta[list(stations)] * u_station[list(stations)] + station_noise[list(stations)]
        source = fractions * eta[list(stations)] * u_station[list(stations)]
        bmat = np.diag(fractions * load).astype(complex)
        coherences: dict[tuple[int, int], complex] = {}
        signs = []
        for edge_idx, (li, lj) in enumerate(LOCAL_PAIRCOMBINE_EDGES):
            a = int(stations[li])
            b = int(stations[lj])
            edge = tuple(sorted((a, b)))
            vis = complex(vtrue[edge_to_index[edge]])
            if a > b:
                vis = np.conj(vis)
            coh = math.sqrt(max(float(source[li] * source[lj]), 0.0)) * vis
            coherences[(li, lj)] = coh
            bmat[li, lj] = coh
            bmat[lj, li] = np.conj(coh)
            signs.append(1.0 if a < b else -1.0)
        derivs = []
        for edge_idx, (li, lj) in enumerate(LOCAL_PAIRCOMBINE_EDGES):
            coh = coherences[(li, lj)]
            deriv = np.zeros((3, 3), dtype=complex)
            deriv[li, lj] = 1j * coh * signs[edge_idx]
            deriv[lj, li] = -1j * np.conj(coh) * signs[edge_idx]
            derivs.append(transform @ deriv @ transform.conj().T)
        b2 = transform @ bmat @ transform.conj().T
        local = total_modes * opt.base.qfi_from_bmat_derivatives(b2, derivs, eig_floor=1e-12)
        embed_local_edge_fisher_sample(out, local, stations, LOCAL_PAIRCOMBINE_EDGES, edge_to_index)
    return 0.5 * (out + out.T)


def paircombine_profile_q_fisher_for_sample(
    profile: dict[str, np.ndarray | object],
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    direct_noise: np.ndarray,
    nu_eff: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    global CORE_ALPHA_FOR_SAMPLE
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    alpha_core = np.asarray(profile["alpha"], dtype=float)
    old_alpha = CORE_ALPHA_FOR_SAMPLE.copy()
    try:
        set_corealpha_for_sample(alpha_core)
        edge_fisher = core_direct_edge_fisher_for_sample_corealpha(
            total_modes=total_modes,
            vtrue=vtrue,
            u_station=u_station,
            eta=eta,
            direct_noise=direct_noise,
            edges=edges,
            edge_to_index=edge_to_index,
        )
    finally:
        CORE_ALPHA_FOR_SAMPLE = old_alpha
    edge_fisher += remote_star_edge_fisher_for_sample_paircombine_profile(
        profile,
        total_modes=total_modes,
        vtrue=vtrue,
        u_station=u_station,
        eta=eta,
        station_noise=station_noise,
        nu_eff=nu_eff,
        edges=edges,
        edge_to_index=edge_to_index,
    )
    edge_fisher += paircombine_edge_fisher_for_sample(
        profile,
        total_modes=total_modes,
        vtrue=vtrue,
        u_station=u_station,
        eta=eta,
        station_noise=station_noise,
        edges=edges,
        edge_to_index=edge_to_index,
    )
    n_station = max(max(edge) for edge in edges) + 1
    fisher = opt.base.closure_fisher_after_gauge_marginalization(edge_fisher, q_basis, edges, n_station)
    return 0.5 * (fisher + fisher.T)

STRATEGIES = [
    ("edge_uniform", "Uniform edge-first", "edge_uniform_dirty", "#0077b6"),
    ("optimal_singlecopy", "Optimized single-copy POVM", "optimal_singlecopy_dirty", "#0a9396"),
    ("promoted_singlecopy", "Promoted score lift", "promoted_singlecopy_dirty", "#9d0208"),
]
STRATEGY_FILTER_TEXT = os.environ.get("FIG2_STRATEGY_FILTER", "").strip()
if STRATEGY_FILTER_TEXT:
    _allowed_strategies = {
        item.strip()
        for item in STRATEGY_FILTER_TEXT.split(",")
        if item.strip()
    }
    STRATEGIES = [item for item in STRATEGIES if item[0] in _allowed_strategies]
    if not STRATEGIES:
        known = ", ".join(
            (
                "edge_uniform",
                "optimal_singlecopy",
                "promoted_singlecopy",
            )
        )
        raise ValueError(f"FIG2_STRATEGY_FILTER selected no strategies; known: {known}")


def observing_night_label(nights: int) -> str:
    suffix = "" if nights == 1 else "s"
    return f"{nights} observing night{suffix}"


def configure_good_runtime() -> None:
    configure_six_station_constants()
    morph.configure_runtime(scan=False)
    morph.amp_rml.SOURCE = GOOD_SOURCE
    # The active benchmark uses the full hub-distance attenuation model:
    # 0.2 dB/km with no additional effective-distance shortening.
    # False-positive occupancies are controlled by environment variables so
    # the manuscript benchmark can use the final noise model.
    aug.FIBER_LENGTH_SCALE = 1.0
    split_sim.EPS_STATION = EPS_STATION_RUN
    split_sim.EPS_PAIR = EPS_PAIR_RUN
    split_sim.EPS_DIRECT_EXTRA = EPS_DIRECT_EXTRA_RUN
    split_sim.FIBER_LENGTH_SCALE = 1.0
    split_sim.FIBER_LOSS_DB_PER_KM = 0.20
    split_sim.SOURCE = GOOD_SOURCE
    aug.LAMBDA_MIN_NM = LAMBDA_MIN_NM_RUN
    aug.LAMBDA_MAX_NM = LAMBDA_MAX_NM_RUN
    aug.LAMBDA_STEP_NM = LAMBDA_STEP_NM_RUN
    aug.POST_AVERAGE_DRIFT_STD = POST_AVERAGE_DRIFT_STD_RUN
    split_sim.aug.LAMBDA_MIN_NM = LAMBDA_MIN_NM_RUN
    split_sim.aug.LAMBDA_MAX_NM = LAMBDA_MAX_NM_RUN
    split_sim.aug.LAMBDA_STEP_NM = LAMBDA_STEP_NM_RUN
    split_sim.aug.POST_AVERAGE_DRIFT_STD = POST_AVERAGE_DRIFT_STD_RUN
    wt.LAMBDA_MIN_NM = LAMBDA_MIN_NM_RUN
    wt.LAMBDA_MAX_NM = LAMBDA_MAX_NM_RUN
    morph.BROADBAND_LAMBDA_MIN_NM = LAMBDA_MIN_NM_RUN
    morph.BROADBAND_LAMBDA_MAX_NM = LAMBDA_MAX_NM_RUN
    morph.BROADBAND_LAMBDA_STEP_NM = LAMBDA_STEP_NM_RUN
    morph.amp_rml.OBSERVING_DAYS = OBSERVING_DAYS
    morph.amp_rml.EXPOSURE_S = BASE_EXPOSURE_S * EXPOSURE_SCALE
    morph.amp_rml.N_TIME_WINDOWS = N_TIME_WINDOWS_RUN
    morph.amp_rml.EXPOSURE_GAP_S = EXPOSURE_GAP_S_RUN
    morph.amp_rml.MODE_FALSE_POSITIVE = EPS_STATION_RUN
    morph.amp_rml.PAIR_FALSE_POSITIVE = EPS_PAIR_RUN
    wt.N_PIX = 64
    val.FIT_N_PIX = RML_FIT_N_PIX_RUN
    val.ADAM_ITER = RML_ADAM_ITER_RUN
    val.ADAM_LR = RML_ADAM_LR_RUN
    val.ADAM_TARGET_AMP_CHI2 = float(os.environ.get("RML_ADAM_TARGET_AMP_CHI2", "1.15"))
    val.ADAM_TARGET_PHASE_CHI2 = float(os.environ.get("RML_ADAM_TARGET_PHASE_CHI2", "1.05"))

    closure_bm.EPS_STATION = EPS_STATION_RUN
    closure_bm.EPS_PAIR = EPS_PAIR_RUN
    closure_bm.EPS_DIRECT_EXTRA = EPS_DIRECT_EXTRA_RUN
    closure_bm.FIBER_LENGTH_SCALE = 1.0
    closure_bm.FIBER_LOSS_DB_PER_KM = 0.20


def apply_sample_stress_runtime() -> None:
    """Apply the requested shorter per-sample exposure to the simulator globals."""
    exposure_s = BASE_EXPOSURE_S * EXPOSURE_SCALE
    aug.OBSERVING_DAYS = OBSERVING_DAYS
    aug.N_TIME_WINDOWS = N_TIME_WINDOWS_RUN
    aug.EXPOSURE_S = exposure_s
    aug.EXPOSURE_GAP_S = EXPOSURE_GAP_S_RUN
    aug.POST_AVERAGE_DRIFT_STD = POST_AVERAGE_DRIFT_STD_RUN
    split_sim.aug.OBSERVING_DAYS = OBSERVING_DAYS
    split_sim.aug.N_TIME_WINDOWS = N_TIME_WINDOWS_RUN
    split_sim.aug.EXPOSURE_S = exposure_s
    split_sim.aug.EXPOSURE_GAP_S = EXPOSURE_GAP_S_RUN
    split_sim.aug.POST_AVERAGE_DRIFT_STD = POST_AVERAGE_DRIFT_STD_RUN
    morph.amp_rml.OBSERVING_DAYS = OBSERVING_DAYS
    morph.amp_rml.N_TIME_WINDOWS = N_TIME_WINDOWS_RUN
    morph.amp_rml.EXPOSURE_S = exposure_s
    morph.amp_rml.EXPOSURE_GAP_S = EXPOSURE_GAP_S_RUN
    wt.OBSERVING_DAYS = OBSERVING_DAYS
    wt.HALF_WIDTH_UAS = max(float(getattr(wt, "HALF_WIDTH_UAS", aug.HALF_WIDTH_UAS)), 110.0)
    aug.HALF_WIDTH_UAS = max(float(aug.HALF_WIDTH_UAS), 110.0)


def scale_remote_coordinates(case: aug.NetworkCase) -> aug.NetworkCase:
    """Return a case with added remote-station east/north coordinates scaled."""
    telescopes = []
    for tel in case.telescopes:
        x_km = REMOTE_X_SCALE * float(tel.x_km) if tel.is_added else float(tel.x_km)
        y_km = REMOTE_Y_SCALE * float(tel.y_km) if tel.is_added else float(tel.y_km)
        diameter_m = REMOTE_DIAMETER_M if tel.is_added else float(tel.diameter_m) * np.sqrt(EXISTING_COUPLED_AREA_FRACTION)
        telescopes.append(
            aug.Telescope(
                name=tel.name,
                x_km=x_km,
                y_km=y_km,
                diameter_m=diameter_m,
                is_added=bool(tel.is_added),
            )
        )
    return aug.NetworkCase(
        key=f"{case.key}_remote{REMOTE_SCALE_TAG}_coupled",
        title=(
            f"{case.title}; remote x x {REMOTE_X_SCALE:g}, y x {REMOTE_Y_SCALE:g}; "
            f"existing coupled area {EXISTING_COUPLED_AREA_FRACTION:g}, remote D={REMOTE_DIAMETER_M:g} m"
        ),
        latitude_deg=case.latitude_deg,
        center_latlon=case.center_latlon,
        telescopes=telescopes,
        hub_km=case.hub_km,
        optimization_score=case.optimization_score,
    )


def thin_bands_every_other_time_row(
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    keep_every: int,
) -> tuple[list[dict[str, np.ndarray]], dict]:
    """Keep one out of every ``keep_every`` time rows, preserving all baselines in a kept row."""
    if keep_every <= 1:
        rows = len(bands[0]["u"]) // len(opt.base.edge_list(len(aug.station_table_from_case(case)[0])))
        return bands, {"keep_every": 1, "rows_before": int(rows), "rows_after": int(rows)}

    stations, _, _, _ = aug.station_table_from_case(case)
    n_edges = len(opt.base.edge_list(len(stations)))
    thinned: list[dict[str, np.ndarray]] = []
    rows_before = None
    rows_after = None
    for band in bands:
        rows = len(band["u"]) // n_edges
        mask = (np.arange(rows) % keep_every) == 0
        copied: dict[str, np.ndarray] = {}
        for key, value in band.items():
            arr = np.asarray(value)
            if arr.ndim == 1 and arr.size == rows * n_edges:
                copied[key] = arr.reshape(rows, n_edges)[mask].reshape(-1).copy()
            else:
                copied[key] = arr.copy()
        thinned.append(copied)
        rows_before = rows
        rows_after = int(np.count_nonzero(mask))
    return thinned, {
        "keep_every": int(keep_every),
        "rows_before": int(rows_before or 0),
        "rows_after": int(rows_after or 0),
    }


def optimize_physical_direct_root_weights(
    bm: closure_bm.AllClosureBenchmark,
    *,
    total_weight: float = 1.0,
    seed: int = 20260610,
    floor: float = 1.0e-4,
) -> tuple[dict[tuple[int, int, int], float], dict[str, object]]:
    """Legacy root-closure direct splitter, retained only for diagnostics.

    This is not the direct model used by the near-match figures or RML run.  The
    plotted direct benchmark uses ``physical_direct_triangle_weights`` below,
    where each three-mode triangle consumes light only from its three stations
    and the constraint is station-wise: sum_{tri contains i} w_tri <= 1.
    """
    triangles = core4_remote.root_independent_triangles(bm.n)

    def project_simplex_with_floor(values: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=float), floor)
        remaining = total_weight - len(values) * floor
        if remaining <= 0.0:
            return np.full(len(values), total_weight / len(values), dtype=float)
        shifted = values - floor
        order = np.sort(shifted)[::-1]
        cumsum = np.cumsum(order)
        active = np.nonzero(order * np.arange(1, len(order) + 1) > (cumsum - remaining))[0]
        theta = (cumsum[active[-1]] - remaining) / (active[-1] + 1) if len(active) else 0.0
        return np.maximum(shifted - theta, 0.0) + floor

    def fisher_from_weights(weight_vec: np.ndarray) -> np.ndarray:
        return core4_remote.direct_fisher_from_root_weights(
            bm,
            dict(zip(triangles, (float(value) for value in weight_vec))),
        )

    def objective(weight_vec: np.ndarray) -> float:
        return core4_remote.strict_mean_coord_rms(fisher_from_weights(weight_vec))

    rng = np.random.default_rng(seed)
    best = np.full(len(triangles), total_weight / len(triangles), dtype=float)
    best_score = objective(best)
    for scale in (0.5, 1.0, 2.0, 3.0):
        for _ in range(700):
            candidate = project_simplex_with_floor(rng.lognormal(mean=0.0, sigma=scale, size=len(triangles)))
            score = objective(candidate)
            if score < best_score:
                best = candidate
                best_score = score

    for width in (0.08, 0.04, 0.02, 0.01, 0.004, 0.002, 0.001):
        improved = True
        while improved:
            improved = False
            for i in range(len(best)):
                for j in range(len(best)):
                    if i == j:
                        continue
                    movable = min(width, best[j] - floor)
                    if movable <= 0.0:
                        continue
                    candidate = best.copy()
                    candidate[i] += movable
                    candidate[j] -= movable
                    score = objective(candidate)
                    if score < best_score - 1e-15:
                        best = candidate
                        best_score = score
                        improved = True

    equal = np.full(len(triangles), total_weight / len(triangles), dtype=float)
    weights = {tri: float(value) for tri, value in zip(triangles, best)}
    return weights, {
        "model": "physical_direct_root_splitter",
        "objective": "legacy diagnostic only; not used by the physical all-triangle direct split",
        "total_weight": float(sum(weights.values())),
        "legacy_global_polling_constraint": (
            "sum_l w_l <= 1 was a conservative old root-closure polling proxy. "
            "It is not the physical all-triangle direct budget used by this script."
        ),
        "floor": float(floor),
        "n_root_closures": int(len(triangles)),
        "min_weight": float(np.min(best)),
        "max_weight": float(np.max(best)),
        "mean_coord_rms": float(best_score),
        "equal_weight_mean_coord_rms": float(objective(equal)),
        "weights": {f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}": float(value) for tri, value in weights.items()},
    }


def physical_direct_triangle_weights(
    n_station: int,
) -> tuple[dict[tuple[int, int, int], float], dict[str, object]]:
    """Uniform all-triangle direct split with exact station photon budgets.

    A scalar three-mode direct-closure receiver on triangle (a,b,c) consumes
    fraction w_abc from each of its three stations.  Since every station belongs
    to C(N-1,2) triangles, w_abc=1/C(N-1,2) saturates, but does not exceed, each
    station's received photon budget.
    """
    per_triangle = 1.0 / math.comb(n_station - 1, 2)
    weights = {tuple(tri): float(per_triangle) for tri in itertools.combinations(range(n_station), 3)}
    station_sums = {
        f"S{i + 1}": float(sum(weight for tri, weight in weights.items() if i in tri))
        for i in range(n_station)
    }
    return weights, {
        "model": "physical_all_triangle_direct_split_station_budget",
        "description": (
            "All C(N,3) scalar three-mode direct-closure receivers are available as split channels. "
            "Each triangle receives w=1/C(N-1,2), so every station's total directed fraction is exactly one."
        ),
        "n_triangle_settings": int(len(weights)),
        "per_triangle_weight": float(per_triangle),
        "station_budget_constraint": "for every station i, sum_{tri contains i} w_tri <= 1",
        "station_weight_sums": station_sums,
        "max_station_weight_sum": float(max(station_sums.values())),
        "total_triangle_weight_sum": float(sum(weights.values())),
    }


def physical_direct_triangle_weighted_fisher_for_sample(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Cycle Fisher for the physical all-triangle direct split model."""
    n_station = max(max(edge) for edge in edges) + 1
    weights = split_sim.DIRECT_ROOT_WEIGHTS or physical_direct_triangle_weights(n_station)[0]
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    fisher = np.zeros((q_basis.shape[1], q_basis.shape[1]), dtype=float)
    for tri, weight in weights.items():
        tri = tuple(tri)
        scalar = split_sim.core_triangle_direct_fisher_for_sample(
            tri,
            total_modes=total_modes,
            vtrue=vtrue,
            u_station=u_station,
            eta=eta,
            direct_noise=direct_noise,
            edges=edges,
            edge_to_index=edge_to_index,
        )
        d = q_basis.T @ split_sim.closure_edge_vector(edges, tri)
        fisher += float(weight) * scalar * np.outer(d, d)
    return 0.5 * (fisher + fisher.T)


CORE_ALPHA_FOR_SAMPLE = np.full(len(core4_remote.CORE), core4_remote.CORE_JOINT_FRACTION, dtype=float)


def set_corealpha_for_sample(alpha_core: np.ndarray) -> None:
    global CORE_ALPHA_FOR_SAMPLE
    CORE_ALPHA_FOR_SAMPLE = np.asarray(alpha_core, dtype=float).reshape(len(core4_remote.CORE)).copy()


def core_direct_edge_fisher_for_sample_corealpha(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    direct_noise: np.ndarray,
    edges: list[tuple[int, int]],
    edge_to_index: dict[tuple[int, int], int],
) -> np.ndarray:
    """Embed one sample of the close-four joint receiver with per-station alpha_i."""
    local_edges = opt.base.edge_list(len(split_sim.CORE_STATIONS))
    local_vis = np.asarray([vtrue[edge_to_index[edge]] for edge in local_edges], dtype=complex)
    subset = list(split_sim.CORE_STATIONS)
    alpha_core = CORE_ALPHA_FOR_SAMPLE
    eta_core = alpha_core * eta[subset]
    station_part = np.maximum(direct_noise[subset] - split_sim.EPS_DIRECT_EXTRA, 0.0)
    noise_core = alpha_core * station_part + split_sim.EPS_DIRECT_EXTRA
    local_edge_fisher = total_modes * split_sim.raw_edge_phase_fisher_station_u(
        local_vis,
        eta_core,
        noise_core,
        u_station[subset],
        local_edges,
    )
    out = np.zeros((len(edges), len(edges)), dtype=float)
    for local_i, edge_i in enumerate(local_edges):
        global_i = edge_to_index[edge_i]
        for local_j, edge_j in enumerate(local_edges):
            global_j = edge_to_index[edge_j]
            out[global_i, global_j] += local_edge_fisher[local_i, local_j]
    return 0.5 * (out + out.T)


def core4_remote_global_split_fisher_for_sample_corealpha(
    split: np.ndarray,
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    direct_noise: np.ndarray,
    nu_eff: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    edge_fisher = core_direct_edge_fisher_for_sample_corealpha(
        total_modes=total_modes,
        vtrue=vtrue,
        u_station=u_station,
        eta=eta,
        direct_noise=direct_noise,
        edges=edges,
        edge_to_index=edge_to_index,
    )
    for idx, (i, j) in enumerate(edges):
        if i in split_sim.CORE_STATIONS and j in split_sim.CORE_STATIONS:
            continue
        f_edge = split_sim.edge_pair_fisher_for_sample(
            i,
            j,
            split[i, j],
            split[j, i],
            total_modes=total_modes,
            u_station=u_station,
            eta=eta,
            station_noise=station_noise,
            nu_eff=nu_eff,
            edge_to_index=edge_to_index,
        )
        edge_fisher[idx, idx] += f_edge
    n_station = max(max(edge) for edge in edges) + 1
    fisher = opt.base.closure_fisher_after_gauge_marginalization(edge_fisher, q_basis, edges, n_station)
    return 0.5 * (fisher + fisher.T)


def root_cov_from_cycle_fisher(
    bm: closure_bm.AllClosureBenchmark,
    fisher_q: np.ndarray,
) -> np.ndarray:
    cov_q = np.linalg.pinv(0.5 * (fisher_q + fisher_q.T), rcond=1e-12)
    root_from_q = bm.w_basis.T @ bm.q_basis
    cov = root_from_q @ cov_q @ root_from_q.T
    return 0.5 * (cov + cov.T)


def edge_root_sigmas(bm: closure_bm.AllClosureBenchmark) -> np.ndarray:
    edge_fisher = bm.edge_fisher_values(bm.uniform_split_matrix())
    edge_cov = np.diag(1.0 / np.maximum(edge_fisher, 1e-300))
    root_cov = bm.w_basis.T @ edge_cov @ bm.w_basis
    return np.sqrt(np.maximum(np.diag(0.5 * (root_cov + root_cov.T)), 1e-300))


def raw_logits_from_projected_remote_split(
    p: np.ndarray,
    bm: closure_bm.AllClosureBenchmark,
) -> np.ndarray:
    raw = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw, -np.inf)
    for i in core4_remote.CORE:
        allowed = np.zeros(bm.n, dtype=bool)
        allowed[list(core4_remote.REMOTE)] = True
        total = core4_remote.CORE_REMOTE_FRACTION
        floor = core4_remote.SPLIT_FLOOR
        remaining = max(total - int(np.sum(allowed)) * floor, 1e-300)
        weights = np.maximum((p[i, allowed] - floor) / remaining, 1e-300)
        raw[i, allowed] = np.log(weights)
    for i in core4_remote.REMOTE:
        allowed = np.ones(bm.n, dtype=bool)
        allowed[i] = False
        total = core4_remote.REMOTE_TOTAL_FRACTION
        floor = core4_remote.SPLIT_FLOOR
        remaining = max(total - int(np.sum(allowed)) * floor, 1e-300)
        weights = np.maximum((p[i, allowed] - floor) / remaining, 1e-300)
        raw[i, allowed] = np.log(weights)
    return raw


def optimize_root_loop_gain_split(
    bm: closure_bm.AllClosureBenchmark,
    *,
    seed: int = 20260610,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optimize the compact-core+remote split on the displayed physical root-loop gains.

    The score acts on the 15 root closures directly.  It rewards the average
    log Fisher gain relative to uniform edge-first, rewards the worst SNR gain,
    and penalizes gain nonuniformity and any loop with gain below one.
    """
    rng = np.random.default_rng(seed)
    triangles = core4_remote.root_independent_triangles(bm.n)
    labels = [f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}" for tri in triangles]
    edge_sigma = edge_root_sigmas(bm)
    core_edge_fisher = core4_remote.core_direct_edge_fisher_matrix(bm)
    active: list[tuple[int, int]] = []
    for i in core4_remote.CORE:
        for j in core4_remote.REMOTE:
            active.append((i, j))
    for i in core4_remote.REMOTE:
        for j in range(bm.n):
            if i != j:
                active.append((i, j))

    def fisher_for_split(p: np.ndarray) -> np.ndarray:
        edge_fisher = core_edge_fisher + core4_remote.remote_edge_fisher_matrix_for_split(bm, p)
        return core4_remote.base.closure_fisher_after_gauge_marginalization(
            edge_fisher,
            bm.q_basis,
            bm.edges,
            bm.n,
        )

    def gains_for_split(p: np.ndarray) -> np.ndarray:
        cov = root_cov_from_cycle_fisher(bm, fisher_for_split(p))
        sigma = np.sqrt(np.maximum(np.diag(cov), 1e-300))
        return edge_sigma / sigma

    def score_gains(gain: np.ndarray) -> float:
        log_gain = np.log(np.maximum(gain, 1e-300))
        below = np.maximum(0.0, -log_gain)
        mean_log_fisher_gain = float(np.mean(2.0 * log_gain))
        min_log_snr_gain = float(np.min(log_gain))
        nonuniformity = float(np.std(log_gain))
        return (
            mean_log_fisher_gain
            + 2.0 * min_log_snr_gain
            - 1.5 * nonuniformity
            - 20.0 * float(np.mean(below * below))
        )

    raw0 = np.zeros((bm.n, bm.n), dtype=float)
    np.fill_diagonal(raw0, -np.inf)
    start_p = core4_remote.project_remote_split(raw0, bm)
    best_raw = raw0.copy()
    best_p = start_p
    best_gain = gains_for_split(best_p)
    best_score = score_gains(best_gain)

    # Include the older q-coordinate mean-RMS optimum as a starting point, then
    # re-optimize with the root-loop gain objective.
    mean_rms_p, mean_rms_info = core4_remote.optimize_split(bm, "mean_rms", seed=20260529)
    mean_rms_gain = gains_for_split(mean_rms_p)
    mean_rms_score = score_gains(mean_rms_gain)
    if mean_rms_score > best_score:
        best_score = mean_rms_score
        best_gain = mean_rms_gain
        best_p = mean_rms_p
        best_raw = raw_logits_from_projected_remote_split(mean_rms_p, bm)

    for scale in (0.4, 0.8, 1.2, 1.8, 2.6, 3.5):
        for _ in range(900):
            candidate = raw0 + rng.normal(scale=scale, size=(bm.n, bm.n))
            np.fill_diagonal(candidate, -np.inf)
            p = core4_remote.project_remote_split(candidate, bm)
            gain = gains_for_split(p)
            score = score_gains(gain)
            if score > best_score:
                best_score = score
                best_raw = candidate
                best_p = p
                best_gain = gain

    for width in (1.2, 0.6, 0.28, 0.12, 0.05, 0.02, 0.008):
        improved = True
        while improved:
            improved = False
            for i, j in active:
                for sign in (-1.0, 1.0):
                    candidate = best_raw.copy()
                    candidate[i, j] += sign * width
                    p = core4_remote.project_remote_split(candidate, bm)
                    gain = gains_for_split(p)
                    score = score_gains(gain)
                    if score > best_score + 1e-12:
                        best_score = score
                        best_raw = candidate
                        best_p = p
                        best_gain = gain
                        improved = True

    log_gain = np.log(np.maximum(best_gain, 1e-300))
    order = np.argsort(best_gain)
    return best_p, {
        "objective": "balanced_root_fisher_gain",
        "description": (
            "Optimize the 15 root closures directly: maximize average log Fisher gain "
            "relative to uniform edge-first while rewarding the minimum SNR gain and "
            "penalizing nonuniformity and any gain below one."
        ),
        "score": float(best_score),
        "min_snr_gain": float(np.min(best_gain)),
        "mean_snr_gain": float(np.mean(best_gain)),
        "median_snr_gain": float(np.median(best_gain)),
        "mean_fisher_gain": float(np.mean(best_gain * best_gain)),
        "geomean_fisher_gain": float(np.exp(np.mean(2.0 * log_gain))),
        "std_log_snr_gain": float(np.std(log_gain)),
        "n_below_unity": int(np.sum(best_gain < 1.0 - 1e-9)),
        "worst_loop_gains": {labels[idx]: float(best_gain[idx]) for idx in order[:5]},
        "best_loop_gains": {labels[idx]: float(best_gain[idx]) for idx in order[-5:][::-1]},
        "all_loop_snr_gains": {label: float(gain) for label, gain in zip(labels, best_gain)},
        "previous_mean_rms_objective": {
            "split_optimization": mean_rms_info,
            "min_snr_gain": float(np.min(mean_rms_gain)),
            "mean_fisher_gain": float(np.mean(mean_rms_gain * mean_rms_gain)),
            "n_below_unity": int(np.sum(mean_rms_gain < 1.0 - 1e-9)),
        },
    }


def make_split_matrices(case: aug.NetworkCase) -> dict[str, np.ndarray]:
    """Build split metadata for the old and corrected direct-closure models."""
    global SPLIT_DIAGNOSTICS
    n = len(case.telescopes)
    edge_uniform = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_uniform[i, j] = 1.0 / (n - 1.0)
    direct_weights = latest_direct_optimized_weights()
    station_schedule, schedule_info = load_generalized_direct_station_schedule()
    direct_station_sums = {
        f"S{i + 1}": float(sum(weight for tri, weight in direct_weights.items() if i in tri))
        for i in range(n)
    }
    generalized_station_sums = {
        f"S{i + 1}": float(np.sum(station_schedule[i]))
        for i in range(n)
    }
    split_sim.set_modular_loop_specs([], schedule_factor=1.0, close_factor=None)
    split_sim.set_direct_root_weights(direct_weights, model="old_direct_optimized_schedule_selected_balanced10")
    SPLIT_DIAGNOSTICS = {
        "closure_coordinate_basis": "balanced_10loop_independent_triangles",
        "closure_loops": [loop_label(tri) for tri in BALANCED_INDEPENDENT_TRIANGLES],
        "old_direct_strategy": "legacy loop-local 3-port direct projection with scalar loop weights",
        "old_direct_weights_source": str(SOURCE18 / "results" / "remote_star_joint_near_summary.json"),
        "old_direct_weights": {loop_label(tri): float(weight) for tri, weight in sorted(direct_weights.items())},
        "old_direct_station_weight_sums": direct_station_sums,
        "new_direct_model": NEW_DIRECT_MODEL,
        "new_direct_strategy": (
            "global coherent N-port raw closure Fisher with one station photon budget"
            if NEW_DIRECT_MODEL == "global_raw_nport"
            else "modular raw 3-port loop receivers embedded in edge space and gauge-marginalized once globally"
        ),
        "corrected_direct_schedule_mode": schedule_info.get("schedule_mode", RAWJE_STATION_SCHEDULE_MODE),
        "corrected_direct_schedule_source": schedule_info["source"],
        "corrected_direct_station_budget_model": schedule_info.get("station_budget_model"),
        "corrected_direct_station_weight_sums": generalized_station_sums,
        "corrected_direct_schedule_summary": schedule_info.get("summary", {}),
    }
    return {
        "edge_uniform": edge_uniform,
    }

    old_loader = closure_bm.rml_cases.load_maunakea_plus3_case
    old_configure = closure_bm.configure_physics
    old_bm_source = closure_bm.ngc.NGC4151
    old_core4_source = core4_remote.ngc.NGC4151

    def configure_closure_benchmark_runtime() -> None:
        old_configure()
        closure_bm.aug.OBSERVING_DAYS = OBSERVING_DAYS
        closure_bm.aug.N_TIME_WINDOWS = N_TIME_WINDOWS_RUN
        closure_bm.aug.EXPOSURE_S = BASE_EXPOSURE_S * EXPOSURE_SCALE
        closure_bm.aug.EXPOSURE_GAP_S = EXPOSURE_GAP_S_RUN
        closure_bm.aug.LAMBDA_MIN_NM = LAMBDA_MIN_NM_RUN
        closure_bm.aug.LAMBDA_MAX_NM = LAMBDA_MAX_NM_RUN
        closure_bm.aug.LAMBDA_STEP_NM = LAMBDA_STEP_NM_RUN
        closure_bm.aug.POST_AVERAGE_DRIFT_STD = POST_AVERAGE_DRIFT_STD_RUN
        closure_bm.aug.FIBER_LENGTH_SCALE = 1.0
        closure_bm.aug.FIBER_LOSS_DB_PER_KM = 0.20
        closure_bm.wt.OBSERVING_DAYS = OBSERVING_DAYS
        closure_bm.wt.SNR_BOOST = 1.0

    closure_bm.rml_cases.load_maunakea_plus3_case = lambda: case
    closure_bm.configure_physics = configure_closure_benchmark_runtime
    closure_bm.ngc.NGC4151 = GOOD_SOURCE
    core4_remote.ngc.NGC4151 = GOOD_SOURCE
    try:
        with morph.patched_variant(GOOD_VARIANT), ngc.patched_source(GOOD_SOURCE):
            bm = closure_bm.AllClosureBenchmark()
            default_payload = CORRECTED_ROOT / "near_match_direct_split_payload.json"
            external_payload = os.environ.get("NEAR_SPLIT_PAYLOAD", "").strip()
            if not external_payload and default_payload.exists():
                external_payload = str(default_payload)
            if external_payload:
                payload_path = Path(external_payload).expanduser().resolve()
                payload = json.loads(payload_path.read_text())
                profile_split = np.asarray(payload["split_matrix"], dtype=float)
                alpha_core = corealpha_variants.alpha_core_array(payload.get("alpha_core", payload["alpha"]), bm)
                split_info = {
                    "source": str(payload_path),
                    "payload_variant": payload.get("variant", "external"),
                    **dict(payload.get("summary", {})),
                }
                if profile_split.shape != (bm.n, bm.n):
                    raise ValueError(
                        f"NEAR_SPLIT_PAYLOAD split matrix has shape {profile_split.shape}, expected {(bm.n, bm.n)}"
                    )
            else:
                profile_split, split_info = optimize_root_loop_gain_split(bm, seed=20260610)
                alpha_core = np.full(len(core4_remote.CORE), float(core4_remote.CORE_JOINT_FRACTION), dtype=float)
            set_corealpha_for_sample(alpha_core)
            split_sim.core4_remote_global_split_fisher_for_sample = core4_remote_global_split_fisher_for_sample_corealpha
            near_edge_fisher = corealpha_variants.core_direct_edge_fisher_matrix_alpha(
                bm,
                alpha_core,
            ) + core4_remote.remote_edge_fisher_matrix_for_split(bm, profile_split)
            near_fisher = core4_remote.base.closure_fisher_after_gauge_marginalization(
                near_edge_fisher,
                bm.q_basis,
                bm.edges,
                bm.n,
            )
            direct_weights, direct_info = physical_direct_triangle_weights(bm.n)
            near_info = {
                "objective": split_info.get("objective", "physical_global_row_normalized_external_root_gain"),
                "description": (
                    "Compact-core stations use the shared phase-frame direct+nuisance receiver; "
                    "remote-involved baselines use one global station-side row-normalized "
                    "edge-first split matrix. No closure-loop schedule weights are used. "
                    "The split is optimized to match the physical all-triangle direct split loop by loop, "
                    "with near/edge<1 candidates rejected. "
                    "The RML simulation uses the same independently optimized core-station alpha_i values as the Fisher objective."
                ),
                "core_joint_alpha": float(np.mean(alpha_core)),
                "core_joint_alpha_core": [float(x) for x in alpha_core],
                "core_joint_alpha_by_station": {
                    bm.names[station]: float(value)
                    for station, value in zip(core4_remote.CORE, alpha_core)
                },
                "metrics": closure_bm.stable_metrics(near_fisher),
                "split_optimization": split_info,
                "station_row_sums": {
                    bm.names[i]: float(np.sum(profile_split[i])) for i in range(bm.n)
                },
                "station_total_budgets": {
                    bm.names[i]: float((alpha_core[list(core4_remote.CORE).index(i)] if i in core4_remote.CORE else 0.0) + np.sum(profile_split[i]))
                    for i in range(bm.n)
                },
            }
            split_sim.set_modular_loop_specs([], schedule_factor=1.0, close_factor=None)
            split_sim.set_direct_root_weights(direct_weights, model=str(direct_info["model"]))
            split_sim.direct_root_weighted_fisher_for_sample = physical_direct_triangle_weighted_fisher_for_sample
            SPLIT_DIAGNOSTICS = {
                "near_strategy": "physical compact-core joint phase-frame plus global row-normalized remote edge-first split",
                "near_core_joint_alpha": float(np.mean(alpha_core)),
                "near_core_joint_alpha_core": [float(x) for x in alpha_core],
                "near_split_payload": external_payload or None,
                "near_split_objective": near_info,
                "direct_strategy": "physical all-triangle three-mode direct closure split",
                "direct_weight_info": direct_info,
                "direct_raw_qfi_metrics_upper_bound": closure_bm.stable_metrics(bm.direct_raw),
                "direct_raw_qfi_caveat": (
                    "The full N-mode raw multiparameter closure QFI is retained only as an upper-bound diagnostic. "
                    "The plotted direct column uses triangle direct receivers with station-wise split sums <= 1."
                ),
            }
            matrices = {
                "edge_uniform": bm.uniform_split_matrix(),
                "core4_remote_optimized": profile_split,
            }
    finally:
        closure_bm.rml_cases.load_maunakea_plus3_case = old_loader
        closure_bm.configure_physics = old_configure
    closure_bm.ngc.NGC4151 = old_bm_source
    core4_remote.ngc.NGC4151 = old_core4_source
    return matrices


def covariance_square_root(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def mixed_receiver_covariances_for_sample(
    *,
    total_modes: float,
    vtrue: np.ndarray,
    u_station: np.ndarray,
    eta: np.ndarray,
    station_noise: np.ndarray,
    base_q: np.ndarray,
    selected_basis: np.ndarray,
    edges: list[tuple[int, int]],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return aggregate covariance on full amplitudes plus selected closures.

    The POVM is evaluated on the conditional one-photon state.  Multiplication
    by ``total_modes * one_photon_occupation`` converts the conditional CFI (or
    QLAN covariance) to the weak-light temporal-mode experiment.
    """
    n_station = len(eta)
    n_edge = len(edges)
    n_closure = base_q.shape[1]
    piston_basis = receiver_design.incidence_basis(edges, n_station)
    parameter_map = receiver_design.mixed_parameter_map(base_q, piston_basis)
    rho, raw_derivatives, one_photon_occupation = receiver_design.conditional_state_and_raw_derivatives(
        vtrue,
        eta,
        u_station,
        station_noise,
        edges,
    )
    optimal_effects, optimal_meta = receiver_design.locally_optimized_effects(
        rho,
        raw_derivatives,
        parameter_map,
        vtrue,
        edges,
        amplitude_fraction=AMPLITUDE_BRANCH_FRACTION,
        phase_fraction=PHASE_BRANCH_FRACTION,
    )
    edge_effects = receiver_design.edge_first_effects(
        vtrue,
        edges,
        n_station,
        amplitude_fraction=AMPLITUDE_BRANCH_FRACTION,
        phase_fraction=PHASE_BRANCH_FRACTION,
    )
    _edge_outcome, edge_fisher, edge_lift = receiver_design.evaluate_effects_in_mixed_space(
        rho,
        raw_derivatives,
        parameter_map,
        edge_effects,
        n_edge=n_edge,
        n_closure=n_closure,
    )
    _single_outcome, single_fisher, single_lift = receiver_design.evaluate_effects_in_mixed_space(
        rho,
        raw_derivatives,
        parameter_map,
        optimal_effects,
        n_edge=n_edge,
        n_closure=n_closure,
    )
    information_scale = float(total_modes) * float(one_photon_occupation)
    if information_scale <= 0.0:
        raise ValueError("non-positive weak-light information scale")
    covariance_base = {
        "edge_uniform": receiver_design.psd_pinv(edge_fisher) / information_scale,
        "optimal_singlecopy": receiver_design.psd_pinv(single_fisher) / information_scale,
        "promoted_singlecopy": single_lift.covariance_v
        / (information_scale * FINITE_NS_QLAN_FISHER_FACTOR),
    }
    # The RML likelihood uses the balanced physical loop list, whereas the
    # receiver algebra uses an orthonormal closure coordinate basis.
    loop_transform = selected_basis.T @ base_q
    science_transform = np.zeros((n_edge + n_closure, n_edge + n_closure), dtype=float)
    science_transform[:n_edge, :n_edge] = np.eye(n_edge)
    science_transform[n_edge:, n_edge:] = loop_transform
    covariances = {
        key: 0.5 * (
            science_transform @ value @ science_transform.T
            + (science_transform @ value @ science_transform.T).T
        )
        for key, value in covariance_base.items()
    }

    derivatives_new = np.einsum("ip,ijk->pjk", parameter_map, raw_derivatives, optimize=True)
    _slds, quantum_fisher = receiver_design.sld_operators(rho, derivatives_new)
    closure_indices = np.arange(n_edge, n_edge + n_closure)
    amplitude_indices = np.arange(n_edge)
    piston_indices = np.arange(n_edge + n_closure, parameter_map.shape[1])
    quantum_closure = receiver_design._schur_target(
        quantum_fisher,
        closure_indices,
        np.concatenate([amplitude_indices, piston_indices]),
    )
    quantum_invsqrt = receiver_design.psd_power(quantum_closure, -0.5)

    def efficiency_summary(fisher: np.ndarray) -> dict[str, float]:
        closure_fisher = receiver_design.closure_effective_fisher(fisher, n_edge, n_closure)
        normalized = quantum_invsqrt @ closure_fisher @ quantum_invsqrt
        values = np.linalg.eigvalsh(0.5 * (normalized + normalized.T))
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    diagnostics: dict[str, object] = {
        "povm_hash": optimal_meta["povm_hash"],
        "povm_objective": optimal_meta["objective"],
        "optimizer": optimal_meta,
        "one_photon_occupation": float(one_photon_occupation),
        "information_scale": information_scale,
        "mixed_dimension": int(n_edge + n_closure),
        "amplitude_dimension": int(n_edge),
        "closure_dimension": int(n_closure),
        "piston_nuisance_dimension": int(n_station - 1),
        "rank_edge": int(np.linalg.matrix_rank(edge_fisher, tol=1.0e-9)),
        "rank_single": int(np.linalg.matrix_rank(single_fisher, tol=1.0e-9)),
        "edge_qfi_efficiency": efficiency_summary(edge_fisher),
        "single_qfi_efficiency": efficiency_summary(single_fisher),
        "promoted_qfi_efficiency": efficiency_summary(
            FINITE_NS_QLAN_FISHER_FACTOR * single_lift.promoted_fisher
        ),
        "qlan_qfi_efficiency": efficiency_summary(single_lift.promoted_fisher),
        "single_lift_kappa_operator": float(single_lift.kappa_operator),
        "edge_lift_kappa_operator": float(edge_lift.kappa_operator),
        "coherent_block_size": int(COHERENT_BLOCK_SIZE),
        "promotion_model": PROMOTION_MODEL,
        "finite_ns_qlan_fisher_factor": FINITE_NS_QLAN_FISHER_FACTOR,
        "finite_block_statement": FINITE_BLOCK_DESCRIPTION,
    }
    return covariances, diagnostics


def simulate_latest_loop_bands(case: aug.NetworkCase, splits: dict[str, np.ndarray]):
    rng = np.random.default_rng(FIG2_RNG_SEED)
    drift_rng = np.random.default_rng(FIG2_RNG_SEED + 37)
    stations, diameters, _, _ = aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n = len(stations)
    edges = opt.base.edge_list(n)
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    base_q = opt.base.orthonormal_cycle_basis(opt.base.root_cycle_basis(edges, n))
    selected_basis = latest_loop_basis(edges)
    selected_to_edge = selected_basis @ np.linalg.pinv(selected_basis.T @ selected_basis, rcond=1e-12)
    direct_weights = latest_direct_optimized_weights()
    station_schedule, schedule_info = load_generalized_direct_station_schedule()

    truth, axis_uas = opt.base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * opt.base.UAS_TO_RAD
    vgrid, uv_axis = opt.base.visibility_grid(truth, fov_rad)
    wavelength_source = getattr(opt.base, "make_source_at_wavelength_nm", None)

    effective_hub_dist = aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n, EPS_STATION_RUN, dtype=float)
    direct_noise = np.full(n, EPS_STATION_RUN + EPS_DIRECT_EXTRA_RUN, dtype=float)
    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    station_piston_std = aug.POST_AVERAGE_DRIFT_STD / np.sqrt(2.0)

    endpoint_coverage = {}
    for wavelength_nm in (aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM):
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            wavelength_nm * 1e-9,
            latitude_deg=case.latitude_deg,
            declination_deg=GOOD_SOURCE.dec_deg,
        )
        endpoint_coverage[f"{wavelength_nm:g}"] = {
            "u": (uu_rows.reshape(-1) / 1e9).tolist(),
            "v": (vv_rows.reshape(-1) / 1e9).tolist(),
        }

    strategy_keys = ["all", "edge_uniform", "old_direct_optimized", "nmode_joint_scheduled"]
    uniform_split = splits["edge_uniform"]
    lam_edges_nm = wavelength_bin_edges_nm()
    bands: list[dict[str, np.ndarray]] = []
    all_amp_sigma = {key: [] for key in strategy_keys}
    all_amp_true = []
    all_amp_data = {key: [] for key in strategy_keys}

    for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
        center_nm = float(math.sqrt(lo_nm * hi_nm))
        lam = center_nm * 1e-9
        freq = opt.base.C_LIGHT / lam
        freq_lo = opt.base.C_LIGHT / (hi_nm * 1e-9)
        freq_hi = opt.base.C_LIGHT / (lo_nm * 1e-9)
        total_modes = aug.EXPOSURE_S * aug.OBSERVING_DAYS * (freq_hi - freq_lo)
        amp_modes = total_modes * AMPLITUDE_BRANCH_FRACTION
        phase_modes = total_modes * PHASE_BRANCH_FRACTION
        u_station = detected_station_mode_occupations(freq, diameters)
        if callable(wavelength_source):
            band_truth, _ = wavelength_source(aug.N_PIX, aug.HALF_WIDTH_UAS, center_nm)
            band_vgrid, band_uv_axis = opt.base.visibility_grid(band_truth, fov_rad)
        else:
            band_vgrid, band_uv_axis = vgrid, uv_axis
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            lam,
            latitude_deg=case.latitude_deg,
            declination_deg=GOOD_SOURCE.dec_deg,
        )
        band = {"u": [], "v": []}
        vis = {key: [] for key in strategy_keys}
        sig = {key: [] for key in strategy_keys}
        sig_q = {key: [] for key in strategy_keys}
        amp_by_strategy = {key: [] for key in strategy_keys}
        amp_true_all = []
        amp_sigma_by_strategy = {key: [] for key in strategy_keys}
        for uu, vv in zip(uu_rows, vv_rows):
            vtrue = opt.base.interp_vis(band_vgrid, band_uv_axis, uu, vv)
            amp = np.abs(vtrue)
            phase = np.angle(vtrue)
            loop_true = phase @ selected_basis
            nu_eff = np.clip(amp, 1e-4, 0.98)
            amp_standard_normal = rng.normal(size=amp.shape)
            sigma_amp_by_key = {
                key: strategy_amplitude_sigma_for_sample(
                    key,
                    total_modes=amp_modes,
                    vtrue=vtrue,
                    uniform_split=uniform_split,
                    direct_weights=direct_weights,
                    station_schedule=station_schedule,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    direct_noise=direct_noise,
                    edges=edges,
                )
                for key in strategy_keys
            }
            measured_amp_by_key = {
                key: (
                    amp.copy()
                    if split_sim.USE_TRUE_AMPLITUDE
                    else np.maximum(amp + amp_standard_normal * sigma_amp_by_key[key], 0.0)
                )
                for key in strategy_keys
            }
            phase_amp_by_key = {
                key: np.maximum(measured_amp_by_key[key], 1e-8)
                for key in strategy_keys
            }

            f_edge = split_sim.edge_fisher_for_sample(
                uniform_split,
                total_modes=phase_modes,
                u_station=u_station,
                eta=eta,
                station_noise=station_noise,
                nu_eff=nu_eff,
                edges=edges,
            )
            sigma_raw = np.minimum(1.0 / np.sqrt(np.maximum(f_edge, 1e-300)), aug.SIGMA_CLIP_RAD)
            raw_noise = rng.normal(scale=sigma_raw)
            station_pistons = drift_rng.normal(scale=station_piston_std, size=n)
            station_pistons -= np.mean(station_pistons)
            residual_drift = np.array([station_pistons[i] - station_pistons[j] for i, j in edges])
            noise_all = raw_noise + residual_drift
            sigma_all = np.sqrt(sigma_raw**2 + aug.POST_AVERAGE_DRIFT_STD**2)
            vis["all"].append(phase_amp_by_key["all"] * np.exp(1j * (phase + noise_all)))
            sig["all"].append(sigma_all)

            paired_loop_z = rng.normal(size=selected_basis.shape[1]) if PAIRED_LOOP_NOISE else None
            if NEW_DIRECT_MODEL == "global_raw_nport":
                new_direct_fisher_q = global_raw_direct_q_fisher_for_sample(
                    total_modes=phase_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    direct_noise=direct_noise,
                    base_q=base_q,
                    edges=edges,
                )
            elif NEW_DIRECT_MODEL in {"raw_triangle_global_gauge", "raw_triangle_station_schedule"}:
                new_direct_fisher_q = corrected_raw_direct_q_fisher_for_sample(
                    station_schedule,
                    total_modes=phase_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    direct_noise=direct_noise,
                    base_q=base_q,
                    edges=edges,
                )
            else:
                raise ValueError(
                    "Unknown FIG2_NEW_DIRECT_MODEL="
                    f"{NEW_DIRECT_MODEL!r}; expected global_raw_nport or raw_triangle_global_gauge"
                )
            for key, fisher_q in (
                (
                    "edge_uniform",
                    q_fisher_from_edge_split_for_sample(
                        uniform_split,
                        total_modes=phase_modes,
                        u_station=u_station,
                        eta=eta,
                        station_noise=station_noise,
                        nu_eff=nu_eff,
                        base_q=base_q,
                        edges=edges,
                    ),
                ),
                (
                    "old_direct_optimized",
                    triangle_weighted_q_fisher_for_sample(
                        direct_weights,
                        total_modes=phase_modes,
                        vtrue=vtrue,
                        u_station=u_station,
                        eta=eta,
                        direct_noise=direct_noise,
                        base_q=base_q,
                        edges=edges,
                    ),
                ),
                ("nmode_joint_scheduled", new_direct_fisher_q),
            ):
                edge_noise, sigma_edge, sigma_loop = selected_loop_noise_from_q_fisher(
                    rng,
                    fisher_q,
                    base_q,
                    selected_basis,
                    selected_to_edge,
                    standard_normals=paired_loop_z,
                )
                data_edge_phase = selected_to_edge @ (loop_true + edge_noise @ selected_basis)
                vis[key].append(phase_amp_by_key[key] * np.exp(1j * data_edge_phase))
                sig[key].append(sigma_edge)
                sig_q[key].append(sigma_loop)

            band["u"].append(uu)
            band["v"].append(vv)
            amp_true_all.append(amp)
            for key in strategy_keys:
                amp_by_strategy[key].append(measured_amp_by_key[key])
                amp_sigma_by_strategy[key].append(sigma_amp_by_key[key])

        band["u"] = np.concatenate(band["u"])
        band["v"] = np.concatenate(band["v"])
        band["amp_true"] = np.concatenate(amp_true_all)
        for key in strategy_keys:
            band[f"amp_{key}"] = np.concatenate(amp_by_strategy[key])
            band[f"amp_sigma_{key}"] = np.concatenate(amp_sigma_by_strategy[key])
        band["amp"] = band["amp_edge_uniform"].copy()
        band["amp_sigma"] = band["amp_sigma_edge_uniform"].copy()
        for key in strategy_keys:
            band[f"vis_{key}"] = np.concatenate(vis[key])
            band[f"sigma_{key}"] = np.concatenate(sig[key])
            if sig_q[key]:
                band[f"sigmaq_{key}"] = np.concatenate(sig_q[key])
        bands.append(band)
        for key in strategy_keys:
            all_amp_sigma[key].append(band[f"amp_sigma_{key}"])
            all_amp_data[key].append(band[f"amp_{key}"])
        all_amp_true.append(band["amp_true"])

    amp_true = np.concatenate(all_amp_true)
    amp_sigma_by_strategy_flat = {
        key: np.concatenate(values)
        for key, values in all_amp_sigma.items()
        if values
    }
    amp_data_by_strategy_flat = {
        key: np.concatenate(values)
        for key, values in all_amp_data.items()
        if values
    }
    stats = {
        "case": case.key,
        "n_station": n,
        "n_closure": int(selected_basis.shape[1]),
        "closure_coordinate_basis": "balanced_10loop_independent_triangles",
        "closure_loops": [loop_label(tri) for tri in BALANCED_INDEPENDENT_TRIANGLES],
        "old_direct_schedule_model": "legacy_direct_optimized_schedule_selected_balanced10",
        "old_direct_weights": {loop_label(tri): float(weight) for tri, weight in sorted(direct_weights.items())},
        "new_direct_model": NEW_DIRECT_MODEL,
        "hcrb_working_point_model": (
            HCRB_WORKING_POINT_MODEL if NEW_DIRECT_MODEL == "global_raw_nport" else None
        ),
        "new_direct_strategy": (
            "global coherent N-port complex-working-point canonical Holevo covariance"
            if NEW_DIRECT_MODEL == "global_raw_nport"
            else "modular raw 3-port station-budget blocks with final global gauge"
        ),
        "corrected_direct_schedule_model": "raw_Je_asymmetric_station_budget_balanced10",
        "corrected_direct_schedule_mode": schedule_info.get("schedule_mode", RAWJE_STATION_SCHEDULE_MODE),
        "corrected_direct_schedule_source": schedule_info["source"],
        "corrected_direct_station_budget_model": schedule_info.get("station_budget_model"),
        "corrected_direct_schedule_summary": schedule_info.get("summary", {}),
        "eps_station": EPS_STATION_RUN,
        "eps_pair": EPS_PAIR_RUN,
        "eps_direct_extra": EPS_DIRECT_EXTRA_RUN,
        "post_average_drift_std_rad": float(getattr(aug, "POST_AVERAGE_DRIFT_STD", 0.0)),
        "fiber_length_scale": aug.FIBER_LENGTH_SCALE,
        "fiber_loss_db_per_km": aug.FIBER_LOSS_DB_PER_KM,
        "amplitude_sigma_model": (
            "strategy-dependent amplitude branch; total sample photons are split into "
            f"{AMPLITUDE_BRANCH_FRACTION:g} amplitude and {PHASE_BRANCH_FRACTION:g} phase/closure branches"
        ),
        "amplitude_branch_fraction": float(AMPLITUDE_BRANCH_FRACTION),
        "phase_branch_fraction": float(PHASE_BRANCH_FRACTION),
        "use_true_amplitude": split_sim.USE_TRUE_AMPLITUDE,
        "phase_noise_model": "selected balanced-loop covariance; pseudo-edge representatives satisfy data_phase @ loop_basis = measured_loop",
        "paired_loop_noise": PAIRED_LOOP_NOISE,
        "endpoint_coverage_g_lambda": endpoint_coverage,
        "source_spectral_model": getattr(opt.base, "SOURCE_COMPONENT_SPECTRAL_MODEL", "achromatic source morphology"),
        "amplitude_snr_median_by_strategy": {
            key: float(np.median(amp_true / np.maximum(value, 1e-300)))
            for key, value in amp_sigma_by_strategy_flat.items()
        },
        "amplitude_noise_rms_by_strategy": {
            key: float(np.sqrt(np.mean((amp_data_by_strategy_flat[key] - amp_true) ** 2)))
            for key in amp_data_by_strategy_flat
        },
    }
    return bands, stats, truth, axis_uas


def simulate_mixed_povm_bands(case: aug.NetworkCase):
    """Simulate all three paired receivers in closure-phase + full-amplitude space."""
    rng = np.random.default_rng(FIG2_RNG_SEED)
    stations, diameters, _, _ = aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n_station = len(stations)
    edges = opt.base.edge_list(n_station)
    n_edge = len(edges)
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    base_q = opt.base.orthonormal_cycle_basis(opt.base.root_cycle_basis(edges, n_station))
    selected_basis = latest_loop_basis(edges)
    n_closure = selected_basis.shape[1]
    selected_to_edge = selected_basis @ np.linalg.pinv(selected_basis.T @ selected_basis, rcond=1.0e-12)
    strategy_keys = ["edge_uniform", "optimal_singlecopy", "promoted_singlecopy"]

    truth, axis_uas = opt.base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * opt.base.UAS_TO_RAD
    vgrid, uv_axis = opt.base.visibility_grid(truth, fov_rad)
    wavelength_source = getattr(opt.base, "make_source_at_wavelength_nm", None)
    effective_hub_dist = aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n_station, EPS_STATION_RUN, dtype=float)
    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)

    endpoint_coverage: dict[str, object] = {}
    for wavelength_nm in (aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM):
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            wavelength_nm * 1.0e-9,
            latitude_deg=case.latitude_deg,
            declination_deg=GOOD_SOURCE.dec_deg,
        )
        endpoint_coverage[f"{wavelength_nm:g}"] = {
            "u": (uu_rows.reshape(-1) / 1.0e9).tolist(),
            "v": (vv_rows.reshape(-1) / 1.0e9).tolist(),
        }

    cache_data: dict[str, np.ndarray] | None = None
    cache_metadata: dict[str, object] = {}
    if RECEIVER_CACHE_PATH.exists():
        with np.load(RECEIVER_CACHE_PATH, allow_pickle=False) as loaded:
            cache_data = {key: np.asarray(loaded[key]) for key in loaded.files}
        if RECEIVER_CACHE_METADATA_PATH.exists():
            cache_metadata = json.loads(RECEIVER_CACHE_METADATA_PATH.read_text())
        expected = {
            "n_station": n_station,
            "n_edge": n_edge,
            "n_closure": n_closure,
            "amplitude_branch_fraction": AMPLITUDE_BRANCH_FRACTION,
            "phase_branch_fraction": PHASE_BRANCH_FRACTION,
            "coherent_block_size": COHERENT_BLOCK_SIZE,
            "promotion_model": PROMOTION_MODEL,
            "exposure_s": float(aug.EXPOSURE_S),
            "n_time_windows": int(len(hour_angles)),
            "lambda_min_nm": float(aug.LAMBDA_MIN_NM),
            "lambda_max_nm": float(aug.LAMBDA_MAX_NM),
            "lambda_step_nm": float(aug.LAMBDA_STEP_NM),
            "eps_station": float(EPS_STATION_RUN),
            "existing_coupled_area_fraction": float(EXISTING_COUPLED_AREA_FRACTION),
            "remote_diameter_m": float(REMOTE_DIAMETER_M),
            "photon_collection_efficiency": float(PHOTON_COLLECTION_EFFICIENCY),
        }
        if (
            FINITE_NS_QLAN_FISHER_FACTOR != 1.0
            or PROMOTION_MODEL != "coherent_score_operator_qlan_surrogate"
        ):
            expected["finite_ns_qlan_fisher_factor"] = (
                FINITE_NS_QLAN_FISHER_FACTOR
            )
        for key, value in expected.items():
            if cache_metadata.get(key) != value:
                raise ValueError(
                    f"Receiver cache metadata mismatch for {key}: "
                    f"{cache_metadata.get(key)!r} != {value!r}"
                )

    built_covariances = {key: [] for key in strategy_keys}
    built_diagnostics: list[dict[str, object]] = []
    lam_edges_nm = wavelength_bin_edges_nm()
    bands: list[dict[str, np.ndarray]] = []
    all_amp_true: list[np.ndarray] = []
    all_amp_sigma = {key: [] for key in strategy_keys}
    all_amp_data = {key: [] for key in strategy_keys}

    for band_index, (lo_nm, hi_nm) in enumerate(zip(lam_edges_nm[:-1], lam_edges_nm[1:])):
        center_nm = float(math.sqrt(lo_nm * hi_nm))
        lam = center_nm * 1.0e-9
        freq = opt.base.C_LIGHT / lam
        freq_lo = opt.base.C_LIGHT / (hi_nm * 1.0e-9)
        freq_hi = opt.base.C_LIGHT / (lo_nm * 1.0e-9)
        total_modes = aug.EXPOSURE_S * aug.OBSERVING_DAYS * (freq_hi - freq_lo)
        u_station = detected_station_mode_occupations(freq, diameters)
        if callable(wavelength_source):
            band_truth, _ = wavelength_source(aug.N_PIX, aug.HALF_WIDTH_UAS, center_nm)
            band_vgrid, band_uv_axis = opt.base.visibility_grid(band_truth, fov_rad)
        else:
            band_vgrid, band_uv_axis = vgrid, uv_axis
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
            lam,
            latitude_deg=case.latitude_deg,
            declination_deg=GOOD_SOURCE.dec_deg,
        )
        band: dict[str, object] = {"u": [], "v": []}
        vis = {key: [] for key in strategy_keys}
        sigma_edge = {key: [] for key in strategy_keys}
        sigma_q = {key: [] for key in strategy_keys}
        sigma_q_cov = {key: [] for key in strategy_keys}
        mixed_cov = {key: [] for key in strategy_keys}
        closure_data = {key: [] for key in strategy_keys}
        closure_reference_rows: list[np.ndarray] = []
        amp_data = {key: [] for key in strategy_keys}
        amp_sigma = {key: [] for key in strategy_keys}
        amp_true_rows: list[np.ndarray] = []
        band_cache_rows = {key: [] for key in strategy_keys}

        for time_index, (uu, vv) in enumerate(zip(uu_rows, vv_rows)):
            vtrue = opt.base.interp_vis(band_vgrid, band_uv_axis, uu, vv)
            amplitude_true = np.abs(vtrue)
            loop_true = np.angle(vtrue) @ selected_basis
            if cache_data is None:
                covariances, diagnostics = mixed_receiver_covariances_for_sample(
                    total_modes=total_modes,
                    vtrue=vtrue,
                    u_station=u_station,
                    eta=eta,
                    station_noise=station_noise,
                    base_q=base_q,
                    selected_basis=selected_basis,
                    edges=edges,
                )
                compact_diagnostics = {
                    "band_index": int(band_index),
                    "time_index": int(time_index),
                    "lambda_center_nm": center_nm,
                    "povm_hash": diagnostics["povm_hash"],
                    "rank_edge": diagnostics["rank_edge"],
                    "rank_single": diagnostics["rank_single"],
                    "edge_qfi_efficiency": diagnostics["edge_qfi_efficiency"],
                    "single_qfi_efficiency": diagnostics["single_qfi_efficiency"],
                    "promoted_qfi_efficiency": diagnostics["promoted_qfi_efficiency"],
                    "single_lift_kappa_operator": diagnostics["single_lift_kappa_operator"],
                    "maximin_qfi_efficiency": diagnostics["optimizer"]["maximin_qfi_efficiency"],
                    "povm_completeness_error": diagnostics["optimizer"]["completeness_error"],
                }
                built_diagnostics.append(compact_diagnostics)
                for key in strategy_keys:
                    band_cache_rows[key].append(covariances[key])
                if time_index % 6 == 0 or time_index + 1 == len(hour_angles):
                    print(
                        "[receiver-cache] "
                        f"band={band_index + 1}/{len(lam_edges_nm) - 1} "
                        f"epoch={time_index + 1}/{len(hour_angles)} "
                        f"single-min={diagnostics['single_qfi_efficiency']['min']:.3f} "
                        f"promoted-min={diagnostics['promoted_qfi_efficiency']['min']:.3f}",
                        flush=True,
                    )
            else:
                covariances = {
                    key: np.asarray(cache_data[f"cov_{key}"][band_index, time_index], dtype=float)
                    for key in strategy_keys
                }

            common_standard_normal = rng.normal(size=n_edge + n_closure)
            for key in strategy_keys:
                covariance = 0.5 * (covariances[key] + covariances[key].T)
                noise = covariance_square_root(covariance) @ common_standard_normal
                measured_amplitude = np.maximum(amplitude_true + noise[:n_edge], 0.0)
                measured_loop = loop_true + noise[n_edge:]
                representative_edge_phase = selected_to_edge @ measured_loop
                vis[key].append(measured_amplitude * np.exp(1j * representative_edge_phase))
                closure_data[key].append(measured_loop)
                amp_data[key].append(measured_amplitude)
                amp_sigma[key].append(np.sqrt(np.maximum(np.diag(covariance)[:n_edge], 1.0e-300)))
                closure_covariance = covariance[n_edge:, n_edge:]
                sigma_q_cov[key].append(closure_covariance)
                sigma_q[key].append(np.sqrt(np.maximum(np.diag(closure_covariance), 1.0e-300)))
                edge_covariance = selected_to_edge @ closure_covariance @ selected_to_edge.T
                sigma_edge[key].append(np.sqrt(np.maximum(np.diag(edge_covariance), 1.0e-300)))
                mixed_cov[key].append(covariance)
            band["u"].append(uu)
            band["v"].append(vv)
            amp_true_rows.append(amplitude_true)
            closure_reference_rows.append(loop_true)

        if cache_data is None:
            for key in strategy_keys:
                built_covariances[key].append(np.asarray(band_cache_rows[key]))
        band["u"] = np.concatenate(band["u"])
        band["v"] = np.concatenate(band["v"])
        band["amp_true"] = np.concatenate(amp_true_rows)
        band["closure_reference"] = np.asarray(closure_reference_rows)
        for key in strategy_keys:
            band[f"vis_{key}"] = np.concatenate(vis[key])
            band[f"sigma_{key}"] = np.concatenate(sigma_edge[key])
            band[f"sigmaq_{key}"] = np.concatenate(sigma_q[key])
            band[f"sigmaqcov_{key}"] = np.asarray(sigma_q_cov[key])
            band[f"mixedcov_{key}"] = np.asarray(mixed_cov[key])
            band[f"closure_{key}"] = np.asarray(closure_data[key])
            band[f"amp_{key}"] = np.concatenate(amp_data[key])
            band[f"amp_sigma_{key}"] = np.concatenate(amp_sigma[key])
            all_amp_sigma[key].append(band[f"amp_sigma_{key}"])
            all_amp_data[key].append(band[f"amp_{key}"])
        band["amp"] = np.asarray(band["amp_edge_uniform"]).copy()
        band["amp_sigma"] = np.asarray(band["amp_sigma_edge_uniform"]).copy()
        all_amp_true.append(np.asarray(band["amp_true"]))
        bands.append(band)

    if cache_data is None:
        RECEIVER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            RECEIVER_CACHE_PATH,
            **{f"cov_{key}": np.asarray(value) for key, value in built_covariances.items()},
        )
        metric_keys = (
            "edge_qfi_efficiency",
            "single_qfi_efficiency",
            "promoted_qfi_efficiency",
        )
        aggregate_efficiency: dict[str, object] = {}
        for metric_key in metric_keys:
            aggregate_efficiency[metric_key] = {
                component: {
                    "min": float(np.min([row[metric_key][component] for row in built_diagnostics])),
                    "mean": float(np.mean([row[metric_key][component] for row in built_diagnostics])),
                    "max": float(np.max([row[metric_key][component] for row in built_diagnostics])),
                }
                for component in ("min", "mean", "max")
            }
        cache_metadata = {
            "definition": (
                "Per-sample aggregate covariance in (full E edge amplitudes, C selected closure phases). "
                "The optimal_singlecopy and promoted_singlecopy entries share the same POVM hash at every sample."
            ),
            "n_station": n_station,
            "n_edge": n_edge,
            "n_closure": n_closure,
            "n_wavelength_bins": len(lam_edges_nm) - 1,
            "n_time_windows": len(hour_angles),
            "exposure_s": float(aug.EXPOSURE_S),
            "lambda_min_nm": float(aug.LAMBDA_MIN_NM),
            "lambda_max_nm": float(aug.LAMBDA_MAX_NM),
            "lambda_step_nm": float(aug.LAMBDA_STEP_NM),
            "eps_station": float(EPS_STATION_RUN),
            "existing_coupled_area_fraction": float(EXISTING_COUPLED_AREA_FRACTION),
            "remote_diameter_m": float(REMOTE_DIAMETER_M),
            "photon_collection_efficiency": float(PHOTON_COLLECTION_EFFICIENCY),
            "amplitude_branch_fraction": AMPLITUDE_BRANCH_FRACTION,
            "phase_branch_fraction": PHASE_BRANCH_FRACTION,
            "coherent_block_size": COHERENT_BLOCK_SIZE,
            "promotion_model": PROMOTION_MODEL,
            "finite_ns_qlan_fisher_factor": FINITE_NS_QLAN_FISHER_FACTOR,
            "phase_povm_objective": PHASE_POVM_OBJECTIVE,
            "finite_block_caveat": (
                "The finite-n_s covariance is obtained by inflating the QLAN covariance with "
                "a universal factor calibrated at one 655-nm near-transit working point; it is "
                "not a per-bin exact finite-copy POVM."
            ),
            "aggregate_efficiency": aggregate_efficiency,
            "povm_hashes": [row["povm_hash"] for row in built_diagnostics],
            "sample_diagnostics": built_diagnostics,
        }
        RECEIVER_CACHE_METADATA_PATH.write_text(json.dumps(cache_metadata, indent=2) + "\n")

    amp_true_flat = np.concatenate(all_amp_true)
    stats = {
        "case": case.key,
        "n_station": n_station,
        "n_edge_amplitude": n_edge,
        "n_closure_phase": n_closure,
        "phase_parameter_space": "closure subspace only",
        "closure_data_representation": (
            "real-valued local closure coordinates, unwrapped about each "
            "wavelength--epoch POVM working point"
        ),
        "amplitude_parameter_space": "complete E-dimensional edge space",
        "existing_coupled_area_fraction": float(EXISTING_COUPLED_AREA_FRACTION),
        "remote_diameter_m": float(REMOTE_DIAMETER_M),
        "photon_collection_efficiency": float(PHOTON_COLLECTION_EFFICIENCY),
        "receiver_cache_npz": str(RECEIVER_CACHE_PATH),
        "receiver_cache_metadata": str(RECEIVER_CACHE_METADATA_PATH),
        "receiver_cache": cache_metadata,
        "amplitude_branch_fraction": AMPLITUDE_BRANCH_FRACTION,
        "phase_branch_fraction": PHASE_BRANCH_FRACTION,
        "coherent_block_size": COHERENT_BLOCK_SIZE,
        "promotion_model": PROMOTION_MODEL,
        "finite_ns_qlan_fisher_factor": FINITE_NS_QLAN_FISHER_FACTOR,
        "paired_noise_draws": True,
        "mixed_likelihood": "full (E+C)x(E+C) covariance including amplitude-phase cross block",
        "endpoint_coverage_g_lambda": endpoint_coverage,
        "source_spectral_model": getattr(opt.base, "SOURCE_COMPONENT_SPECTRAL_MODEL", "achromatic source morphology"),
        "amplitude_snr_median_by_strategy": {
            key: float(np.median(amp_true_flat / np.maximum(np.concatenate(all_amp_sigma[key]), 1.0e-300)))
            for key in strategy_keys
        },
        "amplitude_noise_rms_by_strategy": {
            key: float(
                np.sqrt(np.mean((np.concatenate(all_amp_data[key]) - amp_true_flat) ** 2))
            )
            for key in strategy_keys
        },
    }
    return bands, stats, truth, axis_uas


def simulate_good_bands(case: aug.NetworkCase, splits: dict[str, np.ndarray]):
    split_sim.configure()
    apply_sample_stress_runtime()
    # ``patched_variant`` changes ngc.make_source_factory, while
    # ``patched_source`` actually installs the resulting factory into
    # plot_prl_broadband_clean.base.make_source.  Both are required here
    # because split_sim builds the truth image by calling base.make_source
    # directly rather than going through amp_rml.simulate_case.
    old_closure_basis = val.closure_basis
    val.closure_basis = latest_closure_basis
    try:
        with morph.patched_variant(GOOD_VARIANT), ngc.patched_source(GOOD_SOURCE):
            bands, stats, truth, axis_uas = simulate_mixed_povm_bands(case)
    finally:
        val.closure_basis = old_closure_basis
    bands, thinning_stats = thin_bands_every_other_time_row(bands, case, KEEP_EVERY_TIME_ROW)
    stats["sample_stress_test"] = {
        "run_tag": RUN_TAG,
        "observing_days": OBSERVING_DAYS,
        "base_exposure_s": BASE_EXPOSURE_S,
        "exposure_scale": EXPOSURE_SCALE,
        "effective_exposure_s": BASE_EXPOSURE_S * EXPOSURE_SCALE,
        "sample_cadence_s": SAMPLE_CADENCE_S_RUN,
        "exposure_gap_s": EXPOSURE_GAP_S_RUN,
        "fourier_coverage_h": FOURIER_COVERAGE_H_RUN,
        "eps_station": EPS_STATION_RUN,
        "eps_pair": EPS_PAIR_RUN,
        "eps_direct_extra": EPS_DIRECT_EXTRA_RUN,
        "post_average_drift_std_rad": POST_AVERAGE_DRIFT_STD_RUN,
        "remote_x_scale": REMOTE_X_SCALE,
        "remote_y_scale": REMOTE_Y_SCALE,
        "existing_coupled_area_fraction": EXISTING_COUPLED_AREA_FRACTION,
        "remote_diameter_m": REMOTE_DIAMETER_M,
        "photon_collection_efficiency": PHOTON_COLLECTION_EFFICIENCY,
        "rng_seed": FIG2_RNG_SEED,
        "rml_fit_n_pix": val.FIT_N_PIX,
        "rml_adam_iter": val.ADAM_ITER,
        "rml_adam_lr": val.ADAM_LR,
        "rml_prior_weight": RML_PRIOR_WEIGHT_RUN,
        "rml_tv_weight": RML_TV_WEIGHT_RUN,
        "rml_entropy_weight": RML_ENTROPY_WEIGHT_RUN,
        "rml_morphology_specific_core_prior": "none",
        "per_band_rml": PER_BAND_RML,
        "wavelength_bin_width_nm": LAMBDA_STEP_NM_RUN,
        "display_stretch": "arcsinh",
        "display_asinh_q": DISPLAY_ASINH_Q,
        "legacy_display_log_vmin": DISPLAY_LOG_VMIN,
        "rml_phase_chi2_selection": RML_PHASE_CHI2_SELECTION,
        "rml_phase_chi2_target": RML_PHASE_CHI2_TARGET,
        "rml_phase_chi2_min": RML_PHASE_CHI2_MIN,
        "rml_phase_chi2_max": RML_PHASE_CHI2_MAX,
        "rml_amp_chi2_target": RML_AMP_CHI2_TARGET,
        "rml_amp_chi2_min": RML_AMP_CHI2_MIN,
        "rml_amp_chi2_max": RML_AMP_CHI2_MAX,
        "split_diagnostics": {
            "status": "not used by the paired POVM benchmark",
            "phase_space": "closure subspace only",
            "amplitude_space": "complete E-dimensional edge space",
        },
        "corrected_photon_budget_model": (
            "All three lower panels use the same 50:50 amplitude/phase resource split. "
            "Amplitude data span all E edges, whereas phase data span only the C-dimensional "
            "closure subspace. The first receiver is uniform edge-first, the second is the "
            "locally optimized single-copy POVM, and the third coherently lifts that same POVM's "
            f"score fluctuations in n_s={COHERENT_BLOCK_SIZE} collective blocks."
        ),
        **thinning_stats,
    }
    prior_full = morph.amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = {
        "prior": morph.amp_rml.project_flux_positive(prior, smooth_pix=0.0),
    }
    for strategy, _label, start_name, _color in STRATEGIES:
        starts[start_name] = val.rebin_image_average(
            morph.amp_rml.quick_dirty_start(bands, strategy, truth),
            val.FIT_N_PIX,
        )
    return bands, stats, truth, axis_uas, prior, starts


def bands_for_strategy(bands: list[dict[str, np.ndarray]], strategy: str) -> list[dict[str, np.ndarray]]:
    """Return band views whose amplitude likelihood matches ``strategy``."""
    amp_key = f"amp_{strategy}"
    sigma_key = f"amp_sigma_{strategy}"
    out: list[dict[str, np.ndarray]] = []
    for band in bands:
        if amp_key not in band or sigma_key not in band:
            out.append(dict(band))
            continue
        copied = dict(band)
        copied["amp"] = np.asarray(band[amp_key], dtype=float)
        copied["amp_sigma"] = np.asarray(band[sigma_key], dtype=float)
        out.append(copied)
    return out


def wavelength_bin_edges_nm() -> np.ndarray:
    edges_nm = np.arange(
        LAMBDA_MIN_NM_RUN,
        LAMBDA_MAX_NM_RUN + 0.5 * LAMBDA_STEP_NM_RUN,
        LAMBDA_STEP_NM_RUN,
    )
    edges_nm[-1] = LAMBDA_MAX_NM_RUN
    return edges_nm


def wavelength_bin_centers_and_weights(source: ngc.SourceModel = GOOD_SOURCE) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    weights = []
    for lo_nm, hi_nm in zip(wavelength_bin_edges_nm()[:-1], wavelength_bin_edges_nm()[1:]):
        center_nm = float(np.sqrt(float(lo_nm) * float(hi_nm)))
        freq = opt.base.C_LIGHT / (center_nm * 1.0e-9)
        freq_lo = opt.base.C_LIGHT / (float(hi_nm) * 1.0e-9)
        freq_hi = opt.base.C_LIGHT / (float(lo_nm) * 1.0e-9)
        centers.append(center_nm)
        weights.append(morph.total_source_fnu_jy(source, center_nm) * (freq_hi - freq_lo) / freq)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= float(np.sum(weights_arr))
    return np.asarray(centers, dtype=float), weights_arr


def truth_image_at_wavelength(center_nm: float, n_pix: int, half_width_uas: float) -> tuple[np.ndarray, np.ndarray]:
    with morph.patched_variant(GOOD_VARIANT), ngc.patched_source(GOOD_SOURCE):
        return opt.base.make_source_at_wavelength_nm(n_pix, half_width_uas, float(center_nm))


def aggregate_residual_diagnostics(results: list[dict]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    amp_z = np.concatenate([item["best"]["residual_arrays"]["amp_z"] for item in results])
    phase_z = np.concatenate([item["best"]["residual_arrays"]["phase_z"] for item in results])
    finite_amp = amp_z[np.isfinite(amp_z)]
    finite_phase = phase_z[np.isfinite(phase_z)]
    diagnostics = {
        "amp_reduced_chi2": float(np.mean(finite_amp**2)),
        "amp_z_rms": float(np.sqrt(np.mean(finite_amp**2))),
        "amp_z_p50_abs": float(np.percentile(np.abs(finite_amp), 50.0)),
        "amp_z_p90_abs": float(np.percentile(np.abs(finite_amp), 90.0)),
        "phase_reduced_chi2": float(np.mean(finite_phase**2)),
        "phase_z_rms": float(np.sqrt(np.mean(finite_phase**2))),
        "phase_z_p50_abs": float(np.percentile(np.abs(finite_phase), 50.0)),
        "phase_z_p90_abs": float(np.percentile(np.abs(finite_phase), 90.0)),
        "phase_floor_rad": float(morph.amp_rml.PHASE_FLOOR_RAD),
        "n_amp_samples": int(finite_amp.size),
        "n_phase_samples": int(finite_phase.size),
    }
    return diagnostics, {"amp_z": finite_amp, "phase_z": finite_phase}


def weighted_image_stack(images: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.zeros_like(images[0], dtype=float)
    for image, weight in zip(images, weights):
        stacked += float(weight) * np.asarray(image, dtype=float)
    return morph.amp_rml.project_flux_positive(stacked, smooth_pix=0.0)


def make_single_band_starts(
    band: dict[str, np.ndarray],
    strategy: str,
    start_name: str,
    prior: np.ndarray,
    truth_band: np.ndarray,
    band_index: int,
) -> dict[str, np.ndarray]:
    fit_n_pix = int(prior.shape[0])
    starts = {
        "prior": morph.amp_rml.project_flux_positive(prior, smooth_pix=0.0),
    }
    if start_name != "prior":
        starts[start_name] = val.rebin_image_average(
            morph.amp_rml.quick_dirty_start([band], strategy, truth_band),
            fit_n_pix,
        )
    if EXTERNAL_BAND_START_CACHE:
        with np.load(Path(EXTERNAL_BAND_START_CACHE), allow_pickle=False) as archive:
            if EXTERNAL_BAND_START_KEY not in archive.files:
                raise KeyError(
                    f"External start cache {EXTERNAL_BAND_START_CACHE} lacks "
                    f"{EXTERNAL_BAND_START_KEY!r}"
                )
            external = np.asarray(archive[EXTERNAL_BAND_START_KEY], dtype=float)
        if external.shape[0] <= band_index:
            raise ValueError(
                f"External start cache contains {external.shape[0]} bands; "
                f"requested band {band_index}"
            )
        image = np.asarray(external[band_index], dtype=float)
        if image.shape != prior.shape:
            image = val.rebin_image_average(image, fit_n_pix)
        starts["external_band_start"] = morph.amp_rml.project_flux_positive(
            image, smooth_pix=0.0
        )
    return starts


def chi2_target_score(candidate: dict) -> float:
    residuals = candidate["residuals"]
    amp_chi2 = max(float(residuals["amp_reduced_chi2"]), 1e-300)
    phase_chi2 = max(float(residuals["phase_reduced_chi2"]), 1e-300)
    amp_score = abs(np.log(amp_chi2 / max(RML_AMP_CHI2_TARGET, 1e-300)))
    high_chi_penalty = max(0.0, np.log(amp_chi2 / max(RML_CHI2_THRESHOLD, 1e-300)))
    if RML_PHASE_CHI2_SELECTION == "minimize":
        phase_score = np.log(phase_chi2)
        return float(phase_score + 0.10 * amp_score + 5.0 * high_chi_penalty + 0.001 * candidate["objective"])
    phase_score = abs(np.log(phase_chi2 / max(RML_PHASE_CHI2_TARGET, 1e-300)))
    low_amp_penalty = max(0.0, np.log(max(RML_AMP_CHI2_MIN, 1e-300) / amp_chi2))
    return float(max(phase_score, amp_score) + 2.0 * high_chi_penalty + 1.0 * low_amp_penalty + 0.001 * candidate["objective"])


def chi2_target_passed(candidate: dict) -> bool:
    residuals = candidate["residuals"]
    amp_chi2 = float(residuals["amp_reduced_chi2"])
    phase_chi2 = float(residuals["phase_reduced_chi2"])
    if RML_PHASE_CHI2_SELECTION == "minimize":
        return np.isfinite(phase_chi2) and amp_chi2 < RML_CHI2_THRESHOLD
    if candidate.get("strategy") == "all":
        return amp_chi2 < RML_CHI2_THRESHOLD and 0.75 <= phase_chi2 <= 1.35
    return (
        RML_AMP_CHI2_MIN <= amp_chi2 <= RML_AMP_CHI2_MAX
        and RML_PHASE_CHI2_MIN <= phase_chi2 <= RML_PHASE_CHI2_MAX
    )


def expand_reconstruction_checkpoints(candidate: dict) -> list[dict]:
    """Return the final candidate plus Adam checkpoint images with diagnostics."""
    history = candidate.get("history", {})
    checkpoints = history.pop("_checkpoint_images", None)
    expanded = [candidate]
    if not checkpoints:
        return expanded
    for item in checkpoints:
        image = val.upsample_image_nearest(item["image"], len(candidate["image"]))
        residuals, residual_arrays = val.residual_diagnostics(
            item["image"],
            candidate["bands"],
            candidate["case_obj"],
            candidate["strategy"],
            candidate["axis_uas"],
        )
        metrics = morph.amp_rml.metrics_for(image, candidate["truth"], candidate["axis_uas"])
        expanded.append(
            {
                **candidate,
                "image": image,
                "fit_image": item["image"],
                "history": {
                    key: value
                    for key, value in history.items()
                    if not str(key).startswith("_")
                },
                "checkpoint_iteration": int(item["iteration"]),
                "objective": float(item["values"]["objective"]),
                "metrics": metrics,
                "residuals": residuals,
                "residual_arrays": residual_arrays,
                "validation_score": val.residual_selection_score(residuals, float(item["values"]["objective"])),
            }
        )
    return expanded


def run_strategy_for_bands(strategy: str, label: str, start_name: str, case, bands, truth, axis_uas, prior, starts) -> dict:
    old_strategy = val.STRATEGY
    old_optimizer = val.OPTIMIZER
    old_adam_iter = val.ADAM_ITER
    old_adam_lr = val.ADAM_LR
    try:
        val.STRATEGY = strategy
        val.OPTIMIZER = "adam"
        start_order = []
        for name in (start_name, "external_band_start", "prior"):
            if name in starts and name not in start_order:
                start_order.append(name)
        iter_factors = [1.0, 1.8, 3.0, 4.6]
        lr_factors = [1.0, 0.75, 0.55, 0.40]
        candidates = []
        attempts = []
        best_passed = None
        selection_tag = "minphase" if RML_PHASE_CHI2_SELECTION == "minimize" else "chi2gate"
        for pass_index in range(RML_CHI2_MAX_PASSES):
            val.ADAM_ITER = int(round(RML_ADAM_ITER_RUN * iter_factors[min(pass_index, len(iter_factors) - 1)]))
            val.ADAM_LR = RML_ADAM_LR_RUN * lr_factors[min(pass_index, len(lr_factors) - 1)]
            config = {
                "label": f"adam_{selection_tag}_p{pass_index + 1}",
                "prior": morph.amp_rml.PRIOR_WEIGHT,
                "tv": morph.amp_rml.TV_WEIGHT,
                "entropy": morph.amp_rml.ENTROPY_WEIGHT,
                "step": morph.amp_rml.STEP,
            }
            pass_candidates = []
            for start_idx, candidate_start in enumerate(start_order, start=1):
                done_units = pass_index * len(start_order) + start_idx - 1
                total_units = RML_CHI2_MAX_PASSES * len(start_order)
                bar_width = 22
                filled = int(round(bar_width * done_units / max(total_units, 1)))
                bar = "#" * filled + "." * (bar_width - filled)
                print(
                    f"[chi2-gate] {strategy:24s} [{bar}] pass={pass_index + 1}/{RML_CHI2_MAX_PASSES} "
                    f"start={candidate_start} iter={val.ADAM_ITER} lr={val.ADAM_LR:.4g}",
                    flush=True,
                )
                candidate = val.run_single_reconstruction(
                    case=case,
                    bands=bands,
                    truth=truth,
                    axis_uas=axis_uas,
                    prior=prior,
                    start_name=candidate_start,
                    start=starts[candidate_start],
                    config=config,
                    split_label=f"fig2_{selection_tag}",
                )
                candidate["bands"] = bands
                candidate["case_obj"] = case
                candidate["axis_uas"] = axis_uas
                candidate["truth"] = truth
                expanded = expand_reconstruction_checkpoints(candidate)
                candidates.extend(expanded)
                pass_candidates.extend(expanded)
            best = min(pass_candidates, key=chi2_target_score)
            residuals = best["residuals"]
            amp_chi2 = float(residuals["amp_reduced_chi2"])
            phase_chi2 = float(residuals["phase_reduced_chi2"])
            passed = chi2_target_passed(best)
            attempts.append(
                {
                    "pass": pass_index + 1,
                    "adam_iter": int(val.ADAM_ITER),
                    "adam_lr": float(val.ADAM_LR),
                    "best_start": str(best["start"]),
                    "checkpoint_iteration": int(best.get("checkpoint_iteration", -1)),
                    "amp_chi2": amp_chi2,
                    "phase_chi2": phase_chi2,
                    "chi2_target_score": chi2_target_score(best),
                    "passed": bool(passed),
                }
            )
            print(
                f"[chi2-gate] {strategy:24s} best pass={pass_index + 1}: "
                f"amp_chi2={amp_chi2:.3g}, phase_chi2={phase_chi2:.3g}, passed={passed}",
                flush=True,
            )
            if passed and RML_PHASE_CHI2_SELECTION != "minimize":
                best_passed = best
                break
        passed_candidates = [candidate for candidate in candidates if chi2_target_passed(candidate)]
        best = best_passed or (min(passed_candidates, key=chi2_target_score) if passed_candidates else min(candidates, key=chi2_target_score))
        return {
            "strategy": strategy,
            "label": label,
            "best": best,
            "candidates": candidates,
            "chi2_attempts": attempts,
        }
    finally:
        val.STRATEGY = old_strategy
        val.OPTIMIZER = old_optimizer
        val.ADAM_ITER = old_adam_iter
        val.ADAM_LR = old_adam_lr


def run_strategy_per_band(strategy: str, label: str, start_name: str, case, bands, truth, axis_uas, prior) -> dict:
    centers_nm, weights = wavelength_bin_centers_and_weights(GOOD_SOURCE)
    if len(centers_nm) != len(bands):
        raise ValueError(f"Expected {len(centers_nm)} wavelength bins, got {len(bands)} simulated bands")

    band_results = []
    band_summaries = []
    for band_index, (band, center_nm, weight) in enumerate(zip(bands, centers_nm, weights), start=1):
        bar_width = 18
        filled = int(round(bar_width * (band_index - 1) / max(len(bands), 1)))
        bar = "#" * filled + "." * (bar_width - filled)
        print(
            f"[per-band-rml] {strategy:24s} [{bar}] bin={band_index}/{len(bands)} "
            f"lambda={center_nm:.1f} nm weight={weight:.3f}",
            flush=True,
        )
        truth_band, _axis_band = truth_image_at_wavelength(center_nm, aug.N_PIX, aug.HALF_WIDTH_UAS)
        band_starts = make_single_band_starts(
            band, strategy, start_name, prior, truth_band, band_index - 1
        )
        band_result = run_strategy_for_bands(
            strategy,
            label,
            start_name,
            case,
            [band],
            truth_band,
            axis_uas,
            prior,
            band_starts,
        )
        band_results.append(band_result)
        best = band_result["best"]
        residuals = best["residuals"]
        band_summaries.append(
            {
                "band_index": int(band_index - 1),
                "lambda_center_nm": float(center_nm),
                "photon_weight": float(weight),
                "best_start": str(best["start"]),
                "amp_chi2": float(residuals["amp_reduced_chi2"]),
                "phase_chi2": float(residuals["phase_reduced_chi2"]),
                "global_corr": float(best["metrics"]["global_corr"]),
                "blr_corr": float(best["metrics"]["blr_corr"]),
            }
        )

    display_stack = weighted_image_stack([item["best"]["image"] for item in band_results], weights)
    fit_stack = weighted_image_stack([item["best"]["fit_image"] for item in band_results], weights)
    residuals, residual_arrays = aggregate_residual_diagnostics(band_results)
    metrics = morph.amp_rml.metrics_for(display_stack, truth, axis_uas)
    objective = float(sum(float(weight) * float(item["best"]["objective"]) for item, weight in zip(band_results, weights)))
    selection_tag = "minphase" if RML_PHASE_CHI2_SELECTION == "minimize" else "chi2gate"
    best = {
        "case": case.key,
        "split": f"fig2_per_10nm_{selection_tag}",
        "strategy": strategy,
        "optimizer": "per_band_adam",
        "config": f"per_band_{selection_tag}",
        "start": "photon-weighted per-band stack",
        "objective": objective,
        "amp_objective": float(sum(float(weight) * float(item["best"]["amp_objective"]) for item, weight in zip(band_results, weights))),
        "phase_objective": float(sum(float(weight) * float(item["best"]["phase_objective"]) for item, weight in zip(band_results, weights))),
        "image": display_stack,
        "fit_image": fit_stack,
        "history": {},
        "metrics": metrics,
        "residuals": residuals,
        "residual_arrays": residual_arrays,
        "validation_score": val.residual_selection_score(residuals, objective),
    }
    print(
        f"[per-band-rml] {strategy:24s} aggregate: "
        f"amp_chi2={residuals['amp_reduced_chi2']:.3g}, "
        f"phase_chi2={residuals['phase_reduced_chi2']:.3g}, "
        f"BLR r={metrics['blr_corr']:.3f}, all r={metrics['global_corr']:.3f}",
        flush=True,
    )
    return {
        "strategy": strategy,
        "label": label,
        "best": best,
        "candidates": [],
        "chi2_attempts": band_summaries,
        "band_results": band_results,
        "per_band_rml": True,
    }


def run_strategy(strategy: str, label: str, start_name: str, case, bands, truth, axis_uas, prior, starts) -> dict:
    strategy_bands = bands_for_strategy(bands, strategy)
    if PER_BAND_RML:
        return run_strategy_per_band(strategy, label, start_name, case, strategy_bands, truth, axis_uas, prior)
    strategy_starts = {
        name: value
        for name, value in starts.items()
        if name == "prior" or name == start_name
    }
    return run_strategy_for_bands(strategy, label, start_name, case, strategy_bands, truth, axis_uas, prior, strategy_starts)


def component_spectrum_rows(source: ngc.SourceModel = GOOD_SOURCE) -> list[dict[str, float]]:
    base_fractions = morph.variant_component_fractions(GOOD_VARIANT)
    wavelengths_nm = np.linspace(LAMBDA_MIN_NM_RUN, LAMBDA_MAX_NM_RUN, 101)
    rows = []
    for wavelength_nm in wavelengths_nm:
        total_mjy = 1.0e3 * morph.total_source_fnu_jy(source, float(wavelength_nm))
        component_fluxes = morph.component_flux_fnu_jy(base_fractions, source, float(wavelength_nm))
        fractions = morph.component_flux_fractions(base_fractions, float(wavelength_nm), source=source)
        row = {
            "wavelength_nm": float(wavelength_nm),
            "total_mjy": float(total_mjy),
        }
        for name, fraction in fractions.items():
            row[f"{name}_mjy"] = float(1.0e3 * component_fluxes[name])
            row[f"{name}_fraction"] = float(fraction)
        rows.append(row)
    return rows


def mjy_to_abmag(values_mjy: np.ndarray) -> np.ndarray:
    values_jy = np.asarray(values_mjy, dtype=float) * 1.0e-3
    out = np.full(values_jy.shape, np.nan, dtype=float)
    good = values_jy > 0.0
    out[good] = -2.5 * np.log10(values_jy[good] / 3631.0)
    return out


def normalized_image_for_display(image: np.ndarray) -> np.ndarray:
    clipped = np.asarray(image, dtype=float).copy()
    clipped -= np.percentile(clipped, 0.8)
    scale = float(np.percentile(clipped, 99.7))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(np.abs(clipped)))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(clipped)
    return np.clip(clipped / scale, 0.0, 1.0)


def asinh_display(image: np.ndarray) -> np.ndarray:
    normalized = normalized_image_for_display(image)
    q = max(DISPLAY_ASINH_Q, 1e-6)
    return np.arcsinh(normalized / q) / np.arcsinh(1.0 / q)


def log_display(image: np.ndarray) -> np.ndarray:
    normalized = normalized_image_for_display(image)
    return np.clip(normalized, DISPLAY_LOG_VMIN, 1.0)


def asinh_tick(value: float) -> float:
    q = max(DISPLAY_ASINH_Q, 1e-6)
    return float(np.arcsinh(float(value) / q) / np.arcsinh(1.0 / q))


def plot_component_spectra(ax) -> None:
    rows = component_spectrum_rows()
    wavelengths = np.asarray([row["wavelength_nm"] for row in rows])
    style_map = [
        ("total_mjy", "total", "black", 1.55, "-"),
        ("blr_mjy", "BLR", "#d00000", 1.50, "-"),
        ("core_mjy", "core", "#005f73", 1.05, "-"),
        ("disk_mjy", "disk", "#4361ee", 1.05, "--"),
        ("continuum_mjy", "cont.", "#4361ee", 1.05, "--"),
        ("plume_mjy", "plume", "#f77f00", 1.05, "-."),
        ("halo_mjy", "halo", "#6a994e", 1.05, ":"),
    ]
    plotted_mags = []
    for key, label, color, lw, ls in style_map:
        if key not in rows[0]:
            continue
        values = mjy_to_abmag(np.asarray([row[key] for row in rows]))
        if not np.any(np.isfinite(values)):
            continue
        plotted_mags.append(values[np.isfinite(values)])
        ax.plot(wavelengths, values, color=color, lw=lw, ls=ls, label=label)
    line_marks = [
        (486.1 * (1.0 + NGC4151_REDSHIFT), r"H$\beta$"),
        (656.3 * (1.0 + NGC4151_REDSHIFT), r"H$\alpha$"),
    ]
    total_mag = mjy_to_abmag(np.asarray([row["total_mjy"] for row in rows]))
    y_bottom = float(np.nanmax(total_mag))
    y_top = float(np.nanmin(total_mag))
    for wavelength_nm, label in line_marks:
        if wavelength_nm < LAMBDA_MIN_NM_RUN or wavelength_nm > LAMBDA_MAX_NM_RUN:
            continue
        ax.axvline(wavelength_nm, color="#d00000", lw=0.65, alpha=0.55)
        ax.annotate(
            f"{label}\n{wavelength_nm:.1f} nm",
            xy=(wavelength_nm, y_top + 0.32 * (y_bottom - y_top)),
            xytext=(7.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.45,
            color="#9b0000",
            arrowprops={"arrowstyle": "-", "color": "#9b0000", "lw": 0.55, "shrinkA": 0.0, "shrinkB": 0.0},
        )
    ax.set_xlim(LAMBDA_MIN_NM_RUN, LAMBDA_MAX_NM_RUN)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("AB magnitude")
    if plotted_mags:
        all_mag = np.concatenate(plotted_mags)
        mag_min = float(np.nanmin(all_mag))
        mag_max = float(np.nanmax(all_mag))
        pad = max(0.08 * (mag_max - mag_min), 0.10)
        ax.set_ylim(mag_max + pad, mag_min - pad)
    else:
        ax.invert_yaxis()
    ax.set_title("Source spectrum")
    ax.grid(alpha=0.22, lw=0.45)
    ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.3, columnspacing=0.7, fontsize=5.8)
    ax.set_box_aspect(1)


def result_rows(results: list[dict], truth: np.ndarray, axis_uas: np.ndarray) -> list[dict]:
    rows = []
    for result in results:
        best = result["best"]
        metrics = best["metrics"]
        residuals = best["residuals"]
        rows.append(
            {
                "strategy": result["strategy"],
                "label": result["label"],
                "best_start": best["start"],
                "global_corr": float(metrics["global_corr"]),
                "blr_corr": float(metrics["blr_corr"]),
                "profile_rmse": morph.profile_rmse(truth, best["image"], axis_uas),
                "amp_chi2": float(residuals["amp_reduced_chi2"]),
                "phase_chi2": float(residuals["phase_reduced_chi2"]),
            }
        )
    return rows


def save_selected_result_cache(
    stats: dict[str, object],
    truth: np.ndarray,
    axis_uas: np.ndarray,
    results: list[dict],
) -> Path:
    cache_npz = OUT / f"{OUTPUT_STEM}_selected_display_cache.npz"
    payload = {
        "truth": np.asarray(truth, dtype=float),
        "axis_uas": np.asarray(axis_uas, dtype=float),
    }
    for result in results:
        strategy = result["strategy"]
        payload[f"{strategy}_image"] = np.asarray(result["best"]["image"], dtype=float)
        band_results = result.get("band_results", [])
        if band_results:
            payload[f"{strategy}_band_images"] = np.stack(
                [np.asarray(item["best"]["image"], dtype=float) for item in band_results]
            )
            payload[f"{strategy}_band_fit_images"] = np.stack(
                [np.asarray(item["best"]["fit_image"], dtype=float) for item in band_results]
            )
    np.savez_compressed(cache_npz, **payload)
    stats["selected_display_cache_npz"] = str(cache_npz)
    return cache_npz


def plot_results(case, stats, truth, axis_uas, results: list[dict]) -> tuple[Path, Path]:
    axis = axis_uas
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    result_by = {item.get("plot_slot", item["strategy"]): item for item in results}
    stations, diameters, _names, is_added = aug.station_table_from_case(case)
    plt.rcParams.update(
        {
            "font.size": 8.4,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
        }
    )
    fig = plt.figure(figsize=(7.85, 6.05), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.34)
    fig.subplots_adjust(left=0.075, right=0.900, bottom=0.092, top=0.940)
    image_axes = []
    panel_axes = {}

    ax = fig.add_subplot(gs[0, 0])
    panel_axes["array_topology"] = ax
    for added, marker, color, label in (
        (False, "o", "#005f73", "existing"),
        (True, "^", "#ae2012", f"remote {REMOTE_DIAMETER_M:g} m"),
    ):
        mask = is_added == added
        if np.any(mask):
            ax.scatter(
                stations[mask, 0],
                stations[mask, 1],
                s=30 if added else 25,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
                zorder=3,
            )
    ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=55, marker="*", color="#ca6702", label="hub", zorder=4)
    for i, j in opt.base.edge_list(len(stations)):
        ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.83", lw=0.42, zorder=0)
    core_mask = ~is_added
    core_x0 = float(np.min(stations[core_mask, 0])) - 0.18
    core_x1 = float(np.max(stations[core_mask, 0])) + 0.18
    core_y0 = float(np.min(stations[core_mask, 1])) - 0.18
    core_y1 = float(np.max(stations[core_mask, 1])) + 0.18
    ax.add_patch(
        Rectangle(
            (core_x0, core_y0),
            core_x1 - core_x0,
            core_y1 - core_y0,
            facecolor="none",
            edgecolor="0.20",
            lw=0.65,
            ls="--",
            zorder=5,
        )
    )
    label_offsets = {
        4: (-0.72, 0.34),
        5: (-0.52, 0.34),
        6: (0.18, -0.58),
    }
    for i, (x, y) in enumerate(stations):
        if not is_added[i]:
            continue
        dx, dy = label_offsets.get(i, (0.2, 0.2))
        ax.text(x + dx, y + dy, f"S{i + 1}  {diameters[i]:g}m", fontsize=6.3)
    y_min = float(np.min(stations[:, 1])) - 0.82
    y_max = float(np.max(stations[:, 1])) + 0.82
    y_span = y_max - y_min
    x_min = float(np.min(stations[:, 0])) - 0.70
    x_max = x_min + y_span
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title("Array topology\ncore zoom inset")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.15, borderpad=0.1)

    inset = ax.inset_axes([0.55, 0.48, 0.42, 0.39])
    for i, j in opt.base.edge_list(len(stations)):
        if core_mask[i] and core_mask[j]:
            inset.plot(
                [stations[i, 0], stations[j, 0]],
                [stations[i, 1], stations[j, 1]],
                color="0.70",
                lw=0.50,
                zorder=0,
            )
    inset.scatter(
        stations[core_mask, 0],
        stations[core_mask, 1],
        s=18,
        marker="o",
        color="#005f73",
        edgecolor="white",
        linewidth=0.30,
        zorder=3,
    )
    inset.scatter([case.hub_km[0]], [case.hub_km[1]], s=34, marker="*", color="#ca6702", zorder=4)
    inset_offsets = {
        0: (-0.040, 0.092),
        1: (-0.158, 0.024),
        2: (0.046, -0.024),
    }
    for i in np.where(core_mask)[0]:
        x, y = stations[i]
        dx, dy = inset_offsets[i]
        inset.text(
            x + dx,
            y + dy,
            f"S{i + 1}",
            fontsize=5.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.15},
        )
    inset.text(
        case.hub_km[0] + 0.018,
        case.hub_km[1] - 0.126,
        "hub",
        fontsize=5.1,
        color="#7a3b00",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.12},
    )
    scale_len_km = 0.20
    scale_x0 = core_x0 + 0.080
    scale_x1 = scale_x0 + scale_len_km
    scale_y = core_y0 + 0.070
    tick_half = 0.018
    inset.plot([scale_x0, scale_x1], [scale_y, scale_y], color="0.12", lw=0.75, zorder=6)
    inset.plot([scale_x0, scale_x0], [scale_y - tick_half, scale_y + tick_half], color="0.12", lw=0.75, zorder=6)
    inset.plot([scale_x1, scale_x1], [scale_y - tick_half, scale_y + tick_half], color="0.12", lw=0.75, zorder=6)
    inset.text(
        0.5 * (scale_x0 + scale_x1),
        scale_y + 0.028,
        "0.2 km",
        ha="center",
        va="bottom",
        fontsize=4.7,
        color="0.12",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.10},
        zorder=7,
    )
    inset.set_xlim(core_x0, core_x1)
    inset.set_ylim(core_y0, core_y1)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_aspect("equal", adjustable="box")
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("0.20")

    ax = fig.add_subplot(gs[0, 1])
    panel_axes["uv_coverage"] = ax
    endpoint_items = (
        (f"{LAMBDA_MIN_NM_RUN:g}", "#005f73", 0.50),
        (f"{LAMBDA_MAX_NM_RUN:g}", "#ee9b00", 0.42),
    )
    for wavelength, color, alpha in endpoint_items:
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax.scatter(uu, vv, s=1.2, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax.scatter(-uu, -vv, s=1.2, color=color, alpha=0.62 * alpha)
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 256)
    for theta_uas, ls in ((60.0, ":"), (30.0, "--"), (10.0, "-.")):
        radius_g_lambda = 1.0 / (theta_uas * opt.base.UAS_TO_RAD) / 1.0e9
        ax.plot(
            radius_g_lambda * np.cos(theta_circle),
            radius_g_lambda * np.sin(theta_circle),
            ls=ls,
            lw=0.55,
            color="0.35",
            alpha=0.70,
            label=rf"{theta_uas:g} $\mu$as",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.1, borderpad=0.1)

    ax = fig.add_subplot(gs[0, 2])
    panel_axes["input_source"] = ax
    image_norm = colors.LogNorm(vmin=DISPLAY_LOG_VMIN, vmax=1.0)
    ax.imshow(log_display(truth), origin="lower", extent=extent, cmap="inferno", norm=image_norm)
    ax.set_title("Input source\ncore + disk + BLR")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    ax.set_box_aspect(1)
    image_axes.append(ax)

    bottom_labels = {
        "edge_uniform": "Uniform edge-first\nsingle-copy",
        "optimal_singlecopy": "Optimal POVM\nsingle-copy",
        "promoted_singlecopy": (
            PROMOTED_DISPLAY_LABEL
        ),
    }
    for col, strategy in enumerate(("edge_uniform", "optimal_singlecopy", "promoted_singlecopy")):
        result = result_by[strategy]
        ax = fig.add_subplot(gs[1, col])
        panel_axes[strategy] = ax
        ax.imshow(
            log_display(result["best"]["image"]),
            origin="lower",
            extent=extent,
            cmap="inferno",
            norm=image_norm,
        )
        m = result["best"]["metrics"]
        display_label = result.get("display_label", bottom_labels[strategy])
        ax.set_title(
            f"{display_label}\nBLR r={m['blr_corr']:.3f}, all r={m['global_corr']:.3f}",
            fontsize=8.3,
        )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        ax.set_box_aspect(1)
        image_axes.append(ax)

    cax = fig.add_axes([0.922, 0.135, 0.014, 0.670])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=image_norm, cmap="inferno"),
        cax=cax,
    )
    cbar.set_label("norm. brightness\n(log scale)", fontsize=8.0)
    tick_values = [DISPLAY_LOG_VMIN, 0.1, 0.3, 1.0]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{DISPLAY_LOG_VMIN:g}", "0.1", "0.3", "1"])

    fig.canvas.draw()
    boxes = {
        key: {
            "x0": float(ax.get_position().x0),
            "y0": float(ax.get_position().y0),
            "width": float(ax.get_position().width),
            "height": float(ax.get_position().height),
        }
        for key, ax in panel_axes.items()
    }
    widths = np.asarray([box["width"] for box in boxes.values()])
    heights = np.asarray([box["height"] for box in boxes.values()])
    fig_w_in, fig_h_in = fig.get_size_inches()
    physical_widths = widths * fig_w_in
    physical_heights = heights * fig_h_in
    stats["plot_layout"] = {
        "panel_boxes": boxes,
        "figure_size_inches": [float(fig_w_in), float(fig_h_in)],
        "max_width_delta_inches": float(np.max(physical_widths) - np.min(physical_widths)),
        "max_height_delta_inches": float(np.max(physical_heights) - np.min(physical_heights)),
        "max_width_height_mismatch_inches": float(np.max(np.abs(physical_widths - physical_heights))),
        "column_x0_delta_top_bottom": {
            str(col): float(
                abs(
                    list(boxes.values())[col]["x0"]
                    - list(boxes.values())[3 + col]["x0"]
                )
            )
            for col in range(3)
            if 3 + col < len(boxes)
        },
    }
    png = OUT / f"{OUTPUT_STEM}.png"
    pdf = OUT / f"{OUTPUT_STEM}.pdf"
    cache_npz = OUT / f"{OUTPUT_STEM}_display_cache.npz"
    cache_payload = {
        "truth": np.asarray(truth, dtype=float),
        "axis_uas": np.asarray(axis_uas, dtype=float),
    }
    for strategy, result in result_by.items():
        cache_payload[f"{strategy}_image"] = np.asarray(result["best"]["image"], dtype=float)
    np.savez_compressed(cache_npz, **cache_payload)
    stats["display_cache_npz"] = str(cache_npz)
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    configure_good_runtime()
    val.closure_basis = latest_closure_basis
    case = make_six_station_case()
    # The paired receiver comparison is constructed directly from explicit
    # POVM effects.  No legacy loop-split matrices enter this benchmark.
    splits: dict[str, np.ndarray] = {}
    bands, stats, truth, axis_uas, prior, starts = simulate_good_bands(case, splits)
    if PRECOMPUTE_ONLY:
        print(RECEIVER_CACHE_PATH)
        print(RECEIVER_CACHE_METADATA_PATH)
        return
    morph.configure_runtime(scan=False)
    morph.amp_rml.SOURCE = GOOD_SOURCE
    morph.amp_rml.N_TIME_WINDOWS = N_TIME_WINDOWS_RUN
    morph.amp_rml.EXPOSURE_S = BASE_EXPOSURE_S * EXPOSURE_SCALE
    morph.amp_rml.EXPOSURE_GAP_S = EXPOSURE_GAP_S_RUN
    morph.amp_rml.MODE_FALSE_POSITIVE = EPS_STATION_RUN
    morph.amp_rml.PAIR_FALSE_POSITIVE = EPS_PAIR_RUN
    morph.amp_rml.PRIOR_WEIGHT = GOOD_CFG["prior"]
    morph.amp_rml.TV_WEIGHT = GOOD_CFG["tv"]
    morph.amp_rml.ENTROPY_WEIGHT = GOOD_CFG["entropy"]

    results = []
    for strategy, label, start_name, _color in STRATEGIES:
        print(f"[good-rml] {strategy}", flush=True)
        results.append(run_strategy(strategy, label, start_name, case, bands, truth, axis_uas, prior, starts))

    rows = result_rows(results, truth, axis_uas)
    centers_nm, photon_weights = wavelength_bin_centers_and_weights(GOOD_SOURCE)
    csv_path = OUT / f"{OUTPUT_STEM}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if SKIP_PLOT:
        pdf = None
        png = None
        selected_cache = save_selected_result_cache(stats, truth, axis_uas, results)
    else:
        pdf, png = plot_results(case, stats, truth, axis_uas, results)
        selected_cache = None
    payload = {
        "figure_pdf": None if pdf is None else str(pdf),
        "figure_png": None if png is None else str(png),
        "selected_display_cache_npz": None if selected_cache is None else str(selected_cache),
        "metrics_csv": str(csv_path),
        "source_variant": GOOD_VARIANT.__dict__,
        "prior": GOOD_CFG,
        "stats": stats,
        "rows": rows,
        "component_spectra": component_spectrum_rows(),
        "per_band_rml": PER_BAND_RML,
        "wavelength_bin_centers_nm": [float(value) for value in centers_nm],
        "wavelength_bin_photon_weights": [float(value) for value in photon_weights],
        "band_diagnostics": {
            result["strategy"]: result.get("chi2_attempts", [])
            for result in results
        },
        "note": (
            "Six-station paired-POVM Fig.2 pipeline: NGC 4151 core+disk+BLR H-alpha source, "
            "600-700 nm split into 10 nm approximately monochromatic RML bins, then "
            "photon-weighted stacking for the displayed broadband reconstruction. "
            "Phase information is restricted to the balanced independent 10-loop closure set "
            "{123,124,125,134,136,245,256,346,356,456}. "
            "Amplitude information remains in the complete 15-edge space. The first receiver is "
            "uniform edge-first; the second is the locally optimized single-copy POVM; the third "
            "uses the coherent score-operator fluctuation lift induced by exactly the same POVM. "
            f"For the displayed n_s={COHERENT_BLOCK_SIZE} case, {FINITE_BLOCK_DESCRIPTION} "
            "The single-copy POVM is optimized independently at every wavelength--epoch "
            "sample. The finite-copy correction is calibrated independently for every "
            "wavelength band at its near-transit work point. "
            "Amplitude and phase branches each receive half the photons. "
            "For the discrepancy-controlled comparison, all three receivers use the same "
            "positivity and flux constraints and the common "
            f"(prior, TV, entropy) weights ({RML_PRIOR_WEIGHT_RUN:g}, "
            f"{RML_TV_WEIGHT_RUN:g}, {RML_ENTROPY_WEIGHT_RUN:g}). "
            "No translation-gauge, single-core, compact-structure, or other "
            "morphology-specific penalty is used. "
            "Candidates are selected by the same componentwise discrepancy "
            "criterion targeting amplitude and closure-phase reduced chi-square values "
            "near unity, so the benchmark is not driven by receiver-dependent "
            "regularization, underfitting, or overfitting. "
            "Source and reconstructions "
            f"are shown with a log brightness scale clipped to {DISPLAY_LOG_VMIN:g}-1."
        ),
    }
    json_path = OUT / f"{OUTPUT_STEM}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    if pdf is not None:
        print(pdf)
    if png is not None:
        print(png)
    print(csv_path)
    print(json_path)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
