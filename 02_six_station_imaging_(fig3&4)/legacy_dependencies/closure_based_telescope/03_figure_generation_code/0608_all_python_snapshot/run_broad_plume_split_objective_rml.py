from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CORE_DIR = THIS_DIR.parent / "core"
BUNDLE = THIS_DIR.parents[1]
for _path in (CORE_DIR, THIS_DIR):
    _path_text = str(_path)
    if _path_text in sys.path:
        sys.path.remove(_path_text)
    sys.path.insert(0, _path_text)

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
import run_hawaii3_rml_strategy_comparison as strategy_run
import run_rml_validation_pipeline as val
import test_fig3_split_objective_imaging as split_sim
import core4_joint_remote_split_design as core4_remote


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
MORPH_DIR = ROOT / "rml_remote3_source_morphology_optimization_20260525"
if str(MORPH_DIR) not in sys.path:
    sys.path.insert(0, str(MORPH_DIR))
import optimize_remote3_source_morphology_priors as morph  # noqa: E402


OUT = BUNDLE / "rml_remote3_broad_plume_split_objective_20260608_photonlimited"
OUT.mkdir(parents=True, exist_ok=True)


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
REMOTE_DIAMETER_M = float(os.environ.get("FIG2_REMOTE_DIAMETER_M", "2.0"))
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
RML_PHASE_CHI2_SELECTION = os.environ.get("RML_PHASE_CHI2_SELECTION", "minimize").strip().lower()
RML_PHASE_CHI2_TARGET = float(os.environ.get("RML_PHASE_CHI2_TARGET", "1.0"))
RML_PHASE_CHI2_MIN = float(os.environ.get("RML_PHASE_CHI2_MIN", "0.90"))
RML_PHASE_CHI2_MAX = float(os.environ.get("RML_PHASE_CHI2_MAX", "1.05"))
PER_BAND_RML = os.environ.get("FIG2_PER_BAND_RML", "1").strip().lower() not in {"0", "false", "no"}
SKIP_PLOT = os.environ.get("FIG2_SKIP_PLOT", "0").strip().lower() in {"1", "true", "yes"}
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
    f"_dir{occupancy_tag(EPS_DIRECT_EXTRA_RUN)}_{DRIFT_TAG}"
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

STRATEGIES = [
    ("all", "All visibilities + drift", "all_dirty", "#7b2cbf"),
    ("edge_uniform", "Edge-first closure", "edge_uniform_dirty", "#0077b6"),
    ("core4_remote_optimized", "Core4+remote optimized closure", "core4_remote_optimized_dirty", "#f77f00"),
    ("nmode_joint_scheduled", "Direct Fisher closure-space benchmark", "nmode_joint_scheduled_dirty", "#d00000"),
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
                "all",
                "edge_uniform",
                "core4_remote_optimized",
                "nmode_joint_scheduled",
            )
        )
        raise ValueError(f"FIG2_STRATEGY_FILTER selected no strategies; known: {known}")


def observing_night_label(nights: int) -> str:
    suffix = "" if nights == 1 else "s"
    return f"{nights} observing night{suffix}"


def configure_good_runtime() -> None:
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


def make_split_matrices(case: aug.NetworkCase) -> dict[str, np.ndarray]:
    """Build strict V1 close4 phase-frame plus remote loop-wise schedule."""
    global SPLIT_DIAGNOSTICS
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
            loop_specs, schedule_info = core4_remote.optimize_strict_v1_schedule_weights(bm)
            close_factor = float(schedule_info["close4_phase_frame_schedule_weight"])
            near_fisher = core4_remote.fisher_for_loop_specs(
                bm,
                loop_specs,
                close_factor=close_factor,
                apply_spec_schedule_weights=True,
            )
            near_info = {
                "objective": "strict_v1_loopwise_schedule_ratio_balancing",
                "description": (
                    "Close4 stations use the shared phase-frame direct+nuisance receiver. "
                    "Every root loop involving a remote station uses an edge-first readout "
                    "with loop-internal optimized splitting and optimized schedule weight."
                ),
                "metrics": closure_bm.stable_metrics(near_fisher),
                "schedule_optimization": schedule_info,
            }
            direct_weights, direct_info = core4_remote.uniform_direct_root_weights(
                bm,
                mode="capacity_relaxed_scalar",
            )
            split_sim.set_modular_loop_specs(loop_specs, schedule_factor=1.0, close_factor=close_factor)
            split_sim.set_direct_root_weights(
                direct_weights,
                model="capacity_relaxed_scalar_root_closure_weights",
            )
            budget = schedule_info["photon_budget_diagnostics"]
            profile_split = np.zeros((bm.n, bm.n), dtype=float)
            name_to_index = {name: idx for idx, name in enumerate(bm.names)}
            for station_row in budget["station_rows"]:
                station = name_to_index[str(station_row["station"])]
                branch_budget = float(station_row["remote_branch_budget"])
                for target_name, fraction in station_row["active_normalized_profile"].items():
                    profile_split[station, name_to_index[str(target_name)]] = branch_budget * float(fraction)
            SPLIT_DIAGNOSTICS = {
                "near_strategy": "strict V1 close4 phase-frame direct+nuisance plus remote loop-wise edge-first",
                "near_split_objective": near_info,
                "loop_specs": loop_specs,
                "photon_budget_diagnostics": budget,
                "schedule_averaged_split_profile": budget["split_profile_rows"],
                "direct_strategy": "capacity-relaxed scalar root-closure direct schedule",
                "direct_weight_info": direct_info,
                "direct_root_weights": {
                    f"S{tri[0] + 1}-S{tri[1] + 1}-S{tri[2] + 1}": float(weight)
                    for tri, weight in sorted(direct_weights.items())
                },
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


def simulate_good_bands(case: aug.NetworkCase, splits: dict[str, np.ndarray]):
    split_sim.configure()
    apply_sample_stress_runtime()
    # ``patched_variant`` changes ngc.make_source_factory, while
    # ``patched_source`` actually installs the resulting factory into
    # plot_prl_broadband_clean.base.make_source.  Both are required here
    # because split_sim builds the truth image by calling base.make_source
    # directly rather than going through amp_rml.simulate_case.
    with morph.patched_variant(GOOD_VARIANT), ngc.patched_source(GOOD_SOURCE):
        bands, stats, truth, axis_uas = split_sim.simulate_bands_with_strategies(case, splits)
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
        "rng_seed": FIG2_RNG_SEED,
        "rml_fit_n_pix": val.FIT_N_PIX,
        "rml_adam_iter": val.ADAM_ITER,
        "rml_adam_lr": val.ADAM_LR,
        "rml_prior_weight": RML_PRIOR_WEIGHT_RUN,
        "rml_tv_weight": RML_TV_WEIGHT_RUN,
        "rml_entropy_weight": RML_ENTROPY_WEIGHT_RUN,
        "per_band_rml": PER_BAND_RML,
        "wavelength_bin_width_nm": LAMBDA_STEP_NM_RUN,
        "display_stretch": "arcsinh",
        "display_asinh_q": DISPLAY_ASINH_Q,
        "legacy_display_log_vmin": DISPLAY_LOG_VMIN,
        "rml_phase_chi2_selection": RML_PHASE_CHI2_SELECTION,
        "rml_phase_chi2_target": RML_PHASE_CHI2_TARGET,
        "rml_phase_chi2_min": RML_PHASE_CHI2_MIN,
        "rml_phase_chi2_max": RML_PHASE_CHI2_MAX,
        "split_diagnostics": SPLIT_DIAGNOSTICS,
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
    return starts


def chi2_target_score(candidate: dict) -> float:
    residuals = candidate["residuals"]
    amp_chi2 = max(float(residuals["amp_reduced_chi2"]), 1e-300)
    phase_chi2 = max(float(residuals["phase_reduced_chi2"]), 1e-300)
    amp_score = abs(np.log(amp_chi2))
    high_chi_penalty = max(0.0, np.log(amp_chi2 / max(RML_CHI2_THRESHOLD, 1e-300)))
    if RML_PHASE_CHI2_SELECTION == "minimize":
        phase_score = np.log(phase_chi2)
        return float(phase_score + 0.10 * amp_score + 5.0 * high_chi_penalty + 0.001 * candidate["objective"])
    phase_score = abs(np.log(phase_chi2 / max(RML_PHASE_CHI2_TARGET, 1e-300)))
    return float(phase_score + 0.30 * amp_score + 2.0 * high_chi_penalty + 0.001 * candidate["objective"])


def chi2_target_passed(candidate: dict) -> bool:
    residuals = candidate["residuals"]
    amp_chi2 = float(residuals["amp_reduced_chi2"])
    phase_chi2 = float(residuals["phase_reduced_chi2"])
    if RML_PHASE_CHI2_SELECTION == "minimize":
        return np.isfinite(phase_chi2) and amp_chi2 < RML_CHI2_THRESHOLD
    if candidate.get("strategy") == "all":
        return amp_chi2 < RML_CHI2_THRESHOLD and 0.75 <= phase_chi2 <= 1.35
    return (
        amp_chi2 < RML_CHI2_THRESHOLD
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
        for name in (start_name, "prior"):
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
        band_starts = make_single_band_starts(band, strategy, start_name, prior, truth_band)
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
    if PER_BAND_RML:
        return run_strategy_per_band(strategy, label, start_name, case, bands, truth, axis_uas, prior)
    return run_strategy_for_bands(strategy, label, start_name, case, bands, truth, axis_uas, prior, starts)


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


def asinh_tick(value: float) -> float:
    q = max(DISPLAY_ASINH_Q, 1e-6)
    return float(np.arcsinh(float(value) / q) / np.arcsinh(1.0 / q))


def plot_component_spectra(ax) -> None:
    rows = component_spectrum_rows()
    wavelengths = np.asarray([row["wavelength_nm"] for row in rows])
    style_map = [
        ("total_mjy", "total", "black", 1.55, "-"),
        ("blr_mjy", "BLR", "#d00000", 1.60, "-"),
        ("core_mjy", "core", "#005f73", 1.05, "-"),
        ("disk_mjy", "disk", "#4361ee", 1.05, "--"),
        ("continuum_mjy", "cont.", "#4361ee", 1.05, "--"),
        ("plume_mjy", "plume", "#f77f00", 1.05, "-."),
        ("halo_mjy", "halo", "#6a994e", 1.05, ":"),
    ]
    for key, label, color, lw, ls in style_map:
        if key not in rows[0]:
            continue
        values = np.asarray([row[key] for row in rows])
        ax.plot(wavelengths, values, color=color, lw=lw, ls=ls, label=label)
    line_marks = [
        (486.1 * (1.0 + NGC4151_REDSHIFT), r"H$\beta$"),
        (656.3 * (1.0 + NGC4151_REDSHIFT), r"H$\alpha$"),
    ]
    y_top = max(row["total_mjy"] for row in rows)
    for wavelength_nm, label in line_marks:
        if wavelength_nm < LAMBDA_MIN_NM_RUN or wavelength_nm > LAMBDA_MAX_NM_RUN:
            continue
        ax.axvline(wavelength_nm, color="#d00000", lw=0.65, alpha=0.55)
        ax.annotate(
            f"{label}\n{wavelength_nm:.1f} nm",
            xy=(wavelength_nm, 0.66 * y_top),
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
    ax.set_ylabel(r"$F_\nu$ (mJy)")
    ax.set_title("Component-resolved\nnuclear SED")
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


def plot_results(case, stats, truth, axis_uas, results: list[dict]) -> tuple[Path, Path]:
    axis = axis_uas
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    result_by = {item["strategy"]: item for item in results}
    stations, diameters, _names, is_added = aug.station_table_from_case(case)
    plt.rcParams.update(
        {
            "font.size": 6.9,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.5,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
        }
    )
    fig = plt.figure(figsize=(10.15, 5.40), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.34)
    fig.subplots_adjust(left=0.060, right=0.925, bottom=0.095, top=0.930)
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
        ax.text(x + dx, y + dy, f"S{i + 1}  {diameters[i]:g}m", fontsize=5.1)
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

    inset = ax.inset_axes([0.62, 0.50, 0.34, 0.34])
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
        0: (-0.022, -0.088),
        1: (0.020, 0.064),
        2: (-0.112, 0.012),
        3: (0.046, -0.052),
    }
    for i, (x, y) in enumerate(stations[:4]):
        dx, dy = inset_offsets[i]
        inset.text(
            x + dx,
            y + dy,
            f"S{i + 1}",
            fontsize=4.55,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.15},
        )
    inset.text(
        case.hub_km[0] - 0.090,
        case.hub_km[1] - 0.130,
        "hub",
        fontsize=4.15,
        color="#7a3b00",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.12},
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
    image_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    ax.imshow(asinh_display(truth), origin="lower", extent=extent, cmap="inferno", norm=image_norm)
    ax.set_title("Input source\ncore + disk + BLR")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    ax.set_box_aspect(1)
    image_axes.append(ax)

    ax = fig.add_subplot(gs[0, 3])
    panel_axes["component_spectra"] = ax
    plot_component_spectra(ax)

    bottom_labels = {
        "all": "All visibilities + drift",
        "edge_uniform": "Edge-first closure\nuniform split",
        "core4_remote_optimized": "Approx. optimal closure\nstrict V1 loop-wise",
        "nmode_joint_scheduled": "Direct Fisher benchmark\ncapacity-relaxed",
    }
    for col, strategy in enumerate(("all", "edge_uniform", "core4_remote_optimized", "nmode_joint_scheduled")):
        result = result_by[strategy]
        ax = fig.add_subplot(gs[1, col])
        panel_axes[strategy] = ax
        ax.imshow(
            asinh_display(result["best"]["image"]),
            origin="lower",
            extent=extent,
            cmap="inferno",
            norm=image_norm,
        )
        m = result["best"]["metrics"]
        ax.set_title(
            f"{bottom_labels[strategy]}\nBLR r={m['blr_corr']:.3f}, all r={m['global_corr']:.3f}",
            fontsize=7.0,
        )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        ax.set_box_aspect(1)
        image_axes.append(ax)

    cax = fig.add_axes([0.942, 0.125, 0.012, 0.685])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=image_norm, cmap="inferno"),
        cax=cax,
    )
    cbar.set_label("norm. brightness\n(arcsinh stretch)", fontsize=6.8)
    tick_values = [0.0, 0.03, 0.1, 0.3, 1.0]
    cbar.set_ticks([asinh_tick(value) for value in tick_values])
    cbar.set_ticklabels(["0", "0.03", "0.1", "0.3", "1"])

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
                    - list(boxes.values())[4 + col]["x0"]
                )
            )
            for col in range(4)
        },
    }
    png = OUT / f"{OUTPUT_STEM}.png"
    pdf = OUT / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    configure_good_runtime()
    case = scale_remote_coordinates(hawaii3_compact_case.make_hawaii3_compact_remote_case())
    splits = make_split_matrices(case)
    bands, stats, truth, axis_uas, prior, starts = simulate_good_bands(case, splits)
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
    else:
        pdf, png = plot_results(case, stats, truth, axis_uas, results)
    payload = {
        "figure_pdf": None if pdf is None else str(pdf),
        "figure_png": None if png is None else str(png),
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
            "Final-candidate Fig.2 pipeline: NGC 4151 core+disk+BLR H-alpha source, "
            "600-700 nm split into 10 nm approximately monochromatic RML bins, then "
            "photon-weighted stacking for the displayed broadband reconstruction. "
            "The displayed near-optimal column uses the strict V1 close4 direct-closure+nuisance "
            "receiver with remote-involving root closures implemented by loop-wise optimized "
            "edge-first readout, and "
            "the direct Fisher column uses a capacity-relaxed scalar root-closure schedule "
            "with uniform w_l=(N-1)/C; "
            "per-band RML candidates are selected by minimizing the phase reduced chi-square "
            "under an amplitude chi-square sanity guard when RML_PHASE_CHI2_SELECTION=minimize; "
            "raw-QFI remains only an upper-bound diagnostic. Source and reconstructions "
            "are shown with an arcsinh brightness stretch."
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
