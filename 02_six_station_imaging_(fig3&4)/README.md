# Six-Station Imaging and 12-Seed Statistics (Main-Text Figs. 3 and 4)

This is the most computationally intensive module in the archive. It contains
the results needed for immediate redrawing, the complete 12-seed RML outputs,
and the code required to regenerate the simulated measurements and image
reconstructions from the beginning of the pipeline.

## Directory contents

- `raw_measurements/`: pre-RML measurement records for twelve paired seeds.
  Each NPZ contains the ten-band `(u,v)` coordinates, true amplitudes and
  closure phases, and the noisy amplitudes, closure phases, visibilities,
  standard deviations, and complete mixed covariances for all three receivers.
- `results_12seed_raw/100ms/`: the RML summary, strategy metrics, selected
  display cache, and H-alpha-band audit for every seed.
- `finite_ns3_povm_q2_band*.npz/json`: independently calibrated finite-`n_s=3`
  receivers for each 10-nm wavelength band. The same band calibrations are used
  for all seeds.
- `receiver_cache_*.npz/json`: per-band, per-epoch receiver covariance caches.
- `plotting_data/`: the compact paired-seed correlation and per-loop gain
  tables consumed by the current Fig. 4 plotting routine.
- `results/`: the complete 12-seed summary CSV/JSON files, chi-square table,
  and representative-seed selection record.
- `code/make_maintext_fig3_fig4.py`: the active, self-contained plotting entry
  point for the current single-column Figs. 3 and 4.
- `generated_outputs/`: figures produced by the active plotting entry point.
- `reference_outputs/`: the current manuscript PDFs/PNGs used for strict
  rendering validation. Only the current Fig. 3 and Fig. 4 layouts are kept;
  superseded manuscript figures have been removed from this module.
- `legacy_dependencies/`: the physical-model and RML modules extracted from
  the original project that are required for a full rerun. The active entry
  points use relative paths and do not depend on the original workstation
  directories.

## Quick reproduction from archived RML results

```bash
./run_from_archived_data.sh
```

This command redraws both current main-text figures directly from the archived
representative reconstruction and paired 12-seed statistics, then runs the
archive audit. It produces
`generated_outputs/fig3_fourpanel_singlecolumn.pdf` and
`generated_outputs/fig4_statistics_singlecolumn.pdf`. In the validated
software environment, both PNG files are byte-identical to their counterparts
in `reference_outputs/`.

## Full rerun

```bash
./run_full_pipeline.sh
```

The full workflow generates twelve paired noise realizations, performs the
40-by-40-pixel RML reconstruction independently in every wavelength band,
selects the representative seed, regenerates the numerical summaries, and
renders only the current Fig. 3 and Fig. 4 layouts.
It runs three seeds in parallel by default and writes new results to
`recomputed_12seed_runs/`, leaving the archived results unchanged. This step is
computationally expensive, and last-digit differences may occur across BLAS or
Matplotlib versions.

The canonical current-layout figures are intentionally redrawn by
`run_from_archived_data.sh`, so that the submission artifacts remain tied to
the audited paired ensemble rather than to environment-dependent reruns of the
nonconvex RML optimization.

The `legacy_dependencies/` tree retains source modules and numerical assets
needed by the simulation, but excludes their historical PDF/PNG previews. This
keeps the archive computationally complete without mixing obsolete manuscript
figures with the two canonical outputs.

The full workflow deliberately reuses the archived per-band receiver cache
across seeds. To redesign the finite-copy receivers themselves, first run
`optimize_finite_ns_povm_per_band.py` and
`build_per_band_finite_cache.py`; both programs expose their options through
`--help`.

To regenerate only the pre-RML measurement records, use:

```bash
python3 export_raw_measurements.py --help
```

## Fixed settings used in the paper

- Six stations: 100% coupled area for the three central stations and three
  remote telescopes of diameter 6 m.
- Photon-collection efficiency 0.02, followed by the fiber attenuation applied
  in the simulation code.
- 600--700 nm in ten 10-nm bins and 36 Earth-rotation epochs.
- 100 ms per sample, with equal photon budgets for amplitude and phase.
- Amplitude information in the complete `E=15` edge space and phase
  information in the `C=10` closure space.
- Finite collective block size `n_s=3`, calibrated independently by band and
  reused across seeds.
- RML without a translation gauge or morphology-specific core prior, using a
  common Gaussian/TV/entropy regularization and a receiver-blind chi-square
  selection rule.

The BLR correlation mask is matched to the source model:
`r_BLR +/- max(2.2 sigma_BLR, 10 microarcsec)`. It spans
45.6--98.4 microarcsec in this example.
