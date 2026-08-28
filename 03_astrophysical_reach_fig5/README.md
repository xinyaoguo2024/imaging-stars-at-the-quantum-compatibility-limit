# Astrophysical Reach Estimate (Main-Text Fig. 5)

This module contains the analytic estimate used in main-text Fig. 5, the
fixed-`|g|=0.5` random-phase Monte Carlo, the Fisher-gain distribution of a
representative closure loop, and the source catalog based on literature black
hole mass estimates.

## Quick reproduction

```bash
./run_from_archived_data.sh
```

The command redraws the current main-text Fig. 5 from
`data/fixed_modulus_phase_gain_summary.json` and validates the PNG, the key
parameters, and the raw Monte Carlo arrays. The PNG should be byte-identical to
the submission version in `reference_outputs/`. The canonical outputs are
`generated_outputs/fig5_astrophysical_reach.pdf` and
`generated_outputs/fig5_astrophysical_reach.png`.

## Full Monte Carlo rerun

```bash
./run_full_monte_carlo.sh
```

By default, the script generates 600 positive-semidefinite random-phase
coherence matrices for each of `N=6` and `N=20`. It uses the S1--S2--S3 loop,
which is fixed before sampling, and evaluates the collective-to-single-copy
Fisher gain over AB magnitudes 10--22. New results are written to
`recomputed_data/` and `recomputed_outputs/`; the archived data are not
overwritten.

## Principal inputs

- `T Delta f = 10^11`, `N=20`, `eta=0.2`, and background occupation `10^-11`.
- Every off-diagonal coherence satisfies exactly `|g_ij|=0.5`.
- The mass estimates, uncertainties, and source-selection references are
  documented in `MASS_REVIEW.md` and
  `data/fig5_astrophysical_reach.json`.
- `data/fixed_modulus_phase_gain_samples.npz` stores every sampled gain,
  closure phase, minimum eigenvalue, and magnitude-grid entry rather than only
  ensemble averages.
