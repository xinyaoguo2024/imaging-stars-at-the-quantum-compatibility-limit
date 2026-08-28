# Strict Physical Near With Pair-Combine Taps

This folder contains the strict physical near receiver pipeline requested on 2026-06-13.

## Variables

- `p_ij`: directed station-side split matrix.  It is not tied across stations.
- `alpha_i`: independent compact-core joint-receiver fraction for core stations `S1-S3`.
- `gamma`: 18 directed remote-star fractions, split into core-to-remote and remote-to-core components.
- `q`: pair-combine taps.  For each balanced-loop orientation `(a+b)|c`, part of the light from stations `a`, `b`, and `c` is sent to a receiver that coherently combines `a` and `b` and beats the resulting mode with `c`.

All sinks share the same row-wise station photon budget.  For each station, `alpha + sum(p) + sum(q) = 1` up to floating-point precision.

## Scripts

```bash
python3 27_strict_physical_near_paircombine_20260613/code/run_optimize_paircombine.py --activation-probes 10 --passes 0
```

The command above is the verified smoke/diagnostic run.  A fuller coordinate refinement can be launched with:

```bash
python3 27_strict_physical_near_paircombine_20260613/code/run_optimize_paircombine.py --activation-probes 30 --passes 1
```

## Current Verified Run

The verified run reproduced the folder-18 `current_near` and `remote_star_independent` columns exactly before optimizing.  With `--activation-probes 10 --passes 0`, the best state remained the saved independent-gamma remote-star start; no activated pair-combine tap improved the objective in that short scan.

The diagnostic figure is:

```text
27_strict_physical_near_paircombine_20260613/figures/paircombine_strict_near_loop_gain_diagnostic.png
```

## Corrected Worst-Ratio Objective

On 2026-06-13, the objective was corrected so that the optimization is not a
simple mean over loops.  The new `worst-ratio` objective is dominated by

```text
max_l |log(SNR_near,l / SNR_direct,l)|
```

with small RMS, variance, and bias terms used only as tie-breakers.  This makes
the optimizer reduce the worst per-loop near/direct mismatch instead of hiding
one bad loop inside a good average.

The implementation also caches the remote-star Fisher matrix for fixed
`p,gamma`, which avoids recomputing the same remote-star block during scans over
pair-combine `q,beta,delta` coordinates.

The current corrected-objective run was generated with:

```bash
python3 27_strict_physical_near_paircombine_20260613/code/run_optimize_paircombine.py --objective worst-ratio --activation-probes 0 --passes 0
```

using the best saved checkpoint from the interrupted refinement.  The budget
constraint is satisfied to `8.9e-16` maximum absolute station-budget error.
The resulting near/direct SNR ratios over the ten balanced loops are:

```text
min/mean/max/std = 0.987810 / 0.995026 / 1.005251 / 0.005772
max |log ratio| = 0.012265
```

Main files:

```text
27_strict_physical_near_paircombine_20260613/results/paircombine_strict_near_loop_gains_worst_ratio.csv
27_strict_physical_near_paircombine_20260613/results/paircombine_strict_near_summary_worst_ratio.json
27_strict_physical_near_paircombine_20260613/results/paircombine_modules_worst_ratio.csv
27_strict_physical_near_paircombine_20260613/results/checkpoint_worst-ratio.json
27_strict_physical_near_paircombine_20260613/figures/paircombine_strict_near_loop_gain_diagnostic_worst_ratio.png
```
