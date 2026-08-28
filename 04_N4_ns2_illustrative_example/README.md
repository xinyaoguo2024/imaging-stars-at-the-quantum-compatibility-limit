# `N=4`, `n_s=2` Illustrative Collective Receiver

This module is the complete numerical archive for the Appendix worked example.
It contains the complex working point, the 16-outcome Schur PVM, the
overcomplete `n_s=2,3` POVMs, the unit-weight Holevo benchmark, the
nuisance-aware Fisher gain of every edge, and all matrices required to
construct the joint unitary.

## Quick reproduction

```bash
./run_from_archived_data.sh
```

The script uses the saved NPZ/JSON files to:

1. export the complete 16-outcome table at the final complex working point;
2. redraw the outcome atlas, finite-copy gain comparison, and receiver
   schematic;
3. validate probability normalization, PVM/POVM completeness, CFIM
   reconstruction, the two explicit projection vectors quoted in the
   Appendix, and every risk/gain value in the comparison table.

`data/n4_ns2_pvm_outcomes_legacy_stale.csv` is retained explicitly for
provenance. It was generated at an earlier real working point and must not be
used for the paper results. The canonical table is
`data/n4_ns2_pvm_outcomes.csv`.

## Full optimization

```bash
./run_full_optimization.sh
```

The full workflow independently generates the following in
`recomputed_data/`:

- multistart Riemannian optimization of the `n_s=2,3` Schur PVMs;
- optimization of the `n_s=2,3`, `q=2` Parseval-frame POVMs;
- unit-weight Holevo optimization and an SDP cross-check;
- reconstructed figures and validation residuals.

The PVM/POVM searches produce the best validated local optima in the stated
receiver classes; they are not proofs of global optimality over unrestricted
POVMs. The full Holevo calculation requires SciPy and CVXPY. The quick audit
requires only NumPy and Matplotlib.

## Stored data fields

`n4_ns2_phase_pvm_complexwp.npz` stores `rho`, the six tangent matrices, both
Schur bases, `U_+ (10x10)`, `U_- (6x6)`, all 16 probabilities and derivatives,
the CFIM, covariance, QFIM, restart histories, and outcome-block slices. The
two POVM NPZ files additionally store the Parseval frames and the complete
Born statistics for 32 and 88 outcomes. The archive therefore contains all
effects, probabilities, derivatives, scores, Fisher matrices, and validation
quantities promised in the Appendix. The CSV is a human-readable derived
table, not the sole data source.

