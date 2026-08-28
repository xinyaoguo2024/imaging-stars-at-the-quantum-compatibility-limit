# (N=4,5), (n_s=2,3) phase-only joint-PVM scan

This folder contains a like-for-like numerical extension of the earlier
(N=4,n_s=2) construction.  The benchmark is the full-rank
compound-symmetric state

\[
\rho=\frac{(1-g)I+g\mathbf 1\mathbf 1^{\mathsf T}}{N},\qquad g=0.5,
\]

with all (E=N(N-1)/2) edge phases treated as unknown and all visibility
magnitudes held fixed.  The local coordinate is
(h=\sqrt{n_s}(\phi-\phi_0)).  In this coordinate, repeating the same
single-copy uniform-edge-first POVM (n_s) times has the same CFIM as one
copy, so every number below is directly comparable across (n_s).

## Main numbers

| (N) | (n_s) | raw dimension | joint outcomes | repetitive A-risk | joint A-risk | A-risk gain | worst-direction FI gain | mean FI gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 16 | 16 | 144 | 91.39475 | 1.57558 | 1.23398 | 1.63495 |
| 5 | 2 | 25 | 25 | 400 | 244.05414 | 1.63898 | 1.32701 | 1.69248 |
| 4 | 3 | 64 | 44 | 144 | 79.71724 | 1.80638 | 1.49963 | 1.87872 |
| 5 | 3 | 125 | 85 | 400 | 207.11195 | 1.93132 | 1.52062 | 2.01722 |

Here A-risk means \(\operatorname{Tr}J^{-1}\).  Its gain is the repetitive
risk divided by the joint risk, i.e. the harmonic-mean Fisher gain.  The
corresponding square-root gains are (1.2552,1.2802,1.3440,1.3897),
respectively.  The full Fisher spectra are stored in the per-case JSON files.

The largest generalized eigenvalue of the joint CFIM relative to the QFIM is
(0.6920,0.5967,0.7611,0.6738) for the four rows above, so none of the
solutions violates the QFI matrix bound.

## Measurement class and dimensional reduction

The search is over copy-permutation-invariant projective measurements.  For
(n_s=2), the optimization is performed in the symmetric and antisymmetric
blocks.  For (n_s=3), it uses the Schur decomposition

\[
[3]\oplus[2,1]\oplus[1,1,1].
\]

For (N=5,n_s=3), this reduces the raw 125-dimensional problem to reduced
blocks of dimensions (35,40,10).  The mixed block has multiplicity two, so
its physical POVM effects have rank two; the multiplicity label is not read
out.  This is why the number of outcomes is 85 rather than 125.

The copies are distinct temporal registers.  The total optical Fock state is
still bosonic, but the one-photon-per-register sector is isomorphic to
((\mathbb C^N)^{\otimes n_s}); one should therefore not discard the mixed
and antisymmetric copy-permutation sectors.

## Scope of the result

These are best-of-restart local optima of a nonconvex A-optimal PVM search.
They are explicit positive, complete, permutation-invariant joint POVMs, but
they are **not** global-optimality certificates and are **not** constrained to
be the strict score-preserving \(\Pi\)-lift of the one-copy POVM.  Thus the
table answers “how well can this explicit joint-PVM class do at the chosen
working point?”, not “what is the unique receiver guaranteed by the paired
lift theorem?”.

The final gradient norms are approximately (7.6\times10^{-7}),
(3.2\times10^{-5}), (8.8\times10^{-4}), and (2.1\times10^{-3}).  The
(N=5,n_s=3) number is consequently best read to about three significant
digits.  Exact reconstruction checks give completeness residuals below
(1.3\times10^{-14}), copy-permutation commutators below
(2.2\times10^{-15}), and Fisher reconstruction residuals below
(1.7\times10^{-16}).

## Files and reproduction

- `optimize_phase_pvm_scan.py`: model, Schur reduction, Riemannian optimization,
  diagnostics, and export.
- `validate_saved_scan.py`: reconstructs every full POVM effect and independently
  checks completeness, copy-permutation invariance, probabilities, and CFIM.
- `scan_summary.md` / `scan_summary.csv`: compact comparison tables.
- `n*_ns*_phase_pvm_summary.json`: complete scalar diagnostics and spectra.
- `n*_ns*_phase_pvm.npz`: optimized reduced unitaries, Schur bases, reduced
  states/tangents, probabilities, derivatives, CFIM, and covariance.
- `validation_summary.json`: independent full-space validation.

Run all cases from scratch with

```bash
python3 optimize_phase_pvm_scan.py --cases 4x2,5x2,4x3,5x3 --restarts 20 --steps 2200
```

Continue a saved best solution with

```bash
python3 optimize_phase_pvm_scan.py --cases 5x3 --steps 25000 --refine-existing
```

and validate the saved POVMs with

```bash
python3 validate_saved_scan.py
```
