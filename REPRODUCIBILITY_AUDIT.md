# Reproducibility audit

Audit date: 2026-08-28.

## Scope checked

- Located every figure included by the current main manuscript and Appendix.
- Copied the canonical plotting/simulation/optimization scripts and all direct local
  dependencies into four self-contained modules.
- Replaced workstation-specific paths in executable Python/shell sources with paths
  derived from each module root.
- Preserved canonical PDF/PNG outputs separately from regenerated outputs.
- Added quick rerun and full rerun entry points without overwriting archived raw data.

## Passed checks

- Principle schematic and implementation diagram: regenerated PNGs are byte-identical
  to the manuscript versions.
- Six-station imaging: representative seed is 20260536; regenerated current
  main-text Fig. 3 and Fig. 4 PNGs are byte-identical to the manuscript
  versions. The current figures use a normalized-brightness cutoff and outer
  contour at 0.015, and the Fig. 4 mean/SEM error bars are drawn above the
  paired-seed points. Superseded wide-layout manuscript figures and historical
  dependency previews are excluded from the deliverable; the module audit
  fails if any PDF/PNG/SVG other than the current Fig. 3/4 generated and
  reference copies is present.
- Twelve paired seeds 20260529--20260540 are present. Each has ten-band pre-RML raw
  measurements and four RML deliverables (summary, metrics, display cache, H-alpha audit).
- The raw measurement NPZs include all three receivers, full amplitude records,
  closure-only phase records, and the full mixed covariance—not only final images.
- The astrophysical-reach figure (Fig. 5 in the current manuscript): regenerated
  PNG is byte-identical. Raw Monte Carlo arrays contain 600 accepted
  draws for both N=6 and N=20, with a fixed typical loop and all `|g_ij|=0.5`.
- N=4 example: all 16/32/88 outcome statistics, Schur bases, unitaries/frames, tangents,
  CFIMs, QFIMs, covariance matrices, optimization histories and residuals are present.
  The two explicit port vectors/probabilities and all Appendix comparison values agree.

## Corrected provenance issue

An old `n4_ns2_pvm_outcomes.csv` was generated at an earlier real working point and did
not match the final complex-workpoint NPZ. It is retained as
`n4_ns2_pvm_outcomes_legacy_stale.csv`; the canonical CSV is regenerated directly from
`n4_ns2_phase_pvm_complexwp.npz` and contains all six score components.

## Numerical caveats

- PDF hashes can differ only because Matplotlib embeds creation timestamps; PNGs are used
  for strict rendering comparison.
- Finite-copy PVM/POVM searches are nonconvex local searches within the explicitly stated
  Schur-projective/Parseval-frame classes. Their saved residuals certify feasibility and
  reproduce the quoted performance, not global unrestricted-POVM optimality.
- A complete rerun of the unit-weight Holevo optimization requires SciPy and CVXPY.
  The archived solver cross-checks and residuals are included even when those optional
  packages are not installed.
