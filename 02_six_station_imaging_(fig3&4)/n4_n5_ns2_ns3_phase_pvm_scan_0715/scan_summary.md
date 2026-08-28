# N=4,5 and n_s=2,3 phase-PVM scan

Benchmark: compound-symmetric full-rank state with g=0.5; all edge phases are estimated; local coordinates are h=sqrt(n_s)(phi-phi0).

| N | n_s | raw dim | outcomes | repetitive A-risk | joint A-risk | A-risk gain | min FI gain | mean FI gain | max FI gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 16 | 16 | 144 | 91.3948 | 1.57558 | 1.23398 | 1.63495 | 1.9658 |
| 5 | 2 | 25 | 25 | 400 | 244.054 | 1.63898 | 1.32701 | 1.69248 | 2.11635 |
| 4 | 3 | 64 | 44 | 144 | 79.7172 | 1.80638 | 1.49963 | 1.87872 | 2.27527 |
| 5 | 3 | 125 | 85 | 400 | 207.112 | 1.93132 | 1.52062 | 2.01722 | 2.39931 |

These are best-of-restart local optima of a nonconvex A-optimal projective-measurement search, not global optimality certificates.
For n_s=3, the mixed Schur effect has rank two and its representation multiplicity is not resolved.
