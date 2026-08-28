# Workflow Log

## 2026-06-11 remote-star coherent near attempt

- Started from the six-station old-S2--old-S7 benchmark in `16_six_station_reduced_from7_20260611`.
- Confirmed that the existing near model treats remote-involved loops as compact-core joint Fisher plus pairwise edge-first core-remote/remote-remote beats under an optimized station-side split.
- Implemented a remote-star joint receiver: for each remote station, the three core beams assigned to that remote and the remote beam assigned back to the core enter one local four-mode phase-frame receiver.
- Found and fixed an evaluation-context issue in the new benchmark: Fisher recomputation must run inside the same `GOOD_SOURCE`/spectral-variant patch context as the original optimizer, otherwise wavelength-dependent visibilities fall back to a different source state.
- Tested the unrestricted full remote-star receiver. It strongly exceeds the physical direct split target because it also measures extra core-core phase information.
- Added the reported remote-only variant: local core-core phases inside each remote-star receiver are Schur-complemented as nuisance parameters before embedding the block, so compact-core phase information is not double-counted.
- Scanned scalar gamma from 0 to 1. The selected value is gamma = 0.10, where 10% of each core-remote directed split enters the remote-star receiver and 90% remains pairwise edge-first.
- Outputs: `results/remote_star_joint_near_summary.json`, `results/remote_star_joint_loop_gains.csv`, `results/remote_star_gamma_scan.csv`, `figures/remote_star_joint_loop_gains.png`, and `notes/remote_star_joint_near_note.md`.

## 2026-06-11 independent gamma and optimized direct schedule

- Replaced the scalar gamma-only comparison with an independent directed gamma search.  Each core-to-remote and remote-to-core directed split can choose its own fraction for the remote-star receiver, with the residual kept in pairwise edge-first channels.
- Independent gamma improves the mean root-loop SNR gain versus edge-first from 1.0947 to 1.1059 and improves mean near/direct SNR ratio from 0.9519 to 0.9614.
- Added an optimized all-triangle direct schedule.  It optimizes triangle weights under exact constraints `sum_{tri contains i} w_tri = 1` for every station, so every station's received photon budget is fully used.
- The optimized direct schedule improves the worst root-loop gain versus edge-first from 1.0798 to 1.1127, while the mean gain changes only from 1.1513 to 1.1571.  For the representative loops, S1-S2-S3 improves from 1.0853 to 1.1263 and S1-S2-S4 improves from 1.0798 to 1.1130.
- This confirms that the small direct/edge gain for the compact strong loops is partly a uniform-schedule dilution effect, but not entirely: edge-first is already close to efficient on those strong, balanced loops.
- Added a horizontal bookkeeping benchmark at `sqrt(15/10)=1.2247` to mark the ideal SNR factor one would associate with not spending readout degrees of freedom on the five station-piston/gauge nuisance modes in a six-station phase-frame model.  This is a reference line, not a separately implemented physical receiver.

## 2026-06-11 balanced 10-loop independent-set rerun

- User clarified that the intended benchmark is not an isotropic closure-subspace basis, but a ten-dimensional independent triangle-closure matrix with equal photon budget assigned to each selected loop.
- Created `18_balanced_10loop_independent_set_20260611` as a new result folder copied from the remote-star workflow and changed its root output paths.
- Replaced the default root-loop plotting/evaluation basis with the balanced independent loop set `123, 124, 125, 134, 136, 245, 256, 346, 356, 456`.
- This set has rank 10 and every station appears in five loops; assigning direct weight 0.2 to each selected loop therefore gives station-side budget sum exactly 1 at every station.
- Added explicit projection from q-basis Fisher matrices to the selected triangle closures, so all plotted loop RMS values are computed as `d_l^T Cov_q d_l` for the requested loop vector `d_l`.
- Replaced the near split objective with a balanced-10 target: optimize station split and independent core alphas to match the equal-budget `direct_balanced_10loop` target while treating any selected loop with `SNR_near/SNR_edge < 1` as infeasible.
- Updated scalar and independent remote-star gamma searches to use the same balanced 10-loop direct target.
- After the first balanced run, tightened the gamma search as well: any scalar or independent gamma candidate with a selected-loop gain below edge-first is now infeasible, not merely penalized.
- Refreshed the gamma CSV, summary JSON, note, and plot from the saved balanced near split payload.  The strict independent-gamma result has minimum selected-loop gain 1.000061 versus edge-first and mean gain 1.157274.
