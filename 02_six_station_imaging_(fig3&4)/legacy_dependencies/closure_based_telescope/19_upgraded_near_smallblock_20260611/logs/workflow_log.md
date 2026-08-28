# Workflow Log

## 2026-06-11 upgraded small-block near attempt

- Created `19_upgraded_near_smallblock_20260611` as a separate benchmark folder.
- Tested a Fisher-level upgraded near strategy motivated by the user's proposal: keep only local small receiver blocks and forbid the full six-mode joint receiver.
- Allowed receiver blocks are all three-station triangles containing at least one compact-core station.  This allows `{core, core, remote}` and `{core, remote, remote}` modules, while forbidding the pure remote `{S4,S5,S6}` block.
- Optimized the allowed block weights under exact station-side constraints `sum_{block contains station i} w_block = 1`.
- Used the balanced 10-loop selected closure set from folder 18 and optimized the small-block schedule to match the folder-18 direct optimized schedule while enforcing selected-loop gain at or above edge-first.
- Result: upgraded small-block near has SNR gain min/mean/max versus edge-first `1.1893 / 1.2543 / 1.3848`.
- Relative to direct optimized, upgraded small-block near has SNR ratio min/mean/max `0.9745 / 0.9912 / 1.0071`.
- This strongly suggests that the old remote-star near failed on two-remote loops because it lacked `{core, remote, remote}` small blocks, not because two-remote closure gains are intrinsically unavailable.
