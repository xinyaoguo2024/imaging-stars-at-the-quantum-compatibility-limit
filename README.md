# Paper Figure and Data Reproducibility Archive

This archive accompanies the latest PRX-format manuscript,
*Imaging Stars at the Quantum Compatibility Limit*.
Each of the four modules contains an independent execution entry point, source
code, raw or intermediate data, canonical manuscript outputs, and numerical
validation. No module depends on the historical project folders outside this
archive.

## Directory and figure map

1. `01_principle_and_implementation_schematics_(fig1&2)/`
   - Main-text Fig. 1: array/VCZ imaging and the single-copy--collective
     readout concept;
   - Main-text Fig. 2: memory-assisted implementation block diagram.
2. `02_six_station_imaging_(fig3&4)/`
   - Main-text Fig. 3: array geometry, Fourier coverage, input source, and the
     representative finite-copy collective reconstruction;
   - Main-text Fig. 4: paired 12-seed image-correlation and per-loop
     closure-phase SNR-gain statistics;
   - pre-RML measurement arrays and complete selected RML outputs for all
     twelve seeds.
3. `03_astrophysical_reach_fig5/`
   - Main-text Fig. 5: Schwarzschild-scale astrophysical reach;
   - raw fixed-`|g|` random-phase Monte Carlo samples for `N=6,20` and the
     source catalog.
4. `04_N4_ns2_illustrative_example/`
   - Appendix worked example for `N=4,n_s=2`;
   - the 16-outcome PVM atlas, `n_s=2,3` POVMs, the Holevo limit, and complete
     matrix data.

## One-command quick reproduction

```bash
python3 -m pip install -r requirements.txt
./run_all_quick.sh
```

The quick workflow redraws all submission figures from the archived numerical
results and runs the numerical and completeness audits. It does not repeat the
expensive 12-seed RML calculation, Monte Carlo sampling, or multistart
collective-POVM optimization. Each module README gives the corresponding full
rerun command.

## Audit and data integrity

- `REPRODUCIBILITY_AUDIT.md` records the completed checks and the numerical
  optimization caveats.
- `MANIFEST.tsv` lists the relative path, size, and SHA-256 digest of every
  archived file.
- `SHA256SUMS.txt` supports transport verification with
  `shasum -a 256 -c SHA256SUMS.txt`.
- `audit_archive.py` checks the data fields, dimensions, sample counts,
  normalizations, principal numerical results, and submission PNGs without
  rerunning the expensive optimizations.

PDF hashes may change because Matplotlib embeds creation-time metadata.
Strict rendering comparisons therefore use PNG files, while numerical
receiver comparisons use the probabilities, matrices, and validation
residuals stored in NPZ/JSON files.
