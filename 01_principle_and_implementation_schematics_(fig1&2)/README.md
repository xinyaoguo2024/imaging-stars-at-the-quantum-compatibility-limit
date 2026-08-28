# Principle and Implementation Schematics

This module reproduces the two conceptual and implementation figures in the
main text:

- `reference_outputs/fig1_principle_schematic.*`: the interferometric-imaging
  and single-copy--collective-readout schematic;
- `reference_outputs/fig2_implementation_schematic.*`: the memory encoding,
  joint processing, and final-POVM implementation diagram.

## Quick reproduction

```bash
./run_all.sh
```

The script runs both plotting programs in `code/` and writes PDF, PNG, and SVG
files to `generated_outputs/`. The generated PNG files should be byte-identical
to the submission versions in `reference_outputs/`. PDF hashes may differ only
because of creation-time metadata.

## Dependencies

Python 3, NumPy, and Matplotlib. The programs do not read files outside this
module.

