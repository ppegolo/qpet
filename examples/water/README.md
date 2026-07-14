# qpet examples

## `water_density_cube.py`

Computes the electron density of a water molecule with a metatomic ML model
and with DFT, writing each to a Gaussian CUBE file.

```bash
cd examples
python water_density_cube.py
```

Outputs:

- `water_density_ml.cube` — ML-predicted density (requires a metatomic model
  saved as `pet-scfbench.pt` in this folder; skipped if absent).
- `water_density_dft.cube` — PBE / def2-svp DFT reference density.

The ML branch needs the `metatomic` optional dependencies:

```bash
pip install "qpet[metatomic]"
```

`OUTPUT_NAME`, `BASIS_NAME`, and `BASIS_DEFINITION` at the top of the script
are model-specific and must match the auxiliary basis the model was trained
on. Adjust them for your own model.
