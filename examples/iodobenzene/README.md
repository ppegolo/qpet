# Iodobenzene: density and potential as cube files

```bash
python iodobenzene_cube.py
```

Predicts the density of iodobenzene with PET-ρ-OMol25, runs the
ωB97M-V/def2-TZVPD reference with PySCF, and writes the density and the
electrostatic potential of both on one grid:

- `iodobenzene_density_ml.cube`, `iodobenzene_esp_ml.cube`
- `iodobenzene_density_dft.cube`, `iodobenzene_esp_dft.cube`

The potential is positive beyond the iodine, along the C–I axis: the σ-hole.
