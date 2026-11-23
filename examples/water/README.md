# Water: density and potential as cube files

```bash
python water_density_cube.py model.pt
```

Predicts the density of a water molecule with an exported metatomic model, runs
the ωB97M-V/def2-TZVPD reference with PySCF, and writes the density and the
electrostatic potential of both on one grid:

- `water_density_ml.cube`, `water_esp_ml.cube`
- `water_density_dft.cube`, `water_esp_dft.cube`

`AUX_BASIS` and `HEAD` at the top of the script are the auxiliary basis and the
output name of the model.
