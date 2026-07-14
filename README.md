# qpet

Interface [metatomic](https://github.com/metatensor/metatomic) machine-learning
electron-density models with [PySCF](https://pyscf.org/).

`qpet` predicts resolution-of-identity (RI) density-fitting coefficients from
a trained metatomic model and turns them into evaluable scalar fields — electron
density and electrostatic potential — on equal footing. It also reads
density-matrix and exact-RI representations, and exports any field to a Gaussian
CUBE file.

## What it provides

- **`MetatomicDensityPredictor`** — run a metatomic model, get PySCF-ready RI
  coefficients (with optional charge and dipole normalization).
- **Calculators** — `DMCalculator`, `RICalculator`, `MetatomicCalculator`,
  sharing a common `Calculator` base that exposes `compute_density` and
  `compute_potential`.
- **`Structure`** — a light container wrapping a PySCF `Mole`, ASE `Atoms`, or
  metatomic `System`.
- **RI observables** — charge, dipole, quadrupole from RI coefficients.
- **CUBE export** — `write_density_cube`, `write_esp_cube`, `write_cube`.

## Installation

```bash
pip install -e .            # core: pyscf, numpy, scipy, ase
pip install -e ".[metatomic]"   # + torch, metatensor, metatomic, metatrain
```

## Development

```bash
tox -e tests     # run the test suite
tox -e lint      # ruff format check + ruff check + mypy
tox -e format    # auto-format
```
