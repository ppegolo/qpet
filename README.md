# qpet

Interface [metatomic](https://github.com/metatensor/metatomic) machine-learning
electron-density models with [PySCF](https://pyscf.org/).

`qpet` predicts resolution-of-identity (RI) electron densities from trained
metatomic models and evaluates them alongside exact density-matrix references:
density and electrostatic potential on points, multipoles, CUBE export, and
SCF initial guesses.

## What it provides

- **`DensityModel`** — wraps a metatomic model; `predict(mol)` returns an
  `RIDensity`.
- **Density value objects** — `RIDensity` (from RI coefficients, with
  `density`, `potential`, `charge`, `dipole`, `quadrupole`,
  `normalize_charge`, `constrain_dipole`, `scf_guess`) and `DMDensity`
  (from an AO density matrix). Each carries its own geometry.
- **SCF bridge** — `dm_from_auxdensity` builds a one-diagonalization initial
  guess `dm0` from a predicted density.
- **CUBE export** — `write_density_cube`, `write_esp_cube`, `write_cube`.

```python
from qpet import DensityModel

model = DensityModel("model.pt", head="mtt::rho_c_jfit_coulomb",
                     aux_basis="def2-universal-jfit")
rho = model.predict(mol).normalize_charge()

rho.potential(points)                  # ESP in a.u., points in Angstrom
rho.dipole(origin)                     # multipoles from RI coefficients
rho.scf_guess(mol, xc="pbe")           # dm0 for mf.kernel(dm0=...)
```

## Installation

```bash
pip install -e .   # pyscf, numpy, scipy, ase, torch, metatensor, metatomic, vesin
```

## Development

```bash
tox -e format    # auto-format
tox -e lint      # ruff format check + ruff check + mypy
tox -e tests     # run the test suite
```
