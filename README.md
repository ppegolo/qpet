# qpet

Interface [metatomic](https://github.com/metatensor/metatomic) machine-learning
electron-density models with [PySCF](https://pyscf.org/).

`qpet` predicts resolution-of-identity (RI) electron densities from trained
metatomic models and evaluates them alongside exact density-matrix references,
on equal footing: density and electrostatic potential on arbitrary points and
grids, multipoles, CUBE export, and SCF initial guesses.

## What it provides

- **`DensityModel`** — wraps a model; `predict(mol)` returns an `RIDensity`.
  Two inference backends behind one class: `"metatomic"` runs the exported
  model through metatensor, `"petfast"` runs the same weights through an
  inference-only reimplementation that is several times faster and holds far
  larger systems. `backend="auto"` (the default) picks the fast one for a
  `.petfast` artifact.
- **`FastESP`** — the electrostatic potential in closed form on the GPU,
  replacing the per-(function, point) libcint call behind `RIDensity.potential`
  with a handful of floating-point operations. `RIDensity.potential` uses it
  automatically on a GPU once the grid is large enough to repay building it;
  `fast=True`/`False` overrides that.
- **Density value objects** — `RIDensity` (from RI coefficients, with
  `density`, `potential`, `charge`, `dipole`, `quadrupole`, `select_atoms`,
  `normalize_charge`, `constrain_dipole`, `scf_guess`) and `DMDensity` (from an
  AO density matrix). Each carries its own geometry, so the two are
  interchangeable wherever a field is expected.
- **CUBE utilities** — `CubeGrid` and `becke_grid` for sampling fields on
  grids; `write_density_cube`, `write_esp_cube`, `write_esp_error_cube` and
  `write_cube` for Gaussian CUBE export, `read_cube` for the round trip.
- **Provenance** — `file_fingerprint` and `model_provenance` identify the model
  a result came from by content hash rather than by path.
- **SCF bridge** — `dm_from_auxdensity` builds a one-diagonalization initial
  guess `dm0` from a predicted density.

```python
from qpet import DensityModel, model_provenance

model = DensityModel("model.pt", head="mtt::rho_c_jfit_coulomb",
                     aux_basis="def2-universal-jfit")
rho = model.predict(mol).normalize_charge()

rho.potential(points)                  # ESP in a.u., points in Angstrom
rho.dipole(origin)                     # multipoles from RI coefficients
rho.scf_guess(mol, xc="pbe")           # dm0 for mf.kernel(dm0=...)

model_provenance("model.pt")           # sha256 + versions, for the output file
```

The same call runs the fast backend; the head comes from the artifact, and
everything downstream of `predict` is unchanged:

```python
model = DensityModel("model.petfast", aux_basis="def2-universal-jfit")
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

CI runs `tox -e lint` and `tox -e tests` on every push and pull request.

## License

MIT
