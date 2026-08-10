# qpet

Interface [metatomic](https://github.com/metatensor/metatomic) machine-learning
electron-density models with [PySCF](https://pyscf.org/).

`qpet` predicts resolution-of-identity (RI) electron densities from trained
metatomic models and evaluates them alongside exact density-matrix references,
on equal footing: density and electrostatic potential on points and surfaces,
multipoles, CUBE export, and SCF initial guesses.

## What it provides

- **`DensityModel`** — wraps a metatomic model; `predict(mol)` returns an
  `RIDensity`.
- **Density value objects** — `RIDensity` (from RI coefficients, with
  `density`, `potential`, `charge`, `dipole`, `quadrupole`, `select_atoms`,
  `normalize_charge`, `constrain_dipole`, `scf_guess`) and `DMDensity` (from an
  AO density matrix). Each carries its own geometry, so the two are
  interchangeable wherever a field is expected.
- **Surfaces** — `solvent_accessible_surface` returns points *and* the area
  each one represents; `SurfacePatch` carries points, areas, ownership and the
  rule that built them, because a potential without its surface definition is
  not reportable.
- **ESP evaluation and error** — `esp_on_surface`, and `esp_error`, which
  splits an error into a rigid offset and the shape that survives it.
- **Map export** — `export_esp_maps` writes density, ESP and error maps on one
  shared grid plus runnable PyMOL and ChimeraX scripts; `write_density_cube`,
  `write_esp_cube`, `write_cube` for single maps.
- **Provenance** — `file_fingerprint` and `model_provenance` identify the model
  a result came from by content hash rather than by path.
- **SCF bridge** — `dm_from_auxdensity` builds a one-diagonalization initial
  guess `dm0` from a predicted density.

```python
from qpet import DensityModel, esp_error, model_provenance

model = DensityModel("model.pt", head="mtt::rho_c_jfit_coulomb",
                     aux_basis="def2-universal-jfit")
rho = model.predict(mol).normalize_charge()

rho.potential(points)                  # ESP in a.u., points in Angstrom
rho.dipole(origin)                     # multipoles from RI coefficients
rho.scf_guess(mol, xc="pbe")           # dm0 for mf.kernel(dm0=...)

model_provenance("model.pt")           # sha256 + versions, for the output file
```

### The offset/shape split

`esp_error` reports `rmse`, `offset`, `shape_rmse`, `max_abs`, `contrast` and
`correlation`, area-weighted, with `rmse² = offset² + shape²` holding exactly
by construction.

The split matters because the two halves mean different things. A rigid offset
shifts the whole surface and leaves every *difference* on it intact, so a
correlation-based readout barely notices it while an absolute one is wrong
everywhere. Shape error is the part that changes what the surface says. Report
one RMSE and the two are indistinguishable; `contrast`, the reference's own
area-weighted spread, is what tells you whether a given shape error is small.

```python
stats = esp_error(predicted, reference, patch)
stats["shape_rmse"] / stats["contrast"]   # below 1: the surface still agrees
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
