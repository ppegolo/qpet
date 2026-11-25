# qpet

Machine-learning electron densities on [PySCF](https://pyscf.org/) molecules.

`qpet` evaluates trained [metatomic](https://github.com/metatensor/metatomic)
density models on a PySCF `Mole` and works with the predicted densities next to
DFT references: density and electrostatic potential on points, multipoles, cube
files, and SCF calculations started from the prediction. Lengths are in bohr
and outputs in atomic units, except `build_mole` and file I/O (Å).

Trained models (PET-ρ-OMol25) are at https://huggingface.co/lab-cosmo/pet-rho.

## Usage

```python
from pyscf import dft
from qpet import DensityModel, build_mole, to_scf

mol = build_mole(symbols, positions, basis="def2-svp", ecp="def2-svp")
model = DensityModel.from_release("pet-rho-omol", size="l")   # latest version
# "pet-rho-omol-nospin", "pet-rho-omol-gap-llpr" (ensembles=True): get_available_models()
# from_release(checkpoint_path="my.ckpt") exports a local checkpoint;
# DensityModel("model.pt", aux_basis=..., heads=...) loads an exported model
p = model.predict(mol)        # Prediction: p.rho, p.spin (RIDensity)

p.rho.potential(points)       # electrostatic potential, nuclei included
p.rho.dipole()                # multipoles from the RI coefficients
p.rho.populations()           # electrons per atom

mf = to_scf(p, dft.RKS(mol, xc="wb97m-d4"))   # or .to_gpu() first
mf.kernel()                   # the SCF starts from the predicted density
```

- **`DensityModel`**: one forward pass for all heads (`rho`, `spin`, and LLPR
  ensembles `rho_ensemble`, `spin_ensemble`). Predictions are projected to N
  electrons and 2S unpaired ones in the Coulomb metric by default
  (`normalize=` another metric, or `None` for the raw output);
  `predict(mol, cell=...)` lets the model see periodic neighbours;
  `predict_many` runs a list of isolated molecules.
  `backend="fast"` runs the same weights through an inference-only
  reimplementation for large systems.
- **`RIDensity`, `DMDensity`**: densities that carry their geometry, from RI
  coefficients or from an AO density matrix: `density`, `potential`,
  `potential_at_nuclei`, `integral`, `dipole`,
  `quadrupole`, `select_atoms` (and `RIDensity.gradient`). `RIDensity.fit` is the Coulomb-metric fit of a
  DFT density matrix (`fit_coefficients` for several at once),
  `RIDensity.normalized` the projection, `coulomb_distance` the Coulomb norm
  of the difference of two densities, `to_tensormap`/`from_tensormap` the
  layout of the training targets.
- **`Ensemble`**: members of an LLPR ensemble, recentred on the prediction;
  their spread on any observable is the error estimate.
- **`becke_grid`, `atomic_becke_grid`, `nmae`**: quadrature grids (with the
  atom of every point for Becke populations) and the normalized mean absolute
  error of two densities on one; `aux_density` evaluates several densities
  of one auxiliary basis in a single pass.
- **`CubeGrid`, `Cube`, `write_cube`, `read_cube`**: regular grids and cube
  files; `read_cube` returns a `Cube`.
- **`to_system`**: the metatomic `System` of a `Mole`, with its charge and
  spin multiplicity, to call a model directly.
- **`to_scf`**: the initial guess of a PySCF or gpu4pyscf Kohn-Sham object
  from the prediction (target functional on the predicted density, exact
  exchange from the orbitals of a first diagonalization).

## Installation

```bash
pip install -e .               # pyscf, torch, metatomic, vesin, scipy
pip install -e ".[numba]"      # iterative Coulomb solver, from 10,000 auxiliary functions
pip install -e ".[metatrain]"  # DensityModel.from_release, export to the fast backend
pip install -e ".[gpu4pyscf]"  # GPU integrals and to_scf on the GPU
```

GPU features use gpu4pyscf and CuPy when installed (`to_scf` on a `.to_gpu()`
object, the integrals of `RIDensity.fit` and `DMDensity.potential`).
`to_scf` relies on gpu4pyscf internals and was tested with gpu4pyscf 1.9.2.
`DensityModel.from_release` and the fast backend's export from a checkpoint
need metatrain.

The model runs on the `device` given to `DensityModel`. `RIDensity.fit`,
`DMDensity.density` and `DMDensity.potential` use a GPU when one is available
and give the same float64 result. The float32 closed-form ESP kernel is used
only with `potential(points, fast=True)`.

## Development

```bash
tox -e format    # auto-format
tox -e lint      # ruff format check + ruff check + mypy
tox -e tests     # run the test suite
```

## License

MIT
