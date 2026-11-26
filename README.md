# qpet

Machine-learning electron densities on [PySCF](https://pyscf.org/) molecules.

`qpet` runs [metatomic](https://github.com/metatensor/metatomic) density models,
such as [PET-ρ-OMol25](https://huggingface.co/lab-cosmo/pet-rho), on a PySCF
`Mole`, and computes observables from the predicted densities. Lengths are in
bohr and outputs in atomic units, except for `build_mole` and files (Å).

## Installation

```bash
pip install git+https://github.com/ppegolo/qpet
```

Optional extras: `qpet[numba]` for the iterative Coulomb solver on large
systems, `qpet[gpu4pyscf]` for GPU integrals and SCF. Functionals with D4
dispersion, like the one in the example below, also need `pyscf-dispersion`.

## Quick start

```python
import numpy as np
from pyscf import dft
from qpet import DensityModel, build_mole, to_scf

# formaldehyde, positions in Å
symbols = ["C", "O", "H", "H"]
positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.205], [0.0, 0.943, -0.587], [0.0, -0.943, -0.587]]
mol = build_mole(symbols, positions, basis="def2-svp")

model = DensityModel.from_release("pet-rho-omol", size="l")
prediction = model.predict(mol)

points = np.array([[0.0, 0.0, 5.0]])  # bohr
prediction.rho.potential(points)  # electrostatic potential on points
prediction.rho.dipole()           # dipole moment
prediction.spin.populations()     # unpaired electrons per atom

mf = to_scf(prediction, dft.RKS(mol, xc="wb97m-d4"))  # SCF from the prediction
mf.kernel()
```

With the predicted densities, you can:

- evaluate the density, its gradient and the electrostatic potential on any
  set of points;
- compute multipoles and atomic populations;
- write and read cube files;
- estimate errors from ensembles (`ensembles=True` with
  `pet-rho-omol-gap-llpr`);
- start a Kohn-Sham calculation from the predicted density.

`save_release` writes a released model to disk, as its metatrain checkpoint or,
for a `.pt` output, as a metatomic export.

The docstrings describe the options, and the
[cookbook recipe](https://atomistic-cookbook.org/examples/pet-rho-omol25/pet-rho-omol25.html)
shows a complete example.

## Citing

If you use qpet or PET-ρ-OMol25, please cite

```bibtex
@misc{pegolo2026density,
  title         = {A general-purpose electron density model for molecular chemistry},
  author        = {Pegolo, Paolo and Abbott, Joseph W. and Ceriotti, Michele},
  year          = {2026},
  eprint        = {2611.34732},
  archivePrefix = {arXiv},
}
```

## Development

```bash
tox -e format    # auto-format
tox -e lint      # ruff format check + ruff check + mypy
tox -e tests     # run the test suite
```

## License

MIT
