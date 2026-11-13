"""qpet: ML electron densities on PySCF molecules.

Predicts densities with trained models, projects them to the exact electron
count, and evaluates them next to DFT references on equal footing: density
and electrostatic potential on points, multipoles, cube files, SCF guesses.
Lengths are in bohr except in :func:`build_mole` and file I/O.
"""

from .basis import make_auxmol
from .conditioning import nuclear_charges_of
from .density import DMDensity, Ensemble, RIDensity, nuclear_potential
from .grids import CubeGrid, becke_grid, read_cube, write_cube
from .model import DensityModel, Prediction
from .mole import build_mole
from .normalize import aux_integrals, direction, project
from .scf import to_scf

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - generated at build time
    __version__ = "0.0.0+unknown"

__all__ = [
    "CubeGrid",
    "DMDensity",
    "DensityModel",
    "Ensemble",
    "Prediction",
    "RIDensity",
    "__version__",
    "aux_integrals",
    "becke_grid",
    "build_mole",
    "direction",
    "make_auxmol",
    "nuclear_charges_of",
    "nuclear_potential",
    "project",
    "read_cube",
    "to_scf",
    "write_cube",
]
