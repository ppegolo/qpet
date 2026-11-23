"""qpet: ML electron densities on PySCF molecules.

Predicts densities with trained models, projects them to the exact electron
count, and evaluates them next to DFT references on equal footing: density
and electrostatic potential on points, multipoles, cube files, SCF guesses.
Lengths are in bohr except in :func:`build_mole` and file I/O.
"""

from ._tensormap import tensormap_coefficients
from .backends.metatomic import to_system
from .basis import aux_integrals, make_auxmol
from .density import (
    DMDensity,
    Ensemble,
    RIDensity,
    aux_density,
    fit_coefficients,
    nmae,
)
from .grids import (
    Cube,
    CubeGrid,
    atomic_becke_grid,
    becke_grid,
    read_cube,
    write_cube,
)
from .model import DensityModel, Prediction
from .mole import build_mole
from .normalize import project
from .scf import to_scf

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - generated at build time
    __version__ = "0.0.0+unknown"

__all__ = [
    "Cube",
    "CubeGrid",
    "DMDensity",
    "DensityModel",
    "Ensemble",
    "Prediction",
    "RIDensity",
    "__version__",
    "atomic_becke_grid",
    "aux_density",
    "aux_integrals",
    "becke_grid",
    "build_mole",
    "fit_coefficients",
    "make_auxmol",
    "nmae",
    "project",
    "read_cube",
    "tensormap_coefficients",
    "to_scf",
    "to_system",
    "write_cube",
]
