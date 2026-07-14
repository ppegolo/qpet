"""qpet: interface metatomic ML density models with PySCF.

Predicts RI density-fitting coefficients from metatomic models and evaluates
electron density and electrostatic potential from them, on equal footing.

The metatomic-backed predictor requires the ``metatomic`` optional dependencies
(``pip install qpet[metatomic]``); the rest of the package works with the
core dependencies alone.
"""

from .calculators import Calculator, DMCalculator, MetatomicCalculator, RICalculator
from .conversions import (
    BOHR_TO_ANG,
    ase_to_pyscf,
    pyscf_to_ase,
    pyscf_to_atom_string,
    pyscf_to_system,
    system_to_atom_string,
    system_to_pyscf,
)
from .cube import write_cube, write_density_cube, write_esp_cube
from .observables import get_ri_charge, get_ri_dipole, get_ri_quadrupole
from .structure import Structure

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - _version.py is generated at build time
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "Structure",
    "Calculator",
    "DMCalculator",
    "RICalculator",
    "MetatomicCalculator",
    "BOHR_TO_ANG",
    "ase_to_pyscf",
    "pyscf_to_ase",
    "pyscf_to_atom_string",
    "pyscf_to_system",
    "system_to_atom_string",
    "system_to_pyscf",
    "get_ri_charge",
    "get_ri_dipole",
    "get_ri_quadrupole",
    "write_cube",
    "write_density_cube",
    "write_esp_cube",
]

# The metatomic density predictor pulls in torch and is only importable with the
# ``[metatomic]`` extra installed.
try:
    from .predictor import MetatomicDensityOutput, MetatomicDensityPredictor

    __all__ += ["MetatomicDensityPredictor", "MetatomicDensityOutput"]
except ImportError:  # pragma: no cover - exercised only without the extra
    pass
