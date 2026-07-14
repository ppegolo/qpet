"""qpet: interface metatomic ML density models with PySCF.

Predicts RI electron densities with metatomic models and evaluates them —
density, electrostatic potential, multipoles, CUBE export, SCF initial
guesses — alongside exact density-matrix references, on equal footing.
"""

from .conversions import (
    BOHR_TO_ANG,
    ase_to_pyscf,
    pyscf_to_ase,
    pyscf_to_atom_string,
    pyscf_to_system,
    system_to_atom_string,
    system_to_pyscf,
)
from .cube import (
    BeckeGrid,
    CubeData,
    becke_grid,
    read_cube,
    write_cube,
    write_density_cube,
    write_esp_cube,
)
from .fields import DMDensity, RIDensity, ScalarField
from .model import DensityModel
from .moments import (
    apply_dipole_correction,
    compute_S_vector,
    get_ri_charge,
    get_ri_dipole,
    get_ri_quadrupole,
    normalize_charge_additive,
)
from .scf import dm_from_auxdensity

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - _version.py is generated at build time
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "RIDensity",
    "DMDensity",
    "ScalarField",
    "DensityModel",
    "dm_from_auxdensity",
    "get_ri_charge",
    "get_ri_dipole",
    "get_ri_quadrupole",
    "apply_dipole_correction",
    "compute_S_vector",
    "normalize_charge_additive",
    "write_cube",
    "write_density_cube",
    "write_esp_cube",
    "read_cube",
    "CubeData",
    "becke_grid",
    "BeckeGrid",
    "BOHR_TO_ANG",
    "ase_to_pyscf",
    "pyscf_to_ase",
    "pyscf_to_atom_string",
    "pyscf_to_system",
    "system_to_atom_string",
    "system_to_pyscf",
]
