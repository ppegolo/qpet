"""qpet: interface metatomic ML density models with PySCF.

Predicts RI electron densities with metatomic models and evaluates them —
density, electrostatic potential, multipoles, CUBE export, SCF initial
guesses — alongside exact density-matrix references, on equal footing.
"""

from .basis import make_auxmol, resolve_aux_basis
from .conditioning import (
    atomic_numbers_of,
    check_spin,
    core_electron_count,
    core_electrons_per_atom,
    electron_count,
    nuclear_charges_of,
    spin_from_parity,
)
from .conversions import (
    BOHR_TO_ANG,
    ase_to_pyscf,
    attach_conditioning,
    build_mole,
    pyscf_to_ase,
    pyscf_to_atom_string,
    pyscf_to_system,
    system_to_atom_string,
    system_to_pyscf,
)
from .cube import (
    BeckeGrid,
    CubeData,
    CubeGrid,
    becke_grid,
    read_cube,
    write_cube,
    write_density_cube,
    write_esp_cube,
    write_esp_error_cube,
)
from .esp_fast import FastESP
from .fields import DMDensity, RIDensity, ScalarField, nuclear_potential
from .model import DensityModel
from .moments import (
    DEFAULT_CHARGE_OFFSET,
    apply_dipole_correction,
    compute_S_vector,
    get_ri_charge,
    get_ri_dipole,
    get_ri_quadrupole,
    normalize_charge_additive,
)
from .partition import FragmentDensity, HirshfeldSplit, free_atom_density
from .provenance import (
    Fingerprint,
    file_fingerprint,
    model_provenance,
    provenance_columns,
)
from .scf import dm_from_auxdensity

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - _version.py is generated at build time
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "RIDensity",
    "FragmentDensity",
    "HirshfeldSplit",
    "free_atom_density",
    "DMDensity",
    "ScalarField",
    "DensityModel",
    "FastESP",
    "nuclear_potential",
    "dm_from_auxdensity",
    "spin_from_parity",
    "check_spin",
    "electron_count",
    "core_electron_count",
    "core_electrons_per_atom",
    "atomic_numbers_of",
    "nuclear_charges_of",
    "DEFAULT_CHARGE_OFFSET",
    "get_ri_charge",
    "get_ri_dipole",
    "get_ri_quadrupole",
    "apply_dipole_correction",
    "compute_S_vector",
    "normalize_charge_additive",
    "Fingerprint",
    "file_fingerprint",
    "model_provenance",
    "provenance_columns",
    "write_cube",
    "write_density_cube",
    "write_esp_cube",
    "write_esp_error_cube",
    "read_cube",
    "CubeData",
    "CubeGrid",
    "becke_grid",
    "BeckeGrid",
    "BOHR_TO_ANG",
    "make_auxmol",
    "resolve_aux_basis",
    "attach_conditioning",
    "build_mole",
    "ase_to_pyscf",
    "pyscf_to_ase",
    "pyscf_to_atom_string",
    "pyscf_to_system",
    "system_to_atom_string",
    "system_to_pyscf",
]
