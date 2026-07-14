"""Structure: a light molecular-geometry container with a calculator slot.

Wraps a PySCF ``Mole``, an ASE ``Atoms``, or a metatomic ``System`` behind a
uniform interface. Geometry (positions, atomic numbers, symbols) is always
available; the PySCF ``Mole`` is produced lazily and requires a basis for
non-``Mole`` inputs.

Attach a calculator via :attr:`calc` to enable :meth:`compute_density` and
:meth:`compute_potential`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyscf import gto
from pyscf.data import elements

from qpet.conversions import BOHR_TO_ANG

if TYPE_CHECKING:
    import ase
    from metatomic.torch import System

    from qpet.calculators.base import Calculator


class Structure:
    """
    Molecular structure with an optional scalar-field calculator.

    Accepts a PySCF ``Mole``, ASE ``Atoms``, or metatomic ``System`` as input
    and exposes a uniform geometry API. When constructed from a non-``Mole``
    input, :attr:`mol` is converted lazily and requires ``basis`` to have been
    provided.

    :param atoms: Molecular structure (PySCF ``Mole``, ASE ``Atoms``, or
        metatomic ``System``)
    :param basis: Basis set name for PySCF ``Mole`` conversion (required to
        access :attr:`mol` if ``atoms`` is not already a ``Mole``)
    :param charge: Molecular charge (used when converting to a PySCF ``Mole``)
    :param spin: Number of unpaired electrons (used when converting to a
        PySCF ``Mole``)

    Example::

        from qpet import Structure, DMCalculator
        from pyscf import gto, scf

        mol = gto.M(
            atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", basis="cc-pvdz"
        ).build()
        dm = scf.RHF(mol).run().make_rdm1()

        structure = Structure(mol)
        structure.calc = DMCalculator(dm)
        potential = structure.compute_potential(points)
    """

    def __init__(
        self,
        atoms: gto.Mole | ase.Atoms | System,
        basis: str | None = None,
        charge: int = 0,
        spin: int = 0,
    ) -> None:
        self._original = atoms
        self._basis = basis
        self._charge = charge
        self._spin = spin
        self._mol: gto.Mole | None = None
        self._charges: np.ndarray | None = None
        self.calc: Calculator | None = None

        # If already a Mole, store directly and extract charge/spin from it.
        if isinstance(atoms, gto.Mole):
            self._mol = atoms
            self._charge = atoms.charge
            self._spin = atoms.spin

    @property
    def positions(self) -> np.ndarray:
        """
        Atomic positions in Angstrom, shape (N, 3).

        Always available regardless of whether basis is provided.
        """
        if self._mol is not None:
            return self._mol.atom_coords(unit="Ang")
        elif hasattr(self._original, "get_positions"):  # ase.Atoms
            return np.asarray(self._original.get_positions())
        elif hasattr(self._original, "positions"):  # metatomic System
            return self._original.positions.detach().cpu().numpy()
        else:
            raise TypeError(f"Cannot extract positions from {type(self._original)}")

    @property
    def atomic_numbers(self) -> np.ndarray:
        """
        Atomic numbers, shape (N,).

        Always available regardless of whether basis is provided.
        """
        if self._mol is not None:
            return np.array(
                [self._mol.atom_charge(i) for i in range(self._mol.natm)], dtype=int
            )
        elif hasattr(self._original, "get_atomic_numbers"):  # ase.Atoms
            return np.asarray(self._original.get_atomic_numbers())
        elif hasattr(self._original, "types"):  # metatomic System
            return self._original.types.detach().cpu().numpy().astype(int)
        else:
            raise TypeError(
                f"Cannot extract atomic numbers from {type(self._original)}"
            )

    @property
    def symbols(self) -> list[str]:
        """
        Element symbols as a list of strings.

        Always available regardless of whether basis is provided.
        """
        if self._mol is not None:
            return [self._mol.atom_symbol(i) for i in range(self._mol.natm)]
        elif hasattr(self._original, "get_chemical_symbols"):  # ase.Atoms
            return list(self._original.get_chemical_symbols())
        else:
            return [elements.ELEMENTS[z] for z in self.atomic_numbers]

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        if self._mol is not None:
            return self._mol.natm
        else:
            return len(self.positions)

    @property
    def charge(self) -> int:
        """Molecular charge."""
        return self._charge

    @property
    def spin(self) -> int:
        """Number of unpaired electrons (2S)."""
        return self._spin

    @property
    def charges(self) -> np.ndarray | None:
        """Partial charges (FF or QM-derived), shape (N,)."""
        return self._charges

    @charges.setter
    def charges(self, value: np.ndarray | None) -> None:
        if value is not None:
            value = np.asarray(value)
            if value.shape != (self.n_atoms,):
                raise ValueError(
                    f"charges shape {value.shape} != expected ({self.n_atoms},)"
                )
        self._charges = value

    @property
    def mol(self) -> gto.Mole:
        """
        PySCF ``Mole`` object.

        If the input was not a ``Mole``, this property lazily converts it,
        which requires that ``basis`` was provided at construction.

        :raises ValueError: If basis is required but was not provided
        """
        if self._mol is None:
            if self._basis is None:
                raise ValueError(
                    "basis is required to convert ASE Atoms or metatomic System "
                    "to pyscf Mole. Provide basis= at Structure construction, "
                    "or use a calculator that doesn't require .mol access."
                )
            self._mol = self._to_pyscf()
        return self._mol

    def _to_pyscf(self) -> gto.Mole:
        """Convert the stored input to a PySCF ``Mole``."""
        from qpet.conversions import ase_to_pyscf, system_to_pyscf

        assert self._basis is not None  # guaranteed by the `mol` property
        if hasattr(self._original, "get_positions"):  # ase.Atoms
            return ase_to_pyscf(self._original, self._basis, self._charge, self._spin)
        elif hasattr(self._original, "positions"):  # metatomic System
            return system_to_pyscf(self._original, self._basis)
        else:
            raise TypeError(f"Cannot convert {type(self._original)} to pyscf Mole")

    @property
    def positions_bohr(self) -> np.ndarray:
        """Atomic positions in Bohr, shape (N, 3)."""
        return self.positions / BOHR_TO_ANG

    def compute_potential(self, pts: np.ndarray) -> np.ndarray:
        """
        Compute the electrostatic potential at given points.

        Requires a calculator to be attached via :attr:`calc`.

        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :return: Electrostatic potential in atomic units, shape (N,)
        :raises ValueError: If no calculator is attached
        """
        if self.calc is None:
            raise ValueError("No calculator attached. Set .calc to a Calculator first.")
        return self.calc.compute_potential(self, pts)

    def compute_density(self, pts: np.ndarray, batch_size: int = 5000) -> np.ndarray:
        """
        Compute the electron density at given points.

        Requires a calculator that supports density evaluation.

        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per batch (controls peak memory)
        :return: Electron density in electrons / Bohr^3, shape (N,)
        :raises ValueError: If no calculator is attached
        :raises NotImplementedError: If the calculator has no density model
        """
        if self.calc is None:
            raise ValueError("No calculator attached. Set .calc to a Calculator first.")
        return self.calc.compute_density(self, pts, batch_size)

    def __repr__(self) -> str:
        calc_str = type(self.calc).__name__ if self.calc else "None"
        return (
            f"{type(self).__name__}(n_atoms={self.n_atoms}, "
            f"charge={self._charge:+d}, calc={calc_str})"
        )
