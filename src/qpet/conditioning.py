"""Electron count and spin parity for conditioned density models.

Charge is an input everywhere; spin follows from it. :func:`spin_from_parity`
returns the smallest spin consistent with the electron count, and
:func:`check_spin` rejects a spin of the wrong parity before PySCF would,
closer to the cause.

Electron counts are ECP-aware: an element carrying an effective core potential
(the def2 family assigns one to Rb and heavier) has its core electrons removed
from the calculation, so a reference density integrates to ``Z_eff - charge``
and not to ``Z - charge``. :func:`electron_count` takes either a built
``Mole`` or bare atomic numbers plus the ECP name, and returns the count the
density actually carries.

The same distinction splits two per-atom quantities that PySCF folds into one
column: :func:`atomic_numbers_of` gives the element identity ``Z`` (symbols,
radii), :func:`nuclear_charges_of` the charge the nucleus actually exerts,
``Z_eff = Z - n_core`` (nuclear potential, dipoles, centre of nuclear charge).
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable
from functools import cache
from typing import Any

import numpy as np

#: Lightest element the def2 family gives an effective core potential to.
FIRST_ECP_ELEMENT = 37

__all__ = [
    "atomic_numbers_of",
    "check_spin",
    "core_electron_count",
    "core_electrons_per_atom",
    "electron_count",
    "nuclear_charges_of",
    "spin_from_parity",
]


@cache
def _named_ecp_core(name: str, symbol: str) -> int:
    """Core electrons a named ECP removes from one element (0 if it has none)."""
    from pyscf import gto

    # PySCF prints "ECP <name> not found for <symbol>" for every element
    # without one; that is the common case, not a problem worth reporting.
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = gto.basis.load_ecp(name, symbol)
    return int(loaded[0]) if loaded else 0


def _ecp_core_electrons(symbol: str, ecp: Any) -> int:
    """Core electrons an ECP specification removes from one element."""
    if not ecp:
        return 0
    spec = ecp.get(symbol, ecp.get("default")) if isinstance(ecp, dict) else ecp
    if not spec:
        return 0
    if isinstance(spec, (list, tuple)):  # already-parsed [ncore, blocks]
        return int(spec[0]) if len(spec) else 0
    return _named_ecp_core(str(spec), symbol)


def _is_mole(obj: Any) -> bool:
    return hasattr(obj, "atom_symbol") and hasattr(obj, "natm")


def atomic_numbers_of(mol: Any) -> np.ndarray:
    """Element identity ``Z`` of every atom of a built ``Mole``."""
    from pyscf import gto

    return np.array(
        [gto.charge(mol.atom_symbol(i)) for i in range(mol.natm)], dtype=int
    )


def core_electrons_per_atom(mol_or_numbers: Any, *, ecp: Any = None) -> np.ndarray:
    """
    Core electrons an ECP removes from each atom.

    :param mol_or_numbers: A built PySCF ``Mole`` or an iterable of atomic
        numbers.
    :param ecp: ECP name (or PySCF ECP dict). Required for the atomic-number
        form; for a ``Mole`` it is a fallback used only when the ``Mole`` names
        no ECP itself (auxiliary molecules built outside qpet).
    :return: Integer array, shape ``(n_atoms,)``, zeros without an ECP.
    """
    from pyscf import gto

    if _is_mole(mol_or_numbers):
        mol = mol_or_numbers
        # A Mole built with `ecp=` already carries the reduced nuclear charges.
        applied = np.array([int(mol.atom_nelec_core(i)) for i in range(mol.natm)])
        if applied.any():
            return applied
        # `pyscf.df.addons.make_auxmol` rebuilds `_atm` without the ECP, so an
        # auxiliary Mole reports full-Z charges while still naming its ECP;
        # :func:`qpet.make_auxmol` repairs this, foreign auxmols may not.
        spec = mol.ecp if mol.ecp else ecp
        return np.array(
            [_ecp_core_electrons(mol.atom_symbol(i), spec) for i in range(mol.natm)],
            dtype=int,
        )

    numbers = np.asarray(list(mol_or_numbers), dtype=int)
    return np.array(
        [_ecp_core_electrons(gto.mole._std_symbol(int(z)), ecp) for z in numbers],
        dtype=int,
    )


def core_electron_count(mol_or_numbers: Any, *, ecp: Any = None) -> int:
    """
    Electrons removed by effective core potentials, summed over atoms.

    See :func:`core_electrons_per_atom` for the arguments.
    """
    return int(core_electrons_per_atom(mol_or_numbers, ecp=ecp).sum())


def nuclear_charges_of(mol: Any, *, ecp: Any = None) -> np.ndarray:
    """
    Effective nuclear charge ``Z_eff = Z - n_core`` of every atom.

    :param mol: A built ``Mole``, orbital or auxiliary.
    :param ecp: Fallback ECP name for a ``Mole`` that names none.
    :return: Integer array, shape ``(n_atoms,)``.
    """
    return atomic_numbers_of(mol) - core_electrons_per_atom(mol, ecp=ecp)


def electron_count(
    mol_or_numbers: Any, *, charge: int | None = None, ecp: Any = None
) -> int:
    """
    Number of electrons a density carries, with ECP cores removed.

    :param mol_or_numbers: A built PySCF ``Mole`` — orbital or auxiliary — or
        an iterable of atomic numbers.
    :param charge: Net charge. Required for atomic numbers; for a ``Mole`` it
        defaults to ``mol.charge``.
    :param ecp: ECP name (or PySCF ECP dict) for the atomic-number form; a
        fallback for a ``Mole`` that names no ECP of its own.
    :return: Electron count.
    """
    if _is_mole(mol_or_numbers):
        mol = mol_or_numbers
        net = mol.charge if charge is None else charge
        return int(nuclear_charges_of(mol, ecp=ecp).sum() - net)

    if charge is None:
        raise TypeError("electron_count needs 'charge' for bare atomic numbers.")
    numbers = np.asarray(list(mol_or_numbers), dtype=int)
    core = core_electron_count(numbers, ecp=ecp)
    return int(numbers.sum()) - core - int(charge)


def spin_from_parity(
    atomic_numbers: Iterable[int], *, charge: int, ecp: Any = None
) -> int:
    """
    The smallest spin consistent with the electron count.

    Low-spin assumption: 0 for an even electron count, 1 for an odd one. A
    high-spin ground state (e.g. a transition-metal site) breaks it. ECP cores
    always hold an even number of electrons, so the answer does not depend on
    ``ecp``.

    :param atomic_numbers: Atomic numbers of every atom in the system.
    :param charge: Net charge of the system.
    :param ecp: Effective core potential name, if any.
    :return: Number of unpaired electrons (PySCF's ``spin``, i.e. 2S), 0 or 1.
    """
    return electron_count(atomic_numbers, charge=charge, ecp=ecp) % 2


def check_spin(
    atomic_numbers: Iterable[int], *, charge: int, spin: int, ecp: Any = None
) -> None:
    """
    Reject a spin whose parity contradicts the electron count.

    PySCF would refuse the same combination when the ``Mole`` is built, but far
    from the place the numbers came from. Failing here names both inputs.

    :param atomic_numbers: Atomic numbers of every atom in the system.
    :param charge: Net charge of the system.
    :param spin: Proposed number of unpaired electrons.
    :param ecp: Effective core potential name, if any.
    :raises ValueError: If ``spin`` is negative, exceeds the electron count, or
        has the wrong parity for it.
    """
    n_electrons = electron_count(atomic_numbers, charge=charge, ecp=ecp)
    if spin < 0:
        raise ValueError(f"spin must be non-negative, got {spin}.")
    if spin > n_electrons:
        raise ValueError(
            f"spin {spin} exceeds the electron count {n_electrons} "
            f"(charge {charge:+d})."
        )
    if (n_electrons - spin) % 2 != 0:
        raise ValueError(
            f"spin {spin} is inconsistent with {n_electrons} electrons at charge "
            f"{charge:+d}; the smallest consistent value is {n_electrons % 2}."
        )
