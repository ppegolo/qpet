"""Electron count and spin parity for conditioned density models.

Charge is an input everywhere; spin follows from it. :func:`spin_from_parity`
returns the smallest spin consistent with the electron count, and
:func:`check_spin` rejects a spin of the wrong parity before PySCF would,
closer to the cause.
"""

from __future__ import annotations

import numpy as np


def electron_count(atomic_numbers: np.ndarray, *, charge: int) -> int:
    """
    Number of electrons in a system at a given charge.

    :param atomic_numbers: Atomic numbers of every atom in the system.
    :param charge: Net charge of the system.
    :return: Electron count.
    """
    return int(np.sum(np.asarray(atomic_numbers, dtype=int))) - charge


def spin_from_parity(atomic_numbers: np.ndarray, *, charge: int) -> int:
    """
    The smallest spin consistent with the electron count.

    Low-spin assumption: 0 for an even electron count, 1 for an odd one. A
    high-spin ground state (e.g. a transition-metal site) breaks it.

    :param atomic_numbers: Atomic numbers of every atom in the system.
    :param charge: Net charge of the system.
    :return: Number of unpaired electrons (PySCF's ``spin``, i.e. 2S), 0 or 1.
    """
    return electron_count(atomic_numbers, charge=charge) % 2


def check_spin(atomic_numbers: np.ndarray, *, charge: int, spin: int) -> None:
    """
    Reject a spin whose parity contradicts the electron count.

    PySCF would refuse the same combination when the ``Mole`` is built, but far
    from the place the numbers came from. Failing here names both inputs.

    :param atomic_numbers: Atomic numbers of every atom in the system.
    :param charge: Net charge of the system.
    :param spin: Proposed number of unpaired electrons.
    :raises ValueError: If ``spin`` is negative, exceeds the electron count, or
        has the wrong parity for it.
    """
    n_electrons = electron_count(atomic_numbers, charge=charge)
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
