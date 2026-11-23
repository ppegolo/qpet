"""Auxiliary bases: PySCF names and even-tempered specs ``"etb:<ao_basis>:<beta>"``."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import numpy as np
from pyscf import df, gto
from pyscf.data import elements


def _is_etb_spec(aux_basis: str) -> bool:
    """Whether ``aux_basis`` has the form ``"etb:<ao_basis>:<beta>"``."""
    parts = aux_basis.split(":")
    return len(parts) == 3 and parts[0].lower() == "etb"


@cache
def _build_etb_basis(
    ao_basis: str, atomic_numbers: tuple[int, ...], beta: float
) -> dict[str, object]:
    """
    ``pyscf.df.aug_etb`` of a dummy molecule with one atom of each element.

    The orbital basis ``ao_basis`` comes from the spec, as in metatrain's density
    losses, so the basis does not depend on the basis the molecule is built in.
    """
    symbols = [elements.ELEMENTS[n] for n in atomic_numbers]
    mol = gto.Mole()
    mol.atom = "\n".join(f"{s} 0.0 0.0 {i * 10.0}" for i, s in enumerate(symbols))
    mol.basis = ao_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.build()

    return df.aug_etb(mol, beta=beta)


def _resolve_aux_basis(aux_basis: str, mol: gto.Mole) -> str | dict[str, object]:
    """The name unchanged, or the ETB basis of the elements of ``mol``."""
    if not _is_etb_spec(aux_basis):
        return aux_basis

    _, ao_basis, beta = aux_basis.split(":")
    # `atom_charge` would give Z_eff under an ECP; the element is what selects
    # the orbital basis the ETB is generated from.
    atomic_numbers = tuple(
        sorted({gto.charge(mol.atom_pure_symbol(a)) for a in range(mol.natm)})
    )
    return _build_etb_basis(ao_basis, atomic_numbers, float(beta))


def make_auxmol(mol: gto.Mole, aux_basis: str) -> gto.Mole:
    """
    Auxiliary ``Mole`` of ``mol``, ETB specs included.

    Unlike ``pyscf.df.addons.make_auxmol``, which rebuilds the atoms without the
    ECP, the effective nuclear charges and nuclear model of ``mol`` are kept, so
    ``auxmol.nelectron`` and the nuclear terms agree with ``mol``.
    """
    auxmol = df.addons.make_auxmol(mol, _resolve_aux_basis(aux_basis, mol))
    return copy_nuclei(mol, auxmol)


def copy_nuclei(
    source: gto.Mole, target: gto.Mole, atoms: Sequence[int] | slice = slice(None)
) -> gto.Mole:
    """Give ``target`` the effective nuclear charges and nuclear model of ``atoms``."""
    for column in (gto.mole.CHARGE_OF, gto.mole.NUC_MOD_OF):
        target._atm[:, column] = source._atm[atoms, column]
    return target


def aux_integrals(auxmol: gto.Mole) -> np.ndarray:
    """``q_k = ∫ φ_k dr`` per auxiliary function (zero for l > 0), shape (naux,)."""
    q = np.zeros(auxmol.nao)
    ao_loc = auxmol.ao_loc_nr()
    for ib in range(auxmol.nbas):
        if auxmol.bas_angular(ib) == 0:
            prim = (np.pi / auxmol.bas_exp(ib)) ** 1.5 / np.sqrt(4 * np.pi)
            ctr = auxmol._libcint_ctr_coeff(ib)
            q[ao_loc[ib] : ao_loc[ib] + ctr.shape[1]] = prim @ ctr
    return q
