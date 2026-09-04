"""Auxiliary-basis resolution, including even-tempered (ETB) bases.

An RI density is only meaningful together with the auxiliary basis its
coefficients were fitted in, so every entry point that builds an auxiliary
``Mole`` goes through :func:`make_auxmol`. On top of the PySCF basis names that
``pyscf.df.addons.make_auxmol`` accepts, this adds the even-tempered
specification ``"etb:<ao_basis>:<beta>"`` used by metatrain's density losses
(see ``metatrain.utils.pyscf_loss``), so a model trained against an ETB
auxiliary basis can be evaluated with the same string it was trained with.
"""

from __future__ import annotations

from functools import cache

from pyscf import df, gto
from pyscf.data import elements

__all__ = ["is_etb_spec", "make_auxmol", "resolve_aux_basis"]


def is_etb_spec(aux_basis: str) -> bool:
    """
    Whether ``aux_basis`` is an even-tempered specification.

    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: True for the ETB form.
    """
    parts = aux_basis.split(":")
    return len(parts) == 3 and parts[0].lower() == "etb"


@cache
def _build_etb_basis(
    ao_basis: str, atomic_numbers: tuple[int, ...], beta: float
) -> dict[str, object]:
    """
    Build an even-tempered auxiliary basis with ``pyscf.df.aug_etb``.

    The basis is generated from a dummy molecule holding one atom of each
    element in the *orbital* basis ``ao_basis``, which is how SCFBench-style RI
    datasets (and metatrain's density losses) generate theirs. Taking the
    orbital basis from the spec rather than from the molecule keeps the
    auxiliary basis identical to the one the coefficients were fitted in, even
    if the molecule is built in a different orbital basis.

    :param ao_basis: Orbital (not auxiliary) basis name, e.g. ``"def2-svp"``.
    :param atomic_numbers: Unique atomic numbers to build the basis for.
    :param beta: Even-tempering ratio.
    :return: Mapping from element symbol to basis specification.
    """
    symbols = [elements.ELEMENTS[n] for n in atomic_numbers]
    # Spread the atoms out so the dummy molecule builds without warnings.
    mol = gto.Mole()
    mol.atom = "\n".join(f"{s} 0.0 0.0 {i * 10.0}" for i, s in enumerate(symbols))
    mol.basis = ao_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.build()

    return df.aug_etb(mol, beta=beta)


def resolve_aux_basis(aux_basis: str, mol: gto.Mole) -> str | dict[str, object]:
    """
    Resolve an auxiliary-basis specification for a given molecule.

    :param aux_basis: A PySCF basis name (e.g. ``"def2-universal-jfit"``) or an
        even-tempered specification ``"etb:<ao_basis>:<beta>"``, e.g.
        ``"etb:def2-svp:2.0"``.
    :param mol: Built ``Mole`` whose elements the basis must cover.
    :return: The name unchanged, or the generated ETB basis dictionary.
    """
    if not is_etb_spec(aux_basis):
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
    Build the auxiliary ``Mole`` for ``mol``, ETB specifications included.

    Drop-in replacement for ``pyscf.df.addons.make_auxmol`` that keeps the
    nuclei of ``mol``: PySCF rebuilds the atom table without the ECP, so its
    auxiliary ``Mole`` reports full-``Z`` nuclear charges (and electron count)
    for elements whose cores the calculation never had. Here the effective
    charges and nuclear model of ``mol`` are copied over, so
    ``auxmol.atom_charges()``, ``auxmol.nelectron`` and every nuclear term
    built from them agree with the orbital molecule.

    :param mol: Built ``Mole`` (orbital basis).
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: Auxiliary-basis ``Mole``.
    """
    auxmol = df.addons.make_auxmol(mol, resolve_aux_basis(aux_basis, mol))
    return _copy_nuclei(mol, auxmol)


def _copy_nuclei(source: gto.Mole, target: gto.Mole) -> gto.Mole:
    """Give ``target`` the effective nuclear charges and nuclear model of ``source``."""
    if target.natm != source.natm:  # pragma: no cover - PySCF keeps the atoms
        raise ValueError("auxiliary molecule does not match its orbital molecule")
    target._atm[:, gto.mole.CHARGE_OF] = source._atm[:, gto.mole.CHARGE_OF]
    target._atm[:, gto.mole.NUC_MOD_OF] = source._atm[:, gto.mole.NUC_MOD_OF]
    return target
