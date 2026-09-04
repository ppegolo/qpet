"""Tests for electron count and spin parity."""

from __future__ import annotations

import re

import numpy as np
import pytest

from qpet import check_spin, spin_from_parity
from qpet.conditioning import core_electron_count, electron_count

WATER = np.array([8, 1, 1])  # 10 electrons when neutral


class TestElectronCount:
    def test_follows_charge(self):
        neutral = electron_count(WATER, charge=0)
        assert neutral == 10
        assert electron_count(WATER, charge=1) == neutral - 1
        assert electron_count(WATER, charge=-1) == neutral + 1


class TestSpin:
    def test_parity(self):
        assert spin_from_parity(WATER, charge=0) == 0
        assert spin_from_parity(WATER, charge=1) == 1

    def test_derived_spin_always_passes_the_check(self):
        for charge in (-2, -1, 0, 1, 2):
            spin = spin_from_parity(WATER, charge=charge)
            check_spin(WATER, charge=charge, spin=spin)

    def test_wrong_parity_is_rejected(self):
        message = (
            "spin 1 is inconsistent with 10 electrons at charge +0; "
            "the smallest consistent value is 0."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            check_spin(WATER, charge=0, spin=1)

    def test_out_of_range_spin_is_rejected(self):
        with pytest.raises(
            ValueError, match=re.escape("spin must be non-negative, got -1.")
        ):
            check_spin(WATER, charge=0, spin=-1)
        with pytest.raises(
            ValueError,
            match=re.escape("spin 99 exceeds the electron count 10 (charge +0)."),
        ):
            check_spin(WATER, charge=0, spin=99)

    def test_higher_spin_of_right_parity_is_allowed(self):
        check_spin(WATER, charge=0, spin=2)


class TestECPElectronCount:
    """
    An element with an effective core potential does not carry ``Z`` electrons.

    def2 assigns an ECP to Rb and heavier, so a reference density for such a
    system integrates to ``Z_eff - charge``. Counting ``Z - charge`` instead
    misses 28 electrons for iodine, which is what a charge projection would
    then try to add to the density.
    """

    def _hydrogen_iodide(self):
        from pyscf import gto

        return gto.M(
            atom="I 0 0 0; H 0 0 1.61",
            basis="def2-svp",
            ecp="def2-svp",
            verbose=0,
        )

    def test_iodine_counts_valence_electrons_only(self):
        mol = self._hydrogen_iodide()
        assert electron_count(mol) == 26  # (53 - 28) + 1
        assert electron_count([53, 1], charge=0, ecp="def2-svp") == 26
        assert electron_count([53, 1], charge=1, ecp="def2-svp") == 25
        # Without the ECP there is nothing to remove; that is the wrong answer
        # for a def2 calculation, and the reason the ECP has to travel along.
        assert electron_count([53, 1], charge=0) == 54

    def test_auxiliary_molecule_is_ecp_aware(self):
        """
        PySCF's ``make_auxmol`` rebuilds the atoms without the ECP, so its
        ``auxmol.nelectron`` reports ``Z``; qpet's copies the nuclei back and
        the helper reads either correctly.
        """
        from pyscf import df

        from qpet.basis import make_auxmol

        mol = self._hydrogen_iodide()
        foreign = df.addons.make_auxmol(mol, "def2-universal-jfit")
        assert foreign.nelectron == 54  # the trap
        assert electron_count(foreign) == 26
        auxmol = make_auxmol(mol, "def2-universal-jfit")
        assert auxmol.nelectron == electron_count(auxmol) == mol.nelectron == 26

    def test_element_identity_and_nuclear_charge_are_distinct(self):
        from pyscf import df, gto

        from qpet import atomic_numbers_of, core_electrons_per_atom, nuclear_charges_of
        from qpet.basis import make_auxmol

        mol = self._hydrogen_iodide()
        auxmol = make_auxmol(mol, "def2-universal-jfit")
        for m in (mol, auxmol):
            assert atomic_numbers_of(m).tolist() == [53, 1]
            assert nuclear_charges_of(m).tolist() == [25, 1]
            assert core_electrons_per_atom(m).tolist() == [28, 0]
        # An all-electron auxmol built elsewhere: the ECP name supplies the cores.
        bare = df.addons.make_auxmol(
            gto.M(atom="I 0 0 0; H 0 0 1.61", basis="def2-svp", verbose=0),
            "def2-universal-jfit",
        )
        assert nuclear_charges_of(bare).tolist() == [53, 1]
        assert nuclear_charges_of(bare, ecp="def2-svp").tolist() == [25, 1]
        assert core_electrons_per_atom([53, 1], ecp="def2-svp").tolist() == [28, 0]

    def test_all_electron_molecule_is_unchanged(self):
        from pyscf import gto

        from qpet import atomic_numbers_of, nuclear_charges_of

        mol = gto.M(atom="O 0 0 0; H 0.96 0 0; H -0.24 0.93 0", basis="sto-3g")
        assert atomic_numbers_of(mol).tolist() == [8, 1, 1]
        assert nuclear_charges_of(mol).tolist() == [8, 1, 1]

    def test_core_electron_count(self):
        assert core_electron_count([53, 1], ecp="def2-svp") == 28
        assert core_electron_count([53, 1]) == 0
        assert core_electron_count(WATER, ecp="def2-svp") == 0

    def test_light_elements_are_untouched(self):
        """An all-electron molecule must count the same with or without an ECP
        name: def2 defines no ECP below Rb."""
        from pyscf import gto

        water = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24", basis="def2-svp")
        assert electron_count(water) == 10
        assert electron_count(WATER, charge=0, ecp="def2-svp") == 10
        assert electron_count(WATER, charge=-1, ecp="def2-svp") == 11

    def test_ecp_cores_do_not_change_spin_parity(self):
        """ECP cores hold an even number of electrons, so the low-spin value
        is the same with and without one."""
        assert spin_from_parity([53, 1], charge=0, ecp="def2-svp") == 0
        assert spin_from_parity([53, 1], charge=0) == 0
        assert spin_from_parity([53], charge=0, ecp="def2-svp") == 1
