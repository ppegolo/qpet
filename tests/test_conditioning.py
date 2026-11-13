"""Tests for electron count and spin parity."""

from __future__ import annotations

import re

import numpy as np
import pytest
from pyscf import df, gto
from pyscf.lib.exceptions import BasisNotFoundError

from qpet.conditioning import (
    atomic_numbers_of,
    check_spin,
    electron_count,
    nuclear_charges_of,
    spin_from_parity,
)

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
    """An element with an effective core potential does not carry ``Z`` electrons."""

    def test_electron_count(self, hydrogen_iodide_ri):
        assert electron_count(hydrogen_iodide_ri[0]) == 26  # (53 - 28) + 1
        assert electron_count([53, 1], charge=1, ecp="def2-svp") == 25
        assert electron_count([53, 1], charge=0) == 54
        assert electron_count(WATER, charge=0, ecp="def2-svp") == 10
        with pytest.raises(BasisNotFoundError, match="^no-such-ecp$"):
            electron_count([53, 1], charge=0, ecp="no-such-ecp")

    def test_element_and_nuclear_charge_are_distinct(self, hydrogen_iodide_ri):
        mol, _, auxmol, _ = hydrogen_iodide_ri
        for m in (mol, auxmol):
            assert atomic_numbers_of(m).tolist() == [53, 1]
            assert nuclear_charges_of(m).tolist() == [25, 1]
        # An auxiliary Mole built without the ECP: its name supplies the cores.
        bare = df.addons.make_auxmol(
            gto.M(atom="I 0 0 0; H 0 0 1.61", basis="def2-svp", verbose=0),
            "def2-universal-jfit",
        )
        assert nuclear_charges_of(bare, ecp="def2-svp").tolist() == [25, 1]

    def test_ecp_cores_do_not_change_spin_parity(self):
        assert spin_from_parity([53, 1], charge=0, ecp="def2-svp") == 0
        assert spin_from_parity([53], charge=0, ecp="def2-svp") == 1
