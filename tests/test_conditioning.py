"""Tests for electron count and spin parity."""

from __future__ import annotations

import re

import numpy as np
import pytest

from qpet import check_spin, spin_from_parity
from qpet.conditioning import electron_count

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
