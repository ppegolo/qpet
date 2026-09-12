"""The tabulated independent-atom model, and how it differs from the promolecule."""

from __future__ import annotations

import numpy as np
import pytest

from qpet.it92 import IT92, it92_form_factor
from qpet.scattering import spherical_atom_form_factor


def s_to_q(s):
    return 4.0 * np.pi * np.asarray(s, dtype=float)


def test_form_factor_is_the_electron_count_at_zero():
    for z in (1, 6, 7, 8, 16, 26):
        assert it92_form_factor(z, np.array([0.0]))[0] == pytest.approx(z, abs=0.02)


def test_hydrogen_is_the_bonded_atom_not_the_free_one():
    """The tabulated hydrogen is contracted; the exact free atom is analytic.

    A free hydrogen 1s gives ``f = 16 / (4 + q^2)^2`` in atomic units, which is what
    the Hartree-Fock route returns. The tabulated value is well above it, and mixing
    the two makes a bonded hydrogen look far more contracted than it is.
    """
    s = 0.25
    q_bohr = s_to_q(s) * 0.52917721
    exact = 16.0 / (4.0 + q_bohr**2) ** 2
    promolecule = float(spherical_atom_form_factor(1, np.array([s_to_q(s)]))[0])
    tabulated = float(it92_form_factor(1, np.array([s_to_q(s)]))[0])
    assert promolecule == pytest.approx(exact, rel=2e-3)
    assert tabulated > 1.2 * promolecule


def test_light_elements_sit_above_the_promolecule_atom():
    """A few per cent at intermediate angle, shrinking with atomic number."""
    s = np.array([0.17, 0.25, 0.33])
    gaps = {}
    for z in (6, 7, 8, 16):
        tab = it92_form_factor(z, s_to_q(s))
        hf = spherical_atom_form_factor(z, s_to_q(s))
        gaps[z] = float(np.max(tab / hf - 1.0))
        assert gaps[z] > 0.0
    assert gaps[6] > gaps[16]
    assert gaps[6] < 0.06


def test_table_covers_the_biological_elements():
    for z in (1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 25, 26, 30, 34, 53):
        assert z in IT92
        a, b, c = IT92[z]
        assert len(a) == len(b) == 4
        assert isinstance(c, float)
