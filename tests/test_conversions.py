"""Tests for geometry conversions: charge/spin must survive the round trip."""

from __future__ import annotations

import ase

from qpet import ase_to_pyscf


def test_ase_to_pyscf_propagates_charge_and_spin():
    """
    Charge and spin are physics inputs, not metadata: a silently dropped
    charge would change every downstream density and potential.
    """
    atoms = ase.Atoms(
        "NH3",
        positions=[
            [0.000, 0.000, 0.116],
            [0.000, 0.939, -0.270],
            [0.813, -0.469, -0.270],
            [-0.813, -0.469, -0.270],
        ],
    )

    mol = ase_to_pyscf(atoms, "sto-3g", charge=1, spin=1)

    assert mol.charge == 1
    assert mol.spin == 1
    assert mol.nelectron == 9
