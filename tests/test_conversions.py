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


def test_pyscf_to_system_carries_charge_and_spin():
    """A charged Mole must hand its charge and multiplicity to the System.

    A charge-conditioned model reads these off the System. Without them it
    falls back to a neutral singlet and predicts the wrong species, with no
    error to show for it.
    """
    from pyscf import gto

    from qpet import pyscf_to_system

    mol = gto.M(atom="O 0 0 0; H 0 0.96 0", basis="sto-3g", charge=-1, spin=0)
    system = pyscf_to_system(mol)

    assert set(system.known_data()) == {"charge", "spin_multiplicity"}
    assert float(system.get_data("charge").block().values[0, 0]) == -1.0
    assert float(system.get_data("spin_multiplicity").block().values[0, 0]) == 1.0

    doublet = gto.M(atom="O 0 0 0; H 0 0.96 0", basis="sto-3g", charge=0, spin=1)
    system = pyscf_to_system(doublet)
    assert float(system.get_data("spin_multiplicity").block().values[0, 0]) == 2.0


def test_attach_conditioning_keeps_values_the_caller_set():
    """Data already on the System wins, unless the caller asks to overwrite."""
    import torch
    from metatomic.torch import System

    from qpet import attach_conditioning

    system = System(
        types=torch.tensor([1, 1], dtype=torch.int32),
        positions=torch.zeros((2, 3)),
        cell=torch.zeros((3, 3)),
        pbc=torch.zeros(3, dtype=torch.bool),
    )
    attach_conditioning(system, charge=-1, spin_multiplicity=3)
    attach_conditioning(system, charge=0, spin_multiplicity=1)
    assert float(system.get_data("charge").block().values[0, 0]) == -1.0
    assert float(system.get_data("spin_multiplicity").block().values[0, 0]) == 3.0

    attach_conditioning(system, charge=0, spin_multiplicity=1, overwrite=True)
    assert float(system.get_data("charge").block().values[0, 0]) == 0.0
    assert float(system.get_data("spin_multiplicity").block().values[0, 0]) == 1.0
