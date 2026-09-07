"""Computed interaction channels: Coulomb energy, Hirshfeld volumes, dispersion."""

import numpy as np
import pytest
from pyscf import gto, scf

from qpet import RIDensity
from qpet.conversions import BOHR_TO_ANG
from qpet.interactions import (
    analytic_coulomb_interaction_energy,
    coulomb_interaction_energy,
    free_atom_volume,
    grid_coulomb_interaction_energy,
    hirshfeld_dispersion_energy,
    hirshfeld_volume_ratios,
)

_AUX = "def2-universal-jfit"


def _ri(atom: str, charge: int = 0) -> RIDensity:
    mol = gto.M(atom=atom, basis="def2-svp", charge=charge, verbose=0)
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()
    return RIDensity.from_dm(mol, dm, _AUX)


@pytest.fixture(scope="module")
def water_dimer_ri():
    """Two waters 3 A apart, RHF/sto-3g and Coulomb-fitted; 0-2 is the first."""
    mol = gto.M(
        atom=(
            "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0; O 3.0 0 0; H 3.3 0.9 0; H 3.9 -0.2 0"
        ),
        basis="sto-3g",
        verbose=0,
    )
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()
    return RIDensity.from_dm(mol, dm, _AUX)


def test_coulomb_energy_of_separated_ions_is_the_point_charge_limit():
    separation = 15.0
    lithium = _ri("Li 0 0 0", charge=1)
    fluoride = _ri(f"F {separation} 0 0", charge=-1)
    energy = coulomb_interaction_energy(lithium, fluoride)
    # The fitted charges, not the formal ones: the RI fit of an ion is not
    # exactly integral, and the quadrature has to reproduce what it is given.
    q_li = float(np.sum(lithium.charge(include_nuclear=True)))
    q_f = float(np.sum(fluoride.charge(include_nuclear=True)))
    # 1e-4 rather than machine precision: the grid is truncated at
    # ``support_radius``, which leaves a little of the ion's tail out.
    assert energy == pytest.approx(q_li * q_f * BOHR_TO_ANG / separation, rel=1e-4)


def test_coulomb_energy_is_symmetric_and_attractive_in_a_dimer(water_dimer_ri):
    donor, acceptor = water_dimer_ri.split_hirshfeld([0, 1, 2])
    forward = coulomb_interaction_energy(donor, acceptor)
    backward = coulomb_interaction_energy(acceptor, donor)
    assert forward == pytest.approx(backward, abs=2e-3)
    assert forward < 0.0


def test_free_atom_has_unit_volume_ratio():
    neon = _ri("Ne 0 0 0")
    ratio = hirshfeld_volume_ratios(neon)
    assert ratio.shape == (1,)
    # One only to the accuracy of the density's r^3 moment: the RI fit has a
    # small spurious tail and r^3 weights it heavily (1.06 here, 1.08 on a
    # level-3 grid). The bias is common to all atoms and a fitted coefficient
    # in front of the dispersion channel absorbs it.
    assert ratio[0] == pytest.approx(1.0, abs=0.1)


def test_bound_atoms_lose_volume(water_dimer_ri):
    ratios = hirshfeld_volume_ratios(water_dimer_ri, [0, 1, 2])
    # Hydrogens donate their density to oxygen and shrink the most.
    assert np.all(ratios > 0.2) and np.all(ratios < 1.2)
    assert ratios[1] < 0.8 and ratios[2] < 0.8
    assert ratios[0] > ratios[1] and ratios[0] > ratios[2]


def test_free_atom_volume_grows_with_the_element():
    assert free_atom_volume(1) < free_atom_volume(6) < free_atom_volume(17)


def test_dispersion_is_negative_and_decays_with_separation():
    def dimer(distance: float) -> float:
        mol = gto.M(atom=f"Ne 0 0 0; Ne {distance} 0 0", basis="def2-svp", verbose=0)
        dm = scf.RHF(mol).run(verbose=0).make_rdm1()
        density = RIDensity.from_dm(mol, dm, _AUX)
        return hirshfeld_dispersion_energy(density, [0])

    near, far = dimer(4.0), dimer(6.0)
    assert near < 0.0 and far < 0.0
    # Undamped at these separations, so the ratio is the R^-6 one to ~10 %.
    assert near / far == pytest.approx((6.0 / 4.0) ** 6, rel=0.1)


def test_dispersion_ignores_atoms_beyond_the_environment_radius():
    mol = gto.M(atom="Ne 0 0 0; Ne 4 0 0", basis="def2-svp", verbose=0)
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()
    density = RIDensity.from_dm(mol, dm, _AUX)
    assert hirshfeld_dispersion_energy(density, [0], environment_radius=2.0) == 0.0
    assert np.isfinite(hirshfeld_dispersion_energy(density, [0]))


def test_analytic_and_grid_agree_on_an_ri_split(water_dimer_ri):
    """With no Hirshfeld correction every analytic term is exact.

    What is left is the grid's own error: level-1 quadrature and the
    ``support_radius`` truncation, a few times 1e-5 Hartree here.
    """
    donor = water_dimer_ri.select_atoms([0, 1, 2])
    acceptor = water_dimer_ri.select_atoms([3, 4, 5])
    analytic = analytic_coulomb_interaction_energy(donor, acceptor)
    assert analytic == pytest.approx(
        grid_coulomb_interaction_energy(donor, acceptor), abs=5e-5
    )


def test_analytic_and_grid_agree_on_a_hirshfeld_split(water_dimer_ri):
    """
    To the quadrature error of the grid, which is what the difference measures.

    The grid result moves by about 1e-3 Hartree between levels 1 and 3 -- the
    correction density is a point-charge set and the finer grid samples it
    closer in -- so that is the scale the two methods can agree on.
    """
    donor, acceptor = water_dimer_ri.split_hirshfeld([0, 1, 2])
    analytic = analytic_coulomb_interaction_energy(donor, acceptor)
    assert analytic == pytest.approx(
        grid_coulomb_interaction_energy(donor, acceptor), abs=1e-3
    )
    # Dropping the smallest correction sites changes nothing at 1e-6.
    assert analytic == pytest.approx(
        analytic_coulomb_interaction_energy(donor, acceptor, site_tolerance=0.0),
        abs=1e-6,
    )


def test_unknown_coulomb_method_is_refused(water_dimer_ri):
    donor, acceptor = water_dimer_ri.split_hirshfeld([0, 1, 2])
    with pytest.raises(ValueError, match="unknown method"):
        coulomb_interaction_energy(donor, acceptor, method="quadrature")
