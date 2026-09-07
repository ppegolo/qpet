"""Computed interaction channels: Coulomb energy, Hirshfeld volumes, dispersion."""

import re

import numpy as np
import pytest
from pyscf import gto, scf

from qpet import RIDensity
from qpet.conversions import BOHR_TO_ANG
from qpet.interactions import (
    analytic_coulomb_interaction_energy,
    coulomb_interaction_energy,
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


@pytest.fixture(scope="module")
def neon_dimers():
    """Ne2 at 4 and 6 A, RHF/def2-svp and Coulomb-fitted, by separation."""
    return {distance: _ri(f"Ne 0 0 0; Ne {distance} 0 0") for distance in (4.0, 6.0)}


def test_coulomb_energy_of_separated_ions_is_the_point_charge_limit():
    separation = 15.0
    lithium = _ri("Li 0 0 0", charge=1)
    fluoride = _ri(f"F {separation} 0 0", charge=-1)
    energy = coulomb_interaction_energy(lithium, fluoride)
    # Fitted, not formal, charges: the RI fit of an ion is not exactly integral.
    q_li = float(np.sum(lithium.charge(include_nuclear=True)))
    q_f = float(np.sum(fluoride.charge(include_nuclear=True)))
    # 1e-4: the grid is truncated at ``support_radius``.
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
    # One only to the accuracy of the RI fit's r^3 moment, which weights its tail.
    assert ratio[0] == pytest.approx(1.0, abs=0.1)


def test_bound_atoms_lose_volume(water_dimer_ri):
    ratios = hirshfeld_volume_ratios(water_dimer_ri, [0, 1, 2])
    # Hydrogens donate their density to oxygen and shrink the most.
    assert ratios[1] < 0.8 and ratios[2] < 0.8
    assert ratios[0] > ratios[1] and ratios[0] > ratios[2]
    np.testing.assert_allclose(
        hirshfeld_volume_ratios(water_dimer_ri, [2, 0, 1]), ratios[[2, 0, 1]]
    )


def test_dispersion_is_negative_and_decays_with_separation(neon_dimers):
    near = hirshfeld_dispersion_energy(neon_dimers[4.0], [0])
    far = hirshfeld_dispersion_energy(neon_dimers[6.0], [0])
    assert near < 0.0 and far < 0.0
    # Undamped at these separations, so the ratio is the R^-6 one to ~10 %.
    assert near / far == pytest.approx((6.0 / 4.0) ** 6, rel=0.1)


def test_dispersion_ignores_atoms_beyond_the_environment_radius(neon_dimers):
    density = neon_dimers[4.0]
    assert hirshfeld_dispersion_energy(density, [0], environment_radius=2.0) == 0.0


def test_analytic_and_grid_agree_on_an_ri_split(water_dimer_ri):
    """With no Hirshfeld correction every analytic term is exact; what is left
    is the grid's own error."""
    donor = water_dimer_ri.select_atoms([0, 1, 2])
    acceptor = water_dimer_ri.select_atoms([3, 4, 5])
    analytic = analytic_coulomb_interaction_energy(donor, acceptor)
    assert analytic == pytest.approx(
        grid_coulomb_interaction_energy(donor, acceptor), abs=5e-5
    )


def test_analytic_and_grid_agree_on_a_hirshfeld_split(water_dimer_ri):
    """To the quadrature error of the grid, about 1e-3 Hartree between levels."""
    donor, acceptor = water_dimer_ri.split_hirshfeld([0, 1, 2])
    analytic = analytic_coulomb_interaction_energy(donor, acceptor)
    assert analytic == pytest.approx(
        grid_coulomb_interaction_energy(donor, acceptor), abs=1e-3
    )


def test_unknown_coulomb_method_is_refused(water_dimer_ri):
    donor, acceptor = water_dimer_ri.split_hirshfeld([0, 1, 2])
    message = "unknown method 'quadrature'; use 'analytic' or 'grid'."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        coulomb_interaction_energy(donor, acceptor, method="quadrature")
