"""Hirshfeld fragment partition: additivity, weights, convergence, geometry."""

import re

import numpy as np
import pytest
from pyscf import gto, scf

from qpet import BOHR_TO_ANG, RIDensity
from qpet.partition import free_atom_density, hirshfeld_weights

_AUX = "def2-universal-jfit"


@pytest.fixture(scope="module")
def water_dimer_ri():
    """
    Water donating a hydrogen bond to ammonia (O-H...N along x, N 2.9 A from O),
    RHF/sto-3g, Coulomb-fitted: atoms 0-2 are the water, 3-6 the ammonia. The
    pair is deliberately asymmetric so the two partitions disagree on the
    fragment charges (a mirror-symmetric dimer would hide a bookkeeping error).
    """
    mol = gto.M(
        atom=(
            "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0; "
            "N 2.9 0 0; H 3.3 0.9 0; H 3.3 -0.5 0.8; H 3.3 -0.5 -0.8"
        ),
        basis="sto-3g",
        verbose=0,
    )
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()
    return RIDensity.from_dm(mol, dm, _AUX)


@pytest.fixture(scope="module")
def surface_points():
    rng = np.random.default_rng(0)
    # Around the contact, outside both molecules' cores.
    return rng.uniform([-1.0, -2.5, 1.8], [4.2, 3.0, 3.5], size=(40, 3))


def test_free_atom_density_integrates_to_the_electron_count():
    r = np.linspace(1e-3, 15.0, 20001)
    for z, n_expected in (
        (1, 1.0),
        (8, 8.0),
        (53, 25.0),
    ):  # iodine: def2 ECP, 25 valence
        rho = free_atom_density(z, r)
        n = np.trapezoid(4 * np.pi * r**2 * rho, r)
        assert n == pytest.approx(n_expected, abs=1e-2)


def test_hirshfeld_weights_are_a_partition_of_unity(water_dimer_ri):
    auxmol = water_dimer_ri.auxmol
    pts = np.random.default_rng(1).uniform(-3, 7, size=(200, 3)) / BOHR_TO_ANG
    numbers = np.array([auxmol.atom_charge(i) for i in range(auxmol.natm)])
    in_a = np.array([True, True, True, False, False, False, False])
    w_a = hirshfeld_weights(pts, auxmol.atom_coords(), numbers, in_a)
    w_b = hirshfeld_weights(pts, auxmol.atom_coords(), numbers, ~in_a)
    assert np.all(w_a >= 0) and np.all(w_a <= 1)
    np.testing.assert_allclose(w_a + w_b, 1.0, atol=1e-12)
    # Near an oxygen of A the weight of A is essentially one.
    near_o = np.array([[0.05, 0.02, 0.01]]) / BOHR_TO_ANG
    assert hirshfeld_weights(near_o, auxmol.atom_coords(), numbers, in_a)[0] > 0.999


def test_fragments_sum_exactly_to_the_total(water_dimer_ri, surface_points):
    a, b = water_dimer_ri.split_hirshfeld([0, 1, 2])
    total = water_dimer_ri.potential(surface_points)
    np.testing.assert_allclose(
        a.potential(surface_points) + b.potential(surface_points), total, atol=1e-10
    )
    assert a.electrons + b.electrons == pytest.approx(
        -float(np.sum(water_dimer_ri.charge())), abs=1e-9
    )
    assert a.charge(include_nuclear=True) + b.charge(
        include_nuclear=True
    ) == pytest.approx(
        float(np.sum(water_dimer_ri.charge(include_nuclear=True))), abs=1e-9
    )
    pts = np.array([[0.3, 0.2, 0.1], [2.7, 0.1, 0.0]])
    np.testing.assert_allclose(
        a.density(pts) + b.density(pts), water_dimer_ri.density(pts), rtol=1e-12
    )


def test_hirshfeld_differs_from_ri_split_at_the_contact(water_dimer_ri, surface_points):
    a, _ = water_dimer_ri.split_hirshfeld([0, 1, 2])
    ri = water_dimer_ri.select_atoms([0, 1, 2])
    diff = a.potential(surface_points) - ri.potential(surface_points)
    # A hydrogen-bonded dimer shares density: the two bookkeepings disagree by
    # a visible, but not wild, amount.
    assert 1e-4 < np.abs(diff).max() < 5e-2
    assert abs(a.split.electron_transfer) > 1e-3


def test_correction_converges_with_the_grid(water_dimer_ri, surface_points):
    coarse, _ = water_dimer_ri.split_hirshfeld([0, 1, 2], grid_level=1)
    fine, _ = water_dimer_ri.split_hirshfeld([0, 1, 2], grid_level=4)
    v_coarse = coarse.potential(surface_points)
    v_fine = fine.potential(surface_points)
    assert np.abs(v_coarse - v_fine).max() < 5e-4
    assert coarse.electrons == pytest.approx(fine.electrons, abs=2e-3)


def test_fragment_geometry(water_dimer_ri):
    _, b = water_dimer_ri.split_hirshfeld([0, 1, 2], [3, 4, 5, 6])
    np.testing.assert_array_equal(b.atomic_numbers, [7, 1, 1, 1])


def test_far_apart_fragments_reduce_to_the_ri_split():
    mol = gto.M(
        atom=(
            "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0; "
            "O 30 0 0; H 30.757 0.587 0; H 29.243 0.587 0"
        ),
        basis="sto-3g",
        verbose=0,
    )
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()
    rho = RIDensity.from_dm(mol, dm, _AUX)
    a, b = rho.split_hirshfeld([0, 1, 2])
    assert len(a.split.cell_atoms) == 0
    pts = np.array([[1.5, 1.5, 1.0], [28.0, 1.0, 1.0]])
    np.testing.assert_allclose(
        a.potential(pts), rho.select_atoms([0, 1, 2]).potential(pts)
    )
    assert a.split.electron_transfer == 0.0


_NOT_A_PARTITION = (
    "the two fragments must partition the complex: disjoint and covering all 7 "
    "atoms (got {} + {})."
)


@pytest.mark.parametrize(
    "atoms_a, atoms_b, message",
    [
        ([0, 1], [3, 4, 5, 6], _NOT_A_PARTITION.format(2, 4)),
        ([0, 1, 2], [2, 3, 4, 5, 6], _NOT_A_PARTITION.format(3, 5)),
        ([0, 1, 2, 3, 4, 5, 6], None, "both fragments need at least one atom."),
    ],
)
def test_partition_must_cover_the_complex(water_dimer_ri, atoms_a, atoms_b, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        water_dimer_ri.split_hirshfeld(atoms_a, atoms_b)
