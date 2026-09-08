"""Tests for the density value objects (RIDensity, DMDensity)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from qpet import DMDensity, RIDensity, becke_grid


def test_ri_density_integrates_to_electron_count(water_ri):
    """
    The RI-fitted density must integrate to the molecular electron count:
    a wrong overall scale would silently bias every downstream ESP.
    """
    mol, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)

    grid = becke_grid(mol, level=3)
    n_elec = float(np.sum(grid.weights * rho.density(grid.points)))
    assert abs(n_elec - mol.nelectron) < 0.05


def test_ri_potential_matches_dm(water_ri):
    """
    RIDensity ESP must reproduce the DMDensity ESP within RI accuracy —
    i.e. both representations compute the same physical potential.
    """
    mol, dm, auxmol, coeffs = water_ri
    rho_dm = DMDensity(mol, dm)
    rho_ri = RIDensity(auxmol, coeffs)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-4.0, 4.0, size=(60, 3))
    v_dm = rho_dm.potential(pts)
    v_ri = rho_ri.potential(pts)
    assert np.max(np.abs(v_dm - v_ri)) < 2e-2


class TestConstructorValidation:
    """Shape mismatches must fail at construction, not at compute time."""

    def test_dm_rejects_wrong_shape(self, water_ri):
        mol = water_ri[0]
        with pytest.raises(ValueError, match="does not match"):
            DMDensity(mol, np.zeros((3, 4)))

    def test_dm_rejects_1d(self, water_ri):
        mol = water_ri[0]
        with pytest.raises(ValueError, match="does not match"):
            DMDensity(mol, np.zeros(5))

    def test_ri_rejects_wrong_size(self, water_ri):
        _, _, auxmol, _ = water_ri
        with pytest.raises(ValueError, match="does not match"):
            RIDensity(auxmol, np.zeros(auxmol.nao + 1))


def test_potential_batch_size_invariant(water_ri):
    """
    Changing ``batch_size`` only affects memory; numerical results must be
    identical (a batch-boundary bug would show as small per-batch drift).
    """
    mol, dm, auxmol, coeffs = water_ri
    rng = np.random.default_rng(1)
    pts = rng.uniform(-4.0, 4.0, size=(257, 3))  # not a multiple of any batch

    rho_dm = DMDensity(mol, dm)
    np.testing.assert_allclose(
        rho_dm.potential(pts, batch_size=64),
        rho_dm.potential(pts, batch_size=300),
        atol=1e-12,
    )

    rho_ri = RIDensity(auxmol, coeffs)
    np.testing.assert_allclose(
        rho_ri.potential(pts, batch_size=64),
        rho_ri.potential(pts, batch_size=300),
        atol=1e-12,
    )


def test_geometry_matches_mol(water_ri):
    """Value objects expose the geometry of the Mole they were built from."""
    mol, dm, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)
    np.testing.assert_allclose(rho.positions, mol.atom_coords(unit="Ang"))
    np.testing.assert_array_equal(rho.atomic_numbers, [8, 1, 1])

    rho_dm = DMDensity(mol, dm)
    np.testing.assert_allclose(rho_dm.positions, mol.atom_coords(unit="Ang"))


def test_normalize_charge_returns_exact_count(water_ri):
    """
    ``normalize_charge()`` must return a NEW density integrating to the exact
    electron count, leaving the original untouched (corrections are pure
    functions, not mutations).
    """
    _, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs * 1.03)  # visibly mis-normalized
    q_before = float(rho.charge())

    fixed = rho.normalize_charge()

    assert abs(float(fixed.charge()) + auxmol.nelectron) < 1e-9
    assert abs(float(rho.charge()) - q_before) == 0.0  # original unchanged
    assert fixed is not rho


def test_constrain_dipole_moves_dipole_and_preserves_charge(water_ri):
    """
    ``constrain_dipole`` must land on the requested electronic dipole (at the
    centre of nuclear charge) without changing the integrated charge.
    """
    _, _, auxmol, coeffs = water_ri
    z = auxmol.atom_charges().astype(float)
    r_cnc = z @ auxmol.atom_coords() / z.sum()

    rho = RIDensity(auxmol, coeffs)
    mu_target = rho.dipole(r_cnc) + np.array([0.05, -0.02, 0.01])
    q_before = float(rho.charge())

    fixed = rho.constrain_dipole(mu_target)

    np.testing.assert_allclose(fixed.dipole(r_cnc), mu_target, atol=1e-9)
    assert abs(float(fixed.charge()) - q_before) < 1e-9


def test_select_atoms_partition_is_additive(water_ri):
    """
    The core masking invariant: partitioning atoms into A and B and summing
    the two sub-densities' potentials must reproduce the full potential
    exactly. This is what lets a consumer attribute a complex's density to one
    partner while the other stays in the model's context — a broken slice or
    a leaked/duplicated nucleus would show here.
    """
    _, _, auxmol, coeffs = water_ri  # water: O=0, H=1, H=2
    rho = RIDensity(auxmol, coeffs)

    a = rho.select_atoms([0])  # oxygen
    b = rho.select_atoms([1, 2])  # both hydrogens

    rng = np.random.default_rng(0)
    pts = rng.uniform(-4.0, 4.0, size=(50, 3))

    np.testing.assert_allclose(
        a.potential(pts) + b.potential(pts), rho.potential(pts), atol=1e-10
    )
    np.testing.assert_allclose(
        a.density(pts) + b.density(pts), rho.density(pts), atol=1e-12
    )


def test_select_atoms_geometry_and_charge(water_ri):
    """
    A sub-density carries only the selected atoms' geometry, and its
    integrated electronic charge is the sum of the parts (partition additivity
    of the electron count).
    """
    _, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)

    o = rho.select_atoms([0])
    assert o.auxmol.natm == 1
    np.testing.assert_array_equal(o.atomic_numbers, [8])
    np.testing.assert_allclose(o.positions, rho.positions[[0]])

    h2 = rho.select_atoms([1, 2])
    assert abs(float(o.charge()) + float(h2.charge()) - float(rho.charge())) < 1e-10


def test_select_atoms_validates_input(water_ri):
    """Empty selection and out-of-range indices must fail loudly."""
    _, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)
    with pytest.raises(ValueError, match="empty"):
        rho.select_atoms([])
    with pytest.raises(IndexError, match="out of range"):
        rho.select_atoms([0, 3])


class TestEffectiveCorePotential:
    """
    Under an ECP the nucleus the density sees carries ``Z - n_core``. A
    nuclear term built from ``Z`` overshoots by ``n_core / r`` -- 28/r for
    iodine, several Hartree at any surface point -- so the two field classes
    must agree with each other and with the ECP calculation.
    """

    def test_element_and_nuclear_charge(self, hydrogen_iodide_ri):
        mol, dm, auxmol, coeffs = hydrogen_iodide_ri
        for rho in (DMDensity(mol, dm), RIDensity(auxmol, coeffs)):
            assert rho.nuclear_charges.tolist() == [25, 1]

    def test_ri_potential_matches_dm_with_ecp(self, hydrogen_iodide_ri):
        mol, dm, auxmol, coeffs = hydrogen_iodide_ri
        rho_dm = DMDensity(mol, dm)
        rho_ri = RIDensity(auxmol, coeffs)
        rng = np.random.default_rng(1)
        pts = rng.uniform(-4.0, 4.0, size=(60, 3)) + np.array([0.0, 0.0, 0.8])
        v_dm = rho_dm.potential(pts)
        v_ri = rho_ri.potential(pts, fast=False)
        assert np.max(np.abs(v_dm - v_ri)) < 2e-2

    def test_far_field_is_neutral(self, hydrogen_iodide_ri):
        """A neutral molecule's potential decays; with 53/r it would not."""
        _, _, auxmol, coeffs = hydrogen_iodide_ri
        rho = RIDensity(auxmol, coeffs)
        far = np.array([[30.0, 0.0, 0.0], [0.0, -30.0, 0.0], [0.0, 0.0, 40.0]])
        assert np.max(np.abs(rho.potential(far, fast=False))) < 1e-3

    def test_select_atoms_keeps_the_effective_nucleus(self, hydrogen_iodide_ri):
        _, _, auxmol, coeffs = hydrogen_iodide_ri
        rho = RIDensity(auxmol, coeffs)
        iodine = rho.select_atoms([0])
        assert iodine.atomic_numbers.tolist() == [53]
        assert iodine.nuclear_charges.tolist() == [25]


def test_with_periodic_images_makes_the_density_periodic(water_ri):
    """A molecule in a box: with images, ρ(p) equals ρ(p + lattice vector)."""
    _, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)
    cell = 5.0 * np.eye(3)
    periodic = rho.with_periodic_images(cell)
    assert periodic.auxmol.natm == 27 * auxmol.natm
    assert periodic.coeffs.shape == (27 * auxmol.nao,)
    pts = np.array([[0.3, 0.2, 0.1], [-0.6, 0.4, 0.5]])
    np.testing.assert_allclose(
        periodic.density(pts), periodic.density(pts + cell[0]), rtol=1e-6, atol=1e-8
    )
    # One cell alone is not periodic: a point near the face sees the missing neighbours.
    face = np.array([[2.4, 0.0, 0.0]])
    assert periodic.density(face)[0] > rho.density(face)[0]


def test_with_periodic_images_radius_selects_atoms(water_ri):
    _, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)
    cell = 5.0 * np.eye(3)
    local = rho.with_periodic_images(cell, centre=np.zeros(3), radius=1.0)
    assert 1 <= local.auxmol.natm < 27 * auxmol.natm
    # The oxygen at the origin is kept, and the density there is unchanged.
    origin = np.zeros((1, 3))
    np.testing.assert_allclose(local.density(origin), rho.density(origin), rtol=1e-6)

    message = "centre and radius must be given together."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        rho.with_periodic_images(cell, radius=1.0)
