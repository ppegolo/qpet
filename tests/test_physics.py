"""Physics-invariance tests for the scalar-field calculators.

These exercise scientific invariants: far-field decay, the bare-nucleus
Coulomb limit, an exact cross-check against PySCF's native grid integral, and
the nuclear-singularity clamp.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto, scf

from qpet import BOHR_TO_ANG, DMDensity
from qpet.fields import _DIST_FLOOR, nuclear_potential


@pytest.fixture(scope="module")
def water_dm():
    """Neutral water in sto-3g with an SCF density matrix."""
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        basis="sto-3g",
        verbose=0,
    ).build()
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()
    return mol, dm


def test_far_field_decay_is_fast(water_dm):
    """
    Neutral water has zero monopole; the dipole term dominates at large r.
    The ESP magnitude should fall at least as fast as 1/r² (ratio < 0.4
    when doubling r).
    """
    mol, dm = water_dm
    rho = DMDensity(mol, dm)

    # Sample along +z (dipole direction ≠ 0).
    pts = np.array([[0, 0, r] for r in (5.0, 10.0, 20.0)])
    v = rho.potential(pts)

    assert abs(v[1]) / abs(v[0]) < 0.4, f"|V(10)|/|V(5)| = {abs(v[1]) / abs(v[0]):.3f}"
    assert abs(v[2]) / abs(v[1]) < 0.4, f"|V(20)|/|V(10)| = {abs(v[2]) / abs(v[1]):.3f}"


def test_bare_nucleus_reproduces_coulomb():
    """
    Setting ``dm = 0`` strips the electronic contribution, so
    ``compute_potential`` reduces to the bare nuclear sum ``Σ Z/r``. Check
    V > 0 and exact 1/r scaling for a single nucleus at the origin.
    """
    # spin=1 for neutral Li (3 electrons); the electronic contribution is
    # zeroed out by ``dm=0`` anyway, so charge/spin only affect mol construction.
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0).build()
    dm = np.zeros((mol.nao, mol.nao))

    rho = DMDensity(mol, dm)

    r1, r2 = 3.0, 6.0  # Angstrom
    pts = np.array([[r1, 0, 0], [r2, 0, 0]])
    v = rho.potential(pts)

    assert v[0] > 0 and v[1] > 0
    # Nuclear-only: V = Z/r exactly. Check V(r1)/V(r2) = r2/r1.
    ratio = v[0] / v[1]
    expected = r2 / r1
    assert abs(ratio - expected) / expected < 1e-6


def test_dm_matches_pyscf_int1e_grids(water_dm):
    """
    ``DMDensity.potential - V_nuc`` is literally
    ``-tr(D · int1e_grids)``. Cross-check both paths agree to machine
    precision.
    """
    mol, dm = water_dm
    rho = DMDensity(mol, dm)

    rng = np.random.default_rng(42)
    pts_ang = rng.uniform(-5, 5, size=(50, 3))
    pts_bohr = pts_ang / BOHR_TO_ANG

    v_total = rho.potential(pts_ang)
    v_nuc = nuclear_potential(mol.atom_coords(), rho.atomic_numbers, pts_bohr)
    v_ele_calc = v_nuc - v_total  # recovered electronic contribution

    ints = mol.intor("int1e_grids", grids=pts_bohr)
    v_ele_ref = np.einsum("ij,kij->k", dm, ints)

    assert np.max(np.abs(v_ele_calc - v_ele_ref)) < 1e-10


def test_nuclear_singularity_is_clamped():
    """
    Evaluating V at a point inside ``_DIST_FLOOR`` must stay finite and
    bounded by ``Z / _DIST_FLOOR``.
    """
    positions_bohr = np.array([[0.0, 0.0, 0.0]])
    charges = np.array([8])  # oxygen
    pts_bohr = np.array([[0.001, 0.0, 0.0]])  # inside floor

    v = nuclear_potential(positions_bohr, charges, pts_bohr)
    assert np.isfinite(v[0])
    assert v[0] <= 8 / _DIST_FLOOR + 1e-12
