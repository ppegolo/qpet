"""Tests for RIDensity, DMDensity and Ensemble."""

from __future__ import annotations

import numpy as np
import pytest
from _dummy_model import to_tensormap
from pyscf import gto, scf

from qpet import DMDensity, Ensemble, RIDensity
from qpet.density import _DIST_FLOOR, nuclear_potential


@pytest.fixture(scope="module")
def oh_radical(tmp_path_factory):
    """UHF doublet OH, run with a checkpoint file. Returns ``(mol, mf, chk)``."""
    chk = tmp_path_factory.mktemp("chk") / "oh.chk"
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="6-31g*", spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.chkfile = str(chk)
    mf.run()
    return mol, mf, chk


def _shell(n: int, radius: float) -> np.ndarray:
    """``n`` points on a sphere of ``radius`` bohr."""
    v = np.random.default_rng(0).normal(size=(n, 3))
    return radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def test_fit_reaches_the_target(water_ri):
    mol, dm, _, coeffs = water_ri
    exact = RIDensity.fit(mol, dm, "def2-svp-jkfit", target=10.0)
    assert exact.integral() == pytest.approx(10.0, abs=1e-10)
    assert abs(exact.coeffs - coeffs).max() < 1e-2 * abs(coeffs).max()


def test_ri_potential_is_close_to_the_density_matrix(water_ri):
    """Outside the molecule the Coulomb-metric fit reproduces the potential."""
    mol, dm, auxmol, coeffs = water_ri
    points = _shell(50, 6.0)
    v_dm = DMDensity(mol, dm).potential(points)
    v_ri = RIDensity(auxmol, coeffs).potential(points, fast=False)
    assert np.abs(v_ri - v_dm).max() < 1e-3


def test_dm_potential_is_nuclei_minus_int1e_grids(water_ri):
    mol, dm, _, _ = water_ri
    points = _shell(20, 3.0)
    electrons = np.einsum("pij,ij->p", mol.intor("int1e_grids", grids=points), dm)
    nuclei = nuclear_potential(mol.atom_coords(), mol.atom_charges(), points)
    np.testing.assert_allclose(
        DMDensity(mol, dm).potential(points), nuclei - electrons, atol=1e-12
    )
    np.testing.assert_allclose(
        DMDensity(mol, dm, nuclei=False).potential(points), -electrons, atol=1e-12
    )


def test_dm_dipole_matches_pyscf(water_ri):
    mol, dm, auxmol, coeffs = water_ri
    expected = scf.hf.dip_moment(mol, dm, unit="AU", verbose=0)
    np.testing.assert_allclose(DMDensity(mol, dm).dipole(), expected, atol=1e-10)
    np.testing.assert_allclose(RIDensity(auxmol, coeffs).dipole(), expected, atol=1e-2)


def test_select_atoms_partitions_the_potential(water_ri):
    _, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)
    points = _shell(30, 4.0)
    parts = rho.select_atoms([0]).potential(points) + rho.select_atoms(
        [1, 2]
    ).potential(points)
    np.testing.assert_allclose(parts, rho.potential(points), atol=1e-10)


def test_populations_are_the_integrals_of_select_atoms(water_ri):
    _, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)
    expected = [rho.select_atoms([a]).integral() for a in range(auxmol.natm)]
    np.testing.assert_allclose(rho.populations(), expected, atol=1e-12)


def test_from_chk_gives_total_and_spin_density(oh_radical):
    mol, mf, chk = oh_radical
    total, spin = DMDensity.from_chk(chk)
    alpha, beta = mf.make_rdm1()
    np.testing.assert_allclose(total.dm, alpha + beta, atol=1e-12)
    assert total.integral() == pytest.approx(mol.nelectron)
    assert spin.integral() == pytest.approx(1.0)
    assert total.nuclei and not spin.nuclei


def test_from_tensormap_round_trips(water_ri):
    _, _, auxmol, coeffs = water_ri
    np.testing.assert_array_equal(
        RIDensity.from_tensormap(to_tensormap(auxmol, coeffs), auxmol).coeffs, coeffs
    )


def test_ensemble_members_are_densities(water_ri):
    _, _, auxmol, coeffs = water_ri
    ens = Ensemble(auxmol, coeffs * np.array([0.99, 1.0, 1.01])[:, None])
    points = _shell(10, 5.0)
    v = ens.potential(points)
    np.testing.assert_allclose(v[2], ens[2].potential(points, fast=False), atol=1e-12)
    assert ens.std(lambda r: r.potential(points, fast=False)).shape == (10,)


def test_bare_nucleus_is_coulomb():
    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    points = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 4.0]])
    v = DMDensity(mol, np.zeros((mol.nao, mol.nao))).potential(points)
    np.testing.assert_allclose(v, 2.0 / np.linalg.norm(points, axis=1))
    assert (
        DMDensity(mol, np.zeros((1, 1))).potential(np.zeros((1, 3)))[0]
        == 2.0 / _DIST_FLOOR
    )
