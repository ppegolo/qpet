"""Tests for RIDensity, DMDensity and Ensemble."""

from __future__ import annotations

import numpy as np
import pytest
from _dummy_model import to_tensormap
from pyscf import gto, scf

from qpet import (
    DMDensity,
    Ensemble,
    RIDensity,
    becke_grid,
    fit_coefficients,
    nmae,
    nuclear_potential,
    tensormap_coefficients,
)


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


@pytest.mark.parametrize("n_points", [20, 1200])
def test_dm_potential_is_nuclei_minus_int1e_grids(water_ri, n_points):
    mol, dm, _, _ = water_ri
    points = _shell(n_points, 3.0)
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


def test_quadrupole_matches_the_grid_integral(water_ri):
    mol, dm, auxmol, coeffs = water_ri
    origin = np.array([0.3, -0.2, 0.5])
    points, weights = becke_grid(mol, level=5)
    r, nuc = points - origin, mol.atom_coords() - origin
    nuclei = np.einsum("i,ix,iy->xy", mol.atom_charges(), nuc, nuc)
    for rho in (DMDensity(mol, dm), RIDensity(auxmol, coeffs)):
        electrons = np.einsum("p,px,py->xy", weights * rho.density(points), r, r)
        np.testing.assert_allclose(
            rho.quadrupole(origin), nuclei - electrons, atol=1e-5
        )


def test_neutral_ecp_molecule_has_an_origin_independent_dipole(hydrogen_iodide_ri):
    """26 electrons against the 25 + 1 nuclear charges the ECP leaves, not 53 + 1."""
    _, _, auxmol, coeffs = hydrogen_iodide_ri
    rho = RIDensity(auxmol, coeffs)
    np.testing.assert_allclose(
        rho.dipole(), rho.dipole(np.array([1.0, -2.0, 0.5])), atol=0.05
    )


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


def test_tensormap_round_trips(hydrogen_iodide_ri):
    _, _, auxmol, coeffs = hydrogen_iodide_ri
    tensor = RIDensity(auxmol, coeffs).to_tensormap()
    assert set(tensor.keys.column("atom_type").tolist()) == {1, 53}
    np.testing.assert_array_equal(
        RIDensity.from_tensormap(tensor, auxmol).coeffs, coeffs
    )


def test_tensormap_coefficients_of_ensemble_members(water_ri):
    _, _, auxmol, coeffs = water_ri
    members = coeffs * np.arange(1, 4)[:, None]
    (out,) = tensormap_coefficients(to_tensormap(auxmol, members), n_members=3)
    np.testing.assert_array_equal(out, members)


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


def test_potential_at_a_nucleus_excludes_it(water_ri):
    mol, dm, auxmol, coeffs = water_ri
    r, z = mol.atom_coords(), mol.atom_charges()
    others = z[1:] @ (1 / np.linalg.norm(r[1:] - r[0], axis=1))
    with mol.with_rinv_origin(r[0]):
        electrons = np.einsum("ij,ji", mol.intor("int1e_rinv"), dm)
    v_dm = DMDensity(mol, dm).potential(r[:1], exclude=0)[0]
    assert v_dm == pytest.approx(others - electrons, abs=1e-10)
    point = gto.fakemol_for_charges(r[:1])
    electrons = coeffs @ gto.mole.intor_cross("int2c2e", auxmol, point).ravel()
    rho = RIDensity(auxmol, coeffs)
    assert rho.potential(r[:1], exclude=0)[0] == pytest.approx(
        others - electrons, abs=1e-10
    )
    members = Ensemble(auxmol, np.stack([coeffs, coeffs]))
    np.testing.assert_allclose(
        members.potential(r[:1], exclude=0)[:, 0], others - electrons, atol=1e-10
    )
    with np.errstate(divide="ignore"):
        assert np.isinf(rho.potential(r[:1])[0])


def test_density_matches_unscreened_basis_values(water_ri):
    mol, dm, auxmol, coeffs = water_ri
    points, _ = becke_grid(mol, level=1)
    ao = mol.eval_gto("GTOval", points)
    np.testing.assert_allclose(
        DMDensity(mol, dm).density(points),
        np.einsum("pi,ij,pj->p", ao, dm, ao),
        atol=1e-10,
    )
    aux = auxmol.eval_gto("GTOval", points)
    np.testing.assert_allclose(
        RIDensity(auxmol, coeffs).density(points), aux @ coeffs, atol=1e-10
    )
    members = np.stack([coeffs, 2 * coeffs])
    np.testing.assert_allclose(
        Ensemble(auxmol, members).density(points), members @ aux.T, atol=1e-10
    )


def test_nmae_of_a_density_against_zero_is_100(water_ri):
    mol, dm, _, _ = water_ri
    points, weights = becke_grid(mol)
    rho = DMDensity(mol, dm).density(points)
    assert nmae(rho, 0 * rho, weights, mol.nelectron) == pytest.approx(100, abs=1e-3)
    assert nmae(rho, 0.99 * rho, weights, mol.nelectron) == pytest.approx(1, abs=1e-5)


def test_fit_coefficients_in_slices_and_batches(water_ri, monkeypatch):
    mol, dm, auxmol, coeffs = water_ri
    monkeypatch.setattr("qpet.density._FIT_FLOATS", 1)
    np.testing.assert_allclose(fit_coefficients(mol, auxmol, dm), coeffs, atol=1e-10)
    np.testing.assert_allclose(
        fit_coefficients(mol, auxmol, np.stack([dm, 2 * dm])),
        np.stack([coeffs, 2 * coeffs]),
        atol=1e-10,
    )


def test_coulomb_distance_is_the_coulomb_norm_of_the_difference(water_ri):
    _, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)
    d = 0.1 * coeffs
    assert rho.coulomb_distance(RIDensity(auxmol, 0.9 * coeffs)) == pytest.approx(
        d @ auxmol.intor("int2c2e") @ d
    )
