"""Structure factors and form factors from an RI density."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf.gto import ft_ao

from qpet import BOHR_TO_ANG, RIDensity, becke_grid, compute_S_vector
from qpet.scattering import (
    atomic_form_factors,
    debye_waller,
    electron_structure_factors,
    hkl_to_q,
    read_tsc,
    structure_factors,
    structure_factors_from_form_factors,
    write_tsc,
)


@pytest.fixture(scope="module")
def water(water_ri):
    mol, dm, auxmol, coeffs = water_ri
    return mol, dm, RIDensity(auxmol, coeffs)


@pytest.fixture(scope="module")
def q_vectors():
    """A handful of scattering vectors up to |q| ≈ 6 Å⁻¹ (sin θ/λ ≈ 0.5)."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(12, 3))
    q *= (np.linspace(0.3, 6.0, len(q)) / np.linalg.norm(q, axis=1))[:, None]
    return q


def test_zero_q_is_electron_count(water):
    _, _, rho = water
    f0 = structure_factors(rho, np.zeros((1, 3)))
    n = compute_S_vector(rho.auxmol) @ rho.coeffs
    assert f0.shape == (1,)
    assert f0[0] == pytest.approx(n, rel=1e-10)
    assert abs(f0[0].imag) < 1e-12


def test_analytic_matches_numerical_fourier_transform(water, q_vectors):
    _, _, rho = water
    grid = becke_grid(rho.auxmol, level=4)
    rho_w = rho.density(grid.points) * grid.weights
    numerical = np.exp(1j * (q_vectors @ grid.points.T)) @ rho_w
    analytic = structure_factors(rho, q_vectors)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-3, atol=1e-3)


def test_ri_fit_reproduces_density_matrix_structure_factors(water, q_vectors):
    """Coulomb-metric RI vs the exact AO-pair transform, low resolution."""
    mol, dm, rho = water
    pair = ft_ao.ft_aopair(mol, -q_vectors * BOHR_TO_ANG)  # (N, nao, nao)
    exact = np.einsum("nij,ij->n", pair, dm)
    fitted = structure_factors(rho, q_vectors)
    # The fit is exact in the Coulomb metric, i.e. weighted by 1/q²: tight
    # at low q, looser at high q where the core dominates.
    low = np.linalg.norm(q_vectors, axis=1) < 3.0
    np.testing.assert_allclose(fitted[low], exact[low], rtol=2e-2, atol=2e-2)
    assert np.abs(fitted - exact).max() < 0.2


@pytest.mark.parametrize("partition", ["ri", "hirshfeld"])
def test_form_factors_sum_to_structure_factor(water, q_vectors, partition):
    _, _, rho = water
    f = atomic_form_factors(rho, q_vectors, partition=partition)
    assert f.shape == (len(q_vectors), 3)
    total = structure_factors_from_form_factors(f, q_vectors, rho.positions)
    tol = 1e-12 if partition == "ri" else 2e-3
    np.testing.assert_allclose(total, structure_factors(rho, q_vectors), atol=tol)


def test_hirshfeld_form_factors_at_zero_q_are_hirshfeld_charges(water):
    _, _, rho = water
    f0 = atomic_form_factors(rho, np.zeros((1, 3)), partition="hirshfeld")[0]
    assert f0.real.sum() == pytest.approx(
        rho.coeffs @ compute_S_vector(rho.auxmol), abs=2e-3
    )
    assert f0.real[0] > 8.0  # oxygen pulls density from the hydrogens
    assert np.all(f0.real[1:] < 1.0)


def test_isotropic_debye_waller_scales_by_common_factor(water, q_vectors):
    _, _, rho = water
    u_iso = 0.02
    smeared = structure_factors(rho, q_vectors, adps=np.full(3, u_iso))
    rigid = structure_factors(rho, q_vectors)
    q2 = np.einsum("ni,ni->n", q_vectors, q_vectors)
    np.testing.assert_allclose(smeared, rigid * np.exp(-0.5 * u_iso * q2))


def test_anisotropic_debye_waller_matches_isotropic_for_scalar_tensor():
    q = np.array([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]])
    u = 0.03
    iso = debye_waller(q, np.array([u, u]), 2)
    aniso = debye_waller(q, np.array([u * np.eye(3)] * 2), 2)
    np.testing.assert_allclose(iso, aniso)
    with pytest.raises(ValueError, match="adps must have shape"):
        debye_waller(q, np.ones(3), 2)


def test_hkl_to_q_cubic_cell():
    q = hkl_to_q([[1, 0, 0], [1, 1, 0], [0, 0, 2]], 5.0 * np.eye(3))
    np.testing.assert_allclose(
        np.linalg.norm(q, axis=1),
        2 * np.pi / 5.0 * np.array([1.0, np.sqrt(2.0), 2.0]),
    )


def test_electron_structure_factors_mott_bethe(water, q_vectors):
    _, _, rho = water
    v = electron_structure_factors(rho, q_vectors)
    assert v.shape == (len(q_vectors),)
    assert np.all(np.isfinite(v))
    # Mott–Bethe by hand from the X-ray structure factor.
    nuclear = (
        rho.nuclear_charges[None, :] * np.exp(1j * (q_vectors @ rho.positions.T))
    ).sum(axis=1)
    q2 = np.einsum("ni,ni->n", q_vectors, q_vectors) * BOHR_TO_ANG**2
    expected = 4 * np.pi * (nuclear - structure_factors(rho, q_vectors)) / q2
    np.testing.assert_allclose(v, expected)
    assert np.isnan(electron_structure_factors(rho, np.zeros((1, 3)))[0])


def test_tsc_round_trip(tmp_path, water):
    _, _, rho = water
    hkl = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 3]])
    q = hkl_to_q(hkl, 8.0 * np.eye(3))
    f = atomic_form_factors(rho, q)
    path = tmp_path / "water.tsc"
    write_tsc(path, hkl, f, ["O1", "H1", "H2"], title="water")
    hkl_back, f_back, labels = read_tsc(path)
    assert labels == ["O1", "H1", "H2"]
    np.testing.assert_array_equal(hkl_back, hkl)
    np.testing.assert_allclose(f_back, f, rtol=1e-7, atol=1e-7)
    header = path.read_text().splitlines()[:5]
    assert header[0] == "TITLE: water"
    assert header[1] == "SYMM: expanded"
    assert header[4] == "DATA:"
    with pytest.raises(ValueError, match="does not match"):
        write_tsc(path, hkl, f, ["O1"])
