"""Structure factors and form factors from an RI density."""

from __future__ import annotations

from pathlib import Path

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


def test_electron_atomic_form_factors_sum_to_total(water, q_vectors):
    """The per-atom potential form factors, summed with the nuclear phases, are the total."""
    from qpet.scattering import (
        electron_atomic_form_factors,
        structure_factors_from_form_factors,
    )

    _, _, rho = water
    fe = electron_atomic_form_factors(rho, q_vectors)
    assert fe.shape == (len(q_vectors), rho.positions.shape[0])
    summed = structure_factors_from_form_factors(fe, q_vectors, rho.positions)
    np.testing.assert_allclose(summed, electron_structure_factors(rho, q_vectors))


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


def test_spherical_atom_form_factor_is_electron_count_at_zero_q():
    from qpet.scattering import spherical_atom_form_factor

    for z in (1, 6, 8):
        assert spherical_atom_form_factor(z, np.zeros(1))[0] == pytest.approx(
            z, rel=2e-3
        )
    f = spherical_atom_form_factor(8, np.array([0.0, 5.0, 10.0]))
    assert f[0] > f[1] > f[2] > 0


def test_independent_atom_structure_factors_match_promolecule_transform(
    water, q_vectors
):
    from qpet.partition import free_atom_density
    from qpet.scattering import independent_atom_structure_factors

    _, _, rho = water
    numbers, positions = rho.atomic_numbers, rho.positions
    analytic = independent_atom_structure_factors(numbers, positions, q_vectors)
    # Numerical transform of the promolecule on a Becke grid.
    grid = becke_grid(rho.auxmol, level=4)
    pts_bohr = grid.points / BOHR_TO_ANG
    promol = np.zeros(len(grid.points))
    for z, pos in zip(numbers, positions, strict=True):
        promol += free_atom_density(
            int(z), np.linalg.norm(pts_bohr - pos / BOHR_TO_ANG, axis=1)
        )
    numerical = np.exp(1j * (q_vectors @ grid.points.T)) @ (promol * grid.weights)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-2, atol=2e-2)
    assert analytic[np.linalg.norm(q_vectors, axis=1).argmin()].real == pytest.approx(
        10.0, rel=5e-2
    )


def test_independent_atom_structure_factors_debye_waller(water, q_vectors):
    from qpet.scattering import independent_atom_structure_factors

    _, _, rho = water
    rigid = independent_atom_structure_factors(
        rho.atomic_numbers, rho.positions, q_vectors
    )
    smeared = independent_atom_structure_factors(
        rho.atomic_numbers, rho.positions, q_vectors, adps=np.full(3, 0.02)
    )
    q2 = np.einsum("ni,ni->n", q_vectors, q_vectors)
    np.testing.assert_allclose(smeared, rigid * np.exp(-0.5 * 0.02 * q2))


def test_cell_from_parameters_reproduces_lengths_and_angles():
    from qpet.scattering import cell_from_parameters

    cell = cell_from_parameters(4.726, 9.369, 10.239, 88.796, 85.14, 78.90)
    lengths = np.linalg.norm(cell, axis=1)
    np.testing.assert_allclose(lengths, [4.726, 9.369, 10.239])
    cosines = [cell[1] @ cell[2], cell[0] @ cell[2], cell[0] @ cell[1]] / np.array(
        [lengths[1] * lengths[2], lengths[0] * lengths[2], lengths[0] * lengths[1]]
    )
    np.testing.assert_allclose(np.degrees(np.arccos(cosines)), [88.796, 85.14, 78.90])


def test_hkl_to_q_triclinic_cell_phase():
    """q·r equals 2π h·frac for a triclinic cell, with reciprocal vectors from cross products."""
    from qpet.scattering import cell_from_parameters

    cell = cell_from_parameters(4.726, 9.369, 10.239, 88.796, 85.14, 78.90)
    hkl = np.array([[1, 0, 0], [0, 2, -1], [3, -1, 2], [-2, 4, 5]], dtype=float)
    frac = np.array([[0.1, 0.7, -0.3], [0.45, 0.2, 0.9]])
    q = hkl_to_q(hkl, cell)
    r = frac @ cell
    np.testing.assert_allclose(q @ r.T, 2 * np.pi * hkl @ frac.T, atol=1e-12)
    a, b, c = cell
    vol = a @ np.cross(b, c)
    recip = np.array([np.cross(b, c), np.cross(c, a), np.cross(a, b)]) / vol
    np.testing.assert_allclose(q, 2 * np.pi * hkl @ recip, atol=1e-12)


def test_adp_cif_to_cartesian_matches_reciprocal_axis_exponent():
    """exp(-2π² Σ h_i h_j a*_i a*_j U^ij) == exp(-½ qᵀ U_cart q) on a triclinic cell."""
    from qpet.scattering import adp_cif_to_cartesian, cell_from_parameters, debye_waller

    cell = cell_from_parameters(4.726, 9.369, 10.239, 88.796, 85.14, 78.90)
    u_cif = np.array([[0.0313, 0.0237, -0.0041], [0.0237, 0.0670, -0.0073], [-0.0041, -0.0073, 0.0174]])
    hkl = np.array([[1, 0, 0], [0, 2, -1], [3, -1, 2], [-2, 4, 5]], dtype=float)
    a, b, c = cell
    recip = np.linalg.norm([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=1) / (a @ np.cross(b, c))
    reference = np.exp(-2 * np.pi**2 * np.einsum("ni,ij,nj->n", hkl * recip, u_cif, hkl * recip))
    q = hkl_to_q(hkl, cell)
    ours = debye_waller(q, adp_cif_to_cartesian(u_cif, cell)[None], 1)[:, 0]
    np.testing.assert_allclose(ours, reference, rtol=1e-12)
    # Orthorhombic cell: identity.
    ortho = np.diag([5.93, 12.26, 5.79])
    np.testing.assert_allclose(adp_cif_to_cartesian(u_cif, ortho), u_cif)


def test_predict_coefficients_torch_matches_the_numpy_path():
    """Same coefficients as the numpy path, and a gradient to the positions."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("metatomic")
    model_path = Path(__file__).resolve().parents[3] / "models" / "pet-omol25-bio.pt"
    if not model_path.exists():
        pytest.skip("no density model checkpoint in the workspace")

    from qpet import DensityModel, build_system

    numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    model = DensityModel(
        str(model_path),
        head="mtt::rho_c_jfit_coulomb",
        aux_basis="def2-universal-jfit",
        basis="def2-svp",
        backend="metatomic",
        device="cpu",
    )
    reference = model.predict_systems(build_system(numbers, positions))[0].coeffs
    tensor_positions = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    coefficients = model.predict_coefficients_torch(numbers, tensor_positions)
    assert coefficients.shape == reference.shape
    assert np.abs(coefficients.detach().numpy() - reference).max() < 1e-5
    gradient, = torch.autograd.grad(coefficients.sum(), tensor_positions)
    assert gradient.shape == tensor_positions.shape
    assert float(gradient.abs().max()) > 0.0
