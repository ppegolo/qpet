"""Tests for RI-density observables (Ghost-Trick multipoles, S vector)."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import df, gto

from qpet.moments import (
    apply_dipole_correction,
    compute_S_vector,
    get_ri_charge,
    get_ri_dipole,
)


def test_ghost_charge_matches_analytic_s_vector(water_ri):
    """
    The package has two routes to ``∫ χ_i(r) dr``: the analytic per-shell
    formula (``compute_S_vector``, used for charge normalization) and the
    ghost-orbital cross integral (``get_ri_charge``). They implement the same
    physics and must not drift apart.
    """
    _, _, auxmol, coeffs = water_ri
    n_elec_analytic = float(compute_S_vector(auxmol) @ coeffs)
    q_ghost = float(get_ri_charge(auxmol, coeffs))
    assert abs(q_ghost - (-n_elec_analytic)) < 1e-8


def test_get_ri_charge_recovers_nelectron(water_ri):
    """
    ``get_ri_charge`` on a Coulomb-fit RI density must recover ``-N_electrons``
    (electronic charge convention). A sign flip or unit error would surface
    as a factor-of-many discrepancy.
    """
    mol, _, auxmol, coeffs = water_ri
    q_el = float(get_ri_charge(auxmol, coeffs))
    # 6-31g* / Coulomb fit: ~1e-3 relative accuracy on integrated charge.
    assert abs(q_el - (-mol.nelectron)) < 0.05, (
        f"got q_el={q_el}, expected ≈ {-mol.nelectron}"
    )


def test_apply_dipole_correction_fixes_dipole_and_preserves_charge(water_ri):
    """
    The correction must (a) shift the dipole to the target and (b) leave the
    input's integrated charge unchanged. "Charge-neutral" is a property of the
    correction itself: it doesn't undo the perturbation's charge shift, but it
    adds none of its own.
    """
    _, _, auxmol, coeffs = water_ri
    R_cnc = auxmol.atom_charges() @ auxmol.atom_coords() / auxmol.atom_charges().sum()

    mu_target = get_ri_dipole(auxmol, coeffs, R_cnc)

    # Perturb the coefficients enough to shift the dipole appreciably.
    rng = np.random.default_rng(0)
    coeffs_perturbed = coeffs + 0.01 * rng.standard_normal(coeffs.shape)
    mu_perturbed = get_ri_dipole(auxmol, coeffs_perturbed, R_cnc)
    delta_mu = mu_target - mu_perturbed
    assert np.linalg.norm(delta_mu) > 1e-3, "perturbation too small to be a real test"

    q_before = float(get_ri_charge(auxmol, coeffs_perturbed))
    coeffs_corrected = apply_dipole_correction(auxmol, coeffs_perturbed, delta_mu)

    mu_corrected = get_ri_dipole(auxmol, coeffs_corrected, R_cnc)
    q_after = float(get_ri_charge(auxmol, coeffs_corrected))

    np.testing.assert_allclose(mu_corrected, mu_target, atol=1e-9)
    assert abs(q_after - q_before) < 1e-9, (
        f"correction shifted charge by {q_after - q_before:.2e}"
    )


def test_apply_dipole_correction_linear_molecule():
    """
    Linear N₂ on the z-axis: the Coulomb-metric Gram matrix in the correction
    can be ill-conditioned. ``lstsq`` should return a finite, charge-neutral
    correction without crashing (``np.linalg.solve`` would raise on a truly
    singular Gram).
    """
    mol = gto.M(atom="N 0 0 -0.55; N 0 0 0.55", basis="def2-svp", verbose=0).build()
    auxmol = df.addons.make_auxmol(mol, "def2-universal-jfit")

    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(auxmol.nao)
    q_before = float(get_ri_charge(auxmol, coeffs))

    corrected = apply_dipole_correction(auxmol, coeffs, np.array([0.1, 0.2, 0.3]))

    assert np.all(np.isfinite(corrected))
    q_after = float(get_ri_charge(auxmol, corrected))
    assert abs(q_after - q_before) < 1e-9


def test_normalize_charge_coulomb_metric_is_minimum_coulomb_norm(water_ri):
    """
    ``metric="coulomb"`` must restore the electron count with the correction of
    minimum Coulomb self-energy: ``δc ∝ J⁻¹ S``, which (unlike the identity
    metric) also moves coefficients outside the s channel.
    """
    from qpet.moments import compute_S_vector, normalize_charge_additive

    _, _, auxmol, coeffs = water_ri
    scaled = coeffs * 1.03
    S_vec = compute_S_vector(auxmol)
    J = auxmol.intor("int2c2e")

    fixed = normalize_charge_additive(auxmol, scaled, 10.0, metric="coulomb")
    assert abs(S_vec @ fixed - 10.0) < 1e-10

    delta = fixed - scaled
    # Minimum J-norm among all corrections with the same charge: the KKT
    # solution is δc = γ J⁻¹ S, so J δc must be parallel to S.
    j_delta = J @ delta
    gamma = (j_delta @ S_vec) / (S_vec @ S_vec)
    assert np.linalg.norm(j_delta - gamma * S_vec) < 1e-10 * np.linalg.norm(j_delta)

    identity = normalize_charge_additive(auxmol, scaled, 10.0, metric="identity")
    assert (identity - scaled)[S_vec == 0.0].max() == 0.0  # shape channels untouched
    assert np.abs(delta[S_vec == 0.0]).max() > 0.0  # Coulomb metric uses them
    assert delta @ J @ delta < (identity - scaled) @ J @ (identity - scaled)


def test_normalize_charge_unknown_metric_raises(water_ri):
    from qpet.moments import normalize_charge_additive

    _, _, auxmol, coeffs = water_ri
    with pytest.raises(ValueError, match="Unknown metric"):
        normalize_charge_additive(auxmol, coeffs, 10.0, metric="overlap")


def test_normalize_charge_guard_skips_a_hopeless_correction(water_ri):
    """
    ``max_shift`` must leave the coefficients alone (and say so) when the gap
    is too large to be fitting noise.

    A prediction off by an electron means the model does not know one of the
    elements; spreading that much charge over the whole basis in the Coulomb
    metric turns a local failure into a global one.
    """
    from qpet.moments import compute_S_vector, normalize_charge_additive

    _, _, auxmol, coeffs = water_ri
    S_vec = compute_S_vector(auxmol)

    with pytest.warns(RuntimeWarning, match="charge projection skipped for Be-ish"):
        untouched = normalize_charge_additive(
            auxmol, coeffs, 11.5, metric="coulomb", max_shift=0.5, label="Be-ish"
        )
    np.testing.assert_array_equal(untouched, coeffs)

    # Just inside the limit the correction is applied as usual.
    fixed = normalize_charge_additive(
        auxmol, coeffs, 10.4, metric="coulomb", max_shift=0.5
    )
    assert abs(S_vec @ fixed - 10.4) < 1e-10


def test_nuclear_moments_use_effective_charges(hydrogen_iodide_ri):
    """
    With the nuclear term included, a neutral ECP molecule must come out
    neutral: 26 electrons against a 25 + 1 nuclear frame, not 53 + 1.
    """
    from qpet.moments import get_ri_charge, get_ri_dipole

    mol, _, auxmol, coeffs = hydrogen_iodide_ri
    q_total = float(get_ri_charge(auxmol, coeffs, include_nuclear=True))
    assert abs(q_total) < 0.05
    # The total dipole of a neutral molecule is origin-independent; a
    # full-Z nuclear frame would add 28 * shift to it.
    mu_a = get_ri_dipole(auxmol, coeffs, np.zeros(3), include_nuclear=True)
    mu_b = get_ri_dipole(
        auxmol, coeffs, np.array([1.0, -2.0, 0.5]), include_nuclear=True
    )
    assert np.allclose(mu_a, mu_b, atol=0.05)


def test_coulomb_direction_regularises_a_singular_metric():
    """A rank-deficient metric (coincident functions) still yields a finite
    direction, with a warning naming the shift; a healthy metric warns nothing."""
    import numpy as np
    import pytest

    from qpet.moments import _coulomb_direction

    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    J = A @ A.T
    J[5] = J[4]  # two identical functions: exactly singular
    J[:, 5] = J[:, 4]
    S = rng.normal(size=6)
    S[5] = S[4]  # identical functions have identical integrals

    with pytest.warns(RuntimeWarning, match="not positive definite"):
        direction = _coulomb_direction(J, S, "toy")
    assert np.all(np.isfinite(direction))
    assert abs(direction[4] - direction[5]) < 1e-6 * np.abs(direction).max()

    healthy = A @ A.T + np.eye(6)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected = np.linalg.solve(healthy, S)
        np.testing.assert_allclose(_coulomb_direction(healthy, S, None), expected)
