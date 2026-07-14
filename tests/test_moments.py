"""Tests for RI-density observables (Ghost-Trick multipoles, S vector)."""

from __future__ import annotations

import numpy as np
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
