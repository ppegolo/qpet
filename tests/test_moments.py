"""Tests for RI-density observables (Ghost-Trick multipoles, S vector)."""

from __future__ import annotations

import numpy as np

from qpet.moments import (
    get_ri_charge,
    get_ri_dipole,
)


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


def test_nuclear_moments_use_effective_charges(hydrogen_iodide_ri):
    """
    With the nuclear term included, a neutral ECP molecule must come out
    neutral: 26 electrons against a 25 + 1 nuclear frame, not 53 + 1.
    """
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
