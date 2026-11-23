"""Tests for the projection of RI coefficients onto a fixed integral."""

from __future__ import annotations

import re

import numpy as np
import pytest
from pyscf import gto

from qpet.basis import make_auxmol
from qpet.normalize import aux_integrals, direction, project


def _metric(auxmol, metric):
    if metric == "identity":
        return np.eye(auxmol.nao)
    if metric == "overlap":
        return auxmol.intor("int1e_ovlp")
    J = auxmol.intor("int2c2e")
    if metric == "coulomb_atom":
        block = np.zeros_like(J)
        for _, _, ao0, ao1 in auxmol.aoslice_by_atom():
            block[ao0:ao1, ao0:ao1] = J[ao0:ao1, ao0:ao1]
        return block
    return J


@pytest.mark.parametrize("metric", ["coulomb", "overlap", "identity", "coulomb_atom"])
def test_projection_is_the_minimum_norm_shift(water_ri, metric):
    """The shift reaches the target and is ``∝ M⁻¹q``: ``M Δc`` is parallel to ``q``
    (the condition for the smallest ``Δc·M·Δc`` at fixed ``q·Δc``)."""
    _, _, auxmol, coeffs = water_ri
    scaled = 1.03 * coeffs
    q = aux_integrals(auxmol)
    fixed = project(auxmol, scaled, 10.0, metric)
    assert q @ fixed == pytest.approx(10.0, abs=1e-10)
    m_delta = _metric(auxmol, metric) @ (fixed - scaled)
    gamma = (m_delta @ q) / (q @ q)
    assert np.linalg.norm(m_delta - gamma * q) < 1e-9 * np.linalg.norm(m_delta)


def test_only_coulomb_and_overlap_move_the_shape_channels(water_ri):
    _, _, auxmol, _ = water_ri
    s_free = aux_integrals(auxmol) == 0.0
    for metric in ("identity", "coulomb_atom"):
        d = direction(auxmol, metric)
        assert np.abs(d[s_free]).max() < 1e-12 * np.abs(d).max()
    for metric in ("coulomb", "overlap"):
        assert direction(auxmol, metric)[s_free].any()


def test_project_broadcasts_over_members(water_ri):
    _, _, auxmol, coeffs = water_ri
    members = coeffs * np.array([0.98, 1.0, 1.03])[:, None]
    fixed = project(auxmol, members, 10.0)
    np.testing.assert_allclose(fixed @ aux_integrals(auxmol), 10.0, atol=1e-10)
    np.testing.assert_allclose(fixed[2], project(auxmol, members[2], 10.0))


def test_aux_integrals_match_the_flat_gaussian_overlap(water_ri):
    """Two routes to ``∫ φ_k dr``: the closed form here and the overlap with the
    flat Gaussian behind the multipoles."""
    from qpet.moments import _FLAT, _SCALE

    _, _, auxmol, _ = water_ri
    overlap = _SCALE * gto.intor_cross("int1e_ovlp", auxmol, _FLAT).ravel()
    np.testing.assert_allclose(aux_integrals(auxmol), overlap, atol=1e-8)


def test_unknown_metric_raises(water_ri):
    _, _, auxmol, _ = water_ri
    message = (
        "unknown metric 'l2'; use 'coulomb', 'overlap', 'identity' or 'coulomb_atom'"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        direction(auxmol, "l2")


def test_singular_coulomb_metric_is_regularised():
    """Coincident atoms make ``J`` singular: the projection still reaches the
    target, splits it evenly between the copies, and reports the shift."""
    mol = gto.M(atom="He 0 0 0; He 0 0 0", basis="def2-svp", verbose=0)
    auxmol = make_auxmol(mol, "def2-universal-jfit")
    message = (
        f"Coulomb metric is not positive definite (naux={auxmol.nao}); "
        "the projection used a relative diagonal shift of 1.0e-10."
    )
    with pytest.warns(RuntimeWarning, match=f"^{re.escape(message)}$"):
        fixed = project(auxmol, np.zeros(auxmol.nao), 4.0)
    assert aux_integrals(auxmol) @ fixed == pytest.approx(4.0, abs=1e-10)
    half = auxmol.nao // 2
    np.testing.assert_allclose(
        fixed[:half], fixed[half:], atol=1e-5 * np.abs(fixed).max()
    )


def test_coulomb_atom_uses_one_block_per_species():
    """Two waters: every O (and every H) gets the same one-centre shift."""
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0; "
        "O 5 0 0; H 5.757 0.587 0; H 4.243 0.587 0",
        basis="sto-3g",
        verbose=0,
    )
    auxmol = make_auxmol(mol, "def2-universal-jfit")
    d = direction(auxmol, "coulomb_atom")
    slices = [slice(ao0, ao1) for _, _, ao0, ao1 in auxmol.aoslice_by_atom()]
    np.testing.assert_array_equal(d[slices[0]], d[slices[3]])
    np.testing.assert_array_equal(d[slices[1]], d[slices[5]])
