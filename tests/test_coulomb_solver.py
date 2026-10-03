"""Tests for the iterative Coulomb-metric solver."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto

import qpet._coulomb_solver
import qpet.moments
from qpet._coulomb_solver import CoulombSolver, _FarField
from qpet.basis import make_auxmol
from qpet.moments import compute_S_vector, normalize_charge_additive

_WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.757, 0.587, 0.0)),
    ("H", (-0.757, 0.587, 0.0)),
]


def _water_cluster(n_cells: tuple[int, int, int], spacing: float) -> gto.Mole:
    """Auxiliary Mole of waters on a cubic grid (``spacing`` in Å)."""
    atom = [
        (symbol, tuple(np.add(xyz, spacing * np.array([i, j, k]))))
        for i in range(n_cells[0])
        for j in range(n_cells[1])
        for k in range(n_cells[2])
        for symbol, xyz in _WATER
    ]
    mol = gto.M(atom=atom, basis="sto-3g", verbose=0)
    return make_auxmol(mol, "def2-universal-jfit")


def _j_norm_error(J: np.ndarray, x: np.ndarray, reference: np.ndarray) -> float:
    e = x - reference
    return float(np.sqrt(e @ J @ e / (reference @ J @ reference)))


@pytest.fixture(scope="module")
def cluster():
    """40 waters (120 atoms) spanning about 14 Å, with the exact metric and S vector."""
    auxmol = _water_cluster((4, 5, 2), 3.5)
    return auxmol, auxmol.intor("int2c2e"), compute_S_vector(auxmol)


def test_solve_matches_dense_inverse(cluster):
    """The J-norm error of ``d = J⁻¹ S`` stays within twice the requested tolerance."""
    auxmol, J, S_vec = cluster
    dense = scipy.linalg.cho_solve(scipy.linalg.cho_factor(J), S_vec)
    d = CoulombSolver(auxmol).solve(S_vec, tol=5e-5)
    assert _j_norm_error(J, d, dense) < 1e-4


def test_far_field_matches_int2c2e_between_distant_atoms():
    """Atoms of different waters, 14 Å apart, interact exactly as multipoles."""
    auxmol = _water_cluster((2, 1, 1), 14.0)
    J = auxmol.intor("int2c2e")
    x = np.random.default_rng(0).normal(size=auxmol.nao)
    water = np.array([label[0] // 3 for label in auxmol.ao_labels(fmt=None)])
    expected = np.where(water[:, None] != water[None, :], J, 0.0) @ x
    far = _FarField(auxmol, rnear=5.0).apply(x)
    assert np.linalg.norm(far - expected) < 1e-8 * np.linalg.norm(expected)


def test_normalize_charge_iterative_matches_dense(cluster):
    """Both routes of ``metric="coulomb"`` give the same corrected coefficients."""
    auxmol, J, S_vec = cluster
    coeffs = np.random.default_rng(1).normal(size=auxmol.nao) * 1e-2
    n_electrons = float(S_vec @ coeffs) + 0.7
    dense = normalize_charge_additive(
        auxmol, coeffs, n_electrons, metric="coulomb", iterative=False
    )
    fast = normalize_charge_additive(
        auxmol, coeffs, n_electrons, metric="coulomb", iterative=True
    )
    assert abs(S_vec @ fast - n_electrons) < 1e-10
    assert _j_norm_error(J, fast - coeffs, dense - coeffs) < 1e-4


def test_normalize_charge_switches_to_iterative_solver_by_size(cluster, monkeypatch):
    """``iterative=None`` follows ``ITERATIVE_COULOMB_NAUX``."""
    auxmol, _, S_vec = cluster
    calls = []

    class Spy(CoulombSolver):
        def solve(self, b, tol=5e-5):
            calls.append(tol)
            return super().solve(b, tol)

    monkeypatch.setattr(qpet._coulomb_solver, "CoulombSolver", Spy)
    coeffs = np.zeros(auxmol.nao)
    normalize_charge_additive(auxmol, coeffs, 1.0, metric="coulomb")
    assert not calls
    monkeypatch.setattr(qpet.moments, "ITERATIVE_COULOMB_NAUX", auxmol.nao)
    normalize_charge_additive(auxmol, coeffs, 1.0, metric="coulomb")
    assert len(calls) == 1
