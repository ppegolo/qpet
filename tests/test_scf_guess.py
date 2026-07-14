"""Tests for the RI-density SCF initial guess (``dm_from_auxdensity``)."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import dft, gto

from qpet import RIDensity, dm_from_auxdensity
from qpet.moments import compute_S_vector, normalize_charge_additive


def _run_scf(mol: gto.Mole, xc: str, dm0: np.ndarray | None = None):
    """Run RKS SCF, returning the mean-field object and the cycle count."""
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    n_cycles = 0

    def _count(envs):
        nonlocal n_cycles
        n_cycles += 1

    mf.callback = _count
    mf.kernel(dm0=dm0)
    return mf, n_cycles


@pytest.mark.parametrize(
    ("xc", "strict"),
    [
        ("pbe", True),  # pure GGA: Fock is exact for the given density
        ("b3lyp", False),  # hybrid: exact exchange approximated from minao
        ("tpss", False),  # MGGA: tau approximated (von Weizsäcker)
    ],
)
def test_guess_from_converged_density_accelerates_scf(water_ri, xc, strict):
    """
    The whole point of the guess: a Fock matrix built from a good predicted
    density must land close to the converged solution. Fitting the *converged*
    density and restarting must reach the same energy in fewer cycles than the
    default guess (or at worst equal, for functionals where part of the Fock
    matrix is approximated).
    """
    mol, _, auxmol, _ = water_ri

    mf_ref, n_ref = _run_scf(mol, xc)
    assert mf_ref.converged

    coeffs = RIDensity.from_dm(mol, mf_ref.make_rdm1(), "def2-svp-jkfit").coeffs
    dm0 = dm_from_auxdensity(mol, auxmol, coeffs, xc=xc)

    mf_ml, n_ml = _run_scf(mol, xc, dm0=dm0)
    assert mf_ml.converged
    assert abs(mf_ml.e_tot - mf_ref.e_tot) < 1e-8
    if strict:
        assert n_ml < n_ref, f"guess did not accelerate SCF: {n_ml} vs {n_ref}"
    else:
        assert n_ml <= n_ref, f"guess slowed SCF down: {n_ml} vs {n_ref}"


def test_normalize_charge_additive(water_ri):
    """
    The additive correction must restore the exact electron count without
    touching the input array (callers may reuse their coefficients).
    """
    _, _, auxmol, coeffs = water_ri
    scaled = coeffs * 1.03
    scaled_copy = scaled.copy()

    fixed = normalize_charge_additive(auxmol, scaled, 10.0)

    S_vec = compute_S_vector(auxmol)
    assert abs(S_vec @ fixed - 10.0) < 1e-10
    np.testing.assert_array_equal(scaled, scaled_copy)


def test_open_shell_raises(water_ri):
    """Open-shell input must fail loudly, not silently produce an RKS guess."""
    _, _, auxmol, coeffs = water_ri
    mol_open = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    with pytest.raises(NotImplementedError, match="closed-shell"):
        dm_from_auxdensity(mol_open, auxmol, coeffs, xc="pbe")


def test_range_separated_raises(water_ri):
    """Range-separated hybrids are unsupported and must say so."""
    mol, _, auxmol, coeffs = water_ri
    with pytest.raises(NotImplementedError, match="range-separated"):
        dm_from_auxdensity(mol, auxmol, coeffs, xc="wb97x")


def test_wrong_coeff_size_raises(water_ri):
    mol, _, auxmol, _ = water_ri
    with pytest.raises(ValueError, match="does not match"):
        dm_from_auxdensity(mol, auxmol, np.zeros(auxmol.nao + 1), xc="pbe")
