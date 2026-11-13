"""Tests for to_scf: PySCF calculations started from a density."""

from __future__ import annotations

import pytest
from pyscf import dft, gto

from qpet import Prediction, RIDensity, to_scf

AUX = "def2-universal-jfit"


def _cycles(mf) -> int:
    count = []
    mf.callback = lambda envs: count.append(1)
    mf.kernel()
    assert mf.converged
    return len(count)


def _fitted(mf, mol) -> Prediction:
    """The converged density of ``mf``, RI-fitted as a model would predict it."""
    dm = mf.make_rdm1()
    if dm.ndim == 2:
        return Prediction(rho=RIDensity.fit(mol, dm, AUX, target=mol.nelectron))
    return Prediction(
        rho=RIDensity.fit(mol, dm[0] + dm[1], AUX, target=mol.nelectron),
        spin=RIDensity.fit(mol, dm[0] - dm[1], AUX, target=mol.spin, nuclei=False),
    )


@pytest.mark.parametrize(
    ("xc", "strict"),
    [
        ("pbe", True),
        ("b3lyp", True),
        # the PBE guess potential misses the long-range exchange: no gain on water
        ("wb97x", False),
    ],
)
def test_closed_shell_starts_close_to_the_solution(xc, strict):
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", basis="def2-svp", verbose=0
    )
    reference = dft.RKS(mol, xc=xc)
    n_default = _cycles(reference)
    mf = to_scf(_fitted(reference, mol), mol, xc)
    assert isinstance(mf, dft.rks.RKS)
    n = _cycles(mf)
    assert n < n_default if strict else n <= n_default
    assert mf.e_tot == pytest.approx(reference.e_tot, abs=1e-7)


def test_guess_survives_to_gpu():
    pytest.importorskip("gpu4pyscf")
    cupy = pytest.importorskip("cupy")
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", basis="def2-svp", verbose=0
    )
    reference = dft.RKS(mol, xc="b3lyp").to_gpu()
    n_default = _cycles(reference)
    mf = to_scf(_fitted(reference.to_cpu(), mol), mol, "b3lyp")
    gpu = mf.to_gpu()
    assert cupy.asnumpy(gpu.init_guess) == pytest.approx(mf.init_guess, abs=1e-12)
    assert _cycles(gpu) < n_default
    assert gpu.e_tot == pytest.approx(reference.e_tot, abs=1e-7)


def test_open_shell_uses_both_spin_channels():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="def2-svp", spin=1, verbose=0)
    reference = dft.UKS(mol, xc="pbe")
    n_default = _cycles(reference)
    mf = to_scf(_fitted(reference, mol), mol, "pbe", conv_tol=1e-10)
    assert isinstance(mf, dft.uks.UKS) and mf.conv_tol == 1e-10
    assert mf.init_guess.shape == (2, mol.nao, mol.nao)
    assert _cycles(mf) < n_default
    assert mf.e_tot == pytest.approx(reference.e_tot, abs=1e-7)
