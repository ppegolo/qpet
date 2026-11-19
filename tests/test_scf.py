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


WATER = "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0"


@pytest.mark.parametrize("xc", ["pbe", "b3lyp", "wb97x", "wb97m-v"])
def test_closed_shell_starts_close_to_the_solution(xc):
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    reference = dft.RKS(mol, xc=xc)
    n_default = _cycles(reference)
    mf = to_scf(_fitted(reference, mol), dft.RKS(mol, xc=xc))
    assert _cycles(mf) < n_default
    assert mf.e_tot == pytest.approx(reference.e_tot, abs=1e-7)


@pytest.mark.parametrize("xc", ["pbe", "wb97m-v"])
def test_open_shell_uses_both_spin_channels(xc):
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="def2-svp", spin=1, verbose=0)
    reference = dft.UKS(mol, xc=xc)
    n_default = _cycles(reference)
    mf = to_scf(_fitted(reference, mol), dft.UKS(mol, xc=xc))
    assert mf.init_guess.shape == (2, mol.nao, mol.nao)
    assert _cycles(mf) < n_default
    assert mf.e_tot == pytest.approx(reference.e_tot, abs=1e-7)


def test_gpu_object_gives_the_cpu_guess():
    pytest.importorskip("gpu4pyscf")
    cupy = pytest.importorskip("cupy")
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    reference = dft.RKS(mol, xc="wb97m-v").run()
    n_default = _cycles(dft.RKS(mol, xc="wb97m-v").to_gpu())
    prediction = _fitted(reference, mol)
    cpu = to_scf(prediction, dft.RKS(mol, xc="wb97m-v"))
    gpu = to_scf(prediction, dft.RKS(mol, xc="wb97m-v").to_gpu())
    assert cupy.asnumpy(gpu.init_guess) == pytest.approx(cpu.init_guess, abs=1e-8)
    assert _cycles(gpu) < n_default
    assert gpu.e_tot == pytest.approx(reference.e_tot, abs=1e-7)
