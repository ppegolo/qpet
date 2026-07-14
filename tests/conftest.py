"""Shared fixtures for the qpet test suite."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import df, gto, scf


@pytest.fixture(scope="session")
def water_ri():
    """
    Water RHF/6-31g* density, Coulomb-metric RI-fitted onto def2-svp-jkfit.

    The Coulomb-metric fit is the resolution-of-identity an ML density model
    is trained to reproduce, so it is the right reference for RIDensity
    and for the TensorMap round-trip tests.

    Returns ``(mol, dm, auxmol, coeffs)``.
    """
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        basis="6-31g*",
        verbose=0,
    )
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()

    auxmol = df.addons.make_auxmol(mol, "def2-svp-jkfit")
    j2c = auxmol.intor("int2c2e")
    j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    coeffs = np.linalg.solve(j2c, np.einsum("ijP,ij->P", j3c, dm))
    return mol, dm, auxmol, coeffs
