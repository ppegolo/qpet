"""Shared fixtures for the qpet test suite."""

from __future__ import annotations

import pytest
from pyscf import gto, scf

from qpet import RIDensity


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
    rho = RIDensity.from_dm(mol, dm, "def2-svp-jkfit")
    return mol, dm, rho.auxmol, rho.coeffs
