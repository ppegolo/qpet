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
    rho = RIDensity.fit(mol, dm, "def2-svp-jkfit")
    return mol, dm, rho.auxmol, rho.coeffs


@pytest.fixture(scope="session")
def hydrogen_iodide_ri():
    """
    HI RHF/def2-svp with the def2 ECP on iodine, RI-fitted onto
    def2-universal-jfit.

    Iodine keeps 25 of its 53 electrons: the fixture for everything that must
    tell the element (``Z = 53``) from the nucleus the density sees
    (``Z_eff = 25``).

    Returns ``(mol, dm, auxmol, coeffs)``.
    """
    mol = gto.M(
        atom="I 0 0 0; H 0 0 1.61",
        basis="def2-svp",
        ecp="def2-svp",
        verbose=0,
    )
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()
    rho = RIDensity.fit(mol, dm, "def2-universal-jfit")
    return mol, dm, rho.auxmol, rho.coeffs
