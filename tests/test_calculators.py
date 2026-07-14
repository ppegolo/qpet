"""Tests for the scalar-field calculator base class and RICalculator."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import df, dft, gto, scf

from qpet import BOHR_TO_ANG, Calculator, DMCalculator, RICalculator, Structure


class TestCalculatorBase:
    """Density and potential are on equal footing: both default-raise."""

    def test_density_default_raises(self):
        with pytest.raises(NotImplementedError, match="does not support density"):
            Calculator().compute_density(None, None)

    def test_potential_default_raises(self):
        with pytest.raises(NotImplementedError, match="does not support potential"):
            Calculator().compute_potential(None, None)


@pytest.fixture(scope="module")
def water_ri():
    """
    Water RHF/6-31g* density, Coulomb-metric RI-fitted onto def2-svp-jkfit.

    The Coulomb-metric fit is the resolution-of-identity an ML density model
    is trained to reproduce, so it is the right reference for RICalculator.
    """
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        basis="6-31g*",
        verbose=0,
    ).build()
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()

    auxmol = df.addons.make_auxmol(mol, "def2-svp-jkfit")
    j2c = auxmol.intor("int2c2e")
    j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    coeffs = np.linalg.solve(j2c, np.einsum("ijP,ij->P", j3c, dm))
    return mol, dm, auxmol, coeffs


def test_ri_density_integrates_to_electron_count(water_ri):
    """
    The RI-fitted density must integrate to the molecular electron count:
    a wrong overall scale would silently bias every downstream ESP.
    """
    mol, _, auxmol, coeffs = water_ri
    atoms = Structure(mol)
    atoms.calc = RICalculator(coeffs, auxmol)

    grid = dft.gen_grid.Grids(mol)
    grid.level = 3
    grid.build()
    rho = atoms.compute_density(grid.coords * BOHR_TO_ANG)
    n_elec = float(np.sum(grid.weights * rho))
    assert abs(n_elec - mol.nelectron) < 0.05


def test_ri_potential_matches_dm(water_ri):
    """
    RICalculator ESP must reproduce the DMCalculator ESP within RI accuracy —
    i.e. both pathways compute the same physical electrostatic potential.
    """
    mol, dm, auxmol, coeffs = water_ri
    s_dm = Structure(mol)
    s_dm.calc = DMCalculator(dm)
    s_ri = Structure(mol)
    s_ri.calc = RICalculator(coeffs, auxmol)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-4.0, 4.0, size=(60, 3))
    v_dm = s_dm.compute_potential(pts)
    v_ri = s_ri.compute_potential(pts)
    assert np.max(np.abs(v_dm - v_ri)) < 2e-2
