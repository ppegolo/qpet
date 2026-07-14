"""Resolution-of-identity (RI) scalar-field calculator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyscf import gto

from qpet.conversions import BOHR_TO_ANG

from .base import Calculator, _nuclear_potential

if TYPE_CHECKING:
    from qpet.structure import Structure


class RICalculator(Calculator):
    """
    Scalar-field calculator using RI (resolution-of-identity) density coefficients.

    The electronic potential is ``V_ele(r) = Σ_P c_P (P|r)`` where ``c_P``
    are the fitted coefficients and ``(P|r)`` are Coulomb integrals with the
    auxiliary basis.

    :param coeffs: RI coefficients, shape (naux,)
    :param aux_mol: Auxiliary basis Mole object

    Example::

        from qpet import Structure, RICalculator

        structure = Structure(mol)
        structure.calc = RICalculator(coeffs, aux_mol)
        potential = structure.compute_potential(points)
    """

    def __init__(self, coeffs: np.ndarray, aux_mol: gto.Mole) -> None:
        self.coeffs = coeffs
        self.aux_mol = aux_mol

    def compute_potential(self, atoms: Structure, pts: np.ndarray) -> np.ndarray:
        """
        Compute ESP from RI coefficients.

        :param atoms: Molecular structure (only geometry needed)
        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :return: Electrostatic potential in atomic units, shape (N,)
        """
        pts_bohr = pts / BOHR_TO_ANG

        v_nuc = _nuclear_potential(atoms.positions_bohr, atoms.atomic_numbers, pts_bohr)

        v_ele = np.zeros(len(pts_bohr))
        batch_size = 2000

        for i0 in range(0, len(pts_bohr), batch_size):
            batch_pts = pts_bohr[i0 : i0 + batch_size]
            fakemol = gto.fakemol_for_charges(batch_pts)
            v_Pp = gto.mole.intor_cross("int2c2e", self.aux_mol, fakemol)
            v_ele[i0 : i0 + len(batch_pts)] = self.coeffs @ v_Pp

        return v_nuc - v_ele

    def compute_density(
        self, atoms: Structure, pts: np.ndarray, batch_size: int = 5000
    ) -> np.ndarray:
        """
        Compute electron density ``ρ(r) = Σ_P c_P χ_P(r)``.

        :param atoms: Molecular structure (only geometry needed)
        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per AO batch
        :return: Electron density in electrons / Bohr^3, shape (N,)
        """
        pts_bohr = pts / BOHR_TO_ANG

        rho = np.zeros(len(pts_bohr))
        for p0 in range(0, len(pts_bohr), batch_size):
            p1 = min(p0 + batch_size, len(pts_bohr))
            chi = self.aux_mol.eval_gto("GTOval", pts_bohr[p0:p1])
            rho[p0:p1] = chi @ self.coeffs
        return rho
