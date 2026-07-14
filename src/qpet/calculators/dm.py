"""Density-matrix scalar-field calculator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qpet.conversions import BOHR_TO_ANG

from .base import Calculator, _nuclear_potential

if TYPE_CHECKING:
    from qpet.structure import Structure


class DMCalculator(Calculator):
    """
    Scalar-field calculator using a density matrix in the AO basis.

    The potential is ``V = V_nuc - V_ele``, where ``V_ele`` is computed from
    one-electron integrals contracted with the density matrix.

    Note: the density matrix must correspond to the same molecule and basis
    as the :class:`~qpet.structure.Structure` passed to the compute methods.
    Mismatched ``mol/dm`` pairs will produce incorrect results.

    :param dm: Density matrix in AO basis, shape (nao, nao)

    Example::

        from pyscf import gto, scf
        from qpet import Structure, DMCalculator

        mol = gto.M(
            atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", basis="cc-pvdz"
        ).build()
        dm = scf.RHF(mol).run().make_rdm1()

        structure = Structure(mol)
        structure.calc = DMCalculator(dm)
        potential = structure.compute_potential(points)
    """

    def __init__(self, dm: np.ndarray) -> None:
        self.dm = dm

    def compute_potential(self, atoms: Structure, pts: np.ndarray) -> np.ndarray:
        """
        Compute ESP from the density matrix.

        :param atoms: Molecular structure with pyscf Mole (requires basis)
        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :return: Electrostatic potential in atomic units, shape (N,)
        """
        mol = atoms.mol
        pts_bohr = pts / BOHR_TO_ANG

        v_nuc = _nuclear_potential(atoms.positions_bohr, atoms.atomic_numbers, pts_bohr)

        v_ele = np.zeros(len(pts_bohr))
        batch_size = 500
        for p0 in range(0, len(pts_bohr), batch_size):
            p1 = min(p0 + batch_size, len(pts_bohr))
            ints = mol.intor("int1e_grids", grids=pts_bohr[p0:p1])
            v_ele[p0:p1] = np.einsum("ij,kij->k", self.dm, ints)

        return v_nuc - v_ele

    def compute_density(
        self, atoms: Structure, pts: np.ndarray, batch_size: int = 5000
    ) -> np.ndarray:
        """
        Compute electron density ``ρ(r) = Σ_ij D_ij φ_i(r) φ_j(r)``.

        :param atoms: Molecular structure with pyscf Mole (requires basis)
        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per AO batch
        :return: Electron density in electrons / Bohr^3, shape (N,)
        """
        mol = atoms.mol
        pts_bohr = pts / BOHR_TO_ANG

        rho = np.zeros(len(pts_bohr))
        for p0 in range(0, len(pts_bohr), batch_size):
            p1 = min(p0 + batch_size, len(pts_bohr))
            ao = mol.eval_gto("GTOval", pts_bohr[p0:p1])
            rho[p0:p1] = np.einsum("pi,ij,pj->p", ao, self.dm, ao)
        return rho
