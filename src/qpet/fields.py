"""Density value objects: evaluable electron densities carrying their geometry.

A density object knows everything needed to evaluate its scalar fields —
electron density and electrostatic potential — at arbitrary points:

- :class:`RIDensity` — density expanded in an auxiliary (RI) basis, the
  representation ML models predict and the Coulomb fit produces.
- :class:`DMDensity` — density from an AO-basis density matrix, the exact
  quantum-chemistry reference.

Both carry their molecular geometry (via the PySCF ``Mole``/``auxmol``), so
fields, multipoles, and CUBE export need no separate structure object and
cannot be paired with the wrong molecule.

Points are given in Angstrom; all returned values are in atomic units.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pyscf import gto

from qpet.conversions import BOHR_TO_ANG

# Clamp distance to avoid singularity at nuclei (Bohr).
_DIST_FLOOR: float = 0.1


def _nuclear_potential(
    positions_bohr: np.ndarray,
    charges: np.ndarray,
    pts_bohr: np.ndarray,
    max_chunk_elements: int = 2_000_000,
) -> np.ndarray:
    """
    Nuclear contribution to ESP: ``V_nuc = Σ_A Z_A / |r - R_A|``.

    Points are processed in chunks to bound peak memory — the largest
    intermediate tensor holds about ``3 * max_chunk_elements`` floats
    regardless of grid size or atom count.

    :param positions_bohr: Nuclear positions in Bohr, shape (N_atoms, 3)
    :param charges: Nuclear charges (atomic numbers), shape (N_atoms,)
    :param pts_bohr: Evaluation points in Bohr, shape (N_pts, 3)
    :param max_chunk_elements: Target size (in elements) of the
        ``(pts_chunk, N_atoms)`` distance matrix per batch
    :return: Nuclear potential in atomic units, shape (N_pts,)
    """
    n_atoms = len(charges)
    pts_chunk = max(1, max_chunk_elements // max(n_atoms, 1))
    v_nuc = np.empty(len(pts_bohr))
    for start in range(0, len(pts_bohr), pts_chunk):
        end = min(start + pts_chunk, len(pts_bohr))
        diff = pts_bohr[start:end, np.newaxis, :] - positions_bohr[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        v_nuc[start:end] = np.sum(
            charges[np.newaxis, :] / np.maximum(dist, _DIST_FLOOR), axis=1
        )
    return v_nuc


class ScalarField(Protocol):
    """Anything that can evaluate density/ESP and knows its geometry.

    Satisfied by :class:`RIDensity` and :class:`DMDensity`; the CUBE writers
    accept any implementation.
    """

    @property
    def positions(self) -> np.ndarray:
        """Atomic positions in Angstrom, shape (N, 3)."""
        ...

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers, shape (N,)."""
        ...

    def density(self, pts: np.ndarray, batch_size: int = ...) -> np.ndarray:
        """Electron density at points (Angstrom in, electrons/Bohr^3 out)."""
        ...

    def potential(self, pts: np.ndarray, batch_size: int = ...) -> np.ndarray:
        """Electrostatic potential at points (Angstrom in, a.u. out)."""
        ...


class DMDensity:
    """
    Electron density from an AO-basis density matrix.

    The potential is ``V = V_nuc - V_ele``, where ``V_ele`` is computed from
    one-electron integrals contracted with the density matrix.

    :param mol: Built PySCF ``Mole`` the density matrix belongs to.
    :param dm: Density matrix in AO basis, shape (nao, nao).

    Example::

        from pyscf import gto, scf
        from qpet import DMDensity

        mol = gto.M(atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", basis="cc-pvdz")
        dm = scf.RHF(mol).run().make_rdm1()

        rho = DMDensity(mol, dm)
        potential = rho.potential(points)
    """

    def __init__(self, mol: gto.Mole, dm: np.ndarray) -> None:
        if dm.ndim != 2 or dm.shape != (mol.nao, mol.nao):
            raise ValueError(
                f"dm shape {dm.shape} does not match mol AO basis "
                f"({mol.nao}, {mol.nao})"
            )
        self.mol = mol
        self.dm = dm

    @property
    def positions(self) -> np.ndarray:
        """Atomic positions in Angstrom, shape (N, 3)."""
        return self.mol.atom_coords(unit="Ang")

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers, shape (N,)."""
        return self.mol.atom_charges()

    def density(self, pts: np.ndarray, batch_size: int = 5000) -> np.ndarray:
        """
        Compute electron density ``ρ(r) = Σ_ij D_ij φ_i(r) φ_j(r)``.

        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per AO batch
        :return: Electron density in electrons / Bohr^3, shape (N,)
        """
        pts_bohr = pts / BOHR_TO_ANG
        rho = np.zeros(len(pts_bohr))
        for p0 in range(0, len(pts_bohr), batch_size):
            p1 = min(p0 + batch_size, len(pts_bohr))
            ao = self.mol.eval_gto("GTOval", pts_bohr[p0:p1])
            rho[p0:p1] = np.einsum("pi,pi->p", ao @ self.dm, ao)
        return rho

    def potential(self, pts: np.ndarray, batch_size: int = 500) -> np.ndarray:
        """
        Compute ESP from the density matrix.

        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per batch. Each batch allocates a
            ``(batch, nao, nao)`` integral tensor, so the default is small;
            increase for speed if memory allows.
        :return: Electrostatic potential in atomic units, shape (N,)
        """
        pts_bohr = pts / BOHR_TO_ANG
        v_nuc = _nuclear_potential(
            self.mol.atom_coords(), self.atomic_numbers, pts_bohr
        )
        v_ele = np.zeros(len(pts_bohr))
        for p0 in range(0, len(pts_bohr), batch_size):
            p1 = min(p0 + batch_size, len(pts_bohr))
            ints = self.mol.intor("int1e_grids", grids=pts_bohr[p0:p1])
            v_ele[p0:p1] = ints.reshape(len(ints), -1) @ self.dm.ravel()
        return v_nuc - v_ele

    def __repr__(self) -> str:
        return f"DMDensity(natm={self.mol.natm}, nao={self.mol.nao})"


class RIDensity:
    """
    Electron density expanded in an auxiliary (RI) basis.

    ``ρ(r) = Σ_P c_P χ_P(r)`` with the geometry and basis carried by
    ``auxmol``. The electronic potential is ``V_ele(r) = Σ_P c_P (P|r)``.

    :param auxmol: Auxiliary-basis ``Mole`` (e.g. from
        ``pyscf.df.addons.make_auxmol``).
    :param coeffs: RI coefficients, shape (naux,).

    Example::

        from qpet import RIDensity

        rho = RIDensity(auxmol, coeffs)
        v = rho.potential(points)
        mu = rho.dipole(origin=r_cnc)
    """

    def __init__(self, auxmol: gto.Mole, coeffs: np.ndarray) -> None:
        if coeffs.shape != (auxmol.nao,):
            raise ValueError(
                f"coeffs shape {coeffs.shape} does not match auxiliary basis size "
                f"({auxmol.nao},)"
            )
        self.auxmol = auxmol
        self.coeffs = coeffs

    @property
    def positions(self) -> np.ndarray:
        """Atomic positions in Angstrom, shape (N, 3)."""
        return self.auxmol.atom_coords(unit="Ang")

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers, shape (N,)."""
        return self.auxmol.atom_charges()

    def density(self, pts: np.ndarray, batch_size: int = 5000) -> np.ndarray:
        """
        Compute electron density ``ρ(r) = Σ_P c_P χ_P(r)``.

        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per AO batch
        :return: Electron density in electrons / Bohr^3, shape (N,)
        """
        pts_bohr = pts / BOHR_TO_ANG
        rho = np.zeros(len(pts_bohr))
        for p0 in range(0, len(pts_bohr), batch_size):
            p1 = min(p0 + batch_size, len(pts_bohr))
            chi = self.auxmol.eval_gto("GTOval", pts_bohr[p0:p1])
            rho[p0:p1] = chi @ self.coeffs
        return rho

    def potential(self, pts: np.ndarray, batch_size: int = 2000) -> np.ndarray:
        """
        Compute ESP from the RI coefficients.

        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per batch (controls peak memory)
        :return: Electrostatic potential in atomic units, shape (N,)
        """
        pts_bohr = pts / BOHR_TO_ANG
        v_nuc = _nuclear_potential(
            self.auxmol.atom_coords(), self.atomic_numbers, pts_bohr
        )
        v_ele = np.zeros(len(pts_bohr))
        for p0 in range(0, len(pts_bohr), batch_size):
            p1 = min(p0 + batch_size, len(pts_bohr))
            fakemol = gto.fakemol_for_charges(pts_bohr[p0:p1])
            v_Pp = gto.mole.intor_cross("int2c2e", self.auxmol, fakemol)
            v_ele[p0:p1] = self.coeffs @ v_Pp
        return v_nuc - v_ele

    # -- multipoles (delegate to qpet.moments) ---------------------------- #

    def charge(self, *, include_nuclear: bool = False) -> np.ndarray:
        """Electronic charge ``-N_elec`` (or total charge with nuclei)."""
        from qpet.moments import get_ri_charge

        return get_ri_charge(self.auxmol, self.coeffs, include_nuclear=include_nuclear)

    def dipole(
        self,
        origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        *,
        include_nuclear: bool = False,
    ) -> np.ndarray:
        """Dipole moment at ``origin`` (Bohr), atomic units."""
        from qpet.moments import get_ri_dipole

        return get_ri_dipole(
            self.auxmol, self.coeffs, origin, include_nuclear=include_nuclear
        )

    def quadrupole(
        self,
        origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        *,
        include_nuclear: bool = False,
    ) -> np.ndarray:
        """Primitive quadrupole tensor at ``origin`` (Bohr), atomic units."""
        from qpet.moments import get_ri_quadrupole

        return get_ri_quadrupole(
            self.auxmol, self.coeffs, origin, include_nuclear=include_nuclear
        )

    # -- corrections (return new objects) --------------------------------- #

    def normalize_charge(self, n_electrons: float | None = None) -> RIDensity:
        """
        Return a copy whose density integrates to the exact electron count.

        Minimum-norm additive shift on the s-channel coefficients; l>0
        (shape) channels are untouched.

        :param n_electrons: Target electron count; defaults to
            ``auxmol.nelectron`` (charge- and ECP-aware).
        """
        from qpet.moments import normalize_charge_additive

        if n_electrons is None:
            n_electrons = self.auxmol.nelectron
        return RIDensity(
            self.auxmol,
            normalize_charge_additive(self.auxmol, self.coeffs, n_electrons),
        )

    def constrain_dipole(self, mu_target: np.ndarray) -> RIDensity:
        """
        Return a copy whose electronic dipole matches ``mu_target``.

        Minimum-norm, charge-neutral correction. The target is the electronic
        dipole at the centre of nuclear charge, in e·Bohr.
        """
        from qpet.moments import apply_dipole_correction, get_ri_dipole

        z = self.atomic_numbers.astype(float)
        r_cnc = z @ self.auxmol.atom_coords() / z.sum()
        delta_mu = np.asarray(mu_target) - get_ri_dipole(
            self.auxmol, self.coeffs, r_cnc
        )
        return RIDensity(
            self.auxmol,
            apply_dipole_correction(self.auxmol, self.coeffs, delta_mu),
        )

    # -- SCF bridge -------------------------------------------------------- #

    def scf_guess(
        self, mol: gto.Mole, xc: str = "pbe", *, normalize_charge: bool = False
    ) -> np.ndarray:
        """
        SCF initial-guess density matrix built from this density.

        See :func:`qpet.scf.dm_from_auxdensity`.

        :param mol: Built PySCF ``Mole`` (orbital basis) for the SCF run.
        :param xc: Exchange-correlation functional.
        :param normalize_charge: Normalize the electron count first.
        :return: ``dm0`` for ``mf.kernel(dm0=dm0)``.
        """
        from qpet.scf import dm_from_auxdensity

        return dm_from_auxdensity(
            mol, self.auxmol, self.coeffs, xc, normalize_charge=normalize_charge
        )

    def __repr__(self) -> str:
        return f"RIDensity(natm={self.auxmol.natm}, naux={self.auxmol.nao})"
