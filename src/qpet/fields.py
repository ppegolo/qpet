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

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pyscf import df, gto

from qpet.basis import copy_nuclei, make_auxmol
from qpet.conditioning import atomic_numbers_of, nuclear_charges_of
from qpet.conversions import BOHR_TO_ANG

if TYPE_CHECKING:
    from qpet.esp_fast import FastESP
    from qpet.partition import FragmentDensity

logger = logging.getLogger(__name__)

# Clamp distance to avoid singularity at nuclei (Bohr).
_DIST_FLOOR: float = 0.1

#: Grid size below which :meth:`RIDensity.potential` stays on libcint: smaller
#: than this and the closed form's fit costs more than the evaluation saves.
FAST_ESP_MIN_POINTS = 1000


def nuclear_potential(
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
    :param charges: Effective nuclear charges, shape (N_atoms,): ``Z`` for an
        all-electron atom, ``Z - n_core`` under an ECP (see
        :func:`qpet.nuclear_charges_of`)
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
        """Atomic numbers (element identity ``Z``), shape (N,)."""
        ...

    @property
    def nuclear_charges(self) -> np.ndarray:
        """Effective nuclear charges (``Z`` minus ECP core electrons), shape (N,)."""
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
        """Atomic numbers (element identity ``Z``), shape (N,)."""
        return atomic_numbers_of(self.mol)

    @property
    def nuclear_charges(self) -> np.ndarray:
        """Effective nuclear charges (``Z`` minus ECP core electrons), shape (N,)."""
        return nuclear_charges_of(self.mol)

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
        v_nuc = nuclear_potential(
            self.mol.atom_coords(), self.nuclear_charges, pts_bohr
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

    :param auxmol: Auxiliary-basis ``Mole`` from :func:`qpet.make_auxmol`
        (PySCF's own ``make_auxmol`` drops the ECP from the nuclei, so an ECP
        system built with it gets the wrong nuclear potential).
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
        self._fast: FastESP | None = None
        self._no_fast = False  # the fit was tried for this basis and failed

    @classmethod
    def from_dm(cls, mol: gto.Mole, dm: np.ndarray, aux_basis: str) -> RIDensity:
        """
        Coulomb-metric RI fit of a density matrix.

        This is the resolution-of-identity that density ML models are trained
        to reproduce, so it provides the reference RI density for a given
        quantum-chemistry result. Incore 3-center integrals: peak memory
        nao² × naux floats.

        :param mol: Built PySCF ``Mole`` (orbital basis).
        :param dm: Density matrix in AO basis, shape (nao, nao).
        :param aux_basis: Auxiliary basis set name
            (e.g. ``"def2-universal-jfit"``) or an even-tempered specification
            ``"etb:<ao_basis>:<beta>"`` (e.g. ``"etb:def2-svp:2.0"``).
        :return: Fitted :class:`RIDensity`.
        """
        auxmol = make_auxmol(mol, aux_basis)
        j2c = auxmol.intor("int2c2e")
        j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        coeffs = np.linalg.solve(j2c, np.einsum("ijP,ij->P", j3c, dm))
        return cls(auxmol, coeffs)

    def select_atoms(self, atom_indices: Sequence[int]) -> RIDensity:
        """
        Restrict the density to a subset of atoms.

        Returns a new :class:`RIDensity` whose auxiliary molecule contains only
        the selected atoms and whose coefficients are the corresponding slice.
        Both the nuclear positions and the electronic expansion restrict, so
        :meth:`potential` gives the electrostatic potential produced by exactly
        those atoms (in the density predicted for the full system) — the
        primitive for attributing a complex's density to one partner while
        keeping the other in the model's context.

        For a partition of all atoms into A and B this is exact and additive:
        ``select_atoms(A).potential(p) + select_atoms(B).potential(p)
        == potential(p)``.

        :param atom_indices: Atom indices to keep (into this density's atoms).
        :return: Sub-density on the selected atoms.
        """
        idx = sorted({int(i) for i in atom_indices})
        if not idx:
            raise ValueError("atom_indices is empty; nothing to select.")
        n_atoms = self.auxmol.natm
        if idx[0] < 0 or idx[-1] >= n_atoms:
            raise IndexError(
                f"atom index out of range for {n_atoms}-atom density: {idx}"
            )

        aoslice = self.auxmol.aoslice_by_atom()  # (natm, 4): sh0, sh1, ao0, ao1
        ao_keep = np.concatenate([np.arange(aoslice[a, 2], aoslice[a, 3]) for a in idx])

        sub = gto.Mole()
        sub.atom = [self.auxmol._atom[a] for a in idx]
        sub.basis = self.auxmol.basis
        sub.unit = "Bohr"  # ._atom coordinates are in Bohr
        sub.charge = 0
        # Auxiliary mol is used only for integrals, never SCF; pick a valid
        # spin so build() accepts odd electron counts.
        sub.spin = int(sum(self.auxmol.atom_charge(a) for a in idx)) % 2
        sub.verbose = 0
        sub.build()
        # build() reinstates full-Z nuclei; keep the parent's effective charges.
        copy_nuclei(self.auxmol, sub, idx)

        return RIDensity(sub, self.coeffs[ao_keep])

    def split_hirshfeld(
        self,
        atoms_a: Sequence[int],
        atoms_b: Sequence[int] | None = None,
        **options: object,
    ) -> tuple[FragmentDensity, FragmentDensity]:
        """
        Partition the density into two fragments in real space (Hirshfeld).

        Where :meth:`select_atoms` assigns every auxiliary function to the atom
        it is centred on, this assigns every point in space to the fragments in
        proportion to their promolecular densities, so the shared density at a
        contact is divided by the electrons rather than by the fit's choice of
        functions. Each fragment's potential is its :meth:`select_atoms`
        potential plus a correction integrated on the contact atoms' cells;
        the two potentials sum exactly to :meth:`potential`. See
        :mod:`qpet.partition` for the method and the options
        (``grid_level``, ``contact_radius``, ``environment_radius``, ``softcore``).

        :param atoms_a: Atom indices of the first fragment.
        :param atoms_b: Atom indices of the second fragment; ``None`` takes the
            complement of ``atoms_a``.
        :param options: Forwarded to :class:`~qpet.partition.HirshfeldSplit`.
        :return: ``(fragment_a, fragment_b)``, two
            :class:`~qpet.partition.FragmentDensity` scalar fields.
        """
        from qpet.partition import HirshfeldSplit

        return HirshfeldSplit(self, atoms_a, atoms_b, **options).fragments  # type: ignore[arg-type]

    @property
    def positions(self) -> np.ndarray:
        """Atomic positions in Angstrom, shape (N, 3)."""
        return self.auxmol.atom_coords(unit="Ang")

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers (element identity ``Z``), shape (N,)."""
        return atomic_numbers_of(self.auxmol)

    @property
    def nuclear_charges(self) -> np.ndarray:
        """Effective nuclear charges (``Z`` minus ECP core electrons), shape (N,)."""
        return nuclear_charges_of(self.auxmol)

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

    def potential(
        self, pts: np.ndarray, batch_size: int = 2000, fast: bool | None = None
    ) -> np.ndarray:
        """
        Compute ESP from the RI coefficients.

        Two implementations of the same quantity: libcint, and the closed form
        in :class:`~qpet.esp_fast.FastESP`, which is much faster on a GPU and
        slower than libcint anywhere else. The fast path runs in float32 and so
        carries ~1e-5 relative error, below that of a float32 density prediction
        but not what ``fast=False`` gives you.

        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per batch (controls peak memory)
        :param fast: Which implementation to use. ``None`` (the default) picks
            the closed form when it pays and can be trusted: a CUDA device, at
            least :data:`FAST_ESP_MIN_POINTS` points, and an auxiliary basis
            whose fit passes its residual check. ``True`` forces it and raises
            if the fit fails; ``False`` forces libcint.
        :return: Electrostatic potential in atomic units, shape (N,)
        """
        pts_bohr = pts / BOHR_TO_ANG
        v_nuc = nuclear_potential(
            self.auxmol.atom_coords(), self.nuclear_charges, pts_bohr
        )

        if fast is not False:
            evaluator = self._fast_esp(len(pts_bohr), force=bool(fast))
            if evaluator is not None:
                return v_nuc - evaluator.potential(self.coeffs, pts_bohr)

        v_ele = np.zeros(len(pts_bohr))
        for p0 in range(0, len(pts_bohr), batch_size):
            p1 = min(p0 + batch_size, len(pts_bohr))
            fakemol = gto.fakemol_for_charges(pts_bohr[p0:p1])
            v_Pp = gto.mole.intor_cross("int2c2e", self.auxmol, fakemol)
            v_ele[p0:p1] = self.coeffs @ v_Pp
        return v_nuc - v_ele

    def _fast_esp(self, n_points: int, *, force: bool) -> FastESP | None:
        """This density's closed-form evaluator, built once, or ``None``.

        :param n_points: Size of the grid being asked for; too small a grid does
            not repay the fit.
        :param force: Build regardless of device and grid size, and let a failed
            fit raise.
        """
        import torch

        cuda = torch.cuda.is_available()
        # The gate comes first: a cached evaluator built for an earlier forced
        # call must not turn every later grid into a closed-form one, least of
        # all on a CPU, where it is slower than libcint.
        if not force and (self._no_fast or not cuda or n_points < FAST_ESP_MIN_POINTS):
            return None
        if self._fast is not None:
            return self._fast

        from qpet.esp_fast import FastESP

        device = "cuda" if cuda else "cpu"
        # float32 is well below the error of the density being evaluated, but
        # only on a GPU is it the faster choice.
        dtype = torch.float32 if cuda else torch.float64
        try:
            self._fast = FastESP(self.auxmol, device=device, dtype=dtype)
        except RuntimeError as error:
            # Asked for explicitly a failed fit is an error; left to itself it is
            # a reason to use libcint.
            if force:
                raise
            self._no_fast = True
            logger.info("closed-form ESP unavailable, using libcint: %s", error)
        return self._fast

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

    def normalize_charge(
        self,
        n_electrons: float | None = None,
        *,
        metric: str = "identity",
        max_shift: float | None = None,
        label: str | None = None,
    ) -> RIDensity:
        """
        Return a copy whose density integrates to the exact electron count.

        Minimum-norm additive shift of the coefficients; see
        :func:`qpet.moments.normalize_charge_additive` for the two metrics.

        :param n_electrons: Target electron count; defaults to the ECP-aware
            count of the auxiliary molecule (:func:`qpet.electron_count`).
        :param metric: ``"identity"`` (s-channel shift, shape channels
            untouched) or ``"coulomb"`` (minimum Coulomb self-energy, the one
            that repairs the potential at the molecular surface).
        :param max_shift: Refuse (with a warning) to move more charge than
            this; see :func:`qpet.moments.normalize_charge_additive`.
        :param label: Name of the system for that warning.
        """
        from qpet.conditioning import electron_count
        from qpet.moments import normalize_charge_additive

        if n_electrons is None:
            n_electrons = electron_count(self.auxmol)
        return RIDensity(
            self.auxmol,
            normalize_charge_additive(
                self.auxmol,
                self.coeffs,
                n_electrons,
                metric=metric,
                max_shift=max_shift,
                label=label,
            ),
        )

    def constrain_dipole(self, mu_target: np.ndarray) -> RIDensity:
        """
        Return a copy whose electronic dipole matches ``mu_target``.

        Minimum-norm, charge-neutral correction. The target is the electronic
        dipole at the centre of nuclear charge, in e·Bohr.
        """
        from qpet.moments import apply_dipole_correction, get_ri_dipole

        z = self.nuclear_charges.astype(float)
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
