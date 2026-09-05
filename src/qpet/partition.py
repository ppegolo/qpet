"""Real-space (Hirshfeld) partition of an RI density into two molecular fragments.

:meth:`RIDensity.select_atoms` splits a density by *bookkeeping*: every
auxiliary function belongs to the atom it is centred on, so the "ligand
density" is the sum of the ligand atoms' functions. That is exact and additive,
but where two molecules touch their functions overlap heavily and the Coulomb
fit is free to describe the same piece of density with either set. Which set
it picks is decided by the fit's conditioning, not by the electrons: on
protein-ligand complexes the two bookkeepings (this one and a real-space one)
disagree by up to an electron on how much charge "is the ligand's", and a
density model that gets the electron density right cannot be expected to
reproduce the choice.

The Hirshfeld partition assigns each point in space to the fragments in
proportion to their promolecular densities,
``w_A(r) = rho_A^0(r) / (rho_A^0(r) + rho_B^0(r))``, with ``rho_X^0`` the sum
of spherically averaged free-atom densities of the fragment's atoms. It is
smooth, basis-independent, and lets each molecule keep its own share of the
density it shares with its partner.

Evaluating it costs a grid, but only near the interface: the fragment's
Hirshfeld potential is its :meth:`~RIDensity.select_atoms` potential plus the
potential of the correction density ``w_A(r) rho(r) - rho_A^RI(r)``, and that
integrand vanishes wherever the two bookkeepings agree -- deep inside either
molecule, and everywhere far from the contact. So the correction is
integrated on the Becke quadrature cells of the *contact atoms* only, using the
functions of atoms in their neighbourhood, and represented as point charges.
The total electrostatic potential stays analytic and the cost scales with the
size of the interface, not of the receptor. The two fragments' corrections are
equal and opposite, so ``A.potential(p) + B.potential(p) == rho.potential(p)``
holds to machine precision, exactly as for :meth:`~RIDensity.select_atoms`.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from qpet.conditioning import atomic_numbers_of, nuclear_charges_of
from qpet.conversions import BOHR_TO_ANG

if TYPE_CHECKING:
    from pyscf import gto

    from qpet.fields import RIDensity

logger = logging.getLogger(__name__)

__all__ = ["FragmentDensity", "HirshfeldSplit", "free_atom_density"]

#: Orbital basis of the free-atom calculations behind the promolecular weights.
FREE_ATOM_BASIS = "def2-svp"
#: Elements from this atomic number on are computed with the matching ECP
#: (the def2 bases are only defined against one there).
_FIRST_ECP_ELEMENT = 37
#: Point charges below this magnitude (electrons) are dropped from the
#: correction; they carry nothing and the potential loop is linear in their number.
_CHARGE_FLOOR = 1e-14


@cache
def _free_atom_table(z: int, ecp: str | None) -> tuple[np.ndarray, np.ndarray]:
    """``(r, rho)`` of the spherically averaged free-atom density, r in Bohr."""
    from pyscf import gto
    from pyscf.scf import atom_hf

    symbol = gto.mole._std_symbol(z)
    if ecp is None and z >= _FIRST_ECP_ELEMENT:
        ecp = FREE_ATOM_BASIS
    mol = gto.M(
        atom=f"{symbol} 0 0 0", basis=FREE_ATOM_BASIS, ecp=ecp, spin=None, verbose=0
    )
    with warnings.catch_warnings():
        # PySCF's atomic solver still calls its own deprecated linear-dependence
        # helper; nothing here to act on.
        warnings.simplefilter("ignore", DeprecationWarning)
        _, _, mo_coeff, mo_occ = atom_hf.get_atm_nrhf(mol)[symbol]
    dm = (mo_coeff * mo_occ) @ mo_coeff.T
    # Dense near the nucleus, linear beyond: interpolation stays a plain
    # ``np.interp`` on r, the hot loop of the partition.
    radii = np.concatenate(
        [np.linspace(0.0, 1.0, 2001)[:-1], np.linspace(1.0, 20.0, 7601)]
    )
    points = np.zeros((len(radii), 3))
    points[:, 2] = radii
    ao = mol.eval_gto("GTOval", points)
    rho = np.einsum("pi,ij,pj->p", ao, dm, ao)
    return radii, rho


def free_atom_density(
    z: int, distances_bohr: np.ndarray, *, ecp: str | None = None
) -> np.ndarray:
    """
    Spherically averaged free-atom density of element ``z`` at radial distances.

    A restricted, spherically averaged Hartree-Fock atom in
    :data:`FREE_ATOM_BASIS`, tabulated once per element and interpolated
    log-log. Under an ECP the core electrons are absent, matching a density
    that was computed with the same ECP.

    :param z: Atomic number.
    :param distances_bohr: Distances from the nucleus in Bohr, any shape.
    :param ecp: ECP name, or ``None`` for an all-electron atom (elements from
        rubidium on always use ``def2-svp``'s ECP).
    :return: Density in electrons / Bohr^3, same shape as ``distances_bohr``.
    """
    radii, rho = _free_atom_table(int(z), ecp)
    return np.interp(np.asarray(distances_bohr, dtype=float), radii, rho, right=0.0)


def hirshfeld_weights(
    points_bohr: np.ndarray,
    centres_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    in_fragment: np.ndarray,
    *,
    ecp: str | None = None,
) -> np.ndarray:
    """
    Hirshfeld weight of one fragment, ``rho_A^0 / rho_total^0``, at points.

    :param points_bohr: Points, shape (N, 3), Bohr.
    :param centres_bohr: Nuclear positions, shape (M, 3), Bohr.
    :param atomic_numbers: Element of every nucleus, shape (M,).
    :param in_fragment: Boolean mask, shape (M,): nucleus belongs to the fragment.
    :param ecp: ECP of the free atoms, see :func:`free_atom_density`.
    :return: Weights in ``[0, 1]``, shape (N,); ``0.5`` where no atom has density.
    """
    numerator = np.zeros(len(points_bohr))
    total = np.zeros(len(points_bohr))
    atomic_numbers = np.asarray(atomic_numbers)
    in_fragment = np.asarray(in_fragment, dtype=bool)
    for z in np.unique(atomic_numbers):
        of_z = atomic_numbers == z
        distances = np.linalg.norm(
            points_bohr[:, None, :] - centres_bohr[of_z][None, :, :], axis=2
        )
        rho = free_atom_density(z, distances, ecp=ecp)  # (n_points, n_atoms_z)
        total += rho.sum(axis=1)
        numerator += rho[:, in_fragment[of_z]].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.where(total > 0, numerator / total, 0.5)
    return weights


def _becke_cell_weights(
    points_bohr: np.ndarray,
    centres_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: int,
) -> np.ndarray:
    """
    Becke's fuzzy-cell weight of atom ``cell`` at points, among the given atoms.

    PySCF's own kernel (``VXCgen_grid``: Becke's three-fold smoothing with
    Treutler's atomic-size adjustment on Bragg radii, the DFT grid default),
    called on a neighbourhood instead of the whole molecule: the weight of a
    cell is decided by its neighbours, so the cost stays local.
    """
    import ctypes

    from pyscf.dft import radi
    from pyscf.dft.gen_grid import libdft

    radii = np.sqrt(radi.BRAGG_RADII[np.asarray(atomic_numbers, dtype=int)]) + 1e-200
    ratio = radii.reshape(-1, 1) * (1.0 / radii)
    adjust = np.clip(0.25 * (ratio.T - ratio), -0.5, 0.5)
    table = np.ascontiguousarray(adjust.ravel())
    coords = np.asarray(points_bohr, dtype=float, order="F")
    atoms = np.ascontiguousarray(centres_bohr, dtype=float)
    n_atoms, n_points = len(atoms), len(coords)
    pbecke = np.empty((n_atoms, n_points))
    libdft.VXCgen_grid(
        pbecke.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        atoms.ctypes.data_as(ctypes.c_void_p),
        table.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n_atoms),
        ctypes.c_int(n_points),
    )
    return pbecke[cell] / pbecke.sum(axis=0)


def _pairwise_within(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask over rows of ``a``: within ``radius`` of any row of ``b``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(b).query(a, k=1, distance_upper_bound=radius * (1 + 1e-12))
    return np.isfinite(distance)


class HirshfeldSplit:
    """
    Hirshfeld partition of an :class:`~qpet.RIDensity` into two fragments.

    Built once per complex; :attr:`fragments` are the two
    :class:`FragmentDensity` objects. See the module docstring for the method.

    :param density: The density of the whole complex.
    :param atoms_a: Atom indices of the first fragment (the ligand, typically).
    :param atoms_b: Atom indices of the second fragment; ``None`` takes every
        other atom. Together the two must cover the complex exactly once.
    :param grid_level: PySCF grid level of the atomic quadratures. The
        correction density is the partner's diffuse tails, smooth everywhere,
        so level 1 (the default) is accurate to about 2e-4 Ha at the surface;
        level 2 to 1e-4 at twice the cost.
    :param overlap_radius: A grid point is integrated only if it lies within
        this distance (Angstrom) of an atom of *each* fragment: elsewhere one
        of the two densities that make up the correction is negligible.
    :param cell_radius: Atomic cells considered are those of atoms within this
        distance (Angstrom) of the other fragment; a cell farther away cannot
        reach the overlap zone.
    :param neighbour_radius: Atoms within this distance (Angstrom) of a cell's
        atom define its Becke weight and its promolecular densities, and supply
        the auxiliary functions evaluated on it.
    :param function_radius: Only auxiliary functions of atoms within this
        distance (Angstrom) of a cell's atom are evaluated on it; the most
        diffuse functions of the auxiliary bases are below 1e-7 beyond it.
    :param weight_floor: Grid points whose Becke quadrature weight (Bohr^3) is
        below this are dropped before anything is evaluated on them; they lie
        deep inside a neighbour's cell.
    :param softcore: Regularisation length (Bohr) of ``1/r`` between a
        correction charge and an evaluation point, for points that happen to
        fall onto a grid point. Evaluation points are meant to be outside the
        molecules (surface points); inside, the correction is a quadrature of
        a singular kernel and correspondingly rough.
    :param threads: Cells are integrated in this many threads (PySCF's
        integral and grid kernels release the GIL); ``None`` uses up to eight.
    """

    def __init__(
        self,
        density: RIDensity,
        atoms_a: Sequence[int],
        atoms_b: Sequence[int] | None = None,
        *,
        grid_level: int = 1,
        overlap_radius: float = 6.0,
        cell_radius: float = 10.0,
        neighbour_radius: float = 7.0,
        function_radius: float = 7.0,
        weight_floor: float = 1e-10,
        softcore: float = 0.05,
        threads: int | None = None,
    ) -> None:
        from pyscf import dft

        auxmol = density.auxmol
        n_atoms = auxmol.natm
        a = np.array(sorted({int(i) for i in atoms_a}), dtype=int)
        if atoms_b is None:
            b = np.setdiff1d(np.arange(n_atoms), a)
        else:
            b = np.array(sorted({int(i) for i in atoms_b}), dtype=int)
        if len(a) == 0 or len(b) == 0:
            raise ValueError("both fragments need at least one atom.")
        if np.intersect1d(a, b).size or len(a) + len(b) != n_atoms:
            raise ValueError(
                "the two fragments must partition the complex: disjoint and "
                f"covering all {n_atoms} atoms (got {len(a)} + {len(b)})."
            )
        if a.min() < 0 or max(a.max(), b.max()) >= n_atoms:
            raise IndexError(f"atom index out of range for {n_atoms}-atom density.")

        self.density = density
        self.atoms = (a, b)
        self.grid_level = grid_level
        self.softcore = softcore
        self.ecp = auxmol.ecp if auxmol.ecp else None
        in_a = np.zeros(n_atoms, dtype=bool)
        in_a[a] = True
        self._in_a = in_a
        self._numbers = atomic_numbers_of(auxmol)
        self._centres_bohr = auxmol.atom_coords()

        centres = auxmol.atom_coords(unit="Ang")
        # Cells worth visiting: atoms close enough to the other fragment for
        # their cell to reach the overlap zone.
        candidate = np.zeros(n_atoms, dtype=bool)
        candidate[a] = _pairwise_within(centres[a], centres[b], cell_radius)
        candidate[b] = _pairwise_within(centres[b], centres[a], cell_radius)
        self.cell_atoms = np.flatnonzero(candidate)
        self.environment_atoms = np.zeros(0, dtype=int)
        self.points_bohr = np.zeros((0, 3))
        self.charges = np.zeros(0)
        if len(self.cell_atoms) == 0:
            # Nothing is shared: the Hirshfeld split is the RI split.
            self._finish()
            return

        # Every promolecular density that can matter on a visited cell belongs
        # to an atom within neighbour_radius of it; every function, within
        # function_radius. Both neighbourhoods are cheap to enumerate.
        environment = _pairwise_within(centres, centres[candidate], neighbour_radius)
        self.environment_atoms = np.flatnonzero(environment)
        grids = dft.gen_grid.Grids(auxmol)
        grids.level = grid_level
        atom_grids = grids.gen_atomic_grids(
            auxmol, grids.atom_grid, grids.radi_method, grids.level, grids.prune
        )
        overlap_bohr = overlap_radius / BOHR_TO_ANG
        neighbour_bohr = neighbour_radius / BOHR_TO_ANG
        function_bohr = function_radius / BOHR_TO_ANG
        all_centres = self._centres_bohr
        numbers = self._numbers
        aoslice = auxmol.aoslice_by_atom()

        def integrate_cell(atom: int) -> tuple[np.ndarray, np.ndarray] | None:
            centre = all_centres[atom]
            cell_points, volumes = atom_grids[auxmol.atom_symbol(int(atom))]
            cell_points = cell_points + centre
            neighbours = np.flatnonzero(
                _pairwise_within(all_centres, centre[None, :], neighbour_bohr)
            )
            near_a = neighbours[in_a[neighbours]]
            near_b = neighbours[~in_a[neighbours]]
            selected = _pairwise_within(cell_points, all_centres[near_a], overlap_bohr)
            selected &= _pairwise_within(cell_points, all_centres[near_b], overlap_bohr)
            if not selected.any():
                return None
            cell_points = cell_points[selected]
            volumes = volumes[selected]
            cell_weights = volumes * _becke_cell_weights(
                cell_points,
                all_centres[neighbours],
                numbers[neighbours],
                int(np.flatnonzero(neighbours == atom)[0]),
            )
            heavy = cell_weights > weight_floor
            cell_points = cell_points[heavy]
            cell_weights = cell_weights[heavy]
            if len(cell_points) == 0:
                return None
            functions = np.flatnonzero(
                _pairwise_within(all_centres, centre[None, :], function_bohr)
            )
            local = density.select_atoms(functions.tolist())
            counts = (aoslice[functions, 3] - aoslice[functions, 2]).astype(int)
            local_coeffs_a = local.coeffs * np.repeat(in_a[functions], counts)
            ao = local.auxmol.eval_gto("GTOval", cell_points)
            rho = ao @ local.coeffs
            rho_a_ri = ao @ local_coeffs_a
            w_a = hirshfeld_weights(
                cell_points,
                all_centres[neighbours],
                numbers[neighbours],
                in_a[neighbours],
                ecp=self.ecp,
            )
            q = cell_weights * (w_a * rho - rho_a_ri)
            significant = np.abs(q) > _CHARGE_FLOOR
            return cell_points[significant], q[significant]

        from concurrent.futures import ThreadPoolExecutor

        workers = threads or min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cells = [
                c
                for c in pool.map(integrate_cell, self.cell_atoms.tolist())
                if c is not None
            ]
        points = [c[0] for c in cells]
        charges = [c[1] for c in cells]
        if points:
            self.points_bohr = np.concatenate(points)
            self.charges = np.concatenate(charges)
        self._finish()

    def _finish(self) -> None:
        a, b = self.atoms
        self.fragments = (
            FragmentDensity(self, self.density.select_atoms(a.tolist()), +1.0),
            FragmentDensity(self, self.density.select_atoms(b.tolist()), -1.0),
        )

    @property
    def electron_transfer(self) -> float:
        """
        Electrons the Hirshfeld partition moves to fragment A relative to the
        RI split (``N_A^Hirshfeld - N_A^RI``); fragment B loses the same.
        """
        return float(self.charges.sum())

    def correction_potential(
        self, pts: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        """
        Electronic potential of the correction density at points, for fragment A.

        ``-sum_g q_g / |r - r_g|`` (electrons lower the potential); fragment B's
        correction is the negative of this.

        :param pts: Points in Angstrom, shape (N, 3).
        :param batch_size: Points per block of the ``(points x charges)`` kernel.
        :return: Potential in atomic units, shape (N,).
        """
        pts_bohr = np.asarray(pts, dtype=float) / BOHR_TO_ANG
        if len(self.charges) == 0:
            return np.zeros(len(pts_bohr))
        import torch

        # The kernel sum is a dense (points x charges) product: on a GPU it is
        # instantaneous, on a CPU torch matches numpy and keeps one code path.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        eps2 = self.softcore * self.softcore
        charges = torch.as_tensor(self.charges, dtype=torch.float64, device=device)
        sources = torch.as_tensor(self.points_bohr, dtype=torch.float64, device=device)
        targets = torch.as_tensor(pts_bohr, dtype=torch.float64, device=device)
        out = torch.empty(len(targets), dtype=torch.float64, device=device)
        charge_block = max(1, 8_000_000 // max(batch_size, 1))
        for p0 in range(0, len(targets), batch_size):
            block = targets[p0 : p0 + batch_size]
            acc = torch.zeros(len(block), dtype=torch.float64, device=device)
            for g0 in range(0, len(charges), charge_block):
                d2 = torch.cdist(block, sources[g0 : g0 + charge_block]) ** 2
                acc += torch.rsqrt(d2 + eps2) @ charges[g0 : g0 + charge_block]
            out[p0 : p0 + batch_size] = -acc
        return out.cpu().numpy()

    def weight(self, pts: np.ndarray) -> np.ndarray:
        """Hirshfeld weight of fragment A at points (Angstrom), shape (N,)."""
        return hirshfeld_weights(
            np.asarray(pts, dtype=float) / BOHR_TO_ANG,
            self._centres_bohr,
            self._numbers,
            self._in_a,
            ecp=self.ecp,
        )


class FragmentDensity:
    """
    One fragment of a :class:`HirshfeldSplit`; a :class:`~qpet.ScalarField`.

    Geometry (positions, atomic numbers, nuclear charges) is the fragment's
    atoms; :attr:`ri_part` is the :meth:`~qpet.RIDensity.select_atoms` density
    of the same atoms, whose potential this object corrects.

    :param split: The partition this fragment belongs to.
    :param ri_part: The fragment's RI-split density.
    :param sign: ``+1`` for fragment A, ``-1`` for fragment B.
    """

    def __init__(self, split: HirshfeldSplit, ri_part: RIDensity, sign: float) -> None:
        self.split = split
        self.ri_part = ri_part
        self.sign = float(sign)

    @property
    def auxmol(self) -> gto.Mole:
        """Auxiliary ``Mole`` of the fragment's atoms (the RI part's)."""
        return self.ri_part.auxmol

    @property
    def coeffs(self) -> np.ndarray:
        """RI coefficients of the fragment's own functions (the RI part's)."""
        return self.ri_part.coeffs

    @property
    def positions(self) -> np.ndarray:
        """Atomic positions in Angstrom, shape (N, 3)."""
        return self.ri_part.positions

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers (element identity ``Z``), shape (N,)."""
        return atomic_numbers_of(self.ri_part.auxmol)

    @property
    def nuclear_charges(self) -> np.ndarray:
        """Effective nuclear charges (``Z`` minus ECP core electrons), shape (N,)."""
        return nuclear_charges_of(self.ri_part.auxmol)

    @property
    def electrons(self) -> float:
        """Electron count of the fragment under the Hirshfeld partition."""
        return (
            float(-np.sum(self.ri_part.charge()))
            + self.sign * self.split.electron_transfer
        )

    def density(self, pts: np.ndarray, batch_size: int = 5000) -> np.ndarray:
        """
        Hirshfeld-weighted electron density ``w_A(r) rho(r)`` at points.

        :param pts: Evaluation points in Angstrom, shape (N, 3).
        :param batch_size: Points evaluated per AO batch.
        :return: Electron density in electrons / Bohr^3, shape (N,).
        """
        weight = self.split.weight(pts)
        if self.sign < 0:
            weight = 1.0 - weight
        return weight * self.split.density.density(pts, batch_size=batch_size)

    def potential(
        self, pts: np.ndarray, batch_size: int = 2000, fast: bool | None = None
    ) -> np.ndarray:
        """
        Electrostatic potential of the fragment (nuclei plus Hirshfeld electrons).

        :param pts: Evaluation points in Angstrom, shape (N, 3).
        :param batch_size: Passed to :meth:`RIDensity.potential` for the RI part.
        :param fast: Passed to :meth:`RIDensity.potential` for the RI part.
        :return: Potential in atomic units, shape (N,).
        """
        analytic = self.ri_part.potential(pts, batch_size=batch_size, fast=fast)
        return analytic + self.sign * self.split.correction_potential(pts)

    def charge(self, *, include_nuclear: bool = False) -> float:
        """Electronic charge ``-N_elec`` (or total charge with nuclei)."""
        electronic = -self.electrons
        if include_nuclear:
            return electronic + float(self.nuclear_charges.sum())
        return electronic

    def __repr__(self) -> str:
        return (
            f"FragmentDensity(natm={self.ri_part.auxmol.natm}, "
            f"sign={self.sign:+.0f}, electrons={self.electrons:.4f})"
        )
