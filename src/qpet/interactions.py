"""Interaction energies between two fragments of one density.

:mod:`qpet.partition` splits a density in real space; this module turns that
split into the two interaction channels that can be *computed* from it rather
than fitted:

- :func:`coulomb_interaction_energy`, the classical electrostatic interaction
  of the two fragments including their nuclei,
  ``E = sum_A Z_A V_B(R_A) - int rho_A(r) V_B(r) dr``.
- :func:`hirshfeld_dispersion_energy`, the Tkatchenko-Scheffler pairwise
  dispersion between the fragments. It needs one extra integral per atom, the
  Hirshfeld effective volume :func:`hirshfeld_volume_ratios`, which rescales
  the free-atom polarisability, ``C6`` and radius of every atom by how much of
  its free-atom volume it keeps in the molecule.

Both are quadratures on Becke atomic cells built with the same machinery as
:mod:`qpet.partition`, restricted to the neighbourhood of the fragment of
interest: a Hirshfeld fragment density has no support far from its own atoms,
and a damped ``1/R^6`` sum is short-ranged by construction.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from qpet.conditioning import atomic_numbers_of
from qpet.conversions import BOHR_TO_ANG
from qpet.fields import RIDensity
from qpet.partition import (
    FragmentDensity,
    becke_cell_weights,
    free_atom_density,
    hirshfeld_weights,
    pairwise_within,
)

if TYPE_CHECKING:
    from qpet.fields import ScalarField
    from qpet.partition import HirshfeldSplit

logger = logging.getLogger(__name__)

__all__ = [
    "HARTREE_TO_KCAL",
    "analytic_coulomb_interaction_energy",
    "coulomb_interaction_energy",
    "grid_coulomb_interaction_energy",
    "free_atom_volume",
    "hirshfeld_dispersion_energy",
    "hirshfeld_volume_ratios",
    "ts_dispersion_energy",
]

#: Hartree to kcal/mol (CODATA Hartree, 1 kcal = 4184 J, exact Avogadro).
HARTREE_TO_KCAL = 627.5094740631

#: Steepness of the Tkatchenko-Scheffler Fermi damping, PRL 102, 073005 (2009).
TS_DAMPING = 20.0
#: Range parameter of the same damping; the published PBE value.
TS_RANGE = 0.94


@cache
def _ts_free_atom_parameters(symbol: str) -> tuple[float, float, float] | None:
    """
    ``(alpha, C6, R0)`` of the free atom: Bohr^3, Hartree Bohr^6, Bohr.

    Taken from ASE (a dependency already), which is where the
    Tkatchenko-Scheffler reference values live in this stack:
    ``vdWDB_alphaC6`` (static polarisabilities and ``C6`` of Chu and Dalgarno,
    J. Chem. Phys. 121, 4083 (2004), as tabulated in the TS paper) and, as in
    ASE's own TS implementation, Grimme's D2 van der Waals radii
    (J. Comput. Chem. 27, 1787 (2006)) as the free-atom radii. ``None`` when
    the element is in neither table.
    """
    from ase.calculators.vdwcorrection import vdWDB_alphaC6, vdWDB_Grimme06jcc

    if symbol not in vdWDB_alphaC6 or symbol not in vdWDB_Grimme06jcc:
        return None
    alpha, c6 = np.asarray(vdWDB_alphaC6[symbol], dtype=float)
    r0_ang = float(np.asarray(vdWDB_Grimme06jcc[symbol], dtype=float)[1])
    return float(alpha), float(c6), r0_ang / BOHR_TO_ANG


@cache
def free_atom_volume(z: int, ecp: str | None = None) -> float:
    """
    Volume ``int r^3 rho^0(r) dr`` of the spherically averaged free atom, Bohr^3.

    The reference of the Hirshfeld ratio, computed from the *same* free-atom
    densities that define the Hirshfeld weights
    (:func:`qpet.partition.free_atom_density`: restricted, spherically averaged
    Hartree-Fock in :data:`qpet.partition.FREE_ATOM_BASIS`, ECP-aware), so the
    ratio of a free atom is exactly one.

    :param z: Atomic number.
    :param ecp: ECP name, or ``None``; see :func:`~qpet.partition.free_atom_density`.
    :return: The volume in Bohr^3.
    """
    r = np.linspace(0.0, 20.0, 40001)
    rho = free_atom_density(int(z), r, ecp=ecp)
    return float(np.trapezoid(4.0 * np.pi * r**5 * rho, r))


def _grid_over(
    symbols: Sequence[str], coords_ang: np.ndarray, grid_level: int
) -> tuple[np.ndarray, np.ndarray]:
    """Becke grid (points in Angstrom, weights in Bohr^3) over bare nuclei."""
    from pyscf import dft, gto

    # Only nuclei and elements enter a grid; a one-function basis keeps the
    # Mole buildable for any element, ECP or not.
    mol = gto.Mole()
    mol.atom = [(s, xyz) for s, xyz in zip(symbols, coords_ang, strict=True)]
    mol.basis = {s: [[0, [1.0, 1.0]]] for s in set(symbols)}
    mol.unit = "Angstrom"
    mol.spin = int(sum(gto.charge(s) for s in symbols)) % 2
    mol.verbose = 0
    mol.build()
    grid = dft.gen_grid.Grids(mol)
    grid.level = grid_level
    grid.build()
    return grid.coords * BOHR_TO_ANG, grid.weights


class _FragmentEvaluator:
    """
    ``field``'s electron density at points, with a truncated function set.

    A :class:`~qpet.partition.FragmentDensity` evaluates the *whole complex*'s
    auxiliary basis and then weights it, which on a protein site is tens of
    thousands of functions and turns a grid integral into minutes. The density
    at a point is set by the functions around it, so the complex is truncated
    once to the atoms within ``radius`` of the fragment, and the Hirshfeld
    weight is built from the same neighbourhood (a distant atom changes
    neither). Points where that weight is below ``weight_floor`` -- deep inside
    the partner, where the fragment has no density -- are skipped before
    anything is evaluated on them. Anything that is not a fragment of a
    Hirshfeld split is evaluated as it comes.

    :param field: The density to evaluate.
    :param radius: Atoms within this distance (Angstrom) of one of the
        fragment's own atoms are kept.
    :param weight_floor: Hirshfeld weight below which a point is taken as zero.
    """

    def __init__(
        self, field: ScalarField, radius: float, weight_floor: float = 1e-8
    ) -> None:
        self.field = field
        self.weight_floor = weight_floor
        if not isinstance(field, FragmentDensity):
            return
        split = field.split
        auxmol = split.density.auxmol
        near = np.flatnonzero(
            pairwise_within(auxmol.atom_coords(unit="Ang"), field.positions, radius)
        )
        self.local = split.density.select_atoms(near.tolist())
        self.centres_bohr = auxmol.atom_coords()[near]
        self.numbers = atomic_numbers_of(auxmol)[near]
        self.ecp = split.ecp
        self.in_fragment = np.isin(near, split.atoms[0 if field.sign > 0 else 1])

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Electron density in electrons / Bohr^3 at points (Angstrom)."""
        if not isinstance(self.field, FragmentDensity):
            return np.asarray(self.field.density(points), dtype=float)
        weights = hirshfeld_weights(
            np.asarray(points, dtype=float) / BOHR_TO_ANG,
            self.centres_bohr,
            self.numbers,
            self.in_fragment,
            ecp=self.ecp,
        )
        keep = weights > self.weight_floor
        density = np.zeros(len(points))
        if keep.any():
            density[keep] = weights[keep] * self.local.density(points[keep])
        return density


def _electronic_potential(density: RIDensity, points: np.ndarray) -> np.ndarray:
    """``int n(r') / |r - r'| dr'``, an RI density's electronic potential."""
    from qpet.fields import nuclear_potential

    v_nuc = nuclear_potential(
        density.auxmol.atom_coords(),
        density.nuclear_charges,
        np.asarray(points, dtype=float) / BOHR_TO_ANG,
    )
    return v_nuc - density.potential(points)


def analytic_coulomb_interaction_energy(a: ScalarField, b: ScalarField) -> float:
    """
    Coulomb interaction of two fragments without a quadrature, in Hartree.

    The same energy as :func:`coulomb_interaction_energy`, assembled from
    closed-form pieces instead of a grid. Writing the electron density of
    fragment ``a`` as its RI part plus the Hirshfeld correction,
    ``n_a = n_a^RI + s d`` (``s = +-1``, ``d`` the correction the split carries
    as point charges at the grid sites ``r_g``, ``n_b = n_b^RI - s d``),

    ``E = sum_A Z_A V_b(R_A) - sum_B Z_B V_a^RI,ele(R_B) + c_a^T J_ab c_b
    - s sum_g q_g [V_a^RI,ele(r_g) + V_b^RI(r_g)] + sum_g q_g C(r_g)``

    with ``J_ab`` the two-centre Coulomb integrals between the two auxiliary
    bases and ``C`` the correction's own potential. Every term is exact except
    the last, the self-interaction of the correction density, which is a double
    sum over the correction sites and inherits the point-charge representation
    (see :attr:`HirshfeldSplit.softcore`); it is the one place where this
    result is no better than the grid one.

    Plain :class:`~qpet.RIDensity` fragments have no correction and every term
    is then exact.

    :param a: One fragment.
    :param b: The other fragment: an :class:`~qpet.RIDensity` if ``a`` is one,
        otherwise from the same split as ``a``.
    :return: Interaction energy in Hartree; negative is attractive.
    """
    from pyscf import gto

    split: HirshfeldSplit | None
    if isinstance(a, RIDensity) and isinstance(b, RIDensity):
        split, sign, ri_a, ri_b = None, 1.0, a, b
    elif (
        isinstance(a, FragmentDensity)
        and isinstance(b, FragmentDensity)
        and a.split is b.split
    ):
        split, sign, ri_a, ri_b = a.split, a.sign, a.ri_part, b.ri_part
    else:
        raise ValueError(
            "the analytic Coulomb energy needs two RIDensity objects or two "
            "fragments of the same split."
        )

    energy = float(np.sum(a.nuclear_charges * b.potential(a.positions)))
    energy -= float(
        np.sum(b.nuclear_charges * _electronic_potential(ri_a, b.positions))
    )
    j_ab = gto.mole.intor_cross("int2c2e", ri_a.auxmol, ri_b.auxmol)
    energy += float(ri_a.coeffs @ j_ab @ ri_b.coeffs)

    if split is None or len(split.charges) == 0:
        return energy

    sites = split.points_bohr * BOHR_TO_ANG
    energy -= sign * float(
        np.sum(
            split.charges * (_electronic_potential(ri_a, sites) + ri_b.potential(sites))
        )
    )
    energy += float(np.sum(split.charges * split.correction_potential(sites)))
    return energy


def grid_coulomb_interaction_energy(
    a: ScalarField,
    b: ScalarField,
    *,
    grid_level: int = 1,
    environment_radius: float = 4.0,
    support_radius: float = 6.0,
    function_radius: float = 6.0,
    weight_floor: float = 1e-10,
    batch_size: int = 50_000,
) -> float:
    """
    Electrostatic interaction energy of two fragment densities, in Hartree.

    ``E = sum_A Z_A V_B(R_A) - int rho_A(r) V_B(r) dr`` over the nuclei and the
    electrons of ``a`` in the total potential of ``b``. The electronic integral
    runs on a Becke grid built over ``a``'s atoms and the atoms of ``b`` within
    ``environment_radius`` of them -- the contact atoms, whose cells is where a
    fragment density is hardest to integrate -- with the points farther than
    ``support_radius`` from ``a`` dropped, since ``rho_a`` vanishes there.
    ``V_B`` is evaluated analytically at every retained point. The quadrature is
    the only approximation, so ``E(a, b)`` and ``E(b, a)`` agree to it, and for
    two well-separated fragments the result is the point-charge energy
    ``q_a q_b / R``.

    :param a: One fragment; its nuclei carry the ``Z_A``.
    :param b: The other fragment.
    :param grid_level: PySCF grid level (1 is enough at 1e-3 Hartree).
    :param environment_radius: Atoms of ``b`` within this many Angstrom of an
        atom of ``a`` join the grid.
    :param support_radius: Grid points farther than this from every atom of
        ``a`` are dropped.
    :param function_radius: Truncation radius of the density evaluation, see
        :class:`_FragmentEvaluator`.
    :param weight_floor: Grid points with a quadrature weight (Bohr^3) below
        this are dropped.
    :param batch_size: Points evaluated per block.
    :return: Interaction energy in Hartree; negative is attractive.
    """
    from pyscf.data.elements import ELEMENTS

    pos_a = np.asarray(a.positions, dtype=float)
    pos_b = np.asarray(b.positions, dtype=float)
    near = pairwise_within(pos_b, pos_a, environment_radius)
    symbols = [ELEMENTS[int(z)] for z in a.atomic_numbers] + [
        ELEMENTS[int(z)] for z, n in zip(b.atomic_numbers, near, strict=True) if n
    ]
    points, weights = _grid_over(
        symbols, np.concatenate([pos_a, pos_b[near]]), grid_level
    )
    keep = pairwise_within(points, pos_a, support_radius) & (weights > weight_floor)
    points, weights = points[keep], weights[keep]

    density_of_a = _FragmentEvaluator(a, function_radius)
    electronic = 0.0
    for p0 in range(0, len(points), batch_size):
        block = points[p0 : p0 + batch_size]
        electronic -= float(
            np.sum(
                weights[p0 : p0 + batch_size] * density_of_a(block) * b.potential(block)
            )
        )
    nuclear = float(np.sum(a.nuclear_charges * b.potential(pos_a)))
    return nuclear + electronic


def coulomb_interaction_energy(
    a: ScalarField, b: ScalarField, *, method: str = "analytic", **kwargs: object
) -> float:
    """
    Coulomb interaction energy of two fragment densities, in Hartree.

    ``E = sum_A Z_A V_b(R_A) - int rho_a(r) V_b(r) dr``, the classical
    electrostatic interaction of the two fragments with their nuclei.

    :param a: One fragment; its nuclei carry the ``Z_A``.
    :param b: The other fragment.
    :param method: ``"analytic"`` (the default,
        :func:`analytic_coulomb_interaction_energy`: closed form, no
        quadrature, seconds) or ``"grid"``
        (:func:`grid_coulomb_interaction_energy`: Becke quadrature, the
        reference implementation, minutes on a protein site).
    :param kwargs: Passed to the chosen implementation (the grid one only).
    :return: Interaction energy in Hartree; negative is attractive.
    """
    if method == "analytic":
        if kwargs:
            raise TypeError(f"unexpected arguments for the analytic method: {kwargs}")
        return analytic_coulomb_interaction_energy(a, b)
    if method == "grid":
        return grid_coulomb_interaction_energy(a, b, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown method {method!r}; use 'analytic' or 'grid'.")


def hirshfeld_volume_ratios(
    density: RIDensity,
    atoms: Sequence[int] | None = None,
    *,
    grid_level: int = 1,
    cell_radius: float = 6.0,
    neighbour_radius: float = 7.0,
    function_radius: float = 7.0,
    weight_floor: float = 1e-10,
    threads: int | None = None,
) -> np.ndarray:
    """
    Hirshfeld effective-volume ratios ``V_eff / V_free`` of selected atoms.

    ``V_eff_A = int w_A(r) |r - R_A|^3 rho(r) dr`` with the atomic Hirshfeld
    weight ``w_A = rho_A^0 / sum_B rho_B^0``, divided by
    :func:`free_atom_volume` of the same element. It measures how much of its
    free-atom volume an atom keeps in the molecule, and is what
    Tkatchenko-Scheffler rescales the free-atom dispersion parameters with.

    The ``r^3`` weight makes the integral sensitive to the tail of the density,
    where an RI fit is least accurate, so a free atom comes out slightly above
    one. The bias is shared by every atom and is absorbed by a coefficient
    fitted in front of the dispersion channel.

    The integral is a sum over Becke cells, as in :mod:`qpet.partition`: cells
    of every atom within ``cell_radius`` of a requested atom are visited, and
    each contributes to every requested atom whose Hirshfeld weight is not
    negligible on it. Truncating the cell list is safe because ``w_A`` falls
    off exponentially away from ``A``.

    :param density: Density of the whole system (weights are the *molecular*
        promolecular weights, not a fragment's).
    :param atoms: Atom indices to compute; ``None`` takes every atom.
    :param grid_level: PySCF grid level of the atomic quadratures.
    :param cell_radius: Cells of atoms within this distance (Angstrom) of a
        requested atom are integrated.
    :param neighbour_radius: Atoms within this distance (Angstrom) of a cell's
        atom define its Becke weight and its promolecular densities.
    :param function_radius: Only auxiliary functions of atoms within this
        distance (Angstrom) of a cell's atom are evaluated on it.
    :param weight_floor: Drop grid points whose quadrature weight (Bohr^3) is
        below this; they lie deep inside a neighbour's cell.
    :param threads: Cells are integrated in this many threads; ``None`` uses
        up to eight.
    :return: Ratios in the order of ``atoms``, shape ``(len(atoms),)``.
    """
    from pyscf import dft

    auxmol = density.auxmol
    n_atoms = auxmol.natm
    targets = np.arange(n_atoms) if atoms is None else np.array(atoms, dtype=int)
    if len(targets) == 0:
        return np.zeros(0)
    if targets.min() < 0 or targets.max() >= n_atoms:
        raise IndexError(f"atom index out of range for {n_atoms}-atom density.")

    ecp = auxmol.ecp if auxmol.ecp else None
    numbers = atomic_numbers_of(auxmol)
    centres_bohr = auxmol.atom_coords()
    centres = auxmol.atom_coords(unit="Ang")

    cells = np.flatnonzero(pairwise_within(centres, centres[targets], cell_radius))
    grids = dft.gen_grid.Grids(auxmol)
    grids.level = grid_level
    atom_grids = grids.gen_atomic_grids(
        auxmol, grids.atom_grid, grids.radi_method, grids.level, grids.prune
    )
    neighbour_bohr = neighbour_radius / BOHR_TO_ANG
    function_bohr = function_radius / BOHR_TO_ANG

    def integrate_cell(atom: int) -> np.ndarray:
        volumes = np.zeros(len(targets))
        centre = centres_bohr[atom]
        points, weights = atom_grids[auxmol.atom_symbol(int(atom))]
        points = points + centre
        neighbours = np.flatnonzero(
            pairwise_within(centres_bohr, centre[None, :], neighbour_bohr)
        )
        cell_weights = weights * becke_cell_weights(
            points,
            centres_bohr[neighbours],
            numbers[neighbours],
            int(np.flatnonzero(neighbours == atom)[0]),
        )
        heavy = cell_weights > weight_floor
        points, cell_weights = points[heavy], cell_weights[heavy]
        if len(points) == 0:
            return volumes
        functions = np.flatnonzero(
            pairwise_within(centres_bohr, centre[None, :], function_bohr)
        )
        local = density.select_atoms(functions.tolist())
        rho = local.auxmol.eval_gto("GTOval", points) @ local.coeffs

        distances = np.linalg.norm(
            points[:, None, :] - centres_bohr[neighbours][None, :, :], axis=2
        )
        promolecule = np.zeros(len(points))
        free = {}
        for i, j in enumerate(neighbours):
            free[int(j)] = free_atom_density(numbers[j], distances[:, i], ecp=ecp)
            promolecule += free[int(j)]
        near_target = np.isin(targets, neighbours)
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in np.flatnonzero(near_target):
                t = int(targets[k])
                i = int(np.flatnonzero(neighbours == t)[0])
                w = np.where(promolecule > 0, free[t] / promolecule, 0.0)
                volumes[k] = float(
                    np.sum(cell_weights * w * distances[:, i] ** 3 * rho)
                )
        return volumes

    from concurrent.futures import ThreadPoolExecutor

    workers = threads or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        effective = sum(pool.map(integrate_cell, cells.tolist()))

    reference = np.array([free_atom_volume(int(numbers[i]), ecp) for i in targets])
    return np.asarray(effective) / reference


def ts_dispersion_energy(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    volume_ratios: np.ndarray,
    atoms_a: Sequence[int],
    atoms_b: Sequence[int],
    *,
    s_r: float = TS_RANGE,
    damping: float = TS_DAMPING,
) -> float:
    """
    Tkatchenko-Scheffler dispersion between two groups of atoms, in Hartree.

    ``E = -sum_{A in a, B in b} f(R_AB) C6_AB / R_AB^6`` with Fermi damping
    ``f = 1 / (1 + exp(-damping (R / (s_r (R0_A + R0_B)) - 1)))``. Free-atom
    ``alpha``, ``C6`` and ``R0`` (:func:`_ts_free_atom_parameters`) are scaled
    by the Hirshfeld ratio as ``alpha v``, ``C6 v^2`` and ``R0 v^(1/3)``, and
    combined with the usual ``C6_AB = 2 C6_A C6_B / (alpha_B/alpha_A C6_A +
    alpha_A/alpha_B C6_B)``. Atoms of an element neither table covers are
    dropped with a warning.

    :param atomic_numbers: Element of every atom, shape ``(N,)``.
    :param positions: Positions in Angstrom, shape ``(N, 3)``.
    :param volume_ratios: Hirshfeld ratios of every atom, shape ``(N,)``;
        entries of atoms in neither group are ignored.
    :param atoms_a: Indices of the first group.
    :param atoms_b: Indices of the second group.
    :param s_r: Range parameter of the damping (the PBE value by default; the
        density here is not a PBE density, so this is a convention that a
        fitted coefficient in front of the channel absorbs).
    :param damping: Steepness of the damping.
    :return: Dispersion energy in Hartree; always negative (or zero).
    """
    from pyscf.data.elements import ELEMENTS

    numbers = np.asarray(atomic_numbers, dtype=int)
    coords = np.asarray(positions, dtype=float) / BOHR_TO_ANG
    ratios = np.asarray(volume_ratios, dtype=float)

    def parameters(group: Sequence[int]) -> tuple[np.ndarray, ...]:
        alpha, c6, r0, keep = [], [], [], []
        unknown = set()
        for i in group:
            values = _ts_free_atom_parameters(ELEMENTS[int(numbers[i])])
            if values is None:
                unknown.add(ELEMENTS[int(numbers[i])])
                continue
            v = float(ratios[i])
            alpha.append(values[0] * v)
            c6.append(values[1] * v * v)
            r0.append(values[2] * v ** (1.0 / 3.0))
            keep.append(int(i))
        if unknown:
            logger.warning(
                "no Tkatchenko-Scheffler parameters for %s; those atoms are "
                "left out of the dispersion sum",
                ", ".join(sorted(unknown)),
            )
        return (
            np.array(alpha),
            np.array(c6),
            np.array(r0),
            np.array(keep, dtype=int),
        )

    alpha_a, c6_a, r0_a, keep_a = parameters(list(atoms_a))
    alpha_b, c6_b, r0_b, keep_b = parameters(list(atoms_b))
    if len(keep_a) == 0 or len(keep_b) == 0:
        return 0.0

    c6 = (2.0 * c6_a[:, None] * c6_b[None, :]) / (
        (alpha_b[None, :] / alpha_a[:, None]) * c6_a[:, None]
        + (alpha_a[:, None] / alpha_b[None, :]) * c6_b[None, :]
    )
    r = np.linalg.norm(coords[keep_a][:, None, :] - coords[keep_b][None, :, :], axis=2)
    x = r / (s_r * (r0_a[:, None] + r0_b[None, :]))
    f = 1.0 / (1.0 + np.exp(-damping * (x - 1.0)))
    return -float(np.sum(f * c6 / r**6))


def hirshfeld_dispersion_energy(
    density: RIDensity,
    atoms_a: Sequence[int],
    atoms_b: Sequence[int] | None = None,
    *,
    environment_radius: float = 6.0,
    grid_level: int = 1,
    cell_radius: float = 4.0,
    neighbour_radius: float = 5.0,
    function_radius: float = 4.5,
    s_r: float = TS_RANGE,
    damping: float = TS_DAMPING,
    threads: int | None = None,
) -> float:
    """
    Tkatchenko-Scheffler dispersion between two fragments of one density.

    Computes :func:`hirshfeld_volume_ratios` for the atoms of fragment ``a``
    and for the atoms of ``b`` within ``environment_radius`` of them, then
    sums :func:`ts_dispersion_energy` over those pairs. Pairs farther apart
    contribute little each but are many, so the radius is part of the
    channel's definition and must be kept fixed across a series.

    :param density: Density of the whole complex.
    :param atoms_a: Atom indices of the first fragment.
    :param atoms_b: Atom indices of the second; ``None`` takes the rest.
    :param environment_radius: Atoms of ``b`` within this many Angstrom of an
        atom of ``a`` enter the sum.
    :param grid_level: PySCF grid level of the volume quadratures.
    :param cell_radius: See :func:`hirshfeld_volume_ratios`; tighter by
        default, a uniform bias a fitted coefficient absorbs.
    :param neighbour_radius: See :func:`hirshfeld_volume_ratios`.
    :param function_radius: See :func:`hirshfeld_volume_ratios`.
    :param s_r: Range parameter of the damping.
    :param damping: Steepness of the damping.
    :param threads: Threads for the volume integration.
    :return: Dispersion energy in Hartree; negative.
    """
    auxmol = density.auxmol
    a = np.array(sorted({int(i) for i in atoms_a}), dtype=int)
    b = (
        np.setdiff1d(np.arange(auxmol.natm), a)
        if atoms_b is None
        else np.array(sorted({int(i) for i in atoms_b}), dtype=int)
    )
    centres = auxmol.atom_coords(unit="Ang")
    b = b[pairwise_within(centres[b], centres[a], environment_radius)]
    if len(b) == 0:
        return 0.0
    atoms = np.union1d(a, b)
    ratios = np.zeros(auxmol.natm)
    ratios[atoms] = hirshfeld_volume_ratios(
        density,
        atoms.tolist(),
        grid_level=grid_level,
        cell_radius=cell_radius,
        neighbour_radius=neighbour_radius,
        function_radius=function_radius,
        threads=threads,
    )
    return ts_dispersion_energy(
        atomic_numbers_of(auxmol),
        centres,
        ratios,
        a.tolist(),
        b.tolist(),
        s_r=s_r,
        damping=damping,
    )
