"""
Scattering from an RI density: structure factors and atomic form factors.

The density ``ρ(r) = Σ_P c_P χ_P(r)`` is a sum of atom-centred Gaussians, and
the Fourier transform of every auxiliary function is analytic, so the X-ray
structure factor

    F(q) = ∫ ρ(r) e^{i q·r} dr = Σ_P c_P χ̃_P(q)

is a single matrix product with the same coefficient vector that feeds the
electrostatic potential. No grid, and no lattice sum either: a crystal's
structure factor is the sum over the atoms in one cell of atom-centred
functions, so periodicity only enters the model's neighbour list.

Conventions
-----------
* ``q`` is the scattering vector in Å⁻¹ *including* the ``2π``:
  ``|q| = 4π sin θ / λ``; :func:`hkl_to_q` builds it from Miller indices and a
  cell, ``q = 2π h·B`` with ``B`` the reciprocal cell.
* ``F(q)`` is dimensionless (electrons), with ``F(0)`` the electron count.
* Per-atom form factors ``f_A(q)`` carry no phase of their own centre:
  ``F(q) = Σ_A f_A(q) T_A(q) e^{i q·R_A}``, with ``T_A`` the Debye–Waller
  factor ``exp(-½ qᵀ U_A q)`` of a (Cartesian, Å²) displacement tensor.
* Electron diffraction sees the electrostatic potential; its structure
  factor follows from Mott–Bethe, ``V(q) = 4π (F_nuc(q) − F(q)) / q²`` in
  atomic units (Hartree·Bohr³) with ``F_nuc = Σ_A Z_A e^{i q·R_A}``.

Two atomic partitions are offered. ``"ri"`` assigns every auxiliary function
to the atom it is centred on (analytic, works for crystals); ``"hirshfeld"``
divides space by the promolecule on a Becke grid, which is what Hirshfeld atom
refinement uses. Hirshfeld is periodic-aware: pass ``cell`` to
:func:`atomic_form_factors` and each cell atom competes with the free-atom
densities of the neighbouring cells (the density integrated is the periodic
one), so a boundary atom no longer grabs a periodic neighbour's density.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pyscf import gto

from .conversions import BOHR_TO_ANG
from .cube import BeckeGrid, atomic_grids, becke_grid
from .fields import RIDensity
from .partition import (
    FragmentDensity,
    free_atom_densities,
    free_atom_density,
    free_atom_radial_table,
)

__all__ = [
    "hkl_to_q",
    "cell_from_parameters",
    "adp_cif_to_cartesian",
    "spherical_atom_form_factor",
    "independent_atom_structure_factors",
    "debye_waller",
    "structure_factors",
    "atomic_form_factors",
    "structure_factors_from_form_factors",
    "electron_structure_factors",
    "electron_atomic_form_factors",
    "write_tsc",
    "read_tsc",
]

#: Default spatial-cell side as a fraction of the Hirshfeld ``cutoff``. Smaller
#: cells mean fewer local atoms per cell but more cells, and below ~2/3 of the
#: cutoff the per-cell overhead wins; measured on glycine packed at protein
#: density, 2/3 was ~1.3x faster than either 1/2 or 1.
_BIN_FRACTION = 2.0 / 3.0

#: Density below which an auxiliary function is treated as out of reach (see
#: ``_LocalDensity``). The reach is set by the most diffuse ``jfit`` functions,
#: so tightening this costs nothing measurable; it is kept strict.
_AUX_EPS = 1e-10

#: Floor (Å) on the cell side used for the density evaluation, which otherwise
#: follows the auxiliary reach: cells much smaller than the reach do not shrink
#: the shell selection any further and only multiply the per-cell overhead.
_DENSITY_BIN_FLOOR = 3.0

Density = RIDensity | FragmentDensity
Partition = Literal["ri", "hirshfeld"]


# --------------------------------------------------------------------------- #
# Reciprocal space helpers
# --------------------------------------------------------------------------- #


def hkl_to_q(hkl: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Scattering vectors of Miller indices, ``q = 2π h·B`` with ``B`` the
    reciprocal cell.

    With positions ``r = frac @ cell`` the reciprocal vectors are the rows of
    ``inv(cell).T``, so that ``q·r = 2π h·frac`` for any cell shape.

    :param hkl: Miller indices, shape (N, 3).
    :param cell: Lattice vectors as rows, shape (3, 3), Å.
    :return: Scattering vectors in Å⁻¹ (with the ``2π``), shape (N, 3).
    """
    hkl = np.asarray(hkl, dtype=float)
    cell = np.asarray(cell, dtype=float)
    return 2.0 * np.pi * hkl @ np.linalg.inv(cell).T


def cell_from_parameters(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """
    Lattice vectors (rows, Å) from cell parameters, angles in degrees.

    Standard crystallographic orientation: ``a`` along x, ``b`` in the xy
    plane. Fractional coordinates map to Cartesian as ``frac @ cell``.
    """
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    return np.array(
        [[a, 0.0, 0.0], [b * np.cos(ga), b * np.sin(ga), 0.0], [cx, cy, cz]]
    )


def adp_cif_to_cartesian(u_cif: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Cartesian displacement tensor from the CIF ``U^{ij}`` convention.

    CIF/SHELX ``U^{ij}`` are defined against the reciprocal axes so that the
    Debye–Waller exponent is ``-2π² Σ_ij h_i h_j a*_i a*_j U^{ij}``. With
    ``N = diag(a*, b*, c*)`` (reciprocal lengths without 2π) and ``cell`` the
    lattice vectors as rows, ``U_cart = cellᵀ N U N cell``, which makes
    :func:`debye_waller` (``exp(-½ qᵀ U_cart q)`` with ``q`` from
    :func:`hkl_to_q`) agree with the CIF exponent for every reflection. For an
    orthorhombic cell with axes along x, y, z this is the identity.

    :param u_cif: ``U^{ij}`` tensor(s), shape (3, 3) or (n, 3, 3), Å².
    :param cell: Lattice vectors as rows, shape (3, 3), Å.
    :return: Cartesian tensor(s), same shape, Å².
    """
    cell = np.asarray(cell, dtype=float)
    recip = np.linalg.inv(cell).T  # rows: reciprocal vectors (no 2π)
    m = cell.T @ np.diag(np.linalg.norm(recip, axis=1))
    u = np.asarray(u_cif, dtype=float)
    return np.einsum("ij,...jk,lk->...il", m, u, m)


def debye_waller(q: np.ndarray, adps: np.ndarray | None, n_atoms: int) -> np.ndarray:
    """
    Debye–Waller factors ``T_A(q) = exp(-½ qᵀ U_A q)``.

    :param q: Scattering vectors in Å⁻¹, shape (N, 3).
    :param adps: ``None`` (rigid atoms, all ones), isotropic ``U_iso`` in Å²
        of shape (n_atoms,), or Cartesian displacement tensors in Å² of
        shape (n_atoms, 3, 3).
    :param n_atoms: Number of atoms, to validate ``adps``.
    :return: Factors, shape (N, n_atoms).
    """
    q = np.asarray(q, dtype=float)
    if adps is None:
        return np.ones((len(q), n_atoms))
    u = np.asarray(adps, dtype=float)
    if u.shape == (n_atoms,):
        return np.exp(-0.5 * np.einsum("ni,ni->n", q, q)[:, None] * u[None, :])
    if u.shape == (n_atoms, 3, 3):
        return np.exp(-0.5 * np.einsum("ni,aij,nj->na", q, u, q))
    raise ValueError(
        f"adps must have shape ({n_atoms},) or ({n_atoms}, 3, 3), got {u.shape}"
    )


def _aux_fourier(auxmol: gto.Mole, q: np.ndarray) -> np.ndarray:
    """
    ``χ̃_P(q) = ∫ χ_P(r) e^{i q·r} dr`` for every auxiliary function.

    PySCF's ``ft_ao`` uses ``e^{-i k·r}`` and Bohr⁻¹; the density is real, so
    evaluating at ``-q`` gives the ``e^{+i q·r}`` convention used here.

    :return: Complex array, shape (N, naux).
    """
    from pyscf.gto import ft_ao

    q_bohr = -np.asarray(q, dtype=float) * BOHR_TO_ANG
    return np.asarray(ft_ao.ft_ao(auxmol, q_bohr), dtype=complex)


def _atom_slices(auxmol: gto.Mole) -> np.ndarray:
    """Start index of every atom's block of auxiliary functions, shape (natm,)."""
    return auxmol.aoslice_by_atom()[:, 2]


def _positions_ang(density: Density) -> np.ndarray:
    return np.asarray(density.positions, dtype=float)


# --------------------------------------------------------------------------- #
# Structure factors and form factors
# --------------------------------------------------------------------------- #


def _ri_atomic_terms(
    density: Density, q: np.ndarray, *, q_batch: int = 4096
) -> np.ndarray:
    """``Σ_{P∈A} c_P χ̃_P(q)``, phase of the centre included, shape (N, natm).

    Batched over ``q``: the intermediate ``χ̃_P(q)`` is ``(len(q), naux)``, which for a
    large cell
    (a protein: ~10⁴–10⁵ aux functions) and a full reflection list is tens of GB in one
    array. The
    batch caps it without changing the result.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    coeffs = np.asarray(density.coeffs)[None, :]
    starts = _atom_slices(density.auxmol)
    out = np.empty((len(q), len(starts)), dtype=complex)
    for start in range(0, len(q), q_batch):
        stop = min(start + q_batch, len(q))
        weighted = _aux_fourier(density.auxmol, q[start:stop]) * coeffs
        out[start:stop] = np.add.reduceat(weighted, starts, axis=1)
    return out


def structure_factors(
    density: Density,
    q: np.ndarray,
    *,
    adps: np.ndarray | None = None,
) -> np.ndarray:
    """
    X-ray structure factors ``F(q) = ∫ ρ(r) e^{i q·r} dr`` of an RI density.

    Analytic in the auxiliary basis. With ``adps`` every atom's functions are
    smeared by its Debye–Waller factor (the functions are assigned to the atom
    they are centred on).

    :param density: :class:`~qpet.RIDensity` or fragment thereof.
    :param q: Scattering vectors in Å⁻¹ (with the ``2π``), shape (N, 3);
        see :func:`hkl_to_q`.
    :param adps: Displacement parameters, see :func:`debye_waller`.
    :return: Complex structure factors in electrons, shape (N,).
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    terms = _ri_atomic_terms(density, q)
    if adps is None:
        return terms.sum(axis=1)
    return (terms * debye_waller(q, adps, terms.shape[1])).sum(axis=1)


def atomic_form_factors(
    density: Density,
    q: np.ndarray,
    *,
    partition: Partition = "ri",
    grid_level: int = 3,
    q_batch: int = 256,
    grid_batch: int = 50000,
    cutoff: float | None = 6.0,
    rho_min: float = 1e-8,
    atom_min: float = 1e-9,
    bin_size: float | None = None,
    cell: np.ndarray | None = None,
    images: int = 1,
    device: str | None = None,
    dtype: str | None = None,
    quadrature: str = "becke",
) -> np.ndarray:
    """
    Aspherical atomic form factors ``f_A(q)``, the Fourier transform of each
    atom's share of the density about its own nucleus.

    ``"ri"`` gives every atom the functions centred on it; ``"hirshfeld"``
    integrates ``w_A(r) ρ(r) e^{i q·(r − R_A)}`` on a Becke grid with the
    stockholder weights ``w_A = ρ_A⁰ / Σ_B ρ_B⁰`` of the free atoms. Either way
    ``Σ_A f_A(q) e^{i q·R_A}`` is the structure factor (exactly for ``"ri"``,
    to grid accuracy for ``"hirshfeld"``).

    For a **periodic** cell pass ``cell`` (lattice vectors as rows, Å): the
    Hirshfeld stockholder then competes each cell atom against the free-atom
    densities of the neighbouring cells (``±images`` translations), and the
    density integrated is the periodic one, so an atom near a cell face no
    longer grabs density that belongs to a periodic neighbour. Without ``cell``
    the stockholder is the isolated-molecule one (the Becke grid does not know
    about images), which is only right when the molecule is well separated in
    the cell. ``cell`` has no effect on the ``"ri"`` partition (already exact).

    :param density: :class:`~qpet.RIDensity` or fragment thereof.
    :param q: Scattering vectors in Å⁻¹, shape (N, 3).
    :param partition: ``"ri"`` (default) or ``"hirshfeld"``.
    :param grid_level: PySCF grid level of the Hirshfeld quadrature.
    :param q_batch: Scattering vectors per chunk of the Hirshfeld integral.
    :param cell: Lattice vectors as rows (Å) for a periodic Hirshfeld
        partition; ``None`` for the isolated-molecule stockholder.
    :param images: Lattice-translation range per axis for the periodic
        promolecule (default 1: the 27 surrounding cells).
    :param cutoff: Radius (Å) around a chunk of grid points within which atoms
        contribute to the stockholder weights; ``None`` runs the dense
        reference over every atom. The error falls off with the free-atom tails,
        so 3–6 Å is already at the quadrature noise floor.
    :param rho_min: Quadrature integrand below which a grid point is dropped.
    :param atom_min: Share of a spatial cell's electrons below which a nearby
        atom is dropped from that cell's Fourier sum; the resulting error is
        bounded by ``atom_min`` times the electron count.
    :param bin_size: Side (Å) of the spatial cells the grid is binned into
        before the local atom sets are built; defaults to two thirds of
        ``cutoff``. Smaller cells mean fewer local atoms per chunk but more
        chunks, and the default is the measured optimum.
    :param quadrature: Which grid the Hirshfeld integral runs on. ``"becke"``
        (the default) is the shared molecular grid; ``"atomic"`` gives every atom
        its own single-centre grid, which removes the Becke partition and its
        ``O(N_points x N_atoms)`` construction and is what makes protein-sized
        cells affordable. See :func:`_hirshfeld_atomic`.
    :param dtype: ``"float32"`` runs the grid Fourier sum in single precision
        (an order of magnitude faster on a consumer GPU, ~1e-6 relative error);
        ``None`` keeps it in double.
    :param device: Torch device for the Hirshfeld grid Fourier sum (e.g.
        ``"cuda"``); ``None`` keeps it on NumPy/CPU. The sum
        ``Σ_G e^{i q·r_G} w_A(r_G) ρ(r_G)`` over the quadrature grid is the
        dominant cost and is a dense complex matrix product, so a GPU speeds it
        up several-fold; the result is identical to the CPU path.
    :return: Complex form factors in electrons, shape (N, natm).
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    positions = _positions_ang(density)
    phase = np.exp(-1j * (q @ positions.T))  # (N, natm), removes e^{i q·R_A}
    if partition == "ri":
        return _ri_atomic_terms(density, q) * phase
    if partition != "hirshfeld":
        raise ValueError(f"unknown partition {partition!r}; use 'ri' or 'hirshfeld'")

    promol = None
    if cell is not None:
        promol = density.with_periodic_images(np.asarray(cell), images=images)
    source = promol if promol is not None else density

    if quadrature == "atomic":
        if cutoff is None:
            raise ValueError(
                "the atomic quadrature has no dense reference; give a cutoff"
            )
        return _hirshfeld_atomic(
            density,
            q,
            source,
            promol,
            positions,
            phase,
            grid_level=grid_level,
            cutoff=cutoff,
            rho_min=rho_min,
            device=device,
            q_batch=q_batch,
            dtype=dtype,
        )
    if quadrature != "becke":
        raise ValueError(f"unknown quadrature {quadrature!r}; use 'becke' or 'atomic'")

    grid = becke_grid(density.auxmol, level=grid_level)

    if cutoff is None:
        # Dense reference: the stockholder weights are ``(n_grid, natm)`` and the
        # Fourier sum runs every reflection against every grid point against every atom.
        # Batched over the grid only to cap memory. Exact, but O(N_refl·N_grid·N_atoms)
        # -- hours for a protein. Kept for testing and small systems; large cells should
        # use the local path (a finite ``cutoff``).
        natm = len(positions)
        out = np.zeros((len(q), natm), dtype=complex)
        for start in range(0, len(grid.points), grid_batch):
            stop = min(start + grid_batch, len(grid.points))
            pts = grid.points[start:stop]
            rho = source.density(pts) * grid.weights[start:stop]  # (g,)
            weights = _hirshfeld_atom_weights(
                density, pts, promol_density=promol
            )  # (g, natm)
            out += _grid_fourier_sum(
                q, pts, rho[:, None] * weights, device=device, q_batch=q_batch
            )
        return out * phase

    return _hirshfeld_local(
        density,
        q,
        grid,
        source,
        promol,
        positions,
        phase,
        cutoff=cutoff,
        rho_min=rho_min,
        atom_min=atom_min,
        device=device,
        q_batch=q_batch,
        grid_batch=grid_batch,
        bin_size=bin_size,
        dtype=dtype,
    )


def _grid_fourier_sum(
    q: np.ndarray,
    points_ang: np.ndarray,
    weighted: Any,
    *,
    device: str | None = None,
    q_batch: int = 256,
    dtype: str | None = None,
    keep_on_device: bool = False,
) -> np.ndarray | Any:
    """``Σ_G e^{i q·r_G} weighted[G, A]`` -> ``(N_q, natm)``, on ``device`` or NumPy.

    The exponential and the contraction over the ~10^5 quadrature points dominate the
    Hirshfeld form-factor cost; on a GPU this is a few dense complex matmuls. The NumPy
    path (``device=None``) is the reference and gives an identical result.

    ``dtype="float32"`` runs the phase and the contraction in single precision, which on
    a
    consumer GPU (where double precision runs at 1/64 of the single-precision rate) is
    worth
    an order of magnitude. The phase ``q·r`` is reduced modulo 2π in double precision
    first,
    so the single-precision error is on the *reduced* angle and the form factors keep
    ~1e-6
    relative accuracy; the accumulation across chunks stays in double.
    """
    if device is not None:
        import torch

        single = _is_single(dtype)
        real = torch.float32 if single else torch.float64
        cplx = torch.complex64 if single else torch.complex128
        dev = torch.device(device)
        pts = torch.as_tensor(np.ascontiguousarray(points_ang), device=dev)
        wt = (
            weighted
            if torch.is_tensor(weighted)
            else torch.as_tensor(np.ascontiguousarray(weighted), device=dev)
        ).to(cplx)
        qt = torch.as_tensor(np.ascontiguousarray(q), device=dev)
        summed = torch.empty((len(q), wt.shape[1]), dtype=torch.complex128, device=dev)
        for start in range(0, len(q), q_batch):
            stop = min(start + q_batch, len(q))
            angle = qt[start:stop] @ pts.T  # (n, G), double
            if single:
                angle = torch.remainder(angle, 2.0 * torch.pi)
            wave = torch.polar(
                torch.ones((), device=dev, dtype=real).expand(angle.shape),
                angle.to(real),
            )
            summed[start:stop] = (wave @ wt).to(torch.complex128)
        return summed if keep_on_device else np.asarray(summed.cpu().numpy())

    out = np.empty((len(q), weighted.shape[1]), dtype=complex)
    for start in range(0, len(q), q_batch):
        stop = min(start + q_batch, len(q))
        phases = np.exp(1j * (q[start:stop] @ points_ang.T))  # (n, G)
        out[start:stop] = phases @ weighted
    return out


def _free_atom_grid(
    auxmol: gto.Mole, numbers: np.ndarray, points_bohr: np.ndarray, ecp: str | None
) -> np.ndarray:
    """Free-atom densities of every atom of ``auxmol`` at the points: (G, natm)."""
    centres = auxmol.atom_coords()
    free = np.empty((len(points_bohr), len(centres)))
    for a, (centre, z) in enumerate(zip(centres, np.asarray(numbers), strict=True)):
        dist = np.linalg.norm(points_bohr - centre, axis=1)
        free[:, a] = free_atom_density(int(z), dist, ecp=ecp)
    return free


def _hirshfeld_atom_weights(
    density: Density, points_ang: np.ndarray, promol_density: Density | None = None
) -> np.ndarray:
    """Stockholder weights at the points, shape (G, natm), for the atoms of ``density``.

    The numerator is each ``density`` atom's free-atom density; the denominator is the
    promolecule sum. With ``promol_density`` (a periodic supercell of the same
    coefficients) the denominator runs over the neighbouring cells too, so a cell atom
    competes with its periodic neighbours rather than overreaching into them.
    """
    auxmol = density.auxmol
    ecp = auxmol.ecp if auxmol.ecp else None
    points = np.asarray(points_ang) / BOHR_TO_ANG
    numerator = _free_atom_grid(auxmol, density.atomic_numbers, points, ecp)
    if promol_density is None:
        total = numerator.sum(axis=1, keepdims=True)
    else:
        denom = _free_atom_grid(
            promol_density.auxmol, promol_density.atomic_numbers, points, ecp
        )
        total = denom.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, numerator / total, 1.0 / numerator.shape[1])


def _cell_weights(
    pts_bohr: np.ndarray,
    rho: np.ndarray,
    s_num: np.ndarray,
    s_den: np.ndarray,
    num_pos: np.ndarray,
    num_z: np.ndarray,
    den_pos: np.ndarray,
    den_z: np.ndarray,
    ecp: str | None,
    *,
    atom_min: float,
    device: str | None,
) -> tuple[np.ndarray, Any]:
    """``ρ(r)·w_A(r)`` for one spatial cell, and the atoms of it worth carrying into the
    sum.

    An atom within ``cutoff`` of a cell may still own a negligible share of the density
    in it;
    dropping those columns costs at most ``atom_min`` of the cell's electrons each, so
    the error
    over the whole grid is bounded by ``atom_min`` times the electron count. With
    ``device`` set
    the weights are built there and handed to the Fourier sum without a host round-trip.

    :return: ``(atom indices, weighted integrand)``; the integrand is a torch tensor on
        ``device``, or a NumPy array when ``device`` is ``None``.
    """
    if device is None:
        tot = _free_atom_block(pts_bohr, den_pos[s_den], den_z[s_den], ecp).sum(axis=1)
        num = _free_atom_block(pts_bohr, num_pos[s_num], num_z[s_num], ecp)
        with np.errstate(invalid="ignore", divide="ignore"):
            weighted = rho[:, None] * np.where(
                tot[:, None] > 0, num / tot[:, None], 0.0
            )
        owners = np.abs(weighted).sum(axis=0) > atom_min * np.abs(rho).sum()
        return s_num[owners], weighted[:, owners]

    import torch

    # The stockholder weights stay in double even when the Fourier sum runs single: they
    # are an interpolation and a distance matrix (memory-bound, so double costs little
    # on a GPU), and in single they were the one step that pushed the form factors past
    # 1e-6.
    denominator = _free_atom_block_torch(
        pts_bohr, den_pos[s_den], den_z[s_den], ecp, device=device, dtype="float64"
    ).sum(dim=1)
    numerator = _free_atom_block_torch(
        pts_bohr, num_pos[s_num], num_z[s_num], ecp, device=device, dtype="float64"
    )
    integrand = torch.as_tensor(rho, dtype=numerator.dtype, device=numerator.device)
    shares = integrand[:, None] * torch.where(
        denominator[:, None] > 0, numerator / denominator[:, None], 0.0
    )
    keep = shares.abs().sum(dim=0) > atom_min * integrand.abs().sum()
    return s_num[keep.cpu().numpy()], shares[:, keep]


def _spatial_bins(points_ang: np.ndarray, size: float) -> list[np.ndarray]:
    """Point indices grouped into cubic cells of side ``size`` Å, densest cell first.

    The Becke grid is atom-ordered, not space-ordered: a slice of it spans an atom's
    whole
    radial extent, so the set of atoms within ``cutoff`` of *any* of its points is
    nearly the
    whole molecule and the locality of the stockholder weights is lost. Binning by
    position
    restores it -- a cell of side ``size`` only ever sees the atoms within ``cutoff +
    √3·size/2``
    of its centre.
    """
    keys = np.floor(np.asarray(points_ang) / size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    cuts = np.flatnonzero(np.any(np.diff(keys[order], axis=0) != 0, axis=1)) + 1
    return np.split(order, cuts)


def _box_atom_set(
    positions: np.ndarray,
    tree: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    radii: float | np.ndarray,
) -> np.ndarray:
    """Indices of atoms whose reach (a scalar cutoff, or one radius each) touches the
    box.

    The ball query around the box centre over-reaches by the box diagonal, so the
    candidates are
    filtered on the true point-to-box distance -- worth doing, because the whole point
    of binning
    the grid is to keep these sets small.
    """
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    radii = np.broadcast_to(np.asarray(radii, dtype=float), (len(positions),))
    reach = float(radii.max()) + float(np.linalg.norm(half))
    candidates = np.asarray(tree.query_ball_point(centre, reach), dtype=int)
    if candidates.size == 0:
        return candidates
    offset = np.maximum(np.abs(positions[candidates] - centre) - half, 0.0)
    return np.sort(candidates[np.linalg.norm(offset, axis=1) <= radii[candidates]])


def _free_atom_block(
    pts_bohr: np.ndarray, centres_bohr: np.ndarray, numbers: np.ndarray, ecp: str | None
) -> np.ndarray:
    """Free-atom densities of the given centres at the given points: (g, n_centres)."""
    from scipy.spatial.distance import cdist

    return free_atom_densities(numbers, cdist(pts_bohr, centres_bohr), ecp=ecp)


def _is_single(dtype: str | None) -> bool:
    """Whether ``dtype`` asks for the single-precision grid path."""
    return str(dtype) in ("float32", "single", "complex64")


@cache
def _free_atom_torch_table(
    elements: tuple[int, ...], ecp: str | None, device: str, dtype: str
) -> tuple[Any, Any]:
    """``(radii, table)`` of the free-atom radial densities of ``elements``, stacked, on
    a device.

    Every element is tabulated on the same radial grid, so the interpolation of a whole
    ``(points, atoms)`` block is one bucketised gather rather than a loop.
    """
    import torch

    radii, rows = None, []
    for z in elements:
        r, rho = free_atom_radial_table(int(z), ecp=ecp)
        radii = r
        rows.append(rho)
    dev = torch.device(device)
    return (
        torch.as_tensor(radii, dtype=getattr(torch, dtype), device=dev),
        torch.as_tensor(np.asarray(rows), dtype=getattr(torch, dtype), device=dev),
    )


def _free_atom_block_torch(
    pts_bohr: np.ndarray,
    centres_bohr: np.ndarray,
    numbers: np.ndarray,
    ecp: str | None,
    *,
    device: str,
    dtype: str,
) -> Any:
    """:func:`_free_atom_block` on a torch device: (g, n_centres), same linear
    interpolation."""
    import torch

    elements = tuple(sorted({int(z) for z in np.asarray(numbers)}))
    radii, table = _free_atom_torch_table(elements, ecp, device, dtype)
    row = torch.as_tensor(
        [elements.index(int(z)) for z in np.asarray(numbers)], device=radii.device
    )
    distance = torch.cdist(
        torch.as_tensor(
            np.ascontiguousarray(pts_bohr), dtype=radii.dtype, device=radii.device
        ),
        torch.as_tensor(
            np.ascontiguousarray(centres_bohr), dtype=radii.dtype, device=radii.device
        ),
    )
    index = torch.searchsorted(radii, distance.contiguous()).clamp_(1, len(radii) - 1)
    lower, upper = radii[index - 1], radii[index]
    flat = table.reshape(-1)
    offset = row[None, :] * len(radii) + index
    value = flat[offset - 1] + (flat[offset] - flat[offset - 1]) * (
        (distance - lower) / (upper - lower)
    )
    return value.masked_fill_(distance > radii[-1], 0.0)


def _aux_atom_radius(density: Density, eps: float) -> np.ndarray:
    """Radius (Å) beyond which an atom's auxiliary functions contribute less than
    ``eps``.

    Bounds the contracted radial function of every shell by ``Σ_i |d_i| r^l e^{-α_i
    r²}`` times the
    largest fitting coefficient on it, and takes the outermost radius where that bound
    still
    exceeds ``eps``. Diffuse ``jfit`` functions reach ~10 Å, tight core ones a fraction
    of one, so
    the per-atom radius (rather than one global cutoff) is what makes the local
    evaluation pay.
    """
    auxmol = density.auxmol
    coeffs = np.abs(np.asarray(density.coeffs))
    ao_loc = auxmol.ao_loc_nr()
    r = np.linspace(0.1, 60.0, 600)  # Bohr
    radius = np.zeros(auxmol.natm)
    for shell in range(auxmol.nbas):
        contraction = np.abs(auxmol.bas_ctr_coeff(shell)).sum(axis=1)
        bound = coeffs[ao_loc[shell] : ao_loc[shell + 1]].max() * (
            r[:, None] ** auxmol.bas_angular(shell)
            * np.exp(-np.outer(r**2, auxmol.bas_exp(shell)))
            * contraction
        ).sum(axis=1)
        above = np.flatnonzero(bound > eps)
        if above.size:
            atom = auxmol.bas_atom(shell)
            radius[atom] = max(radius[atom], r[above[-1]])
    return radius * BOHR_TO_ANG


class _LocalDensity:
    """``ρ(r)`` evaluated from the auxiliary functions that reach the points, and no
    others.

    ``RIDensity.density`` evaluates every auxiliary function at every point: an
    ``(n_points, n_aux)`` array that for a protein cell is both the memory wall and the
    dominant
    cost of the Hirshfeld grid, even though each function only reaches a few Å.
    Restricting the
    shells to the atoms whose functions reach the current spatial cell (see
    :func:`_aux_atom_radius`) makes the evaluation local without changing the result
    beyond
    ``eps``; the shells are selected in place on a shallow copy of the ``Mole``, so the
    columns of
    the evaluation line up with a slice of the coefficient vector.
    """

    def __init__(
        self, density: Density, eps: float = _AUX_EPS, batch_size: int = 20000
    ) -> None:
        from scipy.spatial import cKDTree

        self.auxmol = density.auxmol
        self.coeffs = np.asarray(density.coeffs)
        self.batch_size = batch_size
        self.bas_atom = np.asarray(self.auxmol._bas[:, gto.ATOM_OF])
        self.ao_loc = self.auxmol.ao_loc_nr()
        self.positions = _positions_ang(density)
        self.radius = _aux_atom_radius(density, eps)
        self.tree = cKDTree(self.positions)
        self._sub = self.auxmol.copy()

    def __call__(
        self, pts_ang: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        atoms = _box_atom_set(self.positions, self.tree, lo, hi, self.radius)
        if atoms.size == 0:
            return np.zeros(len(pts_ang))
        shells = np.flatnonzero(np.isin(self.bas_atom, atoms))
        counts = self.ao_loc[shells + 1] - self.ao_loc[shells]
        offsets = np.repeat(self.ao_loc[shells], counts)
        columns = (
            offsets
            + np.arange(counts.sum())
            - np.repeat(np.cumsum(counts) - counts, counts)
        )
        self._sub._bas = np.asarray(self.auxmol._bas[shells], dtype=np.int32, order="C")
        coeffs = self.coeffs[columns]
        pts_bohr = pts_ang / BOHR_TO_ANG
        rho = np.empty(len(pts_bohr))
        for start in range(0, len(pts_bohr), self.batch_size):
            stop = start + self.batch_size
            rho[start:stop] = (
                self._sub.eval_gto("GTOval", pts_bohr[start:stop]) @ coeffs
            )
        return rho


def _fits_on_device(device: str, nbytes: float, fraction: float = 0.25) -> bool:
    """Whether an array of ``nbytes`` fits on ``device`` without crowding the work."""
    import torch

    if torch.device(device).type != "cuda":
        return False
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return nbytes < fraction * free


def _grid_integrand(
    source: Density, grid: Any, bin_size: float | None = None
) -> np.ndarray:
    """``ρ(r_G)·w_quad(G)`` on the whole quadrature grid, evaluated cell by cell.

    A pass of its own, because how far the auxiliary functions reach has nothing to do
    with how
    far the free-atom tails do, so the two want different spatial cells; this one
    follows the
    auxiliary reach. The result is one array over the grid (8 bytes a point), which the
    form-factor loop then just indexes.
    """
    evaluate = _LocalDensity(source)
    if bin_size is None:
        bin_size = max(_DENSITY_BIN_FLOOR, float(np.median(evaluate.radius)))
    rho = np.empty(len(grid.points))
    for index in _spatial_bins(grid.points, bin_size):
        pts = grid.points[index]
        rho[index] = evaluate(pts, pts.min(axis=0), pts.max(axis=0))
    return rho * grid.weights


def _hirshfeld_atomic(
    density: Density,
    q: np.ndarray,
    source: Density,
    promol: Density | None,
    positions: np.ndarray,
    phase: np.ndarray,
    *,
    grid_level: int,
    cutoff: float,
    rho_min: float,
    device: str | None,
    q_batch: int,
    dtype: str | None = None,
) -> np.ndarray:
    """Hirshfeld form factors from one single-centre quadrature per atom.

    The stockholder weight confines ``w_A rho`` to atom A, so the integral is atomic and
    wants an atomic grid, not a slice of a molecular Becke grid. Dropping the partition
    removes its ``O(N_points x N_atoms)`` construction, which at protein scale is the
    single largest cost, and collapses the Fourier sum as well: on a shared grid every
    point is summed against every atom whose weight reaches it (tens), whereas here
    each point belongs to exactly one atom.

    The quadrature is sound because ``w_A`` is tiny wherever another nucleus sits --
    ``rho_A^0(R_B) / rho_B^0(R_B)`` is ~1e-4 or less -- so the integrand carries no
    cusp but its own, which a single-centre radial x Lebedev grid is built to resolve.
    """
    from scipy.spatial import cKDTree

    auxmol = density.auxmol
    ecp = auxmol.ecp if auxmol.ecp else None
    num_z = np.asarray(density.atomic_numbers)
    den_pos = _positions_ang(promol) if promol is not None else positions
    den_z = np.asarray(promol.atomic_numbers) if promol is not None else num_z
    den_tree = cKDTree(den_pos)
    table = atomic_grids(auxmol, level=grid_level)
    b = BOHR_TO_ANG

    # The density is the same function of position whatever atom's grid a point came
    # from, so the atomic grids are concatenated and evaluated in one globally binned
    # pass. Doing it per atom instead bins a 25 A sphere into 3 A cells for every atom
    # in turn, and each of those thousands of tiny cells pays a shell selection and an
    # ``eval_gto`` call of its own.
    blocks = [table[auxmol.atom_symbol(a)] for a in range(len(positions))]
    bounds = np.cumsum([0] + [len(weights) for _, weights in blocks])
    points = np.concatenate(
        [
            offsets + centre
            for (offsets, _), centre in zip(blocks, positions, strict=True)
        ]
    )
    integrand = _grid_integrand(
        source, BeckeGrid(points=points, weights=np.concatenate([w for _, w in blocks]))
    )

    out = np.zeros((len(q), len(positions)), dtype=complex)
    for a, centre in enumerate(positions):
        span = slice(bounds[a], bounds[a + 1])
        own_points, rho = points[span], integrand[span]
        keep = np.abs(rho) > rho_min
        if not keep.any():
            continue
        own_points, rho = own_points[keep], rho[keep]

        bohr = own_points / b
        weighted: Any
        local = _box_atom_set(
            den_pos, den_tree, own_points.min(axis=0), own_points.max(axis=0), cutoff
        )
        centres = np.vstack([den_pos[local], centre[None, :]]) / b
        numbers = np.append(den_z[local], num_z[a])
        if device is None:
            free = _free_atom_block(bohr, centres, numbers, ecp)
            total, own = free[:, :-1].sum(axis=1), free[:, -1]
            with np.errstate(invalid="ignore", divide="ignore"):
                weighted = rho * np.where(total > 0, own / total, 0.0)
        else:
            import torch

            free = _free_atom_block_torch(
                bohr, centres, numbers, ecp, device=device, dtype="float64"
            )
            total, own = free[:, :-1].sum(axis=1), free[:, -1]
            share = torch.where(total > 0, own / total, torch.zeros_like(own))
            on_device = torch.as_tensor(rho, dtype=share.dtype, device=share.device)
            weighted = on_device * share
        out[:, a] = _grid_fourier_sum(
            q,
            own_points,
            weighted[:, None],
            device=device,
            q_batch=q_batch,
            dtype=dtype,
        )[:, 0]
    return out * phase


def _hirshfeld_local(
    density: Density,
    q: np.ndarray,
    grid: Any,
    source: Density,
    promol: Density | None,
    positions: np.ndarray,
    phase: np.ndarray,
    *,
    cutoff: float,
    rho_min: float,
    atom_min: float,
    device: str | None,
    q_batch: int,
    grid_batch: int,
    bin_size: float | None = None,
    dtype: str | None = None,
) -> np.ndarray:
    """Atom-local Hirshfeld form factors, matching the dense path to the cutoff/tail
    tolerance.

    The dense grid sum wastes its ``O(N_atoms)`` factor: the stockholder weight
    ``w_A(r)`` is ~0 for
    atoms far from ``r``, and most Becke-grid points sit in near-zero-density radial
    tails. The grid
    is first binned into cubic cells of side ``bin_size`` so that each chunk is
    spatially compact,
    and per cell we

    1. take only the atoms within ``cutoff`` Å of the cell (one box query, see
       :func:`_box_atom_set`),
    2. drop the points whose quadrature integrand ``ρ(r)·w_quad(r)`` is below
    ``rho_min``, and
    3. build the stockholder weights from the local atoms alone and run the Fourier sum
    over the
       surviving points against the atoms that own a share of the cell's density (see
       :func:`_cell_weights`).

    Cost falls from ``O(N_refl·N_grid·N_atoms)`` to ``~O(N_refl·N_grid_kept·N_local)``,
    with
    ``N_local`` set by the atom density inside a ``cutoff``-sized ball rather than by
    the cell.
    """
    from scipy.spatial import cKDTree

    natm = len(positions)
    ecp = density.auxmol.ecp if density.auxmol.ecp else None
    num_pos = positions  # density atoms (numerator), Å
    num_z = np.asarray(density.atomic_numbers)
    den_pos = (
        _positions_ang(promol) if promol is not None else positions
    )  # promolecule (denom)
    den_z = np.asarray(promol.atomic_numbers) if promol is not None else num_z
    num_tree = cKDTree(num_pos)
    den_tree = cKDTree(den_pos)
    integrand = _grid_integrand(source, grid)
    b = BOHR_TO_ANG
    num_bohr, den_bohr = num_pos / b, den_pos / b

    # Every cell's contribution is a ``(n_refl, n_local)`` block, and copying each one
    # back to the host is more traffic than the sum itself once there are thousands of
    # cells. Where the whole ``(n_refl, natm)`` accumulator fits on the device, it stays
    # there and the blocks are scattered into it in place.
    on_device = device is not None and _fits_on_device(device, 16 * len(q) * natm)
    out: Any
    if on_device:
        import torch

        out = torch.zeros((len(q), natm), dtype=torch.complex128, device=device)
    else:
        out = np.zeros((len(q), natm), dtype=complex)
    for cell_index in _spatial_bins(grid.points, bin_size or _BIN_FRACTION * cutoff):
        for start in range(0, len(cell_index), grid_batch):
            index = cell_index[start : start + grid_batch]
            pts = grid.points[index]
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            s_num = _box_atom_set(num_pos, num_tree, lo, hi, cutoff)
            s_den = _box_atom_set(den_pos, den_tree, lo, hi, cutoff)
            if s_num.size == 0:
                continue
            rho = integrand[index]
            keep = np.abs(rho) > rho_min
            if not keep.any():
                continue
            pts, rho, pts_bohr = pts[keep], rho[keep], pts[keep] / b

            owners, weighted = _cell_weights(
                pts_bohr,
                rho,
                s_num,
                s_den,
                num_bohr,
                num_z,
                den_bohr,
                den_z,
                ecp,
                atom_min=atom_min,
                device=device,
            )
            if owners.size == 0:
                continue
            block = _grid_fourier_sum(
                q,
                pts,
                weighted,
                device=device,
                q_batch=q_batch,
                dtype=dtype,
                keep_on_device=on_device,
            )
            if on_device:
                out.index_add_(1, torch.as_tensor(owners, device=out.device), block)
            else:
                out[:, owners] += block
    if on_device:
        out = out.cpu().numpy()
    return out * phase


def structure_factors_from_form_factors(
    form_factors: np.ndarray,
    q: np.ndarray,
    positions: np.ndarray,
    *,
    adps: np.ndarray | None = None,
) -> np.ndarray:
    """
    ``F(q) = Σ_A f_A(q) T_A(q) e^{i q·R_A}`` from per-atom form factors.

    :param form_factors: Shape (N, natm), from :func:`atomic_form_factors`.
    :param q: Scattering vectors in Å⁻¹, shape (N, 3).
    :param positions: Atomic positions in Å, shape (natm, 3).
    :param adps: Displacement parameters, see :func:`debye_waller`.
    :return: Complex structure factors, shape (N,).
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    positions = np.asarray(positions, dtype=float)
    phase = np.exp(1j * (q @ positions.T))
    return (form_factors * debye_waller(q, adps, len(positions)) * phase).sum(axis=1)


def electron_structure_factors(
    density: Density,
    q: np.ndarray,
    *,
    adps: np.ndarray | None = None,
) -> np.ndarray:
    """
    Electron-diffraction structure factors: the Fourier transform of the
    electrostatic potential, ``V(q) = 4π (Σ_A Z_A T_A e^{i q·R_A} − F(q)) / q²``
    (Mott–Bethe), in atomic units (Hartree·Bohr³).

    Nuclear charges are the effective ones under an ECP, consistent with the
    valence-only density. ``q = 0`` has no finite value and is returned as
    ``nan``.

    :param density: :class:`~qpet.RIDensity` or fragment thereof.
    :param q: Scattering vectors in Å⁻¹, shape (N, 3).
    :param adps: Displacement parameters applied to nuclei and electrons alike.
    :return: Complex potential structure factors, shape (N,).
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    positions = _positions_ang(density)
    charges = np.asarray(density.nuclear_charges, dtype=float)
    nuclear = (
        charges[None, :]
        * debye_waller(q, adps, len(charges))
        * np.exp(1j * (q @ positions.T))
    ).sum(axis=1)
    electronic = structure_factors(density, q, adps=adps)
    q2_bohr = np.einsum("ni,ni->n", q, q) * BOHR_TO_ANG**2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 4.0 * np.pi * (nuclear - electronic) / q2_bohr
    out[q2_bohr == 0] = np.nan
    return out


def electron_atomic_form_factors(
    density: Density,
    q: np.ndarray,
    *,
    partition: Partition = "ri",
    grid_level: int = 3,
    q_batch: int = 256,
    grid_batch: int = 50000,
    cutoff: float | None = 6.0,
    rho_min: float = 1e-8,
    atom_min: float = 1e-9,
    bin_size: float | None = None,
    cell: np.ndarray | None = None,
    images: int = 1,
    device: str | None = None,
    dtype: str | None = None,
) -> np.ndarray:
    """
    Per-atom electron (potential) form factors ``f^e_A(q) = 4π (Z_A − f^x_A(q)) / q²``.

    The electron-diffraction counterpart of :func:`atomic_form_factors`: each atom's
    Fourier
    potential about its own nucleus -- the ECP-effective point charge ``Z_A`` minus that
    atom's
    electronic (X-ray) form factor, over ``q²`` (Mott–Bethe). Summed with the nuclear
    phases and
    Debye–Waller factors via :func:`structure_factors_from_form_factors` it reproduces
    :func:`electron_structure_factors` exactly; kept per atom it is the aspherical
    scattering table
    an electron-diffraction Hirshfeld-atom refinement needs (and its response to the
    geometry is the
    X-ray one scaled by ``−4π/q²``, so the exact density-response term carries straight
    over).

    All arguments are as :func:`atomic_form_factors`. ``q = 0`` has no finite value and
    is returned
    as ``nan``.

    :return: Complex electron form factors in atomic units (Hartree·Bohr³), shape (N,
        natm).
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    fx = atomic_form_factors(
        density,
        q,
        partition=partition,
        grid_level=grid_level,
        q_batch=q_batch,
        grid_batch=grid_batch,
        cutoff=cutoff,
        rho_min=rho_min,
        atom_min=atom_min,
        bin_size=bin_size,
        cell=cell,
        images=images,
        device=device,
        dtype=dtype,
    )
    charges = np.asarray(density.nuclear_charges, dtype=float)
    q2_bohr = np.einsum("ni,ni->n", q, q) * BOHR_TO_ANG**2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 4.0 * np.pi * (charges[None, :] - fx) / q2_bohr[:, None]
    out[q2_bohr == 0] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Independent atom model
# --------------------------------------------------------------------------- #


def spherical_atom_form_factor(
    z: int, q_norm: np.ndarray, *, ecp: str | None = None
) -> np.ndarray:
    """
    Form factor of the spherical free atom, ``f(q) = 4π ∫ ρ(r) r² sin(qr)/(qr) dr``.

    The atom is the tabulated Hartree--Fock free atom behind
    :func:`qpet.partition.free_atom_density`, so the independent atom model
    built from it is the Fourier transform of the Hirshfeld promolecule.

    :param z: Atomic number.
    :param q_norm: ``|q|`` in Å⁻¹ (with the ``2π``), any shape.
    :param ecp: ECP of the free atom, see :func:`~qpet.partition.free_atom_density`.
    :return: Form factor in electrons, same shape as ``q_norm``.
    """
    r = np.geomspace(1e-4, 20.0, 4000)  # Bohr
    rho = free_atom_density(z, r, ecp=ecp)
    q_bohr = np.asarray(q_norm, dtype=float).reshape(-1) * BOHR_TO_ANG
    qr = np.outer(q_bohr, r)
    kernel = np.ones_like(qr)
    nonzero = qr > 1e-12
    kernel[nonzero] = np.sin(qr[nonzero]) / qr[nonzero]
    f = 4.0 * np.pi * np.trapezoid(kernel * (rho * r**2)[None, :], r, axis=1)
    return np.asarray(np.reshape(f, np.shape(q_norm)))


def independent_atom_structure_factors(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    q: np.ndarray,
    *,
    adps: np.ndarray | None = None,
    ecp: str | None = None,
) -> np.ndarray:
    """
    Structure factors of the independent atom model (spherical free atoms).

    :param atomic_numbers: Shape (natm,).
    :param positions: Atomic positions in Å, shape (natm, 3).
    :param q: Scattering vectors in Å⁻¹, shape (N, 3).
    :param adps: Displacement parameters, see :func:`debye_waller`.
    :param ecp: ECP of the free atoms.
    :return: Complex structure factors in electrons, shape (N,).
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    numbers = np.asarray(atomic_numbers)
    positions = np.asarray(positions, dtype=float)
    q_norm = np.linalg.norm(q, axis=1)
    t = debye_waller(q, adps, len(numbers))
    phase = np.exp(1j * (q @ positions.T))
    out = np.zeros(len(q), dtype=complex)
    for z in np.unique(numbers):
        sel = numbers == z
        f = spherical_atom_form_factor(int(z), q_norm, ecp=ecp)
        out += f * (t[:, sel] * phase[:, sel]).sum(axis=1)
    return out


# --------------------------------------------------------------------------- #
# .tsc files (NoSpherA2 / Olex2 tabulated scattering contributions)
# --------------------------------------------------------------------------- #


def write_tsc(
    path: str | Path,
    hkl: np.ndarray,
    form_factors: np.ndarray,
    labels: list[str],
    *,
    title: str = "qpet",
) -> None:
    """
    Write per-atom form factors as a ``.tsc`` table for NoSpherA2 (Olex2).

    One row per reflection: ``h k l`` followed by ``re,im`` of every atom's
    form factor, in the order of ``labels``. Symmetry is ``expanded`` (the
    caller supplies every reflection it wants), and no anomalous dispersion.

    :param path: Output file.
    :param hkl: Miller indices, shape (N, 3).
    :param form_factors: Complex form factors, shape (N, natm).
    :param labels: Atom labels as in the refinement model, length natm.
    :param title: Free-text title line.
    """
    hkl = np.asarray(hkl)
    ff = np.asarray(form_factors, dtype=complex)
    if ff.shape != (len(hkl), len(labels)):
        raise ValueError(
            f"form_factors shape {ff.shape} does not match {len(hkl)} reflections "
            f"× {len(labels)} atoms"
        )
    lines = [
        f"TITLE: {title}",
        "SYMM: expanded",
        "AD: FALSE",
        "SCATTERERS: " + " ".join(labels),
        "DATA:",
    ]
    for (h, k, l_), row in zip(hkl, ff, strict=True):
        values = " ".join(f"{v.real:.8e},{v.imag:.8e}" for v in row)
        lines.append(f"{int(h)} {int(k)} {int(l_)} {values}")
    Path(path).write_text("\n".join(lines) + "\n")


def read_tsc(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Read a ``.tsc`` file written by :func:`write_tsc` (or NoSpherA2).

    :return: ``(hkl, form_factors, labels)`` with shapes (N, 3) and (N, natm).
    """
    labels: list[str] = []
    hkl: list[list[int]] = []
    rows: list[list[complex]] = []
    in_data = False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_data:
            fields = line.split()
            hkl.append([int(x) for x in fields[:3]])
            rows.append([complex(*map(float, f.split(","))) for f in fields[3:]])
        elif line.startswith("SCATTERERS:"):
            labels = line.split(":", 1)[1].split()
        elif line.startswith("DATA:"):
            in_data = True
    return np.asarray(hkl, dtype=int), np.asarray(rows, dtype=complex), labels
