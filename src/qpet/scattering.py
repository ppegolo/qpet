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

from pathlib import Path
from typing import Literal

import numpy as np
from pyscf import gto

from .conversions import BOHR_TO_ANG
from .cube import becke_grid
from .fields import RIDensity
from .partition import FragmentDensity, free_atom_density

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
    "write_tsc",
    "read_tsc",
]

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


def _ri_atomic_terms(density: Density, q: np.ndarray) -> np.ndarray:
    """``Σ_{P∈A} c_P χ̃_P(q)``, phase of the centre included, shape (N, natm)."""
    weighted = _aux_fourier(density.auxmol, q) * np.asarray(density.coeffs)[None, :]
    starts = _atom_slices(density.auxmol)
    return np.add.reduceat(weighted, starts, axis=1)


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
    cell: np.ndarray | None = None,
    images: int = 1,
    device: str | None = None,
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

    grid = becke_grid(density.auxmol, level=grid_level)
    promol = None
    if cell is not None:
        promol = density.with_periodic_images(np.asarray(cell), images=images)
    source = promol if promol is not None else density
    rho = source.density(grid.points) * grid.weights  # (G,) periodic electrons if cell
    weights = _hirshfeld_atom_weights(density, grid.points, promol_density=promol)
    weighted = rho[:, None] * weights  # (G, natm)
    out = _grid_fourier_sum(q, grid.points, weighted, device=device, q_batch=q_batch)
    return out * phase


def _grid_fourier_sum(q, points_ang, weighted, *, device=None, q_batch=256):
    """``Σ_G e^{i q·r_G} weighted[G, A]`` -> ``(N_q, natm)``, on ``device`` or NumPy.

    The exponential and the contraction over the ~10^5 quadrature points dominate the
    Hirshfeld form-factor cost; on a GPU this is a few dense complex matmuls. The NumPy
    path (``device=None``) is the reference and gives an identical result.
    """
    if device is not None:
        import torch

        dev = torch.device(device)
        pts = torch.as_tensor(np.ascontiguousarray(points_ang), device=dev)
        wt = torch.as_tensor(
            np.ascontiguousarray(weighted), dtype=torch.complex128, device=dev
        )
        qt = torch.as_tensor(np.ascontiguousarray(q), device=dev)
        out = torch.empty((len(q), wt.shape[1]), dtype=torch.complex128, device=dev)
        for start in range(0, len(q), q_batch):
            stop = min(start + q_batch, len(q))
            wave = torch.exp(1j * (qt[start:stop] @ pts.T))  # (n, G)
            out[start:stop] = wave @ wt
        return out.cpu().numpy()

    out = np.empty((len(q), weighted.shape[1]), dtype=complex)
    for start in range(0, len(q), q_batch):
        stop = min(start + q_batch, len(q))
        wave = np.exp(1j * (q[start:stop] @ points_ang.T))  # (n, G)
        out[start:stop] = wave @ weighted
    return out


def _free_atom_grid(auxmol, numbers, points_bohr, ecp):
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
