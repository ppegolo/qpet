"""
Gaussian CUBE file export for volumetric scalar fields.

Writes electron density and electrostatic potential on a regular grid in a
format that ChimeraX, PyMOL, and VMD load directly. All grid geometry is
written in Bohr (positive grid counts) and field values in atomic units, the
canonical CUBE convention.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qpet.conversions import BOHR_TO_ANG

if TYPE_CHECKING:
    from qpet.fields import ScalarField

logger = logging.getLogger(__name__)


def _auto_grid(
    positions_ang: np.ndarray,
    padding_ang: float,
    spacing_ang: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Build an orthogonal grid that encloses the molecule with uniform spacing.

    :param positions_ang: Atomic positions in Angstrom, shape (N, 3)
    :param padding_ang: Padding added on every side, in Angstrom
    :param spacing_ang: Grid spacing in Angstrom (same on all axes)
    :return: ``(origin_bohr, axes_bohr, dims)`` where ``origin_bohr`` is shape
        ``(3,)``, ``axes_bohr`` is shape ``(3, 3)`` with rows = grid step
        vectors, and ``dims = (nx, ny, nz)``.
    """
    if spacing_ang <= 0:
        raise ValueError(f"spacing must be positive, got {spacing_ang}")
    if padding_ang < 0:
        raise ValueError(f"padding must be non-negative, got {padding_ang}")

    lo = positions_ang.min(axis=0) - padding_ang
    hi = positions_ang.max(axis=0) + padding_ang
    extent = hi - lo
    dims = tuple(int(np.ceil(e / spacing_ang)) for e in extent)

    spacing_bohr = spacing_ang / BOHR_TO_ANG
    origin_bohr = lo / BOHR_TO_ANG
    axes_bohr = np.eye(3) * spacing_bohr

    return origin_bohr, axes_bohr, dims  # type: ignore[return-value]


def _grid_points_ang(
    origin_bohr: np.ndarray,
    axes_bohr: np.ndarray,
    dims: tuple[int, int, int],
) -> np.ndarray:
    """
    Enumerate grid point coordinates in Angstrom, ordered (x, y, z) C-style.

    :return: Points in Angstrom, shape ``(nx*ny*nz, 3)``
    """
    nx, ny, nz = dims
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(float)
    pts_bohr = origin_bohr[None, :] + idx @ axes_bohr
    return pts_bohr * BOHR_TO_ANG


def write_cube(
    path: str | Path,
    atoms: ScalarField,
    field: np.ndarray,
    origin_bohr: np.ndarray,
    axes_bohr: np.ndarray,
    *,
    title1: str = "qpet cube",
    title2: str = "",
) -> None:
    """
    Write a Gaussian-format CUBE file.

    :param path: Output file path
    :param atoms: Density object (provides positions and atomic numbers)
    :param field: Scalar field on the grid, shape ``(nx, ny, nz)``, atomic units
    :param origin_bohr: Grid origin in Bohr, shape ``(3,)``
    :param axes_bohr: Grid step vectors in Bohr, shape ``(3, 3)`` (rows = axes)
    :param title1: First title line
    :param title2: Second title line
    """
    field = np.asarray(field)
    if field.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {field.shape}")
    nx, ny, nz = field.shape

    origin_bohr = np.asarray(origin_bohr, dtype=float).reshape(3)
    axes_bohr = np.asarray(axes_bohr, dtype=float).reshape(3, 3)

    z = atoms.atomic_numbers
    pos_bohr = atoms.positions / BOHR_TO_ANG
    natoms = len(z)

    path = Path(path)
    with open(path, "w") as f:
        f.write(f"{title1}\n")
        f.write(f"{title2}\n")
        f.write(
            f"{natoms:5d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} "
            f"{origin_bohr[2]:12.6f}\n"
        )
        for n, ax in zip((nx, ny, nz), axes_bohr, strict=True):
            f.write(f"{n:5d} {ax[0]:12.6f} {ax[1]:12.6f} {ax[2]:12.6f}\n")
        for zi, ri in zip(z, pos_bohr, strict=True):
            f.write(
                f"{int(zi):5d} {float(zi):12.6f} {ri[0]:12.6f} "
                f"{ri[1]:12.6f} {ri[2]:12.6f}\n"
            )

        for i in range(nx):
            for j in range(ny):
                row = field[i, j]
                for k0 in range(0, nz, 6):
                    chunk = row[k0 : k0 + 6]
                    f.write(" ".join(f"{v:13.5E}" for v in chunk))
                    f.write("\n")

    logger.info("Wrote CUBE file: %s (%d×%d×%d)", path, nx, ny, nz)


def _write_scalar_cube(
    atoms: ScalarField,
    path: str | Path,
    field_fn: Callable[[np.ndarray], np.ndarray],
    *,
    padding: float = 4.0,
    spacing: float = 0.2,
    title1: str = "qpet cube",
    title2: str = "",
) -> None:
    """Sample a scalar field on an auto-sized grid and write a CUBE file.

    ``field_fn`` receives the full point array and is expected to batch
    internally (the compute methods take ``batch_size`` for that).
    """
    origin_bohr, axes_bohr, dims = _auto_grid(atoms.positions, padding, spacing)
    pts_ang = _grid_points_ang(origin_bohr, axes_bohr, dims)
    field = field_fn(pts_ang).reshape(dims)
    write_cube(path, atoms, field, origin_bohr, axes_bohr, title1=title1, title2=title2)


def write_density_cube(
    atoms: ScalarField,
    path: str | Path,
    *,
    padding: float = 4.0,
    spacing: float = 0.2,
    batch_size: int = 5000,
) -> None:
    """
    Write the electron density on an auto-sized grid to a CUBE file.

    :param atoms: Density object supporting ``density`` (RIDensity, DMDensity, ...)
    :param path: Output file path
    :param padding: Padding around the molecule in Angstrom (default 4.0)
    :param spacing: Grid spacing in Angstrom (default 0.2)
    :param batch_size: Points evaluated per batch (controls peak memory)
    """
    _write_scalar_cube(
        atoms,
        path,
        lambda p: atoms.density(p, batch_size=batch_size),
        padding=padding,
        spacing=spacing,
        title1="qpet electron density",
        title2="rho(r) in electrons / Bohr^3",
    )


def write_esp_cube(
    atoms: ScalarField,
    path: str | Path,
    *,
    padding: float = 4.0,
    spacing: float = 0.2,
    batch_size: int = 5000,
) -> None:
    """
    Write the electrostatic potential on an auto-sized grid to a CUBE file.

    :param atoms: Density object supporting ``potential`` (RIDensity, DMDensity, ...)
    :param path: Output file path
    :param padding: Padding around the molecule in Angstrom (default 4.0)
    :param spacing: Grid spacing in Angstrom (default 0.2)
    :param batch_size: Points evaluated per batch (controls peak memory)
    """
    _write_scalar_cube(
        atoms,
        path,
        lambda p: atoms.potential(p, batch_size=batch_size),
        padding=padding,
        spacing=spacing,
        title1="qpet electrostatic potential",
        title2="V(r) in atomic units (Hartree / e)",
    )
