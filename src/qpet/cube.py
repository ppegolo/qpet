"""
Gaussian CUBE file import/export and integration grids.

Writes (and reads back) electron density and electrostatic potential on a
regular grid in a format that ChimeraX, PyMOL, and VMD load directly. All
grid geometry is in Bohr (positive grid counts) and field values in atomic
units, the canonical CUBE convention.

:func:`becke_grid` exposes PySCF's molecular quadrature so arbitrary density
functionals can be integrated in one line::

    grid = becke_grid(mol)
    n_elec = (grid.weights * rho.density(grid.points)).sum()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qpet.conversions import BOHR_TO_ANG

if TYPE_CHECKING:
    from pyscf import gto

    from qpet.fields import ScalarField

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BeckeGrid:
    """Molecular integration grid: ``∫ f dr ≈ Σ weights · f(points)``."""

    points: np.ndarray
    """Grid points in Angstrom, shape (N, 3)."""
    weights: np.ndarray
    """Quadrature weights in Bohr^3, shape (N,)."""


def becke_grid(mol: gto.Mole, level: int = 3) -> BeckeGrid:
    """
    Build a Becke molecular quadrature grid for integrating scalar fields.

    :param mol: Built PySCF ``Mole``.
    :param level: PySCF grid level (higher = denser; default 3).
    :return: :class:`BeckeGrid` with points in Angstrom and weights in Bohr^3.
    """
    from pyscf import dft

    grids = dft.Grids(mol)
    grids.level = level
    grids.build()
    return BeckeGrid(points=grids.coords * BOHR_TO_ANG, weights=grids.weights)


@dataclass(frozen=True)
class CubeData:
    """Parsed contents of a Gaussian CUBE file (geometry in Bohr)."""

    title1: str
    title2: str
    origin_bohr: np.ndarray
    """Grid origin in Bohr, shape (3,)."""
    axes_bohr: np.ndarray
    """Grid step vectors in Bohr, shape (3, 3), rows = axes."""
    dims: tuple[int, int, int]
    atomic_numbers: np.ndarray
    positions_bohr: np.ndarray
    field: np.ndarray
    """Scalar field on the grid, shape ``dims``, atomic units."""

    def points(self) -> np.ndarray:
        """Grid point coordinates in Angstrom, shape ``(nx*ny*nz, 3)``."""
        return _grid_points_ang(self.origin_bohr, self.axes_bohr, self.dims)


def read_cube(path: str | Path) -> CubeData:
    """
    Read a Gaussian-format CUBE file.

    :param path: Input file path.
    :return: :class:`CubeData` with geometry in Bohr and the field reshaped
        to its grid dimensions.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    title1 = lines[0]
    title2 = lines[1]
    parts = lines[2].split()
    natoms = int(parts[0])
    origin = np.array([float(x) for x in parts[1:4]])

    axes = np.zeros((3, 3))
    dims = []
    for i in range(3):
        p = lines[3 + i].split()
        dims.append(int(p[0]))
        axes[i] = [float(x) for x in p[1:4]]

    z = []
    pos = []
    for i in range(natoms):
        p = lines[6 + i].split()
        z.append(int(p[0]))
        pos.append([float(p[2]), float(p[3]), float(p[4])])

    data_tokens: list[float] = []
    for line in lines[6 + natoms :]:
        data_tokens.extend(float(x) for x in line.split())

    return CubeData(
        title1=title1,
        title2=title2,
        origin_bohr=origin,
        axes_bohr=axes,
        dims=(dims[0], dims[1], dims[2]),
        atomic_numbers=np.array(z),
        positions_bohr=np.array(pos),
        field=np.array(data_tokens).reshape(dims),
    )


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


@dataclass(frozen=True)
class CubeGrid:
    """A regular grid, shared by every field that must be compared on it.

    The point of making the grid an object is that ChimeraX and PyMOL can only
    colour one map by another when the two sit on *identical* grids. Building a
    grid from one geometry and then sampling several fields on it — a ligand
    density, the receptor's potential, a predicted-minus-reference difference —
    is what makes those files loadable together.
    """

    origin_bohr: np.ndarray
    """Grid origin in Bohr, shape (3,)."""
    axes_bohr: np.ndarray
    """Grid step vectors in Bohr, shape (3, 3), rows = axes."""
    dims: tuple[int, int, int]
    """Point counts along each axis."""

    @classmethod
    def enclosing(
        cls,
        positions_ang: np.ndarray,
        *,
        padding: float = 4.0,
        spacing: float = 0.2,
    ) -> CubeGrid:
        """
        Build an orthogonal grid enclosing ``positions_ang`` with uniform spacing.

        The positions need not be the atoms of the field later sampled on it: to
        look at a ligand inside a large complex, enclose the ligand and sample
        the complex's field, which keeps the grid small.

        :param positions_ang: Positions to enclose, in Angstrom, shape (N, 3).
        :param padding: Padding added on every side, in Angstrom.
        :param spacing: Grid spacing in Angstrom, the same on all axes.
        """
        origin, axes, dims = _auto_grid(
            np.asarray(positions_ang, dtype=float), padding, spacing
        )
        return cls(origin_bohr=origin, axes_bohr=axes, dims=dims)

    @property
    def n_points(self) -> int:
        """Total number of grid points."""
        nx, ny, nz = self.dims
        return nx * ny * nz

    def points_ang(self) -> np.ndarray:
        """Grid point coordinates in Angstrom, shape ``(n_points, 3)``."""
        return _grid_points_ang(self.origin_bohr, self.axes_bohr, self.dims)

    def sample(self, field_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Evaluate a field on this grid.

        :param field_fn: Callable taking points in Angstrom, shape (N, 3), and
            returning one value per point. It is expected to batch internally.
        :return: The field with shape :attr:`dims`.
        """
        return np.asarray(field_fn(self.points_ang())).reshape(self.dims)

    def __repr__(self) -> str:
        nx, ny, nz = self.dims
        spacing = np.linalg.norm(self.axes_bohr, axis=1) * BOHR_TO_ANG
        return (
            f"CubeGrid({nx}x{ny}x{nz} = {self.n_points} points, "
            f"spacing {spacing[0]:.2f} A)"
        )


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
    grid: CubeGrid | None = None,
    padding: float = 4.0,
    spacing: float = 0.2,
    title1: str = "qpet cube",
    title2: str = "",
) -> None:
    """Sample a scalar field on a grid and write a CUBE file.

    ``field_fn`` receives the full point array and is expected to batch
    internally (the compute methods take ``batch_size`` for that). Without an
    explicit ``grid``, one is auto-sized around ``atoms``.
    """
    if grid is None:
        grid = CubeGrid.enclosing(atoms.positions, padding=padding, spacing=spacing)
    write_cube(
        path,
        atoms,
        grid.sample(field_fn),
        grid.origin_bohr,
        grid.axes_bohr,
        title1=title1,
        title2=title2,
    )


def write_density_cube(
    atoms: ScalarField,
    path: str | Path,
    *,
    grid: CubeGrid | None = None,
    padding: float = 4.0,
    spacing: float = 0.2,
    batch_size: int = 5000,
) -> None:
    """
    Write the electron density to a CUBE file.

    :param atoms: Density object supporting ``density`` (RIDensity, DMDensity, ...)
    :param path: Output file path
    :param grid: Grid to sample on; auto-sized around ``atoms`` when omitted.
        Pass an explicit :class:`CubeGrid` when this map must be loaded
        alongside another one.
    :param padding: Padding around the molecule in Angstrom (ignored with ``grid``)
    :param spacing: Grid spacing in Angstrom (ignored with ``grid``)
    :param batch_size: Points evaluated per batch (controls peak memory)
    """
    _write_scalar_cube(
        atoms,
        path,
        lambda p: atoms.density(p, batch_size=batch_size),
        grid=grid,
        padding=padding,
        spacing=spacing,
        title1="qpet electron density",
        title2="rho(r) in electrons / Bohr^3",
    )


def write_esp_cube(
    atoms: ScalarField,
    path: str | Path,
    *,
    grid: CubeGrid | None = None,
    padding: float = 4.0,
    spacing: float = 0.2,
    batch_size: int = 5000,
) -> None:
    """
    Write the electrostatic potential to a CUBE file.

    :param atoms: Density object supporting ``potential`` (RIDensity, DMDensity, ...)
    :param path: Output file path
    :param grid: Grid to sample on; auto-sized around ``atoms`` when omitted.
        Pass an explicit :class:`CubeGrid` to colour a density isosurface by
        this potential — the two maps must share a grid.
    :param padding: Padding around the molecule in Angstrom (ignored with ``grid``)
    :param spacing: Grid spacing in Angstrom (ignored with ``grid``)
    :param batch_size: Points evaluated per batch (controls peak memory)
    """
    _write_scalar_cube(
        atoms,
        path,
        lambda p: atoms.potential(p, batch_size=batch_size),
        grid=grid,
        padding=padding,
        spacing=spacing,
        title1="qpet electrostatic potential",
        title2="V(r) in atomic units (Hartree / e)",
    )


def write_esp_error_cube(
    predicted: ScalarField,
    reference: ScalarField,
    path: str | Path,
    *,
    grid: CubeGrid | None = None,
    padding: float = 4.0,
    spacing: float = 0.2,
    batch_size: int = 5000,
) -> None:
    """
    Write the *signed* ESP error ``V_predicted − V_reference`` to a CUBE file.

    Both fields are sampled on one grid, so the result can colour the same
    isosurface as the density. The sign is kept: on unseen chemistry the error
    is dominated by a near-constant offset, and a map of ``|ΔV|`` would hide
    exactly the structure worth looking at.

    :param predicted: The model's density field.
    :param reference: The reference density field (a ``DMDensity``, or an
        ``RIDensity`` built from reference coefficients).
    :param path: Output file path.
    :param grid: Grid to sample on; auto-sized around ``predicted`` when omitted.
    :param padding: Padding around the molecule in Angstrom (ignored with ``grid``)
    :param spacing: Grid spacing in Angstrom (ignored with ``grid``)
    :param batch_size: Points evaluated per batch (controls peak memory)
    """
    _write_scalar_cube(
        predicted,
        path,
        lambda p: (
            predicted.potential(p, batch_size=batch_size)
            - reference.potential(p, batch_size=batch_size)
        ),
        grid=grid,
        padding=padding,
        spacing=spacing,
        title1="qpet ESP error (predicted - reference)",
        title2="dV(r) in atomic units (Hartree / e)",
    )
