"""Grids: regular ones for cube files, Becke ones for integrals. Everything in bohr.

A density map coloured by a potential, as ChimeraX or PyMOL show it, needs
both cube files on the same grid: build one :class:`CubeGrid` and sample every
field on it::

    grid = CubeGrid.enclosing(mol.atom_coords())
    write_cube("rho.cube", mol, grid, grid.sample(rho.density))
    write_cube("esp.cube", mol, grid, grid.sample(rho.potential))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyscf import dft, gto

from qpet.conditioning import nuclear_charges_of


@dataclass(frozen=True)
class CubeGrid:
    """Regular grid: ``origin + i a_0 + j a_1 + k a_2`` for ``(i, j, k) < shape``."""

    origin: np.ndarray
    axes: np.ndarray
    """Step vectors as rows, shape (3, 3)."""
    shape: tuple[int, int, int]

    @classmethod
    def enclosing(
        cls, positions: np.ndarray, *, padding: float = 7.5, spacing: float = 0.4
    ) -> CubeGrid:
        """
        Orthogonal grid around ``positions`` with ``padding`` on every side.

        The positions need not be those of the field sampled on it: enclose a
        ligand and sample the field of the whole complex.
        """
        lo = np.min(positions, axis=0) - padding
        extent = np.max(positions, axis=0) + padding - lo
        shape = tuple(int(n) for n in np.ceil(extent / spacing).astype(int) + 1)
        return cls(lo, spacing * np.eye(3), shape)  # type: ignore[arg-type]

    @property
    def points(self) -> np.ndarray:
        """Shape (n_points, 3), C order over (i, j, k) as in a cube file."""
        idx = np.indices(self.shape).reshape(3, -1).T
        return self.origin + idx @ self.axes

    def sample(self, field: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """``field`` on every point, reshaped to :attr:`shape`."""
        return np.asarray(field(self.points)).reshape(self.shape)


@dataclass(frozen=True)
class Cube:
    """Contents of a cube file."""

    grid: CubeGrid
    values: np.ndarray
    numbers: np.ndarray
    positions: np.ndarray


def becke_grid(mol: gto.Mole, level: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Points and weights of PySCF's quadrature: ``∫ f ≈ weights @ f(points)``."""
    grids = dft.Grids(mol)
    grids.level = level
    grids.build()
    return grids.coords, grids.weights


def write_cube(
    path: str | Path,
    mol: gto.Mole,
    grid: CubeGrid,
    values: np.ndarray,
    comment: str = "",
) -> None:
    """
    Gaussian cube file of ``values`` on ``grid``, with the atoms of ``mol``.

    The atom lines carry the element and the nuclear charge, which differ under an ECP.
    """
    values = np.asarray(values).reshape(grid.shape)
    numbers = [gto.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)]
    lines = [
        "qpet",
        comment,
        f"{mol.natm:5d}" + "".join(f"{x:12.6f}" for x in grid.origin),
    ]
    lines += [
        f"{n:5d}" + "".join(f"{x:12.6f}" for x in axis)
        for n, axis in zip(grid.shape, grid.axes, strict=True)
    ]
    lines += [
        f"{z:5d}{q:12.6f}" + "".join(f"{x:12.6f}" for x in r)
        for z, q, r in zip(
            numbers, nuclear_charges_of(mol), mol.atom_coords(), strict=True
        )
    ]
    rows = values.reshape(-1, grid.shape[2])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        for row in rows:
            for k in range(0, len(row), 6):
                f.write(" ".join(f"{v:13.5E}" for v in row[k : k + 6]) + "\n")


def read_cube(path: str | Path) -> Cube:
    with open(path) as f:
        lines = f.read().splitlines()
    first, *origin = lines[2].split()[:4]
    n_atoms = int(first)
    head = np.array([line.split()[:4] for line in lines[3:6]], dtype=float)
    atoms = np.array([line.split()[:5] for line in lines[6 : 6 + n_atoms]], dtype=float)
    shape = tuple(int(n) for n in head[:, 0])
    values = np.array(" ".join(lines[6 + n_atoms :]).split(), dtype=float).reshape(
        shape
    )
    grid = CubeGrid(np.array(origin, dtype=float), head[:, 1:], shape)  # type: ignore[arg-type]
    return Cube(grid, values, atoms[:, 0].astype(int), atoms[:, 2:])
