"""Solvent-accessible surfaces with per-point areas.

A surface here is a point cloud in which every point carries the area it
represents. That is what makes an area-weighted average meaningful: a fixed
point count per atom puts as many points on a hydrogen as on a sulfur, so an
unweighted mean over-counts small atoms.

:class:`SurfacePatch` is the container the ESP metrics consume — any subset of
surface points with areas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

#: Default surface points per atom.
POINTS_PER_ATOM = 200

#: Probe radius in Angstrom (a water molecule).
PROBE_RADIUS = 1.4

#: Bondi van der Waals radii (Å). A deliberate numeric pin: the milestone
#: validation numbers depend on these exact values — do not swap for a derived
#: table (e.g. ``pyscf.data.radii``) without re-baselining.
VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "Se": 1.90,
    "Fe": 1.80,
    "Zn": 1.39,
    "Mg": 1.73,
}


def get_vdw(symbol: str) -> float:
    """Look up a VDW radius, warning on fallback to 1.5 Å."""
    r = VDW_RADII.get(symbol)
    if r is None:
        logger.warning("No VDW radius for element '%s'; using 1.5 Å fallback", symbol)
        return 1.5
    return r


@dataclass(frozen=True)
class SurfacePatch:
    """Points on a molecular surface, with the area each one represents."""

    points: np.ndarray
    """Positions in Angstrom, shape (N, 3)."""
    areas: np.ndarray
    """Area represented by each point, in Angstrom^2, shape (N,)."""
    owner: np.ndarray
    """Index of the partner whose surface each point lies on, shape (N,)."""
    definition: str
    """The rule that built this patch, named by the caller."""

    def __len__(self) -> int:
        return len(self.points)

    @property
    def total_area(self) -> float:
        """Summed area of the patch, in Angstrom^2."""
        return float(self.areas.sum())

    @property
    def weights(self) -> np.ndarray:
        """Areas normalised to sum to one."""
        if len(self.areas) == 0:
            return np.zeros(0)
        total = self.areas.sum()
        assert total > 0, "patch areas must be positive"
        normalised: np.ndarray = self.areas / total
        return normalised

    def __repr__(self) -> str:
        return (
            f"SurfacePatch({len(self)} points, {self.total_area:.1f} A^2, "
            f"definition={self.definition!r})"
        )


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Near-uniform points on the unit sphere, deterministic."""
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * np.arange(count) + 1.0) / count
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = golden * np.arange(count)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def solvent_accessible_surface(
    positions: np.ndarray,
    symbols: list[str],
    *,
    probe_radius: float = PROBE_RADIUS,
    points_per_atom: int = POINTS_PER_ATOM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The solvent-accessible surface of one molecule, with per-point areas.

    Points sit at ``r_vdW + probe`` from each atom; those falling inside another
    atom's accessible sphere are dropped. Each surviving point carries the area
    it represents.

    :param positions: Atomic positions in Angstrom, shape (N, 3).
    :param symbols: Element symbol per atom.
    :param probe_radius: Probe radius in Angstrom.
    :param points_per_atom: Points placed on each atom before culling.
    :return: ``(points, areas)`` in Angstrom and Angstrom^2.
    """
    positions = np.asarray(positions, dtype=float)
    radii = np.array([get_vdw(s) for s in symbols]) + probe_radius
    unit = _fibonacci_sphere(points_per_atom)

    points: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    for atom, (centre, radius) in enumerate(zip(positions, radii, strict=True)):
        candidates = centre + radius * unit
        others = np.arange(len(positions)) != atom
        if others.any():
            distances = cdist(candidates, positions[others])
            buried = np.any(distances < radii[others][None, :], axis=1)
        else:
            buried = np.zeros(len(candidates), dtype=bool)
        kept = candidates[~buried]
        points.append(kept)
        areas.append(np.full(len(kept), 4.0 * np.pi * radius**2 / points_per_atom))

    if not points:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(points), np.concatenate(areas)
