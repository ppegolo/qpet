"""Solvent-accessible surface grids.

The construction matches metatrain's ``compute_surface_points``: a Fibonacci
sphere per atom at ``shell_scale`` times its van der Waals radius, with points
buried inside another atom's scaled sphere culled, each point carrying its share
of its sphere's area. A sequence of scalings gives a RESP-style multi-shell
surface, concatenated in ascending shell order.

The difference is the burial test. Testing every point against every atom is
``n_atoms**2 * points_per_atom`` distance evaluations -- 1.25e9 for a five
thousand atom system, which dominates everything else in an ESP workflow. A point
can only be buried by an atom whose scaled sphere it lies inside, so only atoms
within the largest scaled radius can matter, and a spatial index finds those
directly. Same output, ``O(n log n)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pyscf import gto

#: Points per atomic sphere, matching metatrain's ESP_POINTS_PER_ATOM.
DEFAULT_POINTS_PER_ATOM = 50

#: van der Waals scaling of the default shell, matching metatrain's
#: DEFAULT_ESP_SHELL.
DEFAULT_SHELL_SCALE = 1.4


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniformly spread unit vectors; deterministic."""
    golden = np.pi * (3.0 - np.sqrt(5.0))
    index = np.arange(n)
    z = 1.0 - (2.0 * index + 1.0) / n
    radius = np.sqrt(1.0 - z * z)
    angle = golden * index
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)


def sas_points(
    mol: gto.Mole,
    shell_scale: float | Sequence[float] = DEFAULT_SHELL_SCALE,
    points_per_atom: int = DEFAULT_POINTS_PER_ATOM,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a solvent-accessible surface grid with area weights.

    :param mol: built ``Mole``; only its geometry and atomic charges are used.
    :param shell_scale: van der Waals scaling -- one value, or several whose
        grids are concatenated in ascending order.
    :param points_per_atom: points on each atom's sphere before culling.
    :return: ``(coords, weights)`` -- points in Bohr, shape ``(n, 3)``, and their
        area weights in Bohr^2, shape ``(n,)``.
    """
    from scipy.spatial import cKDTree

    from pyscf.data import radii as pyscf_radii

    scales = (
        (float(shell_scale),)
        if np.isscalar(shell_scale)
        else tuple(float(s) for s in shell_scale)
    )

    centres = mol.atom_coords()  # Bohr
    charges = mol.atom_charges()
    unit = _fibonacci_sphere(points_per_atom)

    all_coords, all_weights = [], []
    for scale in sorted(scales):
        sphere_radii = scale * pyscf_radii.VDW[charges]
        points = (
            centres[:, None, :] + sphere_radii[:, None, None] * unit[None, :, :]
        ).reshape(-1, 3)
        owner = np.repeat(np.arange(mol.natm), points_per_atom)

        # Invert the test: instead of asking each point which atoms might bury it,
        # ask each atom which points fall inside its own sphere. That is one query
        # per atom rather than one per point -- fifty times fewer -- and each query
        # uses its own exact radius, so there is no candidate list to filter.
        point_tree = cKDTree(points)
        buried = np.zeros(len(points), dtype=bool)
        for atom in range(mol.natm):
            inside = point_tree.query_ball_point(centres[atom], sphere_radii[atom])
            if inside:
                index = np.fromiter(inside, dtype=np.int64)
                buried[index[owner[index] != atom]] = True

        area_per_point = 4.0 * np.pi * sphere_radii**2 / points_per_atom
        all_coords.append(points[~buried])
        all_weights.append(np.repeat(area_per_point, points_per_atom)[~buried])

    return np.concatenate(all_coords), np.concatenate(all_weights)
