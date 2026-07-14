"""Shared infrastructure for scalar-field calculators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qpet.structure import Structure

# Clamp distance to avoid singularity at nuclei (Bohr).
_DIST_FLOOR: float = 0.1


def _nuclear_potential(
    positions_bohr: np.ndarray,
    charges: np.ndarray,
    pts_bohr: np.ndarray,
) -> np.ndarray:
    """
    Nuclear contribution to ESP: ``V_nuc = Σ_A Z_A / |r - R_A|``.

    :param positions_bohr: Nuclear positions in Bohr, shape (N_atoms, 3)
    :param charges: Nuclear charges (atomic numbers), shape (N_atoms,)
    :param pts_bohr: Evaluation points in Bohr, shape (N_pts, 3)
    :return: Nuclear potential in atomic units, shape (N_pts,)
    """
    v_nuc = np.zeros(len(pts_bohr))
    for i in range(len(charges)):
        dist = np.linalg.norm(pts_bohr - positions_bohr[i], axis=1)
        v_nuc += charges[i] / np.maximum(dist, _DIST_FLOOR)
    return v_nuc


class Calculator:
    """
    Base class for scalar-field calculators.

    A calculator evaluates a scalar field — electron density or electrostatic
    potential — at arbitrary points given a
    :class:`~qpet.structure.Structure`.

    Density and potential are on equal footing: both :meth:`compute_density`
    and :meth:`compute_potential` default-raise :class:`NotImplementedError`,
    and a subclass overrides whichever field(s) it can produce.
    """

    def compute_density(
        self, atoms: Structure, pts: np.ndarray, batch_size: int = 5000
    ) -> np.ndarray:
        """
        Compute electron density at given points.

        :param atoms: Molecular structure (:class:`~qpet.structure.Structure`)
        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per batch (controls peak memory)
        :return: Electron density in atomic units (electrons / Bohr^3), shape (N,)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support density evaluation."
        )

    def compute_potential(self, atoms: Structure, pts: np.ndarray) -> np.ndarray:
        """
        Compute electrostatic potential at given points.

        :param atoms: Molecular structure (:class:`~qpet.structure.Structure`)
        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :return: Electrostatic potential in atomic units, shape (N,)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support potential evaluation."
        )
