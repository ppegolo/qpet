"""COSMO-style implicit solvent for a predicted RI density.

The solute is placed in a cavity carved out of a dielectric continuum. COSMO takes
the continuum to be a conductor, so the total potential vanishes on the cavity
surface, solves for the surface charges that enforce that, and then scales them by

    f(eps) = (eps - 1) / (eps + x)

to recover a finite dielectric. The solvation energy is ``0.5 * q . V``, half the
interaction of the induced charges with the potential that induced them.

Two pieces already exist here: the cavity surface with area weights is
:func:`qpet.surface.sas_points`, and the solute potential on it is
:class:`qpet.esp_fast.FastESP` plus the nuclei. What this module adds is the solve.

**Why the solve is direct.** The surface matrix is symmetric but *not* positive
definite: on a polyatomic cavity, segments straddling a sphere intersection sit
close enough that the off-diagonal terms overcome the self-interaction and a
negative eigenvalue appears. Conjugate gradients assumes definiteness and breaks
down on exactly those systems -- it converged on a lone ion and failed on ammonium.
A dense factorisation does not care, and at these sizes it is not the expensive
part: thirty thousand segments is a cubic solve of about 1e13 operations, seconds
on a GPU, against a cavity that large needing tens of gigabytes only past a hundred
thousand segments. The matrix is built in row blocks so it is never assembled twice.

**What this is and is not.** The density comes from a model trained on gas-phase
references and is not relaxed in the reaction field, so this is a one-shot
correction to a fixed density, not a self-consistent reaction field. It answers
"what would this density cost to solvate", not "what would this molecule's density
become in solvent".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from pyscf import gto

from qpet.surface import DEFAULT_POINTS_PER_ATOM, sas_points

if TYPE_CHECKING:
    import torch

#: Denominator offset in the dielectric scaling. 0.5 is the usual COSMO choice.
DEFAULT_SCREENING_OFFSET = 0.5

#: Self-interaction of a segment, ``DIAGONAL_FACTOR * sqrt(4 pi / area)``. The
#: standard COSMO value, from the potential at the centre of a charged spherical
#: cap of the same area.
DIAGONAL_FACTOR = 1.07

#: Cavity radii are conventionally larger than the van der Waals surface used for
#: an ESP grid; 1.2 is the common COSMO scaling.
DEFAULT_CAVITY_SCALE = 1.2


def screening_factor(
    epsilon: float, offset: float = DEFAULT_SCREENING_OFFSET
) -> float:
    """``(eps - 1) / (eps + offset)``; 1 for a conductor, 0 for vacuum."""
    if epsilon < 1.0:
        raise ValueError(f"dielectric constant must be at least 1, got {epsilon}")
    return (epsilon - 1.0) / (epsilon + offset)


@dataclass
class CosmoResult:
    """Outcome of one COSMO solve.

    :param energy: solvation energy in Hartree (negative for a stabilised solute).
    :param charges: induced surface charges, already scaled by ``f(eps)``.
    :param potential: solute potential at each segment, atomic units.
    :param coords: segment centres in Bohr.
    :param areas: segment areas in Bohr^2.
    :param iterations: conjugate-gradient iterations used.
    :param residual: final relative residual of the conductor condition.
    """

    energy: float
    charges: np.ndarray
    potential: np.ndarray
    coords: np.ndarray
    areas: np.ndarray
    iterations: int
    residual: float

    @property
    def total_charge(self) -> float:
        """Sum of the induced charges.

        For a conductor enclosing a net charge ``Q`` this approaches ``-f * Q``, so
        it is a direct check on the cavity and the solve.
        """
        return float(self.charges.sum())


class COSMO:
    """COSMO solver for a fixed auxiliary-basis geometry.

    The cavity and the surface interaction depend only on the geometry, so build
    once and solve for as many coefficient vectors as needed.

    :param auxmol: auxiliary-basis ``Mole``, as carried by
        :class:`~qpet.fields.RIDensity`.
    :param epsilon: solvent dielectric constant (78.4 for water).
    :param cavity_scale: van der Waals scaling of the cavity surface.
    :param points_per_atom: segments per atomic sphere before culling.
    :param device: torch device.
    :param dtype: ``torch.float64`` is the safe default here -- the surface matrix
        is indefinite and its conditioning is set by the closest pair of segments.
    :param screening_offset: the ``x`` in ``(eps - 1) / (eps + x)``.
    """

    def __init__(
        self,
        auxmol: gto.Mole,
        epsilon: float = 78.4,
        cavity_scale: float | Sequence[float] = DEFAULT_CAVITY_SCALE,
        points_per_atom: int = DEFAULT_POINTS_PER_ATOM,
        device: str = "cuda",
        dtype: "torch.dtype | None" = None,  # noqa: F821
        screening_offset: float = DEFAULT_SCREENING_OFFSET,
    ) -> None:
        import torch

        from qpet.esp_fast import FastESP

        self.auxmol = auxmol
        self.epsilon = epsilon
        self.factor = screening_factor(epsilon, screening_offset)
        self.dtype = dtype or torch.float64
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        coords, areas = sas_points(auxmol, cavity_scale, points_per_atom)
        self.coords = coords
        self.areas = areas

        self._coords = torch.as_tensor(coords, dtype=self.dtype, device=self.device)
        self._diagonal = torch.as_tensor(
            DIAGONAL_FACTOR * np.sqrt(4.0 * np.pi / areas),
            dtype=self.dtype,
            device=self.device,
        )
        # The potential evaluator runs in its own precision: float32 is plenty for
        # the right-hand side, and it is by far the more expensive half.
        self._esp = FastESP(auxmol, device=str(self.device), dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def solve(self, coeffs: np.ndarray, tolerance: float = 1e-6) -> CosmoResult:
        """Solve for the induced surface charges and the solvation energy.

        :param coeffs: RI coefficients of the solute density.
        :param tolerance: relative residual above which the solve is rejected. The
            factorisation is direct, so this only catches a pathological cavity.
        :raises RuntimeError: if the residual exceeds ``tolerance``.
        """
        import torch

        potential = self.surface_potential(coeffs)
        rhs = -torch.as_tensor(potential, dtype=self.dtype, device=self.device)

        matrix = self._matrix()
        charges = torch.linalg.solve(matrix, rhs)
        residual = float(
            torch.linalg.vector_norm(matrix @ charges - rhs)
            / max(float(torch.linalg.vector_norm(rhs)), 1e-300)
        )
        del matrix
        if residual > tolerance:
            raise RuntimeError(
                f"COSMO solve left a relative residual of {residual:.2e}; the "
                "cavity is likely degenerate (segments coincident or nearly so)"
            )
        charges = charges * self.factor

        energy = 0.5 * float(
            (charges * torch.as_tensor(potential, dtype=self.dtype, device=self.device))
            .sum()
        )
        return CosmoResult(
            energy=energy,
            charges=charges.cpu().numpy(),
            potential=potential,
            coords=self.coords,
            areas=self.areas,
            iterations=1,
            residual=residual,
        )

    def surface_potential(self, coeffs: np.ndarray) -> np.ndarray:
        """Total solute potential at the cavity segments, nuclei included."""
        from qpet.fields import _nuclear_potential

        electronic = self._esp.potential(coeffs, self.coords)
        nuclear = _nuclear_potential(
            self.auxmol.atom_coords(), self.auxmol.atom_charges(), self.coords
        )
        return nuclear - electronic

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _apply(self, vector: "torch.Tensor", chunk: int = 4096) -> "torch.Tensor":
        """Surface interaction matrix applied to a vector, without storing it."""
        import torch

        out = torch.empty_like(vector)
        for start in range(0, len(vector), chunk):
            stop = min(start + chunk, len(vector))
            block = self._coords[start:stop]
            delta = block[:, None, :] - self._coords[None, :, :]
            distance = (delta * delta).sum(dim=-1).sqrt()
            # The diagonal is the segment's self-interaction, not 1/0.
            inverse = torch.where(
                distance > 0, distance.reciprocal(), torch.zeros_like(distance)
            )
            rows = torch.arange(start, stop, device=self.device)
            inverse[rows - start, rows] = self._diagonal[start:stop]
            out[start:stop] = inverse @ vector
        return out

    def _matrix(self) -> "torch.Tensor":
        """Assemble the surface interaction matrix, in row blocks."""
        import torch

        n = len(self.coords)
        matrix = torch.empty((n, n), dtype=self.dtype, device=self.device)
        for start in range(0, n, 4096):
            stop = min(start + 4096, n)
            delta = self._coords[start:stop, None, :] - self._coords[None, :, :]
            distance = (delta * delta).sum(dim=-1).sqrt()
            block = torch.where(
                distance > 0, distance.reciprocal(), torch.zeros_like(distance)
            )
            rows = torch.arange(start, stop, device=self.device)
            block[rows - start, rows] = self._diagonal[start:stop]
            matrix[start:stop] = block
        return matrix
