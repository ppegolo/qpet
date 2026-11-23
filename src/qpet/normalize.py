"""Projection of RI coefficients onto a fixed integral: N for a density, 2S for spin.

The projection is the smallest shift of the coefficients, in a metric ``M``,
that sets the integral ``q·c`` to the target:

    Δc = (target - q·c) d,    d = M⁻¹q / (q·M⁻¹q),

with ``q_k = ∫ φ_k dr`` (:func:`aux_integrals`). The metrics are

* ``"coulomb"``: the two-centre Coulomb matrix ``J``, the shift of minimum
  electrostatic self-energy. It spreads the charge over the diffuse functions
  of every atom and is the one that repairs the potential outside the
  molecule. From :data:`ITERATIVE_COULOMB_NAUX` functions on, ``J⁻¹q`` is
  solved iteratively without forming ``J``.
* ``"overlap"``: the auxiliary overlap matrix, the shift of minimum L² norm of
  the density. Dense: use ``"coulomb"`` or ``"coulomb_atom"`` on large systems.
* ``"identity"``: the shift of minimum norm of the coefficients; only the s
  functions move, in proportion to their integrals.
* ``"coulomb_atom"``: ``J`` restricted to the one-centre blocks. Only s
  functions move, with a radial weighting towards the diffuse ones, and every
  atom of a species takes the same share; linear in the system size.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import scipy.linalg
from pyscf import gto

from qpet.basis import aux_integrals

Metric = Literal["coulomb", "overlap", "identity", "coulomb_atom"]

#: Auxiliary-basis size from which ``J⁻¹q`` is solved iteratively.
ITERATIVE_COULOMB_NAUX = 10000

_MIN_REGULARISATION = 1e-10
_MAX_REGULARISATION = 1e-4


def regularised_cho_factor(
    matrix: np.ndarray, name: str = "matrix"
) -> tuple[tuple[np.ndarray, bool], float]:
    """
    Cholesky factor of ``matrix + ε diag(matrix)`` with the smallest ``ε`` that works.

    ``ε`` is zero, then 1e-10 doubling up to 1e-4: a Tikhonov shift that damps
    the directions in which a positive semi-definite matrix is numerically
    singular (diffuse functions of many nearby atoms).

    :return: The factor as ``scipy.linalg.cho_factor`` gives it, and ``ε``.
    """
    diagonal = np.diag(matrix).copy()
    epsilon = 0.0
    while epsilon <= _MAX_REGULARISATION:
        shifted = matrix.copy()
        shifted[np.diag_indices_from(shifted)] += epsilon * diagonal
        try:
            return scipy.linalg.cho_factor(shifted, overwrite_a=True), epsilon
        except np.linalg.LinAlgError:
            epsilon = _MIN_REGULARISATION if epsilon == 0 else 2 * epsilon
    raise np.linalg.LinAlgError(
        f"{name} could not be factorised even with a relative diagonal shift "
        f"of {_MAX_REGULARISATION:g}."
    )


def _dense_solve(matrix: np.ndarray, q: np.ndarray, name: str) -> np.ndarray:
    factor, epsilon = regularised_cho_factor(matrix, name)
    if epsilon:
        warnings.warn(
            f"{name} is not positive definite (naux={len(q)}); the projection "
            f"used a relative diagonal shift of {epsilon:.1e}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return scipy.linalg.cho_solve(factor, q)


def _atom_blocks_solve(auxmol: gto.Mole, q: np.ndarray) -> np.ndarray:
    x = np.zeros_like(q)
    solved: dict[str, np.ndarray] = {}
    for atom, (sh0, sh1, ao0, ao1) in enumerate(auxmol.aoslice_by_atom()):
        symbol = auxmol.atom_symbol(atom)
        if symbol not in solved:
            block = auxmol.intor("int2c2e", shls_slice=(sh0, sh1, sh0, sh1))
            solved[symbol] = scipy.linalg.cho_solve(
                scipy.linalg.cho_factor(block), q[ao0:ao1]
            )
        x[ao0:ao1] = solved[symbol]
    return x


def direction(
    auxmol: gto.Mole, metric: Metric = "coulomb", *, iterative: bool | None = None
) -> np.ndarray:
    """
    ``d = M⁻¹q / (q·M⁻¹q)``: a shift of the coefficients by ``t d`` adds ``t``.

    One direction serves every density on the same ``auxmol`` (total density,
    spin density, ensemble members).

    :param iterative: ``"coulomb"`` only: solve iteratively (``True``) or by
        Cholesky (``False``); ``None`` picks by the basis size.
    :return: Shape (naux,).
    """
    q = aux_integrals(auxmol)
    if metric == "identity":
        x = q
    elif metric == "overlap":
        x = _dense_solve(auxmol.intor("int1e_ovlp"), q, "auxiliary overlap")
    elif metric == "coulomb_atom":
        x = _atom_blocks_solve(auxmol, q)
    elif metric == "coulomb":
        if iterative is None:
            iterative = auxmol.nao >= ITERATIVE_COULOMB_NAUX
        if iterative:
            try:
                from qpet._coulomb_solver import CoulombSolver
            except ImportError as err:
                raise ImportError(
                    f"the iterative solver for {auxmol.nao} auxiliary functions "
                    "needs numba and threadpoolctl: install `qpet[numba]`, or "
                    "pass iterative=False"
                ) from err

            x = CoulombSolver(auxmol).solve(q)
        else:
            x = _dense_solve(auxmol.intor("int2c2e"), q, "Coulomb metric")
    else:
        raise ValueError(
            f"unknown metric {metric!r}; use 'coulomb', 'overlap', 'identity' "
            "or 'coulomb_atom'"
        )
    return x / (q @ x)


def project(
    auxmol: gto.Mole,
    coeffs: np.ndarray,
    target: float,
    metric: Metric = "coulomb",
) -> np.ndarray:
    """
    Coefficients moved by the smallest shift in ``metric`` to integrate to ``target``.

    :param coeffs: Shape (..., naux); every row is projected to ``target``.
    :return: Same shape as ``coeffs``.
    """
    d = direction(auxmol, metric)
    return coeffs + (target - coeffs @ aux_integrals(auxmol))[..., None] * d
