"""Fused Triton kernel for the closed-form ESP.

The array evaluation in :mod:`qpet.esp_fast` writes its point-by-function
intermediates to memory and reads them back; fused, the pair array never exists.
Angular momentum is a compile-time constant, one specialisation per ``l``, so the
monomials and the radial power unroll with no branching.

Triton has no incomplete gamma, so the screening factor is built from ``erf`` by
the recurrence ``P(a+1, x) = P(a, x) - x^a e^-x / Gamma(a+1)`` starting from
``P(1/2, x) = erf(sqrt(x))``. It is evaluated per block, only where it differs
from one.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_TRITON = False


#: Gamma(k + 3/2) for k = 0, 1, ...; the recurrence's denominators.
_GAMMA_HALF = [math.gamma(k + 1.5) for k in range(8)]


if HAS_TRITON:

    @triton.jit
    def _esp_kernel(
        points_ptr,
        centres_ptr,
        poly_ptr,
        exponents_ptr,
        weights_ptr,
        coefficients_ptr,
        out_ptr,
        n_points,
        n_functions,
        cutoff_squared,
        gamma0,
        gamma1,
        gamma2,
        gamma3,
        L: tl.constexpr,
        N_TERMS: tl.constexpr,
        N_PRIM: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_A: tl.constexpr,
    ):
        point_block = tl.program_id(0)
        offs_p = point_block * BLOCK_P + tl.arange(0, BLOCK_P)
        live_p = offs_p < n_points

        px = tl.load(points_ptr + offs_p * 3 + 0, mask=live_p, other=0.0)
        py = tl.load(points_ptr + offs_p * 3 + 1, mask=live_p, other=0.0)
        pz = tl.load(points_ptr + offs_p * 3 + 2, mask=live_p, other=0.0)

        acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

        for start in range(0, n_functions, BLOCK_A):
            offs_a = start + tl.arange(0, BLOCK_A)
            live_a = offs_a < n_functions

            ax = tl.load(centres_ptr + offs_a * 3 + 0, mask=live_a, other=0.0)
            ay = tl.load(centres_ptr + offs_a * 3 + 1, mask=live_a, other=0.0)
            az = tl.load(centres_ptr + offs_a * 3 + 2, mask=live_a, other=0.0)
            coefficient = tl.load(coefficients_ptr + offs_a, mask=live_a, other=0.0)

            dx = px[:, None] - ax[None, :]
            dy = py[:, None] - ay[None, :]
            dz = pz[:, None] - az[None, :]
            r2 = dx * dx + dy * dy + dz * dz
            # A point sitting exactly on a centre is not physical here, and the
            # guard costs nothing next to the rest of the arithmetic.
            r2 = tl.where(r2 > 1e-30, r2, 1e-30)

            # Each monomial is built by unrolled multiplication rather than by
            # indexing a table of powers: the tracer turns Python containers into
            # tuples and will not index them, so nothing is held in a container.
            # `L` is constexpr, so all of these loops vanish at compile time and
            # the repeated factors are common-subexpression eliminated.
            angular = tl.zeros((BLOCK_P, BLOCK_A), dtype=tl.float32)
            term = 0
            for a in range(L, -1, -1):
                for b in range(L - a, -1, -1):
                    c = L - a - b
                    monomial = tl.full((BLOCK_P, BLOCK_A), 1.0, tl.float32)
                    for _ in range(a):
                        monomial = monomial * dx
                    for _ in range(b):
                        monomial = monomial * dy
                    for _ in range(c):
                        monomial = monomial * dz
                    column = tl.load(
                        poly_ptr + offs_a * N_TERMS + term, mask=live_a, other=0.0
                    )
                    angular += column[None, :] * monomial
                    term += 1

            inverse = tl.rsqrt(r2)
            radial = inverse
            for _ in range(2 * L):
                radial = radial * inverse
            value = angular * radial

            # Screening, only where it is not one. The test is per block: if no
            # pair in this tile is close, the whole branch is skipped.
            if tl.max(tl.where(r2 < cutoff_squared, 1.0, 0.0)) > 0.0:
                screening = tl.zeros((BLOCK_P, BLOCK_A), dtype=tl.float32)
                for primitive in range(N_PRIM):
                    alpha = tl.load(
                        exponents_ptr + offs_a * N_PRIM + primitive,
                        mask=live_a,
                        other=0.0,
                    )
                    weight = tl.load(
                        weights_ptr + offs_a * N_PRIM + primitive,
                        mask=live_a,
                        other=0.0,
                    )
                    x = alpha[None, :] * r2
                    # P(1/2, x) = erf(sqrt(x)), then L steps of the recurrence.
                    gamma_p = tl.math.erf(tl.sqrt(x))
                    if L >= 1:
                        decay = tl.exp(-x)
                        xa = tl.sqrt(x)
                        gamma_p -= xa * decay / gamma0
                        if L >= 2:
                            xa = xa * x
                            gamma_p -= xa * decay / gamma1
                        if L >= 3:
                            xa = xa * x
                            gamma_p -= xa * decay / gamma2
                        if L >= 4:
                            xa = xa * x
                            gamma_p -= xa * decay / gamma3
                    screening += weight[None, :] * gamma_p
                value = tl.where(r2 < cutoff_squared, value * screening, value)

            acc += tl.sum(value * coefficient[None, :], axis=1)

        tl.store(out_ptr + offs_p, acc, mask=live_p)


def esp_group(
    points: torch.Tensor,
    centres: torch.Tensor,
    polynomial: torch.Tensor,
    exponents: torch.Tensor,
    weights: torch.Tensor,
    coefficients: torch.Tensor,
    angular_momentum: int,
    cutoff: float,
    block_points: int = 64,
    block_functions: int = 64,
) -> torch.Tensor:
    """Electronic potential from one angular-momentum group.

    :param points: ``(n_points, 3)`` in Bohr, float32, contiguous.
    :param centres: ``(n_functions, 3)`` centre of each function.
    :param polynomial: ``(n_functions, n_terms)`` angular coefficients.
    :param exponents: ``(n_functions, n_primitives)`` primitive exponents.
    :param weights: ``(n_functions, n_primitives)`` radial weights.
    :param coefficients: ``(n_functions,)`` RI coefficients.
    :param angular_momentum: ``l`` of this group.
    :param cutoff: radius beyond which the screening factor is one.
    :return: ``(n_points,)`` potential contribution.
    """
    if not HAS_TRITON:
        raise RuntimeError("the fused ESP kernel needs Triton")
    if angular_momentum > 4:
        raise NotImplementedError(
            f"the fused ESP kernel handles l <= 4, got l = {angular_momentum}"
        )

    n_points = points.shape[0]
    out = torch.empty(n_points, dtype=torch.float32, device=points.device)
    grid = (triton.cdiv(n_points, block_points),)

    _esp_kernel[grid](
        points,
        centres,
        polynomial,
        exponents,
        weights,
        coefficients,
        out,
        n_points,
        centres.shape[0],
        cutoff * cutoff,
        _GAMMA_HALF[0],
        _GAMMA_HALF[1],
        _GAMMA_HALF[2],
        _GAMMA_HALF[3],
        L=angular_momentum,
        N_TERMS=polynomial.shape[1],
        N_PRIM=exponents.shape[1],
        BLOCK_P=block_points,
        BLOCK_A=block_functions,
        num_warps=4,
    )
    return out
