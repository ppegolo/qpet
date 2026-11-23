"""Electrostatic potential from RI coefficients in closed form, on the GPU.

The potential of a solid-harmonic Gaussian of degree ``l`` and exponent ``alpha``
at displacement ``R`` is

    V(R) = P(R) / |R|^(2l+1) * gammainc(l + 1/2, alpha |R|^2)

with ``P`` a homogeneous polynomial of degree ``l`` and ``gammainc`` the
*regularised* lower incomplete gamma. It is exact at every distance, so there is
no near field to special-case, and it replaces the libcint call
:meth:`qpet.RIDensity.potential` otherwise makes for every (auxiliary
function, grid point) pair.

``P`` and the per-primitive radial weights are recovered by least squares from
PySCF's own integrals rather than written out, and residual-checked at
construction: hardcoding them would mean hardcoding PySCF's real-spherical
ordering and normalisation, where a mistake gives a silently wrong potential
instead of an error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pyscf import gto

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

#: Residual above which a fitted shell is rejected rather than trusted.
FIT_TOLERANCE = 1e-8


def _monomial_powers(degree: int) -> np.ndarray:
    """Exponent triples of every monomial of exactly ``degree``."""
    return np.array(
        [
            (a, b, degree - a - b)
            for a in range(degree, -1, -1)
            for b in range(degree - a, -1, -1)
        ],
        dtype=np.int64,
    )


def _r2_floor(degree: int, dtype: torch.dtype) -> float:
    """Squared distance (Bohr^2) below which ``r^-(2l+1)`` would overflow ``dtype``."""
    import torch

    return max(1e-30, (torch.finfo(dtype).max * 1e-6) ** (-1.0 / (degree + 0.5)))


def _fibonacci_directions(n: int) -> np.ndarray:
    golden = np.pi * (3.0 - np.sqrt(5.0))
    index = np.arange(n)
    z = 1.0 - (2.0 * index + 1.0) / n
    radius = np.sqrt(1.0 - z * z)
    angle = golden * index
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)


def _fit_shell(
    auxmol: gto.Mole, shell: int, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Recover one shell's angular polynomials and radial weights.

    :return: ``(polynomials, exponents, weights, residual)`` -- polynomials of
        shape ``(2l+1, n_terms)`` in the degree-``l`` monomial basis, the shell's
        primitive exponents, radial weights summing to one, and the worst relative
        residual of either fit.
    """
    angular = int(auxmol.bas_angular(shell))
    exponents = np.asarray(auxmol.bas_exp(shell), dtype=float)
    n_components = 2 * angular + 1

    # A lone copy of this shell at the origin: its potential depends on the shell
    # alone, so it can be probed in isolation.
    probe = gto.Mole()
    probe.atom = [["H", (0.0, 0.0, 0.0)]]
    coefficients = np.asarray(auxmol.bas_ctr_coeff(shell)).reshape(len(exponents), -1)
    probe.basis = {
        "H": [
            [
                angular,
                *[
                    [float(e), *[float(c) for c in row]]
                    for e, row in zip(exponents, coefficients, strict=True)
                ],
            ]
        ]
    }
    probe.spin = 1
    probe.unit = "Bohr"
    probe.verbose = 0
    probe.build()

    # -- angular part, from far outside the charge distribution where the gamma
    # factor is one to machine precision.
    far_radius = 12.0 / np.sqrt(exponents.min())
    far_points = directions * far_radius
    far_exact = gto.mole.intor_cross(
        "int2c2e", probe, gto.fakemol_for_charges(far_points)
    )[:n_components]

    powers = _monomial_powers(angular)
    design = np.stack(
        [
            directions[:, 0] ** a * directions[:, 1] ** b * directions[:, 2] ** c
            for a, b, c in powers
        ],
        axis=1,
    )
    polynomials = np.zeros((n_components, len(powers)))
    residual = 0.0
    for component in range(n_components):
        target = far_exact[component] * far_radius ** (angular + 1)
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        polynomials[component] = solution
        scale = max(np.abs(target).max(), 1e-300)
        residual = max(residual, np.abs(design @ solution - target).max() / scale)

    # -- radial part: which primitive contributes how much. The exponents are
    # known, so the profile is linear in the weights.
    radii = np.geomspace(0.05 / np.sqrt(exponents.max()), far_radius, 60)
    direction = directions[0]
    near_points = direction[None, :] * radii[:, None]
    near_exact = gto.mole.intor_cross(
        "int2c2e", probe, gto.fakemol_for_charges(near_points)
    )[:n_components]

    from scipy.special import gammainc

    angular_value = design[0] @ polynomials.T  # (n_components,) along `direction`
    basis = np.stack(
        [gammainc(angular + 0.5, e * radii**2) for e in exponents], axis=1
    )  # (n_radii, n_primitives)
    component = int(np.argmax(np.abs(angular_value)))
    target = near_exact[component] * radii ** (angular + 1) / angular_value[component]
    weights, *_ = np.linalg.lstsq(basis, target, rcond=None)
    predicted = basis @ weights
    residual = max(
        residual, np.abs(predicted - target).max() / max(np.abs(target).max(), 1e-300)
    )

    return polynomials, exponents, weights, residual


class FastESP:
    """Closed-form ESP evaluator for one auxiliary-basis geometry.

    The fit depends only on the auxiliary molecule, so build once and reuse across
    coefficient vectors and grids.

    :param auxmol: the auxiliary ``Mole`` an :class:`~qpet.RIDensity` carries.
    :param device: torch device for the evaluation. ``"cuda"`` is the default and
        falls back to the CPU when no GPU is present; any other value is honoured
        as asked for.
    :param dtype: ``torch.float64`` reproduces libcint to ~1e-12; ``torch.float32``
        is far below the error of a float32 density prediction.
    :param backend: ``"triton"`` for the fused kernel, ``"torch"`` for the array
        implementation, ``None`` to prefer the kernel where it applies.
    """

    def __init__(
        self,
        auxmol: gto.Mole,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        backend: str | None = None,
    ) -> None:
        import torch

        if backend not in (None, "torch", "triton"):
            raise ValueError(
                f"Unknown backend {backend!r}. Use 'torch', 'triton', or None."
            )

        self.auxmol = auxmol
        self.dtype = dtype or torch.float64
        # The default is a request to use a GPU if there is one, so it falls back;
        # anything named explicitly is taken at its word.
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        directions = _fibonacci_directions(96)
        max_angular = max(auxmol.bas_angular(i) for i in range(auxmol.nbas))

        # One monomial table spanning every degree, with each degree's block
        # recorded: functions of different l share the array and must index the
        # same columns.
        blocks, offsets, cursor = [], {}, 0
        for degree in range(max_angular + 1):
            powers = _monomial_powers(degree)
            offsets[degree] = (cursor, cursor + len(powers))
            blocks.append(powers)
            cursor += len(powers)
        all_powers = np.concatenate(blocks)
        n_terms = len(all_powers)

        n_aux = auxmol.nao
        polynomial = np.zeros((n_aux, n_terms))
        angular = np.zeros(n_aux, dtype=np.int64)
        atom = np.zeros(n_aux, dtype=np.int64)
        primitives: list[tuple[np.ndarray, np.ndarray]] = []
        primitive_index = np.zeros(n_aux, dtype=np.int64)

        cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        offset = 0
        worst = 0.0
        for shell in range(auxmol.nbas):
            degree = int(auxmol.bas_angular(shell))
            n_components = 2 * degree + 1
            signature = (
                degree,
                tuple(np.round(np.asarray(auxmol.bas_exp(shell)), 12)),
                tuple(np.round(np.asarray(auxmol.bas_ctr_coeff(shell)).ravel(), 12)),
            )
            if signature not in cache:
                cache[signature] = _fit_shell(auxmol, shell, directions)
            polynomials, exponents, weights, residual = cache[signature]
            worst = max(worst, residual)

            low, high = offsets[degree]
            polynomial[offset : offset + n_components, low:high] = polynomials
            angular[offset : offset + n_components] = degree
            atom[offset : offset + n_components] = auxmol.bas_atom(shell)
            primitive_index[offset : offset + n_components] = len(primitives)
            primitives.append((exponents, weights))
            offset += n_components

        if worst > FIT_TOLERANCE:
            raise RuntimeError(
                f"closed-form ESP fit residual {worst:.2e} exceeds "
                f"{FIT_TOLERANCE:.0e}; the auxiliary basis does not match the "
                "assumed functional form"
            )

        # Pad the ragged primitive lists into one rectangular table.
        max_primitives = max(len(e) for e, _ in primitives)
        exponent_table = np.zeros((len(primitives), max_primitives))
        weight_table = np.zeros((len(primitives), max_primitives))
        for i, (exponents, weights) in enumerate(primitives):
            exponent_table[i, : len(exponents)] = exponents
            weight_table[i, : len(weights)] = weights

        def as_tensor(a: np.ndarray, d: torch.dtype | None = None) -> torch.Tensor:
            return torch.as_tensor(a, dtype=d or self.dtype, device=self.device)

        self._centres = as_tensor(auxmol.atom_coords()[atom])

        # Group by angular momentum. Looping every monomial over every function
        # costs the full pair array once per term even where the coefficient is
        # zero; a degree-l function only ever needs its own (l+1)(l+2)/2 terms, and
        # a basis is dominated by low l.
        self._groups: list[dict[str, Any]] = []
        for degree in range(max_angular + 1):
            members = np.flatnonzero(angular == degree)
            if len(members) == 0:
                continue
            low, high = offsets[degree]
            self._groups.append(
                {
                    "l": degree,
                    "index": as_tensor(members, torch.long),
                    "powers": [
                        tuple(int(x) for x in row) for row in all_powers[low:high]
                    ],
                    "poly": as_tensor(polynomial[members, low:high]),
                    "exponents": as_tensor(exponent_table[primitive_index[members]]),
                    "weights": as_tensor(weight_table[primitive_index[members]]),
                    # Beyond this the gamma factor is one to float32 round-off, so
                    # the screening is skipped and the bare multipole is exact.
                    "cutoff": float(
                        6.5
                        / np.sqrt(
                            np.maximum(exponent_table[primitive_index[members]], 1e-30)
                        ).min()
                    ),
                    # Below it the screened radial factor is at its r -> 0 limit.
                    "r2_floor": _r2_floor(degree, self.dtype),
                }
            )

        from qpet._esp_kernel import HAS_TRITON

        usable = (
            HAS_TRITON
            and self.device.type == "cuda"
            and self.dtype == torch.float32
            and max_angular <= 4
        )
        if backend == "triton" and not usable:
            raise RuntimeError(
                "the fused kernel needs Triton, CUDA, float32 and l <= 4 "
                f"(device={self.device.type}, dtype={self.dtype}, l={max_angular})"
            )
        self.backend = "triton" if (backend != "torch" and usable) else "torch"

        # One compiled callable per group: a shared one re-specialises on each
        # group's l and blows dynamo's recompile limit.
        compile_groups = self.backend == "torch" and self.device.type == "cuda"
        for group in self._groups:
            group["multipole"] = self._make_multipole(group, compile_groups)

    def potential(
        self, coeffs: np.ndarray, points_bohr: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        """Electronic potential ``sum_P c_P (P|r)``.

        :param coeffs: RI coefficients, shape ``(naux,)``.
        :param points_bohr: evaluation points in Bohr, shape ``(n, 3)``.
        :param batch_size: points per batch; the working set is
            ``batch_size * naux``, so this is the memory dial.
        :return: electronic potential in atomic units, shape ``(n,)``.
        """
        import torch

        coefficients = torch.as_tensor(
            np.asarray(coeffs), dtype=self.dtype, device=self.device
        )
        points = torch.as_tensor(
            np.asarray(points_bohr), dtype=self.dtype, device=self.device
        )
        if self.backend == "triton":
            from qpet._esp_kernel import esp_group

            points = points.contiguous()
            total = torch.zeros(len(points), dtype=self.dtype, device=self.device)
            for group in self._groups:
                index = group["index"]
                total += esp_group(
                    points,
                    self._centres[index].contiguous(),
                    group["poly"].contiguous(),
                    group["exponents"].contiguous(),
                    group["weights"].contiguous(),
                    coefficients[index].contiguous(),
                    group["l"],
                    group["cutoff"],
                    group["r2_floor"],
                )
            return total.cpu().numpy()

        out = torch.empty(len(points), dtype=self.dtype, device=self.device)

        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            out[start:stop] = self._batch(coefficients, points[start:stop])
        return out.cpu().numpy()

    def _make_multipole(
        self, group: dict[str, Any], compile_it: bool
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        """Build this group's bare multipole, exact outside the screening cutoff.

        ``l`` and the monomial exponents are captured as Python constants, so the
        loops unroll once per group instead of being guarded on every call.
        """
        import torch

        degree: int = group["l"]
        powers: list[tuple[int, int, int]] = group["powers"]
        floor: float = group["r2_floor"]

        def multipole(
            points: torch.Tensor, centres: torch.Tensor, poly: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            delta = points[:, None, :] - centres[None, :, :]
            squared = (delta * delta).sum(dim=-1).clamp_min(floor)

            # Repeated multiplication: `x ** a` calls pow(), far slower.
            axes = []
            for axis in range(3):
                ladder = [torch.ones_like(squared)]
                for _ in range(degree):
                    ladder.append(ladder[-1] * delta[..., axis])
                axes.append(ladder)

            angular_value = torch.zeros_like(squared)
            for term, (a, b, c) in enumerate(powers):
                angular_value = angular_value + poly[:, term][None, :] * (
                    axes[0][a] * axes[1][b] * axes[2][c]
                )

            # 1 / R^(2l+1) by rsqrt and multiplies, for the same reason.
            inverse = torch.rsqrt(squared)
            radial = inverse
            for _ in range(2 * degree):
                radial = radial * inverse
            return angular_value * radial, squared

        return torch.compile(multipole, dynamic=True) if compile_it else multipole

    def _batch(self, coefficients: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        import torch

        total = torch.zeros(points.shape[0], dtype=self.dtype, device=self.device)
        for group in self._groups:
            total = total + self._batch_group(coefficients, points, group)
        return total

    def _batch_group(
        self, coefficients: torch.Tensor, points: torch.Tensor, group: dict[str, Any]
    ) -> torch.Tensor:
        import torch

        index = group["index"]
        degree = group["l"]
        value, squared = group["multipole"](points, self._centres[index], group["poly"])

        # The gamma factor is one except close in, so it is evaluated only
        # there. Left uncompiled: the test is data-dependent and would break the
        # graph wherever it sat.
        near = squared < group["cutoff"] ** 2
        if bool(near.any()):
            rows, columns = near.nonzero(as_tuple=True)
            near_squared = squared[rows, columns]
            screening = torch.zeros_like(near_squared)
            exponents = group["exponents"][columns]
            weights = group["weights"][columns]
            for primitive in range(exponents.shape[1]):
                screening = screening + weights[:, primitive] * torch.special.gammainc(
                    torch.full_like(near_squared, degree + 0.5),
                    exponents[:, primitive].clamp_min(1e-30) * near_squared,
                )
            value[rows, columns] = value[rows, columns] * screening

        return (value * coefficients[index][None, :]).sum(dim=1)
