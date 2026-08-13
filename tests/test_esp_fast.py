"""The fast ESP path must reproduce the libcint one it replaces.

The closed form it rests on is checked directly too, since it is the one piece
that is derived rather than copied.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto
from pyscf.df.addons import make_auxmol

from qpet.fields import RIDensity, _nuclear_potential

torch = pytest.importorskip("torch")

BOHR = 0.52917721092

#: A handful of directions spread well enough to fit a degree-4 polynomial.
DIRECTIONS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-0.4, 0.9, 0.2],
        [0.3, -0.8, 0.5],
        [-0.6, -0.3, 0.7],
        [0.2, 0.5, -0.8],
        [-0.7, 0.4, -0.6],
        [0.9, -0.2, -0.3],
        [-0.1, -0.9, -0.4],
        [0.5, 0.7, 0.5],
        [-0.8, 0.1, 0.6],
        [0.6, -0.6, -0.5],
        [-0.3, 0.6, 0.8],
    ]
)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1)[:, None]


@pytest.fixture(scope="module")
def system():
    """A few elements and enough angular momentum to exercise l up to 4."""
    mol = gto.M(
        atom="O 0 0 0; H 0.96 0 0; H -0.24 0.93 0; C 2.5 1.0 0.3; N 3.6 1.8 -0.4",
        basis="def2-svp",
        spin=1,
        verbose=0,
    )
    auxmol = make_auxmol(mol, "def2-universal-jfit")
    coefficients = np.random.default_rng(0).normal(size=auxmol.nao)
    return auxmol, coefficients


def _reference_electronic(auxmol, coefficients, points_bohr):
    """libcint's answer for the electronic potential alone."""
    total = RIDensity(auxmol, coefficients).potential(points_bohr * BOHR)
    nuclear = _nuclear_potential(
        auxmol.atom_coords(), auxmol.atom_charges(), points_bohr
    )
    return nuclear - total


def test_closed_form_matches_pyscf():
    """V = P(R)/|R|^(2l+1) * gammainc(l + 1/2, a|R|^2), at every distance."""
    from scipy.special import gammainc

    for angular in range(4):
        for exponent in (0.3, 2.0, 30.0):
            probe = gto.Mole()
            probe.atom = [["H", (0.0, 0.0, 0.0)]]
            probe.basis = {"H": [[angular, [float(exponent), 1.0]]]}
            probe.spin = 1
            probe.unit = "Bohr"
            probe.verbose = 0
            probe.build()

            radii = np.array([0.3, 0.7, 1.5, 3.0, 6.0])
            points = (DIRECTIONS[:, None, :] * radii[None, :, None]).reshape(-1, 3)
            exact = gto.mole.intor_cross(
                "int2c2e", probe, gto.fakemol_for_charges(points)
            )
            far = gto.mole.intor_cross(
                "int2c2e", probe, gto.fakemol_for_charges(DIRECTIONS * 40.0)
            )
            distance = np.linalg.norm(points, axis=1)

            for component in range(2 * angular + 1):
                angular_part = np.repeat(far[component] * 40.0 ** (angular + 1), len(radii))
                predicted = (
                    angular_part
                    / distance ** (angular + 1)
                    * gammainc(angular + 0.5, exponent * distance**2)
                )
                scale = max(np.abs(exact[component]).max(), 1e-300)
                assert np.abs(predicted - exact[component]).max() / scale < 1e-9


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_matches_libcint(system, backend):
    from qpet.esp_fast import FastESP

    if backend == "triton":
        from qpet.esp_kernel import HAS_TRITON

        if not (HAS_TRITON and torch.cuda.is_available()):
            pytest.skip("the fused kernel needs Triton and CUDA")

    auxmol, coefficients = system
    points = np.random.default_rng(1).uniform(-6, 8, (2000, 3)) / BOHR
    reference = _reference_electronic(auxmol, coefficients, points)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = FastESP(
        auxmol, device=device, dtype=torch.float32, backend=backend
    )
    got = evaluator.potential(coefficients, points)

    error = np.abs(got - reference).max() / np.abs(reference).max()
    # float32 throughout; the density it is evaluated from carries ~1e-5.
    assert error < 1e-5, f"{backend}: {error:.2e}"
    assert evaluator.fit_residual < 1e-8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="float64 path needs no GPU")
def test_float64_reproduces_libcint(system):
    from qpet.esp_fast import FastESP

    auxmol, coefficients = system
    points = np.random.default_rng(2).uniform(-6, 8, (400, 3)) / BOHR
    reference = _reference_electronic(auxmol, coefficients, points)

    got = FastESP(auxmol, device="cuda", dtype=torch.float64).potential(
        coefficients, points
    )
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-11
