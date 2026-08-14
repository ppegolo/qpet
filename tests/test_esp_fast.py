"""The closed-form ESP against libcint, and the dispatch that chooses between them."""

from __future__ import annotations

import re

import numpy as np
import pytest
from pyscf import gto
from pyscf.df.addons import make_auxmol

from qpet.conversions import BOHR_TO_ANG
from qpet.fields import FAST_ESP_MIN_POINTS, RIDensity, nuclear_potential

torch = pytest.importorskip("torch")


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
    """libcint's answer for the electronic potential alone.

    ``fast=False`` is load-bearing: without it the reference would dispatch to
    the very evaluator under test on any machine with a GPU.
    """
    total = RIDensity(auxmol, coefficients).potential(
        points_bohr * BOHR_TO_ANG, fast=False
    )
    nuclear = nuclear_potential(
        auxmol.atom_coords(), auxmol.atom_charges(), points_bohr
    )
    return nuclear - total


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_matches_libcint(system, backend):
    from qpet.esp_fast import FastESP

    if backend == "triton":
        from qpet._esp_kernel import HAS_TRITON

        if not (HAS_TRITON and torch.cuda.is_available()):
            pytest.skip("the fused kernel needs Triton and CUDA")

    auxmol, coefficients = system
    points = np.random.default_rng(1).uniform(-6, 8, (2000, 3)) / BOHR_TO_ANG
    reference = _reference_electronic(auxmol, coefficients, points)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = FastESP(auxmol, device=device, dtype=torch.float32, backend=backend)
    got = evaluator.potential(coefficients, points)

    error = np.abs(got - reference).max() / np.abs(reference).max()
    # float32 throughout; the density it is evaluated from carries ~1e-5.
    assert error < 1e-5, f"{backend}: {error:.2e}"


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_matches_libcint_on_and_next_to_the_nuclei(system, backend):
    """Becke grids put points there, where ``r^-(2l+1)`` and the screening meet."""
    from qpet.esp_fast import FastESP

    if backend == "triton":
        from qpet._esp_kernel import HAS_TRITON

        if not (HAS_TRITON and torch.cuda.is_available()):
            pytest.skip("the fused kernel needs Triton and CUDA")

    auxmol, coefficients = system
    direction = np.array([1.0, -0.7, 0.3]) / np.linalg.norm([1.0, -0.7, 0.3])
    points = np.concatenate(
        [auxmol.atom_coords() + d * direction for d in (0.0, 1e-5, 1e-3, 3e-2)]
    )
    reference = _reference_electronic(auxmol, coefficients, points)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = FastESP(auxmol, device=device, dtype=torch.float32, backend=backend)
    got = evaluator.potential(coefficients, points)
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-5


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the float64 path needs a GPU"
)
def test_float64_reproduces_libcint(system):
    auxmol, coefficients = system
    points = np.random.default_rng(2).uniform(-6, 8, (400, 3)) / BOHR_TO_ANG
    reference = _reference_electronic(auxmol, coefficients, points)

    from qpet.esp_fast import FastESP

    got = FastESP(auxmol, device="cuda", dtype=torch.float64).potential(
        coefficients, points
    )
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-11


def test_unknown_backend_is_rejected():
    from qpet.esp_fast import FastESP

    message = "Unknown backend 'cuda'. Use 'torch', 'triton', or None."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        FastESP(gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0), backend="cuda")


def test_small_grid_stays_on_libcint(system):
    """Too few points to repay the fit, so ``fast=None`` answers with libcint,
    even after a forced closed-form call on the same density."""
    auxmol, coefficients = system
    density = RIDensity(auxmol, coefficients)
    points = np.random.default_rng(4).uniform(-6, 8, (FAST_ESP_MIN_POINTS - 1, 3))

    libcint = density.potential(points, fast=False)
    forced = density.potential(points, fast=True)
    tolerance = 1e-5 if torch.cuda.is_available() else 1e-10
    assert np.abs(forced - libcint).max() / np.abs(libcint).max() < tolerance
    # Both went through libcint, so this is equality and not a tolerance.
    assert np.array_equal(density.potential(points), libcint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="auto dispatch needs a GPU")
def test_auto_dispatch_takes_the_fast_path_on_a_large_grid(system):
    """A small grid must not settle the question for a later large one."""
    auxmol, coefficients = system
    density = RIDensity(auxmol, coefficients)
    rng = np.random.default_rng(5)

    density.potential(rng.uniform(-6, 8, (10, 3)))
    points = rng.uniform(-6, 8, (FAST_ESP_MIN_POINTS + 1, 3))

    auto = density.potential(points)
    slow = density.potential(points, fast=False)
    assert not np.array_equal(auto, slow), "the large grid should not have used libcint"
    assert np.abs(auto - slow).max() / np.abs(slow).max() < 1e-5
