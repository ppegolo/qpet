"""The closed-form ESP against libcint, and its opt-in through ``fast=True``."""

from __future__ import annotations

import re

import numpy as np
import pytest
from pyscf import gto
from pyscf.data.nist import BOHR as BOHR_TO_ANG
from pyscf.df.addons import make_auxmol

from qpet.density import FAST_ESP_MIN_POINTS, Ensemble, RIDensity

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

    ``fast=False`` keeps the reference off the evaluator under test.
    """
    density = RIDensity(auxmol, coefficients, nuclei=False)
    return -density.potential(points_bohr, fast=False)


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


def test_libcint_is_the_default_and_fast_is_opt_in(system, monkeypatch):
    """Above ``FAST_ESP_MIN_POINTS`` the default is still libcint, even on a GPU."""
    auxmol, coefficients = system
    density = RIDensity(auxmol, coefficients)
    points = np.random.default_rng(4).uniform(-6, 8, (FAST_ESP_MIN_POINTS + 1, 3))

    libcint = density.potential(points, fast=False)
    assert np.array_equal(density.potential(points), libcint)
    assert density._fast[0] is None

    fast = density.potential(points, fast=True)
    assert density._fast[0] is not None
    tolerance = 1e-5 if torch.cuda.is_available() else 1e-10
    assert np.abs(fast - libcint).max() / np.abs(libcint).max() < tolerance
    if torch.cuda.is_available():
        assert not np.array_equal(fast, libcint)
    assert np.array_equal(density.potential(points), libcint)


def test_ensemble_builds_the_kernel_once(system, monkeypatch):
    from qpet import esp_fast

    auxmol, coefficients = system
    members = coefficients * np.array([0.9, 1.0, 1.1])[:, None]
    ensemble = Ensemble(auxmol, members)
    points = np.random.default_rng(5).uniform(-6, 8, (300, 3))

    built = []
    original = esp_fast.FastESP.__init__

    def counting(self, *args, **kwargs):
        built.append(1)
        original(self, *args, **kwargs)

    monkeypatch.setattr(esp_fast.FastESP, "__init__", counting)

    libcint = ensemble.potential(points)
    assert not built
    fast = ensemble.potential(points, fast=True)
    tolerance = 1e-5 if torch.cuda.is_available() else 1e-10
    assert np.abs(fast - libcint).max() / np.abs(libcint).max() < tolerance
    assert len(built) == 1

    ensemble.std(lambda d: d.potential(points, fast=True))
    assert len(built) == 1

    fresh = Ensemble(auxmol, members)
    fresh.std(lambda d: d.potential(points, fast=True))
    assert len(built) == 2
