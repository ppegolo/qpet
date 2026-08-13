"""The fast ESP and SAS paths must reproduce the slow ones they replace.

Both are optimisations of things that already worked, so the reference is not a
tolerance pulled from the air -- it is libcint for the potential, and the
straightforward all-pairs cull for the surface. The closed form the ESP rests on is
checked directly too, since it is the one piece that is derived rather than copied.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto
from pyscf.data import radii as pyscf_radii
from pyscf.df.addons import make_auxmol

from qpet.fields import RIDensity, _nuclear_potential
from qpet.surface import _fibonacci_sphere, sas_points

torch = pytest.importorskip("torch")

BOHR = 0.52917721092


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

    directions = _fibonacci_sphere(12)
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
            points = (directions[:, None, :] * radii[None, :, None]).reshape(-1, 3)
            exact = gto.mole.intor_cross(
                "int2c2e", probe, gto.fakemol_for_charges(points)
            )
            far = gto.mole.intor_cross(
                "int2c2e", probe, gto.fakemol_for_charges(directions * 40.0)
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


def _reference_surface(mol, shell=1.4, n=50):
    """The all-pairs cull, as metatrain writes it."""
    unit = _fibonacci_sphere(n)
    centres = mol.atom_coords()
    radii = shell * pyscf_radii.VDW[mol.atom_charges()]
    kept = []
    for atom in range(mol.natm):
        points = centres[atom] + radii[atom] * unit
        buried = np.zeros(n, dtype=bool)
        for other in range(mol.natm):
            if other != atom:
                buried |= np.linalg.norm(points - centres[other], axis=1) < radii[other]
        kept.append(points[~buried])
    return np.concatenate(kept)


def test_surface_matches_all_pairs_cull(system):
    auxmol, _ = system
    fast, weights = sas_points(auxmol)
    reference = _reference_surface(auxmol)

    assert fast.shape == reference.shape
    assert np.abs(np.sort(fast, axis=0) - np.sort(reference, axis=0)).max() < 1e-12
    assert len(weights) == len(fast)
    assert (weights > 0).all()


def test_surface_multiple_shells(system):
    """Several scalings concatenate in ascending order, each culled on its own."""
    auxmol, _ = system
    single, _ = sas_points(auxmol, shell_scale=1.4)
    multi, weights = sas_points(auxmol, shell_scale=[1.4, 2.0])

    assert len(multi) > len(single)
    assert np.allclose(multi[: len(single)], single)
    # The outer shell is further out, so it is culled less and weighs more.
    assert weights[len(single) :].min() > weights[: len(single)].min()
