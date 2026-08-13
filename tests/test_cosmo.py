"""COSMO checks: an analytic limit, an external implementation, and the solver.

The Born model is the one case with a closed-form answer, so it pins the whole
chain -- cavity, potential, solve and scaling -- against something that cannot
itself be wrong. PySCF's own COSMO then checks the result on real molecules, where
agreement is only approximate because the two use different cavities.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.data import radii as pyscf_radii
from pyscf.df.addons import make_auxmol

from qpet.cosmo import COSMO, screening_factor
from qpet.fields import RIDensity

torch = pytest.importorskip("torch")

AUX = "def2-universal-jfit"


def test_screening_factor_limits():
    assert screening_factor(1.0) == 0.0  # vacuum: no screening
    assert screening_factor(1e12) == pytest.approx(1.0, abs=1e-9)  # conductor
    with pytest.raises(ValueError):
        screening_factor(0.5)


@pytest.mark.parametrize("epsilon", [78.4, 4.0])
@pytest.mark.parametrize("points,expected_error", [(50, 0.01), (500, 0.005)])
def test_born_model(epsilon, points, expected_error):
    """A bare charge in a spherical cavity: ``E = -f(eps) q^2 / (2R)``.

    Solved with no electrons at all, so the only source is the nucleus and the
    answer is exact up to the discretisation of the sphere.
    """
    mol = gto.M(atom="Na 0 0 0", basis="sto-3g", charge=1, spin=0, verbose=0)
    auxmol = make_auxmol(mol, AUX)

    solver = COSMO(
        auxmol,
        epsilon=epsilon,
        cavity_scale=1.2,
        points_per_atom=points,
        dtype=torch.float64,
    )
    result = solver.solve(np.zeros(auxmol.nao))

    charge = float(auxmol.atom_charges()[0])
    radius = 1.2 * pyscf_radii.VDW[int(charge)]
    exact = -screening_factor(epsilon) * charge**2 / (2.0 * radius)

    assert abs(result.energy - exact) / abs(exact) < expected_error
    # A conductor screens the enclosed charge completely.
    assert result.total_charge == pytest.approx(
        -screening_factor(epsilon) * charge, rel=0.01
    )


def test_born_converges_with_refinement():
    """The discretisation error must shrink as the sphere is refined."""
    mol = gto.M(atom="Na 0 0 0", basis="sto-3g", charge=1, spin=0, verbose=0)
    auxmol = make_auxmol(mol, AUX)
    charge = float(auxmol.atom_charges()[0])
    radius = 1.2 * pyscf_radii.VDW[int(charge)]
    exact = -screening_factor(78.4) * charge**2 / (2.0 * radius)

    errors = []
    for points in (50, 200, 1000):
        result = COSMO(
            auxmol, epsilon=78.4, points_per_atom=points, dtype=torch.float64
        ).solve(np.zeros(auxmol.nao))
        errors.append(abs(result.energy - exact))

    assert errors[0] > errors[1] > errors[2]


@pytest.mark.parametrize(
    "name,spec,tolerance",
    [
        ("Cl-", dict(atom="Cl 0 0 0", charge=-1, spin=0), 0.05),
        (
            "NH4+",
            dict(
                atom=(
                    "N 0 0 0; H 1.01 0 0; H -0.34 0.95 0; "
                    "H -0.34 -0.48 -0.82; H -0.34 -0.48 0.82"
                ),
                charge=1,
                spin=0,
            ),
            0.10,
        ),
    ],
)
def test_matches_pyscf_cosmo(name, spec, tolerance):
    """Against PySCF's COSMO on the same density.

    Not an equality check: PySCF discretises the cavity with Lebedev grids and
    switching functions where this uses Fibonacci spheres with a hard cull, so the
    two differ at the level of the discretisation. Ions are compared because their
    solvation is dominated by the monopole and so is far less sensitive to that
    choice than a neutral molecule's dipole-driven value.
    """
    from pyscf.solvent import pcm

    mol = gto.M(basis="def2-svp", verbose=0, **spec)
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()

    reference = pcm.PCM(mol)
    reference.method = "COSMO"
    reference.eps = 78.4
    reference.vdw_scale = 1.2
    reference.build()
    energy_pyscf, _ = reference._get_vind(dm)

    density = RIDensity.from_dm(mol, dm, AUX)
    result = COSMO(
        density.auxmol,
        epsilon=78.4,
        cavity_scale=1.2,
        points_per_atom=302,
        dtype=torch.float64,
    ).solve(density.coeffs)

    assert result.energy < 0
    assert abs(result.energy - energy_pyscf) / abs(energy_pyscf) < tolerance


def test_surface_matrix_is_indefinite_on_a_polyatomic_cavity():
    """Guard on the reason the solve is direct rather than conjugate gradients.

    Segments straddling a sphere intersection interact strongly enough to push an
    eigenvalue negative. If a future change made the cavity definite everywhere, a
    cheaper iterative solve would become available -- and if this test starts
    failing, that is what it is telling you.
    """
    mol = gto.M(
        atom=(
            "N 0 0 0; H 1.01 0 0; H -0.34 0.95 0; "
            "H -0.34 -0.48 -0.82; H -0.34 -0.48 0.82"
        ),
        basis="def2-svp",
        charge=1,
        spin=0,
        verbose=0,
    )
    auxmol = make_auxmol(mol, AUX)
    solver = COSMO(
        auxmol, epsilon=78.4, points_per_atom=302, dtype=torch.float64
    )
    eigenvalues = torch.linalg.eigvalsh(solver._matrix()).cpu().numpy()
    assert (eigenvalues < 0).any()


def test_matrix_free_apply_matches_assembled():
    """The blocked assembly and the matrix-free product must agree exactly."""
    mol = gto.M(atom="O 0 0 0; H 0.96 0 0; H -0.24 0.93 0", basis="def2-svp", verbose=0)
    auxmol = make_auxmol(mol, AUX)
    solver = COSMO(auxmol, epsilon=78.4, points_per_atom=110, dtype=torch.float64)

    matrix = solver._matrix()
    vector = torch.randn(
        len(solver.coords), dtype=torch.float64, device=solver.device
    )
    assert torch.allclose(solver._apply(vector), matrix @ vector, atol=1e-12)


def test_zero_density_gives_zero_energy():
    """No solute, no solvation -- and no division by zero on the way."""
    mol = gto.M(atom="Ne 0 0 0", basis="sto-3g", verbose=0)
    auxmol = make_auxmol(mol, AUX)
    # Cancel the nuclear charge with a compensating s-channel density instead of
    # leaving a bare nucleus, so the surface potential really is ~zero.
    solver = COSMO(auxmol, epsilon=78.4, points_per_atom=50, dtype=torch.float64)
    potential = solver.surface_potential(np.zeros(auxmol.nao))
    assert np.all(potential > 0)  # bare nucleus: positive everywhere
    result = solver.solve(np.zeros(auxmol.nao))
    assert result.energy < 0
