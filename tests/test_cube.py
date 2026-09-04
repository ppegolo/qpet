"""Tests for CUBE file export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf

from qpet import BOHR_TO_ANG, DMDensity
from qpet.cube import (
    _auto_grid,
    read_cube,
    write_cube,
    write_density_cube,
    write_esp_cube,
)


@pytest.fixture(scope="module")
def water_dm() -> DMDensity:
    """Water molecule with an RHF/sto-3g density matrix."""
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        basis="sto-3g",
        verbose=0,
    ).build()
    dm = scf.RHF(mol).run().make_rdm1()
    return DMDensity(mol, dm)


class TestAutoGrid:
    def test_orthogonal_axes(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        origin, axes, dims = _auto_grid(pos, padding_ang=2.0, spacing_ang=0.5)

        # Axes are diagonal, equal step size.
        np.testing.assert_allclose(axes, np.eye(3) * (0.5 / BOHR_TO_ANG))
        # Spacing in Angstrom of 0.5 over (extent + 2*padding) = (1+4, 4, 4).
        assert dims == (10, 8, 8)

    def test_origin_in_bohr(self):
        pos = np.array([[0.0, 0.0, 0.0]])
        origin, _, _ = _auto_grid(pos, padding_ang=1.0, spacing_ang=0.5)
        np.testing.assert_allclose(origin, np.array([-1.0, -1.0, -1.0]) / BOHR_TO_ANG)

    def test_invalid_inputs(self):
        pos = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            _auto_grid(pos, padding_ang=1.0, spacing_ang=0.0)
        with pytest.raises(ValueError):
            _auto_grid(pos, padding_ang=-1.0, spacing_ang=0.5)


class TestWriteCubeRoundTrip:
    def test_geometry_and_dims(self, tmp_path: Path, water_dm: DMDensity):
        origin, axes, dims = _auto_grid(water_dm.positions, 2.0, 0.5)
        field = np.zeros(dims)
        out = tmp_path / "zero.cube"
        write_cube(out, water_dm, field, origin, axes)

        parsed = read_cube(out)
        assert parsed.dims == dims
        assert len(parsed.atomic_numbers) == water_dm.mol.natm
        np.testing.assert_allclose(parsed.origin_bohr, origin, atol=1e-5)
        np.testing.assert_allclose(parsed.axes_bohr, axes, atol=1e-5)
        np.testing.assert_array_equal(parsed.atomic_numbers, water_dm.atomic_numbers)
        np.testing.assert_allclose(
            parsed.positions_bohr, water_dm.positions / BOHR_TO_ANG, atol=1e-5
        )

    def test_field_values_preserved(self, tmp_path: Path, water_dm: DMDensity):
        origin, axes, dims = _auto_grid(water_dm.positions, 1.0, 1.0)
        rng = np.random.default_rng(0)
        field = rng.standard_normal(dims) * 0.01
        out = tmp_path / "rand.cube"
        write_cube(out, water_dm, field, origin, axes)

        parsed = read_cube(out)
        np.testing.assert_allclose(parsed.field, field, rtol=1e-4, atol=1e-7)


class TestDensityCube:
    def test_density_matches_pyscf_numint(self, water_dm: DMDensity):
        """compute_density must match pyscf.dft.numint.eval_rho."""
        from pyscf.dft import numint

        rng = np.random.default_rng(0)
        pts_ang = rng.uniform(-2.0, 2.0, size=(50, 3))
        pts_bohr = pts_ang / BOHR_TO_ANG
        ao = water_dm.mol.eval_gto("GTOval", pts_bohr)
        rho_ref = numint.eval_rho(water_dm.mol, ao, water_dm.dm)

        rho_ours = water_dm.density(pts_ang)
        np.testing.assert_allclose(rho_ours, rho_ref, rtol=1e-10, atol=1e-12)

    def test_density_positive_near_nucleus(self, tmp_path: Path, water_dm: DMDensity):
        """Density on the cube grid must be positive at points near the O atom."""
        out = tmp_path / "rho.cube"
        write_density_cube(water_dm, out, padding=2.0, spacing=0.4)
        parsed = read_cube(out)

        # Locate cube voxel closest to O nucleus (atom 0).
        pts_ang = parsed.points()
        d2 = np.sum((pts_ang - water_dm.positions[0]) ** 2, axis=1)
        i_near = int(np.argmin(d2))
        rho_flat = parsed.field.reshape(-1)
        assert rho_flat[i_near] > 0.0
        # The peak voxel should hold the largest value on the grid.
        assert rho_flat[i_near] == rho_flat.max()

    def test_density_matches_direct_eval(self, tmp_path: Path, water_dm: DMDensity):
        out = tmp_path / "rho.cube"
        write_density_cube(water_dm, out, padding=2.0, spacing=0.4)
        parsed = read_cube(out)

        pts_ang = parsed.points()
        rho_direct = water_dm.density(pts_ang).reshape(parsed.dims)
        np.testing.assert_allclose(parsed.field, rho_direct, rtol=1e-4, atol=1e-7)


class TestEspCube:
    def test_esp_matches_direct_eval(self, tmp_path: Path, water_dm: DMDensity):
        out = tmp_path / "esp.cube"
        write_esp_cube(water_dm, out, padding=2.0, spacing=0.5)
        parsed = read_cube(out)

        pts_ang = parsed.points()
        v_direct = water_dm.potential(pts_ang).reshape(parsed.dims)
        np.testing.assert_allclose(parsed.field, v_direct, rtol=1e-4, atol=1e-6)


def test_cube_atom_lines_carry_element_and_nuclear_charge(
    tmp_path: Path, hydrogen_iodide_ri
):
    """CUBE atom lines hold ``Z`` and the nuclear charge apart; an ECP splits them."""
    mol, dm, _, _ = hydrogen_iodide_ri
    out = tmp_path / "hi.cube"
    field = np.zeros((2, 2, 2))
    write_cube(out, DMDensity(mol, dm), field, np.zeros(3), 0.5 * np.eye(3))
    lines = out.read_text().splitlines()
    iodine = lines[6].split()
    assert int(iodine[0]) == 53
    assert float(iodine[1]) == 25.0
    assert read_cube(out).atomic_numbers.tolist() == [53, 1]
