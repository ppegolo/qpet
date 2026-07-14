"""Tests for CUBE file export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf

from qpet import BOHR_TO_ANG, DMCalculator, Structure
from qpet.cube import (
    _auto_grid,
    _grid_points_ang,
    write_cube,
    write_density_cube,
    write_esp_cube,
)


@pytest.fixture(scope="module")
def water_dm() -> Structure:
    """Water molecule with an RHF/sto-3g density matrix."""
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        basis="sto-3g",
        verbose=0,
    ).build()
    dm = scf.RHF(mol).run().make_rdm1()
    atoms = Structure(mol)
    atoms.calc = DMCalculator(dm)
    return atoms


def _parse_cube(path: Path) -> dict:
    """Minimal CUBE parser for round-trip tests."""
    with open(path) as f:
        lines = f.read().splitlines()

    title1 = lines[0]
    title2 = lines[1]
    parts = lines[2].split()
    natoms = int(parts[0])
    origin = np.array([float(x) for x in parts[1:4]])

    axes = np.zeros((3, 3))
    dims = []
    for i in range(3):
        p = lines[3 + i].split()
        dims.append(int(p[0]))
        axes[i] = [float(x) for x in p[1:4]]

    z = []
    pos = []
    for i in range(natoms):
        p = lines[6 + i].split()
        z.append(int(p[0]))
        pos.append([float(p[2]), float(p[3]), float(p[4])])

    data_tokens: list[float] = []
    for line in lines[6 + natoms :]:
        data_tokens.extend(float(x) for x in line.split())
    data = np.array(data_tokens).reshape(dims)

    return {
        "title1": title1,
        "title2": title2,
        "natoms": natoms,
        "origin_bohr": origin,
        "axes_bohr": axes,
        "dims": tuple(dims),
        "z": np.array(z),
        "positions_bohr": np.array(pos),
        "field": data,
    }


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
    def test_geometry_and_dims(self, tmp_path: Path, water_dm: Structure):
        origin, axes, dims = _auto_grid(water_dm.positions, 2.0, 0.5)
        field = np.zeros(dims)
        out = tmp_path / "zero.cube"
        write_cube(out, water_dm, field, origin, axes)

        parsed = _parse_cube(out)
        assert parsed["dims"] == dims
        assert parsed["natoms"] == water_dm.n_atoms
        np.testing.assert_allclose(parsed["origin_bohr"], origin, atol=1e-5)
        np.testing.assert_allclose(parsed["axes_bohr"], axes, atol=1e-5)
        np.testing.assert_array_equal(parsed["z"], water_dm.atomic_numbers)
        np.testing.assert_allclose(
            parsed["positions_bohr"], water_dm.positions_bohr, atol=1e-5
        )

    def test_field_values_preserved(self, tmp_path: Path, water_dm: Structure):
        origin, axes, dims = _auto_grid(water_dm.positions, 1.0, 1.0)
        rng = np.random.default_rng(0)
        field = rng.standard_normal(dims) * 0.01
        out = tmp_path / "rand.cube"
        write_cube(out, water_dm, field, origin, axes)

        parsed = _parse_cube(out)
        np.testing.assert_allclose(parsed["field"], field, rtol=1e-4, atol=1e-7)


class TestDensityCube:
    def test_density_matches_pyscf_numint(self, water_dm: Structure):
        """compute_density must match pyscf.dft.numint.eval_rho."""
        from pyscf.dft import numint

        rng = np.random.default_rng(0)
        pts_ang = rng.uniform(-2.0, 2.0, size=(50, 3))
        pts_bohr = pts_ang / BOHR_TO_ANG
        ao = water_dm.mol.eval_gto("GTOval", pts_bohr)
        rho_ref = numint.eval_rho(water_dm.mol, ao, water_dm.calc.dm)

        rho_ours = water_dm.compute_density(pts_ang)
        np.testing.assert_allclose(rho_ours, rho_ref, rtol=1e-10, atol=1e-12)

    def test_density_positive_near_nucleus(self, tmp_path: Path, water_dm: Structure):
        """Density on the cube grid must be positive at points near the O atom."""
        out = tmp_path / "rho.cube"
        write_density_cube(water_dm, out, padding=2.0, spacing=0.4)
        parsed = _parse_cube(out)

        # Locate cube voxel closest to O nucleus (atom 0).
        pts_ang = _grid_points_ang(
            parsed["origin_bohr"], parsed["axes_bohr"], parsed["dims"]
        )
        d2 = np.sum((pts_ang - water_dm.positions[0]) ** 2, axis=1)
        i_near = int(np.argmin(d2))
        rho_flat = parsed["field"].reshape(-1)
        assert rho_flat[i_near] > 0.0
        # The peak voxel should hold the largest value on the grid.
        assert rho_flat[i_near] == rho_flat.max()

    def test_density_matches_direct_eval(self, tmp_path: Path, water_dm: Structure):
        out = tmp_path / "rho.cube"
        write_density_cube(water_dm, out, padding=2.0, spacing=0.4)
        parsed = _parse_cube(out)

        pts_ang = _grid_points_ang(
            parsed["origin_bohr"], parsed["axes_bohr"], parsed["dims"]
        )
        rho_direct = water_dm.compute_density(pts_ang).reshape(parsed["dims"])
        np.testing.assert_allclose(parsed["field"], rho_direct, rtol=1e-4, atol=1e-7)


class TestEspCube:
    def test_esp_matches_direct_eval(self, tmp_path: Path, water_dm: Structure):
        out = tmp_path / "esp.cube"
        write_esp_cube(water_dm, out, padding=2.0, spacing=0.5)
        parsed = _parse_cube(out)

        pts_ang = _grid_points_ang(
            parsed["origin_bohr"], parsed["axes_bohr"], parsed["dims"]
        )
        v_direct = water_dm.compute_potential(pts_ang).reshape(parsed["dims"])
        np.testing.assert_allclose(parsed["field"], v_direct, rtol=1e-4, atol=1e-6)
