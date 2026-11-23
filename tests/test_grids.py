"""Tests for cube grids, cube files and Becke grids."""

from __future__ import annotations

import numpy as np
import pytest

from qpet import (
    CubeGrid,
    RIDensity,
    atomic_becke_grid,
    becke_grid,
    read_cube,
    write_cube,
)


def test_grid_encloses_the_positions_with_padding(water_ri):
    mol = water_ri[0]
    grid = CubeGrid.enclosing(mol.atom_coords(), padding=3.0, spacing=0.5)
    points = grid.points
    assert len(points) == np.prod(grid.shape)
    assert np.all(points.min(axis=0) <= mol.atom_coords().min(axis=0) - 3.0 + 1e-12)
    assert np.all(points.max(axis=0) >= mol.atom_coords().max(axis=0) + 3.0 - 1e-12)


def test_cube_round_trip_keeps_values_order_and_atoms(water_ri, tmp_path):
    mol, _, auxmol, coeffs = water_ri
    grid = CubeGrid.enclosing(mol.atom_coords(), padding=2.0, spacing=0.7)
    values = grid.sample(RIDensity(auxmol, coeffs).density)
    write_cube(tmp_path / "rho.cube", mol, grid, values)
    cube = read_cube(tmp_path / "rho.cube")
    assert cube.grid.shape == grid.shape
    np.testing.assert_allclose(cube.grid.points, grid.points, atol=1e-6)
    np.testing.assert_allclose(cube.values, values, rtol=1e-5, atol=1e-12)
    np.testing.assert_array_equal(cube.numbers, [8, 1, 1])
    np.testing.assert_allclose(cube.positions, mol.atom_coords(), atol=1e-6)


def test_cube_atom_lines_carry_element_and_ecp_charge(hydrogen_iodide_ri, tmp_path):
    mol = hydrogen_iodide_ri[0]
    grid = CubeGrid.enclosing(mol.atom_coords(), padding=1.0, spacing=1.0)
    write_cube(tmp_path / "hi.cube", mol, grid, np.zeros(grid.shape))
    line = (tmp_path / "hi.cube").read_text().splitlines()[6].split()
    assert int(line[0]) == 53 and float(line[1]) == 25.0


def test_becke_grid_integrates_the_density(water_ri):
    mol, _, auxmol, coeffs = water_ri
    points, weights = becke_grid(mol)
    assert weights @ RIDensity(auxmol, coeffs).density(points) == pytest.approx(
        mol.nelectron, abs=0.05
    )


def test_atomic_becke_grid_is_the_becke_grid_by_atom(water_ri):
    mol, _, auxmol, coeffs = water_ri
    rho = RIDensity(auxmol, coeffs)
    points, weights = becke_grid(mol)
    by_atom, atomic_weights, atom = atomic_becke_grid(mol)
    assert np.all(np.diff(atom) >= 0)
    populations = np.bincount(atom, atomic_weights * rho.density(by_atom))
    assert populations.sum() == pytest.approx(weights @ rho.density(points), abs=1e-10)
    assert populations[0] > 7 and populations[1] == pytest.approx(populations[2])
