"""Tests for surfaces and the ESP output layer."""

from __future__ import annotations

import re

import numpy as np
import pytest
from pyscf import gto, scf

from qpet import (
    DMDensity,
    SurfacePatch,
    esp_error,
    esp_on_surface,
    export_esp_maps,
    read_cube,
    solvent_accessible_surface,
)


@pytest.fixture(scope="module")
def water():
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g", verbose=0
    )
    dm = scf.RHF(mol).run(verbose=0).make_rdm1()
    return DMDensity(mol, dm)


@pytest.fixture
def patch():
    """A full single-atom SAS wrapped as a patch."""
    points, areas = solvent_accessible_surface(
        np.zeros((1, 3)), ["C"], points_per_atom=200
    )
    return SurfacePatch(points, areas, np.zeros(len(points), dtype=int), "test")


class TestSurface:
    def test_isolated_atom_area_matches_sphere(self):
        """One atom's SAS is a full sphere, so its areas must sum to 4*pi*r^2."""
        points, areas = solvent_accessible_surface(
            np.zeros((1, 3)), ["C"], points_per_atom=2000
        )
        radius = 1.70 + 1.4
        assert len(points) == 2000
        assert areas.sum() == pytest.approx(4 * np.pi * radius**2, rel=1e-12)
        assert np.linalg.norm(points, axis=1) == pytest.approx(radius, rel=1e-12)

    def test_areas_differ_between_elements(self):
        """Equal point counts on H and S must not mean equal areas."""
        _, h_areas = solvent_accessible_surface(np.zeros((1, 3)), ["H"])
        _, s_areas = solvent_accessible_surface(np.zeros((1, 3)), ["S"])
        assert len(h_areas) == len(s_areas)
        assert s_areas.sum() > 1.5 * h_areas.sum()

    def test_buried_points_are_culled(self):
        """Two overlapping atoms lose the points between them."""
        points, _ = solvent_accessible_surface(
            np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]), ["C", "C"]
        )
        assert len(points) < 2 * 200
        assert len(points) > 0


class TestErrorDecomposition:
    def test_pythagoras_holds(self, patch):
        """``rmse^2 = offset^2 + shape^2`` is the identity the split relies on."""
        rng = np.random.default_rng(0)
        reference = rng.normal(size=len(patch))
        predicted = reference + 0.01 + 0.003 * rng.normal(size=len(patch))

        stats = esp_error(predicted, reference, patch)
        assert stats["rmse"] ** 2 == pytest.approx(
            stats["offset"] ** 2 + stats["shape_rmse"] ** 2, rel=1e-10
        )

    def test_a_pure_shift_is_all_offset(self, patch):
        """A rigid shift: full RMSE, zero shape error, correlation untouched."""
        rng = np.random.default_rng(1)
        reference = rng.normal(size=len(patch))
        predicted = reference + 0.05

        stats = esp_error(predicted, reference, patch)
        assert stats["offset"] == pytest.approx(0.05, rel=1e-10)
        assert stats["shape_rmse"] == pytest.approx(0.0, abs=1e-12)
        assert stats["rmse"] == pytest.approx(0.05, rel=1e-10)
        assert stats["correlation"] == pytest.approx(1.0, rel=1e-10)

    def test_empty_patch_gives_nan(self):
        empty = SurfacePatch(
            np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int), "test"
        )
        assert np.isnan(esp_error(np.zeros(0), np.zeros(0), empty)["rmse"])

    def test_length_mismatch_is_refused(self, patch):
        message = (
            f"predicted (3,) and reference (3,) must both have one value per "
            f"patch point ({len(patch)})."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            esp_error(np.zeros(3), np.zeros(3), patch)

    def test_esp_on_surface_evaluates_at_the_patch_points(self, patch, water):
        values = esp_on_surface(water, patch)
        assert values.shape == (len(patch),)


class TestExport:
    def test_export_writes_consistent_maps(self, tmp_path, water):
        """Maps share one grid, self-error is zero, scripts name the files."""
        written = export_esp_maps(
            tmp_path, water, name="w", reference=water, spacing=0.8, padding=2.0
        )
        assert set(written) == {"density", "esp", "esp_error", "pymol", "chimerax"}

        cubes = [read_cube(written[k]) for k in ("density", "esp", "esp_error")]
        first = cubes[0]
        for other in cubes[1:]:
            assert other.dims == first.dims
            assert other.origin_bohr == pytest.approx(first.origin_bohr)
            assert other.axes_bohr == pytest.approx(first.axes_bohr)

        assert np.abs(cubes[2].field).max() == 0.0

        for text in (written["pymol"].read_text(), written["chimerax"].read_text()):
            assert "w_density.cube" in text
            assert "w_esp.cube" in text
            assert "w_esp_error.cube" in text

    def test_focus_shrinks_the_box(self, tmp_path, water):
        """``focus`` keeps a binding-site box out of a whole complex."""
        wide = export_esp_maps(
            tmp_path / "wide", water, name="w", spacing=0.8, padding=6.0
        )
        narrow = export_esp_maps(
            tmp_path / "narrow",
            water,
            name="w",
            focus=water.positions[:1],
            spacing=0.8,
            padding=1.0,
        )
        assert (
            read_cube(narrow["density"]).field.size
            < read_cube(wide["density"]).field.size
        )
