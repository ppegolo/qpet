"""The electrostatic potential on surfaces: evaluation, error split, export.

:func:`esp_on_surface` samples a field's potential on a
:class:`~qpet.surfaces.SurfacePatch`. :func:`esp_error` splits an ESP error
into a rigid offset and the shape that survives it, area-weighted.
:func:`export_esp_maps` writes density, ESP and error maps on one shared
grid — a viewer can only colour one map by another when the grids are
identical — plus runnable PyMOL and ChimeraX scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .surfaces import SurfacePatch

if TYPE_CHECKING:
    from .cube import CubeGrid
    from .fields import ScalarField

logger = logging.getLogger(__name__)

#: Grid size above which sampling a field becomes slow enough to warn about.
_LARGE_GRID = 5_000_000


def esp_on_surface(
    field: ScalarField, patch: SurfacePatch, *, batch_size: int = 2000
) -> np.ndarray:
    """
    Evaluate a field's electrostatic potential on a patch.

    :param field: Any qpet density field (``RIDensity``, ``DMDensity``).
    :param patch: Where to evaluate.
    :param batch_size: Points per batch, controlling peak memory.
    :return: Potential in atomic units (Hartree / e), one value per point.
    """
    if len(patch) == 0:
        return np.zeros(0)
    return np.asarray(field.potential(patch.points, batch_size=batch_size))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values))


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * values**2)))


def esp_error(
    predicted: np.ndarray, reference: np.ndarray, patch: SurfacePatch
) -> dict[str, float]:
    """
    Decompose an ESP error into its rigid offset and the shape that survives it.

    A rigid offset leaves every *difference* on the surface intact, so a
    correlation-based score is nearly immune to it while an absolute readout is
    not; shape deformation is the failure that changes what the surface says.
    ``rmse^2 = offset^2 + shape_rmse^2`` holds exactly, by construction.

    :param predicted: Predicted potential at the patch points, atomic units.
    :param reference: Reference potential at the same points, atomic units.
    :param patch: The patch, supplying the area weights.
    :return: ``rmse``, ``offset``, ``shape_rmse``, ``max_abs``, ``contrast``
        (area-weighted standard deviation of the reference) and ``correlation``
        (area-weighted Pearson, the quantity an EC score depends on). All values
        are ``nan`` for an empty patch.
    :raises ValueError: If the arrays do not match the patch length.
    """
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape or len(predicted) != len(patch):
        raise ValueError(
            f"predicted {predicted.shape} and reference {reference.shape} must "
            f"both have one value per patch point ({len(patch)})."
        )
    if len(patch) == 0:
        return dict.fromkeys(
            ("rmse", "offset", "shape_rmse", "max_abs", "contrast", "correlation"),
            float("nan"),
        )

    w = patch.weights
    difference = predicted - reference
    offset = _weighted_mean(difference, w)

    centred_ref = reference - _weighted_mean(reference, w)
    centred_pred = predicted - _weighted_mean(predicted, w)
    spread = np.sqrt(np.sum(w * centred_ref**2) * np.sum(w * centred_pred**2))
    correlation = (
        float(np.sum(w * centred_ref * centred_pred) / spread)
        if spread > 0
        else float("nan")
    )

    return {
        "rmse": _weighted_rms(difference, w),
        "offset": offset,
        "shape_rmse": _weighted_rms(difference - offset, w),
        "max_abs": float(np.max(np.abs(difference))),
        "contrast": _weighted_rms(centred_ref, w),
        "correlation": correlation,
    }


def _symmetric_limit(
    density: np.ndarray, potential: np.ndarray, isovalue: float
) -> float:
    """A colour limit from the potentials actually seen on the drawn surface.

    Grid points near the isosurface — within a factor of two of ``isovalue`` in
    density — are the ones the viewer will colour. Taking a high percentile of
    ``|V|`` over those, rather than over the whole box, keeps the scale off the
    nuclear cusps and off the empty corners.
    """
    shell = (density > 0.5 * isovalue) & (density < 2.0 * isovalue)
    sample = np.abs(potential[shell]) if shell.any() else np.abs(potential)
    limit = float(np.percentile(sample, 98)) if sample.size else 0.05
    return limit if limit > 0 else 0.05


def _pymol_script(name: str, isovalue: float, limit: float, has_error: bool) -> str:
    lines = [
        f"# {name}: isodensity surface coloured by electrostatic potential.",
        "# Run with:  pymol " + f"{name}.pml",
        "",
        f"load {name}_density.cube, {name}_density",
        f"load {name}_esp.cube, {name}_esp",
        f"isosurface {name}_surface, {name}_density, {isovalue:g}",
        f"ramp_new {name}_ramp, {name}_esp, [{-limit:.4g}, 0, {limit:.4g}], "
        "[red, white, blue]",
        f"set surface_color, {name}_ramp, {name}_surface",
        f"disable {name}_esp",
        "",
        "# Red is negative potential, blue positive; the range is the 98th",
        "# percentile of |V| on the surface itself, in atomic units.",
    ]
    if has_error:
        lines += [
            "",
            f"load {name}_esp_error.cube, {name}_esp_error",
            f"ramp_new {name}_error_ramp, {name}_esp_error, "
            f"[{-limit:.4g}, 0, {limit:.4g}], [red, white, blue]",
            "# To colour by the error instead of the potential:",
            f"#   set surface_color, {name}_error_ramp, {name}_surface",
            f"disable {name}_esp_error",
        ]
    return "\n".join(lines) + "\n"


def _chimerax_script(name: str, isovalue: float, limit: float, has_error: bool) -> str:
    lines = [
        f"# {name}: isodensity surface coloured by electrostatic potential.",
        f"# Run with:  chimerax {name}.cxc",
        "",
        f"open {name}_density.cube",
        f"open {name}_esp.cube",
        f"volume #1 style surface level {isovalue:g}",
        "hide #2 models",
        f"color sample #1 map #2 palette red:white:blue range {-limit:.4g},{limit:.4g}",
        "",
        "# Red is negative potential, blue positive; the range is the 98th",
        "# percentile of |V| on the surface itself, in atomic units.",
    ]
    if has_error:
        lines += [
            "",
            f"open {name}_esp_error.cube",
            "hide #3 models",
            "# To colour by the error instead of the potential:",
            f"#   color sample #1 map #3 palette red:white:blue "
            f"range {-limit:.4g},{limit:.4g}",
        ]
    return "\n".join(lines) + "\n"


def export_esp_maps(
    directory: str | Path,
    density: ScalarField,
    *,
    name: str = "esp",
    esp_source: ScalarField | None = None,
    reference: ScalarField | None = None,
    focus: np.ndarray | None = None,
    grid: CubeGrid | None = None,
    padding: float = 4.0,
    spacing: float = 0.3,
    isovalue: float = 0.002,
    limit: float | None = None,
    batch_size: int = 5000,
) -> dict[str, Path]:
    """
    Write an isodensity surface and its potential for PyMOL and ChimeraX.

    Every map goes on **one** grid. Neither viewer can colour a surface by a
    second map otherwise, and two separately auto-sized grids will not match
    whenever the fields have different atoms — which is precisely the
    interesting case, a ligand isosurface coloured by what the receptor does to
    it.

    :param directory: Output directory; created if absent.
    :param density: Field whose isosurface is drawn (e.g. the ligand's
        interaction density).
    :param name: Stem for every written file.
    :param esp_source: Field supplying the colouring potential. Defaults to
        ``density``; pass the *partner's* field to show the potential the
        surface sits in.
    :param reference: If given, also write a signed ESP error map,
        ``esp_source − reference``, on the same grid.
    :param focus: Positions the grid must enclose, in Angstrom. Defaults to
        ``density``'s atoms — pass the ligand's positions to keep the box small
        inside a large complex.
    :param grid: An explicit grid, overriding ``focus``, ``padding`` and
        ``spacing``.
    :param padding: Padding around ``focus`` in Angstrom.
    :param spacing: Grid spacing in Angstrom. The default is coarser than the
        cube writers', because these boxes enclose binding sites rather than
        molecules.
    :param isovalue: Density isovalue for the surface, in e/Bohr^3.
    :param limit: Symmetric colour limit in atomic units. Derived from the
        potential on the surface itself when omitted.
    :param batch_size: Points evaluated per batch.
    :return: Mapping of label to written path.
    """
    from .cube import CubeGrid, write_cube

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    esp_source = density if esp_source is None else esp_source

    if grid is None:
        positions = density.positions if focus is None else np.asarray(focus)
        grid = CubeGrid.enclosing(positions, padding=padding, spacing=spacing)
    if grid.n_points > _LARGE_GRID:
        logger.warning(
            "Grid has %d points; sampling the potential there is slow. "
            "Consider a focus region or a coarser spacing.",
            grid.n_points,
        )
    logger.info("Exporting %s on %r", name, grid)

    rho = grid.sample(lambda p: density.density(p, batch_size=batch_size))
    potential = grid.sample(lambda p: esp_source.potential(p, batch_size=batch_size))
    if limit is None:
        limit = _symmetric_limit(rho, potential, isovalue)

    written: dict[str, Path] = {}
    maps = [("density", rho, "electron density", "rho(r) in electrons / Bohr^3")]
    maps.append(("esp", potential, "electrostatic potential", "V(r) in atomic units"))
    if reference is not None:
        error = potential - grid.sample(
            lambda p: reference.potential(p, batch_size=batch_size)
        )
        maps.append(
            ("esp_error", error, "ESP error (predicted - reference)", "dV in a.u.")
        )

    for label, field, title1, title2 in maps:
        path = directory / f"{name}_{label}.cube"
        write_cube(
            path,
            density,
            field,
            grid.origin_bohr,
            grid.axes_bohr,
            title1=f"qpet {title1}",
            title2=title2,
        )
        written[label] = path

    has_error = reference is not None
    for label, suffix, text in (
        ("pymol", ".pml", _pymol_script(name, isovalue, limit, has_error)),
        ("chimerax", ".cxc", _chimerax_script(name, isovalue, limit, has_error)),
    ):
        path = directory / f"{name}{suffix}"
        path.write_text(text)
        written[label] = path

    return written
