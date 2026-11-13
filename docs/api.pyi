"""Proposed public API of qpet (skeleton for review: signatures and docstrings only).

Conventions
-----------
* Lengths in bohr everywhere: positions, evaluation points, grids. Å only in
  ``build_mole`` (as in PySCF) and in file readers/writers.
* Outputs in atomic units: density in e/bohr^3, potential in hartree/e.
* Orbital-side objects are PySCF ``Mole``; ML-side densities live on an
  auxiliary ``Mole`` (``auxmol``) with coefficients in PySCF AO order.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
from pyscf import gto

Metric = Literal["coulomb", "overlap", "identity", "coulomb_atom"]


# ============================================================================
# qpet.mole
# ============================================================================

def build_mole(
    symbols: Sequence[str],
    positions: np.ndarray,
    *,
    basis: str,
    charge: int = 0,
    spin: int | None = None,
    ecp: str | None = None,
) -> gto.Mole:
    """PySCF ``Mole`` from symbols and positions in Å.

    ``spin`` is 2S; ``None`` takes the lowest value allowed by the electron
    parity (with the ECP core removed).
    """


def make_auxmol(mol: gto.Mole, aux_basis: str) -> gto.Mole:
    """Auxiliary ``Mole`` on the nuclei of ``mol``, ECP-aware.

    ``aux_basis`` is a PySCF basis name or ``"etb:<ao_basis>:<beta>"``.
    """


# ============================================================================
# qpet.normalize
# ============================================================================

def project(
    auxmol: gto.Mole,
    coeffs: np.ndarray,
    target: float,
    metric: Metric = "coulomb",
) -> np.ndarray:
    """Smallest change of ``coeffs`` in ``metric`` that makes the density integrate to ``target``.

    With q the integrals of the auxiliary functions, the shift is
    ``(target - q @ c) / (q @ M^-1 q) * M^-1 q`` with M:

    * ``"coulomb"``: the two-centre Coulomb matrix J (iterative solve on large
      systems);
    * ``"overlap"``: the auxiliary overlap S, i.e. the smallest change of the
      density in L2;
    * ``"identity"``: the unit matrix, the smallest change of the coefficients;
    * ``"coulomb_atom"``: J restricted to the diagonal atom blocks, so each
      atom's shift uses only its own functions (linear cost).
    """


# ============================================================================
# qpet.density
# ============================================================================

class ScalarField(Protocol):
    """Anything with a density and a potential on points: RIDensity, DMDensity."""

    @property
    def positions(self) -> np.ndarray:
        """Nuclear positions in bohr, (n_atoms, 3)."""

    def density(self, points: np.ndarray) -> np.ndarray:
        """Density on points (bohr), (n_points,)."""

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Electrostatic potential on points (bohr), (n_points,).

        Includes the nuclei when the field carries them (total charge density),
        not for a spin density.
        """


class RIDensity:
    """Density expanded in an auxiliary basis: ``rho(r) = sum_k c_k phi_k(r)``.

    ``nuclei`` says whether nuclear charges belong to this density: True for
    the total (charge) density, False for a spin density or a difference.
    """

    auxmol: gto.Mole
    coeffs: np.ndarray
    nuclei: bool

    def __init__(self, auxmol: gto.Mole, coeffs: np.ndarray, *, nuclei: bool = True) -> None: ...

    @classmethod
    def fit(
        cls,
        mol: gto.Mole,
        dm: np.ndarray,
        aux_basis: str,
        *,
        target: float | None = None,
        metric: Metric = "coulomb",
        nuclei: bool = True,
    ) -> "RIDensity":
        """Coulomb-metric RI fit of an AO density matrix (DFT reference).

        ``target`` constrains the integral (N for a total density, 2S for a
        spin density); ``None`` leaves the fit unconstrained.
        """

    @classmethod
    def from_tensormap(cls, tensor, auxmol: gto.Mole, *, nuclei: bool = True) -> "RIDensity":
        """Coefficients stored as a metatensor TensorMap (training targets, raw head output)."""

    def normalized(self, target: float, metric: Metric = "coulomb") -> "RIDensity":
        """Copy projected to integrate to ``target``; see :func:`qpet.normalize.project`."""

    def select_atoms(self, atoms: Sequence[int]) -> "RIDensity":
        """The part of the density on the functions of ``atoms`` (and their nuclei)."""

    def integral(self) -> float:
        """Number of electrons (or 2S for a spin density)."""

    def dipole(self, origin: np.ndarray | None = None) -> np.ndarray:
        """Dipole moment in a.u. about ``origin`` (bohr; default the origin), nuclei included if carried."""

    def quadrupole(self, origin: np.ndarray | None = None) -> np.ndarray: ...

    def density(self, points: np.ndarray) -> np.ndarray: ...

    def potential(self, points: np.ndarray, *, fast: bool | None = None) -> np.ndarray:
        """``fast=None`` uses the closed-form GPU kernel when it pays off."""


class DMDensity:
    """Density of an AO density matrix on an orbital ``Mole`` (the DFT reference)."""

    mol: gto.Mole
    dm: np.ndarray
    nuclei: bool

    def __init__(self, mol: gto.Mole, dm: np.ndarray, *, nuclei: bool = True) -> None: ...

    @classmethod
    def from_chk(cls, path: str | Path, mol: gto.Mole | None = None) -> "tuple[DMDensity, DMDensity | None]":
        """Total and spin density from a PySCF checkpoint (spin is None for RKS)."""

    def integral(self) -> float: ...
    def dipole(self, origin: np.ndarray | None = None) -> np.ndarray: ...
    def density(self, points: np.ndarray) -> np.ndarray: ...
    def potential(self, points: np.ndarray) -> np.ndarray: ...


# ============================================================================
# qpet.model
# ============================================================================

class Ensemble:
    """Members of an LLPR (or shallow) ensemble of one density.

    Members are recentred on the prediction: their mean is the prediction
    itself, and each member is projected with the same metric.
    """

    auxmol: gto.Mole
    coeffs: np.ndarray  # (n_members, n_aux)

    def __len__(self) -> int: ...
    def __getitem__(self, k: int) -> RIDensity: ...

    def potential(self, points: np.ndarray) -> np.ndarray:
        """(n_members, n_points); the spread is the error estimate."""

    def std(self, fn: Callable[[RIDensity], np.ndarray]) -> np.ndarray:
        """Standard deviation over members of any observable, e.g. ``lambda r: r.dipole()``."""


@dataclass(frozen=True)
class Prediction:
    """Output of one forward pass on one molecule.

    ``shift`` is the integral moved by the projection (N for rho, 2S for spin),
    kept so that broken predictions can be filtered afterwards.
    """

    rho: RIDensity
    spin: RIDensity | None
    shift: dict[str, float]
    rho_ensemble: Ensemble | None = None
    spin_ensemble: Ensemble | None = None


def to_scf(prediction: Prediction, mol: gto.Mole, xc: str, **scf_options):
    """PySCF mean-field object that starts from the predicted density.

    RKS for a closed shell, UKS otherwise (from ``mol.spin``). The initial
    guess is one diagonalization of the Fock matrix built from the predicted
    rho and spin density (J from the RI coefficients, XC on the grid), so
    ``to_scf(p, mol, "wb97m-v").kernel()`` runs the SCF from the prediction
    and the object is then used as any PySCF calculation.
    """


class DensityModel:
    """A trained density model, evaluated on PySCF molecules.

    Settings are explicit, as in metatomic: the caller states the auxiliary
    basis the model predicts in and the names of its heads.

    :param model: path to the exported model (``.pt`` for ``backend="metatomic"``,
        ``.petfast``-style artifact for ``backend="fast"``).
    :param aux_basis: auxiliary basis of the coefficients.
    :param heads: output names, ``{"rho": ..., "spin": ...}``; ``"spin"`` is optional.
        Ensemble outputs are named the same way (``"rho_ensemble"``,
        ``"spin_ensemble"``) and fill ``Prediction.*_ensemble`` when given.
    :param backend: ``"metatomic"`` or ``"fast"`` (needs the ``fast`` extra).
    :param normalize: metric of the projection applied to every prediction
        (N on rho, 2S on spin), or ``None`` for the raw output.
    :param broken_symmetry: keep the predicted spin density of singlets;
        by default it is set to zero (the spin head was trained on
        unrestricted open shells only).
    :param guard: warn when the projection moves more than this fraction of
        the electron count (a sign the prediction is broken). The projection
        is applied regardless; the moved amount is in ``Prediction.shift``.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        aux_basis: str,
        heads: Mapping[str, str],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        backend: Literal["metatomic", "fast"] = "metatomic",
        normalize: Metric | None = "coulomb",
        broken_symmetry: bool = False,
        guard: float = 1e-3,
        **backend_options,
    ) -> None: ...

    def predict(self, mol: gto.Mole, *, cell: np.ndarray | None = None) -> Prediction:
        """Density of ``mol``.

        ``cell`` (3x3, bohr) makes the model see periodic neighbours; the
        returned density still lives on the atoms of ``mol`` as given, with no
        images (the KSI whole-cell field).
        """

    def predict_many(self, mols: Sequence[gto.Mole], *, batch_size: int | None = None) -> list[Prediction]:
        """Batched forward passes; same result as ``[predict(m) for m in mols]``."""


# ============================================================================
# qpet.grids
# ============================================================================

class CubeGrid:
    """Regular grid; origin and axes in bohr."""

    origin: np.ndarray
    axes: np.ndarray
    shape: tuple[int, int, int]

    @classmethod
    def around(cls, mol: gto.Mole, *, padding: float = 6.0, spacing: float = 0.25) -> "CubeGrid": ...

    @property
    def points(self) -> np.ndarray:
        """(n_points, 3), bohr."""

    def sample(self, fn: Callable[[np.ndarray], np.ndarray], *, batch_size: int = 200_000) -> np.ndarray:
        """``fn`` on all points, reshaped to ``shape``."""


def becke_grid(mol: gto.Mole, level: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Integration points (bohr) and weights."""


def write_cube(path: str | Path, mol: gto.Mole, grid: CubeGrid, values: np.ndarray, comment: str = "") -> None: ...


def read_cube(path: str | Path) -> tuple[gto.Mole, CubeGrid, np.ndarray]: ...


# ============================================================================
# qpet.backends  (internal; selected through DensityModel(backend=...))
# ============================================================================
#
# qpet.backends.metatomic   metatomic AtomisticModel + vesin neighbour lists
# qpet.backends.fast        vendored petfast: inference, spatial partitioning
#                           for large systems, Triton attention
#
# Both implement:
#
#     def __call__(self, mols: Sequence[gto.Mole]) -> dict[str, list[np.ndarray]]
#         """Raw coefficients per head, PySCF order."""
#
# Export of a metatrain checkpoint for the fast backend (needs metatrain):
#
#     python -m qpet.backends.fast.export model.ckpt model.fast --heads rho=... spin=...


# ============================================================================
# Removed from the public API
# ============================================================================
#
# Hirshfeld partition (FragmentDensity, HirshfeldSplit, free_atom_*),
# interactions (Coulomb/dispersion energies), provenance, conditioning helpers,
# metatensor System conversions, FastESP as a class (now behind potential()),
# constrain_dipole, oracle_charges, pre-evaluated dict input,
# predict_coefficients_torch, with_periodic_images, predict_systems
# (metatensor System input), charge_offset and the three-part charge guard.
