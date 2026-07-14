"""Metatomic ML-density scalar-field calculator.

Thin wrapper over :class:`MetatomicDensityPredictor` that plugs into the
uniform :class:`Calculator` API. Predictions are cached per
:class:`~qpet.structure.Structure` instance so repeated compute calls on the
same structure run the model only once.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np

from .base import Calculator
from .ri import RICalculator

if TYPE_CHECKING:
    import torch

    from qpet.structure import Structure


class MetatomicCalculator(Calculator):
    """
    Scalar-field calculator using a metatomic ML model for density prediction.

    Wraps a :class:`~qpet.predictor.MetatomicDensityPredictor`. On first call
    for a given :class:`~qpet.structure.Structure`, the model predicts RI
    coefficients and an :class:`RICalculator` is cached; subsequent calls on the
    same structure reuse the cached calculator.

    Requires the ``metatomic`` optional dependencies (``pip install
    qpet[metatomic]``).

    :param model: Path to saved metatomic model (``.pt``) or pre-loaded Module
    :param output_name: Metatensor output name (e.g., ``"mtt::aux_dens"``)
    :param basis_name: Auxiliary basis set name (e.g., ``"cc-pvdz-ri"``)
    :param basis_definition: Dict mapping ``(angular_momentum, atomic_number)``
        to the number of radial channels
    :param device: Torch device (default ``"cpu"``)
    :param dtype: Torch dtype (default ``torch.float32``)

    Example::

        from qpet import Structure, MetatomicCalculator

        calc = MetatomicCalculator(
            model="density_model.pt",
            output_name="mtt::aux_dens",
            basis_name="cc-pvdz-ri",
            basis_definition={(0, 1): 3, (1, 1): 2, (0, 6): 5, ...},
        )

        receptor = Structure(...); receptor.calc = calc
        ligand = Structure(...); ligand.calc = calc

    For batched inference (e.g. virtual screening), reach through to
    ``calc.predictor.predict([...])``.
    """

    def __init__(
        self,
        model: str | torch.nn.Module,  # noqa: F821
        output_name: str,
        basis_name: str,
        basis_definition: dict[tuple[int, int], int],
        calc_basis_name: str | None = None,
        device: str = "cpu",
        dtype: torch.dtype | None = None,  # noqa: F821
        *,
        normalize_charge: bool = False,
        normalize_dipole: bool = False,
        dipole_output_name: str = "mtt::electric_dipole",
    ) -> None:
        import torch

        from qpet.predictor import MetatomicDensityPredictor

        if dtype is None:
            dtype = torch.float32

        loaded_model: torch.nn.Module
        if isinstance(model, str):
            from metatomic.torch import load_atomistic_model

            loaded_model = load_atomistic_model(model)
        else:
            loaded_model = model

        self.predictor = MetatomicDensityPredictor(
            loaded_model,
            output_name=output_name,
            basis_name=basis_name,
            basis_definition=basis_definition,
            device=device,
            dtype=dtype,
            calc_basis_name=calc_basis_name,
            normalize_charge=normalize_charge,
            normalize_dipole=normalize_dipole,
            dipole_output_name=dipole_output_name,
        )
        self._cache: weakref.WeakKeyDictionary[Structure, RICalculator] = (
            weakref.WeakKeyDictionary()
        )

    def _get_ri_calculator(self, atoms: Structure) -> RICalculator:
        """Return a cached :class:`RICalculator` for this structure instance."""
        if atoms not in self._cache:
            out = self.predictor.predict_from_pyscf(atoms.mol)
            self._cache[atoms] = RICalculator(out["ri_coeffs"], out["auxmol"])
        return self._cache[atoms]

    def compute_potential(self, atoms: Structure, pts: np.ndarray) -> np.ndarray:
        """
        Compute ESP using ML-predicted density.

        :param atoms: Molecular structure (requires basis on ``.mol``)
        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :return: Electrostatic potential in atomic units, shape (N,)
        """
        return self._get_ri_calculator(atoms).compute_potential(atoms, pts)

    def compute_density(
        self, atoms: Structure, pts: np.ndarray, batch_size: int = 5000
    ) -> np.ndarray:
        """
        Compute electron density from ML-predicted RI coefficients.

        :param atoms: Molecular structure (requires basis on ``.mol``)
        :param pts: Evaluation points in Angstrom, shape (N, 3)
        :param batch_size: Points evaluated per AO batch
        :return: Electron density in electrons / Bohr^3, shape (N,)
        """
        return self._get_ri_calculator(atoms).compute_density(atoms, pts, batch_size)

    def clear_cache(self) -> None:
        """Clear the per-structure RI calculator cache."""
        self._cache.clear()
