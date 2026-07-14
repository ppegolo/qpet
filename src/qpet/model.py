"""Metatomic model wrapper: predict RI densities from geometries.

This module owns every metatomic / metatensor / torch / vesin touchpoint in
qpet — model loading, neighbor lists, forward passes, and the conversion from
TensorMap output to PySCF coefficient ordering. Treat any metatomic import
outside this file as a design error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from pyscf import df, gto

from qpet.conversions import pyscf_to_system, system_to_pyscf
from qpet.fields import RIDensity
from qpet.moments import (
    apply_dipole_correction,
    get_ri_dipole,
    normalize_charge_additive,
)

if TYPE_CHECKING:
    import torch
    from metatensor.torch import TensorMap
    from metatomic.torch import AtomisticModel, System


def _tensormap_to_pyscf_coefficients(
    tensor: TensorMap,
) -> list[np.ndarray]:
    """
    Convert a TensorMap to a list of numpy arrays for PySCF.

    Extracts coefficient values from TensorMap format and converts them to the ordering
    expected by PySCF, handling angular momentum component reordering for p-orbitals.

    :param tensor: TensorMap containing density fitting coefficients with keys indexed
        by angular momentum, inversion parity, and atomic species.
    :return: List of numpy arrays, one per system, containing the coefficient values
        sorted by atom index, angular momentum, radial channel, and magnetic quantum
        number
    """
    import torch

    sys_list, atm_list, ang_list = [], [], []
    rad_list, m_list, val_list = [], [], []

    for key, block in tensor.items():
        angular = int(key[0])

        vals = block.values
        device = vals.device
        S, C, P = vals.shape

        # Extract indices
        sys_idx = block.samples.values[:, 0].to(device, dtype=torch.long)
        atm_idx = block.samples.values[:, 1].to(device, dtype=torch.long)
        rad_idx = block.properties.values[:, 0].to(device, dtype=torch.long)
        m_idx = block.components[0].values[:, 0].to(device, dtype=torch.long)

        # PySCF p-orbital reordering. metatensor stores real-SH components as
        # (m=-1, 0, +1) = (py, pz, px); PySCF expects (px, py, pz). The permutation
        # `[2, 0, 1]` reorders values into PySCF order; m_idx is left untouched and
        # acts as a stable sort key below — sorting by ascending m then yields the
        # already-permuted (px, py, pz) ordering.
        if angular == 1:
            vals = vals[:, [2, 0, 1], :]

        # Reshape index grids for broadcasting
        # sys/atm: (S, 1, 1) -> (S, C, P)
        sys_grid = sys_idx.view(S, 1, 1).expand(S, C, P)
        atm_grid = atm_idx.view(S, 1, 1).expand(S, C, P)
        # m: (1, C, 1) -> (S, C, P)
        m_grid = m_idx.view(1, C, 1).expand(S, C, P)
        # rad: (1, 1, P) -> (S, C, P)
        rad_grid = rad_idx.view(1, 1, P).expand(S, C, P)
        # ang: scalar -> (S, C, P)
        ang_grid = torch.full((S, C, P), angular, dtype=torch.long, device=device)

        mask = ~torch.isnan(vals)

        sys_list.append(sys_grid[mask])
        atm_list.append(atm_grid[mask])
        ang_list.append(ang_grid[mask])
        rad_list.append(rad_grid[mask])
        m_list.append(m_grid[mask])
        val_list.append(vals[mask])

    if not val_list:
        return []

    sys_all = torch.cat(sys_list)
    atm_all = torch.cat(atm_list)
    ang_all = torch.cat(ang_list)
    rad_all = torch.cat(rad_list)
    m_all = torch.cat(m_list)
    val_all = torch.cat(val_list)

    # To sort by (sys, atm, ang, rad, m), we perform sequential stable sorts
    # in reverse order (least significant key first).
    idx = torch.arange(len(val_all), device=device)

    idx = idx[torch.argsort(m_all[idx], stable=True)]
    idx = idx[torch.argsort(rad_all[idx], stable=True)]
    idx = idx[torch.argsort(ang_all[idx], stable=True)]
    idx = idx[torch.argsort(atm_all[idx], stable=True)]
    idx = idx[torch.argsort(sys_all[idx], stable=True)]

    sys_sorted = sys_all[idx]
    val_sorted = val_all[idx]

    # Split the sorted values into a list of tensors (one per system)
    # Find the indices where the system index changes
    changes = (sys_sorted[1:] != sys_sorted[:-1]).nonzero(as_tuple=True)[0] + 1

    # torch.tensor_split requires the boundary indices on CPU
    result_tuple = torch.tensor_split(val_sorted, changes.cpu())

    return [res.detach().cpu().numpy() for res in result_tuple]


class DensityModel:
    """
    Metatomic ML density model predicting :class:`~qpet.fields.RIDensity`.

    When constructed with a ``dict`` of pre-evaluated TensorMaps (one model
    forward pass feeding several consumers), no forward pass is performed —
    the relevant heads are read from the dict.

    :param model: Pre-loaded metatomic model or path to a ``.pt`` file, or a
        ``dict`` of pre-evaluated TensorMaps from a previous forward pass.
    :param head: Name of the density head in the model output
        (e.g. ``"mtt::rho_c_jfit_coulomb"``).
    :param aux_basis: Auxiliary basis set name the model was trained on
        (e.g. ``"def2-universal-jfit"``).
    :param basis: Orbital basis used to build a ``Mole`` in
        :meth:`predict_systems`; unused by :meth:`predict`, which takes the
        caller's ``Mole`` as-is.
    :param device: Torch device (default ``"cpu"``).
    :param dtype: Torch dtype (default ``torch.float32``).
    :param normalize_charge: If True, shift s-channel coefficients so each
        predicted density integrates to the exact electron count.
    :param constrain_dipole: If True, apply a charge-neutral correction so
        the electric dipole matches the model-predicted value.
    :param dipole_head: Name of the dipole output in the model (only used
        when ``constrain_dipole=True``).

    Example::

        from qpet import DensityModel

        model = DensityModel(
            "density_model.pt",
            head="mtt::rho_c_jfit_coulomb",
            aux_basis="def2-universal-jfit",
        )
        rho = model.predict(mol)
        v = rho.potential(points)  # or density(pts), dipole(), scf_guess(mol)
    """

    def __init__(
        self,
        model: str | torch.nn.Module | dict,  # type: ignore[type-arg]  # noqa: F821
        head: str,
        aux_basis: str,
        basis: str | None = None,
        device: str = "cpu",
        dtype: torch.dtype | None = None,  # noqa: F821
        *,
        normalize_charge: bool = False,
        constrain_dipole: bool = False,
        dipole_head: str = "mtt::electric_dipole",
    ) -> None:
        import torch

        if dtype is None:
            dtype = torch.float32

        self._model = model
        self._head = head
        self._device = device
        self._dtype = dtype

        self.aux_basis = aux_basis
        self.basis = basis
        self.normalize_charge = normalize_charge
        self.constrain_dipole = constrain_dipole
        self.dipole_head = dipole_head

        # Model-level setup only needed when model is not pre-evaluated.
        self._loaded_model: AtomisticModel | None
        if isinstance(model, dict):
            self._nl_calculators = []
            self._options = None
            self._loaded_model = None
            return

        from metatomic.torch import ModelEvaluationOptions, ModelOutput

        if isinstance(model, str):
            from metatomic.torch import load_atomistic_model

            self._loaded_model = load_atomistic_model(model)
        else:
            self._loaded_model = cast("AtomisticModel", model)

        self._loaded_model = self._loaded_model.to(device=device, dtype=dtype)

        if constrain_dipole:
            available = set(self._loaded_model.capabilities().outputs.keys())
            if dipole_head not in available:
                raise ValueError(
                    f"constrain_dipole=True but the model does not support "
                    f"'{dipole_head}'; available outputs: {sorted(available)}"
                )

        outputs = {head: ModelOutput(per_atom=True)}
        if constrain_dipole:
            outputs[dipole_head] = ModelOutput(per_atom=False)
        self._options = ModelEvaluationOptions(length_unit="angstrom", outputs=outputs)
        self._neighbor_list_options = self._loaded_model.requested_neighbor_lists()

        import vesin.metatomic

        self._nl_calculators = [
            vesin.metatomic.NeighborList(options=options, length_unit="angstrom")
            for options in self._neighbor_list_options
        ]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, mols: gto.Mole | list[gto.Mole]) -> RIDensity | list[RIDensity]:
        """
        Predict the RI density for PySCF molecule(s).

        Charge and spin are taken from each ``Mole``; the auxiliary basis is
        built on the molecule as given.

        :param mols: A built ``Mole`` or a list of them.
        :return: A single :class:`RIDensity` for a single ``Mole`` input, a
            list for list input.
        """
        single = isinstance(mols, gto.Mole)
        mol_list: list[gto.Mole] = [mols] if single else list(mols)
        systems = [
            pyscf_to_system(mol, dtype=self._dtype, device=self._device)
            for mol in mol_list
        ]
        densities = self._predict(systems, mol_list)
        return densities[0] if single else densities

    def predict_systems(
        self,
        systems: System | list[System],
        *,
        charge: int = 0,
        spin: int | None = None,
    ) -> list[RIDensity]:
        """
        Predict RI densities for metatomic System(s).

        Requires the ``basis`` constructor argument (a ``Mole`` must be built
        to define the auxiliary basis and electron count). ``charge`` and
        ``spin`` apply to every system.

        :param systems: Single System or list of Systems.
        :param charge: Total molecular charge (default 0).
        :param spin: Number of unpaired electrons (2S); ``None`` lets PySCF
            auto-determine the multiplicity.
        :return: List of :class:`RIDensity`, one per system.
        """
        if self.basis is None:
            raise ValueError(
                "predict_systems requires the 'basis' constructor argument "
                "to build PySCF molecules from bare geometries."
            )
        if not isinstance(systems, list):
            systems = [systems]
        mols = [
            system_to_pyscf(system, basis=self.basis, charge=charge, spin=spin)
            for system in systems
        ]
        return self._predict(systems, mols)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _predict(self, systems: list[System], mols: list[gto.Mole]) -> list[RIDensity]:
        """Shared prediction path: forward pass, conversion, corrections."""
        auxmols = [df.addons.make_auxmol(mol, self.aux_basis) for mol in mols]
        outputs = self._evaluate(systems)
        coeffs_list = _tensormap_to_pyscf_coefficients(outputs[self._head])

        if self.normalize_charge:
            coeffs_list = [
                normalize_charge_additive(auxmol, coeffs, mol.nelectron)
                for mol, auxmol, coeffs in zip(mols, auxmols, coeffs_list, strict=True)
            ]

        if self.constrain_dipole:
            dipole_tm = outputs.get(self.dipole_head)
            if dipole_tm is None:
                raise ValueError(
                    f"constrain_dipole=True but model outputs have no key "
                    f"'{self.dipole_head}'; available keys: {sorted(outputs)}"
                )
            targets = self._extract_dipole_targets(dipole_tm, mols)
            coeffs_list = [
                self._constrain_dipole(auxmol, coeffs, mu_target)
                for auxmol, coeffs, mu_target in zip(
                    auxmols, coeffs_list, targets, strict=True
                )
            ]

        densities = []
        for auxmol, coeffs in zip(auxmols, coeffs_list, strict=True):
            if len(coeffs) != auxmol.nao:
                raise ValueError(
                    f"Predicted {len(coeffs)} coefficients but auxiliary basis "
                    f"has {auxmol.nao} functions. Check that aux_basis matches "
                    f"the basis the model was trained on."
                )
            densities.append(RIDensity(auxmol, coeffs))
        return densities

    def _evaluate(self, systems: list[System]) -> dict[str, TensorMap]:
        """Run a single forward pass (or return the pre-evaluated dict)."""
        import torch

        if isinstance(self._model, dict):
            return self._model
        assert self._loaded_model is not None  # set whenever model is not a dict

        for nl in self._nl_calculators:
            nl.add_neighbor_list(systems=systems)

        prepared = [s.to(device=self._device, dtype=self._dtype) for s in systems]
        with torch.no_grad():
            return self._loaded_model(prepared, self._options, False)

    @staticmethod
    def _extract_dipole_targets(
        dipole_tm: TensorMap, mols: list[gto.Mole]
    ) -> list[np.ndarray]:
        """
        Extract per-system electronic dipole targets at the centre of nuclear charge.

        :param dipole_tm: TensorMap with dipole output from the model.
        :param mols: Per-system PySCF Mole objects (orbital basis).
        :return: List of shape-(3,) arrays, one per system, in e·Bohr at R_cnc.
        """
        block = dipole_tm[0]
        values = block.values.detach().cpu().numpy()  # (n_systems, 3, 1)
        dipoles_cart = values[..., 0]  # (n_systems, 3)

        targets = []
        for dipole_total, mol in zip(dipoles_cart, mols, strict=True):
            z = mol.atom_charges().astype(float)
            coords = mol.atom_coords()  # Bohr
            R_cnc = z @ coords / z.sum()  # centre of nuclear charge
            mu_nuc_0 = z @ coords
            mu_el_cnc = dipole_total - mu_nuc_0 + mol.nelectron * R_cnc
            targets.append(mu_el_cnc)
        return targets

    @staticmethod
    def _constrain_dipole(
        auxmol: gto.Mole, coeffs: np.ndarray, mu_el_target: np.ndarray
    ) -> np.ndarray:
        """Charge-neutral correction moving the dipole to ``mu_el_target``."""
        z = auxmol.atom_charges().astype(float)
        R_cnc = z @ auxmol.atom_coords() / z.sum()  # Bohr
        delta_mu = mu_el_target - get_ri_dipole(auxmol, coeffs, R_cnc)
        return apply_dipole_correction(auxmol, coeffs, delta_mu)
