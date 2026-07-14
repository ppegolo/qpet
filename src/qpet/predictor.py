"""Metatomic density predictor.

Wraps a pre-trained metatomic model to produce density fitting (RI)
coefficients on a PySCF auxiliary basis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from pyscf import df, gto

from .conversions import pyscf_to_system, system_to_atom_string
from .observables import apply_dipole_correction
from .ri_normalization import (
    compute_n_electrons,
    s_normalization_to_tensormap,
)

if TYPE_CHECKING:
    from metatensor.torch import TensorMap
    from metatomic.torch import System


class MetatomicDensityOutput(TypedDict):
    """Output from metatomic density prediction."""

    auxmol: gto.Mole
    ri_coeffs: np.ndarray


def _check_metatomic_dependencies() -> None:
    """Check that metatomic dependencies are available."""
    missing = []
    try:
        import metatensor.torch  # noqa: F401
    except ImportError:
        missing.append("metatensor-torch")
    try:
        import metatomic.torch  # noqa: F401
    except ImportError:
        missing.append("metatomic-torch")
    try:
        import metatrain  # noqa: F401
    except ImportError:
        missing.append("metatrain")

    if missing:
        raise ImportError(
            f"Metatomic integration requires: {', '.join(missing)}. "
            "Install with: pip install qpet[metatomic]"
        )


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

        # PySCF p-orbital reordering
        if angular == 1:
            vals = vals[:, [2, 0, 1], :]

        # 4. Reshape for broadcasting
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

    # torch.tensor_split takes boundary indices and splits the tensor
    result_tuple = torch.tensor_split(
        val_sorted, changes.cpu() if changes.is_cuda else changes
    )

    return [res.detach().cpu().numpy() for res in result_tuple]


class MetatomicDensityPredictor:
    """
    Predicts electron density RI coefficients using a metatomic model.

    Wraps a metatomic model to predict density fitting coefficients in a basis set
    representation, converting outputs to PySCF-compatible format for use with a
    :class:`~qpet.calculators.RICalculator` or the ready-made
    :class:`~qpet.calculators.MetatomicCalculator` wrapper.

    :param model: Pre-trained metatomic model for predicting density coefficients
    :param output_name: Name of the metatensor output to extract from the model
    :param basis_name: Name of the auxiliary basis set (e.g., "def2-universal-jfit")
    :param basis_definition: Dictionary mapping ``(angular_momentum, atomic_number)``
        tuples to the number of radial channels for that combination
    :param device: Device to run computations on
    :param dtype: Data type for tensor computations

    Example::

        from qpet import MetatomicDensityPredictor
        from metatomic.torch import load_atomistic_model

        model = load_atomistic_model("density_model.pt")

        basis_def = {
            (0, 1): 3,
            (1, 1): 2,
            (0, 6): 5,
            (1, 6): 4,
            (2, 6): 3,
        }

        predictor = MetatomicDensityPredictor(
            model=model,
            output_name="mtt::aux_dens",
            basis_name="def2-universal-jfit",
            basis_definition=basis_def,
        )

        result = predictor.predict_from_pyscf(mol)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        output_name: str,
        basis_name: str,
        basis_definition: dict[tuple[int, int], int],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        calc_basis_name: str | None = None,
        *,
        normalize_charge: bool = False,
        normalize_dipole: bool = False,
        dipole_output_name: str = "mtt::electric_dipole",
    ) -> None:
        _check_metatomic_dependencies()

        from metatomic.torch import ModelEvaluationOptions, ModelOutput

        self.device = device
        self.dtype = dtype

        self.model = model.to(device=device, dtype=dtype)
        self.output_name = output_name

        if normalize_dipole:
            available = set(model._capabilities.outputs.keys())
            if dipole_output_name not in available:
                raise ValueError(
                    f"normalize_dipole=True but the model does not support "
                    f"'{dipole_output_name}'; available outputs: {sorted(available)}"
                )

        outputs: dict[str, ModelOutput] = {output_name: ModelOutput(per_atom=True)}
        if normalize_dipole:
            outputs[dipole_output_name] = ModelOutput(per_atom=False)
        self.options = ModelEvaluationOptions(length_unit="angstrom", outputs=outputs)
        self.neighbor_list_options = model.module.requested_neighbor_lists()

        self.basis_name = basis_name
        self.basis_definition = basis_definition

        self.calc_basis_name = (
            calc_basis_name if calc_basis_name is not None else basis_name
        )

        self.normalize_charge = normalize_charge
        self.normalize_dipole = normalize_dipole
        self.dipole_output_name = dipole_output_name

    def _build_auxiliary_mols(
        self, systems: list[System]
    ) -> tuple[list[gto.Mole], list[gto.Mole]]:
        """
        Build PySCF auxiliary molecules for density fitting.

        :param systems: List of atomistic system objects
        :return: Tuple of (orbital mols, auxiliary mols)
        """
        mols = []
        auxmols = []
        for system in systems:
            # spin=None lets PySCF auto-determine parity; the auxiliary mol is only
            # used for integrals (not SCF), so spin is irrelevant here.
            mol = gto.Mole()
            mol.atom = system_to_atom_string(system)
            mol.basis = self.basis_name
            mol.verbose = 0
            mol.spin = None
            mol.build()
            mols.append(mol)
            auxmol = df.addons.make_auxmol(mol, self.basis_name)
            auxmols.append(auxmol)
        return mols, auxmols

    def _validate_coefficients(
        self,
        auxmol: gto.Mole,
        coeffs: np.ndarray,
    ) -> None:
        """
        Validate that predicted coefficients match auxiliary basis size.

        :param auxmol: PySCF auxiliary molecule
        :param coeffs: Predicted coefficient array
        :raises ValueError: If coefficient count doesn't match basis size
        """
        expected_naux = auxmol.nao
        if len(coeffs) != expected_naux:
            raise ValueError(
                f"Predicted {len(coeffs)} coefficients but auxiliary basis has "
                f"{expected_naux} functions. Check basis_definition parameter."
            )

    def predict(self, systems: System | list[System]) -> list[MetatomicDensityOutput]:
        """
        Predict RI coefficients for metatomic System(s).

        :param systems: Single System or list of Systems
        :return: List of :class:`MetatomicDensityOutput` dicts with auxmol and ri_coeffs
        """
        from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

        if not isinstance(systems, list):
            systems = [systems]

        mols, auxmols = self._build_auxiliary_mols(systems)

        prepared_systems = [
            get_system_with_neighbor_lists(system, self.neighbor_list_options).to(
                device=self.device, dtype=self.dtype
            )
            for system in systems
        ]

        with torch.no_grad():
            result = self.model(prepared_systems, self.options, False)
            tensor = result[self.output_name]

            if self.normalize_charge:
                n_electrons, s_orbital_normalization = self._get_ri_norm_factors(mols)
                self._apply_charge_normalization(
                    tensor, n_electrons, s_orbital_normalization
                )
        coeffs_list = _tensormap_to_pyscf_coefficients(tensor)

        if self.normalize_dipole:
            dipole_tm = result[self.dipole_output_name]
            dipole_targets = self._extract_dipole_targets(dipole_tm, mols)
            coeffs_list = self._apply_dipole_normalization(
                coeffs_list, auxmols, dipole_targets
            )

        results: list[MetatomicDensityOutput] = []
        for auxmol, coeffs in zip(auxmols, coeffs_list, strict=True):
            self._validate_coefficients(auxmol, coeffs)
            results.append({"auxmol": auxmol, "ri_coeffs": coeffs})

        return results

    def predict_from_pyscf(self, mol: gto.Mole) -> MetatomicDensityOutput:
        """
        Predict RI coefficients for a PySCF Mole object.

        :param mol: PySCF Mole object
        :return: :class:`MetatomicDensityOutput` with auxmol and ri_coeffs
        """
        system = pyscf_to_system(mol, dtype=self.dtype, device=self.device)
        results = self.predict(system)
        return results[0]

    def _get_ri_norm_factors(
        self, mols: list[gto.Mole]
    ) -> tuple[torch.Tensor, TensorMap]:
        """
        Compute S-orbital normalization factors for RI charge normalization.

        :param mols: List of PySCF Mole objects for the original systems
        :return: Tuple of (n_electrons tensor, s_orbital_normalization TensorMap)
        """
        import metatensor.torch as mts

        n_electrons = torch.tensor(
            [compute_n_electrons(mol) for mol in mols],
            device=self.device,
            dtype=self.dtype,
        )

        s_orbital_normalization = mts.join(
            [
                s_normalization_to_tensormap(
                    mol,
                    self.basis_name,
                    system_idx=i,
                )
                for i, mol in enumerate(mols)
            ],
            axis="samples",
            different_keys="union",
        )

        return n_electrons, s_orbital_normalization

    def _apply_charge_normalization(
        self,
        tensor: TensorMap,
        n_electrons: torch.Tensor,
        s_orbital_normalization: TensorMap,
    ) -> None:
        """
        Apply RI charge normalization to the predicted coefficients in-place.

        :param tensor: TensorMap containing the predicted coefficients to be
            normalized.
        :param n_electrons: TensorMap containing the total number of electrons for each
            system.
        :param s_orbital_normalization: TensorMap containing the S-orbital normalization
            factors for each atom.
        """
        n_systems = n_electrons.shape[0]
        device = tensor[0].values.device
        dtype = tensor[0].values.dtype

        q_mol_total = torch.zeros(n_systems, device=device, dtype=dtype)
        s_norm_sq_mol_total = torch.zeros(n_systems, device=device, dtype=dtype)

        for k, b in tensor.items():
            if k["o3_lambda"] != 0:
                continue
            c = b.values.squeeze(1)  # [samples, num_radial]
            S_vec = (
                s_orbital_normalization.block(k)
                .values.squeeze(1)
                .to(dtype=dtype, device=device)
            )  # [samples, num_radial]
            system_idx = b.samples.values[:, 0].to(torch.int64)

            q_atom = torch.sum(c * S_vec, dim=-1)  # [samples]
            q_mol_total.scatter_add_(0, system_idx, q_atom)

            s_norm_atom = torch.sum(S_vec * S_vec, dim=-1)  # [samples]
            s_norm_sq_mol_total.scatter_add_(0, system_idx, s_norm_atom)

        delta_N = n_electrons - q_mol_total
        gamma_mol = delta_N / (s_norm_sq_mol_total + 1e-8)  # [B]

        for k, b in tensor.items():
            if k["o3_lambda"] != 0:
                continue
            c = b.values.squeeze(1)  # [samples, num_radial]
            S_vec = s_orbital_normalization.block(k).values.to(
                dtype=dtype, device=device
            )
            system_idx = b.samples.values[:, 0].to(torch.int64)

            gamma_atom = (
                gamma_mol[system_idx].unsqueeze(-1).unsqueeze(-1)
            )  # [samples, 1]
            b.values[:] += gamma_atom * S_vec

    def _extract_dipole_targets(
        self, dipole_tm: TensorMap, mols: list[gto.Mole]
    ) -> list[np.ndarray]:
        """
        Extract per-system electronic dipole targets at the centre of nuclear charge.

        The ML model predicts the total molecular dipole (nuclear + electronic) at
        origin 0. We convert to the electronic dipole at the centre of nuclear charge
        (R_cnc), which is origin-independent for neutral systems and well-defined for
        charged ones:

            μ_el(R_cnc) = μ_total(0) − μ_nuc(0) + N_elec · R_cnc

        :param dipole_tm: TensorMap with key ``mtt::electric_dipole`` from the model.
        :param mols: Per-system PySCF Mole objects (orbital basis).
        :return: List of shape-(3,) arrays, one per system, in e·Bohr at R_cnc.
        """
        block = dipole_tm[0]
        values = block.values.detach().cpu().numpy()  # (n_systems, 3, 1)
        dipoles_cart = values[..., 0]  # (n_systems, 3), [x, y, z] in e·Bohr

        targets = []
        for dipole_total, mol in zip(dipoles_cart, mols, strict=True):
            z = mol.atom_charges().astype(float)
            coords = mol.atom_coords()  # Bohr
            R_cnc = z @ coords / z.sum()  # centre of nuclear charge, Bohr
            mu_nuc_0 = z @ coords  # Σ Z_A R_A, nuclear dipole at origin 0
            mu_el_cnc = dipole_total - mu_nuc_0 + mol.nelectron * R_cnc
            targets.append(mu_el_cnc)
        return targets

    def _apply_dipole_normalization(
        self,
        coeffs_list: list[np.ndarray],
        auxmols: list[gto.Mole],
        dipole_targets: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Apply charge-neutral dipole correction to RI coefficients.

        For each system computes the current electronic dipole from the RI density,
        then applies :func:`~qpet.observables.apply_dipole_correction` to bring
        the dipole to the ML-predicted target without changing the total charge.

        :param coeffs_list: Per-system RI coefficient arrays.
        :param auxmols: Per-system PySCF auxiliary Mole objects.
        :param dipole_targets: Per-system target electronic dipole in e·Bohr.
        :return: Corrected per-system RI coefficient arrays.
        """
        from .observables import get_ri_dipole

        corrected = []
        for coeffs, auxmol, mu_el_target in zip(
            coeffs_list, auxmols, dipole_targets, strict=True
        ):
            z = auxmol.atom_charges().astype(float)
            R_cnc = z @ auxmol.atom_coords() / z.sum()  # Bohr
            mu_ri = get_ri_dipole(auxmol, coeffs, R_cnc)
            delta_mu = mu_el_target - mu_ri
            corrected.append(apply_dipole_correction(auxmol, coeffs, delta_mu))
        return corrected

    # def _apply_charge_correction(
    #     self,
    #     systems: list[System],
    #     tensor: TensorMap,
    #     n_electrons: torch.Tensor,
    #     s_orbital_normalization: TensorMap,
    # ) -> None:

    #     n_systems = len(systems)
    #     device = tensor[0].values.device
    #     dtype = tensor[0].values.dtype

    #     q_mol_total = torch.zeros(n_systems, device=device, dtype=dtype)
    #     s_norm_sq_mol_total = torch.zeros(n_systems, device=device, dtype=dtype)

    #     for k, b in tensor.items():
    #         if k["o3_lambda"] != 0:
    #             continue
    #         c = b.values.squeeze(1)  # [samples, num_radial]
    #         S_vec = (
    #             s_orbital_normalization.block(k)
    #             .values.squeeze(1)
    #             .to(dtype=dtype, device=device)
    #         )  # [samples, num_radial]
    #         system_idx = b.samples.values[:, 0].to(torch.int64)

    #         q_atom = torch.sum(c * S_vec, dim=-1)  # [samples]
    #         q_mol_total.scatter_add_(0, system_idx, q_atom)

    #         s_norm_atom = torch.sum(S_vec * S_vec, dim=-1)  # [samples]
    #         s_norm_sq_mol_total.scatter_add_(0, system_idx, s_norm_atom)

    #     delta_N = n_electrons - q_mol_total

    #     # build batch constraint matrix
    #     M = torch.zeros((n_systems, 4, 4), device=device, dtype=dtype)
    #     M[:, 0, 0] = [len(system) for system in systems]
    #     pos_sum =
