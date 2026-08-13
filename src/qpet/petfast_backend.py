"""petfast-backed density model: same output as :class:`~qpet.model.DensityModel`.

:class:`~qpet.model.DensityModel` runs the metatomic export of a PET model and
converts its ``TensorMap`` output into PySCF coefficient order. ``petfast`` runs the
same weights through an inference-only reimplementation that skips metatensor
entirely and returns the coefficients already in that order, several times faster
and able to hold far larger systems.

Everything downstream of the coefficients is unchanged: this returns the same
:class:`~qpet.fields.RIDensity` objects, so cubes, potentials, moments and SCF
guesses all work exactly as before.

The two paths are numerically equivalent up to float round-off, which
``tests/test_petfast_backend.py`` checks against the metatomic path directly. If
``petfast`` is not installed, importing this module raises with instructions.

Usage::

    from qpet.petfast_backend import PetFastDensityModel

    model = PetFastDensityModel(
        "model.petfast",                      # from petfast.export.export_checkpoint
        aux_basis="def2-universal-jfit",
        device="cuda",
    )
    rho = model.predict(mol)
    v = rho.potential(points)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pyscf import gto

from qpet.basis import make_auxmol
from qpet.conversions import BOHR_TO_ANG, system_to_pyscf
from qpet.fields import RIDensity
from qpet.moments import normalize_charge_additive

if TYPE_CHECKING:
    import torch
    from metatomic.torch import System


def _require_petfast() -> Any:
    try:
        import petfast
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "PetFastDensityModel needs the petfast package. Install it, then turn a "
            "metatrain checkpoint into an artifact with "
            "`python -c \"from petfast.export import export_checkpoint; "
            "export_checkpoint('model.ckpt', 'model.petfast')\"`."
        ) from error
    return petfast


class PetFastDensityModel:
    """Predict :class:`~qpet.fields.RIDensity` with a petfast artifact.

    A drop-in alternative to :class:`~qpet.model.DensityModel` for the density
    head. Charge and spin are read off each ``Mole``, exactly as there.

    :param artifact: path to a ``.petfast`` artifact, or a ready
        ``petfast.Calculator``.
    :param aux_basis: auxiliary basis the model was trained on, as in
        :class:`~qpet.model.DensityModel`.
    :param basis: orbital basis used to build a ``Mole`` in
        :meth:`predict_systems`; unused by :meth:`predict`.
    :param device: torch device.
    :param dtype: torch dtype; float32 is the sensible default, float64 costs
        roughly twice the memory for round-off-level differences.
    :param normalize_charge: shift the s-channel so each density integrates to
        the exact electron count, as in :class:`~qpet.model.DensityModel`.
    :param ecp: effective core potential for :meth:`predict_systems`.
    :param max_atoms_per_pass: above this many atoms, evaluate with petfast's
        spatial partitioning instead of in one pass. Partitioning is exact but
        processes each block's halo as well as its core, so it is worth using
        only when a system does not fit; ``None`` never partitions.
    :param calculator_options: forwarded to ``petfast.Calculator`` -- e.g.
        ``autocast_dtype`` or ``compile``. Note that float16 measures a few
        percent relative error on density coefficients, which is unlikely to be
        acceptable here.
    """

    def __init__(
        self,
        artifact: str | Any,
        aux_basis: str,
        basis: str | None = None,
        device: str = "cpu",
        dtype: torch.dtype | None = None,  # noqa: F821
        *,
        normalize_charge: bool = False,
        ecp: str | None = None,
        max_atoms_per_pass: int | None = 150_000,
        **calculator_options: Any,
    ) -> None:
        import torch

        petfast = _require_petfast()

        if dtype is None:
            dtype = torch.float32

        if isinstance(artifact, str):
            self.calculator = petfast.Calculator(
                artifact, device=device, dtype=dtype, **calculator_options
            )
        else:
            self.calculator = artifact

        self.aux_basis = aux_basis
        self.basis = basis
        self.ecp = ecp
        self.normalize_charge = normalize_charge
        self.max_atoms_per_pass = max_atoms_per_pass
        self._dtype = dtype
        self._device = device

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, mols: gto.Mole | list[gto.Mole]) -> RIDensity | list[RIDensity]:
        """Predict the RI density for PySCF molecule(s).

        :param mols: a built ``Mole`` or a list of them.
        :return: one :class:`~qpet.fields.RIDensity` per input, unwrapped for a
            single ``Mole``.
        """
        single = isinstance(mols, gto.Mole)
        mol_list: list[gto.Mole] = [mols] if single else list(mols)
        densities = [self._predict_one(mol) for mol in mol_list]
        return densities[0] if single else densities

    def predict_systems(
        self,
        systems: System | list[System],
        *,
        charge: int = 0,
        spin: int | None = None,
    ) -> list[RIDensity]:
        """Predict RI densities for metatomic ``System``\\ (s).

        Provided for interface parity with :class:`~qpet.model.DensityModel`;
        petfast itself needs only positions and atomic numbers.

        :param systems: a single ``System`` or a list.
        :param charge: total charge, applied to every system.
        :param spin: number of unpaired electrons (2S), or ``None`` to let PySCF
            choose the multiplicity.
        """
        if self.basis is None:
            raise ValueError(
                "predict_systems requires the 'basis' constructor argument "
                "to build PySCF molecules from bare geometries."
            )
        if not isinstance(systems, list):
            systems = [systems]
        mols = [
            system_to_pyscf(
                system, basis=self.basis, charge=charge, spin=spin, ecp=self.ecp
            )
            for system in systems
        ]
        return [self._predict_one(mol) for mol in mols]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _predict_one(self, mol: gto.Mole) -> RIDensity:
        """One molecule: geometry in, coefficients out, wrapped as an RIDensity."""
        import torch

        auxmol = make_auxmol(mol, self.aux_basis)

        # Same extraction DensityModel uses, so the two backends see identical
        # inputs: positions in angstrom, and the per-atom charges PySCF reports.
        positions = torch.tensor(
            mol.atom_coords() * BOHR_TO_ANG, dtype=self._dtype, device=self._device
        )
        types = torch.tensor(
            [mol.atom_charge(i) for i in range(mol.natm)],
            dtype=torch.int32,
            device=self._device,
        )
        # PySCF stores spin as the number of unpaired electrons; the model was
        # conditioned on the multiplicity.
        charge = int(mol.charge)
        spin_multiplicity = int(mol.spin) + 1

        partition = (
            self.max_atoms_per_pass is not None
            and positions.shape[0] > self.max_atoms_per_pass
        )
        if partition:
            coefficients, _ = self.calculator.compute_partitioned(
                positions, types, charge=charge, spin_multiplicity=spin_multiplicity
            )
        else:
            coefficients, _ = self.calculator.compute(
                positions, types, charge=charge, spin_multiplicity=spin_multiplicity
            )

        coeffs = coefficients.detach().cpu().to(torch.float64).numpy()

        if len(coeffs) != auxmol.nao:
            raise ValueError(
                f"Predicted {len(coeffs)} coefficients but auxiliary basis has "
                f"{auxmol.nao} functions. Check that aux_basis matches the basis "
                "the model was trained on."
            )

        if self.normalize_charge:
            coeffs = normalize_charge_additive(auxmol, coeffs, mol.nelectron)

        return RIDensity(auxmol, np.asarray(coeffs))
