"""Inference-only PET: the weights of the metatomic model, run several times faster."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from pyscf import gto
from pyscf.data.nist import BOHR


class FastBackend:
    """
    :param model: an artifact written by :func:`qpet.backends.fast.export`.
    :param outputs: as for :class:`~qpet.backends.metatomic.MetatomicBackend`.
    :param max_atoms_per_pass: above this many atoms the system is evaluated
        in spatial blocks (exact, but each block also runs its halo).
    :param options: forwarded to the calculator (``compile``, ``autocast_dtype``).
    """

    def __init__(
        self,
        model: str | Path,
        outputs: Mapping[str, str | None],
        device: str,
        dtype: torch.dtype,
        max_atoms_per_pass: int | None = 150_000,
        **options,
    ) -> None:
        from petfast import Calculator

        self.calculator = Calculator(str(model), device=device, dtype=dtype, **options)
        if set(outputs) != {self.calculator.target_name}:
            raise ValueError(
                f"the artifact holds {self.calculator.target_name!r} only; "
                f"asked for {sorted(outputs)}"
            )
        self.name = self.calculator.target_name
        self.max_atoms_per_pass = max_atoms_per_pass
        self.device, self.dtype = device, dtype

    def __call__(
        self, mols: Sequence[gto.Mole], cell: np.ndarray | None = None
    ) -> dict[str, list[np.ndarray]]:
        return {self.name: [self._one(mol, cell) for mol in mols]}

    def _one(self, mol: gto.Mole, cell: np.ndarray | None) -> np.ndarray:
        positions = torch.tensor(
            mol.atom_coords() * BOHR, dtype=self.dtype, device=self.device
        )
        types = torch.tensor(
            [gto.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)],
            dtype=torch.int32,
            device=self.device,
        )
        periodic = cell is not None
        if periodic:
            cell = torch.tensor(cell * BOHR, dtype=self.dtype, device=self.device)
        partition = (
            self.max_atoms_per_pass is not None and mol.natm > self.max_atoms_per_pass
        )
        if partition and periodic:
            raise NotImplementedError("the partitioned route has no periodic images")
        compute = (
            self.calculator.compute_partitioned
            if partition
            else self.calculator.compute
        )
        coefficients, _ = compute(
            positions,
            types,
            cell=cell,
            pbc=periodic,
            charge=int(mol.charge),
            spin_multiplicity=int(mol.spin) + 1,
        )
        return coefficients.detach().cpu().double().numpy()
