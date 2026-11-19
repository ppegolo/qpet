"""Inference through metatomic: an exported model and vesin neighbour lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyscf import gto
from pyscf.data.nist import BOHR


def _scalar(value: float, name: str, like: torch.Tensor) -> Any:
    from metatensor.torch import Labels, TensorBlock, TensorMap

    zero = torch.zeros((1, 1), dtype=torch.int32, device=like.device)
    block = TensorBlock(
        values=torch.tensor([[float(value)]], dtype=like.dtype, device=like.device),
        samples=Labels("system", zero),
        components=[],
        properties=Labels(name, zero),
    )
    return TensorMap(Labels.single().to(device=like.device), [block])


def system(
    mol: gto.Mole, cell: np.ndarray | None, dtype: torch.dtype, device: str
) -> Any:
    """metatomic ``System`` of ``mol`` (Å), with its charge and multiplicity.

    The types are the atomic numbers, not the ECP-reduced nuclear charges.
    """
    from metatomic.torch import System

    positions = torch.tensor(mol.atom_coords() * BOHR, dtype=dtype, device=device)
    periodic = cell is not None
    out = System(
        types=torch.tensor(
            [gto.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)],
            dtype=torch.int32,
            device=device,
        ),
        positions=positions,
        cell=torch.tensor(
            cell * BOHR if periodic else np.zeros((3, 3)), dtype=dtype, device=device
        ),
        pbc=torch.tensor([periodic] * 3, device=device),
    )
    out.add_data("charge", _scalar(mol.charge, "charge", positions))
    out.add_data(
        "spin_multiplicity", _scalar(mol.spin + 1, "spin_multiplicity", positions)
    )
    return out


class MetatomicBackend:
    """
    :param model: path to an exported model, or an ``AtomisticModel``.
    :param outputs: names of the outputs to evaluate; an ensemble output maps
        to the output it is the ensemble of, a prediction to ``None``.
    """

    def __init__(
        self,
        model: str | Path | Any,
        outputs: Mapping[str, str | None],
        device: str,
        dtype: torch.dtype,
    ) -> None:
        import vesin.metatomic
        from metatomic.torch import (
            ModelEvaluationOptions,
            ModelOutput,
            load_atomistic_model,
        )

        if isinstance(model, str | Path):
            model = load_atomistic_model(str(model))
        self.model = model.to(device=device, dtype=dtype)
        available = set(self.model.capabilities().outputs)
        missing = sorted(set(outputs) - available)
        if missing:
            raise ValueError(
                f"the model has no output {missing}; available: {sorted(available)}"
            )
        self.outputs = dict(outputs)
        self.options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs={name: ModelOutput(per_atom=True) for name in outputs},
        )
        self.neighbors = [
            vesin.metatomic.NeighborList(options=o, length_unit="angstrom")
            for o in self.model.requested_neighbor_lists()
        ]
        self.device, self.dtype = device, dtype

    def __call__(
        self, mols: Sequence[gto.Mole], cell: np.ndarray | None = None
    ) -> dict[str, list[np.ndarray]]:
        """Raw coefficients per output and molecule: (naux,) or (n_members, naux)."""
        from qpet._tensormap import to_pyscf

        systems = [system(mol, cell, self.dtype, self.device) for mol in mols]
        for nl in self.neighbors:
            nl.add_neighbor_list(systems=systems)
        with torch.no_grad():
            result = self.model(systems, self.options, False)
        out = {}
        for name, base in self.outputs.items():
            n = 1
            if base is not None:
                key = result[base].keys.entry(0)
                n = (
                    result[name].block(key).values.shape[-1]
                    // result[base].block(key).values.shape[-1]
                )
            out[name] = [c.cpu().double().numpy() for c in to_pyscf(result[name], n)]
        return out
