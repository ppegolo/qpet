"""Scripted AtomisticModel with fixed outputs, and PySCF coefficients as TensorMaps.

No ``from __future__ import annotations`` here: TorchScript needs the real
annotation objects.
"""

from typing import Dict, List, Optional  # noqa: UP035

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)


def to_tensormap(auxmol, coeffs: np.ndarray) -> TensorMap:
    """
    Coefficients (naux,) or members (n_members, naux) in the layout of a model output.

    Blocks keyed by (l, 1, Z), samples (system 0, atom), components m with the
    l=1 order (y, z, x), properties the radial channels, member-major.
    """
    members = np.atleast_2d(coeffs)
    blocks: dict = {}
    ao_loc = auxmol.ao_loc_nr()
    for ib in range(auxmol.nbas):
        ell, atom = auxmol.bas_angular(ib), int(auxmol.bas_atom(ib))
        key = (ell, int(auxmol.atom_charge(atom)))
        n_ctr = auxmol.bas_nctr(ib)
        values = members[:, ao_loc[ib] : ao_loc[ib + 1]].reshape(
            len(members), n_ctr, 2 * ell + 1
        )
        if ell == 1:
            values = values[:, :, [1, 2, 0]]
        blocks.setdefault(key, {}).setdefault(atom, []).append(values)
    keys, out = [], []
    for (ell, z), atoms in sorted(blocks.items()):
        values = np.stack([np.concatenate(atoms[a], axis=1) for a in sorted(atoms)])
        n_atoms, n_members, n_rad, n_m = values.shape
        values = values.transpose(0, 3, 1, 2).reshape(n_atoms, n_m, n_members * n_rad)
        keys.append([ell, 1, z])
        out.append(
            TensorBlock(
                values=torch.tensor(values, dtype=torch.float64),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, a] for a in sorted(atoms)], dtype=torch.int32),
                ),
                components=[
                    Labels(
                        ["o3_mu"],
                        torch.arange(-ell, ell + 1, dtype=torch.int32)[:, None],
                    )
                ],
                properties=Labels(
                    ["n"], torch.arange(n_members * n_rad, dtype=torch.int32)[:, None]
                ),
            )
        )
    return TensorMap(
        Labels(
            ["o3_lambda", "o3_sigma", "atom_type"],
            torch.tensor(keys, dtype=torch.int32),
        ),
        out,
    )


class _Fixed(torch.nn.Module):
    n_calls: int

    def __init__(self, outputs: Dict[str, TensorMap]):  # noqa: UP006
        super().__init__()
        self.fixed = outputs
        self.n_calls = 0

    def forward(
        self,
        systems: List[System],  # noqa: UP006
        outputs: Dict[str, ModelOutput],  # noqa: UP006
        selected_atoms: Optional[Labels],  # noqa: UP045
    ) -> Dict[str, TensorMap]:  # noqa: UP006
        self.n_calls += 1
        result: Dict[str, TensorMap] = {}  # noqa: UP006
        for name in outputs:
            result[name] = self.fixed[name]
        return result


def fixed_model(outputs: Dict[str, TensorMap]) -> AtomisticModel:  # noqa: UP006
    """AtomisticModel that returns ``outputs`` whatever the systems."""
    capabilities = ModelCapabilities(
        outputs={name: ModelOutput(per_atom=True) for name in outputs},
        atomic_types=[1, 8],
        interaction_range=0.0,
        length_unit="angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    scripted = torch.jit.script(_Fixed(outputs))
    return AtomisticModel(scripted.eval(), ModelMetadata(), capabilities)
