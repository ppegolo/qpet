"""Scripted AtomisticModel with fixed outputs.

No ``from __future__ import annotations`` here: TorchScript needs the real
annotation objects.
"""

from typing import Dict, List, Optional  # noqa: UP035

import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)

from qpet._tensormap import from_pyscf as to_tensormap  # noqa: F401


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
