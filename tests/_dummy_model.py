"""Scripted dummy AtomisticModel for model-mode tests.

Lives in its own module WITHOUT ``from __future__ import annotations``:
TorchScript inspects real annotation objects and cannot handle the stringized
annotations that future-import turns them into.
"""

from typing import Dict, List, Optional, Tuple  # noqa: UP035

import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)


class _Dummy(torch.nn.Module):
    """Returns fixed TensorMaps; counts forward passes."""

    n_calls: int

    def __init__(self, rho: TensorMap, dip: TensorMap):
        super().__init__()
        self.rho = rho
        self.dip = dip
        self.n_calls = 0

    def forward(
        self,
        systems: List[System],  # noqa: UP006
        outputs: Dict[str, ModelOutput],  # noqa: UP006
        selected_atoms: Optional[Labels],  # noqa: UP045
    ) -> Dict[str, TensorMap]:  # noqa: UP006
        self.n_calls += 1
        result: Dict[str, TensorMap] = {}  # noqa: UP006
        if "mtt::rho_c_jfit_coulomb" in outputs:
            result["mtt::rho_c_jfit_coulomb"] = self.rho
        if "mtt::electric_dipole" in outputs:
            result["mtt::electric_dipole"] = self.dip
        return result


def wrap_dummy(
    rho: TensorMap,
    dip: TensorMap,
    outputs: Dict[str, ModelOutput],  # noqa: UP006
) -> Tuple[AtomisticModel, torch.jit.ScriptModule]:  # noqa: UP006
    """Script a ``_Dummy`` and wrap it as an ``AtomisticModel``."""
    capabilities = ModelCapabilities(
        outputs=outputs,
        atomic_types=[1, 8],
        interaction_range=0.0,
        length_unit="angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    scripted = torch.jit.script(_Dummy(rho, dip))
    scripted.eval()
    return AtomisticModel(scripted, ModelMetadata(), capabilities), scripted
