"""Model outputs and training targets stored as TensorMaps, in PySCF order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from metatensor.torch import TensorMap


def tensormap_coefficients(tensor: TensorMap, n_members: int = 1) -> list[np.ndarray]:
    """
    Coefficients of every system of a TensorMap (training targets, raw model
    outputs) in PySCF order, as float64 arrays: (naux,) per system, or
    (n_members, naux) for an ensemble output.
    """
    return [c.detach().cpu().double().numpy() for c in to_pyscf(tensor, n_members)]


def to_pyscf(tensor: TensorMap, n_members: int = 1) -> list[torch.Tensor]:
    """
    Coefficients of every system, sorted by atom, l, radial channel and m.

    Blocks are keyed by (l, parity, species) with samples (system, atom),
    components m and properties n. metatensor orders the l=1 components
    (y, z, x), PySCF (x, y, z). An ensemble output stacks its members along the
    properties, member-major (metatrain's LLPR layout); with ``n_members > 1``
    each system comes back as (n_members, naux).

    :return: One tensor per system, shape (naux,) or (n_members, naux), on the
        device of the TensorMap and keeping its autograd graph.
    """
    import torch

    keys, values = [], []
    for key, block in tensor.items():
        vals = block.values
        if int(key[0]) == 1:
            vals = vals[:, [2, 0, 1], :]
        n_samples, n_m, n_prop = vals.shape
        n_rad = n_prop // n_members
        vals = vals.reshape(n_samples, n_m, n_members, n_rad).permute(2, 0, 1, 3)
        device = vals.device
        samples = block.samples.values.to(device, dtype=torch.long)
        m = torch.arange(n_m, device=device)
        rad = block.properties.values[:n_rad, -1].to(device, dtype=torch.long)
        grid = torch.stack(
            torch.broadcast_tensors(
                samples[:, 0, None, None],
                samples[:, 1, None, None],
                torch.full((1, 1, 1), int(key[0]), device=device),
                rad[None, None, :],
                m[None, :, None],
            ),
            dim=-1,
        )
        mask = ~torch.isnan(vals[0])
        keys.append(grid[mask])
        values.append(vals[:, mask])
    if not values:
        return []
    keys_all = torch.cat(keys)
    values_all = torch.cat(values, dim=1)
    order = torch.arange(len(keys_all), device=keys_all.device)
    for column in reversed(range(5)):
        order = order[torch.argsort(keys_all[order, column], stable=True)]
    system = keys_all[order, 0]
    bounds = (system[1:] != system[:-1]).nonzero(as_tuple=True)[0] + 1
    parts = torch.tensor_split(values_all[:, order], bounds.cpu(), dim=1)
    return [p[0] if n_members == 1 else p for p in parts]
