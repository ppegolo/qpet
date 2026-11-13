"""Spatial partitioning for systems larger than the device can hold at once.

PET is a local model: an atom's prediction depends only on atoms within a finite
radius. So a large system can be cut into spatial blocks, each evaluated with a
halo of surrounding atoms it needs but does not own, and the blocks' core results
concatenated. The answer is identical to evaluating the whole system, provided the
halo is wide enough.

**How wide.** With ``L`` message-passing layers, a core atom's output depends on
atoms up to ``L + 1`` hops away -- ``L`` for its node features, one more for the
reversed messages its edges carry -- each hop at most one pair cutoff ``a``, so
``(L + 1) * a`` carries the output. The adaptive cutoff normally forces one more
hop -- an atom at the edge of the halo has its cutoff solved from *its*
neighbourhood, which reaches further -- but solving the cutoffs once for the whole
system removes that dependence. Since the halo is a shell, a hop off its radius is
nearly half its volume.

**What it costs.** A block of side ``D`` with halo ``h`` processes
``((D + 2h) / D)^3`` times as many atoms as it owns. For a three-layer model with
``a`` around 6 A the halo is roughly 24 A, so blocks have to be a few hundred
angstrom across before the overhead falls under two. Partitioning is what makes
arbitrarily large systems *possible*, not what makes them fast: use it when the
system does not fit, not before.
"""

from typing import NamedTuple

import torch


class Partition(NamedTuple):
    """One block: the atoms it owns, and the atoms it needs to evaluate them.

    :param core: indices into the full system of the atoms this block owns.
    :param support: indices of core plus halo, in the order handed to the model.
    :param core_in_support: positions of ``core`` within ``support``.
    """

    core: torch.Tensor
    support: torch.Tensor
    core_in_support: torch.Tensor


def partition_system(
    positions: torch.Tensor,
    halo: float,
    max_core_atoms: int,
) -> list[Partition]:
    """Cut a system into spatial blocks, each with a halo of the atoms it needs.

    Blocks are a regular grid over the bounding box, sized so that a block holds
    about ``max_core_atoms`` at the system's mean density. Real systems are not
    uniform, so individual blocks vary; the grid is a partition of space, so every
    atom is owned by exactly one block regardless.

    :param positions: ``(n_atoms, 3)``.
    :param halo: distance beyond a block's bounds to include as support.
    :param max_core_atoms: target number of owned atoms per block.
    :return: one :class:`Partition` per non-empty block.
    """
    n_atoms = positions.shape[0]
    if n_atoms <= max_core_atoms:
        everything = torch.arange(n_atoms, device=positions.device)
        return [Partition(everything, everything, everything)]

    lower = positions.min(dim=0).values
    upper = positions.max(dim=0).values
    extent = (upper - lower).clamp_min(1e-6)

    # Split the longest current block edge, repeatedly, until there are enough
    # blocks. Sizing each axis independently against a target side length instead
    # rounds up three times over and overshoots badly -- asking for two blocks in a
    # roughly cubic box gives 2x2x2. Blocks want to be as few and as cubic as
    # possible: the halo is a shell around each one, so it is paid per block and in
    # proportion to surface area.
    n_blocks_target = max(1, -(-n_atoms // max_core_atoms))
    divisions = [1, 1, 1]
    while divisions[0] * divisions[1] * divisions[2] < n_blocks_target:
        longest = max(range(3), key=lambda axis: float(extent[axis]) / divisions[axis])
        divisions[longest] += 1

    # Block index of every atom, then a single sort so each block's members are
    # contiguous -- cheaper than testing every atom against every block.
    block_size = torch.stack(
        [extent[axis] / divisions[axis] for axis in range(3)]
    ).clamp_min(1e-6)
    coordinates = ((positions - lower) / block_size).floor().long()
    for axis in range(3):
        coordinates[:, axis].clamp_(0, divisions[axis] - 1)
    flat = (
        coordinates[:, 0] * divisions[1] * divisions[2]
        + coordinates[:, 1] * divisions[2]
        + coordinates[:, 2]
    )

    partitions: list[Partition] = []
    for block in torch.unique(flat):
        core = torch.nonzero(flat == block).squeeze(-1)
        if core.numel() == 0:
            continue
        block_lower = positions.index_select(0, core).min(dim=0).values - halo
        block_upper = positions.index_select(0, core).max(dim=0).values + halo
        inside = (positions >= block_lower).all(dim=1) & (positions <= block_upper).all(
            dim=1
        )
        support = torch.nonzero(inside).squeeze(-1)

        # Where each core atom sits inside `support`. Both are sorted ascending,
        # so a search over the support gives the mapping directly.
        core_in_support = torch.searchsorted(support, core)
        partitions.append(Partition(core, support, core_in_support))
    return partitions


def gather_core_coefficients(
    coefficients: torch.Tensor,
    counts_per_atom: torch.Tensor,
    core_in_support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only the core atoms' coefficients from a block's flat output.

    The coefficient vector is atom-major with a variable number of entries per
    atom, so selecting atoms means selecting variable-length runs.

    :param coefficients: flat coefficients for every atom of the support.
    :param counts_per_atom: how many entries each support atom contributed.
    :param core_in_support: which support atoms to keep.
    :return: ``(coefficients, counts)`` for the core atoms, in ``core`` order.
    """
    offsets = torch.cumsum(counts_per_atom, dim=0) - counts_per_atom
    core_counts = counts_per_atom.index_select(0, core_in_support)
    core_offsets = offsets.index_select(0, core_in_support)

    total = int(core_counts.sum())
    if total == 0:
        return coefficients[:0], core_counts

    # Expand each kept run into explicit indices: start offset of the run each
    # output slot belongs to, plus the slot's position within that run.
    run_of_slot = torch.repeat_interleave(
        torch.arange(core_counts.shape[0], device=core_counts.device), core_counts
    )
    starts = torch.cumsum(core_counts, dim=0) - core_counts
    position_in_run = (
        torch.arange(total, device=core_counts.device) - starts[run_of_slot]
    )
    return coefficients[core_offsets[run_of_slot] + position_in_run], core_counts


def halo_radius(
    num_gnn_layers: int, pair_cutoff: float, cutoffs_supplied: bool = False
) -> float:
    """Support radius a block needs beyond its own atoms.

    A core atom's node features after ``L`` layers depend on atoms up to ``L`` hops
    away, and its edge features one hop further: each edge also carries the
    reversed message from its neighbour's own last layer. ``L + 1`` hops of halo
    carries both.

    One more hop is needed for the adaptive cutoff: an atom at the edge of the
    halo has its cutoff solved from *its* neighbourhood. Solve the cutoffs
    globally beforehand and that hop is unnecessary -- which is worth a lot,
    because the halo is a shell and its cost grows as its volume.
    """
    hops = num_gnn_layers + 1 if cutoffs_supplied else num_gnn_layers + 2
    return hops * pair_cutoff
