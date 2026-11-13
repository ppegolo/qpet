"""Geometry preprocessing: neighbor list -> adaptive cutoffs -> flat edges + NEF view.

The canonical layout here is the **flat edge array** ``(n_edges, ...)``, with the
node-edge-feature (NEF) grid ``(n_nodes, max_neighbors, ...)`` built as a view on
top of it. metatrain does the opposite -- it lives in NEF and encodes edge reversal
as a flat index ``j * max_neighbors + position``, which bakes the grid width into
the message-passing indices. Keeping edges flat makes reversal a plain gather by
``corresponding_edges``, independent of how the grid is shaped, which is what lets
the grid later be bucketed by neighbor count without touching message passing.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch

#: Grouping boundaries for the neighbor grid. Atoms whose neighbor count falls in
#: the same interval share a grid, whose width is that group's true maximum. Wider
#: groups mean fewer kernel launches but more padding; the attention cost is
#: quadratic in the width, so the ladder is tight where the counts actually sit.
DEFAULT_BUCKET_BOUNDS = (8, 12, 16, 20, 24, 32, 48, 64, 96, 128)

#: Floor and fraction for the group-merge threshold when ``min_bucket_rows`` is not
#: given: groups holding fewer than ``max(floor, n_atoms // divisor)`` atoms are
#: merged into a neighbouring one. A grid of a handful of rows pays a full set of
#: kernel launches for almost no work, and how few is "few" scales with the system.
DEFAULT_MIN_BUCKET_ROWS_FLOOR = 128
DEFAULT_MIN_BUCKET_ROWS_DIVISOR = 16

#: Fraction of the model's cutoff to search on the first attempt. The adaptive
#: cutoff typically settles far below the cutoff the neighbor list is built at --
#: on a trained 8 A model the adapted radii sit around 3-5 A, so a full-cutoff
#: search finds roughly seven times more pairs than survive. Searching short and
#: rebuilding on the rare miss is exact (see `preprocess`) and much cheaper.
DEFAULT_SEARCH_FRACTION = 0.75


class Bucket(NamedTuple):
    """One neighbor grid, covering the atoms whose neighbor counts are similar.

    :param nodes: ``(n_rows,)`` node indices this grid covers.
    :param edge_indices: ``(n_rows, width)`` flat edge index of each slot; the
        value at padded slots is arbitrary and always masked.
    :param mask: ``(n_rows, width)`` bool, true for real neighbors.
    :param edge_ids: ``(n_real,)`` flat edge indices in row-major slot order, so
        that ``flat[edge_ids] = grid[mask]`` moves a grid result back to the flat
        edge array.
    """

    nodes: torch.Tensor
    edge_indices: torch.Tensor
    mask: torch.Tensor
    edge_ids: torch.Tensor


class Batch(NamedTuple):
    """Preprocessed geometry for one evaluation.

    Flat (per-edge) arrays are the source of truth; the buckets are views onto them.
    """

    species_index: torch.Tensor  # (n_nodes,)   species -> contiguous index
    centers: torch.Tensor  # (n_edges,)
    neighbors: torch.Tensor  # (n_edges,)
    edge_vectors: torch.Tensor  # (n_edges, 3)
    cutoff_factors: torch.Tensor  # (n_edges,)
    corresponding_edges: torch.Tensor  # (n_edges,)  index of the reversed edge
    buckets: list[Bucket]
    #: Radius the neighbor list had to reach for this batch to be exact: the
    #: cutoff solver's ``reach``, or the largest supplied cutoff. If the list was
    #: built at or below it, the batch is invalid.
    required_radius: float


def cutoff_func_bump(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """Bump (C-infinity) cutoff function; identical to metatrain's."""
    scaled = (values - (cutoff - width)) / width
    clamped = scaled.clamp(1e-6, 1.0 - 1e-6)
    return 0.5 * (1 + torch.tanh(1 / torch.tan(math.pi * clamped)))


def _smoothed_count(
    r_per_atom: torch.Tensor,
    edge_distances: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    width: float,
    inv_max_cutoff: float,
    target_count: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``n_total(r)`` and ``dn_total/dr``, the root-finder's objective.

    ``n_total(r) = sum_j bump(d_j, r, w) + target_count * (r / max_cutoff)**3``.
    The cubic baseline makes it monotonic on ``[0, max_cutoff]`` with exactly one
    crossing of ``target_count``.
    """
    r_per_edge = r_per_atom[centers]
    scaled = (edge_distances - (r_per_edge - width)) / width
    active = (scaled > 0.0) & (scaled < 1.0)
    smaller = scaled <= 0.0

    # Clamp into the open interval before the trig form: matches the epsilon in
    # cutoff_func_bump, and keeps the derivative finite at the endpoints.
    safe = scaled.clamp(1e-6, 1.0 - 1e-6)
    s = math.pi * safe
    sin_s = torch.sin(s)
    cot_s = torch.cos(s) / sin_s
    tanh_cot = torch.tanh(cot_s)

    f = torch.where(active, 0.5 * (1.0 + tanh_cot), smaller.to(scaled.dtype))
    sech_sq = 1.0 - tanh_cot * tanh_cot
    df_dr = ((0.5 * math.pi / width) * sech_sq / (sin_s * sin_s)) * active.to(
        scaled.dtype
    )

    n = torch.zeros(num_nodes, dtype=edge_distances.dtype, device=edge_distances.device)
    n.index_add_(0, centers, f)
    dn = torch.zeros_like(n)
    dn.index_add_(0, centers, df_dr)

    x = r_per_atom * inv_max_cutoff
    return (
        n + target_count * x.pow(3),
        dn + 3.0 * target_count * x.pow(2) * inv_max_cutoff,
    )


def adaptive_cutoffs(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    target_count: float,
    num_nodes: int,
    max_cutoff: float,
    width: float,
    iterations: int = 10,
) -> tuple[torch.Tensor, float]:
    """Per-atom cutoff solving ``n_total(r) = target_count`` (Newton + bisection).

    The trailing correction step is kept even though inference needs no gradients:
    it is not only a gradient hook, it also shifts the returned value by the
    residual and applies the ``[max_cutoff / 16, max_cutoff]`` clamp.

    :return: ``(cutoffs, reach)``, with ``reach`` the largest radius at which the
        count was evaluated or a cutoff returned. The fixed iteration count leaves
        the root unconverged, so the result depends on the count at every probe,
        and a neighbour list is only long enough if it reaches past ``reach``.
    """
    inv_max_cutoff = 1.0 / max_cutoff
    r_lo = torch.zeros(
        num_nodes, dtype=edge_distances.dtype, device=edge_distances.device
    )
    r_hi = torch.full_like(r_lo, max_cutoff)
    r = 0.5 * r_hi
    reach = r.new_tensor(0.5 * max_cutoff)

    for _ in range(iterations):
        n, dn = _smoothed_count(
            r, edge_distances, centers, num_nodes, width, inv_max_cutoff, target_count
        )
        f = n - target_count
        below = f <= 0
        r_lo = torch.where(below, r, r_lo)
        r_hi = torch.where(below, r_hi, r)
        r_newton = r - f / dn.clamp_min(1e-6)
        inside = (r_newton >= r_lo) & (r_newton <= r_hi)
        r = torch.where(inside, r_newton, 0.5 * (r_lo + r_hi))
        if num_nodes > 0:
            reach = torch.maximum(reach, r.max())

    n_root, dn_root = _smoothed_count(
        r, edge_distances, centers, num_nodes, width, inv_max_cutoff, target_count
    )
    residual = n_root - target_count
    cutoffs = (r - residual / dn_root.clamp_min(1e-6)).clamp(
        max_cutoff / 16.0, max_cutoff
    )
    if num_nodes > 0:
        reach = torch.maximum(reach, cutoffs.max())
    return cutoffs, float(reach)


def _corresponding_edges(
    centers: torch.Tensor, neighbors: torch.Tensor, cell_shifts: torch.Tensor
) -> torch.Tensor:
    """For each edge ``i -> j``, the index of the edge ``j -> i``."""
    if centers.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=centers.device)

    centers = centers.to(torch.int64)
    neighbors = neighbors.to(torch.int64)
    cell_shifts = cell_shifts.to(torch.int64)

    lowest = cell_shifts.amin(dim=0)
    shifted = cell_shifts - lowest
    shifted_reversed = -cell_shifts - lowest

    extent = shifted.amax(dim=0) + 1
    n_nodes = centers.amax() + 1
    size_z = extent[2]
    size_yz = extent[1] * size_z
    size_xyz = extent[0] * size_yz
    size_total = n_nodes * size_xyz

    def encode(first, second, shifts):
        return (
            first * size_total
            + second * size_xyz
            + shifts[:, 0] * size_yz
            + shifts[:, 1] * size_z
            + shifts[:, 2]
        )

    forward = encode(centers, neighbors, shifted).argsort()
    backward = encode(neighbors, centers, shifted_reversed).argsort()
    out = torch.empty_like(centers)
    out[forward] = backward
    return out


def _group_bounds(
    counts: torch.Tensor, bounds: Sequence[int], min_rows: int
) -> list[tuple[int, int]]:
    """Turn the neighbor-count histogram into ``(low, high]`` groups.

    Groups are taken from ``bounds``, then any group with fewer than ``min_rows``
    atoms is merged into the next wider one so that no grid is launched for a
    handful of rows. Returns count ranges, not widths: each group's width is its
    own true maximum, decided by the caller.
    """
    largest = int(counts.max()) if counts.numel() > 0 else 0
    ladder = [b for b in sorted(bounds) if b < largest] + [largest]

    groups: list[tuple[int, int]] = []
    low = -1
    for high in ladder:
        groups.append((low, high))
        low = high

    # Merge undersized groups into the next wider one.
    merged: list[tuple[int, int]] = []
    pending_low: int | None = None
    for low, high in groups:
        start = low if pending_low is None else pending_low
        rows = int(((counts > start) & (counts <= high)).sum())
        if rows == 0 or rows < min_rows:
            pending_low = start
            continue
        merged.append((start, high))
        pending_low = None

    if pending_low is not None:
        # The widest group has nowhere to merge upward, so it merges *downward*
        # into its predecessor instead -- widening that grid rather than launching
        # a whole extra one for a handful of atoms. A single atom with an unusually
        # large neighbor count would otherwise get a grid to itself.
        if merged:
            merged[-1] = (merged[-1][0], largest)
        else:
            merged.append((pending_low, largest))
    return merged


def _build_buckets(
    centers: torch.Tensor,
    num_nodes: int,
    bounds: Sequence[int],
    min_rows: int | None,
) -> list[Bucket]:
    """Partition atoms by neighbor count into one grid per group.

    A single grid padded to the global maximum wastes work on every atom with
    fewer neighbors than the busiest one; since attention is quadratic in the grid
    width, that waste is superlinear. Grouping atoms of similar degree keeps each
    grid close to the counts it actually holds.
    """
    if min_rows is None:
        min_rows = max(
            DEFAULT_MIN_BUCKET_ROWS_FLOOR,
            num_nodes // DEFAULT_MIN_BUCKET_ROWS_DIVISOR,
        )

    counts = torch.zeros(
        num_nodes, dtype=torch.int64, device=centers.device
    ).scatter_add_(0, centers, torch.ones_like(centers))
    full_width = int(counts.max()) if counts.numel() > 0 else 0

    # One left-packed index grid at full width, built once; each group then reads
    # the rows it owns and truncates the columns it does not need.
    order = torch.argsort(centers, stable=True)
    starts = torch.cumsum(counts, dim=0) - counts
    position = torch.arange(
        centers.shape[0], device=centers.device
    ) - starts.index_select(0, centers.index_select(0, order))
    full_indices = torch.zeros(
        num_nodes * max(full_width, 1), dtype=torch.int64, device=centers.device
    )
    if full_width > 0:
        full_indices[centers.index_select(0, order) * full_width + position] = order
    full_indices = full_indices.view(num_nodes, max(full_width, 1))[:, :full_width]

    arange = torch.arange(full_width, device=centers.device)
    full_mask = arange.view(1, -1).expand(num_nodes, -1) < counts.view(-1, 1)

    buckets: list[Bucket] = []
    for low, high in _group_bounds(counts, bounds, min_rows):
        nodes = torch.nonzero((counts > low) & (counts <= high)).squeeze(-1)
        if nodes.numel() == 0:
            continue
        width = int(counts.index_select(0, nodes).max())
        edge_indices = full_indices.index_select(0, nodes)[:, :width]
        mask = full_mask.index_select(0, nodes)[:, :width]
        buckets.append(
            Bucket(
                nodes=nodes,
                edge_indices=edge_indices,
                mask=mask,
                edge_ids=edge_indices[mask],
            )
        )
    return buckets


def preprocess(
    positions: torch.Tensor,
    species_index: torch.Tensor,
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    cell_shifts: torch.Tensor,
    cell: torch.Tensor | None,
    cutoff: float,
    cutoff_width: float,
    cutoff_width_adaptive: float,
    num_neighbors_adaptive: float,
    bucket_bounds: Sequence[int] = DEFAULT_BUCKET_BOUNDS,
    min_bucket_rows: int | None = None,
    atomic_cutoffs: torch.Tensor | None = None,
) -> Batch:
    """Neighbor list -> :class:`Batch`.

    :param positions: ``(n_nodes, 3)``.
    :param species_index: ``(n_nodes,)`` contiguous species indices.
    :param centers, neighbors: ``(n_edges,)`` node indices of each pair.
    :param cell_shifts: ``(n_edges, 3)`` integer periodic images.
    :param cell: ``(3, 3)`` or ``None`` for a non-periodic system.
    :param bucket_bounds: neighbor-count boundaries for grouping atoms into grids.
        Pass ``()`` for a single grid padded to the global maximum.
    :param min_bucket_rows: groups smaller than this are merged into a neighbour;
        ``None`` scales it with the system size.
    :param atomic_cutoffs: per-atom adapted cutoffs to use instead of solving for
        them. Supplying them is what lets a partitioned run drop a hop of halo:
        the solve is the only step that needs an atom's *own* full neighbourhood,
        so once it is done globally, a block only needs the atoms whose features
        it reads, not the atoms those need to be solved.

    **On searching shorter than the cutoff.** The neighbor list may be built at a
    radius below ``cutoff`` and the result is still exact, provided the cutoff
    solver never probed at or beyond that radius -- which the returned
    ``required_radius`` lets the caller check.

    The reason is that ``cutoff_func_bump`` is *exactly* zero, not merely small,
    once ``d >= r``: its argument clamps to ``1 - 1e-6``, whose cotangent is about
    -3e5, and ``tanh`` of that is -1 in floating point. So a pair omitted from the
    list contributes exactly nothing to the smoothed neighbor count at any probe
    radius below the search radius, and the root finder takes the same steps it
    would have taken with the full list. A probe beyond the search radius sees a
    count that is too small, which can send a Newton step the wrong way; the
    solver is not iterated to convergence, so that changes its answer.
    """
    num_nodes = positions.shape[0]
    if cell is None:
        edge_vectors = positions[neighbors] - positions[centers]
    else:
        edge_vectors = (
            positions[neighbors]
            - positions[centers]
            + cell_shifts.to(cell.dtype) @ cell
        )
    edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15

    if atomic_cutoffs is None:
        atomic_cutoffs, required_radius = adaptive_cutoffs(
            centers,
            edge_distances,
            num_neighbors_adaptive,
            num_nodes,
            cutoff,
            cutoff_width_adaptive,
        )
    else:
        required_radius = float(atomic_cutoffs.max()) if num_nodes > 0 else 0.0
    # PET indexes edge ij against ji, so the pair cutoff must be symmetric.
    pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0

    keep = torch.nonzero(edge_distances <= pair_cutoffs).squeeze(-1)
    centers = centers.index_select(0, keep)
    neighbors = neighbors.index_select(0, keep)
    edge_vectors = edge_vectors.index_select(0, keep)
    cell_shifts = cell_shifts.index_select(0, keep)
    edge_distances = edge_distances.index_select(0, keep)
    pair_cutoffs = pair_cutoffs.index_select(0, keep)

    cutoff_factors = cutoff_func_bump(edge_distances, pair_cutoffs, cutoff_width)

    return Batch(
        species_index=species_index,
        centers=centers,
        neighbors=neighbors,
        edge_vectors=edge_vectors,
        cutoff_factors=cutoff_factors,
        corresponding_edges=_corresponding_edges(centers, neighbors, cell_shifts),
        buckets=_build_buckets(centers, num_nodes, bucket_bounds, min_bucket_rows),
        required_radius=required_radius,
    )


try:
    import vesin.torch as _vesin_torch

    HAS_VESIN_TORCH = True
except ImportError:  # pragma: no cover - exercised only without vesin-torch
    HAS_VESIN_TORCH = False


def neighbor_list(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    pbc: bool,
    cutoff: float,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full neighbor list. Returns ``(centers, neighbors, cell_shifts)``.

    ``vesin.torch`` builds the list on whatever device the positions are on. The
    plain ``vesin`` path goes through numpy, which on CUDA means a round trip and
    a synchronisation on every call -- an order of magnitude slower, and enough to
    make the neighbor list a third of the total runtime.

    :param backend: ``"torch"`` or ``"numpy"`` to pin one; ``None`` picks the torch
        backend whenever it is installed.
    """
    if backend is None:
        backend = "torch" if HAS_VESIN_TORCH else "numpy"
    if backend == "torch" and not HAS_VESIN_TORCH:
        raise RuntimeError("vesin.torch is not installed; `pip install vesin-torch`")

    if backend == "torch":
        box = (
            cell
            if cell is not None
            else torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)
        )
        builder = _vesin_torch.NeighborList(cutoff=cutoff, full_list=True)
        try:
            centers, neighbors, shifts = builder.compute(
                points=positions, box=box, periodic=pbc, quantities="ijS"
            )
        except RuntimeError as error:
            if "out of memory" not in str(error):
                raise
            # vesin's CUDA path allocates with cudaMalloc, outside PyTorch's
            # caching allocator, so a large cached pool can starve it even when
            # plenty is free overall. Returning the cache usually fixes it; if it
            # does not, the host build still works and is only slower.
            torch.cuda.empty_cache()
            try:
                centers, neighbors, shifts = builder.compute(
                    points=positions, box=box, periodic=pbc, quantities="ijS"
                )
            except RuntimeError:
                return neighbor_list(positions, cell, pbc, cutoff, backend="numpy")
        return (
            centers.to(torch.int64),
            neighbors.to(torch.int64),
            shifts.to(torch.int64),
        )

    import vesin

    points = positions.detach().cpu().numpy()
    # vesin requires the box to match the points' dtype, including for the
    # all-zero box that stands in for a non-periodic system.
    box_array = (
        cell.detach().cpu().numpy().astype(points.dtype)
        if cell is not None
        else torch.zeros(3, 3, dtype=positions.dtype).numpy()
    )
    i, j, shifts = vesin.NeighborList(cutoff=cutoff, full_list=True).compute(
        points=points,
        box=box_array,
        periodic=pbc,
        quantities="ijS",
    )
    device = positions.device
    return (
        torch.as_tensor(i, dtype=torch.int64, device=device),
        torch.as_tensor(j, dtype=torch.int64, device=device),
        torch.as_tensor(shifts, dtype=torch.int64, device=device),
    )
