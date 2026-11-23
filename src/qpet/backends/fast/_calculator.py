"""User-facing entry point: positions and atomic numbers in, PySCF coefficients out."""

from collections.abc import Sequence

import torch

from .model import DEFAULT_MAX_CHUNK_TOKENS, PetInference
from .neighbors import (
    DEFAULT_BUCKET_BOUNDS,
    DEFAULT_SEARCH_FRACTION,
    Batch,
    adaptive_cutoffs,
    neighbor_list,
    preprocess,
)
from .partition import (
    gather_core_coefficients,
    halo_radius,
    partition_system,
)
from .postprocess import postprocess

#: Atoms used to estimate the adaptive cutoff before partitioning. Only needs to
#: be a representative patch of bulk, so it is deliberately small.
_PROBE_ATOMS = 4096

#: Rough device memory per atom of a block's *support*, measured on a three-layer
#: d=256 model in float32. Only used to pick a first block size; being wrong is
#: recoverable, since running out of memory halves the block size and retries.
_BYTES_PER_ATOM = 110_000


class Calculator:
    """Evaluate a frozen PET density model.

    :param artifact_path: file written by
        :func:`qpet.backends.fast.export.export`.
    :param device: torch device to run on.
    :param dtype: ``torch.float64`` (default, matches training) or
        ``torch.float32``.
    :param bucket_bounds: neighbor-count boundaries used to group atoms into
        separate neighbor grids. ``()`` gives a single grid padded to the global
        maximum, which is what metatrain does.
    :param min_bucket_rows: groups with fewer atoms than this are merged into a
        neighbour; ``None`` scales the threshold with the system size.
    :param compile: compile the backbone with ``torch.compile``. Costs a one-off
        compilation on the first call and speeds up every call after it. Compiled
        with ``dynamic=True``, so a new atom count does not trigger a recompile --
        with static shapes each new size costs another full compilation, which is
        the wrong trade for one-shot evaluation of differently sized molecules.
    :param neighbor_backend: ``"torch"`` builds the neighbor list on the GPU via
        vesin.torch, ``"numpy"`` goes through the CPU vesin and a host round trip.
        ``None`` prefers the torch backend when it is installed.
    :param search_fraction: fraction of the model's cutoff to search on the first
        attempt. The adaptive cutoff usually settles well below the model cutoff,
        so a full-cutoff search finds several times more pairs than survive it.
        Searching short is exact -- pairs beyond the adapted cutoff contribute
        exactly zero -- and the result is checked, with a rebuild at the full
        cutoff on the rare miss. Pass ``1.0`` to always search the full cutoff.
    :param max_chunk_tokens: cap on how much goes through a transformer layer, and
        through the message-combination MLP, at a time. Peak memory is set by these
        transients rather than by the persistent per-edge messages, so capping them
        bounds the working set and lets much larger systems fit. ``None`` processes
        everything whole.
    :param attention_backend: ``"triton"`` for the fused gated-attention kernel,
        ``"sdpa"`` for PyTorch's, ``None`` (default) to use the kernel wherever it
        applies and fall back otherwise.
    :param autocast_dtype: run the GNN backbone in reduced precision, typically
        ``torch.float16``. **Opt-in**: it is an accuracy/speed trade, not a free
        win. Prefer float16 over bfloat16 here -- the activations are norm-bounded
        so there is no overflow to protect against, and float16's extra mantissa
        bits measure several times more accurate. The heads and readouts always
        stay in ``dtype``.
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        bucket_bounds: Sequence[int] = DEFAULT_BUCKET_BOUNDS,
        min_bucket_rows: int | None = None,
        compile: bool = False,
        autocast_dtype: torch.dtype | None = None,
        attention_backend: str | None = None,
        neighbor_backend: str | None = None,
        search_fraction: float = DEFAULT_SEARCH_FRACTION,
        max_chunk_tokens: int | None = DEFAULT_MAX_CHUNK_TOKENS,
    ) -> None:
        artifact = torch.load(artifact_path, weights_only=False, map_location="cpu")
        if artifact["format"] != 3:
            raise ValueError(
                f"artifact format {artifact['format']} is not supported; re-export "
                "the checkpoint with qpet.backends.fast.export"
            )

        self.device = torch.device(device)
        self.dtype = dtype
        self.model = PetInference(artifact).to(self.device, dtype)
        self.model.fold()
        self.atomic_types = artifact["atomic_types"]
        self.target_names = list(artifact["targets"])
        self.members = {
            name: target["layout"]["members"]
            for name, target in artifact["targets"].items()
        }
        self.constants: dict[str, dict[str, torch.Tensor]] = {
            name: {
                key: (
                    value.to(self.device, dtype)
                    if value.is_floating_point()
                    else value.to(self.device)
                )
                for key, value in target["post"].items()
                if isinstance(value, torch.Tensor)
            }
            for name, target in artifact["targets"].items()
        }
        self.blocks = {
            name: [b["n_m"] * b["n_p"] for b in target["blocks"]]
            for name, target in artifact["targets"].items()
        }
        self.llpr = {
            name: _move(target["llpr"], self.device, dtype)
            for name, target in artifact["targets"].items()
            if target.get("llpr") is not None
        }
        self.bucket_bounds = bucket_bounds
        self.min_bucket_rows = min_bucket_rows
        self.autocast_dtype = autocast_dtype
        self.model.set_attention_backend(attention_backend)
        self.model.max_chunk_tokens = max_chunk_tokens
        self.neighbor_backend = neighbor_backend
        if not 0.0 < search_fraction <= 1.0:
            raise ValueError(
                f"search_fraction must be in (0, 1], not {search_fraction}"
            )
        # Grows toward the model cutoff if a system ever needs more, and stays
        # there: repeatedly rebuilding costs more than searching a little wide.
        self._search_radius = self.model.cutoff * search_fraction

        if autocast_dtype is not None and self.device.type == "cpu":
            raise ValueError(
                "autocast_dtype is only supported on CUDA; on CPU it is slower "
                "than full precision"
            )

        self._forward = self.model.forward
        if compile:
            self._forward = torch.compile(self.model.forward, dynamic=True)

    def raw_blocks(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        cell: torch.Tensor | None = None,
        pbc: bool = False,
        charge: int = 0,
        spin_multiplicity: int = 1,
        atomic_cutoffs: torch.Tensor | None = None,
        names: list[str] | None = None,
        ensembles: list[str] | None = None,
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor], torch.Tensor]:
        """Model output before post-processing: per target, one tensor per block.

        ``charge`` and ``spin_multiplicity`` are used only by models trained with
        system conditioning; the defaults match metatrain's (neutral singlet).
        """
        positions = positions.to(self.device, self.dtype)
        types = types.to(self.device)
        if cell is not None:
            cell = cell.to(self.device, self.dtype)

        unknown = sorted(set(types.tolist()) - set(self.atomic_types))
        if unknown:
            raise ValueError(
                f"atomic types {unknown} are not supported by this model "
                f"(it knows {self.atomic_types})"
            )

        batch = self._build_batch(positions, types, cell, pbc, atomic_cutoffs)
        charge_tensor = torch.tensor([charge], dtype=torch.long, device=self.device)
        spin_tensor = torch.tensor(
            [spin_multiplicity], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            blocks, features = self._forward(
                batch, charge_tensor, spin_tensor, self.autocast_dtype, names, ensembles
            )
        return blocks, features, types

    def _build_batch(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: bool,
        atomic_cutoffs: torch.Tensor | None = None,
    ) -> Batch:
        """Neighbor list plus preprocessing, retrying if the search was too short.

        The adapted cutoffs are only known after preprocessing, so the guard can
        only be checked afterwards. A miss means the list omitted pairs the model
        needs, and the only fix is to rebuild at the full cutoff.
        """
        species_index = self.model.species_to_species_index[types.long()]
        radius = self._search_radius
        while True:
            centers, neighbors, cell_shifts = neighbor_list(
                positions, cell, pbc, radius, backend=self.neighbor_backend
            )
            batch = preprocess(
                positions=positions,
                species_index=species_index,
                centers=centers,
                neighbors=neighbors,
                cell_shifts=cell_shifts,
                cell=cell,
                cutoff=self.model.cutoff,
                cutoff_width=self.model.cutoff_width,
                cutoff_width_adaptive=self.model.cutoff_width_adaptive,
                num_neighbors_adaptive=self.model.num_neighbors_adaptive,
                bucket_bounds=self.bucket_bounds,
                min_bucket_rows=self.min_bucket_rows,
                atomic_cutoffs=atomic_cutoffs,
            )
            if batch.required_radius < radius or radius >= self.model.cutoff:
                # Remember what this system needed, so similar ones do not miss.
                self._search_radius = min(
                    self.model.cutoff,
                    max(self._search_radius, batch.required_radius * 1.1),
                )
                return batch
            radius = self.model.cutoff

    def compute(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        cell: torch.Tensor | None = None,
        pbc: bool = False,
        charge: int = 0,
        spin_multiplicity: int = 1,
        atomic_cutoffs: torch.Tensor | None = None,
        names: list[str] | None = None,
        ensembles: list[str] | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Predict auxiliary-basis coefficients in PySCF order.

        :param positions: ``(n_atoms, 3)`` in the model's length unit.
        :param types: ``(n_atoms,)`` atomic numbers.
        :param cell: ``(3, 3)``, or ``None`` for a non-periodic system.
        :param pbc: whether the cell is periodic.
        :param charge: total charge; only used with system conditioning.
        :param spin_multiplicity: ``2S + 1``; only used with system conditioning.
        :param names: targets to evaluate, all by default.
        :param ensembles: targets whose LLPR deviations to add, as ``<name>/llpr``.
        :return: per target, ``(coefficients, counts_per_atom)``: ``coefficients``
            is ``(n_coefficients, n_members)``, atom-major and ordered as PySCF's
            auxiliary basis expects; ``counts_per_atom`` gives the per-atom lengths.
        """
        blocks, features, types = self.raw_blocks(
            positions,
            types,
            cell,
            pbc,
            charge,
            spin_multiplicity,
            atomic_cutoffs,
            names,
            ensembles,
        )
        out = {
            name: postprocess(raw, types, self.constants[name])
            for name, raw in blocks.items()
        }
        for name, llf in features.items():
            out[name + "/llpr"] = postprocess(
                self._deviations(name, llf, types),
                types,
                self.constants[name],
                deviations=True,
            )
        return out

    def _deviations(
        self, name: str, features: torch.Tensor, types: torch.Tensor
    ) -> list[torch.Tensor]:
        """LLPR deviations of the members from their mean, one
        ``(n_members, n_atoms, n_m * n_p)`` tensor per block, as metatrain's LLPR
        computes them: one linear map of the last-layer features per atomic type
        and block, onto the properties of that type."""
        llpr = self.llpr[name]
        k = llpr["members"]
        out = []
        for j, width in enumerate(self.blocks[name]):
            values = features[:, llpr["columns"][j]]
            deviations = values.new_zeros(values.shape[0], k, width)
            for z, weights, properties in zip(
                llpr["atomic_types"], llpr["weights"], llpr["properties"], strict=True
            ):
                mask = types == z
                if weights[j].shape[0] == 0 or not bool(mask.any()):
                    continue
                n_p = len(properties[j])
                rows = int(mask.sum())
                members = (values[mask] @ weights[j].T).reshape(rows, k, -1, n_p)
                n_m = members.shape[2]
                padded = values.new_zeros(rows, k, n_m, width // n_m)
                padded[..., properties[j]] = members
                deviations[mask] = padded.reshape(rows, k, width)
            deviations = deviations - deviations.mean(dim=1, keepdim=True)
            out.append(deviations.transpose(0, 1))
        return out

    def compute_partitioned(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        charge: int = 0,
        spin_multiplicity: int = 1,
        max_core_atoms: int | None = None,
        names: list[str] | None = None,
        ensembles: list[str] | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Same as :meth:`compute`, for systems too large to evaluate at once.

        The system is cut into spatial blocks, each evaluated with a halo of the
        atoms it needs but does not own, and the blocks' core results
        concatenated. The result is the same as :meth:`compute` would give -- the
        halo is sized so that no core atom's computation can see the boundary --
        but peak memory is set by the largest block rather than by the system.

        Use it when the system does not fit. It is not a speed optimisation: a
        block processes its halo as well as its core, so the total work is
        somewhere between 1.5 and 3 times a single evaluation, depending on how
        large the blocks can be.

        Periodic systems are not supported: a block's halo would have to be built
        from periodic images, which the flat index bookkeeping here does not do.

        :param max_core_atoms: target atoms owned per block. Larger is more
            efficient -- the halo is amortised over more atoms -- until it stops
            fitting. ``None`` estimates one from free device memory and halves it
            on each out-of-memory, which is the setting to use unless you are
            deliberately measuring.
        :param names: targets to evaluate, all by default.
        :return: as :meth:`compute`, in the input atom order.
        """
        positions = positions.to(self.device, self.dtype)
        types = types.to(self.device)

        # Solve the adaptive cutoffs once, for the whole system. This is the only
        # step that needs an atom's own full neighbourhood, so doing it here lets
        # every block skip it -- and that removes a hop from the halo, which is
        # nearly half its volume. It is affordable precisely because it is the
        # cheap part: a neighbour list and a root find over per-edge scalars, not
        # per-edge feature vectors.
        global_cutoffs = self._global_adaptive_cutoffs(positions)
        halo = halo_radius(
            self.model.num_gnn_layers,
            float(global_cutoffs.max()),
            cutoffs_supplied=True,
        )

        if max_core_atoms is None:
            max_core_atoms = self._estimate_core_atoms()

        # Blocks are sized from an estimate, so running out of memory is expected
        # rather than exceptional: halve and start over. Restarting the whole
        # partition (instead of just the failed block) keeps every block the same
        # size, which keeps the halo overhead uniform.
        while True:
            try:
                pieces, partitions = self._evaluate_blocks(
                    positions,
                    types,
                    halo,
                    max_core_atoms,
                    charge,
                    spin_multiplicity,
                    global_cutoffs,
                    names,
                    ensembles,
                )
                break
            except torch.OutOfMemoryError:
                if max_core_atoms <= 1000:
                    raise
                max_core_atoms //= 2
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Blocks come out in spatial order; put the atoms back in input order.

        order = torch.argsort(torch.cat([p.core for p in partitions]))
        out = {}
        for name, per_block in pieces.items():
            all_counts = torch.cat([counts for _, counts in per_block])
            all_coefficients = torch.cat(
                [coefficients for coefficients, _ in per_block]
            )
            offsets = torch.cumsum(all_counts, dim=0) - all_counts
            ordered_counts = all_counts.index_select(0, order)
            ordered_offsets = offsets.index_select(0, order)
            total = int(ordered_counts.sum())
            run_of_slot = torch.repeat_interleave(
                torch.arange(ordered_counts.shape[0], device=self.device),
                ordered_counts,
            )
            starts = torch.cumsum(ordered_counts, dim=0) - ordered_counts
            within = torch.arange(total, device=self.device) - starts[run_of_slot]
            out[name] = (
                all_coefficients[ordered_offsets[run_of_slot] + within],
                ordered_counts,
            )
        return out

    def _estimate_core_atoms(self) -> int:
        """A first guess at block size, from free device memory."""
        if self.device.type != "cuda":
            return 50_000
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        per_atom = _BYTES_PER_ATOM * element_size / 4
        # Leave headroom, and assume a block's support is about twice its core.
        support = int(0.7 * free_bytes / per_atom)
        return max(1000, support // 2)

    def _evaluate_blocks(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        halo: float,
        max_core_atoms: int,
        charge: int,
        spin_multiplicity: int,
        global_cutoffs: torch.Tensor,
        names: list[str] | None,
        ensembles: list[str] | None,
    ) -> tuple[dict[str, list[tuple[torch.Tensor, torch.Tensor]]], list]:
        """Evaluate every block once, at a fixed block size."""
        partitions = partition_system(positions, halo, max_core_atoms)
        pieces: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for partition in partitions:
            results = self.compute(
                positions.index_select(0, partition.support),
                types.index_select(0, partition.support),
                charge=charge,
                spin_multiplicity=spin_multiplicity,
                atomic_cutoffs=global_cutoffs.index_select(0, partition.support),
                names=names,
                ensembles=ensembles,
            )
            for name, (coefficients, per_atom) in results.items():
                pieces.setdefault(name, []).append(
                    gather_core_coefficients(
                        coefficients, per_atom, partition.core_in_support
                    )
                )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return pieces, partitions

    def _global_adaptive_cutoffs(self, positions: torch.Tensor) -> torch.Tensor:
        """Solve every atom's adapted cutoff against the whole system.

        Kept separate from :meth:`_build_batch` because the neighbour list it needs
        is thrown away immediately: only the per-atom scalars are carried forward,
        so this costs a list build and a root find, not the memory of a batch.
        """
        radius = self._search_radius
        while True:
            centers, neighbors, _ = neighbor_list(
                positions, None, False, radius, backend=self.neighbor_backend
            )
            distances = (
                torch.norm(
                    positions.index_select(0, neighbors)
                    - positions.index_select(0, centers),
                    dim=-1,
                )
                + 1e-15
            )
            cutoffs, largest = adaptive_cutoffs(
                centers,
                distances,
                self.model.num_neighbors_adaptive,
                positions.shape[0],
                self.model.cutoff,
                self.model.cutoff_width_adaptive,
            )
            del centers, neighbors, distances
            if largest < radius or radius >= self.model.cutoff:
                self._search_radius = min(
                    self.model.cutoff, max(self._search_radius, largest * 1.1)
                )
                return cutoffs
            radius = self.model.cutoff


def _move(value, device: torch.device, dtype: torch.dtype):
    """``value`` with every tensor on ``device``, floating ones in ``dtype``."""
    if isinstance(value, torch.Tensor):
        return (
            value.to(device, dtype) if value.is_floating_point() else value.to(device)
        )
    if isinstance(value, list):
        return [_move(v, device, dtype) for v in value]
    if isinstance(value, dict):
        return {k: _move(v, device, dtype) for k, v in value.items()}
    return value
