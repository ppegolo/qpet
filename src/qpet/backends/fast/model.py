"""Inference-only PET backbone.

The backbone mirrors ``metatrain.pet.modules.backend.PETBackend`` name for name, and
the exporter lays the heads and readouts out as this module expects, so the frozen
weights load with ``strict=True``. That is deliberate: a structural mistake in this
reimplementation shows up as a load error rather than as silently wrong numbers.

Two things are done differently from metatrain, both verified equivalent:

* **Edge reversal is a flat gather.** metatrain encodes it as an index into the
  ``(n_nodes, max_neighbors)`` grid, which ties message passing to the grid width.
  Here messages live in a flat ``(n_edges, d)`` array and reversal is
  ``messages[corresponding_edges]``.
* **Attention applies the cutoff as a per-key gate.** metatrain materialises an
  ``(n_nodes, T, T)`` additive bias ``log(clamp(c_k, 1e-15))`` -- which depends on
  the key index only -- and hands it to SDPA. The identity
  ``softmax(qk + log c_k) = normalise(c_k * exp(qk))`` removes the tensor.

Padded slots of the grid differ from metatrain's (which leave edge 0's data there);
both carry a zero gate, which excludes them from the attention.
"""

import math

import torch
from torch import nn

from .attention import gated_attention
from .neighbors import Batch, Bucket

#: Default cap on how many *tokens* (rows times grid width) go through a layer at
#: once, and equally on how many edges go through the message-combination MLP.
#: Tokens rather than rows because that is what sizes the tensors: a bucket of ten
#: thousand atoms with a width of twenty-five is four hundred times the work of one
#: with a width of one. At this cap the widest single transient -- the feedforward
#: expansion, at two times d_feedforward -- stays a few hundred megabytes.
DEFAULT_MAX_CHUNK_TOKENS = 1 << 17


def _chunks(n_items: int, max_items: int | None) -> list[slice]:
    """Split ``n_items`` into consecutive slices of at most ``max_items``."""
    if max_items is None or n_items <= max_items:
        return [slice(0, n_items)]
    step = max(1, max_items)
    return [
        slice(start, min(start + step, n_items)) for start in range(0, n_items, step)
    ]


class FeedForward(nn.Module):
    """SwiGLU feed-forward, as PET parametrises it.

    ``w_in`` emits ``2 * dim_feedforward`` and the *first* chunk is the value while
    the second gates it: ``v * sigmoid(g)``. Not the conventional operand order --
    it is copied literally from metatrain.
    """

    def __init__(self, d_model: int, dim_feedforward: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * dim_feedforward)
        self.w_out = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(value * torch.sigmoid(gate))


class AttentionBlock(nn.Module):
    """Multi-head attention with a per-key cutoff gate.

    The gate enters as the additive bias ``log(clamp(c_k, 1e-15))``, which depends
    on the key index only. Both backends in :mod:`qpet.backends.fast.attention`
    exploit that: the Triton kernel takes the gate as a vector, and the SDPA
    fallback passes the
    bias with shape ``(n_nodes, 1, 1, n_tokens)`` so it broadcasts over queries
    instead of being materialised as ``(n_nodes, T, T)`` the way metatrain's
    ``repeat`` does. Neither ever holds the attention matrix in global memory --
    doing that by hand costs ``n_nodes * n_heads * T^2`` floats, about a gigabyte
    at thirty thousand atoms.
    """

    def __init__(self, total_dim: int, num_heads: int) -> None:
        super().__init__()
        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.input_linear = nn.Linear(total_dim, 3 * total_dim)
        self.output_linear = nn.Linear(total_dim, total_dim)
        self.num_heads = num_heads
        self.head_dim = total_dim // num_heads
        self.scale = self.head_dim**-0.5
        # None lets qpet.backends.fast.attention choose; "triton"/"sdpa" pin it.
        self.attention_backend: str | None = None

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        :param x: ``(n_nodes, n_tokens, total_dim)``.
        :param gate: ``(n_nodes, n_tokens)`` cutoff factor of each *key*.
        """
        n_nodes, n_tokens, total_dim = x.shape
        qkv = self.input_linear(x).reshape(
            n_nodes, n_tokens, 3, self.num_heads, self.head_dim
        )
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        out = gated_attention(
            queries, keys, values, gate, self.scale, force=self.attention_backend
        )
        out = out.transpose(1, 2).reshape(n_nodes, n_tokens, total_dim)
        return self.output_linear(out)


class TransformerLayer(nn.Module):
    """One PreLN transformer layer over ``[central token, neighbor tokens]``.

    The central token has **two distinct behaviours**, selected by whether the node
    dimension differs from the model dimension:

    * ``dim_node_features != d_model`` ("expanded"): the node is contracted into
      the attention, expanded back out, and gets its own residual connection and
      MLP, so its residual stream stays at the wider node dimension.
    * ``dim_node_features == d_model``: none of those modules exist, and the node
      output is the **bare attention output** -- no residual, no MLP. This is not
      just a parameter saving; it is a different computation, and it is what
      metatrain does whenever ``d_node == d_pet``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.attention = AttentionBlock(d_model, n_heads)
        self.norm_attention = nn.RMSNorm(d_model)
        self.norm_mlp = nn.RMSNorm(d_model)
        self.mlp = FeedForward(d_model, dim_feedforward)

        self.expanded_node_features = dim_node_features != d_model
        if self.expanded_node_features:
            self.center_contraction = nn.Linear(dim_node_features, d_model)
            self.center_expansion = nn.Linear(d_model, dim_node_features)
            self.norm_center_features = nn.RMSNorm(dim_node_features)
            self.center_mlp = FeedForward(dim_node_features, 2 * dim_node_features)

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param node: ``(n_nodes, 1, dim_node_features)``.
        :param edge: ``(n_nodes, n_neighbors, d_model)``.
        :param gate: ``(n_nodes, n_neighbors + 1)``.
        """
        if self.expanded_node_features:
            node_in = self.center_contraction(node)
        else:
            node_in = node
        tokens = torch.cat([node_in, edge], dim=1)
        updated = self.attention(self.norm_attention(tokens), gate)
        node_update, edge_update = torch.split(
            updated, [1, updated.shape[1] - 1], dim=1
        )

        if self.expanded_node_features:
            node = node + self.center_expansion(node_update)
            node = node + self.center_mlp(self.norm_center_features(node))
        else:
            node = node_update

        edge = edge + edge_update
        edge = edge + self.mlp(self.norm_mlp(edge))
        return node, edge


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, n_heads, dim_node_features, dim_feedforward)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            node, edge = layer(node, edge, gate)
        return node, edge


class SphericalHarmonics(nn.Module):
    """Real spherical harmonics up to ``l_max``, in metatrain's (sphericart) order."""

    def __init__(self, l_max: int) -> None:
        super().__init__()
        self.l_max = l_max
        self.register_buffer("F", torch.empty((l_max + 1) * (l_max + 2) // 2))
        for l in range(l_max + 1):  # noqa: E741
            for m in range(l + 1):
                self.F[l * (l + 1) // 2 + m] = (-1) ** m * math.sqrt(
                    (2 * l + 1)
                    / (2 * math.pi)
                    * math.factorial(l - m)
                    / math.factorial(l + m)
                )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """:param xyz: ``(n, 3)`` nonzero vectors. :return: ``(n, (l_max + 1)^2)``."""
        xyz = xyz / torch.sqrt((xyz**2).sum(dim=1, keepdim=True))
        x, y, z = xyz.unbind(1)
        L = self.l_max

        def lm(l: int, m: int) -> int:  # noqa: E741
            return l * (l + 1) // 2 + m

        Q = [torch.ones_like(z)] + [z] * (lm(L, L) if L else 0)
        for l in range(1, L + 1):  # noqa: E741
            Q[lm(l, l)] = -(2 * l - 1) * Q[lm(l - 1, l - 1)]
            Q[lm(l, l - 1)] = -z * Q[lm(l, l)]
            for m in range(l - 1):
                Q[lm(l, m)] = (
                    (2 * l - 1) * z * Q[lm(l - 1, m)] - (l + m - 1) * Q[lm(l - 2, m)]
                ) / (l - m)

        s, c = [torch.zeros_like(x)], [torch.ones_like(x)]
        for m in range(1, L + 1):
            s.append(x * s[m - 1] + y * c[m - 1])
            c.append(x * c[m - 1] - y * s[m - 1])

        Y = []
        for l in range(L + 1):  # noqa: E741
            Y += [self.F[lm(l, -m)] * Q[lm(l, -m)] * s[-m] for m in range(-l, 0)]
            Y.append(self.F[lm(l, 0)] * Q[lm(l, 0)] / math.sqrt(2.0))
            Y += [self.F[lm(l, m)] * Q[lm(l, m)] * c[m] for m in range(1, l + 1)]
        return torch.stack(Y, dim=1)


class CartesianTransformer(nn.Module):
    """One GNN layer: build edge tokens from geometry, then attend.

    The geometry is embedded either linearly from the edge vector and length, or,
    with ``geometry_l_max``, from the length-scaled spherical harmonics followed by
    a layer norm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int,
        n_layers: int,
        n_species: int,
        is_first: bool,
        geometry_l_max: int | None = None,
    ) -> None:
        super().__init__()
        self.is_first = is_first
        self.trans = Transformer(
            d_model, n_layers, n_heads, dim_node_features, dim_feedforward
        )
        self.geometry_l_max = geometry_l_max
        if geometry_l_max is None:
            self.edge_embedder = nn.Linear(4, d_model)
        else:
            self.spherical_harmonics = SphericalHarmonics(geometry_l_max)
            self.register_buffer(
                "geometry_norm_factors",
                torch.tensor(
                    [
                        math.sqrt(4 * math.pi / (2 * l + 1))
                        for l in range(geometry_l_max + 1)  # noqa: E741
                        for _ in range(2 * l + 1)
                    ]
                ),
            )
            self.edge_embedder = nn.Linear((geometry_l_max + 1) ** 2, d_model)
            self.rmsnorm = nn.LayerNorm(d_model)
        n_merge = 2 if is_first else 3
        self.compress = nn.Sequential(
            nn.Linear(n_merge * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        if not is_first:
            self.neighbor_embedder = nn.Embedding(n_species, d_model)

        # Filled by fold_compress() once the weights are loaded; see there.
        self.folded = False

    def fold_compress(self) -> None:
        """Precompute the compression of each input stream separately.

        ``compress[0]`` is a linear map over three concatenated ``d_model`` blocks,
        so it splits into three independent maps whose outputs are summed --
        ``W [a;b;c] = W_a a + W_b b + W_c c``. That is worth doing three times over:

        * the concatenation itself disappears, and with it a ``3 * d_model`` wide
          intermediate that is pure memory traffic;
        * ``W_a`` composed with the geometry embedder is another linear map, so the
          two collapse into a single ``4 -> d_model`` matrix (unless the embedding
          ends in a layer norm, which leaves ``W_a`` on its own);
        * ``W_b`` composed with the neighbor-species embedding is just that
          embedding table with its rows mapped, so it collapses into a table.

        What is left is one ``d_model -> d_model`` map on the messages, a tiny map
        on the geometry, and a lookup -- a third of the original matmul.

        Done after loading rather than in ``__init__`` so that the module tree
        still matches metatrain's and the weights still load with ``strict=True``.
        """
        d_model = self.compress[0].out_features
        dtype = self.compress[0].weight.dtype
        # Composing two matrices loses precision that evaluating them in sequence
        # does not, and the parameters here are float32 when the calculator runs
        # in it. Folding in float64 and casting back keeps the composition faithful;
        # doing it in float32 costs about five decimal digits in the final output.
        with torch.no_grad():
            weight = self.compress[0].weight.double()
            offset = 0

            # Geometry: compress[0] restricted to the edge block, composed with the
            # 4 -> d_model embedder.
            block = weight[:, offset : offset + d_model]
            bias = self.compress[0].bias.double()
            if self.geometry_l_max is None:
                bias = block @ self.edge_embedder.bias.double() + bias
                block = block @ self.edge_embedder.weight.double()
            self.register_buffer("folded_geometry_weight", block.contiguous().to(dtype))
            self.register_buffer("folded_geometry_bias", bias.to(dtype))
            offset += d_model

            if not self.is_first:
                block = weight[:, offset : offset + d_model]
                self.register_buffer(
                    "folded_species_table",
                    (self.neighbor_embedder.weight.double() @ block.T).to(dtype),
                )
                offset += d_model

            self.register_buffer(
                "folded_message_weight",
                weight[:, offset : offset + d_model].contiguous().to(dtype),
            )
        self.folded = True

    def forward(
        self,
        node: torch.Tensor,
        messages: torch.Tensor,
        neighbor_species: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_distances: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All neighbor arguments are in grid layout ``(n_nodes, n_neighbors, ...)``."""
        if self.geometry_l_max is None:
            geometry = torch.cat([edge_vectors, edge_distances[:, :, None]], dim=2)
        else:
            # Padded slots hold zero vectors, which have no direction; any unit
            # vector does, since the attention never reads those slots.
            unit_z = torch.zeros_like(edge_vectors)
            unit_z[..., 2] = 1.0
            safe = torch.where(
                (edge_vectors == 0).all(dim=2, keepdim=True), unit_z, edge_vectors
            )
            harmonics = self.spherical_harmonics(safe.reshape(-1, 3)).reshape(
                edge_vectors.shape[:2] + (-1,)
            )
            geometry = self.rmsnorm(
                self.edge_embedder(
                    edge_distances[:, :, None] * harmonics * self.geometry_norm_factors
                )
            )

        if self.folded:
            edge_tokens = torch.nn.functional.linear(
                geometry, self.folded_geometry_weight, self.folded_geometry_bias
            )
            edge_tokens = edge_tokens + torch.nn.functional.linear(
                messages, self.folded_message_weight
            )
            if not self.is_first:
                edge_tokens = edge_tokens + self.folded_species_table[neighbor_species]
            edge_tokens = self.compress[2](self.compress[1](edge_tokens))
        else:
            edge_embedding = (
                self.edge_embedder(geometry)
                if self.geometry_l_max is None
                else geometry
            )
            if self.is_first:
                tokens = torch.cat([edge_embedding, messages], dim=2)
            else:
                tokens = torch.cat(
                    [
                        edge_embedding,
                        self.neighbor_embedder(neighbor_species),
                        messages,
                    ],
                    dim=2,
                )
            edge_tokens = self.compress(tokens)

        node_out, edge_out = self.trans(node[:, None, :], edge_tokens, gate)
        return node_out.squeeze(1), edge_out


class SystemConditioningEmbedding(nn.Module):
    """Per-system charge and spin multiplicity, embedded and broadcast to atoms.

    The result is added to the node features after **every** GNN layer.

    :param d_out: output dimension, equal to ``d_node``.
    :param max_charge: supports charges in ``[-max_charge, max_charge]``.
    :param max_spin_multiplicity: supports ``2S + 1`` in ``[1, max_spin_multiplicity]``.
    """

    def __init__(self, d_out: int, max_charge: int, max_spin_multiplicity: int) -> None:
        super().__init__()
        self.max_charge = max_charge
        self.max_spin_multiplicity = max_spin_multiplicity
        self.charge_embedding = nn.Embedding(2 * max_charge + 1, d_out)
        self.spin_multiplicity_embedding = nn.Embedding(max_spin_multiplicity, d_out)
        self.project = nn.Sequential(
            nn.Linear(2 * d_out, d_out), nn.SiLU(), nn.Linear(d_out, d_out)
        )

    def forward(
        self,
        charge: torch.Tensor,
        spin_multiplicity: torch.Tensor,
        system_indices: torch.Tensor,
    ) -> torch.Tensor:
        if bool((charge < -self.max_charge).any() or (charge > self.max_charge).any()):
            raise ValueError(
                f"charge must be in [{-self.max_charge}, {self.max_charge}], "
                f"got {charge.tolist()}"
            )
        if bool(
            (spin_multiplicity < 1).any()
            or (spin_multiplicity > self.max_spin_multiplicity)
        ):
            raise ValueError(
                f"spin_multiplicity must be in [1, {self.max_spin_multiplicity}], "
                f"got {spin_multiplicity.tolist()}"
            )
        charge_embedded = self.charge_embedding(charge + self.max_charge)
        spin_embedded = self.spin_multiplicity_embedding(spin_multiplicity - 1)
        per_system = self.project(torch.cat([charge_embedded, spin_embedded], dim=-1))
        return per_system[system_indices]


class Readout(nn.Module):
    """Affine readout, optionally with an independent map per central-atom species.

    Rows of a gated readout must arrive sorted by species, described by ``groups``:
    one ``(species, start, stop)`` per species present.
    """

    def __init__(self, d_in: int, d_out: int, n_groups: int | None) -> None:
        super().__init__()
        shape = (d_out, d_in) if n_groups is None else (n_groups, d_out, d_in)
        self.gated = n_groups is not None
        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape[:-1]))

    def forward(
        self,
        x: torch.Tensor,
        groups: list[tuple[int, int, int]],
        bias_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """:param bias_scale: ``(n_rows, 1)`` multiplier on the bias, if any."""
        if not self.gated:
            out = torch.nn.functional.linear(x, self.weight)
            bias = self.bias
            return out + (bias if bias_scale is None else bias * bias_scale)
        out = x.new_empty(x.shape[0], self.weight.shape[1])
        for group, start, stop in groups:
            bias = self.bias[group]
            if bias_scale is not None:
                bias = bias * bias_scale[start:stop]
            out[start:stop] = (
                torch.nn.functional.linear(x[start:stop], self.weight[group]) + bias
            )
        return out


class TargetHead(nn.Module):
    """Heads and readouts of one atomic-basis target on the shared backbone features.

    The readouts come as ``members`` sets (a shallow ensemble, or one) reading
    ``head_sets`` head sets (one per member, or one shared); each set has
    ``heads_per_layer`` heads, and block ``j`` reads head ``j % heads_per_layer``.
    """

    def __init__(self, hypers: dict, target: dict, n_species: int) -> None:
        super().__init__()
        layout = target["layout"]
        self.blocks = target["blocks"]

        def head(d_in: int, d_out: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Linear(d_in, d_out), nn.SiLU()]
            for _ in range(hypers["num_head_layers"] - 1):
                layers += [nn.Linear(d_out, d_out), nn.SiLU()]
            return nn.Sequential(*layers)

        n_groups = n_species if layout["readout_gated"] else None
        self.heads_per_layer = layout["heads_per_layer"]
        self.node_heads, self.edge_heads = (
            nn.ModuleList(
                [
                    nn.ModuleList(
                        [head(d_in, d_head) for _ in range(self.heads_per_layer)]
                    )
                    for _ in range(layout["head_sets"])
                ]
            )
            for d_in, d_head in (
                (hypers["d_node"], hypers["d_head_node"]),
                (hypers["d_pet"], hypers["d_head_edge"]),
            )
        )
        self.node_readouts, self.edge_readouts = (
            nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            Readout(d_head, b["n_m"] * b["n_p"], n_groups)
                            for b in self.blocks
                        ]
                    )
                    for _ in range(layout["members"])
                ]
            )
            for d_head in (hypers["d_head_node"], hypers["d_head_edge"])
        )

    @property
    def gated(self) -> bool:
        return self.node_readouts[0][0].gated

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        centers: torch.Tensor,
        factors: torch.Tensor,
        weight_sums: torch.Tensor,
        groups: list[tuple[int, int, int]],
        max_chunk_tokens: int | None,
        features: bool = False,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """One ``(n_members, n_nodes, n_m * n_p)`` tensor per block and, with
        ``features``, metatrain's last-layer features: the node and the pooled
        edge features of each head of the first head set, head after head."""
        n_nodes = node.shape[0]
        node_hidden = [[head(node) for head in heads] for heads in self.node_heads]

        # Edges are pooled onto their centre before the readout, which is affine:
        # sum_j c_ij (W h_ij + b) = W (sum_j c_ij h_ij) + b sum_j c_ij. The edge
        # heads run over every edge, chunked like the backbone.
        pooled = [
            [edge.new_zeros(n_nodes, head[-2].out_features) for head in heads]
            for heads in self.edge_heads
        ]
        for edges in _chunks(edge.shape[0], max_chunk_tokens):
            chunk, weights, chunk_centers = edge[edges], factors[edges], centers[edges]
            for heads, sums in zip(self.edge_heads, pooled, strict=True):
                for head, total in zip(heads, sums, strict=True):
                    total.index_add_(0, chunk_centers, head(chunk) * weights)

        members = []
        for m, (node_readouts, edge_readouts) in enumerate(
            zip(self.node_readouts, self.edge_readouts, strict=True)
        ):
            s = m if len(self.node_heads) > 1 else 0
            members.append(
                [
                    node_readout(node_hidden[s][j % self.heads_per_layer], groups)
                    + edge_readout(
                        pooled[s][j % self.heads_per_layer], groups, weight_sums
                    )
                    for j, (node_readout, edge_readout) in enumerate(
                        zip(node_readouts, edge_readouts, strict=True)
                    )
                ]
            )
        blocks = [torch.stack(per_member) for per_member in zip(*members, strict=True)]
        if not features:
            return blocks, None
        heads = zip(node_hidden[0], pooled[0], strict=True)
        return blocks, torch.cat([torch.cat(pair, dim=-1) for pair in heads], dim=-1)


class PetInference(nn.Module):
    """PET backbone with one :class:`TargetHead` per target, inference only.

    :param artifact: the dict written by :func:`qpet.backends.fast.export.export`.
    """

    def __init__(self, artifact: dict) -> None:
        super().__init__()
        hypers = artifact["hypers"]
        self.cutoff = hypers["cutoff"]
        self.cutoff_width = hypers["cutoff_width"]
        self.cutoff_width_adaptive = hypers["cutoff_width_adaptive"]
        self.num_neighbors_adaptive = hypers["num_neighbors_adaptive"]
        self.num_gnn_layers = hypers["num_gnn_layers"]

        d_pet = hypers["d_pet"]
        d_node = hypers["d_node"]
        n_species = len(artifact["atomic_types"])

        self.register_buffer(
            "species_to_species_index",
            torch.full((max(artifact["atomic_types"]) + 1,), -1, dtype=torch.long),
        )
        for index, species in enumerate(artifact["atomic_types"]):
            self.species_to_species_index[species] = index

        self.gnn_layers = nn.ModuleList(
            [
                CartesianTransformer(
                    d_pet,
                    hypers["num_heads"],
                    d_node,
                    hypers["d_feedforward"],
                    hypers["num_attention_layers"],
                    n_species,
                    is_first=(i == 0),
                    geometry_l_max=hypers["geometry_embedding_l_max"],
                )
                for i in range(hypers["num_gnn_layers"])
            ]
        )
        # NOTE: these are LayerNorm regardless of the `normalization` hyper --
        # metatrain hardcodes them (backend.py), unlike every other norm in PET.
        self.combination_norms = nn.ModuleList(
            [nn.LayerNorm(2 * d_pet) for _ in range(hypers["num_gnn_layers"])]
        )
        self.combination_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d_pet, 2 * d_pet),
                    nn.SiLU(),
                    nn.Linear(2 * d_pet, d_pet),
                )
                for _ in range(hypers["num_gnn_layers"])
            ]
        )
        # "feedforward" featurization reads out from the last GNN layer only.
        self.node_embedders = nn.ModuleList([nn.Embedding(n_species, d_node)])
        self.edge_embedder = nn.Embedding(n_species, d_pet)

        # Set by the calculator; None processes everything in one piece.
        self.max_chunk_tokens: int | None = None
        self.has_system_conditioning = hypers["system_conditioning"]
        if self.has_system_conditioning:
            self.system_conditioning = SystemConditioningEmbedding(
                d_node, hypers["max_charge"], hypers["max_spin_multiplicity"]
            )

        self.target_names = list(artifact["targets"])
        self.targets = nn.ModuleDict(
            {
                _module_name(name): TargetHead(hypers, target, n_species)
                for name, target in artifact["targets"].items()
            }
        )

        # The artifact is float64, and folded readouts use all of it.
        self.double()
        state = dict(artifact["backbone"])
        for name, target in artifact["targets"].items():
            prefix = f"targets.{_module_name(name)}."
            state.update({prefix + k: v for k, v in target["state"].items()})
        self.load_state_dict(state, strict=True)
        self.eval()

    def fold(self) -> None:
        """Precompute the folded compression weights on every GNN layer.

        Called by the calculator *after* the model is moved to its final device and
        dtype -- folding is a matrix composition, so it has to happen at the
        precision the model will run at, not at whatever the tree was built with.
        """
        for layer in self.gnn_layers:
            layer.fold_compress()

    def set_attention_backend(self, backend: str | None) -> None:
        """Pin every attention block to ``"triton"`` or ``"sdpa"``; ``None`` is auto."""
        if backend not in (None, "triton", "sdpa"):
            raise ValueError(
                f"unknown attention backend {backend!r}; use 'triton', 'sdpa' or None"
            )
        for module in self.modules():
            if isinstance(module, AttentionBlock):
                module.attention_backend = backend

    @staticmethod
    def _to_grid(flat: torch.Tensor, bucket: Bucket) -> torch.Tensor:
        """Gather a flat per-edge array into one bucket's grid, zeroing padded slots."""
        return PetInference._to_grid_rows(flat, bucket.edge_indices, bucket.mask)

    @staticmethod
    def _to_grid_rows(
        flat: torch.Tensor, edge_indices: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Same, for an arbitrary slice of a bucket's rows."""
        grid = flat[edge_indices]
        shaped = mask.reshape(mask.shape + (1,) * (flat.dim() - 1))
        return torch.where(shaped, grid, torch.zeros_like(grid))

    def features(
        self,
        batch: Batch,
        charge: torch.Tensor | None = None,
        spin_multiplicity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GNN.

        :param charge: per-system total charge, only used when the model was
            trained with system conditioning.
        :param spin_multiplicity: per-system ``2S + 1``, likewise.
        :return: ``(node_features, edge_features)``; node features are
            ``(n_nodes, d_node)`` and edge features are flat ``(n_edges, d_pet)``.
        """
        conditioning: torch.Tensor | None = None
        if self.has_system_conditioning:
            # One system per call, so every atom maps to system 0.
            system_indices = torch.zeros(
                batch.species_index.shape[0],
                dtype=torch.long,
                device=batch.species_index.device,
            )
            conditioning = self.system_conditioning(
                charge, spin_multiplicity, system_indices
            )

        neighbor_species = batch.species_index[batch.neighbors]
        messages = self.edge_embedder(neighbor_species)
        node = self.node_embedders[0](batch.species_index)

        # Geometry does not change between GNN layers, so each bucket's static
        # grids are built once here; only the messages are re-gridded per layer.
        geometry = []
        for bucket in batch.buckets:
            edge_vectors = self._to_grid(batch.edge_vectors, bucket)
            # PET recomputes the distance from the gridded vectors, with this epsilon.
            edge_distances = torch.sqrt((edge_vectors**2).sum(dim=2) + 1e-15)
            cutoff_grid = self._to_grid(batch.cutoff_factors, bucket)
            # The central token is never cut off; padded neighbors are fully
            # suppressed. Built explicitly rather than by slicing cutoff_grid,
            # which is zero-width for an isolated atom.
            central = torch.ones(
                cutoff_grid.shape[0],
                1,
                dtype=cutoff_grid.dtype,
                device=cutoff_grid.device,
            )
            geometry.append(
                (
                    edge_vectors,
                    edge_distances,
                    self._to_grid(neighbor_species, bucket),
                    torch.cat([central, cutoff_grid], dim=1),
                )
            )

        for layer, norm, mlp in zip(
            self.gnn_layers, self.combination_norms, self.combination_mlps, strict=True
        ):
            # The scatter buffers are allocated from what the layer actually
            # returns, not from the incoming features: under autocast the layer
            # emits half precision while the embeddings that seed `node` and
            # `messages` are full precision, and an index-put will not cross that.
            updated_node: torch.Tensor | None = None
            outgoing: torch.Tensor | None = None
            for bucket, (vectors, distances, species, gate) in zip(
                batch.buckets, geometry, strict=True
            ):
                # Rows per chunk follow from the token budget and this grid's
                # width, so a wide grid is split more finely than a narrow one.
                width = bucket.mask.shape[1] + 1
                max_rows = (
                    None
                    if self.max_chunk_tokens is None
                    else max(1, self.max_chunk_tokens // width)
                )
                for rows in _chunks(bucket.nodes.shape[0], max_rows):
                    nodes = bucket.nodes[rows]
                    indices = bucket.edge_indices[rows]
                    mask = bucket.mask[rows]
                    node_out, edge_grid = layer(
                        node.index_select(0, nodes),
                        self._to_grid_rows(messages, indices, mask),
                        species[rows],
                        vectors[rows],
                        distances[rows],
                        gate[rows],
                    )
                    if updated_node is None:
                        updated_node = torch.empty(
                            node.shape[0],
                            node_out.shape[-1],
                            dtype=node_out.dtype,
                            device=node_out.device,
                        )
                        outgoing = torch.empty(
                            messages.shape[0],
                            edge_grid.shape[-1],
                            dtype=edge_grid.dtype,
                            device=edge_grid.device,
                        )
                    updated_node[nodes] = node_out
                    outgoing[indices[mask]] = edge_grid[mask]
            assert updated_node is not None and outgoing is not None

            node = updated_node
            if conditioning is not None:
                node = node + conditioning

            # The message combination runs on the flat edge array, so the grid
            # chunking above does not cover it, and it is where the memory goes:
            # every full-width (n_edges, d_pet) array costs 1.3 GB at sixty
            # thousand atoms. Gathering the reversed messages per chunk, and
            # accumulating into `messages` in place, keeps two such arrays live
            # instead of four. Reading `outgoing` while writing `messages` is safe
            # -- they are different buffers.
            for edges in _chunks(outgoing.shape[0], self.max_chunk_tokens):
                forward = outgoing[edges]
                reversed_ = outgoing[batch.corresponding_edges[edges]]
                messages[edges] = (
                    messages[edges]
                    + forward
                    + mlp(norm(torch.cat([forward, reversed_], dim=-1)))
                )

        return node, messages

    def forward(
        self,
        batch: Batch,
        charge: torch.Tensor | None = None,
        spin_multiplicity: torch.Tensor | None = None,
        autocast_dtype: torch.dtype | None = None,
        names: list[str] | None = None,
        features: list[str] | None = None,
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor]]:
        """
        :param autocast_dtype: run the GNN backbone in this reduced precision. The
            heads and readouts stay in the model's own dtype: they are cheap
            (a few percent of the total) and carry the whole output range, so
            keeping them exact roughly halves the error for almost no time.
        :param names: targets to evaluate, all by default.
        :param features: targets whose last-layer features to return as well.
        :return: per target, one ``(n_members, n_nodes, n_m * n_p)`` tensor per
            block, in the order the blocks are listed in the artifact; and the
            last-layer features of the targets in ``features``.
        """
        if autocast_dtype is None:
            node, edge = self.features(batch, charge, spin_multiplicity)
        else:
            device_type = batch.edge_vectors.device.type
            with torch.autocast(device_type, dtype=autocast_dtype):
                node, edge = self.features(batch, charge, spin_multiplicity)
            full = batch.edge_vectors.dtype
            node, edge = node.to(full), edge.to(full)

        # A gated readout works on contiguous runs of one species, so the atoms
        # are put in species order for the readouts and back at the end.
        centers = batch.centers
        groups: list[tuple[int, int, int]] = []
        rank: torch.Tensor | None = None
        if any(target.gated for target in self.targets.values()):
            order = torch.argsort(batch.species_index, stable=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(order.shape[0], device=order.device)
            node = node.index_select(0, order)
            centers = rank[centers]
            counts = torch.bincount(batch.species_index).tolist()
            start = 0
            for group, count in enumerate(counts):
                if count:
                    groups.append((group, start, start + count))
                start += count

        factors = batch.cutoff_factors[:, None]
        weight_sums = factors.new_zeros(node.shape[0], 1).index_add(0, centers, factors)
        out: dict[str, list[torch.Tensor]] = {}
        last_layer: dict[str, torch.Tensor] = {}
        features = [] if features is None else features
        for name in self.target_names if names is None else names:
            blocks, llf = self.targets[_module_name(name)](
                node,
                edge,
                centers,
                factors,
                weight_sums,
                groups,
                self.max_chunk_tokens,
                name in features,
            )
            out[name] = [b if rank is None else b[:, rank] for b in blocks]
            if llf is not None:
                last_layer[name] = llf if rank is None else llf[rank]
        return out, last_layer


def _module_name(target: str) -> str:
    """``nn.ModuleDict`` keys cannot contain dots; target names have ``::``."""
    return target.replace(".", "_").replace(":", "_")
