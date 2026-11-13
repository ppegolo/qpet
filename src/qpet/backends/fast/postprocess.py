"""Post-processing: raw PET blocks -> PySCF-ordered auxiliary-basis coefficients.

Two halves, split along the dependency boundary:

* :func:`build_postprocess_constants` runs once, inside the exporter, and does all
  the metatensor work: it recovers the scaler and the composition baseline by
  probing the real modules, and derives the validity mask and the PySCF
  permutation from the target's layout.
* :func:`postprocess` runs per evaluation and is four tensor ops on plain torch
  tensors -- no ``TensorMap``, no ``Labels``, no device synchronisation.

The ordering convention is metatrain's
:func:`metatrain.utils.loss._flatten_to_pyscf_order`, the same one the
``density_mse_via_c`` loss trains against: blocks by ascending ``o3_lambda``,
``l = 1`` components reordered to PySCF's ``(x, y, z)``, and the radial index
major within an angular channel.
"""

import torch


def build_postprocess_constants(
    model, target_name: str, multipliers: list[torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    """Bake the scaler, composition baseline and PySCF layout into plain tensors.

    :param model: a loaded ``metatrain.pet.PET``, in float64.
    :param target_name: the atomic-basis target.
    :return: the constants consumed by :func:`postprocess`:
        ``scales`` ``(n_types, n_wide)``, ``baseline`` ``(n_types, n_wide)``,
        ``valid`` ``(n_types, n_wide)`` bool, ``perm`` ``(n_wide,)``,
        ``type_index`` ``(max_type + 1,)``, and ``block_sizes``.
    """
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatomic.torch import ModelOutput, System

    layout = model.dataset_info.targets[target_name].layout
    shapes = model.output_shapes[target_name]
    types = sorted(model.atomic_types)
    n_types = len(types)

    type_pos = [i for i, n in enumerate(layout.keys.names) if n.endswith("atom_type")]
    other_pos = [
        i for i, n in enumerate(layout.keys.names) if not n.endswith("atom_type")
    ]

    # Dense property axis per (lambda, sigma): the union over atomic types, exactly
    # as metatrain's densification builds it.
    union: dict[tuple, Labels] = {}
    for key, block in layout.items():
        k = tuple(int(key.values[i]) for i in other_pos)
        union[k] = (
            block.properties if k not in union else union[k].union(block.properties)
        )

    lam_sigma, dense_cols, valid_cols, factor_cols = [], [], [], []
    for dict_key, shape in shapes.items():
        lam = int(dict_key.rsplit("o3_lambda_", 1)[1].split("_")[0])
        sigma = int(dict_key.rsplit("o3_sigma_", 1)[1])
        n_m = shape[0] if len(shape) > 1 else 1
        props = union[(lam, sigma)]
        n_p = len(props)
        lam_sigma.append((lam, sigma))
        dense_cols.append((n_m, n_p))

        valid = torch.zeros(n_types, n_m, n_p, dtype=torch.bool)
        factor = torch.ones(n_types, n_m, n_p, dtype=torch.float64)
        for t_i, t in enumerate(types):
            for position, (key, block) in enumerate(layout.items()):
                irrep = tuple(int(key.values[i]) for i in other_pos)
                same_irrep = irrep == (lam, sigma)
                same_type = [int(key.values[i]) for i in type_pos] == [t]
                if same_irrep and same_type:
                    valid[t_i, :, props.select(block.properties)] = True
                    if multipliers is not None:
                        factor[t_i, :, props.select(block.properties)] = multipliers[
                            position
                        ].to(torch.float64)
        valid_cols.append(valid.reshape(n_types, -1))
        factor_cols.append(factor.reshape(n_types, -1))

    probe = System(
        types=torch.tensor(types, dtype=torch.int32),
        positions=torch.zeros(n_types, 3, dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False] * 3),
    )

    def _constant_tensormap(fill: float) -> "TensorMap":
        blocks = []
        samples = Labels(
            ["system", "atom"],
            torch.stack(
                [
                    torch.zeros(n_types, dtype=torch.int32),
                    torch.arange(n_types, dtype=torch.int32),
                ],
                dim=1,
            ),
        )
        for (_, shape), comps, props in zip(
            shapes.items(),
            model.component_labels[target_name],
            model.property_labels[target_name],
            strict=True,
        ):
            blocks.append(
                TensorBlock(
                    values=torch.full(
                        [n_types] + list(shape), fill, dtype=torch.float64
                    ),
                    samples=samples,
                    components=comps,
                    properties=props,
                )
            )
        return TensorMap(model.key_labels[target_name], blocks)

    # Scaler: a multiplier per property *and per atomic type* (metatrain scales
    # per-atom targets separately for each type). The probe holds one atom of each
    # type in order, so scaling a tensor of ones reads the whole table off at once.
    scaled = model.scaler.apply_scales(
        [probe],
        {target_name: _constant_tensormap(1.0)},
        use_per_target_scales=True,
        use_per_property_scales=True,
    )[target_name]
    scales = torch.cat([b.values.reshape(n_types, -1) for b in scaled.blocks()], dim=1)

    # Composition: a per-atom-type constant. It exists only for the invariant
    # (lambda = 0) channel -- a constant cannot be nonzero for lambda > 0 -- so
    # every other block contributes a zero baseline.
    composition = model.additive_models[0]
    was_training = composition.training
    composition.train()  # train mode keeps the composition output dense
    with torch.no_grad():
        base = composition(
            [probe], {target_name: ModelOutput(sample_kind="atom")}, None
        )[target_name]
    composition.train(was_training)
    base_by_irrep = {
        tuple(int(v) for v in key.values): block.values.reshape(n_types, -1)
        for key, block in base.items()
    }
    baseline = torch.cat(
        [
            base_by_irrep.get(
                irrep, torch.zeros(n_types, n_m * n_p, dtype=torch.float64)
            )
            for irrep, (n_m, n_p) in zip(lam_sigma, dense_cols, strict=True)
        ],
        dim=1,
    )

    # Dense -> PySCF permutation.
    pieces, offset = [], 0
    for (lam, _), (n_m, n_p) in zip(lam_sigma, dense_cols, strict=True):
        index = torch.arange(offset, offset + n_m * n_p).reshape(n_m, n_p)
        if lam == 1:
            # metatensor orders l=1 as m = -1, 0, +1; PySCF uses x, y, z.
            index = index[[2, 0, 1], :]
        # (m, n) -> (n, m): the radial index is major within an angular channel.
        pieces.append((lam, index.transpose(0, 1).reshape(-1)))
        offset += n_m * n_p
    perm = torch.cat([idx for _, idx in sorted(pieces, key=lambda p: p[0])])

    type_index = torch.full((max(types) + 1,), -1, dtype=torch.long)
    for i, t in enumerate(types):
        type_index[t] = i

    valid_mask = torch.cat(valid_cols, dim=1)

    # Padded slots -- properties an element does not own -- carry NaN in the
    # composition baseline, because that is the fill value densification uses.
    # They are dropped by the validity mask, so they never reach the output, but
    # leaving NaN in the tables means every intermediate carries NaN and any future
    # reduction placed before the mask would silently poison everything. Replace
    # them with the identity of each operation. A NaN in a *live* slot would be a
    # real defect, so that is an error rather than something to paper over.
    for name, table in (("scales", scales), ("baseline", baseline)):
        live_nan = int((torch.isnan(table) & valid_mask).sum())
        if live_nan:
            raise ValueError(
                f"{live_nan} NaN entries in the {name} table at live (unmasked) "
                "slots; the checkpoint's post-processing weights are incomplete"
            )
    scales = torch.where(valid_mask, scales, torch.ones_like(scales))
    llpr_scales = scales * torch.cat(factor_cols, dim=1)
    baseline = torch.where(valid_mask, baseline, torch.zeros_like(baseline))

    return {
        "scales": scales,
        "llpr_scales": llpr_scales,
        "baseline": baseline,
        "valid": valid_mask,
        "perm": perm,
        "type_index": type_index,
        "block_sizes": [n_m * n_p for n_m, n_p in dense_cols],
    }


def postprocess(
    raw_blocks: list[torch.Tensor],
    atom_types: torch.Tensor,
    constants: dict[str, torch.Tensor],
    deviations: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw per-block PET output -> flat PySCF-ordered coefficients, per member.

    With ``deviations``, the blocks are LLPR deviations: they take the scales and
    the calibration multipliers, and no baseline.

    :param raw_blocks: one ``(n_members, n_atoms, n_m * n_p)`` tensor per block, in
        the order the model emits them.
    :param atom_types: ``(n_atoms,)`` atomic numbers.
    :param constants: output of :func:`build_postprocess_constants`.
    :return: ``(coefficients, counts_per_atom)``. ``coefficients`` is
        ``(n_coefficients, n_members)``, atom-major and in PySCF order within each
        atom; ``counts_per_atom`` gives the per-atom lengths.
    """
    rows = constants["type_index"][atom_types.long()]
    wide = torch.cat([b.reshape(b.shape[0], b.shape[1], -1) for b in raw_blocks], -1)
    if deviations:
        wide = wide * constants["llpr_scales"][rows]
    else:
        wide = wide * constants["scales"][rows] + constants["baseline"][rows]
    wide = wide[..., constants["perm"]]
    mask = constants["valid"][rows][:, constants["perm"]]
    return wide[:, mask].T, mask.sum(dim=1)
