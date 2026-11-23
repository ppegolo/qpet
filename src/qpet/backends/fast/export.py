"""Freeze a metatrain PET checkpoint into a self-contained inference artifact.

This is the only module that imports metatrain. It runs once per trained model
and writes a plain ``torch.save`` dict holding the weights, the handful of
hyperparameters the inference path needs, the auxiliary-basis block layout, and
the baked post-processing constants (scaler, composition baseline, validity mask,
PySCF permutation).

Keeping metatrain on this side of the boundary is deliberate: PET checkpoints are
at format version 19 with eighteen upgrade steps behind them, and metatrain's own
loader is the only thing that should ever have to know about that. The runtime
package reads the artifact and depends on torch alone.
"""

import math
from typing import Any

import torch

ARTIFACT_FORMAT = 3

#: Hyperparameter values this inference package reproduces. Anything else raises:
#: a reimplementation that silently computes the wrong thing is the one failure
#: mode a reimplementation has that ``torch.compile`` does not.
_REQUIRED_HYPERS = {
    "featurizer_type": "feedforward",
    "transformer_type": "PreLN",
    "normalization": "RMSNorm",
    "activation": "SwiGLU",
    "cutoff_function": "Bump",
    "adaptive_cutoff_method": "solver",
    "zbl": False,
    "attention_temperature": 1.0,
}

#: Backend modules that hold one target's heads and readouts.
_PER_TARGET_MODULES = (
    "node_heads",
    "edge_heads",
    "node_last_layers",
    "edge_last_layers",
    "node_heads_ensemble",
    "edge_heads_ensemble",
    "node_last_layers_ensemble_readout",
    "edge_last_layers_ensemble_readout",
    "node_last_layers_ensemble_head",
    "edge_last_layers_ensemble_head",
)


class UnsupportedModelError(RuntimeError):
    """The checkpoint uses a feature this inference package does not implement."""


def _check_supported(hypers: dict[str, Any], target_info) -> None:
    problems: list[str] = []

    for key, expected in _REQUIRED_HYPERS.items():
        got = hypers.get(key)
        if got != expected:
            problems.append(f"{key}={got!r} (supported: {expected!r})")

    if hypers.get("num_neighbors_adaptive") is None:
        problems.append(
            "num_neighbors_adaptive=None (only the adaptive-cutoff path is supported)"
        )
    if hypers.get("long_range", {}).get("enable", False):
        problems.append("long_range.enable=True (not supported)")
    if hypers.get("atomic_charge_conditioning"):
        problems.append("atomic_charge_conditioning=True (not supported)")
    if hypers.get("conservation") is not None:
        problems.append("conservation (not supported)")

    if not getattr(target_info, "is_spherical", False):
        problems.append("target is not spherical")
    if not getattr(target_info, "is_atomic_basis", False):
        problems.append("target is not an atomic-basis target")
    if target_info.layout.sample_names != ["system", "atom"]:
        problems.append(f"target samples are {target_info.layout.sample_names}")

    if problems:
        raise UnsupportedModelError(
            "this checkpoint is outside the supported subset:\n  - "
            + "\n  - ".join(problems)
        )


def _readout_weights(readout, n_groups: int):
    """``(weight, bias)`` of a readout as one linear map, or one per atomic type.

    A mixture-of-experts readout routes on the central-atom type alone, so for each
    type it is a fixed combination of linear experts: a gated linear map.
    """
    if type(readout).__name__ == "LinearReadout":
        bias = readout.bias
        if bias is None:
            bias = torch.zeros(readout.weight.shape[:-1], dtype=readout.weight.dtype)
        return readout.weight * readout.scale, bias * readout.scale

    groups = torch.arange(n_groups)
    scores = torch.softmax(
        readout.routing_matrix(
            torch.nn.functional.silu(readout.group_embedding(groups))
        ),
        dim=-1,
    )
    top, index = torch.topk(scores, readout.num_topk, dim=-1)
    routing = torch.zeros_like(scores).scatter(-1, index, top)
    weight, bias = 0.0, 0.0
    for e, expert in enumerate(readout.routed_experts):
        w, b = _readout_weights(expert, n_groups)
        weight = weight + routing[:, e, None, None] * w
        bias = bias + routing[:, e, None] * b
    for expert in readout.shared_experts:
        w, b = _readout_weights(expert, n_groups)
        weight, bias = weight + w, bias + b
    return weight, bias


def _heads_and_readouts(backend, target_name: str):
    """Head and readout weights of one target, in the layout of the fast backend.

    Heads come as one list of head sets (one set, or one per ensemble member with
    ``scope="head"``), each holding ``heads_per_layer`` MLPs. Readouts come as one
    list of per-block maps per ensemble member (a single member without
    ensembling). Member ``m`` reads head set ``m`` when there is one per member.
    """
    scope = backend.shallow_ensemble_scope
    state: dict[str, torch.Tensor] = {}
    gated = set()

    for side in ("node", "edge"):
        if scope == "head":
            heads = getattr(backend, f"{side}_heads_ensemble")[target_name]
            head_sets = list(heads.values())
            per_layer = [
                member[0]
                for member in getattr(backend, f"{side}_last_layers_ensemble_head")[
                    target_name
                ].values()
            ]
        elif scope == "readout":
            head_sets = [getattr(backend, f"{side}_heads")[target_name]]
            layer = getattr(backend, f"{side}_last_layers_ensemble_readout")[
                target_name
            ][0]
            members = len(next(iter(layer.values())))
            per_layer = [
                {key: by_member[f"member_{e}"] for key, by_member in layer.items()}
                for e in range(members)
            ]
        else:
            head_sets = [getattr(backend, f"{side}_heads")[target_name]]
            per_layer = [getattr(backend, f"{side}_last_layers")[target_name][0]]

        for s, heads in enumerate(head_sets):
            for h, head in enumerate(heads):
                linears = [m for m in head if isinstance(m, torch.nn.Linear)]
                for k, linear in enumerate(linears):
                    state[f"{side}_heads.{s}.{h}.{2 * k}.weight"] = linear.weight
                    state[f"{side}_heads.{s}.{h}.{2 * k}.bias"] = linear.bias

        for m, readouts in enumerate(per_layer):
            for b, readout in enumerate(readouts.values()):
                weight, bias = _readout_weights(readout, backend.num_atomic_species)
                gated.add(weight.dim() == 3)
                state[f"{side}_readouts.{m}.{b}.weight"] = weight
                state[f"{side}_readouts.{m}.{b}.bias"] = bias

    assert len(gated) == 1
    layout = {
        "heads_per_layer": int(backend.heads_per_layer[target_name]),
        "head_sets": len(head_sets),
        "members": len(per_layer),
        "readout_gated": gated.pop(),
    }
    return state, layout


def load_model(checkpoint_path: str):
    """Load a metatrain PET checkpoint as a float64 model in eval mode.

    Hypers missing from the checkpoint take their inert defaults: checkpoints from
    other metatrain branches can carry a version number whose upgrade steps never
    added them.
    """
    from metatrain.pet import PET
    from metatrain.utils.architectures import get_default_hypers

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    checkpoint = PET.upgrade_checkpoint(checkpoint)
    for key, value in get_default_hypers("pet")["model"].items():
        checkpoint["model_data"]["model_hypers"].setdefault(key, value)
    return PET.load_checkpoint(checkpoint, "export").double().eval()


def export(
    checkpoint_path: str,
    artifact_path: str,
    targets: list[str] | None = None,
    llpr: str | None = None,
) -> dict[str, Any]:
    """Read a metatrain PET checkpoint and write the frozen inference artifact.

    :param checkpoint_path: ``.ckpt`` produced by ``mtt train``.
    :param artifact_path: where to write the artifact.
    :param targets: targets to export, all of them by default; they share the
        backbone and are evaluated in one pass.
    :param llpr: an exported metatrain LLPR model of the same checkpoint; its
        ensembles become the ``mtt::aux::<target>_ensemble`` outputs.
    :return: the artifact dict that was written.
    """
    from .postprocess import build_postprocess_constants

    model = load_model(checkpoint_path)
    known = model.dataset_info.targets
    targets = list(model.target_names) if targets is None else targets
    unknown = sorted(set(targets) - set(known))
    if unknown:
        raise ValueError(f"unknown targets {unknown}; have {list(known)}")

    hypers = dict(model.hypers)
    for name in targets:
        _check_supported(hypers, known[name])
    if model.backend.num_readout_layers != 1:
        raise UnsupportedModelError("only one readout layer is supported")

    d_head = hypers["d_head"]
    if isinstance(d_head, dict):
        d_head_node, d_head_edge = int(d_head["node"]), int(d_head["edge"])
    else:
        d_head_node = d_head_edge = int(d_head)

    ensembles = {} if llpr is None else _llpr_targets(llpr)
    exported = {}
    for name in targets:
        # Block layout, in the order the model emits its prediction blocks.
        blocks = []
        for dict_key, shape in model.output_shapes[name].items():
            lam = int(dict_key.rsplit("o3_lambda_", 1)[1].split("_")[0])
            sigma = int(dict_key.rsplit("o3_sigma_", 1)[1])
            n_m = shape[0] if len(shape) > 1 else 1
            blocks.append(
                {
                    "key": dict_key,
                    "o3_lambda": lam,
                    "o3_sigma": sigma,
                    "n_m": n_m,
                    "n_p": shape[-1],
                }
            )
        with torch.no_grad():
            state, layout = _heads_and_readouts(model.backend, name)
        exported[name] = {
            "blocks": blocks,
            "layout": layout,
            "state": {k: v.detach().cpu().contiguous() for k, v in state.items()},
            "post": build_postprocess_constants(
                model, name, ensembles.get(name, {}).pop("multipliers", None)
            ),
            "llpr": ensembles.get(name),
        }

    backbone = {
        k[len("backend.") :]: v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
        if k.startswith("backend.") and k.split(".")[1] not in _PER_TARGET_MODULES
    }
    artifact = {
        "format": ARTIFACT_FORMAT,
        "atomic_types": list(model.atomic_types),
        "hypers": {
            "cutoff": float(hypers["cutoff"]),
            "cutoff_width": float(hypers["cutoff_width"]),
            "cutoff_width_adaptive": float(hypers["cutoff_width_adaptive"]),
            "num_neighbors_adaptive": float(hypers["num_neighbors_adaptive"]),
            "d_pet": int(hypers["d_pet"]),
            "d_node": int(hypers["d_node"]),
            "d_head_node": d_head_node,
            "d_head_edge": d_head_edge,
            "num_head_layers": int(hypers["num_head_layers"]),
            "d_feedforward": int(hypers["d_feedforward"]),
            "num_heads": int(hypers["num_heads"]),
            "num_gnn_layers": int(hypers["num_gnn_layers"]),
            "num_attention_layers": int(hypers["num_attention_layers"]),
            "system_conditioning": bool(hypers["system_conditioning"]),
            "max_charge": int(hypers["max_charge"]),
            "max_spin_multiplicity": int(hypers["max_spin_multiplicity"]),
            "geometry_embedding_l_max": hypers.get("geometry_embedding_l_max"),
        },
        "backbone": backbone,
        "targets": exported,
    }
    torch.save(artifact, artifact_path)
    return artifact


def _llpr_targets(path: str) -> dict[str, dict[str, Any]]:
    """Ensemble weights of each target of an exported LLPR model, per atomic type
    and block, with the columns of the last-layer features and the properties each
    reads, and the calibration multipliers of each block of the target's layout."""
    from metatomic.torch import load_atomistic_model

    module = load_atomistic_model(path).module
    state = module.state_dict()
    out = {}
    for name, target in module.targets.named_children():
        prefix = f"targets.{name}."
        if not any(k.startswith(prefix + "ensemble_layers.") for k in state):
            continue
        n_blocks = len(target.feature_indices)
        columns = [
            target.columns[target.column_offsets[f] : target.column_offsets[f + 1]]
            for f in target.feature_indices
        ]
        weights, properties = [], []
        for g in range(len(target.atomic_types)):
            index = [g * n_blocks + j for j in range(n_blocks)]
            weights.append(
                [state[f"{prefix}ensemble_layers.{i}.weight"].double() for i in index]
            )
            properties.append(
                [
                    target.properties[
                        target.property_offsets[i] : target.property_offsets[i + 1]
                    ]
                    for i in index
                ]
            )
        offsets, flat = target.multiplier_offsets, state[prefix + "multipliers"]
        out[name] = {
            "atomic_types": list(target.atomic_types),
            "members": target.num_outputs[0]
            // math.prod(target.core_layout.block(0).values.shape[1:]),
            "columns": columns,
            "weights": weights,
            "properties": properties,
            "multipliers": [
                flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)
            ],
        }
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=export.__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("artifact")
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--llpr", help="exported LLPR model of the same checkpoint")
    args = parser.parse_args()
    export(args.checkpoint, args.artifact, args.targets, args.llpr)
