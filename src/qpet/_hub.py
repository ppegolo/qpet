"""Released models: downloaded from Hugging Face, exported once, cached."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = "lab-cosmo/pet-rho"
#: auxiliary basis of each model family, matched by name prefix
AUX_BASIS = {"pet-rho-omol": "def2-universal-jfit"}
#: preferred sizes when none is requested
SIZES = ("s", "m", "xs", "l", "xl")
CHECKPOINT_NAME = re.compile(
    r"^(?P<model>pet-[\w-]+?)-(?P<size>xs|s|m|l|xl)-v?(?P<version>\d+\.\d+\.\d+)\.ckpt$"
)


@lru_cache(maxsize=1)
def _releases() -> tuple[tuple[str, str, str], ...]:
    from huggingface_hub import list_repo_files

    matches = [
        CHECKPOINT_NAME.match(f.removeprefix("models/"))
        for f in list_repo_files(REPO)
        if f.startswith("models/")
    ]
    return tuple(
        sorted((m["model"], m["size"], m["version"]) for m in matches if m is not None)
    )


def get_available_models() -> list[str]:
    """Models released in ``lab-cosmo/pet-rho``."""
    return sorted({model for model, _, _ in _releases()})


def exported_model(
    model: str | None,
    size: str | None,
    version: str,
    checkpoint_path: str | Path | None,
) -> tuple[str | None, Path]:
    """Model name (``None`` if unknown) and exported ``.pt`` of a release or
    of a local checkpoint, exporting it on first use."""
    if (model is None) == (checkpoint_path is None):
        raise ValueError(
            "pass either model= or checkpoint_path=; released models: "
            f"{get_available_models()}"
        )
    cache = Path(os.environ.get("QPET_CACHE", Path.home() / ".cache" / "qpet"))
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        match = CHECKPOINT_NAME.match(checkpoint.name)
        path = cache / f"{checkpoint.stem}-{checkpoint.stat().st_mtime_ns}.pt"
        if not path.exists():
            _export(str(checkpoint), path)
        return match["model"] if match else None, path

    assert model is not None
    version = version.removeprefix("v")
    if (
        size is None
        or version == "latest"
        or not (cache / f"{model}-{size}-v{version}.pt").exists()
    ):
        size, version = _resolve(model, size, version)
    path = cache / f"{model}-{size}-v{version}.pt"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        _export(hf_hub_download(REPO, path.stem + ".ckpt", subfolder="models"), path)
    return model, path


def _resolve(model: str, size: str | None, version: str) -> tuple[str, str]:
    """Size and version of a release, as requested or the defaults."""
    released = [(s, v) for m, s, v in _releases() if m == model]
    if not released:
        raise ValueError(
            f"no released model {model!r}; available: {get_available_models()}"
        )
    sizes = sorted({s for s, _ in released}, key=SIZES.index)
    if size is None:
        size = sizes[0]
    elif size not in sizes:
        raise ValueError(f"{model} has no size {size!r}; available: {sizes}")
    versions = sorted(
        (v for s, v in released if s == size),
        key=lambda v: tuple(int(x) for x in v.split(".")),
    )
    if version == "latest":
        version = versions[-1]
    elif version not in versions:
        raise ValueError(
            f"{model}-{size} has no version {version}; available: {versions}"
        )
    return size, version


def _export(checkpoint: str, path: Path) -> None:
    try:
        from metatrain.utils.io import load_model
    except ImportError as err:
        raise ImportError(
            f"exporting {checkpoint} needs metatrain: install `qpet[metatrain]`"
        ) from err

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    load_model(checkpoint).export().save(str(partial))
    partial.rename(path)


def aux_basis(name: str | None, model: Any) -> str:
    """Auxiliary basis from the model metadata, or from the family of ``name``."""
    basis = model.metadata().extra.get("aux_basis")
    if basis is None and name is not None:
        basis = next((b for p, b in AUX_BASIS.items() if name.startswith(p)), None)
    if basis is None:
        raise ValueError(
            f"the auxiliary basis of {name or 'this model'} is unknown: pass aux_basis="
        )
    return basis


def heads(outputs: Mapping[str, Any], ensembles: bool) -> dict[str, str]:
    """
    Heads by output name: the density is the one per-atom ``mtt::`` target, the
    spin density ``<density>_spin``, their LLPR ensembles
    ``mtt::aux::<target>_ensemble``.
    """
    rho = [
        name
        for name, output in outputs.items()
        if name.startswith("mtt::")
        and "aux::" not in name
        and not name.endswith("_spin")
        and output.per_atom
    ]
    if len(rho) != 1:
        raise ValueError(f"cannot tell the density among {rho}: pass heads=")
    target = rho[0].removeprefix("mtt::")
    names = {
        "rho": rho[0],
        "spin": f"{rho[0]}_spin",
        "rho_ensemble": f"mtt::aux::{target}_ensemble",
        "spin_ensemble": f"mtt::aux::{target}_spin_ensemble",
    }
    out = {
        role: name
        for role, name in names.items()
        if name in outputs and (ensembles or not role.endswith("_ensemble"))
    }
    if ensembles and "rho_ensemble" not in out:
        raise ValueError(f"{rho[0]} has no LLPR ensemble")
    return out
