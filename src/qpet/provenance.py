"""Trace a result back to the weights that produced it.

A predicted density is only as reproducible as the model file behind it, and
model files get overwritten in place. :func:`file_fingerprint` identifies one by
content, and :func:`model_provenance` wraps it with the versions of the code
that read it, so that a CSV or a report can carry its own audit trail.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Bytes read per hashing step. Model files are tens of megabytes.
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Fingerprint:
    """What identifies a file on disk, by content rather than by name."""

    path: str
    """Absolute path the file was read from."""
    name: str
    """Basename, the part that usually appears in a report."""
    sha256: str
    """SHA-256 of the file contents."""
    size_bytes: int
    """File size in bytes."""
    modified: str
    """Modification time, ISO-8601 in UTC."""

    @property
    def short_sha256(self) -> str:
        """First 12 hex digits, enough to tell two checkpoints apart."""
        return self.sha256[:12]

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary, for JSON or CSV columns."""
        return asdict(self)


def file_fingerprint(path: str | Path) -> Fingerprint:
    """
    Identify a file by the hash of its contents.

    :param path: File to fingerprint.
    :return: The :class:`Fingerprint`.
    :raises FileNotFoundError: If the path does not exist or is not a file.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"not a file: {resolved}")

    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)

    stat = resolved.stat()
    return Fingerprint(
        path=str(resolved),
        name=resolved.name,
        sha256=digest.hexdigest(),
        size_bytes=stat.st_size,
        modified=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(
            timespec="seconds"
        ),
    )


def model_provenance(
    model_path: str | Path, *, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    A record tying an output to the model file and the code that ran it.

    :param model_path: The model artefact the result came from.
    :param extra: Caller-supplied fields to merge in, such as the dataset used
        or the surface definition. Overwrites nothing that is generated here.
    :return: A JSON-serialisable dictionary.
    """
    from . import __version__

    record: dict[str, Any] = {
        "model": file_fingerprint(model_path).as_dict(),
        "qpet_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra:
        record.update({k: v for k, v in extra.items() if k not in record})
    return record


def provenance_columns(record: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a provenance record into columns for a per-row CSV.

    :param record: The output of :func:`model_provenance`.
    :return: Flat ``{column: value}``, one level deep, dictionaries prefixed by
        their key.
    """
    columns: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            columns.update({f"{key}_{sub}": item for sub, item in value.items()})
        else:
            columns[key] = value
    return columns
