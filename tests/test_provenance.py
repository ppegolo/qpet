"""Tests for tracing an output back to the model file that produced it."""

from __future__ import annotations

import hashlib

import pytest

from qpet.provenance import file_fingerprint, model_provenance, provenance_columns


@pytest.fixture
def artefact(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights, pretending")
    return path


class TestFileFingerprint:
    def test_hashes_the_contents(self, artefact):
        fingerprint = file_fingerprint(artefact)
        assert fingerprint.sha256 == hashlib.sha256(artefact.read_bytes()).hexdigest()
        assert fingerprint.size_bytes == artefact.stat().st_size
        assert fingerprint.name == "model.pt"
        assert fingerprint.short_sha256 == fingerprint.sha256[:12]

    def test_separates_two_checkpoints_at_the_same_path(self, artefact):
        before = file_fingerprint(artefact)
        artefact.write_bytes(b"weights, retrained")
        assert file_fingerprint(artefact).sha256 != before.sha256

    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            file_fingerprint(tmp_path / "absent.pt")


class TestModelProvenance:
    def test_records_the_model_and_the_code(self, artefact):
        record = model_provenance(artefact)
        assert record["model"]["sha256"] == file_fingerprint(artefact).sha256
        assert record["qpet_version"]
        assert record["generated"].endswith("+00:00")

    def test_extra_fields_cannot_overwrite_the_generated_ones(self, artefact):
        record = model_provenance(
            artefact, extra={"dataset": "pockets.zip", "qpet_version": "lies"}
        )
        assert record["dataset"] == "pockets.zip"
        assert record["qpet_version"] != "lies"


class TestProvenanceColumns:
    def test_flattens_one_level_with_a_prefix(self, artefact):
        columns = provenance_columns(model_provenance(artefact))
        assert columns["model_sha256"] == file_fingerprint(artefact).sha256
        assert "model" not in columns
        assert all(not isinstance(value, dict) for value in columns.values())
