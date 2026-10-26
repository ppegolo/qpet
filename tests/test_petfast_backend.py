"""The two backends must agree coefficient for coefficient.

They reach PySCF's auxiliary-basis order by different routes -- one sorts a
TensorMap's indices, the other applies a permutation baked at export time -- and an
ordering mismatch would quietly permute the density rather than raise.

Skipped unless ``QPET_TEST_CHECKPOINT`` points at a metatrain PET density checkpoint
and ``petfast`` is installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pyscf import gto

petfast = pytest.importorskip("petfast", reason="petfast is not installed")

CHECKPOINT = os.environ.get("QPET_TEST_CHECKPOINT")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT or not Path(CHECKPOINT).exists(),
    reason="set QPET_TEST_CHECKPOINT to a metatrain PET density checkpoint",
)

AUX_BASIS = os.environ.get("QPET_TEST_AUX_BASIS", "def2-universal-jfit")


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    """Both artifacts from the same checkpoint: metatomic ``.pt`` and ``.petfast``."""
    from metatrain.pet import PET
    from petfast.export import export_checkpoint

    directory = tmp_path_factory.mktemp("petfast_backend")
    checkpoint = torch.load(CHECKPOINT, weights_only=False, map_location="cpu")
    model = PET.load_checkpoint(PET.upgrade_checkpoint(checkpoint), "export")

    metatomic_path = directory / "model.pt"
    model.export().save(str(metatomic_path))

    artifact_path = directory / "model.petfast"
    artifact = export_checkpoint(CHECKPOINT, str(artifact_path))
    return str(metatomic_path), str(artifact_path), artifact["target_name"]


def _molecules():
    """A neutral singlet and a charged doublet, to exercise the conditioning too."""
    return [
        gto.M(
            atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
            basis="def2-svp",
            verbose=0,
        ),
        gto.M(
            atom="N 0 0 0; H 1.01 0 0; H -0.34 0.95 0; H -0.34 -0.48 -0.82",
            basis="def2-svp",
            charge=1,
            spin=1,
            verbose=0,
        ),
    ]


def test_matches_metatomic_backend(exports):
    """Both molecules, with and without the (unguarded) charge correction."""
    from qpet.model import DensityModel

    metatomic_path, artifact_path, head = exports
    for mol in _molecules():
        for normalize in (False, True):
            reference = DensityModel(
                metatomic_path,
                head=head,
                aux_basis=AUX_BASIS,
                dtype=torch.float64,
                normalize_charge=normalize,
                charge_guard=None,
            ).predict(mol)
            fast = DensityModel(
                artifact_path,
                aux_basis=AUX_BASIS,
                dtype=torch.float64,
                normalize_charge=normalize,
                charge_guard=None,
            ).predict(mol)

            scale = np.abs(reference.coeffs).mean()
            error = np.abs(fast.coeffs - reference.coeffs).max() / scale
            assert error < 1e-12, f"{mol.atom}, normalize={normalize}: {error:.2e}"
            if normalize:
                from qpet import DEFAULT_CHARGE_OFFSET

                target = mol.nelectron + DEFAULT_CHARGE_OFFSET
                assert abs(fast.charge() + target) < 1e-6


def test_matches_metatomic_backend_periodic(exports):
    """A tight periodic cell: petfast must include periodic images, like metatomic.

    A 6 A box puts images well inside the cutoff.
    """
    from qpet import build_system
    from qpet.model import DensityModel

    metatomic_path, artifact_path, head = exports
    numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]])
    common = dict(aux_basis=AUX_BASIS, basis="def2-svp", dtype=torch.float64)
    reference = DensityModel(metatomic_path, head=head, **common)
    fast = DensityModel(artifact_path, **common)
    for box in (6.0, 9.0):
        cell = np.eye(3) * box
        ref = reference.predict_systems(build_system(numbers, positions, cell))[0]
        fst = fast.predict_systems(build_system(numbers, positions, cell))[0]
        error = np.abs(fst.coeffs - ref.coeffs).max() / np.abs(ref.coeffs).mean()
        # Float64 round-off, ~1e-9.
        assert error < 1e-8, f"periodic {box} A box: {error:.2e}"
