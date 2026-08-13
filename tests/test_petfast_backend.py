"""The petfast backend must agree with the metatomic one, coefficient for coefficient.

Both paths flatten the model's per-irrep output into PySCF's auxiliary-basis order,
but they arrive there by different routes -- one sorts a TensorMap's sample and
property indices, the other applies a permutation baked at export time. An ordering
mismatch would not raise; it would quietly permute the density. So these compare the
coefficient vectors directly, in float64, where agreement should be at round-off.

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


@pytest.mark.parametrize("index", [0, 1])
def test_matches_metatomic_backend(exports, index):
    from qpet.model import DensityModel
    from qpet.petfast_backend import PetFastDensityModel

    metatomic_path, artifact_path, head = exports
    mol = _molecules()[index]

    reference = DensityModel(
        metatomic_path, head=head, aux_basis=AUX_BASIS, dtype=torch.float64
    ).predict(mol)
    fast = PetFastDensityModel(
        artifact_path, aux_basis=AUX_BASIS, dtype=torch.float64
    ).predict(mol)

    assert fast.coeffs.shape == reference.coeffs.shape
    scale = np.abs(reference.coeffs).mean()
    error = np.abs(fast.coeffs - reference.coeffs).max() / scale
    assert error < 1e-12, f"coefficients differ by {error:.2e}"


def test_ordering_is_not_merely_a_permutation(exports):
    """Guard: equal *sorted* coefficients would pass a sloppy comparison.

    If the two backends disagreed only in ordering, the coefficient multiset would
    still match. Checking that the sorted vectors agree far better than the raw
    ones would is meaningless -- so instead assert the density is the same where it
    is evaluated, which no permutation survives.
    """
    from qpet.model import DensityModel
    from qpet.petfast_backend import PetFastDensityModel

    metatomic_path, artifact_path, head = exports
    mol = _molecules()[0]

    reference = DensityModel(
        metatomic_path, head=head, aux_basis=AUX_BASIS, dtype=torch.float64
    ).predict(mol)
    fast = PetFastDensityModel(
        artifact_path, aux_basis=AUX_BASIS, dtype=torch.float64
    ).predict(mol)

    points = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.3, 0.1], [-1.2, 0.8, 0.4], [2.0, -1.0, 0.7]]
    )
    density_reference = reference.density(points)
    density_fast = fast.density(points)
    assert np.abs(density_fast - density_reference).max() < 1e-10 * max(
        1.0, np.abs(density_reference).max()
    )


def test_normalize_charge_matches(exports):
    """The charge correction must behave the same on both backends."""
    from qpet.model import DensityModel
    from qpet.petfast_backend import PetFastDensityModel

    metatomic_path, artifact_path, head = exports
    mol = _molecules()[0]

    reference = DensityModel(
        metatomic_path,
        head=head,
        aux_basis=AUX_BASIS,
        dtype=torch.float64,
        normalize_charge=True,
    ).predict(mol)
    fast = PetFastDensityModel(
        artifact_path,
        aux_basis=AUX_BASIS,
        dtype=torch.float64,
        normalize_charge=True,
    ).predict(mol)

    scale = np.abs(reference.coeffs).mean()
    assert np.abs(fast.coeffs - reference.coeffs).max() / scale < 1e-12
    # And the correction actually did something measurable.
    assert abs(fast.charge() + mol.nelectron) < 1e-6
