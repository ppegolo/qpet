"""DensityModel.from_release against the exported model of the same release.

Skipped unless ``QPET_TEST_RELEASE_REFERENCE`` points at the exported ``.pt`` of
``pet-rho-omol-l`` v1.0.0. The checkpoint comes from Hugging Face when the
repository is reachable, and from ``QPET_TEST_RELEASE_CHECKPOINT`` (a local copy
of the ``.ckpt``) through ``checkpoint_path`` when that is set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from pyscf import gto

from qpet import DensityModel, get_available_models

REFERENCE = os.environ.get("QPET_TEST_RELEASE_REFERENCE")
CHECKPOINT = os.environ.get("QPET_TEST_RELEASE_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    REFERENCE is None,
    reason="set QPET_TEST_RELEASE_REFERENCE to the exported pet-rho-omol-l",
)


def _on_hub():
    try:
        return "pet-rho-omol" in get_available_models()
    except Exception:
        return False


@pytest.mark.parametrize("source", ["local", "hub"])
def test_release_matches_its_export(source, tmp_path, monkeypatch):
    pytest.importorskip("metatrain")
    monkeypatch.setenv("QPET_CACHE", str(tmp_path / "cache"))
    if source == "local":
        if CHECKPOINT is None:
            pytest.skip("set QPET_TEST_RELEASE_CHECKPOINT to a local .ckpt")
        link = tmp_path / "pet-rho-omol-l-v1.0.0.ckpt"
        link.symlink_to(CHECKPOINT)
        release = DensityModel.from_release(checkpoint_path=link, dtype=torch.float64)
    elif _on_hub():
        release = DensityModel.from_release(
            "pet-rho-omol", size="l", dtype=torch.float64
        )
    else:
        pytest.skip("lab-cosmo/pet-rho is not reachable")

    reference = DensityModel(
        REFERENCE,
        aux_basis="def2-universal-jfit",
        heads={
            "rho": "mtt::rho_c_jfit_coulomb",
            "spin": "mtt::rho_c_jfit_coulomb_spin",
        },
        dtype=torch.float64,
    )
    assert len(list((tmp_path / "cache").glob("pet-rho-omol-l-v1.0.0*.pt"))) == 1
    for mol in (
        gto.M(atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", basis="def2-svp"),
        gto.M(atom="N 0 0 0; O 0 0 1.15", basis="def2-svp", spin=1),
    ):
        got, expected = release.predict(mol), reference.predict(mol)
        np.testing.assert_allclose(got.rho.coeffs, expected.rho.coeffs, atol=1e-5)
        np.testing.assert_allclose(got.spin.coeffs, expected.spin.coeffs, atol=1e-5)
