"""The fast backend against the metatomic one, on a real checkpoint.

Skipped unless ``QPET_TEST_CHECKPOINT`` points at a metatrain PET density
checkpoint; exporting it needs metatrain.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from pyscf import gto

from qpet import DensityModel

CHECKPOINT = os.environ.get("QPET_TEST_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    CHECKPOINT is None, reason="set QPET_TEST_CHECKPOINT to a PET density checkpoint"
)
AUX = "def2-universal-jfit"


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    from qpet.backends.fast.export import export, load_model

    path = tmp_path_factory.mktemp("models")
    load_model(CHECKPOINT).export().save(str(path / "model.pt"))
    artifact = export(CHECKPOINT, str(path / "model.fast"))
    heads = {"spin" if n.endswith("_spin") else "rho": n for n in artifact["targets"]}
    return [
        DensityModel(
            path / f"model.{suffix}",
            aux_basis=AUX,
            heads=heads,
            backend=backend,
            dtype=torch.float64,
            normalize=None,
            broken_symmetry=True,
        )
        for suffix, backend in (("pt", "metatomic"), ("fast", "fast"))
    ]


def _molecules():
    rng = np.random.default_rng(0)
    water = gto.M(atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", basis="def2-svp")
    anion = gto.M(
        atom="C 0 0 0; O 1.25 0 0; O -0.6 1.1 0; H -0.6 -1.0 0",
        basis="def2-svp",
        charge=-1,
    )
    radical = gto.M(atom="N 0 0 0; O 0 0 1.15", basis="def2-svp", spin=1)
    cluster = gto.M(
        atom=[("O", rng.uniform(0, 8, 3)) for _ in range(12)], basis="sto-3g"
    )
    return [water, anion, radical, cluster]


@pytest.mark.parametrize("index", range(4))
def test_backends_agree(models, index):
    metatomic, fast = (m.predict(_molecules()[index]) for m in models)
    np.testing.assert_allclose(fast.rho.coeffs, metatomic.rho.coeffs, atol=1e-8)
    if fast.spin is not None:
        np.testing.assert_allclose(fast.spin.coeffs, metatomic.spin.coeffs, atol=1e-8)


def test_backends_agree_in_a_periodic_cell(models):
    mol = gto.M(atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", basis="def2-svp")
    cell = 6.0 * np.eye(3)
    metatomic, fast = (m.predict(mol, cell=cell).rho.coeffs for m in models)
    np.testing.assert_allclose(fast, metatomic, rtol=0, atol=1e-8)


def test_partitioned_route_is_exact(models, monkeypatch):
    import qpet.backends.fast._calculator as calculator

    blocks = []

    def counted(*args, **kwargs):
        blocks.extend(calculator_partition(*args, **kwargs))
        return blocks

    calculator_partition = calculator.partition_system
    monkeypatch.setattr(calculator, "partition_system", counted)
    mol = _molecules()[3]
    fast = models[1]
    whole = fast.predict(mol).rho.coeffs
    monkeypatch.setattr(fast.backend, "max_atoms_per_pass", 4)
    parts = fast.predict(mol).rho.coeffs
    assert len(blocks) > 1
    np.testing.assert_allclose(parts, whole, rtol=0, atol=1e-10)


@pytest.mark.skipif(
    not os.environ.get("QPET_TEST_LLPR"),
    reason="set QPET_TEST_LLPR to the exported LLPR model of QPET_TEST_CHECKPOINT",
)
def test_llpr_ensembles_agree(tmp_path):
    from qpet.backends.fast import ensemble_name
    from qpet.backends.fast.export import export

    llpr = os.environ["QPET_TEST_LLPR"]
    artifact = export(CHECKPOINT, str(tmp_path / "model.fast"), llpr=llpr)
    heads = {}
    for name, target in artifact["targets"].items():
        role = "spin" if name.endswith("_spin") else "rho"
        heads[role] = name
        if target["llpr"] is not None:
            heads[role + "_ensemble"] = ensemble_name(name)
    metatomic, fast = (
        DensityModel(
            path,
            aux_basis=AUX,
            heads=heads,
            backend=backend,
            dtype=torch.float64,
            normalize=None,
            broken_symmetry=True,
        )
        for path, backend in ((llpr, "metatomic"), (tmp_path / "model.fast", "fast"))
    )
    mol = _molecules()[2]
    a, b = fast.predict(mol), metatomic.predict(mol)
    for role in ("rho", "spin"):
        x, y = getattr(a, role + "_ensemble"), getattr(b, role + "_ensemble")
        if y is not None:
            np.testing.assert_allclose(x.coeffs, y.coeffs, atol=1e-6)
