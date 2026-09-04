"""Tests for ``DensityModel``.

Dict mode: a single model forward pass produces a ``dict[str, TensorMap]``
that can feed multiple downstream consumers without re-evaluating the model.

Model mode: a minimal scripted ``AtomisticModel`` verifies the real inference
path — constructor capability checks and a SINGLE forward pass even when the
dipole head is also required.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from pyscf import dft, gto

from qpet import (
    BOHR_TO_ANG,
    DEFAULT_CHARGE_OFFSET,
    DensityModel,
    DMDensity,
    RIDensity,
)

HEAD = "mtt::rho_c_jfit_coulomb"
DIPOLE_HEAD = "mtt::electric_dipole"

# ── helpers ───────────────────────────────────────────────────────────


def _ri_coeffs_to_tensormap(
    mol: gto.Mole,
    auxmol: gto.Mole,
    ri_coeffs: np.ndarray,
    system_idx: int = 0,
) -> TensorMap:
    """Pack flat Coulomb-metric RI coefficients into a ``TensorMap``.

    The structure mirrors what a metatomic density model produces so that
    the TensorMap→PySCF conversion can round-trip the values.
    """
    ao_loc = auxmol.ao_loc_nr()

    shells_by_key: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(
        list
    )
    for ib in range(auxmol.nbas):
        L = auxmol.bas_angular(ib)
        atom_idx = int(auxmol.bas_atom(ib))
        Z = int(auxmol.atom_charge(atom_idx))
        start = ao_loc[ib]
        nfn = ao_loc[ib + 1] - ao_loc[ib]
        nctr = nfn // (2 * L + 1)
        shells_by_key[(L, Z)].append((atom_idx, start, nctr, ib))

    keys_list: list[list[int]] = []
    blocks: list[TensorBlock] = []

    for (L, Z), shells in sorted(shells_by_key.items()):
        atoms_in_type = sorted({s[0] for s in shells})

        # All atoms of the same Z have the same shell structure for this L.
        n_comp = 2 * L + 1
        n_radial = sum(s[2] for s in shells if s[0] == atoms_in_type[0])

        values = np.zeros((len(atoms_in_type), n_comp, n_radial))
        atom_to_sample = {a: i for i, a in enumerate(atoms_in_type)}

        for sample_atom_idx, sample in atom_to_sample.items():
            atom_shells = sorted(
                (s for s in shells if s[0] == sample_atom_idx),
                key=lambda x: x[3],
            )
            rad_offset = 0
            for _atom_idx, start, nctr, _ib in atom_shells:
                for i_ctr in range(nctr):
                    for m in range(n_comp):
                        ao_idx = start + i_ctr * n_comp + m
                        val = ri_coeffs[ao_idx]
                        if L == 1:
                            # PySCF stores (px, py, pz); metatensor expects
                            # (py, pz, px) at component positions 0, 1, 2
                            # (corresponding to m = -1, 0, +1).
                            pyscf_m = m  # 0→px, 1→py, 2→pz
                            mt_m = {0: 2, 1: 0, 2: 1}[pyscf_m]
                            values[sample, mt_m, rad_offset + i_ctr] = val
                        else:
                            values[sample, m, rad_offset + i_ctr] = val
                rad_offset += nctr

        keys_list.append([L, 1, Z])

        m_vals = [[m] for m in range(-L, L + 1)]

        blocks.append(
            TensorBlock(
                values=torch.tensor(values, dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(
                        [[system_idx, a] for a in atoms_in_type], dtype=torch.int32
                    ),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.tensor(m_vals, dtype=torch.int32),
                    )
                ],
                properties=Labels(
                    names=["n"],
                    values=torch.tensor(
                        [[n] for n in range(n_radial)], dtype=torch.int32
                    ),
                ),
            )
        )

    keys = Labels(
        names=["o3_lambda", "o3_sigma", "atom_type"],
        values=torch.tensor(keys_list, dtype=torch.int32),
    )
    return TensorMap(keys=keys, blocks=blocks)


def _random_dipole_tensormap(n_systems: int) -> TensorMap:
    """Build a dummy per-system dipole TensorMap with shape (n_systems, 3, 1)."""
    rng = np.random.default_rng(0)
    vals = torch.tensor(
        rng.uniform(-1.0, 1.0, size=(n_systems, 3, 1)), dtype=torch.float64
    )
    return TensorMap(
        keys=Labels(
            names=["_"],
            values=torch.tensor([[0]], dtype=torch.int32),
        ),
        blocks=[
            TensorBlock(
                values=vals,
                samples=Labels(
                    names=["system"],
                    values=torch.tensor(
                        [[i] for i in range(n_systems)], dtype=torch.int32
                    ),
                ),
                components=[
                    Labels(
                        names=["xyz"],
                        values=torch.tensor([[0], [1], [2]], dtype=torch.int32),
                    )
                ],
                properties=Labels(
                    names=["_"],
                    values=torch.tensor([[0]], dtype=torch.int32),
                ),
            )
        ],
    )


def _dict_model(water_tensormap: TensorMap, **kwargs) -> DensityModel:
    return DensityModel(
        model={HEAD: water_tensormap, **kwargs.pop("extra_heads", {})},
        head=HEAD,
        aux_basis="def2-svp-jkfit",
        **kwargs,
    )


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def water_tensormap(water_ri):
    """Reference RI coefficients repacked as a ``TensorMap``."""
    mol, _, auxmol, coeffs = water_ri
    return _ri_coeffs_to_tensormap(mol, auxmol, coeffs)


# ── dict-mode tests ───────────────────────────────────────────────────


def test_dict_mode_predict_roundtrip(water_ri, water_tensormap):
    """RI coefficients round-trip through dict-mode ``predict``."""
    mol, _, _, coeffs_ref = water_ri

    rho = _dict_model(water_tensormap, normalize_charge=False).predict(mol)
    assert isinstance(rho, RIDensity)
    np.testing.assert_allclose(rho.coeffs, coeffs_ref, atol=1e-10)


def test_dict_mode_density_matches_ri(water_ri, water_tensormap):
    """Density from dict-mode ``DensityModel`` matches direct ``RIDensity``."""
    mol, _, auxmol, coeffs = water_ri

    grid = dft.gen_grid.Grids(mol)
    grid.level = 3
    grid.build()
    pts = grid.coords * BOHR_TO_ANG

    rho_ref = RIDensity(auxmol, coeffs).density(pts)
    rho_pred = (
        _dict_model(water_tensormap, normalize_charge=False).predict(mol).density(pts)
    )
    np.testing.assert_allclose(rho_pred, rho_ref, atol=1e-8)


def test_dict_mode_potential_matches_dm(water_ri, water_tensormap):
    """ESP from a dict-mode prediction reproduces the density-matrix ESP."""
    mol, dm, _, _ = water_ri

    rng = np.random.default_rng(42)
    pts = rng.uniform(-4.0, 4.0, size=(30, 3))

    v_ref = DMDensity(mol, dm).potential(pts)
    v_pred = (
        _dict_model(water_tensormap, normalize_charge=False).predict(mol).potential(pts)
    )
    np.testing.assert_allclose(v_pred, v_ref, atol=2e-3)


def test_dict_mode_charge_normalization(water_ri):
    """
    ``normalize_charge=True`` must drive the integrated RI charge to
    ``-N_electrons`` and leave the caller-owned TensorMap unmodified.
    (Regression: the normalization helpers once lacked their torch import,
    so this path raised ``NameError`` — it must stay exercised.)
    """
    from qpet import get_ri_charge

    mol, _, auxmol, coeffs = water_ri

    # Perturb so the integrated charge is visibly wrong before normalization.
    rng = np.random.default_rng(0)
    coeffs_bad = coeffs + 0.05 * rng.standard_normal(coeffs.shape)
    assert abs(float(get_ri_charge(auxmol, coeffs_bad)) + mol.nelectron) > 1e-3

    tm = _ri_coeffs_to_tensormap(mol, auxmol, coeffs_bad)
    snapshot = [b.values.clone() for b in tm.blocks()]

    rho = _dict_model(tm, normalize_charge=True).predict(mol)

    target = mol.nelectron + DEFAULT_CHARGE_OFFSET
    assert abs(float(rho.charge()) + target) < 1e-6
    for b, snap in zip(tm.blocks(), snapshot, strict=True):
        assert torch.equal(b.values, snap), "caller TensorMap was mutated"


def test_dict_mode_dipole_constraint(water_ri, water_tensormap):
    """``constrain_dipole=True`` reads the dipole head from the dict."""
    mol, _, _, coeffs_ref = water_ri

    model = _dict_model(
        water_tensormap,
        extra_heads={DIPOLE_HEAD: _random_dipole_tensormap(n_systems=1)},
        constrain_dipole=True,
        dipole_head=DIPOLE_HEAD,
    )

    rho = model.predict(mol)
    assert len(rho.coeffs) == len(coeffs_ref)
    # The dipole correction should have modified the coefficients.
    assert not np.allclose(rho.coeffs, coeffs_ref, atol=1e-10)


def test_dict_mode_missing_dipole_head_raises(water_ri, water_tensormap):
    """``constrain_dipole=True`` without the dipole key in dict raises."""
    model = _dict_model(water_tensormap, constrain_dipole=True, dipole_head=DIPOLE_HEAD)
    with pytest.raises(
        ValueError, match="constrain_dipole=True.*no key.*electric_dipole"
    ):
        model.predict(water_ri[0])


def test_predict_systems_requires_basis(water_ri, water_tensormap):
    """``predict_systems`` without an orbital basis must fail loudly."""
    from qpet import pyscf_to_system

    model = _dict_model(water_tensormap)  # no basis argument
    system = pyscf_to_system(water_ri[0], dtype=torch.float64)
    with pytest.raises(ValueError, match="basis"):
        model.predict_systems(system)


def test_predict_systems_matches_predict(water_ri, water_tensormap):
    """The System entry point must produce the same density as the Mole one."""
    from qpet import pyscf_to_system

    mol = water_ri[0]
    rho_mol = _dict_model(water_tensormap).predict(mol)

    model = _dict_model(water_tensormap, basis="6-31g*")
    system = pyscf_to_system(mol, dtype=torch.float64)
    (rho_sys,) = model.predict_systems(system)

    np.testing.assert_allclose(rho_sys.coeffs, rho_mol.coeffs, atol=1e-10)


# ── model-mode tests (scripted dummy AtomisticModel) ──────────────────

from _dummy_model import wrap_dummy  # noqa: E402
from metatomic.torch import ModelOutput  # noqa: E402


def test_model_mode_single_forward_pass(water_ri, water_tensormap):
    """
    The real inference path (AtomisticModel, capability check, neighbor
    lists) must run exactly ONE forward pass even when ``constrain_dipole``
    also needs the dipole head — a second pass would double inference cost.
    """
    mol, _, _, coeffs_ref = water_ri

    atomistic, scripted = wrap_dummy(
        water_tensormap,
        _random_dipole_tensormap(1),
        {HEAD: ModelOutput(per_atom=True), DIPOLE_HEAD: ModelOutput(per_atom=False)},
    )

    model = DensityModel(
        model=atomistic,
        head=HEAD,
        aux_basis="def2-svp-jkfit",
        dtype=torch.float64,
        constrain_dipole=True,
        dipole_head=DIPOLE_HEAD,
    )
    rho = model.predict(mol)

    assert int(scripted.n_calls) == 1, "expected exactly one forward pass"
    assert rho.coeffs.shape == coeffs_ref.shape
    # Dipole constraint must have modified the raw prediction.
    assert not np.allclose(rho.coeffs, coeffs_ref, atol=1e-10)


def test_model_mode_missing_capability_raises(water_tensormap):
    """Constructor must reject constrain_dipole if the model can't provide it."""
    atomistic, _ = wrap_dummy(
        water_tensormap,
        _random_dipole_tensormap(1),
        {HEAD: ModelOutput(per_atom=True)},  # no dipole head
    )

    with pytest.raises(ValueError, match="does not support"):
        DensityModel(
            model=atomistic,
            head=HEAD,
            aux_basis="def2-svp-jkfit",
            dtype=torch.float64,
            constrain_dipole=True,
        )


def test_charge_projection_is_the_default(water_ri, water_tensormap):
    """
    A ``DensityModel`` normalizes the charge unless told not to.

    The default target is the ECP-aware electron count plus the systematic
    offset of the reference RI fits, in the Coulomb metric; the raw model
    output is one explicit argument away.
    """
    from qpet import electron_count

    mol, _, _, coeffs_ref = water_ri
    default = _dict_model(water_tensormap).predict(mol)
    raw = _dict_model(water_tensormap, normalize_charge=False).predict(mol)

    target = electron_count(mol) + DEFAULT_CHARGE_OFFSET
    assert abs(float(default.charge()) + target) < 1e-9
    np.testing.assert_allclose(raw.coeffs, coeffs_ref, atol=1e-10)
    assert np.abs(default.coeffs - raw.coeffs).max() > 0.0


def test_charge_projection_guard_names_the_system(water_ri, water_tensormap):
    """A too-large correction is skipped, with the offender named."""
    mol, _, auxmol, coeffs = water_ri
    broken = _ri_coeffs_to_tensormap(mol, auxmol, coeffs * 0.8)

    model = _dict_model(broken, charge_guard=0.5)
    with pytest.warns(RuntimeWarning, match=r"system 0 \(H2O1, charge \+0\)"):
        rho = model.predict(mol)
    np.testing.assert_allclose(rho.coeffs, coeffs * 0.8, atol=1e-12)


def test_missing_ecp_is_flagged():
    """A def2 basis on a heavy element without an ECP is a silent wrong answer."""
    from qpet.model import _warn_if_ecp_is_missing

    iodide = gto.M(atom="I 0 0 0; H 0 0 1.61", basis="def2-svp", verbose=0)
    water = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24", basis="def2-svp")

    with pytest.warns(RuntimeWarning, match="no 'ecp' set"):
        _warn_if_ecp_is_missing([iodide], None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_ecp_is_missing([iodide], "def2-svp")
        _warn_if_ecp_is_missing([water], None)
