"""Tests for DensityModel on a scripted model that returns fixed coefficients."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import torch
from _dummy_model import fixed_model, to_tensormap

from qpet import DensityModel, aux_integrals, direction, to_system

RHO, SPIN, ENSEMBLE = "mtt::rho", "mtt::rho_spin", "mtt::aux::rho_ensemble"
AUX = "def2-svp-jkfit"


@pytest.fixture(scope="module")
def water_cation(water_ri):
    cation = water_ri[0].copy()
    cation.charge, cation.spin = 1, 1
    return cation.build()


@pytest.fixture(scope="module")
def members(water_ri):
    _, _, _, coeffs = water_ri
    noise = np.random.default_rng(0).normal(size=(4, len(coeffs))) * 1e-3
    return coeffs + 0.01 + noise


def _model(water_ri, heads, members=None, **kwargs):
    _, _, auxmol, coeffs = water_ri
    outputs = {
        RHO: to_tensormap(auxmol, 0.98 * coeffs),
        SPIN: to_tensormap(auxmol, 0.1 * coeffs),
    }
    if members is not None:
        outputs[ENSEMBLE] = to_tensormap(auxmol, members)
    names = {"rho": RHO, "spin": SPIN, "rho_ensemble": ENSEMBLE}
    return DensityModel(
        fixed_model(outputs),
        aux_basis=AUX,
        heads={role: names[role] for role in heads},
        dtype=torch.float64,
        **kwargs,
    )


def test_raw_prediction_round_trips_the_coefficients(water_ri):
    mol, _, _, coeffs = water_ri
    p = _model(water_ri, ["rho"], normalize=None).predict(mol)
    np.testing.assert_allclose(p.rho.coeffs, 0.98 * coeffs, atol=1e-12)
    assert p.spin is None and p.shift == {}


def test_density_is_projected_to_the_electron_count(water_ri):
    mol, _, auxmol, coeffs = water_ri
    with pytest.warns(RuntimeWarning, match="projected anyway"):
        p = _model(water_ri, ["rho"]).predict(mol)
    assert p.rho.integral() == pytest.approx(mol.nelectron, abs=1e-10)
    shift = mol.nelectron - aux_integrals(auxmol) @ (0.98 * coeffs)
    assert p.shift["rho"] == pytest.approx(shift)
    np.testing.assert_allclose(p.rho.coeffs, 0.98 * coeffs + shift * direction(auxmol))


def test_guard_stays_quiet_on_a_small_shift(water_ri):
    model = _model(water_ri, ["rho"], guard=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.predict(water_ri[0])


def test_spin_is_projected_to_the_unpaired_electrons(water_ri, water_cation):
    p = _model(water_ri, ["rho", "spin"], guard=1.0).predict(water_cation)
    assert p.rho.integral() == pytest.approx(9.0, abs=1e-10)
    assert p.spin.integral() == pytest.approx(1.0, abs=1e-10)
    assert p.rho.nuclei and not p.spin.nuclei


def test_spin_of_a_singlet_is_zero_unless_asked(water_ri):
    mol = water_ri[0]
    assert (
        not _model(water_ri, ["rho", "spin"], guard=1.0).predict(mol).spin.coeffs.any()
    )
    kept = _model(
        water_ri, ["rho", "spin"], broken_symmetry=True, normalize=None
    ).predict(mol)
    assert kept.spin.coeffs.any()


def test_ensemble_is_recentred_on_the_prediction_and_projected(water_ri, members):
    mol, _, auxmol, _ = water_ri
    p = _model(water_ri, ["rho", "rho_ensemble"], members, guard=1.0).predict(mol)
    ens = p.rho_ensemble
    assert len(ens) == 4
    np.testing.assert_allclose(ens.coeffs.mean(axis=0), p.rho.coeffs, atol=1e-12)
    np.testing.assert_allclose(
        ens.coeffs @ aux_integrals(auxmol), mol.nelectron, atol=1e-10
    )
    spread = members - members.mean(axis=0)
    np.testing.assert_allclose(
        ens.coeffs - p.rho.coeffs,
        spread - np.outer(spread @ aux_integrals(auxmol), direction(auxmol)),
        atol=1e-12,
    )


def test_raw_ensemble_is_still_recentred(water_ri, members):
    mol, _, _, _ = water_ri
    p = _model(water_ri, ["rho", "rho_ensemble"], members, normalize=None).predict(mol)
    np.testing.assert_allclose(
        p.rho_ensemble.coeffs.mean(axis=0), p.rho.coeffs, atol=1e-12
    )


def test_predict_many_matches_predict(water_ri, water_cation):
    model = _model(water_ri, ["rho", "spin"], guard=1.0)
    many = model.predict_many([water_ri[0], water_cation])
    np.testing.assert_allclose(
        many[1].spin.coeffs, model.predict(water_cation).spin.coeffs
    )


def test_wrong_basis_names_the_system(water_ri):
    mol = water_ri[0]
    model = _model(water_ri, ["rho"], normalize=None)
    model.aux_basis = "def2-universal-jfit"
    with pytest.raises(ValueError, match=re.escape("for H2O1 (charge +0)")):
        model.predict(mol)


def test_unknown_head_role_raises(water_ri):
    with pytest.raises(ValueError, match="heads needs 'rho'"):
        DensityModel("unused.pt", aux_basis=AUX, heads={"spin": SPIN})


def test_to_system_carries_elements_charge_and_multiplicity(
    hydrogen_iodide_ri, water_cation
):
    mol = hydrogen_iodide_ri[0]
    system = to_system(mol, dtype=torch.float64)
    assert system.types.tolist() == [53, 1]
    np.testing.assert_allclose(system.positions.numpy(), mol.atom_coords(unit="Ang"))
    assert not system.pbc.any()
    cation = to_system(water_cation)
    assert cation.get_data("charge").block().values.item() == 1
    assert cation.get_data("spin_multiplicity").block().values.item() == 2
