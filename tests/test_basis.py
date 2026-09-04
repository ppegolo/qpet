"""Tests for auxiliary-basis resolution, including even-tempered bases."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import df, gto, scf

from qpet import RIDensity, becke_grid, make_auxmol, resolve_aux_basis
from qpet.basis import is_etb_spec


@pytest.fixture(scope="module")
def water():
    return gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        basis="def2-svp",
        verbose=0,
    )


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("etb:def2-svp:2.0", True),
        ("ETB:def2-svp:2.0", True),
        ("def2-universal-jfit", False),
        ("etb:def2-svp", False),
    ],
)
def test_is_etb_spec(spec, expected):
    assert is_etb_spec(spec) is expected


def test_named_basis_is_passed_through(water):
    """A plain basis name must reach PySCF untouched."""
    assert resolve_aux_basis("def2-universal-jfit", water) == "def2-universal-jfit"
    assert (
        make_auxmol(water, "def2-universal-jfit").nao
        == df.addons.make_auxmol(water, "def2-universal-jfit").nao
    )


def test_etb_auxmol_matches_aug_etb(water):
    """
    The ETB spec must reproduce ``df.aug_etb`` exactly: the coefficients a model
    was trained on are only meaningful in the very basis they were fitted in, so
    an off-by-one shell here silently corrupts every downstream field.
    """
    aux = make_auxmol(water, "etb:def2-svp:2.0")
    ref = df.addons.make_auxmol(water, df.aug_etb(water, beta=2.0))

    # Compared through the basis as PySCF exposes it: the row order of
    # `_bas`/`_env` follows a set iteration in `df.aug_etb`, so it is not stable.
    assert aux.nao == ref.nao
    assert aux.ao_labels() == ref.ao_labels()
    assert np.array_equal(aux.aoslice_by_atom(), ref.aoslice_by_atom())
    # The overlap matrix is the basis itself, in the AO ordering the coefficients
    # are indexed by: if a shell were off by one, this could not match.
    assert np.allclose(aux.intor("int1e_ovlp"), ref.intor("int1e_ovlp"))


def test_etb_basis_follows_the_spec_not_the_molecule():
    """
    The orbital basis in the spec, not the molecule's own basis, generates the
    ETB — a model trained on ``etb:def2-svp`` stays evaluable on a molecule
    built in another orbital basis.
    """
    mol_svp = gto.M(atom="O 0 0 0; H 0 0 1.0", basis="def2-svp", spin=1, verbose=0)
    mol_tzvp = gto.M(atom="O 0 0 0; H 0 0 1.0", basis="def2-tzvp", spin=1, verbose=0)

    aux_a = make_auxmol(mol_svp, "etb:def2-svp:2.0")
    aux_b = make_auxmol(mol_tzvp, "etb:def2-svp:2.0")
    assert aux_a.nao == aux_b.nao

    beta_sensitive = make_auxmol(mol_svp, "etb:def2-svp:2.5")
    assert beta_sensitive.nao != aux_a.nao


@pytest.fixture(scope="module")
def water_etb(water):
    """Water RHF density, Coulomb-metric RI-fitted onto an ETB auxiliary basis."""
    dm = scf.RHF(water).run(verbose=0).make_rdm1()
    return RIDensity.from_dm(water, dm, "etb:def2-svp:2.0")


def test_ri_fit_in_etb_basis_integrates_to_electron_count(water, water_etb):
    """End-to-end: an ETB-fitted RI density carries the right electron count."""
    grid = becke_grid(water, level=3)
    n_elec = float(np.sum(grid.weights * water_etb.density(grid.points)))
    assert abs(n_elec - water.nelectron) < 0.05


def test_select_atoms_on_an_etb_density(water_etb):
    """
    ``select_atoms`` rebuilds a sub-``Mole`` from ``auxmol.basis``, which an ETB
    spec makes a dictionary rather than a name; the partition must stay exact
    and additive, as it is for a named basis.
    """
    oxygen = water_etb.select_atoms([0])
    hydrogens = water_etb.select_atoms([1, 2])
    assert oxygen.auxmol.nao + hydrogens.auxmol.nao == water_etb.auxmol.nao

    rng = np.random.default_rng(0)
    pts = rng.uniform(-4.0, 4.0, size=(50, 3))
    parts = oxygen.potential(pts) + hydrogens.potential(pts)
    assert np.max(np.abs(parts - water_etb.potential(pts))) < 1e-12


def test_auxmol_keeps_the_effective_nuclei(hydrogen_iodide_ri):
    """``make_auxmol`` hands back the nuclei of the orbital molecule, ECP included."""
    mol, _, auxmol, _ = hydrogen_iodide_ri
    assert auxmol.atom_charges().tolist() == mol.atom_charges().tolist() == [25, 1]
    assert auxmol.nelectron == mol.nelectron == 26
