"""Utilities for RI charge normalization.

Functions to compute the s-orbital normalization vector and electron count
needed to enforce charge conservation in predicted RI coefficients.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from pyscf import df, gto
from pyscf.gto.mole import charge as _symbol_to_Z

if TYPE_CHECKING:
    from metatensor.torch import TensorMap


def _true_Z(mol: gto.Mole, atom_idx: int) -> int:
    """True atomic number regardless of ECP core (ECP-safe)."""
    return int(_symbol_to_Z(mol.atom_symbol(atom_idx)))


def compute_S_vector(auxmol: gto.Mole) -> np.ndarray:
    r"""
    Analytical integral ``\int \chi_i(r) d^3r`` for every aux basis function.

    Only l=0 shells contribute; entries for l>0 shells are zero.

    :param auxmol: PySCF auxiliary Mole object.
    :return: S vector of shape (nao,), float64.
    """
    S_vec = np.zeros(auxmol.nao)
    ao_loc = auxmol.ao_loc_nr()

    for ib in range(auxmol.nbas):
        if auxmol.bas_angular(ib) != 0:
            continue

        exps = auxmol.bas_exp(ib)
        coeffs = auxmol._libcint_ctr_coeff(ib)
        start_ao = ao_loc[ib]
        nctr = coeffs.shape[1]

        for i_ctr in range(nctr):
            c_k = coeffs[:, i_ctr]
            prim_integrals = (np.pi / exps) ** 1.5 / np.sqrt(4 * np.pi)
            S_vec[start_ao + i_ctr] = np.sum(c_k * prim_integrals)

    return S_vec


def s_normalization_to_tensormap(
    mol: gto.Mole,
    auxbasis_arg: str | dict[str, str],
    system_idx: int,
) -> TensorMap:
    """
    Build a TensorMap of s-orbital S integrals for the given auxiliary basis.

    Only o3_lambda=0 blocks are included (l>0 shells discarded). Structure
    mirrors RI coefficient TensorMaps: keys ``(o3_lambda=0, o3_sigma=1,
    atom_type=Z)``, samples ``(system, atom)``, components ``(o3_mu=0)``,
    properties ``(n=0, ..., N_radial_s-1)``.

    :param mol: PySCF Mole (orbital basis), used to build the auxiliary mol.
    :param auxbasis_arg: Auxiliary basis name (str) or dict.
    :param system_idx: Integer index of this molecule in the dataset.
    :return: ``metatensor.torch.TensorMap`` of s-orbital S integrals.
    """
    import torch
    from metatensor.torch import Labels, TensorBlock, TensorMap

    auxmol = df.addons.make_auxmol(mol, auxbasis_arg)
    S_vec = compute_S_vector(auxmol)
    ao_loc = auxmol.ao_loc_nr()

    s_shells_by_atom: dict[int, list[float]] = defaultdict(list)
    for sh in range(auxmol.nbas):
        if auxmol.bas_angular(sh) != 0:
            continue
        atom_idx = int(auxmol.bas_atom(sh))
        ao_start = ao_loc[sh]
        nctr = ao_loc[sh + 1] - ao_start
        for i_ctr in range(nctr):
            s_shells_by_atom[atom_idx].append(float(S_vec[ao_start + i_ctr]))

    zs_present: set[int] = {_true_Z(mol, i) for i in s_shells_by_atom}

    keys_vals, blocks = [], []
    for Z in sorted(zs_present):
        atom_indices = [i for i in range(mol.natm) if _true_Z(mol, i) == Z]
        n_radial = len(s_shells_by_atom[atom_indices[0]])

        tensor = np.zeros((len(atom_indices), 1, n_radial), dtype=np.float64)
        for local_idx, global_idx in enumerate(atom_indices):
            for n, val in enumerate(s_shells_by_atom[global_idx]):
                tensor[local_idx, 0, n] = val

        keys_vals.append([0, 1, Z])
        blocks.append(
            TensorBlock(
                values=torch.tensor(tensor, dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(
                        [[system_idx, i] for i in atom_indices], dtype=torch.int32
                    ),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.tensor([[0]], dtype=torch.int32),
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
        values=torch.tensor(keys_vals, dtype=torch.int32),
    )
    return TensorMap(keys=keys, blocks=blocks)


def compute_n_electrons(mol: gto.Mole) -> int:
    """
    Number of electrons in the DFT calculation.

    For all-electron calculations this is ``sum(Z) - charge``. For ECP
    calculations PySCF subtracts the core electrons replaced by the
    pseudopotential, so ``mol.nelectron`` counts only the valence electrons
    explicitly treated in the SCF.

    :param mol: Built PySCF Mole object.
    :return: Number of electrons.
    """
    return mol.nelectron
