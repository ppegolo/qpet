"""Multipole moments of an RI density.

``∫ r^n φ_k dr`` is the cross integral of ``φ_k`` with a flat Gaussian divided
by the Gaussian's normalization. Electrons count as negative charge.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto

_ALPHA = 1e-12
_FLAT = gto.M(atom="X 0 0 0", basis={"X": [[0, [_ALPHA, 1.0]]]}, verbose=0)
_SCALE = (2 * _ALPHA / np.pi) ** -0.75


def _electronic(
    auxmol: gto.Mole, coeffs: np.ndarray, origin: np.ndarray, intor: str, comp: int
) -> np.ndarray:
    with auxmol.with_common_orig(origin):
        ints = gto.intor_cross(intor, auxmol, _FLAT, comp=comp)
    return -_SCALE * (ints.reshape(comp, auxmol.nao) @ coeffs)


def dipole(
    auxmol: gto.Mole, coeffs: np.ndarray, origin: np.ndarray, nuclei: bool
) -> np.ndarray:
    """Dipole moment about ``origin`` (bohr), shape (3,)."""
    mu = _electronic(auxmol, coeffs, origin, "int1e_r", 3)
    if nuclei:
        mu += auxmol.atom_charges() @ (auxmol.atom_coords() - origin)
    return mu


def quadrupole(
    auxmol: gto.Mole, coeffs: np.ndarray, origin: np.ndarray, nuclei: bool
) -> np.ndarray:
    """Primitive quadrupole tensor ``Σ q r r`` about ``origin`` (bohr), shape (3, 3)."""
    q = _electronic(auxmol, coeffs, origin, "int1e_rr", 9).reshape(3, 3)
    if nuclei:
        r = auxmol.atom_coords() - origin
        q += np.einsum("i,ix,iy->xy", auxmol.atom_charges(), r, r)
    return q
