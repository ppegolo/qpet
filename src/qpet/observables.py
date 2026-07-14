"""RI density observables via the Ghost Trick.

Extracts electronic multipoles (charge, dipole) from RI fitting coefficients
using a near-flat Gaussian ghost orbital as a proxy for constant/linear basis
functions.  See DENSITY_ROADMAP.md §Implementation Reference for derivation.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from pyscf import gto

# Ghost Atom Configuration
_GHOST_ALPHA = 1e-12
_NORM_FACTOR = 1.0 / (2 * _GHOST_ALPHA / np.pi) ** 0.75


def _get_or_create_ghost(ghost: gto.Mole | None) -> gto.Mole:
    """Creates a default ghost atom at the global origin if not provided."""
    if ghost is not None:
        return ghost
    return gto.M(
        atom=[("X", (0.0, 0.0, 0.0))],
        basis={"X": [[0, [_GHOST_ALPHA, 1.0]]]},
        spin=0,
        charge=0,
    )


def get_ri_charge(
    auxmol: gto.Mole,
    ri_coeffs: np.ndarray,
    *,
    ghost: gto.Mole | None = None,
    include_nuclear: bool = False,
) -> np.ndarray:
    """
    Returns the electronic charge (-N_elec) as a 0D numpy array.

    :param auxmol: PySCF auxiliary Mole returned by :class:`MetatomicDensityPredictor`
    :param ri_coeffs: RI fitting coefficients, shape ``(naux,)``
    :param origin: Expansion origin in **Bohr** (ignored for charge)
    :param ghost: Optional pre-configured ghost molecule; if None, a default flat
        Gaussian at the global origin is used.
    :param include_nuclear: If True, the total charge (nuclear + electronic) is
        returned; otherwise, only the electronic contribution is returned.
    :return: Electronic charge as a 0D numpy array
    """
    ghost_mol = _get_or_create_ghost(ghost)

    ovlp_cross = gto.intor_cross("int1e_ovlp", auxmol, ghost_mol).flatten()
    n_elec = np.dot(ri_coeffs, ovlp_cross * _NORM_FACTOR)
    if include_nuclear:
        n_nuc = auxmol.atom_charges().sum()
        return np.array(n_nuc - n_elec)
    return np.array(-n_elec)


def get_ri_dipole(
    auxmol: gto.Mole,
    ri_coeffs: np.ndarray,
    origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    *,
    ghost: gto.Mole | None = None,
    include_nuclear: bool = False,
) -> np.ndarray:
    """
    Returns the electronic dipole moment shifted to the target origin.

    :param auxmol: PySCF auxiliary Mole returned by :class:`MetatomicDensityPredictor`
    :param ri_coeffs: RI fitting coefficients, shape ``(naux,)``
    :param origin: Expansion origin in **Bohr**.  For origin-independent dipoles use the
        ligand centre of mass or centre of nuclear charge.
    :param ghost: Optional pre-configured ghost molecule; if None, a default flat
        Gaussian at the global origin is used.
    :return: Electronic dipole vector (shape ``(3,)``) in atomic units, shifted to the
        target origin
    """
    ghost_mol = _get_or_create_ghost(ghost)
    A = np.asarray(origin)

    # 1. Compute Dipole at global origin (0,0,0)
    dip_cross = gto.intor_cross("int1e_r", auxmol, ghost_mol, comp=3)
    int_r = dip_cross.reshape(3, auxmol.nao) * _NORM_FACTOR
    mu_0 = -np.dot(int_r, ri_coeffs)

    # Fast path if no origin shift is requested and no nuclear terms needed
    if np.allclose(A, 0.0) and not include_nuclear:
        return mu_0

    # 2. Shift to origin A: mu_A = mu_0 + N_elec * A
    ovlp_cross = gto.intor_cross("int1e_ovlp", auxmol, ghost_mol).flatten()
    n_elec = np.dot(ri_coeffs, ovlp_cross * _NORM_FACTOR)

    mu_elec = mu_0 + n_elec * A

    if include_nuclear:
        mu_nuc = np.sum(auxmol.atom_charges()[:, None] * auxmol.atom_coords(), axis=0)
        # shift nuclear dipole to origin A: mu_nuc_A = mu_nuc - N_nuc * A
        n_nuc = auxmol.atom_charges().sum()
        mu_nuc -= n_nuc * A
        return mu_elec + mu_nuc

    return mu_elec


def apply_dipole_correction(
    auxmol: gto.Mole,
    ri_coeffs: np.ndarray,
    delta_mu_bohr: np.ndarray,
) -> np.ndarray:
    """
    Minimum-norm, charge-neutral correction to RI coefficients to fix the dipole.

    Finds the smallest ``||δc||`` satisfying:

    - ``Σ_i δc_i ∫ r_α χ_i dr = -Δμ_α`` (adjusts the electronic dipole)
    - ``Σ_i δc_i ∫ χ_i dr = 0`` (charge-neutral)

    :param auxmol: PySCF auxiliary Mole object.
    :param ri_coeffs: Current RI coefficients, shape ``(nao,)``.
    :param delta_mu_bohr: Dipole change needed, shape ``(3,)``, in e·Bohr.
    :return: Corrected RI coefficients, shape ``(nao,)``.
    """
    ghost_mol = _get_or_create_ghost(None)

    ovlp_cross = gto.intor_cross("int1e_ovlp", auxmol, ghost_mol).flatten()
    S = ovlp_cross * _NORM_FACTOR  # ∫ χ_i dr, shape (nao,)

    dip_cross = gto.intor_cross("int1e_r", auxmol, ghost_mol, comp=3)
    D = dip_cross.reshape(3, auxmol.nao) * _NORM_FACTOR  # D[α,i] = ∫ r_α χ_i dr

    # Coulomb-metric inner product: <a, b>_J = a^T J b, where J = (P|Q) is SPD.
    J = auxmol.intor("int2c2e")
    cho = scipy.linalg.cho_factor(J)
    Jinv_S = scipy.linalg.cho_solve(cho, S)  # J^{-1} S, shape (nao,)
    Jinv_D = scipy.linalg.cho_solve(cho, D.T)  # J^{-1} D^T, shape (nao, 3)

    # J-project D against S so the correction is charge-neutral:
    # coef[α] = <D[α], S>_J / <S, S>_J = (D @ Jinv_S)[α] / (S @ Jinv_S)
    coef = (D @ Jinv_S) / (S @ Jinv_S)  # (3,)
    Jinv_Dperp = Jinv_D - np.outer(Jinv_S, coef)  # J^{-1} D_perp^T, shape (nao, 3)

    # Minimum J-norm correction: δc = J^{-1} D_perp^T λ, where G λ = -Δμ
    G = D @ Jinv_Dperp  # (3, 3) Coulomb-metric Gram matrix
    lam = np.linalg.solve(G, -delta_mu_bohr)
    delta_c = Jinv_Dperp @ lam

    return ri_coeffs + delta_c


def get_ri_quadrupole(
    auxmol: gto.Mole,
    ri_coeffs: np.ndarray,
    origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    *,
    ghost: gto.Mole | None = None,
    include_nuclear: bool = False,
) -> np.ndarray:
    """
    Returns the primitive electronic quadrupole tensor shifted to the target origin.
    The quadrupole is returned in its primitive form (without traceless correction)
    to preserve the simple translation properties.

    :param auxmol: PySCF auxiliary Mole returned by :class:`MetatomicDensityPredictor`
    :param ri_coeffs: RI fitting coefficients, shape ``(naux,)``
    :param origin: Expansion origin in **Bohr**.  For origin-independent dipoles use the
        ligand centre of mass or centre of nuclear charge.
    :param ghost: Optional pre-configured ghost molecule; if None, a default flat
        Gaussian at the global origin is used.
    :param include_nuclear: If True, the total quadrupole (nuclear + electronic) is
        returned; otherwise, only the electronic contribution is returned.
    :return: Primitive electronic quadrupole tensor (shape ``(3, 3)``) in atomic units,
        shifted to the target origin
    """
    ghost_mol = _get_or_create_ghost(ghost)
    A = np.asarray(origin)

    # 1. Compute primitive Quadrupole at global origin (0,0,0)
    quad_cross = gto.intor_cross("int1e_rr", auxmol, ghost_mol, comp=9)
    int_rr = quad_cross.reshape(9, auxmol.nao) * _NORM_FACTOR
    Q_0 = -np.dot(int_rr, ri_coeffs).reshape(3, 3)

    # Fast path if no origin shift is requested and no nuclear terms needed
    if np.allclose(A, 0.0) and not include_nuclear:
        return Q_0

    # 2. Get lower moments for analytical translation
    ovlp_cross = gto.intor_cross("int1e_ovlp", auxmol, ghost_mol).flatten()
    n_elec = np.dot(ri_coeffs, ovlp_cross * _NORM_FACTOR)

    dip_cross = gto.intor_cross("int1e_r", auxmol, ghost_mol, comp=3)
    int_r = dip_cross.reshape(3, auxmol.nao) * _NORM_FACTOR
    mu_0 = -np.dot(int_r, ri_coeffs)

    # 3. Shift to origin A: Q_A = Q_0 - (A ⊗ mu_0) - (mu_0 ⊗ A) - N_elec*(A ⊗ A)
    A_out_mu = np.outer(A, mu_0)
    mu_out_A = np.outer(mu_0, A)
    A_out_A = np.outer(A, A)

    Q_elec = Q_0 - A_out_mu - mu_out_A - (n_elec * A_out_A)

    if include_nuclear:
        # Shift coordinates relative to 'origin' A
        shifted_coords = auxmol.atom_coords() - A
        charges = auxmol.atom_charges()

        Q_nuc_A = np.einsum("i,ix,iy->xy", charges, shifted_coords, shifted_coords)

        return Q_elec + Q_nuc_A

    return Q_elec
