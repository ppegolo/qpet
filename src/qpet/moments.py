"""RI-density observables and coefficient corrections.

Multipoles (charge, dipole, quadrupole) are extracted with the Ghost Trick:
cross integrals against a near-flat Gaussian ghost orbital, rescaled by its
normalization, converge to the analytic moment integrals
``∫ r^k χ_i(r) dr`` as the ghost exponent goes to zero.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto

from qpet.conditioning import nuclear_charges_of

# Ghost Atom Configuration
_GHOST_ALPHA = 1e-12
_NORM_FACTOR = 1.0 / (2 * _GHOST_ALPHA / np.pi) ** 0.75
_DEFAULT_GHOST: gto.Mole | None = None


def _get_or_create_ghost(ghost: gto.Mole | None) -> gto.Mole:
    """Returns the provided ghost Mole, or the module-level cached default."""
    global _DEFAULT_GHOST
    if ghost is not None:
        return ghost
    if _DEFAULT_GHOST is None:
        _DEFAULT_GHOST = gto.M(
            atom=[("X", (0.0, 0.0, 0.0))],
            basis={"X": [[0, [_GHOST_ALPHA, 1.0]]]},
            spin=0,
            charge=0,
        )
    return _DEFAULT_GHOST


def get_ri_charge(
    auxmol: gto.Mole,
    ri_coeffs: np.ndarray,
    *,
    ghost: gto.Mole | None = None,
    include_nuclear: bool = False,
) -> np.ndarray:
    """
    Returns the electronic charge (-N_elec) as a 0D numpy array.

    :param auxmol: PySCF auxiliary Mole from a DensityCalculator or RICalculator
    :param ri_coeffs: RI fitting coefficients, shape ``(naux,)``
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
        n_nuc = nuclear_charges_of(auxmol).sum()
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

    :param auxmol: PySCF auxiliary Mole from a DensityCalculator or RICalculator
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
        z_nuc = nuclear_charges_of(auxmol)
        mu_nuc = np.sum(z_nuc[:, None] * auxmol.atom_coords(), axis=0)
        # shift nuclear dipole to origin A: mu_nuc_A = mu_nuc - N_nuc * A
        n_nuc = z_nuc.sum()
        mu_nuc -= n_nuc * A
        return mu_elec + mu_nuc

    return mu_elec


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

    :param auxmol: PySCF auxiliary Mole from a DensityCalculator or RICalculator
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
        charges = nuclear_charges_of(auxmol)

        Q_nuc_A = np.einsum("i,ix,iy->xy", charges, shifted_coords, shifted_coords)

        return Q_elec + Q_nuc_A

    return Q_elec
