"""RI-density observables and coefficient corrections.

Multipoles (charge, dipole, quadrupole) are extracted with the Ghost Trick:
cross integrals against a near-flat Gaussian ghost orbital, rescaled by its
normalization, converge to the analytic moment integrals
``∫ r^k χ_i(r) dr`` as the ghost exponent goes to zero.

The analytic S vector (``∫ χ_i dr``, l=0 shells only) provides the same k=0
integral in closed form and drives the minimum-norm charge normalization;
a test pins the two implementations against each other.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg
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

    # Minimum J-norm correction: δc = J^{-1} D_perp^T λ, where G λ = -Δμ.
    # lstsq handles rank-deficient G — e.g. a linear molecule on an axis where
    # D_perp spans fewer than 3 dipole directions — returning the minimum-norm
    # solution and silently dropping the unachievable component.
    G = D @ Jinv_Dperp  # (3, 3) Coulomb-metric Gram matrix
    lam, *_ = np.linalg.lstsq(G, -delta_mu_bohr, rcond=None)
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


#: Mean charge error of the *reference* RI fits (electrons): an unconstrained
#: Coulomb-metric fit of a def2 density in ``def2-universal-jfit`` integrates to
#: slightly more than the exact electron count. Measured at +0.0117 e over the
#: labelled pocket sets; used as the default target offset at inference so the
#: projection aims at what the references actually carry.
DEFAULT_CHARGE_OFFSET = 0.0117


#: Relative diagonal shift of the first regularised retry of the Coulomb
#: solve; doubled on every further retry up to :data:`_MAX_REGULARISATION`.
_MIN_REGULARISATION = 1e-10
_MAX_REGULARISATION = 1e-4


def _coulomb_direction(
    J: np.ndarray, S_vec: np.ndarray, label: str | None
) -> np.ndarray:
    """
    ``J⁻¹ S`` by Cholesky, regularised if the metric is not positive definite.

    On large systems the auxiliary basis can be numerically linearly dependent
    (diffuse functions of many nearby atoms), so ``J`` is positive
    semi-definite to working precision and the factorisation fails. A Tikhonov
    shift ``J + ε diag(J)`` damps exactly those directions; ``ε`` starts tiny
    and doubles until the factorisation succeeds, and a warning reports it.
    """
    try:
        return scipy.linalg.cho_solve(scipy.linalg.cho_factor(J), S_vec)
    except np.linalg.LinAlgError:
        pass
    diagonal = np.diag(J).copy()
    epsilon = _MIN_REGULARISATION
    while epsilon <= _MAX_REGULARISATION:
        try:
            factor = scipy.linalg.cho_factor(J + np.diag(epsilon * diagonal))
        except np.linalg.LinAlgError:
            epsilon *= 2.0
            continue
        warnings.warn(
            f"Coulomb metric of {label or 'system'} is not positive definite "
            f"(naux={len(S_vec)}); charge projection used a relative diagonal "
            f"shift of {epsilon:.1e}.",
            RuntimeWarning,
            stacklevel=3,
        )
        return scipy.linalg.cho_solve(factor, S_vec)
    raise np.linalg.LinAlgError(
        f"Coulomb metric of {label or 'system'} could not be factorised even "
        f"with a relative diagonal shift of {_MAX_REGULARISATION:g}."
    )


def normalize_charge_additive(
    auxmol: gto.Mole,
    coeffs: np.ndarray,
    n_electrons: float,
    *,
    metric: str = "identity",
    max_shift: float | None = None,
    label: str | None = None,
) -> np.ndarray:
    """
    Shift the coefficients so the density integrates to ``n_electrons``.

    The correction is the smallest one, in the chosen metric, that fixes the
    electron count:

    - ``"identity"``: ``δc ∝ S`` (``S_i = ∫χ_i dr``), i.e. only the s-function
      coefficients move and the l>0 (shape) channels are untouched. Cheap, but
      it dumps the missing charge into the tight s functions, which the
      potential outside the molecule barely sees.
    - ``"coulomb"``: ``δc ∝ J⁻¹ S`` with ``J = (P|Q)``, the correction of
      minimum Coulomb self-energy. It spreads the charge over the diffuse
      functions of every atom, which is what fixes the electrostatic potential
      at the molecular surface. Costs one ``int2c2e`` and one Cholesky
      factorisation of the auxiliary basis.

    :param auxmol: PySCF auxiliary Mole object.
    :param coeffs: RI coefficients, shape (naux,). Not modified.
    :param n_electrons: Target electron count.
    :param metric: ``"identity"`` or ``"coulomb"``.
    :param max_shift: If given and the correction would move more than this
        many electrons, the coefficients are returned unchanged and a warning
        is issued. A gap that large is not fitting noise, and spreading it over
        the basis would turn a local failure into a global one. The limit has
        to scale with the system: see ``DensityModel.charge_guard_relative``.
    :param label: Name of the system, used in the ``max_shift`` warning.
    :return: Corrected coefficients, shape (naux,).
    """
    S_vec = compute_S_vector(auxmol)
    shift = n_electrons - S_vec @ coeffs
    if max_shift is not None and abs(shift) > max_shift:
        warnings.warn(
            f"charge projection skipped for {label or 'system'}: the predicted "
            f"density is off by {shift:+.3f} e (limit {max_shift:g} e), far "
            f"more than fitting noise; spreading it over the basis would turn "
            f"a local failure into a global one.",
            RuntimeWarning,
            stacklevel=2,
        )
        return coeffs.copy()
    if metric == "identity":
        direction = S_vec
    elif metric == "coulomb":
        direction = _coulomb_direction(auxmol.intor("int2c2e"), S_vec, label)
    else:
        raise ValueError(f"Unknown metric {metric!r}. Use 'identity' or 'coulomb'.")
    gamma = shift / (S_vec @ direction)
    return coeffs + gamma * direction
