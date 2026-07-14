"""SCF initial guess from an RI (auxiliary-basis) electron density.

Builds one Fock matrix from a predicted density — the Coulomb term
analytically from the RI coefficients (``J_ij = Σ_P c_P (ij|P)``), the XC
term numerically on a DFT grid — and diagonalizes it once. The resulting
density matrix is a high-quality ``dm0`` for
:meth:`pyscf.scf.hf.SCF.kernel`, typically saving several SCF cycles over
the default guess.

Two approximations, both inherent to starting from a density instead of
orbitals:

- Hybrid functionals: the exact-exchange term needs a density matrix, so the
  standard initial guess (minao) is used for that contribution only.
- Meta-GGAs: the kinetic-energy density is approximated by the von
  Weizsäcker form ``|∇ρ|² / 8ρ`` (exact for a single orbital).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from pyscf import df, dft, gto
from pyscf.dft import numint
from pyscf.scf import hf as scf_hf

from qpet.moments import normalize_charge_additive

# Grid points processed per batch when assembling the XC matrix; bounds the
# (chunk, nao) AO-value intermediates.
_GRID_CHUNK = 16384


def _xc_matrix(
    mol: gto.Mole,
    auxmol: gto.Mole,
    coeffs: np.ndarray,
    grids: dft.Grids,
    xc: str,
) -> np.ndarray:
    """
    XC potential matrix ``V_xc[ρ_RI]`` in the AO basis of ``mol``.

    The density (and its gradient for GGA/MGGA) is evaluated on the grid
    directly from the auxiliary-basis expansion.
    """
    xctype = dft.libxc.xc_type(xc)
    nao = mol.nao
    vmat = np.zeros((nao, nao))
    if xctype == "HF":  # pure exact exchange: no grid contribution
        return vmat

    deriv = 0 if xctype == "LDA" else 1
    ni = numint.NumInt()

    for p0 in range(0, grids.weights.size, _GRID_CHUNK):
        p1 = min(p0 + _GRID_CHUNK, grids.weights.size)
        coords = grids.coords[p0:p1]
        weights = grids.weights[p0:p1]
        ao_aux = numint.eval_ao(auxmol, coords, deriv=deriv)
        ao = numint.eval_ao(mol, coords, deriv=deriv)

        if xctype == "LDA":
            rho = ao_aux @ coeffs
        else:
            rho = np.einsum("xgi,i->xg", ao_aux, coeffs)
            if xctype == "MGGA":
                # von Weizsäcker tau; the floor avoids 0/0 in the far field.
                tau = (rho[1:4] ** 2).sum(axis=0) / (8.0 * np.maximum(rho[0], 1e-30))
                rho = np.vstack([rho, tau])

        vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype=xctype)[1]
        wv = weights * vxc  # (nvar, ngrid)

        # Symmetric contributions are halved here and restored by the final
        # ``vmat + vmat.T``.
        wv[0] *= 0.5
        if xctype == "LDA":
            vmat += (ao * wv[0][:, None]).T @ ao
        else:
            aow = np.einsum("xgi,xg->gi", ao[:4], wv[:4])
            vmat += ao[0].T @ aow
            if xctype == "MGGA":
                # tau matrix element: 0.5 Σ_x ∂φ_i ∂φ_j v_tau (then halved
                # again for the symmetrization) → factor 0.25.
                wtau = 0.25 * wv[4]
                for x in (1, 2, 3):
                    vmat += (ao[x] * wtau[:, None]).T @ ao[x]

    return vmat + vmat.T


def dm_from_auxdensity(
    mol: gto.Mole,
    auxmol: gto.Mole,
    coeffs: np.ndarray,
    xc: str = "pbe",
    *,
    normalize_charge: bool = False,
) -> np.ndarray:
    """
    SCF initial-guess density matrix from RI density-fitting coefficients.

    Assembles ``F = h_core + J[c] + V_xc[ρ_RI]`` (plus a minao-based exact
    exchange term for hybrids), diagonalizes once, and returns the
    closed-shell density matrix of the occupied orbitals — effectively one
    free SCF iteration from the predicted density.

    :param mol: Built PySCF Mole (orbital basis) for the SCF calculation.
    :param auxmol: Auxiliary-basis Mole matching ``coeffs``.
    :param coeffs: RI density coefficients, shape (naux,).
    :param xc: Exchange-correlation functional (LDA/GGA/MGGA/global hybrid;
        range-separated hybrids are not supported).
    :param normalize_charge: If True, apply a minimum-norm additive shift to
        the s-channel coefficients so the density integrates to
        ``mol.nelectron`` before building the Fock matrix.
    :return: Initial-guess density matrix ``dm0``, shape (nao, nao), for
        ``mf.kernel(dm0=dm0)``.
    :raises NotImplementedError: For open-shell molecules or range-separated
        hybrid functionals.
    :raises ValueError: If ``coeffs`` does not match the auxiliary basis.
    """
    if mol.spin != 0:
        raise NotImplementedError(
            "dm_from_auxdensity supports closed-shell (RKS) molecules only"
        )
    if coeffs.shape != (auxmol.nao,):
        raise ValueError(
            f"coeffs shape {coeffs.shape} does not match auxiliary basis size "
            f"({auxmol.nao},)"
        )
    omega = dft.libxc.rsh_coeff(xc)[0]
    if omega != 0:
        raise NotImplementedError(f"range-separated hybrid '{xc}' is not supported")

    if normalize_charge:
        coeffs = normalize_charge_additive(auxmol, coeffs, mol.nelectron)

    grids = dft.Grids(mol)
    grids.build()

    fock = scf_hf.get_hcore(mol)
    fock = fock + _xc_matrix(mol, auxmol, coeffs, grids, xc)

    # Coulomb term directly from the RI expansion. Incore 3-center integrals:
    # peak memory nao² × naux floats.
    j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    fock += j3c @ coeffs

    hyb = dft.libxc.hybrid_coeff(xc)
    if hyb != 0:
        # Exact exchange needs a density matrix; the RI density provides
        # none, so use the standard initial guess for this term only.
        dm_minao = scf_hf.init_guess_by_minao(mol)
        vk = scf_hf.get_jk(mol, dm_minao, hermi=1)[1]
        fock -= 0.5 * hyb * vk

    s1e = mol.intor_symmetric("int1e_ovlp")
    _, mo_coeff = scipy.linalg.eigh(fock, s1e)
    nocc = mol.nelectron // 2
    mocc = mo_coeff[:, :nocc]
    return 2.0 * (mocc @ mocc.T)
