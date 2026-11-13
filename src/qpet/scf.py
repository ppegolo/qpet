"""PySCF calculations started from a predicted density."""

from __future__ import annotations

import numpy as np
import scipy.linalg
from pyscf import df, dft, gto
from pyscf.dft import numint
from pyscf.scf import hf

from qpet.model import Prediction

_GRID_CHUNK = 16384
_J_FLOATS = 100_000_000


def _coulomb(mol: gto.Mole, auxmol: gto.Mole, coeffs: np.ndarray) -> np.ndarray:
    """``J_μν = Σ_k (μν|k) c_k``, a slice of auxiliary shells at a time."""
    j = np.zeros((mol.nao, mol.nao))
    ao_loc = auxmol.ao_loc_nr()
    per_function = mol.nao**2
    s0 = 0
    while s0 < auxmol.nbas:
        s1 = s0 + 1
        while (
            s1 < auxmol.nbas
            and (ao_loc[s1 + 1] - ao_loc[s0]) * per_function < _J_FLOATS
        ):
            s1 += 1
        shls = (0, mol.nbas, 0, mol.nbas, s0, s1)
        j += (
            df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1", shls_slice=shls)
            @ coeffs[ao_loc[s0] : ao_loc[s1]]
        )
        s0 = s1
    return j


def _xc(
    mol: gto.Mole, prediction: Prediction, grids: dft.Grids, xc: str, spin: bool
) -> np.ndarray:
    """``V_xc`` of the prediction (LDA or GGA); (2, nao, nao) with ``spin``."""
    xctype = dft.libxc.xc_type(xc)
    deriv = 0 if xctype == "LDA" else 1
    ni = numint.NumInt()
    auxmol = prediction.rho.auxmol
    m = (
        prediction.spin.coeffs
        if spin and prediction.spin is not None
        else 0 * prediction.rho.coeffs
    )
    vmat = np.zeros((2 if spin else 1, mol.nao, mol.nao))
    for p0 in range(0, grids.weights.size, _GRID_CHUNK):
        coords, weights = (
            grids.coords[p0 : p0 + _GRID_CHUNK],
            grids.weights[p0 : p0 + _GRID_CHUNK],
        )
        aux = numint.eval_ao(auxmol, coords, deriv=deriv)
        ao = numint.eval_ao(mol, coords, deriv=deriv)
        rho, mag = aux @ prediction.rho.coeffs, aux @ m
        if spin:
            rho = np.stack([(rho + mag) / 2, (rho - mag) / 2])
            if deriv:
                rho[:, 0] = np.maximum(rho[:, 0], 0)
            else:
                rho = np.maximum(rho, 0)
        vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype=xctype)[1]
        wv = (weights * vxc).reshape(len(vmat), -1, len(weights))
        wv[:, 0] *= 0.5
        for s in range(len(vmat)):
            if deriv:
                vmat[s] += ao[0].T @ np.einsum("xgi,xg->gi", ao, wv[s])
            else:
                vmat[s] += (ao * wv[s][0][:, None]).T @ ao
    vmat = vmat + vmat.transpose(0, 2, 1)
    return vmat if spin else vmat[0]


def to_scf(
    prediction: Prediction, mol: gto.Mole, xc: str, *, guess_xc: str = "pbe", **options
):
    """
    PySCF Kohn-Sham object that starts from the predicted density.

    RKS for a closed shell, UKS when ``mol`` is open-shell or the prediction
    carries a spin density (a broken-symmetry singlet). The initial guess is one
    diagonalization in the potential of the prediction: ``h + J[ρ] + V_xc[ρα, ρβ]``
    with ``ρα,β = (ρ ± m)/2``, J from the RI coefficients and ``V_xc`` from the
    semilocal ``guess_xc`` (exact exchange needs orbitals). Then
    ``to_scf(p, mol, "wb97m-v").kernel()`` runs the SCF from the prediction, and
    the object is used as any PySCF calculation.

    :param options: set on the object (``conv_tol``, ``max_cycle``, ``chkfile``, ...).
    """
    unrestricted = mol.spin > 0 or (
        prediction.spin is not None and prediction.spin.coeffs.any()
    )
    mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=xc).set(**options)
    fock = hf.get_hcore(mol) + _coulomb(
        mol, prediction.rho.auxmol, prediction.rho.coeffs
    )
    fock = fock + _xc(mol, prediction, mf.grids.build(), guess_xc, unrestricted)
    s = mf.get_ovlp()

    def occupied(f: np.ndarray, n: int) -> np.ndarray:
        c = scipy.linalg.eigh(f, s)[1][:, :n]
        return c @ c.T

    if unrestricted:
        mf.init_guess = np.array(
            [occupied(f, n) for f, n in zip(fock, mol.nelec, strict=True)]
        )
    else:
        mf.init_guess = 2 * occupied(fock, mol.nelectron // 2)
    return mf
