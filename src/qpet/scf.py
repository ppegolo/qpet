"""PySCF calculations started from a predicted density."""

from __future__ import annotations

import importlib

import numpy as np
from pyscf import df, dft, gto, lib
from pyscf.dft import numint

from qpet.model import Prediction

_GRID_CHUNK = 16384
_J_FLOATS = 100_000_000

#: GGA functional of the guess for a meta-GGA target, whose τ is not known from
#: ρ. Other meta-GGAs fall back to PBE.
GGA_FOR_META_GGA = {"WB97M-V": "wb97x-v"}


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


class _PredictedDensity(numint.NumInt):
    """
    ``NumInt`` that reads the density of a DM tagged with ``aux_coeffs``.

    PySCF's ``nr_rks``, ``nr_uks`` and ``nr_nlc_vxc`` see the density only
    through :meth:`_gen_rho_evaluator`; here it returns ρ and ∇ρ of
    ``Σ_k c_k φ_k`` on the grid, zero where the fit goes negative.
    """

    def __init__(self, auxmol: gto.Mole) -> None:
        super().__init__()
        self.auxmol = auxmol

    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        coeffs = getattr(dms, "aux_coeffs", None)
        if coeffs is None:
            return super()._gen_rho_evaluator(mol, dms, hermi, with_lapl, grids)
        rho = np.empty((4, grids.weights.size))
        for p0 in range(0, grids.weights.size, _GRID_CHUNK):
            aux = numint.eval_ao(
                self.auxmol, grids.coords[p0 : p0 + _GRID_CHUNK], deriv=1
            )
            rho[:, p0 : p0 + _GRID_CHUNK] = aux @ coeffs
        rho[:, rho[0] < 0] = 0
        p1 = 0

        def make_rho(idm, ao, mask, xctype):
            nonlocal p1
            p0, p1 = p1, p1 + ao.shape[-2]
            return (rho[0, p0:p1] if xctype == "LDA" else rho[:, p0:p1]).copy()

        return make_rho, 1, mol.nao


def to_scf(prediction: Prediction, mf):
    """
    Set the initial guess of the Kohn-Sham object ``mf`` from the prediction.

    The guess is the occupied space of the Fock matrix of ``mf``'s functional
    with J, the semilocal and the VV10 terms from the predicted ρ (and m for
    UKS: ρα,β = (ρ ± m)/2). The exact exchange of a hybrid needs orbitals: it
    comes from those of the same Fock matrix with PBE in place of the hybrid's
    semilocal part, which alone is an incomplete functional. τ is not known
    from ρ, so a meta-GGA is replaced in the guess by :data:`GGA_FOR_META_GGA`
    (ωB97X-V for ωB97M-V); the SCF itself is ``mf``'s functional. The models
    are trained on ωB97M-V densities: another functional starts off its target.

    Exact exchange, diagonalization and occupations use ``mf``'s own methods,
    so a GPU object (``.to_gpu()``) computes them on the GPU. Returns ``mf``:
    ``to_scf(p, dft.RKS(mol, xc="wb97m-v")).kernel()`` runs the SCF from the
    prediction.
    """
    mol = mf.mol
    ni = _PredictedDensity(prediction.rho.auxmol)
    xc = mf.xc
    if ni._xc_type(xc) == "MGGA":
        xc = GGA_FOR_META_GGA.get(xc.upper(), "pbe")
    unrestricted = mf.istype("UHF")
    rho = prediction.rho.coeffs
    m = prediction.spin.coeffs if prediction.spin is not None else 0 * rho
    grids = dft.Grids(mol)
    grids.level = mf.grids.level
    grids.build()

    def tagged(coeffs):
        return lib.tag_array(np.zeros((mol.nao, mol.nao)), aux_coeffs=coeffs)

    def semilocal(name):
        if unrestricted:
            dms = [tagged((rho + m) / 2), tagged((rho - m) / 2)]
            return xp.asarray(ni.nr_uks(mol, grids, name, dms)[2])
        return xp.asarray(ni.nr_rks(mol, grids, name, tagged(rho))[2])

    h = mf.get_hcore()
    xp = importlib.import_module(type(h).__module__.split(".")[0])
    fock = _coulomb(mol, prediction.rho.auxmol, rho)
    nlc = xc if dft.libxc.is_nlc(xc) else (mf.nlc if xc == mf.xc and mf.nlc else None)
    if nlc:
        nlcgrids = dft.Grids(mol)
        nlcgrids.level = mf.nlcgrids.level
        nlcgrids.build()
        fock = fock + ni.nr_nlc_vxc(mol, nlcgrids, nlc, tagged(rho))[2]
    fock = h + xp.asarray(fock)
    s = mf.get_ovlp()

    def density_matrix(f):
        e, c = mf.eig(f, s)
        return mf.make_rdm1(c, mf.get_occ(e, c))

    if not dft.libxc.is_hybrid_xc(xc):
        mf.init_guess = density_matrix(fock + semilocal(xc))
        return mf
    dm = density_matrix(fock + semilocal("pbe"))
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
    vk = hyb * mf.get_k(mol, dm) if hyb else 0
    if omega and alpha != hyb:
        vk = vk + (alpha - hyb) * mf.get_k(mol, dm, omega=omega)
    mf.init_guess = density_matrix(
        fock + semilocal(xc) - (vk if unrestricted else 0.5 * vk)
    )
    return mf
