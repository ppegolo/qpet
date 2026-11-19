"""PySCF calculations started from a predicted density."""

from __future__ import annotations

import contextlib
import importlib

import numpy as np
from pyscf import df, dft, gto, lib
from pyscf.dft import numint

from qpet.model import Prediction

_GRID_CHUNK = 4096
_J_FLOATS = 100_000_000

#: GGA functional of the guess for a meta-GGA target, whose τ is not known from
#: ρ. Other meta-GGAs fall back to PBE.
GGA_FOR_META_GGA = {"WB97M-V": "wb97x-v", "WB97M-D4": "wb97x-v"}


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


def _rho_on_grid(auxmol: gto.Mole, coeffs: np.ndarray, grids):
    """
    ρ and ∇ρ of ``Σ_k c_k φ_k`` for each row of ``coeffs``, (n, 4, ngrids),
    zero where the fit goes negative. On the GPU for a gpu4pyscf grid.
    """
    if isinstance(grids.coords, np.ndarray):
        rho = np.empty((len(coeffs), 4, grids.weights.size))
        for p0 in range(0, grids.weights.size, _GRID_CHUNK):
            aux = numint.eval_ao(auxmol, grids.coords[p0 : p0 + _GRID_CHUNK], deriv=1)
            rho[..., p0 : p0 + _GRID_CHUNK] = np.einsum("xgk,ck->cxg", aux, coeffs)
    else:
        import cupy
        from gpu4pyscf.dft.numint import NumInt

        ni = NumInt().build(auxmol, None)
        opt = ni.gdftopt
        c = opt.sort_orbitals(cupy.asarray(coeffs), axis=[1])
        rho = cupy.empty((len(coeffs), 4, grids.weights.size))
        # the grid caches the screening of one basis: the orbital one, not this
        orbital_screening, grids._non0ao_idx = grids._non0ao_idx, None
        try:
            p1 = 0
            for ao, idx, weight, _ in ni.block_loop(
                opt._sorted_mol, grids, opt._sorted_mol.nao, 1
            ):
                p0, p1 = p1, p1 + weight.size
                rho[..., p0:p1] = cupy.einsum("xkg,ck->cxg", ao, c[:, idx])
        finally:
            grids._non0ao_idx = orbital_screening
    for r in rho:
        r[:, r[0] < 0] = 0
    return rho


class _PredictedDensity(numint.NumInt):
    """
    ``NumInt`` that reads the density of a DM tagged with ``aux_rho``.

    PySCF's ``nr_rks`` and ``nr_uks`` see the density only
    through :meth:`_gen_rho_evaluator`; here it returns the tagged ρ and ∇ρ
    (4, ngrids) block by block.
    """

    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        rho = getattr(dms, "aux_rho", None)
        if rho is None:
            return super()._gen_rho_evaluator(mol, dms, hermi, with_lapl, grids)
        p1 = 0

        def make_rho(idm, ao, mask, xctype):
            nonlocal p1
            p0, p1 = p1, p1 + ao.shape[-2]
            return (rho[0, p0:p1] if xctype == "LDA" else rho[:, p0:p1]).copy()

        return make_rho, 1, mol.nao


@contextlib.contextmanager
def _predicted_density_on_gpu(channels):
    """
    gpu4pyscf's ``nr_rks`` and ``nr_uks`` evaluate ρ with the
    module function ``eval_rho``, per grid block in order, each GPU on its
    range of the grid, α then β for UKS. Within this context ``eval_rho``
    returns the next block of ``channels`` (n, 4, ngrids) instead.
    """
    import cupy
    import gpu4pyscf.dft.numint as gpu_numint

    ngrids = channels.shape[-1]
    position: dict[int, list[int]] = {}

    def eval_rho(mol, ao, dm, *args, xctype="LDA", **kwargs):
        device = cupy.cuda.Device().id
        start, calls = position.setdefault(
            device, [gpu_numint.gen_grid_range(ngrids, device)[0], 0]
        )
        channel = calls % len(channels)
        block = channels[channel, :, start : start + ao.shape[-1]]
        position[device] = [
            start + ao.shape[-1] if channel == len(channels) - 1 else start,
            calls + 1,
        ]
        return cupy.array(block[0] if xctype == "LDA" else block)

    original = gpu_numint.eval_rho
    gpu_numint.eval_rho = eval_rho
    try:
        yield
    finally:
        gpu_numint.eval_rho = original


def to_scf(prediction: Prediction, mf):
    """
    Set the initial guess of the Kohn-Sham object ``mf`` from the prediction.

    The guess is the occupied space of the Fock matrix of ``mf``'s functional
    with J and the semilocal terms from the predicted ρ (and m for UKS:
    ρα,β = (ρ ± m)/2). The exact exchange of a hybrid needs orbitals: it comes
    from those of the same Fock matrix with PBE in place of the hybrid's
    semilocal part, which alone is an incomplete functional. τ is not known
    from ρ, so a meta-GGA is replaced in the guess by :data:`GGA_FOR_META_GGA`
    (the semilocal part of ωB97X-V for ωB97M), and VV10, whose cost grows with
    the square of the grid, is left out; the SCF itself is ``mf``'s
    functional. The models are trained on ωB97M densities, best matched by
    ωB97M-D4 (dispersion does not enter the density); another functional
    starts off its target.

    The grids, exact exchange, diagonalization and occupations are ``mf``'s,
    and so is its ``NumInt`` code: a GPU object (``.to_gpu()``) computes the
    whole guess on the GPU, J with gpu4pyscf's 3-centre integrals. Returns ``mf``:
    ``to_scf(p, dft.RKS(mol, xc="wb97m-d4")).kernel()`` runs the SCF from the
    prediction.
    """
    mol = mf.mol
    xc = mf.xc
    if dft.libxc.xc_type(xc) == "MGGA":
        xc = GGA_FOR_META_GGA.get(xc.upper(), "pbe")
    unrestricted = mf.istype("UHF")
    auxmol, rho = prediction.rho.auxmol, prediction.rho.coeffs
    m = prediction.spin.coeffs if prediction.spin is not None else 0 * rho
    h = mf.get_hcore()
    xp = importlib.import_module(type(h).__module__.split(".")[0])
    gpu = xp is not np

    grids = mf.grids
    if grids.coords is None:  # built and pruned as mf does on its first iteration
        grids.build()
        total = _rho_on_grid(auxmol, rho[None], grids)[0, 0]
        grids.prune_by_density_(total, mf.small_rho_cutoff)
    channels = np.stack([(rho + m) / 2, (rho - m) / 2]) if unrestricted else rho[None]
    on_grid = _rho_on_grid(auxmol, channels, grids)
    kind = "nr_uks" if unrestricted else "nr_rks"

    def semilocal(name):
        """V_xc of ``name`` on the predicted density, with ``mf``'s ``NumInt`` code."""
        if gpu:
            from gpu4pyscf.dft.numint import NumInt

            dm = xp.zeros((mol.nao, mol.nao))
            dms = (dm, dm) if unrestricted else dm
            with _predicted_density_on_gpu(on_grid):
                return getattr(NumInt(), kind)(mol, grids, name, dms)[2]
        tagged = [
            lib.tag_array(np.zeros((mol.nao, mol.nao)), aux_rho=v) for v in on_grid
        ]
        dms = tagged if unrestricted else tagged[0]
        return getattr(_PredictedDensity(), kind)(mol, grids, name, dms)[2]

    if gpu:
        from gpu4pyscf.df.int3c2e_bdiv import contract_int3c2e_auxvec

        fock = h + contract_int3c2e_auxvec(mol, auxmol, xp.asarray(rho))
    else:
        fock = h + _coulomb(mol, auxmol, rho)
    s = mf.get_ovlp()

    def density_matrix(f):
        e, c = mf.eig(f, s)
        return mf.make_rdm1(c, mf.get_occ(e, c))

    if not dft.libxc.is_hybrid_xc(xc):
        mf.init_guess = density_matrix(fock + semilocal(xc))
        return mf
    dm = density_matrix(fock + semilocal("pbe"))
    omega, alpha, hyb = numint.NumInt().rsh_and_hybrid_coeff(xc, spin=mol.spin)
    vk = hyb * mf.get_k(mol, dm) if hyb else 0
    if omega and alpha != hyb:
        vk = vk + (alpha - hyb) * mf.get_k(mol, dm, omega=omega)
    mf.init_guess = density_matrix(
        fock + semilocal(xc) - (vk if unrestricted else 0.5 * vk)
    )
    return mf
