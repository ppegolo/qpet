"""Exact Coulomb-metric solves ``J d = s`` for large auxiliary bases.

``J = (P|Q)`` is never formed. A product ``J x`` is split by atom distance:

- *near field*: atom pairs within ``rnear`` use the exact ``int2c2e`` blocks,
  computed once and stored (float64, upper triangle by atom);
- *far field*: beyond ``rnear`` the auxiliary functions no longer overlap, and a
  solid-harmonic Gaussian has a single exact multipole moment, so each atom is
  replaced by its multipoles (``l <= 4``) and pairs interact through the
  Cartesian derivatives of ``1/r``. The remaining error is the Gaussian overlap of
  the most diffuse functions: 3e-5 in the J-norm at 7 Å, 6e-7 at 8 Å.

The system is solved by preconditioned conjugate gradients with an overlapping
additive Schwarz preconditioner, ``M⁻¹ = Σ_D R_Dᵀ J_DD⁻¹ R_D``, whose domains are
the atoms within ``rs`` of centres spaced about ``h`` apart.

Cost grows linearly in the near field and preconditioner and quadratically in
the far-field pair loop.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from math import gamma

import numba as nb
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from pyscf import gto, lib
from pyscf.gto import moleintor
from scipy.linalg import lapack
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from qpet.conversions import BOHR_TO_ANG
from qpet.moments import regularised_cho_factor

#: Highest angular momentum of the auxiliary basis the far field supports.
_LMAX = 4

_Monomial = tuple[int, int, int]

#: Cartesian monomials ``x^i y^j z^k`` with ``i + j + k <= _LMAX``, by degree.
_MONOMIALS = [
    (i, j, n - i - j)
    for n in range(_LMAX + 1)
    for i in range(n, -1, -1)
    for j in range(n - i, -1, -1)
]
_NMONO = len(_MONOMIALS)
_MONOMIAL_INDEX = {m: a for a, m in enumerate(_MONOMIALS)}
_MONOMIAL_SIGN = np.array([(-1.0) ** sum(m) for m in _MONOMIALS])

#: Row-strided chunks over which the parallel loops keep private accumulators.
_NCHUNK = 96

_MAXITER = 300


def _atom_slices(
    auxmol: gto.Mole,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Per atom, first/last shell and first/last AO (shells are atom-ordered)."""
    bas_atom = auxmol._bas[:, 0]
    ao_loc = auxmol.ao_loc_nr()
    atoms = np.arange(auxmol.natm)
    sh0 = np.searchsorted(bas_atom, atoms, "left")
    sh1 = np.searchsorted(bas_atom, atoms, "right")
    return sh0, sh1, ao_loc[sh0], ao_loc[sh1]


def _int2c2e(
    auxmol: gto.Mole, bra: NDArray[np.int64], ket: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Exact ``int2c2e`` block between two lists of shells of ``auxmol``."""
    bas = np.ascontiguousarray(auxmol._bas[np.concatenate([bra, ket])])
    return moleintor.getints(
        "int2c2e_sph",
        auxmol._atm,
        bas,
        auxmol._env,
        shls_slice=(0, len(bra), len(bra), len(bra) + len(ket)),
    )


# ---------------------------------------------------------------------------
# Near field
# ---------------------------------------------------------------------------


class _NearField:
    """Exact ``J[A, B]`` blocks for atom pairs ``B >= A`` within ``rnear`` (Å)."""

    def __init__(self, auxmol: gto.Mole, rnear: float, nthreads: int) -> None:
        sh0, sh1, self.ao0, self.ao1 = _atom_slices(auxmol)
        coords = auxmol.atom_coords()
        neighbours = cKDTree(coords).query_ball_point(coords, rnear / BOHR_TO_ANG)
        # sorted, so the first entry of each list is the atom itself
        self.neighbours = [
            n[n >= a] for a, n in enumerate(map(np.sort, map(np.array, neighbours)))
        ]
        nrows = self.ao1 - self.ao0
        ncols = np.array([(self.ao1[n] - self.ao0[n]).sum() for n in self.neighbours])
        self.panel_offset = np.concatenate([[0], np.cumsum(nrows * ncols)])
        self.col_offset = np.concatenate([[0], np.cumsum(ncols)])
        self.data = np.empty(self.panel_offset[-1])
        self.cols = np.empty(self.col_offset[-1], dtype=np.int32)

        def fill(atom: int) -> None:
            n = self.neighbours[atom]
            kets = np.concatenate([np.arange(sh0[j], sh1[j]) for j in n])
            block = _int2c2e(auxmol, np.arange(sh0[atom], sh1[atom]), kets)
            self.data[self.panel_offset[atom] : self.panel_offset[atom + 1]] = (
                block.ravel()
            )
            self.cols[self.col_offset[atom] : self.col_offset[atom + 1]] = (
                np.concatenate([np.arange(self.ao0[j], self.ao1[j]) for j in n])
            )

        with ThreadPoolExecutor(nthreads) as pool:
            list(pool.map(fill, range(auxmol.natm)))

    def apply(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return _near_apply(
            self.data,
            self.panel_offset,
            self.col_offset,
            self.cols,
            self.ao0,
            self.ao1,
            x,
        )


@nb.njit(parallel=True, cache=True)
def _near_apply(data, panel_offset, col_offset, cols, ao0, ao1, x):  # type: ignore[no-untyped-def]
    """``J x`` from the stored panels; off-diagonal blocks apply their transpose too."""
    natm = len(ao0)
    partial = np.zeros((_NCHUNK, x.shape[0]))
    for chunk in nb.prange(_NCHUNK):  # type: ignore[attr-defined]
        y = partial[chunk]
        for atom in range(chunk, natm, _NCHUNK):
            nrow = ao1[atom] - ao0[atom]
            ncol = col_offset[atom + 1] - col_offset[atom]
            for r in range(nrow):
                base = panel_offset[atom] + r * ncol
                xr = x[ao0[atom] + r]
                acc = 0.0
                for k in range(ncol):
                    v = data[base + k]
                    col = cols[col_offset[atom] + k]
                    acc += v * x[col]
                    if k >= nrow:
                        y[col] += v * xr
                y[ao0[atom] + r] += acc
    return partial.sum(axis=0)


# ---------------------------------------------------------------------------
# Far field
# ---------------------------------------------------------------------------


def _double_factorial(n: int) -> float:
    return 1.0 if n <= 0 else float(np.prod(np.arange(n, 0, -2)))


def _solid_harmonic_coefficients(
    ell: int,
) -> tuple[list[_Monomial], NDArray[np.float64]]:
    """
    Monomial coefficients of the orthonormal real solid harmonics ``r^l Y_lm``.

    Read off PySCF's own spherical GTOs so the ``m`` order and signs match the
    integrals, then normalised on the unit sphere by exact product quadrature.

    :return: The degree-``l`` monomials and their ``(nmonomial, 2l + 1)`` coefficients.
    """
    mol = gto.M(atom="H 0 0 0", basis={"H": [[ell, [1.0, 1.0]]]}, spin=1, verbose=0)
    points = np.random.default_rng(ell).normal(size=(400, 3))
    angular = (
        mol.eval_gto("GTOval_sph", points) / np.exp(-np.sum(points**2, 1))[:, None]
    )
    monomials = [m for m in _MONOMIALS if sum(m) == ell]

    def powers(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.stack(
            [xyz[:, 0] ** i * xyz[:, 1] ** j * xyz[:, 2] ** k for i, j, k in monomials],
            1,
        )

    coefficients = np.linalg.lstsq(powers(points), angular, rcond=None)[0]

    z, w = np.polynomial.legendre.leggauss(2 * ell + 4)
    phi = (np.arange(4 * ell + 8) + 0.5) * 2 * np.pi / (4 * ell + 8)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    s = np.sqrt(1 - zz**2)
    sphere = np.stack([s * np.cos(pp), s * np.sin(pp), zz], -1).reshape(-1, 3)
    weights = np.repeat(w, len(phi)) * 2 * np.pi / len(phi)
    norm = ((powers(sphere) @ coefficients) ** 2 * weights[:, None]).sum(0)
    return monomials, coefficients / np.sqrt(norm)


def _multipole_map(auxmol: gto.Mole) -> sp.csr_matrix:
    """
    Sparse ``(naux, natm * _NMONO)`` map from coefficients to the Cartesian
    monomial multipoles of each atom.

    A function ``R(r) Y_lm`` with ``R = Σ_k c_k r^l exp(-a_k r²)`` has the single
    moment ``Q = Σ_k c_k ∫ r^(2l+2) exp(-a_k r²) dr``; outside overlap its
    potential is that of the point multipole ``4π/(2l+1)!! Q r^l Y_lm``.
    """
    ao_loc = auxmol.ao_loc_nr()
    harmonics = {ell: _solid_harmonic_coefficients(ell) for ell in range(_LMAX + 1)}
    rows, cols, vals = [], [], []
    for ib in range(auxmol.nbas):
        ell = auxmol.bas_angular(ib)
        exps = auxmol.bas_exp(ib)
        ctr = auxmol._libcint_ctr_coeff(ib)[:, 0]
        moment = np.sum(ctr * gamma(ell + 1.5) / (2 * exps ** (ell + 1.5)))
        prefactor = moment * 4 * np.pi / _double_factorial(2 * ell + 1)
        monomials, coefficients = harmonics[ell]
        for m in range(2 * ell + 1):
            for t, monomial in enumerate(monomials):
                rows.append(ao_loc[ib] + m)
                cols.append(auxmol.bas_atom(ib) * _NMONO + _MONOMIAL_INDEX[monomial])
                vals.append(prefactor * coefficients[t, m])
    return sp.csr_matrix((vals, (rows, cols)), shape=(auxmol.nao, auxmol.natm * _NMONO))


def _derivative_program(
    nmax: int,
) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]
]:
    """
    Flat recursion tables for the Cartesian derivatives ``D_g = ∂^g (1/r)``.

    With ``c`` the first nonzero index of ``g`` and ``d = g - e_c``::

        r² D_g = -x_c D_{g-e_c} - d_c D_{g-2e_c}
                 - 2 Σ_i d_i x_i D_{g-e_i} - Σ_i d_i (d_i - 1) D_{g-2e_i}

    Each row holds up to 8 terms ``(source, component, coefficient)``; component
    3 stands for the factor 1.

    :return: ``source``, ``component``, ``coefficient`` tables, and the table
        ``pair[a, b]`` of the index of ``D_{a+b}`` for monomials ``a`` and ``b``.
    """
    orders = [
        (i, j, n - i - j)
        for n in range(nmax + 1)
        for i in range(n, -1, -1)
        for j in range(n - i, -1, -1)
    ]
    position = {g: t for t, g in enumerate(orders)}
    source = np.zeros((len(orders), 8), np.int64)
    component = np.full((len(orders), 8), 3, np.int64)
    coefficient = np.zeros((len(orders), 8))

    def lowered(g: _Monomial, k: int, times: int = 1) -> _Monomial:
        h = list(g)
        h[k] -= times
        return (h[0], h[1], h[2])

    for t, g in enumerate(orders):
        if sum(g) == 0:
            continue
        c = next(k for k in range(3) if g[k] > 0)
        d = lowered(g, c)
        terms = [(position[lowered(g, c)], c, -1.0)]
        if d[c] >= 1:
            terms.append((position[lowered(g, c, 2)], 3, -float(d[c])))
        for i in range(3):
            if d[i] >= 1:
                terms.append((position[lowered(g, i)], i, -2.0 * d[i]))
            if d[i] >= 2:
                terms.append((position[lowered(g, i, 2)], 3, -float(d[i] * (d[i] - 1))))
        for k, (s, comp, coef) in enumerate(terms):
            source[t, k], component[t, k], coefficient[t, k] = s, comp, coef
    pair = np.array(
        [
            [position[(a[0] + b[0], a[1] + b[1], a[2] + b[2])] for b in _MONOMIALS]
            for a in _MONOMIALS
        ]
    )
    return source, component, coefficient, pair


_DERIVATIVES = _derivative_program(2 * _LMAX)


@nb.njit(parallel=True, cache=True)
def _far_apply(coords, u, rnear, source, component, coefficient, pair, sign):  # type: ignore[no-untyped-def]
    """
    ``psi[B, b] = Σ_{A: |R_AB| > rnear} Σ_a (-1)^|a| u[A, a] D_{a+b}(r_B - r_A)``.

    Each pair is visited once and fills both ends, using ``D_g(-R) = (-1)^|g| D_g(R)``.
    """
    natm, nmono = u.shape
    nder = source.shape[0]
    partial = np.zeros((_NCHUNK, natm, nmono))
    for chunk in nb.prange(_NCHUNK):  # type: ignore[attr-defined]
        psi = partial[chunk]
        deriv = np.zeros(nder)
        xyz = np.ones(4)
        for b_atom in range(chunk, natm, _NCHUNK):
            for a_atom in range(b_atom):
                for k in range(3):
                    xyz[k] = coords[b_atom, k] - coords[a_atom, k]
                r2 = xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2
                if r2 <= rnear * rnear:
                    continue
                deriv[0] = 1.0 / np.sqrt(r2)
                for t in range(1, nder):
                    acc = 0.0
                    for k in range(8):
                        acc += (
                            coefficient[t, k]
                            * xyz[component[t, k]]
                            * deriv[source[t, k]]
                        )
                    deriv[t] = acc / r2
                for a in range(nmono):
                    ua = sign[a] * u[a_atom, a]
                    back = 0.0
                    for b in range(nmono):
                        dv = deriv[pair[a, b]]
                        psi[b_atom, b] += ua * dv
                        back += u[b_atom, b] * dv
                    psi[a_atom, a] += sign[a] * back
    return partial.sum(axis=0)


class _FarField:
    """Multipole-multipole interaction of all atom pairs beyond ``rnear`` (Å)."""

    def __init__(self, auxmol: gto.Mole, rnear: float) -> None:
        self.map = _multipole_map(auxmol)
        self.map_t = self.map.T.tocsr()
        self.coords = np.ascontiguousarray(auxmol.atom_coords())
        self.rnear = rnear / BOHR_TO_ANG
        self.natm = auxmol.natm

    def apply(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        u = (self.map_t @ x).reshape(self.natm, _NMONO)
        psi = _far_apply(self.coords, u, self.rnear, *_DERIVATIVES, _MONOMIAL_SIGN)
        return self.map @ psi.ravel()


# ---------------------------------------------------------------------------
# Schwarz preconditioner
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _pack_upper(matrix, out):  # type: ignore[no-untyped-def]
    n, p = matrix.shape[0], 0
    for i in range(n):
        for j in range(i, n):
            out[p] = matrix[i, j]
            p += 1


def _spd_inverse_packed(matrix: NDArray[np.float64], out: NDArray[np.float64]) -> None:
    """
    Write the inverse of the SPD ``matrix`` into ``out``, packed upper triangle by row.

    A domain block can be numerically singular (diffuse functions of nearby
    atoms); a small diagonal shift is harmless in a preconditioner.
    """
    (factor, lower), _ = regularised_cho_factor(matrix, "domain block")
    inverse, _ = lapack.dpotri(factor, lower=lower, overwrite_c=1)
    _pack_upper(inverse, out)


@nb.njit(parallel=True, cache=True)
def _schwarz_apply(inverse, inverse_offset, index, index_offset, r):  # type: ignore[no-untyped-def]
    """``Σ_D R_Dᵀ A_D R_D r`` with each ``A_D`` symmetric, packed upper triangle."""
    ndomain = len(index_offset) - 1
    partial = np.zeros((_NCHUNK // 2, r.shape[0]))
    for chunk in nb.prange(_NCHUNK // 2):  # type: ignore[attr-defined]
        out = partial[chunk]
        for k in range(chunk, ndomain, _NCHUNK // 2):
            n = index_offset[k + 1] - index_offset[k]
            ix = index[index_offset[k] : index_offset[k + 1]]
            p = inverse_offset[k]
            for i in range(n):
                ri = r[ix[i]]
                acc = inverse[p] * ri
                for j in range(i + 1, n):
                    v = inverse[p + j - i]
                    acc += v * r[ix[j]]
                    out[ix[j]] += v * ri
                out[ix[i]] += acc
                p += n - i
    return partial.sum(axis=0)


class _Schwarz:
    """
    Overlapping additive Schwarz preconditioner.

    Centres are picked greedily so that every atom lies within ``h`` Å of one;
    a domain is the atoms within ``rs`` Å of its centre, and its matrix is the
    exact ``int2c2e`` of the member shells, stored inverted.
    """

    def __init__(self, auxmol: gto.Mole, rs: float, h: float, nthreads: int) -> None:
        sh0, sh1, ao0, ao1 = _atom_slices(auxmol)
        coords = auxmol.atom_coords()
        centres: list[int] = []
        for a in range(auxmol.natm):
            if not centres or (
                np.min(np.linalg.norm(coords[centres] - coords[a], axis=1))
                > h / BOHR_TO_ANG
            ):
                centres.append(a)
        members = [
            np.sort(np.array(m))
            for m in cKDTree(coords).query_ball_point(coords[centres], rs / BOHR_TO_ANG)
        ]
        sizes = np.array([(ao1[m] - ao0[m]).sum() for m in members])
        self.index_offset = np.concatenate([[0], np.cumsum(sizes)])
        self.inverse_offset = np.concatenate([[0], np.cumsum(sizes * (sizes + 1) // 2)])
        self.index = np.empty(self.index_offset[-1], dtype=np.int64)
        self.inverse = np.empty(self.inverse_offset[-1])

        def build(k: int) -> None:
            shells = np.concatenate([np.arange(sh0[j], sh1[j]) for j in members[k]])
            block = _int2c2e(auxmol, shells, shells)
            self.index[self.index_offset[k] : self.index_offset[k + 1]] = (
                np.concatenate([np.arange(ao0[j], ao1[j]) for j in members[k]])
            )
            _spd_inverse_packed(
                block, self.inverse[self.inverse_offset[k] : self.inverse_offset[k + 1]]
            )

        # One thread per domain: nested OpenMP/BLAS threads inside the pool
        # oversubscribe the cores and roughly double the build time.
        nomp = lib.num_threads()
        lib.num_threads(1)
        try:
            with threadpool_limits(1), ThreadPoolExecutor(nthreads) as pool:
                list(pool.map(build, range(len(members))))
        finally:
            lib.num_threads(nomp)

    def apply(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        return _schwarz_apply(
            self.inverse, self.inverse_offset, self.index, self.index_offset, r
        )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class CoulombSolver:
    """
    Solver for ``J x = b`` with ``J = (P|Q)`` of an auxiliary ``Mole``.

    The operator and the preconditioner are built once and reused by every
    :meth:`solve`.

    :param auxmol: PySCF auxiliary Mole object (angular momentum up to 4).
    :param rnear: Pair distance (Å) up to which ``J`` is computed exactly; beyond
        it the multipole far field is used.
    :param rs: Radius (Å) of the Schwarz domains.
    :param h: Spacing (Å) of the Schwarz domain centres.
    """

    def __init__(
        self,
        auxmol: gto.Mole,
        rnear: float = 7.0,
        rs: float = 4.5,
        h: float = 4.0,
    ) -> None:
        if max(auxmol.bas_angular(ib) for ib in range(auxmol.nbas)) > _LMAX:
            raise ValueError(f"far field supports l <= {_LMAX} only")
        nthreads = lib.num_threads()
        self._near = _NearField(auxmol, rnear, nthreads)
        self._far = _FarField(auxmol, rnear)
        self._preconditioner = _Schwarz(auxmol, rs, h, nthreads)

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """``J x``."""
        return self._near.apply(x) + self._far.apply(x)

    def solve(self, b: NDArray[np.float64], tol: float = 5e-5) -> NDArray[np.float64]:
        """
        ``J⁻¹ b`` by preconditioned conjugate gradients.

        Iterates until ``sqrt(r·M r / b·x) < tol`` with ``r = b - J x`` and ``M``
        the preconditioner. For ``M ≈ J⁻¹`` this estimates the relative J-norm
        error of ``x``, which for ``b`` the S vector is the quantity the charge
        correction depends on.

        :param b: Right-hand side, shape (naux,).
        :param tol: Target relative J-norm error.
        :return: The solution, shape (naux,).
        """
        x = np.zeros_like(b)
        r = b.copy()
        z = self._preconditioner.apply(r)
        p, rz = z.copy(), r @ z
        for _ in range(_MAXITER):
            Ap = self.matvec(p)
            alpha = rz / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            z = self._preconditioner.apply(r)
            rz_new = r @ z
            if np.sqrt(abs(rz_new) / abs(b @ x)) < tol:
                return x
            p = z + (rz_new / rz) * p
            rz = rz_new
        warnings.warn(
            f"Coulomb solve did not reach the J-norm tolerance {tol:g} in "
            f"{_MAXITER} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )
        return x
