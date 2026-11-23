"""Densities that carry their geometry: evaluated on points, integrated, compared.

* :class:`RIDensity`: a density expanded in an auxiliary basis, as the models
  predict it and as the RI fit of a DFT density gives it.
* :class:`DMDensity`: the density of an AO density matrix, the DFT reference.
* :class:`Ensemble`: the members of an ensemble of one predicted density.

Points and origins are in bohr; densities come out in e/bohr³ and potentials
in hartree/e. ``nuclei`` says whether the nuclear charges belong to the
density: True for a total density, False for a spin density.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pyscf import df, gto, lib
from pyscf.dft import gen_grid

from qpet import moments
from qpet.basis import copy_nuclei, make_auxmol
from qpet.normalize import Metric, aux_integrals, project

if TYPE_CHECKING:
    from metatensor.torch import TensorMap

    from qpet.esp_fast import FastESP

logger = logging.getLogger(__name__)

#: Grid size below which :meth:`RIDensity.potential` stays on libcint.
FAST_ESP_MIN_POINTS = 1000

#: Points per block of basis-function values.
_BLOCK = 128 * gen_grid.BLKSIZE

#: Three-centre integrals held at a time in :func:`fit_coefficients` on the CPU.
_FIT_FLOATS = 100_000_000


def nuclear_potential(
    positions: np.ndarray,
    charges: np.ndarray,
    points: np.ndarray,
    chunk: int = 2_000_000,
) -> np.ndarray:
    """``Σ_A Z_A / |r - R_A|`` on points (bohr), about ``chunk`` distances at a time."""
    step = max(1, chunk // max(len(charges), 1))
    v = np.empty(len(points))
    for p0 in range(0, len(points), step):
        dist = np.linalg.norm(points[p0 : p0 + step, None] - positions[None], axis=2)
        v[p0 : p0 + step] = (charges / dist).sum(axis=1)
    return v


def _nuclei(mol: gto.Mole, points: np.ndarray, exclude: int | None) -> np.ndarray:
    """Potential of the nuclei of ``mol``, all but atom ``exclude``."""
    keep = np.arange(mol.natm) != (-1 if exclude is None else exclude)
    return nuclear_potential(mol.atom_coords()[keep], mol.atom_charges()[keep], points)


def _ao_blocks(mol: gto.Mole, points: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Screened basis-function values on consecutive blocks of points."""
    for p0 in range(0, len(points), _BLOCK):
        block = points[p0 : p0 + _BLOCK]
        mask = gen_grid.make_mask(mol, block)
        yield p0, mol.eval_gto("GTOval", block, non0tab=mask)


def _aux_density(
    auxmol: gto.Mole, coeffs: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """``Σ_k c_k φ_k(r)`` on points; ``coeffs`` of shape (..., naux)."""
    rho = np.empty(coeffs.shape[:-1] + (len(points),))
    for p0, ao in _ao_blocks(auxmol, points):
        rho[..., p0 : p0 + len(ao)] = coeffs @ ao.T
    return rho


def _aux_projection(mol: gto.Mole, auxmol: gto.Mole, dms: np.ndarray) -> np.ndarray:
    """``Σ_μν (k|μν) P_μν`` for every matrix of ``dms`` (n, nao, nao): (n, naux)."""
    try:
        import cupy
        from gpu4pyscf.df.int3c2e_bdiv import contract_int3c2e_dm
    except (ImportError, RuntimeError):
        pass
    else:
        return np.stack(
            [
                cupy.asnumpy(contract_int3c2e_dm(mol, auxmol, cupy.asarray(dm)))
                for dm in dms
            ]
        )
    packed = np.stack([lib.pack_tril(dm + dm.T - np.diag(dm.diagonal())) for dm in dms])
    ao_loc = auxmol.ao_loc_nr()
    d = np.empty((len(dms), auxmol.nao))
    s0 = 0
    while s0 < auxmol.nbas:
        s1 = s0 + 1
        while (
            s1 < auxmol.nbas
            and (ao_loc[s1 + 1] - ao_loc[s0]) * packed.shape[1] < _FIT_FLOATS
        ):
            s1 += 1
        shls = (0, mol.nbas, 0, mol.nbas, s0, s1)
        ints = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s2ij", shls_slice=shls)
        d[:, ao_loc[s0] : ao_loc[s1]] = packed @ ints
        s0 = s1
    return d


def fit_coefficients(mol: gto.Mole, auxmol: gto.Mole, dms: np.ndarray) -> np.ndarray:
    """
    Coulomb-metric RI fit of AO density matrices, as in the training references.

    The three-centre integrals are contracted on the GPU when gpu4pyscf is
    installed, a slice of auxiliary shells at a time otherwise.

    :param dms: shape (..., nao, nao).
    :return: Coefficients, shape (..., naux).
    """
    d = _aux_projection(mol, auxmol, dms.reshape(-1, mol.nao, mol.nao))
    coeffs = np.linalg.solve(auxmol.intor("int2c2e"), d.T).T
    return coeffs.reshape(dms.shape[:-2] + (auxmol.nao,))


def nmae(a: np.ndarray, b: np.ndarray, weights: np.ndarray, norm: float) -> float:
    """
    ``100 ∫|a - b| dr / norm``: the normalized mean absolute error, in percent.

    :param a, b: two densities on the points of a quadrature grid
        (:func:`qpet.becke_grid`).
    :param norm: the number of electrons for a total density.
    """
    return 100 * float(weights @ np.abs(a - b)) / norm


class ScalarField(Protocol):
    """A density evaluated on points: :class:`RIDensity`, :class:`DMDensity`."""

    nuclei: bool

    def density(self, points: np.ndarray) -> np.ndarray: ...

    def potential(self, points: np.ndarray) -> np.ndarray: ...


def _aux_potential(
    auxmol: gto.Mole, coeffs: np.ndarray, points: np.ndarray, batch: int
) -> np.ndarray:
    """``Σ_k c_k (k|r)`` on points; ``coeffs`` of shape (..., naux)."""
    v = np.empty(coeffs.shape[:-1] + (len(points),))
    for p0 in range(0, len(points), batch):
        fake = gto.fakemol_for_charges(points[p0 : p0 + batch])
        v[..., p0 : p0 + batch] = coeffs @ gto.mole.intor_cross("int2c2e", auxmol, fake)
    return v


class RIDensity:
    """
    ``ρ(r) = Σ_k c_k φ_k(r)`` on the auxiliary ``Mole`` ``auxmol``.

    :param auxmol: from :func:`qpet.make_auxmol`, which keeps the ECP charges
        of the nuclei.
    :param coeffs: shape (naux,), PySCF order.
    :param nuclei: whether the nuclear charges belong to this density.
    """

    def __init__(
        self, auxmol: gto.Mole, coeffs: np.ndarray, *, nuclei: bool = True
    ) -> None:
        if coeffs.shape != (auxmol.nao,):
            raise ValueError(
                f"coeffs shape {coeffs.shape} does not match the auxiliary basis "
                f"({auxmol.nao},)"
            )
        self.auxmol = auxmol
        self.coeffs = coeffs
        self.nuclei = nuclei
        self._fast: FastESP | None = None
        self._no_fast = False

    @classmethod
    def fit(
        cls,
        mol: gto.Mole,
        dm: np.ndarray,
        aux_basis: str,
        *,
        target: float | None = None,
        nuclei: bool = True,
    ) -> RIDensity:
        """
        Coulomb-metric RI fit of an AO density matrix, as in the training references.

        :param target: constrain the integral (N for a total density, 2S for a
            spin density); ``None`` leaves the fit unconstrained. The
            constrained fit is the unconstrained one projected in the Coulomb
            metric.
        """
        auxmol = make_auxmol(mol, aux_basis)
        coeffs = fit_coefficients(mol, auxmol, dm)
        if target is not None:
            coeffs = project(auxmol, coeffs, target, "coulomb")
        return cls(auxmol, coeffs, nuclei=nuclei)

    @classmethod
    def from_tensormap(
        cls, tensor: TensorMap, auxmol: gto.Mole, *, nuclei: bool = True
    ) -> RIDensity:
        """Coefficients of a one-system TensorMap (training target, raw output)."""
        from qpet._tensormap import to_pyscf

        (coeffs,) = to_pyscf(tensor)
        return cls(auxmol, coeffs.detach().cpu().double().numpy(), nuclei=nuclei)

    def to_tensormap(self) -> TensorMap:
        """The coefficients in the layout of a model output, as one system."""
        from qpet._tensormap import from_pyscf

        return from_pyscf(self.auxmol, self.coeffs)

    def normalized(self, target: float, metric: Metric = "coulomb") -> RIDensity:
        """Copy projected to integrate to ``target``, see :func:`project`."""
        return RIDensity(
            self.auxmol,
            project(self.auxmol, self.coeffs, target, metric),
            nuclei=self.nuclei,
        )

    def select_atoms(self, atoms: Sequence[int]) -> RIDensity:
        """
        The functions and nuclei of ``atoms`` only.

        For a partition of the atoms into A and B the potentials add up:
        ``select_atoms(A).potential(r) + select_atoms(B).potential(r) == potential(r)``.
        """
        idx = sorted({int(i) for i in atoms})
        slices = self.auxmol.aoslice_by_atom()
        keep = np.concatenate([np.arange(slices[a, 2], slices[a, 3]) for a in idx])
        sub = gto.Mole()
        sub.atom = [self.auxmol._atom[a] for a in idx]
        sub.basis = self.auxmol.basis
        sub.unit = "Bohr"
        sub.spin = int(sum(self.auxmol.atom_charge(a) for a in idx)) % 2
        sub.verbose = 0
        sub.build()
        copy_nuclei(self.auxmol, sub, idx)
        return RIDensity(sub, self.coeffs[keep], nuclei=self.nuclei)

    def integral(self) -> float:
        """Number of electrons, or 2S for a spin density."""
        return float(aux_integrals(self.auxmol) @ self.coeffs)

    def coulomb_distance(self, other: RIDensity) -> float:
        """``∫∫ Δρ(r) Δρ(r') / |r - r'|`` of the difference with ``other``,
        a density in the same auxiliary basis."""
        d = self.coeffs - other.coeffs
        return float(d @ self.auxmol.intor("int2c2e") @ d)

    def populations(self) -> np.ndarray:
        """
        Integral of each atom's functions, shape (natm,).

        Atomic charges are ``auxmol.atom_charges() - populations()``;
        for a spin density these are the spin populations.
        """
        slices = self.auxmol.aoslice_by_atom()
        atom = np.repeat(np.arange(self.auxmol.natm), slices[:, 3] - slices[:, 2])
        weights = aux_integrals(self.auxmol) * self.coeffs
        return np.bincount(atom, weights, minlength=self.auxmol.natm)

    def dipole(self, origin: np.ndarray | None = None) -> np.ndarray:
        """Dipole moment about ``origin`` (bohr, default 0), nuclei included if carried.

        Electrons count as negative charge.
        """
        origin = np.zeros(3) if origin is None else origin
        return moments.dipole(self.auxmol, self.coeffs, origin, self.nuclei)

    def quadrupole(self, origin: np.ndarray | None = None) -> np.ndarray:
        """Primitive quadrupole tensor about ``origin`` (bohr, default 0)."""
        origin = np.zeros(3) if origin is None else origin
        return moments.quadrupole(self.auxmol, self.coeffs, origin, self.nuclei)

    def density(self, points: np.ndarray) -> np.ndarray:
        return _aux_density(self.auxmol, self.coeffs, points)

    def potential(
        self,
        points: np.ndarray,
        *,
        fast: bool | None = None,
        batch: int = 2000,
        exclude: int | None = None,
    ) -> np.ndarray:
        """
        Electrostatic potential: nuclei minus electrons, or minus a spin density.

        :param fast: the closed-form kernel of :class:`qpet.esp_fast.FastESP`
            (float32 on a GPU, ~1e-5 relative error) instead of libcint.
            ``None`` uses it on a GPU from :data:`FAST_ESP_MIN_POINTS` points
            on, when its fit to the basis passes; ``True`` forces it.
        :param exclude: atom whose nucleus is left out, for the potential at
            that nucleus.
        """
        evaluator = (
            None if fast is False else self._fast_esp(len(points), force=bool(fast))
        )
        if evaluator is not None:
            v = -evaluator.potential(self.coeffs, points)
        else:
            v = -_aux_potential(self.auxmol, self.coeffs, points, batch)
        if self.nuclei:
            v += _nuclei(self.auxmol, points, exclude)
        return v

    def _fast_esp(self, n_points: int, *, force: bool) -> FastESP | None:
        import torch

        cuda = torch.cuda.is_available()
        if not force and (self._no_fast or not cuda or n_points < FAST_ESP_MIN_POINTS):
            return None
        if self._fast is None:
            from qpet.esp_fast import FastESP

            dtype = torch.float32 if cuda else torch.float64
            try:
                self._fast = FastESP(
                    self.auxmol, device="cuda" if cuda else "cpu", dtype=dtype
                )
            except RuntimeError as error:
                if force:
                    raise
                self._no_fast = True
                logger.info("closed-form ESP unavailable, using libcint: %s", error)
        return self._fast

    def __repr__(self) -> str:
        return f"RIDensity(natm={self.auxmol.natm}, naux={self.auxmol.nao})"


class DMDensity:
    """
    Density of an AO density matrix on ``mol``.

    :param dm: shape (nao, nao); ``alpha + beta`` for a total density,
        ``alpha - beta`` for a spin density.
    """

    def __init__(self, mol: gto.Mole, dm: np.ndarray, *, nuclei: bool = True) -> None:
        if dm.shape != (mol.nao, mol.nao):
            raise ValueError(
                f"dm shape {dm.shape} does not match the AO basis "
                f"({mol.nao}, {mol.nao})"
            )
        self.mol = mol
        self.dm = dm
        self.nuclei = nuclei

    @classmethod
    def from_chk(
        cls, path: str | Path, mol: gto.Mole | None = None
    ) -> tuple[DMDensity, DMDensity | None]:
        """Total and spin density of a PySCF checkpoint (spin None if restricted)."""
        mol = lib.chkfile.load_mol(str(path)) if mol is None else mol
        scf = lib.chkfile.load(str(path), "scf")
        c, occ = scf["mo_coeff"], scf["mo_occ"]
        if c.ndim == 2:
            return cls(mol, (c * occ) @ c.T), None
        alpha, beta = ((c[s] * occ[s]) @ c[s].T for s in range(2))
        return cls(mol, alpha + beta), cls(mol, alpha - beta, nuclei=False)

    def integral(self) -> float:
        return float(np.einsum("ij,ji->", self.dm, self.mol.intor("int1e_ovlp")))

    def dipole(self, origin: np.ndarray | None = None) -> np.ndarray:
        origin = np.zeros(3) if origin is None else origin
        with self.mol.with_common_orig(origin):
            mu = -np.einsum("xij,ji->x", self.mol.intor("int1e_r"), self.dm)
        if self.nuclei:
            mu += self.mol.atom_charges() @ (self.mol.atom_coords() - origin)
        return mu

    def quadrupole(self, origin: np.ndarray | None = None) -> np.ndarray:
        origin = np.zeros(3) if origin is None else origin
        with self.mol.with_common_orig(origin):
            rr = self.mol.intor("int1e_rr").reshape(3, 3, self.mol.nao, -1)
        q = -np.einsum("xyij,ji->xy", rr, self.dm)
        if self.nuclei:
            r = self.mol.atom_coords() - origin
            q += np.einsum("i,ix,iy->xy", self.mol.atom_charges(), r, r)
        return q

    def density(self, points: np.ndarray) -> np.ndarray:
        """On the GPU when torch has one."""
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dm = torch.from_numpy(np.ascontiguousarray(self.dm, dtype=float)).to(device)
        rho = np.empty(len(points))
        for p0, ao in _ao_blocks(self.mol, points):
            ao = torch.from_numpy(ao).to(device)
            rho[p0 : p0 + len(ao)] = ((ao @ dm) * ao).sum(1).cpu().numpy()
        return rho

    def potential(
        self, points: np.ndarray, batch: int = 500, *, exclude: int | None = None
    ) -> np.ndarray:
        """
        On the GPU from :data:`FAST_ESP_MIN_POINTS` points on when gpu4pyscf
        is installed.

        :param exclude: atom whose nucleus is left out, for the potential at
            that nucleus.
        """
        v = None
        if len(points) >= FAST_ESP_MIN_POINTS:
            try:
                import cupy
                from gpu4pyscf.gto.int3c1e import int1e_grids
            except (ImportError, RuntimeError):
                pass
            else:
                dm = cupy.asarray(self.dm)
                v = -cupy.asnumpy(int1e_grids(self.mol, cupy.asarray(points), dm=dm))
        if v is None:
            v = np.empty(len(points))
            for p0 in range(0, len(points), batch):
                ints = self.mol.intor("int1e_grids", grids=points[p0 : p0 + batch])
                v[p0 : p0 + batch] = -(ints.reshape(len(ints), -1) @ self.dm.ravel())
        if self.nuclei:
            v += _nuclei(self.mol, points, exclude)
        return v

    def __repr__(self) -> str:
        return f"DMDensity(natm={self.mol.natm}, nao={self.mol.nao})"


class Ensemble:
    """
    Members of an ensemble of one predicted density, shape (n_members, naux).

    The members are recentred on the prediction (their mean is the prediction)
    and projected like it, so every member has the same integral.
    """

    def __init__(
        self, auxmol: gto.Mole, coeffs: np.ndarray, *, nuclei: bool = True
    ) -> None:
        self.auxmol = auxmol
        self.coeffs = coeffs
        self.nuclei = nuclei

    def __len__(self) -> int:
        return len(self.coeffs)

    def __getitem__(self, k: int) -> RIDensity:
        return RIDensity(self.auxmol, self.coeffs[k], nuclei=self.nuclei)

    def density(self, points: np.ndarray) -> np.ndarray:
        """Density of every member, shape (n_members, n_points)."""
        return _aux_density(self.auxmol, self.coeffs, points)

    def potential(
        self, points: np.ndarray, batch: int = 2000, *, exclude: int | None = None
    ) -> np.ndarray:
        """Potential of every member, shape (n_members, n_points)."""
        v = -_aux_potential(self.auxmol, self.coeffs, points, batch)
        if self.nuclei:
            v += _nuclei(self.auxmol, points, exclude)
        return v

    def std(self, observable: Callable[[RIDensity], np.ndarray]) -> np.ndarray:
        """Standard deviation over the members of ``observable(member)``."""
        return np.std([observable(self[k]) for k in range(len(self))], axis=0)
