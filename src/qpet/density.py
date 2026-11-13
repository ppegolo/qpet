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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pyscf import df, gto, lib

from qpet.basis import copy_nuclei, make_auxmol
from qpet.conditioning import nuclear_charges_of
from qpet.moments import get_ri_dipole, get_ri_quadrupole
from qpet.normalize import aux_integrals, project

if TYPE_CHECKING:
    from metatensor.torch import TensorMap

    from qpet.esp_fast import FastESP

logger = logging.getLogger(__name__)

#: Closest approach to a nucleus in the nuclear potential (bohr).
_DIST_FLOOR = 0.1

#: Grid size below which :meth:`RIDensity.potential` stays on libcint.
FAST_ESP_MIN_POINTS = 1000


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
        v[p0 : p0 + step] = (charges / np.maximum(dist, _DIST_FLOOR)).sum(axis=1)
    return v


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
        j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s2ij")
        dm_tril = lib.pack_tril(dm + dm.T - np.diag(dm.diagonal()))
        coeffs = np.linalg.solve(auxmol.intor("int2c2e"), dm_tril @ j3c)
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

    def normalized(self, target: float, metric: str = "coulomb") -> RIDensity:
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

    def populations(self) -> np.ndarray:
        """
        Integral of each atom's functions, shape (natm,).

        Atomic charges are ``nuclear_charges_of(auxmol) - populations()``;
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
        return get_ri_dipole(
            self.auxmol, self.coeffs, origin, include_nuclear=self.nuclei
        )

    def quadrupole(self, origin: np.ndarray | None = None) -> np.ndarray:
        """Primitive quadrupole tensor about ``origin`` (bohr, default 0)."""
        origin = np.zeros(3) if origin is None else origin
        return get_ri_quadrupole(
            self.auxmol, self.coeffs, origin, include_nuclear=self.nuclei
        )

    def density(self, points: np.ndarray, batch: int = 5000) -> np.ndarray:
        rho = np.empty(len(points))
        for p0 in range(0, len(points), batch):
            rho[p0 : p0 + batch] = (
                self.auxmol.eval_gto("GTOval", points[p0 : p0 + batch]) @ self.coeffs
            )
        return rho

    def potential(
        self, points: np.ndarray, *, fast: bool | None = None, batch: int = 2000
    ) -> np.ndarray:
        """
        Electrostatic potential: nuclei minus electrons, or minus a spin density.

        :param fast: the closed-form kernel of :class:`qpet.esp_fast.FastESP`
            (float32 on a GPU, ~1e-5 relative error) instead of libcint.
            ``None`` uses it on a GPU from :data:`FAST_ESP_MIN_POINTS` points
            on, when its fit to the basis passes; ``True`` forces it.
        """
        evaluator = (
            None if fast is False else self._fast_esp(len(points), force=bool(fast))
        )
        if evaluator is not None:
            v = -evaluator.potential(self.coeffs, points)
        else:
            v = -_aux_potential(self.auxmol, self.coeffs, points, batch)
        if self.nuclei:
            v += nuclear_potential(
                self.auxmol.atom_coords(), nuclear_charges_of(self.auxmol), points
            )
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
            mu += nuclear_charges_of(self.mol) @ (self.mol.atom_coords() - origin)
        return mu

    def density(self, points: np.ndarray, batch: int = 5000) -> np.ndarray:
        rho = np.empty(len(points))
        for p0 in range(0, len(points), batch):
            ao = self.mol.eval_gto("GTOval", points[p0 : p0 + batch])
            rho[p0 : p0 + batch] = np.einsum("pi,pi->p", ao @ self.dm, ao)
        return rho

    def potential(self, points: np.ndarray, batch: int = 500) -> np.ndarray:
        v = np.empty(len(points))
        for p0 in range(0, len(points), batch):
            ints = self.mol.intor("int1e_grids", grids=points[p0 : p0 + batch])
            v[p0 : p0 + batch] = -(ints.reshape(len(ints), -1) @ self.dm.ravel())
        if self.nuclei:
            v += nuclear_potential(
                self.mol.atom_coords(), nuclear_charges_of(self.mol), points
            )
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

    def potential(self, points: np.ndarray, batch: int = 2000) -> np.ndarray:
        """Potential of every member, shape (n_members, n_points)."""
        v = -_aux_potential(self.auxmol, self.coeffs, points, batch)
        if self.nuclei:
            v += nuclear_potential(
                self.auxmol.atom_coords(), nuclear_charges_of(self.auxmol), points
            )
        return v

    def std(self, observable: Callable[[RIDensity], np.ndarray]) -> np.ndarray:
        """Standard deviation over the members of ``observable(member)``."""
        return np.std([observable(self[k]) for k in range(len(self))], axis=0)
