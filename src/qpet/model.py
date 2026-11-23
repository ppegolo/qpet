"""Density models evaluated on PySCF molecules."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pyscf import gto

from qpet.backends import Backend
from qpet.basis import aux_integrals, make_auxmol
from qpet.density import Ensemble, RIDensity
from qpet.normalize import Metric, direction

HEADS = ("rho", "spin", "rho_ensemble", "spin_ensemble")


@dataclass(frozen=True)
class Prediction:
    """
    Output of a model on one molecule.

    ``shift`` holds the integral moved by the projection (``"rho"``, and
    ``"spin"`` when predicted), to filter broken predictions afterwards.
    """

    rho: RIDensity
    spin: RIDensity | None = None
    rho_ensemble: Ensemble | None = None
    spin_ensemble: Ensemble | None = None
    shift: dict[str, float] = field(default_factory=dict)


class DensityModel:
    """
    A trained density model, evaluated on PySCF molecules.

    Settings are explicit: the auxiliary basis the model predicts in and the
    names of its outputs.

    :param model: exported model: a ``.pt`` ``AtomisticModel`` (or the loaded
        object) for ``backend="metatomic"``, an artifact of
        :func:`qpet.backends.fast.export` for ``backend="fast"``.
    :param aux_basis: auxiliary basis of the coefficients, a PySCF name or
        ``"etb:<ao_basis>:<beta>"``.
    :param heads: output names by role: ``"rho"`` (required), ``"spin"``,
        ``"rho_ensemble"``, ``"spin_ensemble"``.
    :param backend: ``"metatomic"`` or ``"fast"``.
    :param normalize: metric of the projection of every prediction (N on rho,
        2S on spin, see :mod:`qpet.normalize`), or ``None`` for the raw output.
    :param broken_symmetry: keep the predicted spin density of singlets. By
        default it is zero: the spin head was trained on open shells only.
    :param warn_shift: warn when the projection moves more than this fraction
        of the electron count, a sign that the prediction is broken. It only
        warns: the projection is applied regardless.
    :param backend_options: forwarded to the backend (``max_atoms_per_pass``,
        ``compile``, ``autocast_dtype`` for ``"fast"``).
    """

    def __init__(
        self,
        model: str | Path | Any,
        *,
        aux_basis: str,
        heads: Mapping[str, str],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        backend: Literal["metatomic", "fast"] = "metatomic",
        normalize: Metric | None = "coulomb",
        broken_symmetry: bool = False,
        warn_shift: float = 1e-3,
        **backend_options: Any,
    ) -> None:
        unknown = sorted(set(heads) - set(HEADS))
        if "rho" not in heads or unknown:
            raise ValueError(
                f"heads needs 'rho' and takes only {HEADS}; got {sorted(heads)}"
            )
        self.aux_basis = aux_basis
        self.heads = dict(heads)
        self.normalize = normalize
        self.broken_symmetry = broken_symmetry
        self.warn_shift = warn_shift
        outputs = {
            name: heads[role.removesuffix("_ensemble")]
            if role.endswith("_ensemble")
            else None
            for role, name in heads.items()
        }
        self.backend: Backend
        if backend == "metatomic":
            from qpet.backends.metatomic import MetatomicBackend

            self.backend = MetatomicBackend(
                model, outputs, device, dtype, **backend_options
            )
        elif backend == "fast":
            from qpet.backends.fast import FastBackend

            self.backend = FastBackend(model, outputs, device, dtype, **backend_options)
        else:
            raise ValueError(f"unknown backend {backend!r}; use 'metatomic' or 'fast'")

    def predict(self, mol: gto.Mole, *, cell: np.ndarray | None = None) -> Prediction:
        """
        Density of ``mol``; charge and spin are read from it.

        :param cell: lattice vectors as rows (bohr): the model sees periodic
            neighbours, and the density stays on the atoms of ``mol`` as given.
        """
        return self._predict([mol], cell)[0]

    def predict_many(
        self, mols: Sequence[gto.Mole], *, batch_size: int = 1
    ) -> list[Prediction]:
        """
        ``[predict(mol) for mol in mols]``, ``batch_size`` molecules per pass.

        For isolated molecules (no ``cell``). The fast backend evaluates one
        molecule per pass whatever ``batch_size``.
        """
        out: list[Prediction] = []
        for start in range(0, len(mols), batch_size):
            out += self._predict(mols[start : start + batch_size], None)
        return out

    def _predict(
        self, mols: Sequence[gto.Mole], cell: np.ndarray | None
    ) -> list[Prediction]:
        raw = self.backend(mols, cell)
        return [
            self._assemble(
                mol, {role: raw[name][i] for role, name in self.heads.items()}
            )
            for i, mol in enumerate(mols)
        ]

    def _assemble(self, mol: gto.Mole, coeffs: dict[str, np.ndarray]) -> Prediction:
        auxmol = make_auxmol(mol, self.aux_basis)
        for role, c in coeffs.items():
            if c.shape[-1] != auxmol.nao:
                raise ValueError(
                    f"{self.heads[role]} gives {c.shape[-1]} coefficients for "
                    f"{_label(mol)} but {self.aux_basis} has {auxmol.nao} functions: "
                    "is it the basis of the model?"
                )
        if mol.spin == 0 and not self.broken_symmetry:
            coeffs |= {
                role: np.zeros_like(c)
                for role, c in coeffs.items()
                if role.startswith("spin")
            }

        for role in ("rho", "spin"):
            members = coeffs.get(f"{role}_ensemble")
            if members is not None:
                coeffs[f"{role}_ensemble"] = (
                    members - members.mean(axis=0) + coeffs[role]
                )

        targets = {"rho": float(mol.nelectron), "spin": float(mol.spin)}
        shift: dict[str, float] = {}
        if self.normalize is not None:
            d, q = direction(auxmol, self.normalize), aux_integrals(auxmol)
            for role in ("rho", "spin"):
                if role not in coeffs:
                    continue
                shift[role] = targets[role] - q @ coeffs[role]
                if abs(shift[role]) > self.warn_shift * mol.nelectron:
                    warnings.warn(
                        f"the {role} prediction of {_label(mol)} is off by "
                        f"{shift[role]:+.3f}, more than {self.warn_shift:g} of its "
                        f"{mol.nelectron} electrons: projected anyway",
                        RuntimeWarning,
                        stacklevel=4,
                    )
                coeffs[role] = coeffs[role] + shift[role] * d
                members = coeffs.get(f"{role}_ensemble")
                if members is not None:
                    coeffs[f"{role}_ensemble"] = (
                        members + (targets[role] - members @ q)[:, None] * d
                    )

        def density(role: str) -> RIDensity | None:
            return (
                RIDensity(auxmol, coeffs[role], nuclei=role == "rho")
                if role in coeffs
                else None
            )

        def ensemble(role: str) -> Ensemble | None:
            name = f"{role}_ensemble"
            return (
                Ensemble(auxmol, coeffs[name], nuclei=role == "rho")
                if name in coeffs
                else None
            )

        return Prediction(
            rho=RIDensity(auxmol, coeffs["rho"]),
            spin=density("spin"),
            rho_ensemble=ensemble("rho"),
            spin_ensemble=ensemble("spin"),
            shift=shift,
        )


def _label(mol: gto.Mole) -> str:
    from collections import Counter

    counts = Counter(mol.atom_pure_symbol(i) for i in range(mol.natm))
    return (
        "".join(f"{s}{n}" for s, n in sorted(counts.items()))
        + f" (charge {mol.charge:+d})"
    )
