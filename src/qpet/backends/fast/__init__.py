"""Inference-only PET: the weights of the metatomic model, run several times faster.

Export a metatrain checkpoint once::

    python -m qpet.backends.fast.export model.ckpt model.fast

and pass the artifact to ``DensityModel(..., backend="fast")``. Every target of
the artifact comes from one pass of the shared backbone. An LLPR ensemble
(exported with ``--llpr``) or a shallow ensemble exposes its members under
metatomic's name for them, ``mtt::aux::<target>_ensemble``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from pyscf import gto
from pyscf.data.nist import BOHR

from ._calculator import Calculator


def ensemble_name(target: str) -> str:
    """metatomic's name for the ensemble output of ``target``."""
    return "mtt::aux::" + target.replace("mtt::", "") + "_ensemble"


class FastBackend:
    """
    :param model: an artifact written by :func:`qpet.backends.fast.export.export`.
    :param outputs: as for :class:`~qpet.backends.metatomic.MetatomicBackend`.
    :param max_atoms_per_pass: above this many atoms the system is evaluated
        in spatial blocks of this many atoms (exact, but each block also runs
        its halo).
    :param options: forwarded to the calculator (``compile``, ``autocast_dtype``).
    """

    def __init__(
        self,
        model: str | Path,
        outputs: Mapping[str, str | None],
        device: str,
        dtype: torch.dtype,
        max_atoms_per_pass: int | None = 150_000,
        **options,
    ) -> None:
        self.calculator = Calculator(str(model), device=device, dtype=dtype, **options)
        available = set(self.calculator.target_names) | {
            ensemble_name(t)
            for t, n in self.calculator.members.items()
            if n > 1 or t in self.calculator.llpr
        }
        missing = sorted(set(outputs) - available)
        if missing:
            raise ValueError(
                f"the model has no output {missing}; available: {sorted(available)}"
            )
        self.outputs = dict(outputs)
        self.targets = sorted({base or name for name, base in self.outputs.items()})
        self.llpr = sorted(
            base for base in self.outputs.values() if base in self.calculator.llpr
        )
        self.max_atoms_per_pass = max_atoms_per_pass
        self.device, self.dtype = device, dtype

    def __call__(
        self, mols: Sequence[gto.Mole], cell: np.ndarray | None = None
    ) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {name: [] for name in self.outputs}
        for mol in mols:
            members = self._one(mol, cell)
            for name, base in self.outputs.items():
                c = members[base or name]
                if base in self.llpr:
                    c = c.mean(axis=0) + members[base + "/llpr"]
                out[name].append(c if base else c.mean(axis=0))
        return out

    def _one(self, mol: gto.Mole, cell: np.ndarray | None) -> dict[str, np.ndarray]:
        """Members of every target, (n_members, naux)."""
        positions = torch.tensor(
            mol.atom_coords() * BOHR, dtype=self.dtype, device=self.device
        )
        types = torch.tensor(
            [gto.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)],
            dtype=torch.int32,
            device=self.device,
        )
        conditioning = {"charge": int(mol.charge), "spin_multiplicity": mol.spin + 1}
        partition = (
            self.max_atoms_per_pass is not None and mol.natm > self.max_atoms_per_pass
        )
        if partition and cell is not None:
            raise NotImplementedError("the partitioned route has no periodic images")
        if partition:
            result = self.calculator.compute_partitioned(
                positions,
                types,
                max_core_atoms=self.max_atoms_per_pass,
                names=self.targets,
                ensembles=self.llpr,
                **conditioning,
            )
        else:
            periodic = cell is not None
            result = self.calculator.compute(
                positions,
                types,
                cell=torch.tensor(cell * BOHR, dtype=self.dtype) if periodic else None,
                pbc=periodic,
                names=self.targets,
                ensembles=self.llpr,
                **conditioning,
            )
        return {
            name: coefficients.T.detach().cpu().double().numpy()
            for name, (coefficients, _) in result.items()
        }
