"""Inference backends of :class:`qpet.DensityModel`.

A backend is called on a list of molecules (and an optional cell, bohr) and
returns the raw coefficients of every requested output in PySCF order:
``{name: [array per molecule]}``, each (naux,) or (n_members, naux).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from pyscf import gto


class Backend(Protocol):
    def __call__(
        self, mols: Sequence[gto.Mole], cell: np.ndarray | None = None
    ) -> dict[str, list[np.ndarray]]: ...
