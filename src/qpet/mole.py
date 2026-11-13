"""PySCF molecules for the models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pyscf import gto


def build_mole(
    symbols: Sequence[str],
    positions: np.ndarray,
    *,
    basis: str,
    charge: int = 0,
    spin: int | None = None,
    ecp: str | None = None,
) -> gto.Mole:
    """
    PySCF ``Mole`` from symbols and positions in Å.

    :param spin: 2S; ``None`` takes the lowest value the electron count allows.
    :param ecp: needed for the elements whose basis comes with one (def2 from
        Rb on): without it the molecule is all-electron.
    """
    if spin is None:
        spin = (sum(gto.charge(s) for s in symbols) - charge) % 2
    return gto.M(
        atom=list(
            zip(symbols, np.asarray(positions, dtype=float).tolist(), strict=True)
        ),
        basis=basis,
        charge=charge,
        spin=spin,
        ecp=ecp,
        verbose=0,
    )
