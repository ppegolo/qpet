"""Conversion utilities between PySCF, ASE, and metatomic data structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyscf import gto
from pyscf.data import elements
from pyscf.lib.parameters import BOHR as BOHR_TO_ANG

if TYPE_CHECKING:
    import ase
    import torch
    from metatensor.torch import TensorMap
    from metatomic.torch import System

__all__ = [
    "BOHR_TO_ANG",
    "attach_conditioning",
    "pyscf_to_system",
    "system_to_pyscf",
    "system_to_atom_string",
    "ase_to_pyscf",
    "pyscf_to_ase",
    "pyscf_to_atom_string",
]


def attach_conditioning(
    system: System,
    charge: float,
    spin_multiplicity: float = 1.0,
    *,
    overwrite: bool = False,
) -> System:
    """
    Attach the charge and spin multiplicity a model conditions on to a System.

    Charge- and spin-conditioned models read these two per-system scalars with
    ``system.get_data(...)``; a System without them makes the model fall back to
    its own defaults, which are a neutral singlet. Predicting a charged species
    from a bare System is therefore silently wrong, so every entry point that
    builds a System attaches them.

    The layout is the one metatrain writes and reads (see
    ``metatrain.utils.system_data.get_system_data_transform``): a single block
    under ``Labels.single()`` keys, one ``"system"`` sample, no components, and
    one property named after the field.

    :param system: System to annotate, modified in place.
    :param charge: Total molecular charge.
    :param spin_multiplicity: Spin multiplicity 2S+1 (default 1, a singlet).
    :param overwrite: Replace values already present instead of keeping them.
        Without it, data the caller attached wins.
    :return: The same System, for chaining.
    """
    known = set(system.known_data())
    for name, value in (
        ("charge", charge),
        ("spin_multiplicity", spin_multiplicity),
    ):
        if name in known and not overwrite:
            continue
        system.add_data(
            name,
            _per_system_scalar(
                value,
                name,
                dtype=system.positions.dtype,
                device=system.positions.device,
            ),
            override=overwrite,
        )
    return system


def _per_system_scalar(
    value: float, name: str, dtype: torch.dtype | None, device: torch.device | None
) -> TensorMap:
    """A one-value per-system TensorMap, in the shape metatrain expects."""
    import torch
    from metatensor.torch import Labels, TensorBlock, TensorMap

    return TensorMap(
        Labels.single().to(device=device),
        [
            TensorBlock(
                values=torch.tensor([[float(value)]], dtype=dtype, device=device),
                samples=Labels(
                    "system", torch.zeros((1, 1), dtype=torch.int32, device=device)
                ),
                components=[],
                properties=Labels(
                    name, torch.zeros((1, 1), dtype=torch.int32, device=device)
                ),
            )
        ],
    )


def pyscf_to_system(
    mol: gto.Mole,
    dtype: torch.dtype | None = None,
    device: str = "cpu",
) -> System:
    """
    Convert a PySCF Mole object to a metatomic System.

    The molecule's charge and spin multiplicity travel with the System, so a
    charge-conditioned model sees the species the caller asked for rather than
    a neutral singlet.

    :param mol: PySCF Mole object (built)
    :param dtype: Torch dtype for positions (defaults to ``torch.float32``)
    :param device: Device to place tensors on
    :return: metatomic System object
    """
    import torch
    from metatomic.torch import System

    if dtype is None:
        dtype = torch.float32

    atom_types = torch.tensor(
        [mol.atom_charge(i) for i in range(mol.natm)],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.tensor(
        mol.atom_coords() * BOHR_TO_ANG,
        dtype=dtype,
        device=device,
    )
    cell = torch.zeros((3, 3), dtype=dtype, device=device)

    system = System(
        types=atom_types,
        positions=positions,
        cell=cell,
        pbc=torch.tensor([False, False, False]).to(device=device),
    )
    # mol.spin is 2S = n_alpha - n_beta, so the multiplicity is mol.spin + 1.
    return attach_conditioning(system, mol.charge, mol.spin + 1)


def system_to_pyscf(
    system: System,
    basis: str,
    charge: int = 0,
    spin: int | None = None,
    verbose: int = 0,
    ecp: str | None = None,
) -> gto.Mole:
    """
    Convert a metatomic System to a PySCF Mole object.

    :param system: metatomic System object
    :param basis: Basis set name for PySCF
    :param charge: Total molecular charge
    :param spin: Number of unpaired electrons (2S); ``None`` lets PySCF
        auto-determine the ground-state multiplicity from the electron count
    :param verbose: PySCF verbosity level
    :param ecp: Effective core potential name. Required for elements whose
        basis is defined against an ECP (def2 for I and other heavy p-block):
        without it ``mol.nelectron`` counts all Z electrons instead of Z_eff,
        so anything keyed on the electron count is wrong.
    :return: Built PySCF Mole object
    """
    atom_str = system_to_atom_string(system)
    return gto.M(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=verbose,
        ecp=ecp,
    )


def _format_atom_line(symbol: str, pos: np.ndarray) -> str:
    """Single atom line for PySCF: ``"X  x  y  z"`` in Angstrom (8-decimal)."""
    return f"{symbol}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}"


def system_to_atom_string(system: System) -> str:
    """
    Convert a metatomic System to a PySCF-compatible atom string.

    :param system: metatomic System object with types and positions attributes
    :return: Multi-line string with element symbols and Cartesian coordinates
        formatted for PySCF molecule construction (positions in Angstrom)
    """
    types = system.types.detach().cpu().numpy()
    positions = system.positions.detach().cpu().numpy()

    return "\n".join(
        _format_atom_line(elements.ELEMENTS[int(types[i])], positions[i])
        for i in range(len(types))
    )


def ase_to_pyscf(
    atoms: ase.Atoms,
    basis: str,
    charge: int = 0,
    spin: int | None = None,
    verbose: int = 0,
) -> gto.Mole:
    """
    Convert an ASE Atoms object to a PySCF Mole object.

    :param atoms: ASE Atoms object (positions in Angstrom)
    :param basis: Basis set name for PySCF
    :param charge: Total molecular charge
    :param spin: Number of unpaired electrons (2S); ``None`` lets PySCF
        auto-determine the ground-state multiplicity from the electron count
    :param verbose: PySCF verbosity level
    :return: Built PySCF Mole object
    """
    atom_str = "\n".join(
        _format_atom_line(symbol, pos)
        for symbol, pos in zip(
            atoms.get_chemical_symbols(), atoms.get_positions(), strict=True
        )
    )
    return gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin, verbose=verbose)


def pyscf_to_ase(mol: gto.Mole) -> ase.Atoms:
    """
    Convert a PySCF Mole object to an ASE Atoms object.

    :param mol: PySCF Mole object (built)
    :return: ASE Atoms object with positions in Angstrom
    """
    import ase

    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    positions = mol.atom_coords() * BOHR_TO_ANG
    return ase.Atoms(symbols=symbols, positions=positions)


def pyscf_to_atom_string(mol: gto.Mole) -> str:
    """
    Convert a PySCF Mole object to an atom string format.

    :param mol: PySCF Mole object
    :return: Multi-line string with element symbols and Cartesian coordinates
        (positions in Angstrom)
    """
    return "\n".join(
        _format_atom_line(mol.atom_symbol(i), mol.atom_coord(i) * BOHR_TO_ANG)
        for i in range(mol.natm)
    )
