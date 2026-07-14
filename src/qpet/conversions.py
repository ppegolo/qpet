"""Conversion utilities between PySCF, ASE, and metatomic data structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import gto
from pyscf.data import elements
from pyscf.lib.parameters import BOHR as BOHR_TO_ANG

if TYPE_CHECKING:
    import ase
    import torch
    from metatomic.torch import System

__all__ = [
    "BOHR_TO_ANG",
    "pyscf_to_system",
    "system_to_pyscf",
    "system_to_atom_string",
    "ase_to_pyscf",
    "pyscf_to_ase",
    "pyscf_to_atom_string",
]


def pyscf_to_system(
    mol: gto.Mole,
    dtype: torch.dtype | None = None,
    device: str = "cpu",
) -> System:
    """
    Convert a PySCF Mole object to a metatomic System.

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

    return System(
        types=atom_types,
        positions=positions,
        cell=cell,
        pbc=torch.tensor([False, False, False]).to(device=device),
    )


def system_to_pyscf(
    system: System,
    basis: str,
    verbose: int = 0,
) -> gto.Mole:
    """
    Convert a metatomic System to a PySCF Mole object.

    :param system: metatomic System object
    :param basis: Basis set name for PySCF
    :param verbose: PySCF verbosity level
    :return: Built PySCF Mole object
    """
    atom_str = system_to_atom_string(system)
    mol = gto.M(atom=atom_str, basis=basis, verbose=verbose)
    return mol.build()


def system_to_atom_string(system: System) -> str:
    """
    Convert a metatomic System to a PySCF-compatible atom string.

    :param system: metatomic System object with types and positions attributes
    :return: Multi-line string with element symbols and Cartesian coordinates
        formatted for PySCF molecule construction (positions in Angstrom)
    """
    atoms = []
    types = system.types.detach().cpu().numpy()
    positions = system.positions.detach().cpu().numpy()

    for i in range(len(types)):
        symbol = elements.ELEMENTS[int(types[i])]
        x, y, z = positions[i]
        atoms.append(f"{symbol}  {x:.8f}  {y:.8f}  {z:.8f}")

    return "\n".join(atoms)


def ase_to_pyscf(
    atoms: ase.Atoms,
    basis: str,
    charge: int = 0,
    spin: int = 0,
    verbose: int = 0,
) -> gto.Mole:
    """
    Convert an ASE Atoms object to a PySCF Mole object.

    :param atoms: ASE Atoms object (positions in Angstrom)
    :param basis: Basis set name for PySCF
    :param charge: Total molecular charge
    :param spin: Number of unpaired electrons (2S)
    :param verbose: PySCF verbosity level
    :return: Built PySCF Mole object
    """
    atom_lines = []
    for symbol, pos in zip(
        atoms.get_chemical_symbols(), atoms.get_positions(), strict=True
    ):
        atom_lines.append(f"{symbol}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}")
    atom_str = "\n".join(atom_lines)
    return gto.M(
        atom=atom_str, basis=basis, charge=charge, spin=spin, verbose=verbose
    ).build()


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
    atoms = []
    for i in range(mol.natm):
        symbol = mol.atom_symbol(i)
        coord = mol.atom_coord(i) * BOHR_TO_ANG
        atoms.append(f"{symbol}  {coord[0]:.8f}  {coord[1]:.8f}  {coord[2]:.8f}")

    return "\n".join(atoms)
