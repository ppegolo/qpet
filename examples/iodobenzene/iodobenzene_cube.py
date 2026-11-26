"""Density and electrostatic potential of iodobenzene, from PET-ρ-OMol25 and from DFT.

    python iodobenzene_cube.py

Writes four cube files on the same grid.
"""

from pyscf import dft

from qpet import CubeGrid, DensityModel, DMDensity, build_mole, write_cube

symbols = ["I", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H"]
positions = [
    [3.3779, -0.1824, 0.0468],
    [1.2924, -0.0699, 0.0182],
    [0.6756, 1.1716, -0.0221],
    [-0.7105, 1.2399, -0.0413],
    [-1.4703, 0.0794, -0.0205],
    [-0.8409, -1.1561, 0.0197],
    [0.5446, -1.2375, 0.0392],
    [1.2682, 2.0757, -0.0385],
    [-1.1948, 2.2072, -0.0728],
    [-2.5505, 0.1379, -0.0355],
    [-1.4273, -2.0655, 0.0362],
    [1.0357, -2.2002, 0.0707],
]
mol = build_mole(symbols, positions, basis="def2-tzvpd", ecp="def2-tzvpd")
grid = CubeGrid.enclosing(mol.atom_coords())

model = DensityModel.from_release("pet-rho-omol", size="l")
predicted = model.predict(mol).rho

mf = dft.RKS(mol, xc="wb97m-v").run()
reference = DMDensity(mol, mf.make_rdm1())

for name, rho in (("ml", predicted), ("dft", reference)):
    write_cube(f"iodobenzene_density_{name}.cube", mol, grid, grid.sample(rho.density))
    write_cube(f"iodobenzene_esp_{name}.cube", mol, grid, grid.sample(rho.potential))
