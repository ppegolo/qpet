"""Density and electrostatic potential of water, from a model and from DFT.

    python water_density_cube.py model.pt

Writes four cube files on the same grid. ``AUX_BASIS`` and ``HEAD`` are those
of the model.
"""

import sys

from pyscf import dft

from qpet import CubeGrid, DensityModel, DMDensity, build_mole, write_cube

AUX_BASIS = "def2-universal-jfit"
HEAD = "mtt::rho_c_jfit_coulomb"

mol = build_mole(
    ["O", "H", "H"],
    [[0.0, 0.0, 0.0], [0.7575, 0.5871, 0.0], [-0.7575, 0.5871, 0.0]],
    basis="def2-tzvpd",
)
grid = CubeGrid.enclosing(mol.atom_coords())

model = DensityModel(sys.argv[1], aux_basis=AUX_BASIS, heads={"rho": HEAD})
predicted = model.predict(mol).rho

mf = dft.RKS(mol, xc="wb97m-v").run()
reference = DMDensity(mol, mf.make_rdm1())

for name, rho in (("ml", predicted), ("dft", reference)):
    write_cube(f"water_density_{name}.cube", mol, grid, grid.sample(rho.density))
    write_cube(f"water_esp_{name}.cube", mol, grid, grid.sample(rho.potential))
