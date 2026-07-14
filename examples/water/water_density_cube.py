"""Water electron density: ML model vs DFT, exported as CUBE files.

Computes the electron density of a single water molecule two ways and writes
each to a Gaussian CUBE file:

1. ML inference -- a metatomic density model (assumed present as ``model.pt``
   next to this script) predicts RI density-fitting coefficients, which qpet
   expands into a real-space density.
2. DFT reference -- a PBE / def2-svp calculation with PySCF.

Open the resulting ``.cube`` files in ChimeraX, PyMOL, or VMD to compare the
ML-predicted density against the DFT reference.

Run from this directory::

    python water_density_cube.py
"""

from __future__ import annotations

from pathlib import Path

from pyscf import dft, gto

from qpet import DensityModel, DMDensity, write_density_cube

HERE = Path(__file__).parent

# Water geometry in Angstrom (experimental gas-phase structure).
WATER = "O 0.0000 0.0000 0.0000; H 0.7575 0.5871 0.0000; H -0.7575 0.5871 0.0000"

# Orbital basis. The DFT reference uses it, and the metatomic model is assumed
# to have been trained on densities computed in the same basis.
ORBITAL_BASIS = "def2-svp"

# Model-specific configuration. These values must match the auxiliary basis
# and output head the model was trained with -- edit them for your model.
MODEL_FILE = HERE / "pet-scfbench.pt"
OUTPUT_NAME = "mtt::rho_c_jfit_coulomb"
BASIS_NAME = "def2-universal-jfit"


def density_from_ml() -> None:
    """Predict the density with the metatomic model and write a CUBE file."""
    model = DensityModel(
        model=str(MODEL_FILE),
        head=OUTPUT_NAME,
        aux_basis=BASIS_NAME,
    )
    rho = model.predict(gto.M(atom=WATER, basis=ORBITAL_BASIS))

    out = HERE / "water_density_ml.cube"
    write_density_cube(rho, out)
    print(f"ML density  -> {out.name}")


def density_from_dft() -> None:
    """Run a PBE / def2-svp DFT calculation and write a CUBE file."""
    mol = gto.M(atom=WATER, basis=ORBITAL_BASIS)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.kernel()

    rho = DMDensity(mol, mf.make_rdm1())

    out = HERE / "water_density_dft.cube"
    write_density_cube(rho, out)
    print(f"DFT density -> {out.name}")


if __name__ == "__main__":
    if MODEL_FILE.exists():
        try:
            density_from_ml()
        except Exception as exc:  # e.g. model exported with an older metatomic
            print(f"Skipping ML inference: could not run {MODEL_FILE.name}: {exc}")
    else:
        print(f"Skipping ML inference: {MODEL_FILE.name} not found in {HERE}.")
    density_from_dft()
