"""Scalar-field calculators: electron density and electrostatic potential.

Each calculator evaluates a field from a particular density representation
(density matrix, RI coefficients, or a metatomic ML model) given a
:class:`~qpet.structure.Structure`.
"""

from .base import Calculator
from .dm import DMCalculator
from .metatomic import MetatomicCalculator
from .ri import RICalculator

__all__ = [
    "Calculator",
    "DMCalculator",
    "RICalculator",
    "MetatomicCalculator",
]
