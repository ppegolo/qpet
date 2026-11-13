"""Inference backends of :class:`qpet.DensityModel`.

A backend is called on a list of molecules (and an optional cell, bohr) and
returns the raw coefficients of every requested output in PySCF order:
``{name: [array per molecule]}``, each (naux,) or (n_members, naux).
"""
