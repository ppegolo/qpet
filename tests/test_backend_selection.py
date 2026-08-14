"""One DensityModel over two backends: which one runs, and what it refuses."""

from __future__ import annotations

import re

import pytest

from qpet.model import DensityModel

AUX = "def2-universal-jfit"
HEAD = "mtt::rho_c_jfit_coulomb"


def test_auto_picks_metatomic_for_a_metatomic_model():
    """Pre-evaluated TensorMaps need no file, so this runs without a checkpoint."""
    model = DensityModel({HEAD: object()}, head=HEAD, aux_basis=AUX)
    assert model.backend == "metatomic"


def test_metatomic_backend_needs_a_head():
    message = (
        "the metatomic backend needs 'head': the name of the density output, "
        "e.g. 'mtt::rho_c_jfit_coulomb'."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        DensityModel({HEAD: object()}, aux_basis=AUX)


def test_petfast_backend_rejects_pre_evaluated_tensormaps():
    """The one refusal a user can trip without petfast installed."""
    message = (
        "backend='petfast' cannot take pre-evaluated TensorMaps; pass a "
        "'.petfast' artifact path or a petfast.Calculator."
    )
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        DensityModel({HEAD: object()}, aux_basis=AUX, backend="petfast")


def test_petfast_backend_rejects_the_dipole_constraint():
    """A petfast artifact carries no dipole head, so the option cannot be honoured."""
    message = (
        "constrain_dipole is a metatomic-backend feature: the petfast artifact "
        "carries only the density head, so there is no dipole output to "
        "constrain against."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        DensityModel("model.petfast", aux_basis=AUX, constrain_dipole=True)
