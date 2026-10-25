"""Model wrapper: predict RI densities from geometries.

One class, :class:`DensityModel`, over two inference backends:

* ``"metatomic"`` — the exported metatomic model, run through metatensor with
  vesin neighbor lists. The reference path, and the one that supports the dipole
  constraint and oracle-charge conditioning.
* ``"petfast"`` — an inference-only reimplementation of the same network that
  skips metatensor entirely, returning coefficients already in PySCF order. It
  is faster and holds larger systems, and is numerically equivalent up to float
  round-off.

The backend changes only how coefficients are produced; everything after that is
shared.

This module owns every metatomic / metatensor / torch / vesin touchpoint in
qpet — model loading, neighbor lists, forward passes, and the conversion from
TensorMap output to PySCF coefficient ordering. Treat any metatomic import
outside this file as a design error.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pyscf import gto

from qpet.basis import make_auxmol
from qpet.conditioning import FIRST_ECP_ELEMENT, electron_count, nuclear_charges_of
from qpet.conversions import (
    BOHR_TO_ANG,
    attach_conditioning,
    pyscf_to_system,
    system_to_pyscf,
)
from qpet.fields import RIDensity
from qpet.moments import (
    DEFAULT_CHARGE_OFFSET,
    apply_dipole_correction,
    get_ri_dipole,
    normalize_charge_additive,
)

if TYPE_CHECKING:
    import torch
    from metatensor.torch import TensorMap
    from metatomic.torch import AtomisticModel, System

#: Backends :class:`DensityModel` can run.
BACKENDS = ("metatomic", "petfast")


def _warn_if_ecp_is_missing(mols: list[gto.Mole], ecp: str | None) -> None:
    """Warn when a heavy element is being treated all-electron by accident."""
    if ecp:
        return
    heavy = sorted(
        {
            mol.atom_symbol(i)
            for mol in mols
            for i in range(mol.natm)
            if gto.charge(mol.atom_symbol(i)) >= FIRST_ECP_ELEMENT
        }
    )
    if not heavy:
        return
    warnings.warn(
        f"no 'ecp' set but the system contains {', '.join(heavy)}: the molecule "
        f"is built all-electron, so its electron count is Z rather than Z_eff. "
        f"Pass the ECP of the training data, e.g. ecp='def2-tzvp'.",
        RuntimeWarning,
        stacklevel=3,
    )


def _system_label(mol: gto.Mole, index: int) -> str:
    """Short, printable name of a molecule for warnings."""
    from collections import Counter

    counts = Counter(mol.atom_symbol(i) for i in range(mol.natm))
    formula = "".join(f"{s}{n}" for s, n in sorted(counts.items()))
    return f"system {index} ({formula}, charge {mol.charge:+d})"


def _check_coefficient_count(auxmol: gto.Mole, coeffs: np.ndarray, label: str) -> None:
    """Fail, naming the system, when a prediction does not fit the auxiliary basis."""
    if len(coeffs) != auxmol.nao:
        raise ValueError(
            f"Predicted {len(coeffs)} coefficients for {label} but the auxiliary "
            f"basis has {auxmol.nao} functions; check that aux_basis matches the "
            "basis the model was trained on."
        )


def _tensormap_to_pyscf_tensors(
    tensor: TensorMap,
) -> list[torch.Tensor]:
    """
    Convert a TensorMap to a list of torch tensors in PySCF order, one per system.

    The tensors keep whatever autograd graph the model built, which is
    what :meth:`DensityModel.predict_coefficients_torch` needs; the numpy
    wrapper below detaches.

    Extracts coefficient values from TensorMap format and converts them to the ordering
    expected by PySCF, handling angular momentum component reordering for p-orbitals.

    :param tensor: TensorMap containing density fitting coefficients with keys indexed
        by angular momentum, inversion parity, and atomic species.
    :return: List of tensors, one per system, with the coefficients sorted
        by atom index, angular momentum, radial channel and magnetic quantum
        number.
    """
    import torch

    sys_list, atm_list, ang_list = [], [], []
    rad_list, m_list, val_list = [], [], []

    for key, block in tensor.items():
        angular = int(key[0])

        vals = block.values
        device = vals.device
        S, C, P = vals.shape

        # Extract indices
        sys_idx = block.samples.values[:, 0].to(device, dtype=torch.long)
        atm_idx = block.samples.values[:, 1].to(device, dtype=torch.long)
        rad_idx = block.properties.values[:, 0].to(device, dtype=torch.long)
        m_idx = block.components[0].values[:, 0].to(device, dtype=torch.long)

        # PySCF p-orbital reordering. metatensor stores real-SH components as
        # (m=-1, 0, +1) = (py, pz, px); PySCF expects (px, py, pz). The permutation
        # `[2, 0, 1]` reorders values into PySCF order; m_idx is left untouched and
        # acts as a stable sort key below — sorting by ascending m then yields the
        # already-permuted (px, py, pz) ordering.
        if angular == 1:
            vals = vals[:, [2, 0, 1], :]

        # Reshape index grids for broadcasting
        # sys/atm: (S, 1, 1) -> (S, C, P)
        sys_grid = sys_idx.view(S, 1, 1).expand(S, C, P)
        atm_grid = atm_idx.view(S, 1, 1).expand(S, C, P)
        # m: (1, C, 1) -> (S, C, P)
        m_grid = m_idx.view(1, C, 1).expand(S, C, P)
        # rad: (1, 1, P) -> (S, C, P)
        rad_grid = rad_idx.view(1, 1, P).expand(S, C, P)
        # ang: scalar -> (S, C, P)
        ang_grid = torch.full((S, C, P), angular, dtype=torch.long, device=device)

        mask = ~torch.isnan(vals)

        sys_list.append(sys_grid[mask])
        atm_list.append(atm_grid[mask])
        ang_list.append(ang_grid[mask])
        rad_list.append(rad_grid[mask])
        m_list.append(m_grid[mask])
        val_list.append(vals[mask])

    if not val_list:
        return []

    sys_all = torch.cat(sys_list)
    atm_all = torch.cat(atm_list)
    ang_all = torch.cat(ang_list)
    rad_all = torch.cat(rad_list)
    m_all = torch.cat(m_list)
    val_all = torch.cat(val_list)

    # To sort by (sys, atm, ang, rad, m), we perform sequential stable sorts
    # in reverse order (least significant key first).
    idx = torch.arange(len(val_all), device=device)

    idx = idx[torch.argsort(m_all[idx], stable=True)]
    idx = idx[torch.argsort(rad_all[idx], stable=True)]
    idx = idx[torch.argsort(ang_all[idx], stable=True)]
    idx = idx[torch.argsort(atm_all[idx], stable=True)]
    idx = idx[torch.argsort(sys_all[idx], stable=True)]

    sys_sorted = sys_all[idx]
    val_sorted = val_all[idx]

    # Split the sorted values into a list of tensors (one per system)
    # Find the indices where the system index changes
    changes = (sys_sorted[1:] != sys_sorted[:-1]).nonzero(as_tuple=True)[0] + 1

    # torch.tensor_split requires the boundary indices on CPU
    result_tuple = torch.tensor_split(val_sorted, changes.cpu())

    return list(result_tuple)


def _tensormap_to_pyscf_coefficients(tensor: TensorMap) -> list[np.ndarray]:
    """PySCF-ordered coefficients as numpy arrays, one per system."""
    import torch

    return [
        res.detach().cpu().to(torch.float64).numpy()
        for res in _tensormap_to_pyscf_tensors(tensor)
    ]


def _require_petfast() -> Any:
    try:
        import petfast
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='petfast' needs the petfast package, and a '.petfast' "
            "artifact built from a metatrain checkpoint with "
            "petfast.export.export_checkpoint."
        ) from error
    return petfast


def _resolve_backend(backend: str, model: Any) -> str:
    """Which backend a ``(backend, model)`` pair asks for.

    ``"auto"`` reads it off the model: a ``.petfast`` artifact path or a ready
    ``petfast.Calculator`` means the fast path, anything else the metatomic one.
    """
    if backend != "auto":
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. Use 'auto' or one of {BACKENDS}."
            )
        return backend
    if isinstance(model, str) and model.endswith(".petfast"):
        return "petfast"
    if type(model).__module__.split(".")[0] == "petfast":
        return "petfast"
    return "metatomic"


class DensityModel:
    """
    ML density model predicting :class:`~qpet.fields.RIDensity`.

    Runs either inference backend (see the module docstring); ``backend="auto"``
    picks ``"petfast"`` for a ``.petfast`` artifact or a ready
    ``petfast.Calculator``, and ``"metatomic"`` otherwise. Both accept the same
    :meth:`predict` / :meth:`predict_systems` calls and return the same objects.

    When constructed with a ``dict`` of pre-evaluated TensorMaps (one model
    forward pass feeding several consumers), no forward pass is performed —
    the relevant heads are read from the dict. Metatomic backend only.

    :param model: Pre-loaded metatomic model, path to a ``.pt`` file, or a
        ``dict`` of pre-evaluated TensorMaps from a previous forward pass; for
        the petfast backend, a path to a ``.petfast`` artifact or a ready
        ``petfast.Calculator``.
    :param head: Name of the density head in the model output
        (e.g. ``"mtt::rho_c_jfit_coulomb"``). Required by the metatomic
        backend; a petfast artifact carries its own head and ignores this.
    :param aux_basis: Auxiliary basis the model was trained on: a PySCF basis
        set name (e.g. ``"def2-universal-jfit"``) or an even-tempered
        specification ``"etb:<ao_basis>:<beta>"`` (e.g. ``"etb:def2-svp:2.0"``),
        where ``ao_basis`` is the *orbital* basis the ETB was generated from.
    :param basis: Orbital basis used to build a ``Mole`` in
        :meth:`predict_systems`; unused by :meth:`predict`, which takes the
        caller's ``Mole`` as-is.
    :param device: Torch device (default ``"cpu"``).
    :param dtype: Torch dtype (default ``torch.float32``).
    :param backend: ``"auto"`` (default), ``"metatomic"`` or ``"petfast"``.
    :param normalize_charge: Shift the coefficients so each predicted density
        integrates to the expected electron count, or to the number of unpaired
        electrons 2S for a spin head (default True): the model is not
        charge-conserving. Pass ``normalize_charge=False`` for the raw model
        output. With :meth:`predict_systems` this needs ``ecp`` set for ECP
        elements, otherwise the target count is Z rather than Z_eff.
    :param charge_metric: Metric of the charge correction, ``"coulomb"``
        (default; minimum Coulomb self-energy, spreads the missing charge over
        the diffuse functions and repairs the surface potential) or
        ``"identity"`` (s-channel shift only). See
        :func:`qpet.moments.normalize_charge_additive`.
    :param charge_offset: Added to the electron count to form the target
        (default :data:`qpet.moments.DEFAULT_CHARGE_OFFSET`). The reference RI
        fits are unconstrained and integrate to slightly more than the exact
        count; this is the mean of that basis-incompleteness offset, so the
        target is what a perfect model of the *reference* would predict.
    :param charge_guard: Floor, in electrons, of the limit above which the
        projection is skipped with a warning (default 0.5; ``None`` disables it).
    :param charge_guard_relative: Part of that limit per electron (default 1e-3).
    :param charge_guard_per_charge: Part of that limit per electron and unit of
        net charge (default 3e-4).
    :param ecp: Effective core potential name used to build molecules in
        :meth:`predict_systems`; unused by :meth:`predict`, which takes the
        caller's ``Mole`` as-is. Set it whenever ``basis`` implies an ECP
        (def2 for I and other heavy p-block).
    :param oracle_charges: If True, compute GFN2-xTB partial charges for each
        system (via ``tblite``) and attach them as the ``"mtt::oracle_charges"``
        conditioning field that oracle-conditioned models read. Harmless for
        models trained without the conditioning; required to get the
        conditioned accuracy from models trained with it.
    :param constrain_dipole: If True, apply a charge-neutral correction so
        the electric dipole matches the model-predicted value. Metatomic
        backend only.
    :param dipole_head: Name of the dipole output in the model (only used
        when ``constrain_dipole=True``).
    :param broken_symmetry: Spin heads only (head name ending in ``_spin``).
        By default the spin density of a singlet (``mol.spin == 0``) is set to
        zero: the spin head was trained on unrestricted states only, so on a
        closed-shell singlet it predicts a spurious broken-symmetry pattern.
        Pass True to keep the prediction for broken-symmetry singlets.
    :param max_atoms_per_pass: petfast backend only. Above this many atoms,
        evaluate with petfast's spatial partitioning instead of in one pass.
        Partitioning is exact but processes each block's halo as well as its
        core, so it is worth using only when a system does not fit; ``None``
        never partitions.
    :param backend_options: petfast backend only; forwarded to
        ``petfast.Calculator`` — e.g. ``autocast_dtype`` or ``compile``. Note
        that float16 measures a few percent relative error on density
        coefficients, which is unlikely to be acceptable here.

    Example::

        from qpet import DensityModel

        model = DensityModel(
            "density_model.pt",
            head="mtt::rho_c_jfit_coulomb",
            aux_basis="def2-universal-jfit",
        )
        rho = model.predict(mol)
        v = rho.potential(points)  # or density(pts), dipole(), scf_guess(mol)

        # Same call, the fast backend; the head comes from the artifact.
        fast = DensityModel("density_model.petfast", aux_basis="def2-universal-jfit")
    """

    def __init__(
        self,
        model: str | torch.nn.Module | dict | Any,  # type: ignore[type-arg]  # noqa: F821
        head: str | None = None,
        aux_basis: str | None = None,
        basis: str | None = None,
        device: str = "cpu",
        dtype: torch.dtype | None = None,  # noqa: F821
        *,
        backend: str = "auto",
        normalize_charge: bool = True,
        charge_metric: str = "coulomb",
        charge_offset: float = DEFAULT_CHARGE_OFFSET,
        charge_guard: float | None = 0.5,
        charge_guard_relative: float = 1e-3,
        charge_guard_per_charge: float = 3e-4,
        constrain_dipole: bool = False,
        oracle_charges: bool = False,
        dipole_head: str = "mtt::electric_dipole",
        ecp: str | None = None,
        broken_symmetry: bool = False,
        max_atoms_per_pass: int | None = 150_000,
        **backend_options: Any,
    ) -> None:
        import torch

        if aux_basis is None:
            raise ValueError(
                "DensityModel needs 'aux_basis': the auxiliary basis the model "
                "was trained on, e.g. 'def2-universal-jfit'."
            )
        if dtype is None:
            dtype = torch.float32

        self.backend = _resolve_backend(backend, model)

        self._model = model
        self._head = head
        self._device = device
        self._dtype = dtype

        self.aux_basis = aux_basis
        self.basis = basis
        self.ecp = ecp
        self.normalize_charge = normalize_charge
        self.broken_symmetry = broken_symmetry
        if charge_metric not in ("identity", "coulomb"):
            raise ValueError(
                f"Unknown charge_metric {charge_metric!r}. Use 'identity' or 'coulomb'."
            )
        self.charge_metric = charge_metric
        self.charge_offset = float(charge_offset)
        self.charge_guard = charge_guard
        self.charge_guard_relative = charge_guard_relative
        self.charge_guard_per_charge = charge_guard_per_charge
        self.constrain_dipole = constrain_dipole
        self.oracle_charges = oracle_charges
        self.dipole_head = dipole_head
        self.max_atoms_per_pass = max_atoms_per_pass

        self._loaded_model: AtomisticModel | None = None
        self._calculator: Any = None

        if self.backend == "petfast":
            self._init_petfast(model, device, dtype, backend_options)
            return

        if head is None:
            raise ValueError(
                "the metatomic backend needs 'head': the name of the density "
                "output, e.g. 'mtt::rho_c_jfit_coulomb'."
            )
        if backend_options:
            raise TypeError(
                f"unexpected keyword argument(s) {sorted(backend_options)} for the "
                "metatomic backend; these are forwarded to petfast.Calculator and "
                "apply only to backend='petfast'."
            )

        # Model-level setup only needed when model is not pre-evaluated.
        if isinstance(model, dict):
            self._nl_calculators = []
            self._options = None
            self._loaded_model = None
            return

        from metatomic.torch import ModelEvaluationOptions, ModelOutput

        if isinstance(model, str):
            from metatomic.torch import load_atomistic_model

            self._loaded_model = load_atomistic_model(model)
        else:
            self._loaded_model = cast("AtomisticModel", model)

        self._loaded_model = self._loaded_model.to(device=device, dtype=dtype)

        if constrain_dipole:
            available = set(self._loaded_model.capabilities().outputs.keys())
            if dipole_head not in available:
                raise ValueError(
                    f"constrain_dipole=True but the model does not support "
                    f"'{dipole_head}'; available outputs: {sorted(available)}"
                )

        outputs = {head: ModelOutput(per_atom=True)}
        if constrain_dipole:
            outputs[dipole_head] = ModelOutput(per_atom=False)
        self._options = ModelEvaluationOptions(length_unit="angstrom", outputs=outputs)
        self._neighbor_list_options = self._loaded_model.requested_neighbor_lists()

        import vesin.metatomic

        self._nl_calculators = [
            vesin.metatomic.NeighborList(options=options, length_unit="angstrom")
            for options in self._neighbor_list_options
        ]

    @property
    def _is_spin_head(self) -> bool:
        return (self._head or "").endswith("_spin")

    @property
    def _zero_singlet_spin(self) -> bool:
        return self._is_spin_head and not self.broken_symmetry

    def _norm_target(self, mol: gto.Mole) -> float:
        """Integral the normalised density is projected to: 2S for a spin head."""
        if self._is_spin_head:
            return float(mol.spin)
        return electron_count(mol) + self.charge_offset

    def _init_petfast(
        self,
        model: Any,
        device: str,
        dtype: torch.dtype,  # noqa: F821
        options: dict[str, Any],
    ) -> None:
        """Set up the petfast backend: load the artifact, or adopt a Calculator."""
        if self.constrain_dipole:
            raise ValueError(
                "constrain_dipole is a metatomic-backend feature: the petfast "
                "artifact carries only the density head, so there is no dipole "
                "output to constrain against."
            )
        if self.oracle_charges:
            raise ValueError(
                "oracle_charges is a metatomic-backend feature; petfast reads its "
                "conditioning from charge and spin multiplicity directly."
            )
        if isinstance(model, dict):
            raise TypeError(
                "backend='petfast' cannot take pre-evaluated TensorMaps; pass a "
                "'.petfast' artifact path or a petfast.Calculator."
            )

        if isinstance(model, str):
            petfast = _require_petfast()
            self._calculator = petfast.Calculator(
                model, device=device, dtype=dtype, **options
            )
        else:
            if options:
                raise TypeError(
                    f"unexpected keyword argument(s) {sorted(options)}: they are "
                    "forwarded to petfast.Calculator, which is already built."
                )
            self._calculator = model
        self._head = self._calculator.target_name

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, mols: gto.Mole | list[gto.Mole]) -> RIDensity | list[RIDensity]:
        """
        Predict the RI density for PySCF molecule(s).

        Charge and spin are taken from each ``Mole``; the auxiliary basis is
        built on the molecule as given.

        :param mols: A built ``Mole`` or a list of them.
        :return: A single :class:`RIDensity` for a single ``Mole`` input, a
            list for list input.
        """
        single = isinstance(mols, gto.Mole)
        mol_list: list[gto.Mole] = [mols] if single else list(mols)
        # petfast reads the geometry off the Mole, so the System conversion is
        # skipped rather than built and thrown away.
        systems = (
            []
            if self.backend == "petfast"
            else [
                pyscf_to_system(mol, dtype=self._dtype, device=self._device)
                for mol in mol_list
            ]
        )
        densities = self._predict(systems, mol_list)
        return densities[0] if single else densities

    def predict_systems(
        self,
        systems: System | list[System],
        *,
        charge: int = 0,
        spin: int | None = None,
    ) -> list[RIDensity]:
        """
        Predict RI densities for metatomic System(s).

        Requires the ``basis`` constructor argument (a ``Mole`` must be built
        to define the auxiliary basis and electron count). ``charge`` and
        ``spin`` apply to every system.

        :param systems: Single System or list of Systems.
        :param charge: Total molecular charge (default 0).
        :param spin: Number of unpaired electrons (2S); ``None`` lets PySCF
            auto-determine the multiplicity.
        :return: List of :class:`RIDensity`, one per system.
        """
        if self.basis is None:
            raise ValueError(
                "predict_systems requires the 'basis' constructor argument "
                "to build PySCF molecules from bare geometries."
            )
        if not isinstance(systems, list):
            systems = [systems]
        mols = [
            system_to_pyscf(
                system, basis=self.basis, charge=charge, spin=spin, ecp=self.ecp
            )
            for system in systems
        ]
        _warn_if_ecp_is_missing(mols, self.ecp)
        # A charge-conditioned model reads these off the System, so the caller's
        # `charge`/`spin` have to travel with it and not only with the Mole.
        # Data the caller already attached wins.
        for system, mol in zip(systems, mols, strict=True):
            attach_conditioning(system, mol.charge, mol.spin + 1)
        return self._predict(systems, mols)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _predict(self, systems: list[System], mols: list[gto.Mole]) -> list[RIDensity]:
        """Shared prediction path: coefficients from the backend, then corrections."""
        auxmols = [make_auxmol(mol, self.aux_basis) for mol in mols]

        outputs: dict[str, TensorMap] = {}
        if self.backend == "petfast":
            # Systems carry the cell; ``predict`` passes none for its molecules.
            if len(systems) == len(mols):
                coeffs_list = [
                    self._petfast_coefficients(mol, system)
                    for system, mol in zip(systems, mols, strict=True)
                ]
            else:
                coeffs_list = [self._petfast_coefficients(mol) for mol in mols]
        else:
            if self.oracle_charges:
                # Same GFN2 computation metatrain runs at training time; a failed
                # oracle leaves the field absent and the model runs oracle-free.
                from .conversions import attach_oracle_charges

                for system, mol in zip(systems, mols, strict=True):
                    attach_oracle_charges(system, mol.charge, mol.spin + 1)
            assert self._head is not None  # the constructor rejects a bare head
            outputs = self._evaluate(systems)
            coeffs_list = _tensormap_to_pyscf_coefficients(outputs[self._head])
        if self._zero_singlet_spin:
            coeffs_list = [
                np.zeros_like(c) if mol.spin == 0 else c
                for mol, c in zip(mols, coeffs_list, strict=True)
            ]

        for index, (mol, auxmol, coeffs) in enumerate(
            zip(mols, auxmols, coeffs_list, strict=True)
        ):
            _check_coefficient_count(auxmol, coeffs, _system_label(mol, index))

        if self.normalize_charge:
            coeffs_list = [
                normalize_charge_additive(
                    auxmol,
                    coeffs,
                    self._norm_target(mol),
                    metric=self.charge_metric,
                    max_shift=self._max_shift(electron_count(mol), mol.charge),
                    label=_system_label(mol, index),
                )
                for index, (mol, auxmol, coeffs) in enumerate(
                    zip(mols, auxmols, coeffs_list, strict=True)
                )
            ]

        if self.constrain_dipole:
            dipole_tm = outputs.get(self.dipole_head)
            if dipole_tm is None:
                raise ValueError(
                    f"constrain_dipole=True but model outputs have no key "
                    f"'{self.dipole_head}'; available keys: {sorted(outputs)}"
                )
            targets = self._extract_dipole_targets(dipole_tm, mols)
            coeffs_list = [
                self._constrain_dipole(auxmol, coeffs, mu_target)
                for auxmol, coeffs, mu_target in zip(
                    auxmols, coeffs_list, targets, strict=True
                )
            ]

        return [
            RIDensity(auxmol, coeffs)
            for auxmol, coeffs in zip(auxmols, coeffs_list, strict=True)
        ]

    def predict_coefficients_torch(
        self,
        atomic_numbers: np.ndarray,
        positions: torch.Tensor,
        cell: np.ndarray | None = None,
        *,
        charge: int = 0,
        spin: int | None = None,
    ) -> torch.Tensor:
        """
        RI coefficients as a torch tensor keeping the graph to ``positions``.

        The numpy path (:meth:`predict_systems`) detaches, which is what most
        callers want. This one is for differentiating a property of the density
        with respect to the geometry -- the derivative of a structure factor,
        say. Only the metatomic backend is supported.

        The charge correction uses the direction of the input geometry: the
        integrals ``∫χ_P`` do not depend on where the functions sit, so for the
        default ``identity`` metric this is exact, while ``coulomb`` neglects
        the geometry dependence of ``J = (P|Q)``.

        :param atomic_numbers: Shape (natm,).
        :param positions: Torch tensor of Cartesian positions in Å, shape
            (natm, 3); set ``requires_grad_()`` on it before calling.
        :param cell: Lattice vectors as rows in Å, ``None`` if isolated.
        :return: Coefficients in PySCF order, shape (naux,).
        """
        import torch

        from .conversions import build_system
        from .moments import charge_direction, compute_S_vector

        if self.backend != "metatomic":
            raise ValueError("predict_coefficients_torch needs the metatomic backend")
        if self._loaded_model is None:
            raise ValueError("no loaded model to differentiate")
        if self.basis is None:
            raise ValueError(
                "predict_coefficients_torch requires the 'basis' constructor "
                "argument to build a PySCF molecule from a bare geometry."
            )

        numbers = np.asarray(atomic_numbers)
        reference = positions.detach().cpu().to(torch.float64).numpy()
        system = build_system(numbers, reference, cell)
        mol = system_to_pyscf(
            system, basis=self.basis, charge=charge, spin=spin, ecp=self.ecp
        )
        auxmol = make_auxmol(mol, self.aux_basis)
        attach_conditioning(system, mol.charge, mol.spin + 1)

        system = system.to(device=self._device, dtype=self._dtype)
        system.positions = positions.to(device=self._device, dtype=self._dtype)
        for nl in self._nl_calculators:
            nl.add_neighbor_list(systems=[system])
        outputs = self._loaded_model([system], self._options, False)
        tensors = _tensormap_to_pyscf_tensors(outputs[self._head])
        coefficients = tensors[0].to(torch.float64)
        if self._zero_singlet_spin and mol.spin == 0:
            coefficients = torch.zeros_like(coefficients)

        if self.normalize_charge:
            device = coefficients.device
            S_vec = compute_S_vector(auxmol)
            integrals = torch.as_tensor(S_vec, dtype=coefficients.dtype, device=device)
            shift = self._norm_target(mol) - integrals @ coefficients
            direction = charge_direction(
                auxmol,
                S_vec,
                float(shift),
                metric=self.charge_metric,
                max_shift=self._max_shift(electron_count(mol), mol.charge),
                label=_system_label(mol, 0),
            )
            if direction is not None:
                coefficients = coefficients + shift * torch.as_tensor(
                    direction, dtype=coefficients.dtype, device=device
                )
        return coefficients

    def _max_shift(self, n_electrons: float, charge: int = 0) -> float | None:
        """The projection guard for a system of this size and charge, in electrons."""
        if self.charge_guard is None:
            return None
        limit = max(self.charge_guard, self.charge_guard_relative * n_electrons)
        if charge:
            limit = max(limit, self.charge_guard_per_charge * n_electrons * abs(charge))
        return limit

    def _petfast_coefficients(
        self, mol: gto.Mole, system: System | None = None
    ) -> np.ndarray:
        """One molecule through petfast, in PySCF auxiliary-basis order.

        The same extraction the metatomic path sees — positions in angstrom and
        the atomic numbers, not the ECP-reduced nuclear charges — so the two
        backends are fed identical inputs. The cell comes from ``system``, since
        the ``Mole`` carries none.
        """
        import torch

        positions = torch.tensor(
            mol.atom_coords() * BOHR_TO_ANG, dtype=self._dtype, device=self._device
        )
        types = torch.tensor(
            [gto.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)],
            dtype=torch.int32,
            device=self._device,
        )
        # PySCF stores spin as the number of unpaired electrons; the model was
        # conditioned on the multiplicity.
        conditioning = {
            "charge": int(mol.charge),
            "spin_multiplicity": int(mol.spin) + 1,
        }

        cell = None
        pbc = False
        if system is not None:
            cell_t = system.cell.to(dtype=self._dtype, device=self._device)
            pbc = bool(system.pbc.any()) and bool(torch.any(cell_t != 0))
            if pbc:
                cell = cell_t

        partition = (
            self.max_atoms_per_pass is not None
            and positions.shape[0] > self.max_atoms_per_pass
        )
        if partition and pbc:
            raise NotImplementedError(
                "petfast's partitioned (blocked) route has no periodic images; a "
                "periodic system above max_atoms_per_pass needs the metatomic backend "
                "or a smaller pass. See petfast.calculator."
            )
        compute = (
            self._calculator.compute_partitioned
            if partition
            else self._calculator.compute
        )
        coefficients, _ = compute(positions, types, cell=cell, pbc=pbc, **conditioning)
        return np.asarray(
            coefficients.detach().cpu().to(torch.float64).numpy(), dtype=float
        )

    def _evaluate(self, systems: list[System]) -> dict[str, TensorMap]:
        """Run a single forward pass (or return the pre-evaluated dict)."""
        import torch

        if isinstance(self._model, dict):
            return self._model
        assert self._loaded_model is not None  # set whenever model is not a dict

        for nl in self._nl_calculators:
            nl.add_neighbor_list(systems=systems)

        prepared = [s.to(device=self._device, dtype=self._dtype) for s in systems]
        with torch.no_grad():
            return self._loaded_model(prepared, self._options, False)

    @staticmethod
    def _extract_dipole_targets(
        dipole_tm: TensorMap, mols: list[gto.Mole]
    ) -> list[np.ndarray]:
        """
        Extract per-system electronic dipole targets at the centre of nuclear charge.

        :param dipole_tm: TensorMap with dipole output from the model.
        :param mols: Per-system PySCF Mole objects (orbital basis).
        :return: List of shape-(3,) arrays, one per system, in e·Bohr at R_cnc.
        """
        block = dipole_tm[0]
        values = block.values.detach().cpu().numpy()  # (n_systems, 3, 1)
        dipoles_cart = values[..., 0]  # (n_systems, 3)

        targets = []
        for dipole_total, mol in zip(dipoles_cart, mols, strict=True):
            z = nuclear_charges_of(mol).astype(float)
            coords = mol.atom_coords()  # Bohr
            R_cnc = z @ coords / z.sum()  # centre of nuclear charge
            mu_nuc_0 = z @ coords
            mu_el_cnc = dipole_total - mu_nuc_0 + mol.nelectron * R_cnc
            targets.append(mu_el_cnc)
        return targets

    @staticmethod
    def _constrain_dipole(
        auxmol: gto.Mole, coeffs: np.ndarray, mu_el_target: np.ndarray
    ) -> np.ndarray:
        """Charge-neutral correction moving the dipole to ``mu_el_target``."""
        z = nuclear_charges_of(auxmol).astype(float)
        R_cnc = z @ auxmol.atom_coords() / z.sum()  # Bohr
        delta_mu = mu_el_target - get_ri_dipole(auxmol, coeffs, R_cnc)
        return apply_dipole_correction(auxmol, coeffs, delta_mu)
