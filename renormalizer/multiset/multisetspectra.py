# -*- coding: utf-8 -*-

import numpy as np

from renormalizer.model.basis import BasisSHO
from renormalizer.mps import Mpo, MpDm, Mps, ThermalProp
from renormalizer.mps.lib import _sum
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mps import MsEvolveMethod, MultisetMps
from renormalizer.multiset.multiset_tdjob import MultisetTdJob, _state_bond_dims
from renormalizer.utils import CompressConfig, EvolveConfig, EvolveMethod, Quantity


def _state_inner_product(bra, ket) -> complex:
    return complex(
        bra.conj().dot(ket)
        * np.conjugate(bra.coeff)
        * ket.coeff
    )


def _multiset_overlap(bra: MultisetMps, ket: MultisetMps) -> complex:
    total = 0j
    for alpha in range(bra.N_electron):
        total += _state_inner_product(bra.msmps[alpha], ket.msmps[alpha])
    return complex(total)


def _multiset_dipole_overlap(bra: MultisetMps, ket: MultisetMps, dipole) -> complex:
    total = 0j
    for alpha in range(bra.N_electron):
        for beta in range(ket.N_electron):
            total += float(dipole[alpha]) * float(dipole[beta]) * _state_inner_product(
                bra.msmps[alpha], ket.msmps[beta]
            )
    return complex(total)


def _scale_multiset_state(state: MultisetMps, factor: float) -> MultisetMps:
    for alpha in range(state.N_electron):
        state.msmps[alpha].scale(float(factor), inplace=True)
    return state


class MultisetSpectraZeroT(MultisetTdJob):
    def __init__(
        self,
        model=None,
        spectratype: str = "abs",
        max_bonddim=None,
        ms_model: MultisetModel = None,
        evolve_config: EvolveConfig = None,
        compress_config: CompressConfig = None,
        offset: Quantity = Quantity(0),
        dump_mps: str = None,
        dump_dir: str = None,
        job_name: str = None,
        expand: bool = False,
    ):
        if spectratype not in ["abs", "emi"]:
            raise ValueError(f"Unsupported spectratype: {spectratype}")

        self.spectratype = spectratype
        self.offset = offset
        self.temperature = Quantity(0, "K")
        self.expand = expand
        self._autocorr = []
        self._bond_dims = []

        if ms_model is None:
            if model is None or max_bonddim is None:
                raise ValueError("Either provide `ms_model` or both `model` and `max_bonddim`.")
            self.ms_model = MultisetModel(
                model,
                max_bonddim=max_bonddim,
                temperature=Quantity(0, "K"),
                evolve_config=evolve_config,
                compress_config=compress_config,
                auto_init=False,
            )
        else:
            if ms_model.temperature != Quantity(0, "K"):
                raise ValueError("`MultisetSpectraZeroT` only supports zero-temperature `ms_model`.")
            if spectratype == "emi":
                if ms_model.MsMps is None:
                    raise ValueError("`ms_model` should already contain an excited multiset state for emission.")
                if evolve_config is not None:
                    ms_model.evolve_config = evolve_config
                if compress_config is not None:
                    ms_model.compress_config = compress_config
                self.ms_model = ms_model
            else:
                self.ms_model = MultisetModel(
                    ms_model.model,
                    max_bonddim=None,
                    temperature=Quantity(0, "K"),
                    method=ms_model.method,
                    evolve_config=ms_model.evolve_config if evolve_config is None else evolve_config,
                    compress_config=ms_model.compress_config if compress_config is None else compress_config,
                    auto_init=False,
                )

        self.model = self.ms_model.model
        self.h_mpo = Mpo(self.ms_model.init_model, offset=self.offset)
        if spectratype == "emi":
            if evolve_config is None or isinstance(self.ms_model.evolve_config.method, MsEvolveMethod):
                evolve_config = EvolveConfig(method=EvolveMethod.tdvp_ps, adaptive=False)
        super().__init__(
            evolve_config=self.ms_model.evolve_config if spectratype == "abs" else evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
        )

    def init_mps(self):
        if self.spectratype == "emi":
            return self.init_mps_emi()
        return self.init_mps_abs()

    def init_mps_abs(self):
        ket = self._init_weighted_ket(self._get_dipole_vector())
        if self.expand:
            ket = self._expand_initial_ket(ket)
        return ket.copy(), ket

    def init_mps_emi(self):
        if self.ms_model.MsMps is None:
            ket = self._init_weighted_ket(np.ones(self.model.n_edofs))
        else:
            ket = self.ms_model.MsMps.copy()
            for alpha in range(ket.N_electron):
                ket.msmps[alpha].compress_config = self.ms_model.compress_config

        dipole = self._get_dipole_vector()
        components = []
        for alpha in range(ket.N_electron):
            state = ket.msmps[alpha].copy()
            state.scale(float(dipole[alpha]), inplace=True)
            state.compress_config = self.ms_model.compress_config
            components.append(state)

        ket = _sum(components, compress=False)
        ket.compress_config = self.ms_model.compress_config
        ket.evolve_config = self.evolve_config
        return ket.copy(), ket

    def evolve_single_step(self, evolve_dt):
        bra, ket = self.latest_mps
        if isinstance(ket, MultisetMps):
            ket = self.ms_model.evolve_state(ket, evolve_dt, normalize=False)
        else:
            ket = ket.evolve(self.h_mpo, evolve_dt, normalize=False)
        return bra, ket

    def process_mps(self, mps):
        bra, ket = mps
        if isinstance(bra, MultisetMps):
            self._autocorr.append(_multiset_overlap(bra, ket))
        else:
            self._autocorr.append(complex(bra.conj().dot(ket)))
        self._bond_dims.append(_state_bond_dims(mps))

    @property
    def autocorr(self):
        return np.array(self._autocorr)

    @property
    def bond_dims(self):
        return np.array(self._bond_dims, dtype=object)

    def get_dump_dict(self):
        return {
            "temperature": self.temperature.as_au(),
            "time series": self.evolve_times,
            "autocorr": self.autocorr,
            "bond_dims": self.bond_dims,
        }

    def init_mp(self):
        init_mp = Mps.hartree_product_state(model=self.ms_model.init_model)
        init_mp.compress_config = self.ms_model.compress_config
        return init_mp

    def _get_dipole_vector(self) -> np.ndarray:
        dipole = getattr(self.model, "dipole", None)
        if dipole is None:
            raise ValueError("`model.dipole` is required for MultisetSpectraZeroT.")

        if isinstance(dipole, dict):
            try:
                dipole = [dipole[dof] for dof in self.model.e_dofs]
            except KeyError as exc:
                raise ValueError(
                    "`model.dipole` must contain one absorption coefficient per electronic dof."
                ) from exc

        dipole = np.asarray(dipole, dtype=float)
        if dipole.ndim == 0:
            dipole = np.repeat(dipole, self.model.n_edofs)
        elif dipole.ndim > 1:
            dipole = dipole.reshape(-1)

        if len(dipole) != self.model.n_edofs:
            raise ValueError(
                f"`model.dipole` must resolve to one absorption coefficient per electronic excitation. "
                f"Expected {self.model.n_edofs}, got {len(dipole)}."
            )
        return dipole

    def _init_weighted_ket(self, weights) -> MultisetMps:
        phi_g = self.init_mp()
        weights = np.asarray(weights, dtype=float)
        msmps = []
        for alpha in range(self.ms_model.N_electron):
            state = phi_g.copy()
            state.scale(float(weights[alpha]), inplace=True)
            state.compress_config = self.ms_model.compress_config
            msmps.append(state)

        return MultisetMps(
            self.ms_model.MsModel,
            self.ms_model.N_electron,
            temperature=Quantity(0, "K"),
            init_model=self.ms_model.init_model,
            method=self.ms_model.method,
            msmps=msmps,
        )

    def _expand_initial_ket(self, ket: MultisetMps, coef: float = 1e-10, use_hint: bool = True) -> MultisetMps:
        initial_norm = _multiset_overlap(ket, ket).real

        self.ms_model.set_mps(ket)
        self.ms_model.expand_bond_dimension_multiset(coef=coef, use_hint=use_hint)
        expanded_ket = self.ms_model.MsMps

        expanded_norm = _multiset_overlap(expanded_ket, expanded_ket).real
        if expanded_norm <= 0:
            raise ValueError("Expanded multiset spectra initial state has non-positive norm.")

        if not np.isclose(initial_norm, expanded_norm):
            _scale_multiset_state(expanded_ket, np.sqrt(initial_norm / expanded_norm))

        return expanded_ket


class MultisetSpectraFiniteT(MultisetTdJob):
    def __init__(
        self,
        model=None,
        spectratype: str = "abs",
        temperature: Quantity = Quantity(298, "K"),
        insteps: int = 1,
        thermal_init_method: str = "imaginary_time_exact",
        max_bonddim=None,
        ms_model: MultisetModel = None,
        evolve_config: EvolveConfig = None,
        compress_config: CompressConfig = None,
        ievolve_config: EvolveConfig = None,
        offset: Quantity = Quantity(0),
        dump_mps: str = None,
        dump_dir: str = None,
        job_name: str = None,
        expand: bool = True,
    ):
        if spectratype not in ["abs", "emi"]:
            raise ValueError(f"Unsupported spectratype: {spectratype}")
        if temperature == 0:
            raise ValueError("`MultisetSpectraFiniteT` requires a non-zero temperature.")

        self.spectratype = spectratype
        self.temperature = temperature
        self.insteps = insteps
        self.thermal_init_method = thermal_init_method
        self.offset = offset
        self.expand = expand
        self._autocorr = []
        self._bond_dims = []

        if ms_model is None:
            if model is None or max_bonddim is None:
                raise ValueError("Either provide `ms_model` or both `model` and `max_bonddim`.")
            self.ms_model = MultisetModel(
                model,
                max_bonddim=max_bonddim,
                temperature=temperature,
                method=self._normalize_thermal_init_method(),
                evolve_config=evolve_config,
                compress_config=compress_config,
                auto_init=False,
            )
        else:
            if evolve_config is not None:
                ms_model.evolve_config = evolve_config
            if compress_config is not None:
                ms_model.compress_config = compress_config
            self.ms_model = ms_model

        self.model = self.ms_model.model
        self.h_mpo = Mpo(self.ms_model.init_model, offset=self.offset)
        self.h_mpo_gs = Mpo(self.ms_model.init_model, offset=Quantity(0))
        self.icompress_config = self.ms_model.compress_config
        self.local_evolve_config = EvolveConfig(method=EvolveMethod.tdvp_ps, adaptive=False)
        self.ievolve_config = (
            EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps)
            if ievolve_config is None else ievolve_config
        )

        super().__init__(
            evolve_config=self.ms_model.evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
        )

    def _normalize_thermal_init_method(self):
        method = self.thermal_init_method.lower().replace("-", "_").replace(" ", "_")
        if method in ["exact", "imaginary_time_exact"]:
            return "imaginary_time_exact"
        if method in ["propagate", "imaginary_time", "imaginary_time_propagate"]:
            return "imaginary_time_propagate"
        if method in ["thermofield", "thermo_field"]:
            raise ValueError("Thermo-field dynamics is not supported for multiset spectra.")
        raise ValueError(f"Unsupported thermal_init_method: {self.thermal_init_method}")

    def init_mps(self):
        if self.spectratype == "emi":
            return self.init_mps_emi()
        return self.init_mps_abs()

    def init_mps_abs(self):
        thermal_mpdm = self.init_mp()
        ket = self._broadcast_local_state(thermal_mpdm, self._get_dipole_vector())
        if self.expand:
            ket = self._expand_initial_multiset_state(ket)
        self._set_multiset_hamiltonian_offset(self.offset)
        return ket.copy(), ket

    def init_mps_emi(self):
        thermal_state = self._init_excited_thermal_state()
        if self.expand:
            thermal_state = self._expand_initial_multiset_state(thermal_state)
        self._set_multiset_hamiltonian_offset(self.offset)
        return thermal_state.copy(), thermal_state

    def init_mp(self, method=None):
        method = self._normalize_thermal_init_method() if method is None else method
        if method == "imaginary_time_exact":
            return self._exact_local_thermal_mpdm(self.ms_model.init_model)
        if method == "imaginary_time_propagate":
            local_state = MpDm.max_entangled_gs(self.ms_model.init_model)
            local_state.compress_config = self.icompress_config
            tp = ThermalProp(
                local_state,
                h_mpo_model=self.ms_model.init_model,
                evolve_config=EvolveConfig(method=EvolveMethod.tdvp_ps, adaptive=False),
                auto_expand=False,
            )
            tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
            thermal_state = tp.latest_mps
            thermal_state.compress_config = self.icompress_config
            thermal_state.evolve_config = self.evolve_config
            return thermal_state
        raise ValueError(f"Unsupported thermal initialisation method: {method}")

    def _exact_local_thermal_mpdm(self, model):
        beta = self.temperature.to_beta()
        condition = {}
        for basis in model.basis:
            if not isinstance(basis, BasisSHO):
                continue
            weights = np.exp(-0.5 * beta * basis.omega * np.arange(basis.nbas, dtype=float))
            weights /= np.linalg.norm(weights)
            condition[basis.dof] = weights

        thermal_mps = Mps.hartree_product_state(model=model, condition=condition)
        thermal_mpdm = MpDm.from_mps(thermal_mps)
        thermal_mpdm.compress_config = self.icompress_config
        thermal_mpdm.evolve_config = self.evolve_config
        return thermal_mpdm

    def _init_excited_thermal_state(self):
        method = self._normalize_thermal_init_method()
        if method == "imaginary_time_exact":
            msmps = [
                self._exact_local_thermal_mpdm(self.ms_model.MsModel[alpha][alpha])
                for alpha in range(self.ms_model.N_electron)
            ]
            return self._build_multiset_state(msmps)

        msmps = []
        for alpha in range(self.ms_model.N_electron):
            state = MpDm.max_entangled_gs(self.ms_model.MsModel[alpha][alpha])
            state.compress_config = self.icompress_config
            state.evolve_config = self.evolve_config
            msmps.append(state)
        return self._imaginary_time_propagate_multiset(self._build_multiset_state(msmps))

    def _build_multiset_state(self, msmps):
        for state in msmps:
            state.compress_config = self.icompress_config
            state.evolve_config = self.evolve_config
        return MultisetMps(
            self.ms_model.MsModel,
            self.ms_model.N_electron,
            temperature=self.temperature,
            init_model=self.ms_model.init_model,
            method=self._normalize_thermal_init_method(),
            msmps=msmps,
        )

    def _imaginary_time_propagate_multiset(self, state: MultisetMps):
        if self.insteps is None:
            raise ValueError("`insteps` must be defined for imaginary-time propagation.")
        evolve_dt = self.temperature.to_beta() / (2j * self.insteps)
        original_evolve_config = self.ms_model.evolve_config
        self.ms_model.evolve_config = self.ievolve_config
        try:
            for _ in range(self.insteps):
                state = self.ms_model.evolve_state(state, evolve_dt, normalize=True)
            return state
        finally:
            self.ms_model.evolve_config = original_evolve_config

    def _broadcast_local_state(self, local_state, weights=None):
        if weights is None:
            weights = np.ones(self.ms_model.N_electron)
        weights = np.asarray(weights, dtype=float)
        msmps = []
        for alpha in range(self.ms_model.N_electron):
            state = local_state.copy()
            state.scale(float(weights[alpha]), inplace=True)
            state.compress_config = self.icompress_config
            state.evolve_config = self.evolve_config
            msmps.append(state)

        return MultisetMps(
            self.ms_model.MsModel,
            self.ms_model.N_electron,
            temperature=self.temperature,
            init_model=self.ms_model.init_model,
            method=self._normalize_thermal_init_method(),
            msmps=msmps,
        )

    def _collapse_with_dipole(self, state: MultisetMps):
        dipole = self._get_dipole_vector()
        components = []
        for alpha in range(state.N_electron):
            component = state.msmps[alpha].copy()
            component.scale(float(dipole[alpha]), inplace=True)
            component.compress_config = self.icompress_config
            components.append(component)
        return _sum(components, compress=False)

    def _expand_initial_multiset_state(self, state: MultisetMps, coef: float = 1e-10) -> MultisetMps:
        self.ms_model.set_mps(state)
        self.ms_model.expand_bond_dimension_multiset(coef=coef, use_hint=True)
        expanded_state = self.ms_model.MsMps
        for alpha in range(expanded_state.N_electron):
            expanded_state.msmps[alpha].compress_config = self.icompress_config
            expanded_state.msmps[alpha].evolve_config = self.evolve_config
        return expanded_state

    def _set_multiset_hamiltonian_offset(self, energy):
        if not isinstance(energy, Quantity):
            energy = Quantity(energy)
        for alpha in range(self.ms_model.N_electron):
            for beta in range(self.ms_model.N_electron):
                if len(self.ms_model.MsModel[alpha][beta].ham_terms) == 0:
                    self.ms_model.MsMpo.msmpo[alpha][beta] = []
                elif alpha == beta:
                    self.ms_model.MsMpo.msmpo[alpha][beta] = Mpo(
                        model=self.ms_model.MsModel[alpha][beta],
                        terms=None,
                        offset=energy,
                    )
                else:
                    self.ms_model.MsMpo.msmpo[alpha][beta] = Mpo(
                        model=self.ms_model.MsModel[alpha][beta],
                        terms=None,
                        offset=Quantity(0),
                    )
        self.ms_model._refresh_mpo_cache()

    def evolve_single_step(self, evolve_dt):
        bra, ket = self.latest_mps
        if len(self.evolve_times) % 2 == 1:
            ket = self._evolve_finite_temperature_branch(ket, evolve_dt)
        else:
            bra = self._evolve_finite_temperature_branch(bra, -evolve_dt)
        return bra, ket

    def _evolve_finite_temperature_branch(self, state, evolve_dt):
        if isinstance(state, MultisetMps):
            state = self._evolve_ground_multiset(state, -evolve_dt)
            return self.ms_model.evolve_state(state, evolve_dt, normalize=False)

        return self._evolve_ground_mpdm(state, evolve_dt)

    def _evolve_ground_multiset(self, state: MultisetMps, evolve_dt):
        msmps = []
        for mpdm in state.msmps:
            msmps.append(self._evolve_ground_mpdm(mpdm, evolve_dt))
        return MultisetMps(
            state.MsModel,
            state.N_electron,
            temperature=state.temperature,
            init_model=state.init_model,
            method=state.method,
            msmps=msmps,
        )

    def _evolve_ground_mpdm(self, mpdm, evolve_dt):
        mpdm = self._ensure_array_qn(mpdm)
        mpdm.compress_config = self.icompress_config
        mpo_prop = self._exact_local_ground_propagator(-1.0j * evolve_dt)
        new_mpdm = mpdm.apply(mpo_prop, canonicalise=True)
        new_mpdm.evolve_config = self.local_evolve_config
        new_mpdm.compress_config = self.icompress_config
        return new_mpdm

    def _exact_local_ground_propagator(self, x):
        mpo = Mpo()
        if np.iscomplex(x):
            mpo.to_complex(inplace=True)
        mpo.model = self.ms_model.init_model

        for basis in self.ms_model.init_model.basis:
            if not isinstance(basis, BasisSHO):
                raise TypeError("Exact multiset finite-T ground propagation only supports BasisSHO.")
            omega = basis.omega[0] if np.ndim(basis.omega) > 0 else basis.omega
            diag = np.exp(x * omega * np.arange(basis.nbas, dtype=float))
            mpo.append(np.diag(diag).reshape(1, basis.nbas, basis.nbas, 1))

        qn_size = mpo.model.qn_size
        mpo.qn = [np.zeros((1, qn_size), dtype=int)] * (len(mpo) + 1)
        mpo.qnidx = len(mpo) - 1
        mpo.qntot = np.zeros(qn_size, dtype=int)
        mpo.to_right = False
        return mpo

    @staticmethod
    def _ensure_array_qn(mpdm):
        mpdm.qn = [np.asarray(qn) for qn in mpdm.qn]
        return mpdm

    def process_mps(self, mps):
        bra, ket = mps
        if isinstance(bra, MultisetMps):
            if self.spectratype == "emi":
                ft = _multiset_dipole_overlap(bra, ket, self._get_dipole_vector())
            else:
                ft = _multiset_overlap(bra, ket)
        else:
            ft = _state_inner_product(bra, ket)
        if self.spectratype == "emi":
            ft = np.conjugate(ft)
        self._autocorr.append(ft)
        self._bond_dims.append(_state_bond_dims(mps))

    def stop_evolve_criteria(self):
        return False

    @property
    def autocorr(self):
        return np.array(self._autocorr)

    @property
    def bond_dims(self):
        return np.array(self._bond_dims, dtype=object)

    def get_dump_dict(self):
        return {
            "temperature": self.temperature.as_au(),
            "time series": self.evolve_times,
            "autocorr": self.autocorr,
            "bond_dims": self.bond_dims,
        }

    def _get_dipole_vector(self) -> np.ndarray:
        dipole = getattr(self.model, "dipole", None)
        if dipole is None:
            raise ValueError("`model.dipole` is required for MultisetSpectraFiniteT.")

        if isinstance(dipole, dict):
            try:
                dipole = [dipole[dof] for dof in self.model.e_dofs]
            except KeyError as exc:
                raise ValueError(
                    "`model.dipole` must contain one absorption coefficient per electronic dof."
                ) from exc

        dipole = np.asarray(dipole, dtype=float)
        if dipole.ndim == 0:
            dipole = np.repeat(dipole, self.model.n_edofs)
        elif dipole.ndim > 1:
            dipole = dipole.reshape(-1)

        if len(dipole) != self.model.n_edofs:
            raise ValueError(
                f"`model.dipole` must resolve to one absorption coefficient per electronic excitation. "
                f"Expected {self.model.n_edofs}, got {len(dipole)}."
            )
        return dipole
