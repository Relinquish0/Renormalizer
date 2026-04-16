# -*- coding: utf-8 -*-

import numpy as np

from renormalizer.mps import Mpo, Mps
from renormalizer.mps.lib import _sum
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mps import MsEvolveMethod, MultisetMps
from renormalizer.multiset.multiset_tdjob import MultisetTdJob, _state_bond_dims
from renormalizer.utils import CompressConfig, EvolveConfig, Quantity


def _multiset_overlap(bra: MultisetMps, ket: MultisetMps) -> complex:
    total = 0j
    for alpha in range(bra.N_electron):
        total += bra.msmps[alpha].conj().dot(ket.msmps[alpha])
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
        expand = False
    ):
        if spectratype not in ["abs", "emi"]:
            raise ValueError(f"Unsupported spectratype: {spectratype}")

        self.spectratype = spectratype
        self.offset = offset
        self._emi_excited_state = None

        if ms_model is None:
            if model is None or max_bonddim is None:
                raise ValueError("Either provide `ms_model` or both `model` and `max_bonddim`.")
            ms_model = MultisetModel(
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
                self._emi_excited_state = ms_model.MsMps.copy()
                if evolve_config is not None:
                    ms_model.evolve_config = evolve_config
                if compress_config is not None:
                    ms_model.compress_config = compress_config
            else:
                ms_model = MultisetModel(
                    ms_model.model,
                    max_bonddim=None,
                    temperature=Quantity(0, "K"),
                    method=ms_model.method,
                    evolve_config=ms_model.evolve_config if evolve_config is None else evolve_config,
                    compress_config=ms_model.compress_config if compress_config is None else compress_config,
                    auto_init=False,
                )

        self.ms_model = ms_model
        self.model = self.ms_model.model
        self.temperature = Quantity(0, "K")
        self._autocorr = []
        self._bond_dims = []
        self.expand = expand
        self._emi_gs_mpo = None

        if self.spectratype == "emi":
            self._emi_gs_mpo = Mpo(self.ms_model.init_model, offset=self.offset)
            if evolve_config is None or isinstance(self.ms_model.evolve_config.method, MsEvolveMethod):
                self._emi_evolve_config = EvolveConfig()
            else:
                self._emi_evolve_config = evolve_config
            job_evolve_config = self._emi_evolve_config
        else:
            self._emi_evolve_config = None
            job_evolve_config = self.ms_model.evolve_config

        super().__init__(
            evolve_config=job_evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
        )

    def init_mp(self):
        init_mp = Mps.hartree_product_state(model=self.ms_model.init_model)
        init_mp.compress_config = self.ms_model.compress_config
        return init_mp

    def _get_abs_dipole_vector(self) -> np.ndarray:
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

    def _build_absorption_ket(self) -> MultisetMps:
        return self._init_weighted_ket(self._get_abs_dipole_vector())

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

    def _build_emission_state(self) -> MultisetMps:
        if self._emi_excited_state is not None:
            excited_state = self._emi_excited_state.copy()
            for alpha in range(excited_state.N_electron):
                excited_state.msmps[alpha].compress_config = self.ms_model.compress_config
            return excited_state
        return self._init_weighted_ket(np.ones(self.model.n_edofs))

    def _contract_emission_ground_state(self, excited_state: MultisetMps):
        dipole = self._get_abs_dipole_vector()
        ground_components = []
        for alpha in range(excited_state.N_electron):
            state = excited_state.msmps[alpha].copy()
            state.scale(float(dipole[alpha]), inplace=True)
            state.compress_config = self.ms_model.compress_config
            ground_components.append(state)

        ground_state = _sum(ground_components, compress=False)
        ground_state.compress_config = self.ms_model.compress_config
        ground_state.evolve_config = self._emi_evolve_config
        return ground_state

    def init_mps(self):
        if self.spectratype == "abs":
            if self.expand:
                ket = self._expand_initial_ket(self._build_absorption_ket())
            else:
                ket = self._build_absorption_ket()
        else:
            ket = self._contract_emission_ground_state(self._build_emission_state())
        bra = ket.copy()
        return bra, ket

    def process_mps(self, mps):
        bra, ket = mps
        if self.spectratype == "abs":
            self._autocorr.append(_multiset_overlap(bra, ket))
        else:
            self._autocorr.append(complex(bra.conj().dot(ket)))
        self._bond_dims.append(_state_bond_dims(mps))

    def evolve_single_step(self, evolve_dt):
        bra, ket = self.latest_mps
        if self.spectratype == "abs":
            new_ket = self.ms_model.evolve_state(ket, evolve_dt, normalize=False)
        else:
            new_ket = ket.evolve(self._emi_gs_mpo, evolve_dt, normalize=False)
        return bra, new_ket

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
