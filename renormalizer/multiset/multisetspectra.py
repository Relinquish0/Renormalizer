# -*- coding: utf-8 -*-

import logging
from typing import Union

import numpy as np

from renormalizer.model.basis import BasisSHO
from renormalizer.mps import Mpo, MpDm, Mps, ThermalProp
from renormalizer.mps.lib import _sum
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mps import (
    ElectronicAncillaMultisetMps,
    MsEvolveMethod,
    MultisetMps,
    _state_inner_product,
)
from renormalizer.multiset.multiset_tdjob import MultisetTdJob, _state_bond_dims
from renormalizer.utils import CompressConfig, EvolveConfig, EvolveMethod, Quantity


logger = logging.getLogger(__name__)


def _multiset_pair_overlaps(bra: MultisetMps, ket: MultisetMps) -> np.ndarray:
    overlaps = np.zeros((bra.N_electron, ket.N_electron), dtype=np.complex128)
    if getattr(bra, "electronic_ancilla", False):
        for alpha in range(bra.N_electron):
            for beta in range(ket.N_electron):
                for ancilla in range(bra.n_anc):
                    overlaps[alpha, beta] += _state_inner_product(
                        bra.get(alpha, ancilla),
                        ket.get(beta, ancilla),
                    )
        return overlaps

    for alpha in range(bra.N_electron):
        for beta in range(ket.N_electron):
            overlaps[alpha, beta] = _state_inner_product(bra.msmps[alpha], ket.msmps[beta])
    return overlaps


def _multiset_overlap(
    bra: MultisetMps, ket: MultisetMps, return_pair_overlaps: bool = False
) -> Union[complex, np.ndarray]:
    overlaps = _multiset_pair_overlaps(bra, ket)
    if return_pair_overlaps:
        return overlaps
    return complex(np.trace(overlaps))


def _multiset_dipole_overlap(bra: MultisetMps, ket: MultisetMps, dipole) -> complex:
    total = 0j
    if getattr(bra, "electronic_ancilla", False):
        for ancilla in range(bra.n_anc):
            for alpha in range(bra.N_electron):
                for beta in range(ket.N_electron):
                    total += float(dipole[alpha]) * float(dipole[beta]) * _state_inner_product(
                        bra.get(alpha, ancilla),
                        ket.get(beta, ancilla),
                    )
        return complex(total)

    for alpha in range(bra.N_electron):
        for beta in range(ket.N_electron):
            total += float(dipole[alpha]) * float(dipole[beta]) * _state_inner_product(
                bra.msmps[alpha], ket.msmps[beta]
            )
    return complex(total)


def _multiset_cross_overlap(bra: MultisetMps, ket: MultisetMps) -> complex:
    return complex(_multiset_pair_overlaps(bra, ket).sum())


def _scale_multiset_state(state: MultisetMps, factor: float) -> MultisetMps:
    for mps in state.msmps:
        mps.scale(factor, inplace=True)
    return state


class MultisetSpectraZeroT(MultisetTdJob):
    def __init__(
        self,
        model=None,
        spectratype: str = "abs",
        max_bonddim=None,
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
        self._autocorr_components = []
        self._bond_dims = []

        if model is None or max_bonddim is None:
            raise ValueError("Both `model` and `max_bonddim` are required.")
        self.ms_model = MultisetModel(
            model,
            max_bonddim=max_bonddim,
            temperature=Quantity(0, "K"),
            evolve_config=evolve_config,
            compress_config=compress_config,
            auto_init=False,
        )

        self.model = self.ms_model.model
        self.h_mpo = Mpo(self.ms_model.init_model, offset=self.offset)
        super().__init__(
            evolve_config=self.ms_model.evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
        )

    def init_mps(self):
        if self.spectratype == "emi":
            return self.init_mps_emi()
        return self.init_mps_abs()

    def init_mps_abs(self):
        ket = self._init_dipole()
        if self.expand:
            ket = self._expand_ket_bonddim(ket)
        return ket.copy(), ket

    def init_mps_emi(self):
        ket = self._init_dipole()
        if self.expand:
            ket = self._expand_ket_bonddim(ket)
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
            component_autocorr = _multiset_overlap(bra, ket, return_pair_overlaps=True)
            if self.spectratype == "emi":
                self._autocorr.append(complex(component_autocorr.sum()))
            else:
                self._autocorr.append(complex(np.trace(component_autocorr)))
            self._autocorr_components.append(component_autocorr)
        else:
            component_autocorr = np.asarray([[complex(bra.conj().dot(ket))]])
            self._autocorr.append(component_autocorr[0, 0])
            self._autocorr_components.append(component_autocorr)
        self._bond_dims.append(_state_bond_dims(mps))

    @property
    def autocorr(self):
        return np.array(self._autocorr)

    @property
    def autocorr_components(self):
        return np.array(self._autocorr_components)

    @property
    def bond_dims(self):
        return np.array(self._bond_dims, dtype=object)

    def get_dump_dict(self):
        return {
            "temperature": self.temperature.as_au(),
            "time series": self.evolve_times,
            "time_series": self.evolve_times,
            "autocorr": self.autocorr,
            "autocorr_components": self.autocorr_components,
            "bond_dims": self.bond_dims,
        }

    def init_mp(self):
        init_mp = Mps.hartree_product_state(model=self.ms_model.init_model)
        init_mp.compress_config = self.ms_model.compress_config
        return init_mp

    def _get_dipole(self) -> np.ndarray:
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

    def _init_dipole(self):
        dipole_weights = self._get_dipole()
        phi_g = self.init_mp() # HF state
        dipole_weights = np.asarray(dipole_weights, dtype=float)
        msmps = []
        for alpha in range(self.ms_model.N_electron):
            state = phi_g.copy()
            state.scale(float(dipole_weights[alpha]), inplace=True)
            state.compress_config = self.ms_model.compress_config
            msmps.append(state)

        ket = MultisetMps(
            self.ms_model.MsModel,
            self.ms_model.N_electron,
            temperature=Quantity(0, "K"),
            init_model=self.ms_model.init_model,
            method=self.ms_model.method,
            msmps=msmps,
        )

        return ket

    def _expand_ket_bonddim(self, ket: MultisetMps, coef: float = 1e-10, use_hint: bool = True) -> MultisetMps:
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
        evolve_config: EvolveConfig = None,
        compress_config: CompressConfig = None,
        ievolve_config: EvolveConfig = None,
        offset: Quantity = Quantity(0),
        dump_mps: str = None,
        dump_dir: str = None,
        job_name: str = None,
        expand: bool = True,
        electronic_ancilla: bool = False,
        n_electronic_ancilla: int = None,
        electronic_purification: str = "thermal",
        electronic_beta: float = None,
        electronic_temperature: Quantity = None,
        electronic_initial_distribution=None,
        electronic_hamiltonian=None,
        electronic_hamiltonian_source: str = "msmodel",
    ):
        if spectratype not in ["abs", "emi"]:
            raise ValueError(f"Unsupported spectratype: {spectratype}")
        if temperature == 0:
            raise ValueError("`MultisetSpectraFiniteT` requires a non-zero temperature.")
        if thermal_init_method not in ["imaginary_time_exact", "imaginary_time_propagate"]:
            raise ValueError(f"Unsupported thermal_init_method: {thermal_init_method}")
        if electronic_purification not in ["thermal", "diagonal"]:
            raise ValueError(f"Unsupported electronic_purification: {electronic_purification}")

        self.spectratype = spectratype
        self.temperature = temperature
        self.insteps = insteps
        self.thermal_init_method = thermal_init_method # "imaginary_time_exact" or "imaginary_time_propagate"
        self.offset = offset
        self.expand = expand
        self.electronic_ancilla = electronic_ancilla
        self.n_electronic_ancilla = n_electronic_ancilla
        self.electronic_purification = electronic_purification
        self.electronic_beta = electronic_beta
        self.electronic_temperature = electronic_temperature
        self.electronic_initial_distribution = electronic_initial_distribution
        self.electronic_hamiltonian = electronic_hamiltonian
        self.electronic_hamiltonian_source = electronic_hamiltonian_source
        self._autocorr = []
        self._autocorr_components = []
        self._bond_dims = []

        if model is None or max_bonddim is None:
            raise ValueError("Both `model` and `max_bonddim` are required.")
        self.ms_model = MultisetModel(
            model,
            max_bonddim=max_bonddim,
            temperature=temperature,
            method=self.thermal_init_method,
            evolve_config=evolve_config,
            compress_config=compress_config,
            auto_init=False,
        )

        self.model = self.ms_model.model
        self.h_mpo = Mpo(self.ms_model.init_model, offset=self.offset)
        self.h_mpo_gs = Mpo(self.ms_model.init_model, offset=Quantity(0))
        self.icompress_config = self.ms_model.compress_config
        self.local_evolve_config = EvolveConfig(method=EvolveMethod.tdvp_ps, adaptive=False)
        self.ievolve_config = (
            EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps)
            if ievolve_config is None else ievolve_config
        )

        job_evolve_config = self.ms_model.evolve_config if spectratype == "abs" else self.local_evolve_config
        super().__init__(
            evolve_config=job_evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
        )

    def init_mps(self):
        if self.spectratype == "emi":
            return self.init_mps_emi()
        elif self.spectratype == "abs":
            return self.init_mps_abs()

    def init_mps_abs(self):
        thermal_mpdm = self._init_ground_thermal_state()
        ket = self._init_dipole(thermal_mpdm)
        if self.expand:
            ket = self._expand_initial_multiset_state(ket)
        self._set_multiset_hamiltonian_offset(self.offset)
        return ket.copy(), ket

    def init_mps_emi(self):
        thermal_state = self._init_excited_thermal_state()
        ket = self._init_dipole(thermal_state)
        if self.expand:
            ket = self._expand_initial_multiset_state(ket)
        self.ms_model.set_mps(ket)
        excited_energy = Quantity(self.ms_model.Hamiltonian())
        logger.info(
            "Finite-temperature multiset emission subtracts excited-state carrier energy %s au (%s eV)",
            excited_energy.as_au(),
            excited_energy.as_au() / Quantity(1, "eV").as_au(),
        )
        self._set_multiset_hamiltonian_offset(excited_energy + self.offset)
        return ket.copy(), ket

    def _electronic_ancilla_dim(self) -> int:
        return self.ms_model.N_electron if self.n_electronic_ancilla is None else int(self.n_electronic_ancilla)

    def _electronic_beta_value(self) -> float:
        if self.electronic_beta is not None:
            return float(self.electronic_beta)
        if self.electronic_temperature is not None:
            return self.electronic_temperature.to_beta()
        return self.temperature.to_beta()

    def _electronic_hamiltonian_matrix(self) -> np.ndarray:
        if self.electronic_hamiltonian is not None:
            h_e = np.asarray(self.electronic_hamiltonian, dtype=np.complex128)
        elif self.electronic_hamiltonian_source == "msmodel":
            if not hasattr(self.model, "mol_list") or not hasattr(self.model, "j_matrix"):
                raise ValueError("`electronic_hamiltonian_source='msmodel'` requires a Holstein-like model.")
            h_e = np.asarray(self.model.j_matrix, dtype=np.complex128).copy()
            for alpha, mol in enumerate(self.model.mol_list):
                h_e[alpha, alpha] = mol.elocalex + mol.e0
        else:
            raise ValueError(
                f"Unsupported electronic_hamiltonian_source: {self.electronic_hamiltonian_source}"
            )

        if h_e.shape != (self.ms_model.N_electron, self.ms_model.N_electron):
            raise ValueError(
                f"Electronic Hamiltonian shape mismatch: expected "
                f"{(self.ms_model.N_electron, self.ms_model.N_electron)}, got {h_e.shape}."
            )
        return h_e

    def _build_electronic_purification_coefficients(self) -> np.ndarray:
        n_phys = self.ms_model.N_electron
        n_anc = self._electronic_ancilla_dim()

        if self.electronic_purification == "diagonal":
            if self.electronic_initial_distribution is None:
                raise ValueError("`electronic_initial_distribution` is required for diagonal purification.")
            probs = np.asarray(self.electronic_initial_distribution, dtype=np.float64).reshape(-1)
            if len(probs) != n_phys:
                raise ValueError(
                    f"Electronic initial distribution length mismatch: expected {n_phys}, got {len(probs)}."
                )
            probs = probs / probs.sum()
            nonzero = int(np.count_nonzero(probs > 1e-14))
            if n_anc < nonzero:
                raise ValueError("`n_electronic_ancilla` is too small to purify the requested diagonal state.")
            coeff = np.zeros((n_phys, n_anc), dtype=np.complex128)
            ancilla_slots = iter(range(n_anc))
            for alpha, prob in enumerate(probs):
                if prob <= 1e-14:
                    continue
                coeff[alpha, next(ancilla_slots)] = np.sqrt(prob)
            return coeff

        beta = self._electronic_beta_value()
        h_e = self._electronic_hamiltonian_matrix()
        evals, evecs = np.linalg.eigh(h_e)
        weights = np.exp(-beta * evals.real)
        weights /= weights.sum()
        nonzero = int(np.count_nonzero(weights > 1e-14))
        if n_anc < nonzero:
            raise ValueError("`n_electronic_ancilla` is too small to purify the thermal electronic state.")

        coeff_full = evecs @ np.diag(np.sqrt(weights))
        if n_anc == n_phys:
            return coeff_full
        if n_anc < n_phys:
            return coeff_full[:, :n_anc]
        coeff = np.zeros((n_phys, n_anc), dtype=np.complex128)
        coeff[:, :n_phys] = coeff_full
        return coeff

    def _build_electronic_ancilla_state(self, local_state, weights=None):
        coeff = self._build_electronic_purification_coefficients()
        if weights is None:
            weights = np.ones(self.ms_model.N_electron, dtype=np.complex128)
        weights = np.asarray(weights, dtype=np.complex128).reshape(-1)
        msmps = []
        for alpha in range(self.ms_model.N_electron):
            for ancilla in range(coeff.shape[1]):
                component = local_state.copy()
                component.coeff *= weights[alpha] * coeff[alpha, ancilla]
                component.compress_config = self.icompress_config
                component.evolve_config = self.evolve_config
                msmps.append(component)
        return ElectronicAncillaMultisetMps(
            self.ms_model.MsModel,
            self.ms_model.N_electron,
            temperature=self.temperature,
            init_model=self.ms_model.init_model,
            method=self.thermal_init_method,
            msmps=msmps,
            n_anc=coeff.shape[1],
        )

    def _init_ground_thermal_state(self):
        if self.thermal_init_method == "imaginary_time_exact":
            return self._exact_ground_thermal_mpdm(self.ms_model.init_model)
        elif self.thermal_init_method == "imaginary_time_propagate":
            return self._propagate_ground_thermal_mpdm(self.ms_model.init_model)

    def _exact_ground_thermal_mpdm(self, model):
        '''
        Build the ground state in finite temperature
        Only phonon terms are purification
        '''
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

    def _propagate_ground_thermal_mpdm(self, model):
        '''
        Build the ground matrix product density matrix in finite temperature
        by using singleset imagine time evolution
        Because in absorption spectra, only ground state are purification.
        There is no electron-phonon coupling in the system-bath
        '''
        local_state = self._max_entangled_ground_mpdm(model, set_evolve_config=False)
        tp = ThermalProp(
            local_state,
            h_mpo_model=model,
            evolve_config=EvolveConfig(method=EvolveMethod.tdvp_ps, adaptive=False),
            auto_expand=False,
        )
        tp.evolve(None, self.insteps, self.temperature.to_beta() / 2j)
        thermal_state = tp.latest_mps
        thermal_state.compress_config = self.icompress_config
        thermal_state.evolve_config = self.evolve_config
        return thermal_state
    
    def _propagate_excited_thermal_msmpdm(self, state: MultisetMps):
        '''
        Build the excited multiset matrix product density matrix in finite temperature
        by using multiset imagine time evolution
        Because in emission spectra, electron-phonon coupling is considered
        '''
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

    def _max_entangled_ground_mpdm(self, model, set_evolve_config=True):
        state = MpDm.max_entangled_gs(model)
        state.compress_config = self.icompress_config
        if set_evolve_config:
            state.evolve_config = self.evolve_config
        return state

    def _init_excited_thermal_state(self):
        method = self.thermal_init_method
        if self.electronic_ancilla:
            local_state = self._max_entangled_ground_mpdm(self.ms_model.init_model)
            return self._propagate_excited_thermal_msmpdm(self._build_electronic_ancilla_state(local_state))

        if method == "imaginary_time_exact":
            logger.warning(
                "Finite-temperature multiset emission does not support a physically exact "
                "diagonal-block thermal initialisation. Falling back to "
                "`imaginary_time_propagate` for the excited thermal state."
            )
            msmps = [
                self._max_entangled_ground_mpdm(self.ms_model.MsModel[alpha][alpha])
                for alpha in range(self.ms_model.N_electron)
            ]
            return self._propagate_excited_thermal_msmpdm(self._build_multiset_state(msmps))
        elif self.thermal_init_method == "imaginary_time_propagate":
            msmps = [
                self._max_entangled_ground_mpdm(self.ms_model.MsModel[alpha][alpha])
                for alpha in range(self.ms_model.N_electron)
            ]
            return self._propagate_excited_thermal_msmpdm(self._build_multiset_state(msmps))

    def _build_multiset_state(self, msmps):
        for state in msmps:
            state.compress_config = self.icompress_config
            state.evolve_config = self.evolve_config
        if self.electronic_ancilla:
            return ElectronicAncillaMultisetMps(
                self.ms_model.MsModel,
                self.ms_model.N_electron,
                temperature=self.temperature,
                init_model=self.ms_model.init_model,
                method=self.thermal_init_method,
                msmps=msmps,
                n_anc=self._electronic_ancilla_dim(),
            )
        return MultisetMps(
            self.ms_model.MsModel,
            self.ms_model.N_electron,
            temperature=self.temperature,
            init_model=self.ms_model.init_model,
            method=self.thermal_init_method,
            msmps=msmps,
        )

    def _broadcast_local_state(self, local_state, weights=None):
        if self.electronic_ancilla:
            return self._build_electronic_ancilla_state(local_state, weights)

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

        return self._build_multiset_state(msmps)

    def _init_dipole(self, state):
        dipole = self._get_dipole()
        if not isinstance(state, MultisetMps):
            return self._broadcast_local_state(state, dipole)

        if getattr(state, "electronic_ancilla", False):
            msmps = []
            for alpha in range(state.N_electron):
                for ancilla in range(state.n_anc):
                    component = state.get(alpha, ancilla).copy()
                    component.scale(float(dipole[alpha]), inplace=True)
                    component.compress_config = self.icompress_config
                    component.evolve_config = self.evolve_config
                    msmps.append(component)
            return self._build_multiset_state(msmps)

        msmps = []
        for alpha in range(state.N_electron):
            component = state.msmps[alpha].copy()
            component.scale(float(dipole[alpha]), inplace=True)
            component.compress_config = self.icompress_config
            component.evolve_config = self.evolve_config
            msmps.append(component)
        return self._build_multiset_state(msmps)

    def _expand_initial_multiset_state(self, state: MultisetMps, coef: float = 1e-10) -> MultisetMps:
        self.ms_model.set_mps(state)
        self.ms_model.expand_bond_dimension_multiset(coef=coef, use_hint=True)
        expanded_state = self.ms_model.MsMps
        for component in expanded_state.msmps:
            component.compress_config = self.icompress_config
            component.evolve_config = self.evolve_config
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
        return self._build_multiset_state(msmps)

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
            component_autocorr = _multiset_pair_overlaps(bra, ket)
            if self.spectratype == "emi":
                ft = np.conjugate(complex(component_autocorr.sum()))
                component_autocorr = np.conjugate(component_autocorr)
            else:
                ft = complex(np.trace(component_autocorr))
        else:
            ft = _state_inner_product(bra, ket)
            if self.spectratype == "emi":
                ft = np.conjugate(ft)
            component_autocorr = np.asarray([[ft]])
        self._autocorr.append(ft)
        self._autocorr_components.append(component_autocorr)
        self._bond_dims.append(_state_bond_dims(mps))

    def stop_evolve_criteria(self):
        return False

    @property
    def autocorr(self):
        return np.array(self._autocorr)

    @property
    def autocorr_components(self):
        return np.array(self._autocorr_components)

    @property
    def bond_dims(self):
        return np.array(self._bond_dims, dtype=object)

    def get_dump_dict(self):
        return {
            "temperature": self.temperature.as_au(),
            "time series": self.evolve_times,
            "time_series": self.evolve_times,
            "autocorr": self.autocorr,
            "autocorr_components": self.autocorr_components,
            "bond_dims": self.bond_dims,
        }

    def _get_dipole(self) -> np.ndarray:
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
