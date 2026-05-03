# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime

import numpy as np
from scipy import integrate

from renormalizer.model.basis import BasisSHO
from renormalizer.model.model import Model
from renormalizer.mps import MpDm, Mps, ThermalProp
from renormalizer.mps.mpo import Mpo
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mpo import MultisetBlockMpo
from renormalizer.multiset.multiset_mps import MsEvolveMethod, MultisetMps
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod,
    Quantity,
)
from renormalizer.utils.constant import mobility2au

logger = logging.getLogger(__name__)


def _thermal_coefficients_from_theta(basis: BasisSHO, temperature: Quantity):
    ratio = np.exp(-0.5 * temperature.to_beta() * basis.omega)
    ratio = np.clip(ratio, 0.0, 1.0 - np.finfo(float).eps)
    theta = np.arctanh(ratio)
    weights = np.tanh(theta) ** np.arange(basis.nbas, dtype=float)
    weights /= np.cosh(theta)
    weights /= np.linalg.norm(weights)
    return weights


def _calc_r_square_multiset(e_occupations):
    r_list = np.arange(0, len(e_occupations))
    if np.allclose(e_occupations, np.zeros_like(e_occupations)):
        return 0.0
    r_mean_square = np.average(r_list, weights=e_occupations) ** 2
    mean_r_square = np.average(r_list**2, weights=e_occupations)
    return float(mean_r_square - r_mean_square)


def _state_bond_dims(state):
    if isinstance(state, MultisetMps):
        return [list(mps.bond_dims) for mps in state.msmps]
    if hasattr(state, "bond_dims"):
        return list(state.bond_dims)
    if isinstance(state, (tuple, list)):
        return [_state_bond_dims(item) for item in state]
    return None


class MultisetTdJob(object):
    """
    A lightweight time-evolution driver for multiset calculations.

    This class intentionally mirrors the control flow of ``TdMpsJob`` while
    remaining agnostic to the exact multiset state layout. Concrete jobs can
    store a single ``MultisetMps`` or a tuple/list of states for correlation
    function calculations.
    """

    def __init__(
        self,
        evolve_config: EvolveConfig = None,
        dump_mps: str = None,
        dump_dir: str = None,
        job_name: str = None,
        if_startup_substeps: bool = False,
        startup_substeps_n: int = 10,
    ):
        logger.info(
            "Creating multiset TD job. dump_dir: %s. job_name: %s",
            dump_dir,
            job_name,
        )
        if evolve_config is None:
            self.evolve_config: EvolveConfig = EvolveConfig()
        else:
            self.evolve_config = evolve_config
        logger.info(f"evolve_config: {self.evolve_config}")
        logger.info("Step 0/?. Preparing multiset state in the initial state.")

        self.evolve_times = [0]
        self.info_interval = 1
        if dump_mps in [None, "all", "one"]:
            self.dump_mps = dump_mps
        else:
            raise ValueError(f"dump_mps should be None, 'all', 'one'. Got {dump_mps}")

        self._dump_mps = None
        self.dump_dir = dump_dir
        self.job_name = job_name
        self.if_startup_substeps = if_startup_substeps
        self.startup_substeps_n = startup_substeps_n

        mps = self.init_mps()
        logger.info(f"Initial multiset state: {str(mps)}")
        bond_dims = _state_bond_dims(mps)
        if bond_dims is not None:
            logger.info("Initial multiset bond dimensions: %s", bond_dims)
        if mps is None:
            raise ValueError("init_mps should return a multiset state. Got None")
        self.latest_mps = mps
        self.process_mps(mps)
        logger.info("Multiset TD job created.")

    def init_mps(self):
        raise NotImplementedError

    def process_mps(self, mps):
        raise NotImplementedError

    def evolve_single_step(self, evolve_dt):
        raise NotImplementedError

    def get_dump_dict(self):
        raise NotImplementedError

    def stop_evolve_criteria(self):
        return False

    def _run_startup_substeps(self, evolve_dt):
        abs_dt = abs(evolve_dt)
        if abs_dt == 0:
            raise ValueError("startup substeps require a non-zero evolve_dt")

        phase = evolve_dt / abs_dt
        previous_abs_time = 0.0
        substeps = np.logspace(np.log10(abs_dt * 1e-5), np.log10(abs_dt), self.startup_substeps_n)
        new_mps = self.latest_mps

        for current_abs_time in substeps:
            sub_dt = phase * float(current_abs_time - previous_abs_time)
            new_mps = self.evolve_single_step(sub_dt)
            self.latest_mps = new_mps
            previous_abs_time = float(current_abs_time)

        return new_mps

    def _checkpoint_state_path(self):
        if not self._defined_output_path:
            return None
        if self._dump_mps == "all":
            return os.path.join(self.dump_dir, f"{self.job_name}_mps_{len(self.evolve_times)-1}.npz")
        return os.path.join(self.dump_dir, f"{self.job_name}_mps.npz")

    def _dump_state(self, state, fname):
        root, ext = os.path.splitext(fname)
        if ext == "":
            ext = ".npz"
            fname = root + ext

        if hasattr(state, "dump"):
            state.dump(fname)
            return

        if isinstance(state, (tuple, list)):
            item_paths = []
            for idx, item in enumerate(state):
                item_path = f"{root}_{idx}{ext}"
                self._dump_state(item, item_path)
                item_paths.append(item_path)
            np.savez(
                fname,
                version="0.1",
                kind=type(state).__name__,
                item_paths=np.array(item_paths, dtype=object),
            )
            return

        logger.warning("State type %s does not support checkpoint dump.", type(state).__name__)

    def dump_dict(self):
        if not self._defined_output_path:
            raise ValueError("Dump dir or job name not set")

        d = self.get_dump_dict()
        os.makedirs(self.dump_dir, exist_ok=True)
        file_path = os.path.join(self.dump_dir, self.job_name + ".npz")
        bak_path = file_path + ".bak"

        if os.path.exists(file_path):
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.rename(file_path, bak_path)

        np.savez(file_path, **d)

        if os.path.exists(bak_path):
            os.remove(bak_path)

        if self._dump_mps is not None:
            self._dump_state(self.latest_mps, self._checkpoint_state_path())

    def evolve(self, evolve_dt=None, nsteps=None, evolve_time=None):
        if (evolve_dt is not None) and (nsteps is not None) and (evolve_time is not None):
            logger.warning("Both evolve_time and nsteps are defined for evolution. The evolve_time is omitted")
            case = 1
        elif (evolve_dt is None) and (nsteps is not None) and (evolve_time is not None):
            evolve_dt = evolve_time / float(nsteps)
            logger.info(f"The evolve_dt is {evolve_dt}")
            case = 1
        elif (evolve_dt is not None) and (nsteps is not None) and (evolve_time is None):
            case = 1
        elif (evolve_dt is not None) and (nsteps is None) and (evolve_time is not None):
            nsteps = int(abs(evolve_time) // abs(evolve_dt)) + 1
            case = 1
        elif (evolve_dt is not None) and (nsteps is None) and (evolve_time is None):
            logger.info("evolution will stop by `stop_evolve_criteria`")
            nsteps = int(1e10)
            case = 2
        else:
            raise ValueError(
                f"The input parameters evolve_dt:{evolve_dt}, nsteps:{nsteps}, "
                f"evolve_time:{evolve_time} do not meet the requirements!"
            )

        if case == 1:
            target_steps = len(self.evolve_times) + nsteps - 1
            target_time = self.evolve_times[-1] + nsteps * evolve_dt
        else:
            target_steps = "?"
            target_time = "?"

        wall_times = [datetime.now()]

        if (
            self.if_startup_substeps
            and nsteps > 0
            and len(self.evolve_times) == 1
            and np.isclose(self.latest_evolve_time, 0)
        ):
            logger.info(
                "step %s/%s, at time %s/%s begin with %s startup substeps.",
                len(self.evolve_times),
                target_steps,
                self.latest_evolve_time,
                target_time,
                self.startup_substeps_n,
            )

            new_mps = self._run_startup_substeps(evolve_dt)
            self.evolve_times.append(self.latest_evolve_time + evolve_dt)
            self.process_mps(new_mps)
            self.latest_mps = new_mps

            evolution_wall_time = datetime.now()
            time_cost = evolution_wall_time - wall_times[-1]
            wall_times.append(evolution_wall_time)

            if self.info_interval is not None:
                mps_abstract = str(new_mps)
                bond_dims = _state_bond_dims(new_mps)
                if bond_dims is not None:
                    mps_abstract += f" bond_dims={bond_dims}"
                self._dump_mps = self.dump_mps
            else:
                mps_abstract = ""
                self._dump_mps = None

            logger.info(
                "step %s complete, time cost %s. %s",
                len(self.evolve_times) - 1,
                time_cost,
                mps_abstract,
            )

            if self._defined_output_path:
                try:
                    self.dump_dict()
                except IOError:
                    logger.exception("dumping dict failed with IOError")
                dump_wall_time = datetime.now()
                logger.info(f"Dumping time cost {dump_wall_time - evolution_wall_time}")

            if self.stop_evolve_criteria():
                logger.info("Criteria to stop the evolution has met. Stop the evolution")
                logger.info(f"{len(wall_times)-1} steps of evolution complete!")
                logger.info("Normal termination. Time cost: %s", wall_times[-1] - wall_times[0])
                return self

            nsteps -= 1

        for i in range(nsteps):
            if self.stop_evolve_criteria():
                logger.info("Criteria to stop the evolution has met. Stop the evolution")
                break

            step_str = "step {}/{}, at time {}/{}".format(
                len(self.evolve_times), target_steps, self.latest_evolve_time, target_time
            )
            logger.info("%s begin.", step_str)

            new_mps = self.evolve_single_step(evolve_dt)

            self.evolve_times.append(self.latest_evolve_time + evolve_dt)
            self.process_mps(new_mps)
            self.latest_mps = new_mps

            evolution_wall_time = datetime.now()
            time_cost = evolution_wall_time - wall_times[-1]
            wall_times.append(evolution_wall_time)

            if self.info_interval is not None and i % self.info_interval == 0:
                mps_abstract = str(new_mps)
                bond_dims = _state_bond_dims(new_mps)
                if bond_dims is not None:
                    mps_abstract += f" bond_dims={bond_dims}"
                self._dump_mps = self.dump_mps
            else:
                mps_abstract = ""
                self._dump_mps = None

            logger.info(
                "step %s complete, time cost %s. %s",
                len(self.evolve_times) - 1,
                time_cost,
                mps_abstract,
            )

            if self._defined_output_path:
                try:
                    self.dump_dict()
                except IOError:
                    logger.exception("dumping dict failed with IOError")
                dump_wall_time = datetime.now()
                logger.info(f"Dumping time cost {dump_wall_time - evolution_wall_time}")

        logger.info(f"{len(wall_times)-1} steps of evolution complete!")
        logger.info("Normal termination. Time cost: %s", wall_times[-1] - wall_times[0])
        return self

    @property
    def latest_evolve_time(self):
        return self.evolve_times[-1]

    @property
    def evolve_times_array(self):
        return np.array(self.evolve_times)

    @property
    def _defined_output_path(self):
        return self.dump_dir is not None and self.job_name is not None


class MultisetChargeDiffusionDynamics(MultisetTdJob):
    """
    Multiset counterpart of charge-diffusion dynamics using ``MultisetModel``
    as the evolution engine and ``MultisetTdJob`` as the orchestration layer.
    """

    def __init__(
        self,
        model: Model = None,
        max_bonddim=None,
        temperature: Quantity = Quantity(0, "K"),
        method: str = "thermo_field",
        evolve_config: EvolveConfig = None,
        compress_config: CompressConfig = None,
        ms_model: MultisetModel = None,
        initial_site: int = None,
        stop_at_edge: bool = True,
        edge_threshold: float = 1e-4,
        use_init_hint: bool = True,
        dump_mps: str = None,
        dump_dir: str = None,
        job_name: str = None,
        if_startup_substeps: bool = False,
        startup_substeps_n: int = 10,
        observables: dict = None,
    ):
        if ms_model is None:
            if model is None or max_bonddim is None:
                raise ValueError("Either provide `ms_model` or both `model` and `max_bonddim`.")
            ms_model = MultisetModel(
                model,
                max_bonddim=max_bonddim,
                temperature=temperature,
                method=method,
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
        self.temperature = self.ms_model.temperature
        self.initial_site = self.ms_model.N_electron // 2 if initial_site is None else initial_site
        self.stop_at_edge = stop_at_edge
        self.edge_threshold = edge_threshold
        self.use_init_hint = use_init_hint
        self.observables = {
            "energy": False,
            "r_square": False,
            "e_occupations": True,
            "ph_occupations": False,
            "rho": False,
            "coherent_length": False,
            "trace": False,
            "purity": False,
        }
        if observables is not None:
            self.observables.update(observables)

        self.energies = []
        self.r_square_array = []
        self.e_occupations_array = []
        self.ph_occupations_array = []
        self.reduced_density_matrices = []
        self.coherent_length_array = []
        self.purity_array = []
        self.trace_array = []

        super().__init__(
            evolve_config=self.ms_model.evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
            if_startup_substeps=if_startup_substeps,
            startup_substeps_n=startup_substeps_n,
        )

    def init_mp(self, method=None):
        method = self.ms_model.method if method is None else method
        if self.temperature == 0:
            return Mps.hartree_product_state(model=self.ms_model.init_model)

        logger.info(f"Initialising multiset finite-temperature state with {method}")
        if method in ["imaginary_time_exact"]:
            logger.info("Purification method: imaginary_time_exact")
            beta = self.temperature.to_beta()
            condition = {}
            for basis in self.ms_model.init_model.basis:
                if not isinstance(basis, BasisSHO):
                    continue
                weights = np.exp(-0.5 * beta * basis.omega * np.arange(basis.nbas, dtype=float))
                weights /= np.linalg.norm(weights)
                condition[basis.dof] = weights
            thermal_mps = Mps.hartree_product_state(model=self.ms_model.init_model, condition=condition)
            return MpDm.from_mps(thermal_mps)

        if method in ["thermo_field"]:
            logger.info("Purification method: thermo_field_dynamics")
            condition = {}
            for basis in self.ms_model.init_model.basis:
                if not isinstance(basis, BasisSHO):
                    continue
                condition[basis.dof] = _thermal_coefficients_from_theta(basis, self.temperature)
            thermal_mps = Mps.hartree_product_state(model=self.ms_model.init_model, condition=condition)
            return MpDm.from_mps(thermal_mps)

        if method in ["imaginary_time_propagate"]:
            logger.info("Purification method: imaginary_time_propagate")
            local_state = MpDm.max_entangled_gs(self.ms_model.init_model)
            icompress_config = CompressConfig(
                CompressCriteria.fixed,
                max_bonddim=1,
            )
            local_state.compress_config = icompress_config
            tp = ThermalProp(
                local_state,
                h_mpo_model=self.ms_model.init_model,
                evolve_config=EvolveConfig(method=EvolveMethod.tdvp_ps),
                auto_expand=False,
            )
            nsteps = max(20, len(local_state))
            tp.evolve(None, nsteps, self.temperature.to_beta() / 2j)
            thermal_state = tp.latest_mps
            logger.info("[thermal init] imaginary-time tdvp bond dims: %s", thermal_state.bond_dims)
            logger.info("[thermal init] ph occupations: %s", thermal_state.ph_occupations)
            return thermal_state

        raise ValueError(f"Unsupported finite-temperature method: {method}")

    def _init_msmps(self, local_state):
        if isinstance(local_state, MultisetMps):
            return local_state
        return MultisetMps(
            self.ms_model.MsModel,
            self.ms_model.N_electron,
            temperature=self.temperature,
            init_model=self.ms_model.init_model,
            method=self.ms_model.method,
            init_mp=local_state,
        )

    def _fc_excitation(self, state: MultisetMps, alpha: int):
        for beta in range(state.N_electron):
            if beta != alpha:
                state.msmps[beta].scale(1e-10, inplace=True)
        state.ms_normalize("mps_only")

    def _set_hamiltonian_offset(self, energy):
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

    def init_mps(self):
        state = self._init_msmps(self.init_mp())
        self._fc_excitation(state, self.initial_site)

        logger.debug(
            f"[init] mp_norms after fc_excitation: "
            f"{[state.msmps[a].mp_norm for a in range(self.ms_model.N_electron)]}"
        )

        self.ms_model.set_mps(state)
        energy = Quantity(self.ms_model.Hamiltonian())
        logger.debug(f"[init] E0 = {energy.as_au():.6f} a.u.")
        self._set_hamiltonian_offset(energy)

        self.ms_model.expand_bond_dimension_multiset(coef=1e-10, use_hint=self.use_init_hint)
        self.ms_model.MsMps.ms_normalize("mps_only")

        logger.debug(
            f"[init] mp_norms after expand+normalize: "
            f"{[self.ms_model.MsMps.msmps[a].mp_norm for a in range(self.ms_model.N_electron)]}"
        )
        return self.ms_model.MsMps

    def process_mps(self, mps):
        self.ms_model.set_mps(mps)
        e_occupations = mps.e_occupations_multiset
        rho = None

        self.e_occupations_array.append(e_occupations)
        if self.observables["energy"]:
            self.energies.append(self.ms_model.Hamiltonian())
        if self.observables["r_square"]:
            self.r_square_array.append(_calc_r_square_multiset(e_occupations))
        if self.observables["ph_occupations"]:
            self.ph_occupations_array.append(mps.ph_occupations_multiset)
        if any(self.observables[key] for key in ("rho", "coherent_length", "trace", "purity")):
            rho = mps.rho_el()
        if self.observables["rho"]:
            self.reduced_density_matrices.append(rho)
        if self.observables["coherent_length"]:
            self.coherent_length_array.append(np.abs(rho).sum() - np.trace(rho).real)
        if self.observables["trace"]:
            self.trace_array.append(np.trace(rho).real)
        if self.observables["purity"]:
            self.purity_array.append(np.trace(rho @ rho).real)

        logger.info(f"e occupations: {self.e_occupations_array[-1]}")
        if self.observables["ph_occupations"]:
            logger.info(f"ph occupations: {self.ph_occupations_array[-1]}")

    def evolve_single_step(self, evolve_dt):
        new_mps = self.ms_model.evolve_state(self.latest_mps, evolve_dt)
        self.ms_model.set_mps(new_mps)
        return new_mps

    def stop_evolve_criteria(self):
        return (
            self.stop_at_edge
            and len(self.e_occupations_array) > 0
            and self.edge_threshold < self.e_occupations_array[-1][0]
        )

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["mol list"] = self.ms_model.model.to_dict()
        dump_dict["temperature"] = self.temperature.as_au()
        dump_dict["time series"] = list(self.evolve_times)
        dump_dict["electron occupations array"] = self.e_occupations_array
        if self.observables["energy"]:
            dump_dict["energy array"] = self.energies
        if self.observables["r_square"]:
            dump_dict["r square array"] = self.r_square_array
        if self.observables["ph_occupations"]:
            dump_dict["phonon occupations array"] = self.ph_occupations_array
        if self.observables["rho"]:
            dump_dict["reduced density matrices"] = self.reduced_density_matrices
        if self.observables["coherent_length"]:
            dump_dict["coherent length array"] = self.coherent_length_array
        if self.observables["trace"]:
            dump_dict["rho trace array"] = self.trace_array
        if self.observables["purity"]:
            dump_dict["purity array"] = self.purity_array
        return dump_dict
