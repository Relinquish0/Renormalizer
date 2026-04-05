# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime

import numpy as np
from scipy import integrate

from renormalizer.model.model import Model
from renormalizer.mps import MpDm
from renormalizer.mps.mpo import Mpo
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mpo import MultisetBlockMpo
from renormalizer.multiset.multiset_mps import MsEvolveMethod, MultisetMps
from renormalizer.utils import CompressConfig, EvolveConfig, Quantity
from renormalizer.utils.constant import mobility2au

logger = logging.getLogger(__name__)


def _calc_r_square_multiset(e_occupations):
    r_list = np.arange(0, len(e_occupations))
    if np.allclose(e_occupations, np.zeros_like(e_occupations)):
        return 0.0
    r_mean_square = np.average(r_list, weights=e_occupations) ** 2
    mean_r_square = np.average(r_list**2, weights=e_occupations)
    return float(mean_r_square - r_mean_square)


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

        mps = self.init_mps()
        logger.info(f"Initial multiset state: {str(mps)}")
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

        self.energies = []
        self.r_square_array = []
        self.e_occupations_array = []
        self.reduced_density_matrices = []
        self.coherent_length_array = []
        self.purity_array = []
        self.trace_array = []

        super().__init__(
            evolve_config=self.ms_model.evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
        )

    def init_mps(self):
        self.ms_model.reset_mps()
        self.ms_model.cdd_init_mps(
            initial_site=self.initial_site,
            use_hint=self.use_init_hint,
        )
        return self.ms_model.MsMps

    def process_mps(self, mps):
        self.ms_model.set_mps(mps)

        energy = self.ms_model.Hamiltonian()
        self.energies.append(energy)

        rho = self.ms_model.rho_el()
        e_occupations = np.diag(rho).real
        self.e_occupations_array.append(e_occupations)
        self.r_square_array.append(_calc_r_square_multiset(e_occupations))
        self.reduced_density_matrices.append(rho)
        self.coherent_length_array.append(np.abs(rho).sum() - np.trace(rho).real)
        self.trace_array.append(np.trace(rho).real)
        self.purity_array.append(np.trace(rho @ rho).real)

        logger.info(f"e occupations: {self.e_occupations_array[-1]}")

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
        dump_dict["energy array"] = self.energies
        dump_dict["r square array"] = self.r_square_array
        dump_dict["electron occupations array"] = self.e_occupations_array
        dump_dict["reduced density matrices"] = self.reduced_density_matrices
        dump_dict["coherent length array"] = self.coherent_length_array
        dump_dict["rho trace array"] = self.trace_array
        dump_dict["purity array"] = self.purity_array
        return dump_dict


class MultisetTransportKubo(MultisetTdJob):
    r"""
    Multiset counterpart of ``renormalizer.transport.kubo.TransportKubo``.

    The control flow mirrors the singleset implementation:
    1. Prepare a finite-temperature multiset state representing ``e^{-beta H / 2}``
    2. Apply the multiset current operator to obtain ``J e^{-beta H / 2}``
    3. Real-time propagate bra and ket states with the multiset TDVP-PS kernel
    4. Evaluate the current-current correlation function and mobility
    """

    def __init__(
        self,
        model: Model = None,
        max_bonddim=None,
        temperature: Quantity = Quantity(298, "K"),
        distance_matrix: np.ndarray = None,
        insteps: int = 1,
        thermal_init_method: str = "thermofield",
        ievolve_config: EvolveConfig = None,
        compress_config: CompressConfig = None,
        evolve_config: EvolveConfig = None,
        ms_model: MultisetModel = None,
        dump_mps: str = None,
        dump_dir: str = None,
        job_name: str = None,
        thermal_dump_path: str = None,
        properties=None,
    ):
        if temperature == 0:
            raise ValueError("Can't set temperature to 0.")
        if properties is not None:
            raise NotImplementedError("MultisetTransportKubo does not support `properties` yet.")

        if ms_model is None:
            if model is None or max_bonddim is None:
                raise ValueError("Either provide `ms_model` or both `model` and `max_bonddim`.")
            ms_model = MultisetModel(
                model,
                max_bonddim=max_bonddim,
                temperature=temperature,
                method="thermo_field",
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
        self.temperature = temperature
        self.distance_matrix = distance_matrix
        self.thermal_init_method = thermal_init_method
        self._auto_corr = []
        self._auto_corr_decomposition = []

        if ievolve_config is None:
            self.ievolve_config = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps)
            if insteps is None:
                self.ievolve_config.adaptive = True
                self.ievolve_config.guess_dt = temperature.to_beta() / 1e5j
                insteps = 1
        else:
            self.ievolve_config = ievolve_config
        self.insteps = insteps

        if compress_config is None:
            self.compress_config = self.ms_model.compress_config
        else:
            self.compress_config = compress_config

        if thermal_dump_path is not None:
            self.thermal_dump_path = thermal_dump_path
        elif dump_dir is not None and job_name is not None:
            self.thermal_dump_path = os.path.join(dump_dir, job_name + "_impdm.npz")
        else:
            self.thermal_dump_path = None

        self._ms_construct_current_operator()

        super().__init__(
            evolve_config=self.ms_model.evolve_config,
            dump_mps=dump_mps,
            dump_dir=dump_dir,
            job_name=job_name,
        )

    def _ms_construct_current_operator(self):
        logger.info("constructing multiset current operator")

        mol_num = self.model.n_edofs
        if self.distance_matrix is None:
            logger.info("Constructing distance matrix based on a periodic one-dimension chain.")
            self.distance_matrix = np.arange(mol_num).reshape(-1, 1) - np.arange(mol_num).reshape(1, -1)
            self.distance_matrix[0][-1] = 1
            self.distance_matrix[-1][0] = -1

        holstein_terms = [[[] for _ in range(self.ms_model.N_electron)] for _ in range(self.ms_model.N_electron)]
        peierls_terms = [[[] for _ in range(self.ms_model.N_electron)] for _ in range(self.ms_model.N_electron)]

        for ham_op in self.model.ham_terms:
            electron_ops = []
            for dof_idx, dof_name in enumerate(ham_op.dofs):
                if dof_name in self.model.e_dofs:
                    electron_ops.append((dof_idx, dof_name, self.model.e_dofs.index(dof_name)))

            if len(electron_ops) == 0:
                continue
            if len(electron_ops) != 2:
                raise ValueError(f"Unsupported current operator term in multiset Kubo: {ham_op}")

            (dof_op_idx1, dof_name1, e_idx1), (dof_op_idx2, dof_name2, e_idx2) = electron_ops
            if e_idx1 == e_idx2:
                continue

            if len(ham_op.dofs) not in (2, 3):
                raise NotImplementedError("Complex vibration potential not implemented")

            if len(ham_op.dofs) == 3:
                phonon_dof_idx = ({0, 1, 2} - {dof_op_idx1, dof_op_idx2}).pop()
                if ham_op.split_symbol[phonon_dof_idx] not in (r"b^\dagger+b", r"b^\dagger + b", "x"):
                    raise NotImplementedError("Only linear phonon-assisted current is supported")

            symbol1 = ham_op.split_symbol[dof_op_idx1]
            symbol2 = ham_op.split_symbol[dof_op_idx2]
            if {symbol1, symbol2} != {r"a^\dagger", "a"}:
                raise ValueError(f"Unknown symbol: {symbol1}, {symbol2}")

            if symbol1 == r"a^\dagger":
                alpha, beta = dof_name1, dof_name2
                factor = self.distance_matrix[e_idx1][e_idx2]
            else:
                alpha, beta = dof_name2, dof_name1
                factor = self.distance_matrix[e_idx2][e_idx1]

            if np.isclose(factor, 0):
                continue

            current_op = self.ms_model._reset_all_MsOp(ham_op * factor)
            if len(ham_op.dofs) == 2:
                holstein_terms[alpha][beta].append(current_op)
            else:
                peierls_terms[alpha][beta].append(current_op)

        self.j_oper = MultisetBlockMpo(
            self.ms_model.basis_set,
            holstein_terms,
            self.ms_model.N_electron,
            compress_config=self.compress_config,
        )
        logger.info("multiset current operator active blocks: %s", len(self.j_oper._active_pairs))

        j_oper2 = MultisetBlockMpo(
            self.ms_model.basis_set,
            peierls_terms,
            self.ms_model.N_electron,
            compress_config=self.compress_config,
        )
        self.j_oper2 = j_oper2 if j_oper2.has_terms else None

    def _normalize_thermal_init_method(self):
        method = self.thermal_init_method.lower().replace("-", "_").replace(" ", "_")
        if method in ["thermofield", "thermo_field"]:
            return "thermo_field"
        if method == "imaginary_time":
            return "imaginary_time"
        raise ValueError(f"Unsupported thermal_init_method: {self.thermal_init_method}")

    def _broadcast_local_state(self, local_state):
        ms_state = MultisetMps.__new__(MultisetMps)
        ms_state.MsModel = self.ms_model.MsModel
        ms_state.N_electron = self.ms_model.N_electron
        ms_state.temperature = self.temperature
        ms_state.init_model = self.ms_model.init_model
        ms_state.method = self._normalize_thermal_init_method()
        ms_state.msmps = [local_state.copy() for _ in range(self.ms_model.N_electron)]
        return ms_state

    def _load_thermal_state(self):
        if self.thermal_dump_path is None:
            return None
        try:
            logger.info("Try load multiset thermal state from %s", self.thermal_dump_path)
            return MultisetMps.load(
                self.ms_model.MsModel,
                self.ms_model.N_electron,
                self.thermal_dump_path,
                init_model=self.ms_model.init_model,
            )
        except FileNotFoundError:
            logger.info("No multiset thermal state found at %s", self.thermal_dump_path)
            return None

    def _build_thermal_seed_state(self):
        method = self._normalize_thermal_init_method()
        if method == "thermo_field":
            state = MultisetMps(
                self.ms_model.MsModel,
                self.ms_model.N_electron,
                temperature=self.temperature,
                init_model=self.ms_model.init_model,
                method="thermo_field",
            )
            state.ms_normalize("mps_only")
            return state

        local_state = MpDm.max_entangled_gs(self.ms_model.init_model)
        local_state.compress_config = self.compress_config
        state = self._broadcast_local_state(local_state)
        state.ms_normalize("mps_only")
        return state

    def _imaginary_time_propagate(self, mpdm: MultisetMps):
        if self.insteps is None:
            raise ValueError("`insteps` must be defined for imaginary-time initialisation.")
        evolve_dt = self.temperature.to_beta() / (2j * self.insteps)
        original_evolve_config = self.ms_model.evolve_config
        self.ms_model.evolve_config = self.ievolve_config
        try:
            state = mpdm
            for _ in range(self.insteps):
                state = self.ms_model.evolve_state(state, evolve_dt, normalize=True)
            return state
        finally:
            self.ms_model.evolve_config = original_evolve_config

    def _set_hamiltonian_offset(self, energy):
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
        mpdm = self._load_thermal_state()
        if mpdm is None:
            mpdm = self._build_thermal_seed_state()
            if self._normalize_thermal_init_method() == "imaginary_time":
                mpdm = self._imaginary_time_propagate(mpdm)
            if self.thermal_dump_path is not None:
                mpdm.dump(self.thermal_dump_path)

        self.ms_model.set_mps(mpdm)
        energy = self.ms_model.Hamiltonian()
        self._set_hamiltonian_offset(energy)

        bra_mpdm = mpdm.copy()
        ket_mpdm = self.j_oper.apply(mpdm)
        if self.j_oper2 is None:
            return bra_mpdm, ket_mpdm

        ket_mpdm2 = self.j_oper2.apply(mpdm)
        return bra_mpdm, ket_mpdm, ket_mpdm2

    def process_mps(self, mps):
        if self.j_oper2 is None:
            bra_mpdm, ket_mpdm = mps
            self._auto_corr.append(-self.j_oper.matrix_element(bra_mpdm, ket_mpdm))
            return

        bra_mpdm, ket_mpdm, ket_mpdm2 = mps
        ft1 = -self.j_oper.matrix_element(bra_mpdm, ket_mpdm)
        ft2 = -self.j_oper.matrix_element(bra_mpdm, ket_mpdm2)
        ft3 = -self.j_oper2.matrix_element(bra_mpdm, ket_mpdm)
        ft4 = -self.j_oper2.matrix_element(bra_mpdm, ket_mpdm2)
        self._auto_corr.append(ft1 + ft2 + ft3 + ft4)
        self._auto_corr_decomposition.append([ft1, ft2, ft3, ft4])

    def evolve_single_step(self, evolve_dt):
        if self.j_oper2 is None:
            prev_bra_mpdm, prev_ket_mpdm = self.latest_mps
            prev_ket_mpdm2 = None
        else:
            prev_bra_mpdm, prev_ket_mpdm, prev_ket_mpdm2 = self.latest_mps

        latest_bra_mpdm = self.ms_model.evolve_state(prev_bra_mpdm, evolve_dt, normalize=False)
        latest_ket_mpdm = self.ms_model.evolve_state(prev_ket_mpdm, evolve_dt, normalize=False)
        if self.j_oper2 is None:
            return latest_bra_mpdm, latest_ket_mpdm

        latest_ket_mpdm2 = self.ms_model.evolve_state(prev_ket_mpdm2, evolve_dt, normalize=False)
        return latest_bra_mpdm, latest_ket_mpdm, latest_ket_mpdm2

    def stop_evolve_criteria(self):
        corr = self.auto_corr
        if len(corr) < 10:
            return False
        last_corr = corr[-10:]
        first_corr = corr[0]
        return (
            np.abs(last_corr.mean()) < 1e-5 * np.abs(first_corr)
            and last_corr.std() < 1e-5 * np.abs(first_corr)
        )

    @property
    def auto_corr(self) -> np.ndarray:
        return np.array(self._auto_corr)

    @property
    def auto_corr_decomposition(self) -> np.ndarray:
        return np.array(self._auto_corr_decomposition)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict["mol list"] = self.ms_model.model.to_dict()
        dump_dict["temperature"] = self.temperature.as_au()
        dump_dict["time series"] = self.evolve_times
        dump_dict["auto correlation"] = self.auto_corr
        dump_dict["auto correlation decomposition"] = self.auto_corr_decomposition
        dump_dict["mobility"] = self.calc_mobility()[1]
        return dump_dict

    def calc_mobility(self):
        time_series = self.evolve_times
        corr_real = self.auto_corr.real
        inte = integrate.trapz(corr_real, time_series)
        mobility_in_au = inte / self.temperature.as_au()
        mobility = mobility_in_au / mobility2au
        return mobility_in_au, mobility
