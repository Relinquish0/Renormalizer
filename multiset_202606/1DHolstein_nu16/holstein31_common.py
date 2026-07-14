# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

from renormalizer.model import HolsteinModel, Mol, Phonon
from renormalizer.mps.backend import USE_GPU
from renormalizer.multiset import MultisetChargeDiffusionDynamics
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod,
    Quantity,
)
from renormalizer.utils.log import package_logger as logger


N = int(os.environ.get("N", "31"))
OMEGA_0 = float(os.environ.get("OMEGA_0", "1.0"))
J = float(os.environ.get("J", "1.0"))
G = float(os.environ.get("G", "1.0"))
NU_MAX = int(os.environ.get("NU_MAX", "16"))
MODEL_SCHEME = int(os.environ.get("MODEL_SCHEME", "4"))

MAX_BONDDIM = int(os.environ.get("MAX_BONDDIM", "16"))
EVOLVE_DT = float(os.environ.get("EVOLVE_DT", "0.1"))
N_SNAPSHOTS = int(os.environ.get("N_SNAPSHOTS", "500"))
IF_STARTUP_SUBSTEPS = os.environ.get("IF_STARTUP_SUBSTEPS", "1").lower() not in {
    "0",
    "false",
    "no",
}
STARTUP_SUBSTEPS_N = int(os.environ.get("STARTUP_SUBSTEPS_N", "10"))
LONG_RANGE_PERIODIC = os.environ.get("LONG_RANGE_PERIODIC", "0").lower() in {
    "1",
    "true",
    "yes",
}


def build_long_range_j_matrix(n, j, alpha, periodic=False):
    j_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            distance = abs(i - k)
            if periodic:
                distance = min(distance, n - distance)
            j_matrix[i, k] = j / distance**alpha
    return j_matrix


def build_model(alpha, scheme):
    phonon_dim = NU_MAX
    lam = G**2 * OMEGA_0
    displacement = np.sqrt(2.0 * lam) / OMEGA_0

    ph = Phonon.simple_phonon(
        Quantity(OMEGA_0),
        Quantity(displacement),
        phonon_dim,
    )
    mol = Mol(Quantity(0), [ph])
    j_matrix = build_long_range_j_matrix(N, J, alpha, periodic=LONG_RANGE_PERIODIC)
    return HolsteinModel([mol] * N, j_matrix, scheme=scheme)


def _output_paths(alpha, label):
    alpha_tag = f"alpha{alpha:g}"
    output_xlsx = os.environ.get(
        "OUTPUT_XLSX",
        f"Holstein31_{alpha_tag}_{MAX_BONDDIM}bd_{label}.xlsx",
    )
    output_npz = os.environ.get(
        "OUTPUT_NPZ",
        f"Holstein31_{alpha_tag}_{MAX_BONDDIM}bd_{label}.npz",
    )
    return output_xlsx, output_npz


def _log_common(alpha, label, output_xlsx, output_npz, scheme):
    logger.info("GPU enabled: %s", USE_GPU)
    logger.info("job type: %s", label)
    logger.info("sites: %d", N)
    logger.info(
        "parameters: omega0=%s, J=%s, g=%s, nu_max=%d, alpha=%s",
        OMEGA_0,
        J,
        G,
        NU_MAX,
        alpha,
    )
    logger.info("long-range periodic distance: %s", LONG_RANGE_PERIODIC)
    logger.info("model scheme: %d", scheme)
    logger.info("phonon local dimension: %d", NU_MAX)
    logger.info("initial FC site: %d", N // 2)
    logger.info("maximum bond dimension: %d", MAX_BONDDIM)
    logger.info("evolve time step: %s", EVOLVE_DT)
    logger.info("number of stored snapshots: %d", N_SNAPSHOTS)
    logger.info(
        "startup substeps enabled: %s, count: %d",
        IF_STARTUP_SUBSTEPS,
        STARTUP_SUBSTEPS_N,
    )
    logger.info("output xlsx: %s", output_xlsx)
    logger.info("output npz: %s", output_npz)


def run_multiset(alpha):
    if N_SNAPSHOTS < 1:
        raise ValueError("N_SNAPSHOTS must be at least 1.")

    output_xlsx, output_npz = _output_paths(alpha, "multiset")
    model = build_model(alpha, scheme=2)
    dynamics_job = MultisetChargeDiffusionDynamics(
        model=model,
        max_bonddim=MAX_BONDDIM,
        initial_site=N // 2,
        stop_at_edge=False,
        if_startup_substeps=IF_STARTUP_SUBSTEPS,
        startup_substeps_n=STARTUP_SUBSTEPS_N,
        if_rdm=True,
    )

    _log_common(alpha, "multiset", output_xlsx, output_npz, scheme=2)
    logger.info("0th population: %s", dynamics_job.e_occupations_array[0])
    dynamics_job.evolve(evolve_dt=EVOLVE_DT, nsteps=N_SNAPSHOTS - 1)

    populations = np.array(dynamics_job.e_occupations_array)
    pd.DataFrame(populations).to_excel(output_xlsx, index=False, header=False)
    np.savez(
        output_npz,
        time_series=np.array(dynamics_job.evolve_times),
        e_occupations=populations,
        S_maxbond=np.array(dynamics_job.S_maxbond_array),
        rdm_el=np.array(dynamics_job.rdm_el_array),
        S_el=np.array(dynamics_job.S_el_array),
    )


def run_singleset(alpha):
    if N_SNAPSHOTS < 1:
        raise ValueError("N_SNAPSHOTS must be at least 1.")

    output_xlsx, output_npz = _output_paths(alpha, "singleset")
    model = build_model(alpha, scheme=MODEL_SCHEME)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=EVOLVE_DT)
    evolve_config.if_startup_substeps = IF_STARTUP_SUBSTEPS
    evolve_config.startup_substeps_n = STARTUP_SUBSTEPS_N
    compress_config = CompressConfig(
        CompressCriteria.fixed,
        max_bonddim=MAX_BONDDIM,
    )
    dynamics_job = ChargeDiffusionDynamics(
        model,
        evolve_config=evolve_config,
        compress_config=compress_config,
        init_electron=InitElectron.fc,
        stop_at_edge=False,
        rdm=True,
    )

    _log_common(alpha, "singleset", output_xlsx, output_npz, scheme=MODEL_SCHEME)
    logger.info("0th population: %s", dynamics_job.e_occupations_array[0])
    dynamics_job.evolve(evolve_dt=EVOLVE_DT, nsteps=N_SNAPSHOTS - 1)

    populations = np.array(dynamics_job.e_occupations_array)
    bond_entropy = np.array(dynamics_job.bond_vn_entropy_array)
    pd.DataFrame(populations).to_excel(output_xlsx, index=False, header=False)
    np.savez(
        output_npz,
        time_series=np.array(dynamics_job.evolve_times),
        e_occupations=populations,
        r_square=np.array(dynamics_job.r_square_array),
        energies=np.array(dynamics_job.energies),
        ph_occupations=np.array(dynamics_job.ph_occupations_array),
        reduced_density_matrices=np.array(dynamics_job.reduced_density_matrices),
        k_occupations=np.array(dynamics_job.k_occupations_array),
        eph_vn_entropy=np.array(dynamics_job.eph_vn_entropy_array),
        bond_vn_entropy=bond_entropy,
        S_maxbond=np.max(bond_entropy, axis=1),
        S_el=np.array(dynamics_job.eph_vn_entropy_array),
        coherent_length=np.array(dynamics_job.coherent_length_array),
    )
