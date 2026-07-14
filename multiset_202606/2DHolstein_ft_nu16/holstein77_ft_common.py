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


NROW = int(os.environ.get("NROW", "7"))
NCOL = int(os.environ.get("NCOL", "7"))
OMEGA_0 = float(os.environ.get("OMEGA_0", "1.0"))
J = float(os.environ.get("J", "1.0"))
G = float(os.environ.get("G", "1.0"))
NU_MAX = int(os.environ.get("NU_MAX", "16"))
MODEL_SCHEME = int(os.environ.get("MODEL_SCHEME", "4"))
PERIODIC = os.environ.get("PERIODIC", "0").lower() in {"1", "true", "yes"}

MAX_BONDDIM = int(os.environ.get("MAX_BONDDIM", "16"))
EVOLVE_DT = float(os.environ.get("EVOLVE_DT", "0.1"))
N_SNAPSHOTS = int(os.environ.get("N_SNAPSHOTS", "251"))
INITIAL_SITE = os.environ.get("INITIAL_SITE")
IF_STARTUP_SUBSTEPS = os.environ.get("IF_STARTUP_SUBSTEPS", "1").lower() not in {
    "0",
    "false",
    "no",
}
STARTUP_SUBSTEPS_N = int(os.environ.get("STARTUP_SUBSTEPS_N", "10"))


def site_index(ix, iy, ncol):
    return ix * ncol + iy


def center_site():
    return site_index(NROW // 2, NCOL // 2, NCOL)


def build_2d_j_matrix(nrow, ncol, j, periodic=False):
    j_matrix = np.zeros((nrow * ncol, nrow * ncol), dtype=float)
    for ix in range(nrow):
        for iy in range(ncol):
            current = site_index(ix, iy, ncol)

            if periodic or ix + 1 < nrow:
                neighbor = site_index((ix + 1) % nrow, iy, ncol)
                if neighbor != current:
                    j_matrix[current, neighbor] = j
                    j_matrix[neighbor, current] = j

            if periodic or iy + 1 < ncol:
                neighbor = site_index(ix, (iy + 1) % ncol, ncol)
                if neighbor != current:
                    j_matrix[current, neighbor] = j
                    j_matrix[neighbor, current] = j

    return j_matrix


def build_model(scheme):
    lam = G**2 * OMEGA_0
    displacement = np.sqrt(2.0 * lam) / OMEGA_0

    ph = Phonon.simple_phonon(
        Quantity(OMEGA_0),
        Quantity(displacement),
        NU_MAX,
    )
    mol = Mol(Quantity(0), [ph])
    j_matrix = build_2d_j_matrix(NROW, NCOL, J, periodic=PERIODIC)
    return HolsteinModel([mol] * (NROW * NCOL), j_matrix, scheme=scheme)


def temperature_quantity(reduced_temperature):
    return Quantity(reduced_temperature * OMEGA_0)


def thermal_phonon_occupation(reduced_temperature):
    if reduced_temperature <= 0:
        return 0.0
    return float(1.0 / np.expm1(1.0 / reduced_temperature))


def _temperature_tag(reduced_temperature):
    return f"T{reduced_temperature:g}"


def _output_paths(reduced_temperature, label):
    temp_tag = _temperature_tag(reduced_temperature)
    output_xlsx = os.environ.get(
        "OUTPUT_XLSX",
        f"Holstein77_{temp_tag}_{MAX_BONDDIM}bd_{label}.xlsx",
    )
    output_npz = os.environ.get(
        "OUTPUT_NPZ",
        f"Holstein77_{temp_tag}_{MAX_BONDDIM}bd_{label}.npz",
    )
    return output_xlsx, output_npz


def _as_array(values):
    try:
        return np.asarray(values)
    except ValueError:
        return np.asarray(values, dtype=object)


def _initial_site():
    return center_site() if INITIAL_SITE is None else int(INITIAL_SITE)


def _log_common(reduced_temperature, label, output_xlsx, output_npz, scheme):
    temp = temperature_quantity(reduced_temperature)
    logger.info("GPU enabled: %s", USE_GPU)
    logger.info("job type: %s", label)
    logger.info("lattice: %d x %d (%d sites)", NROW, NCOL, NROW * NCOL)
    logger.info(
        "parameters: omega0=%s, J=%s, g=%s, nu_max=%d, T*=%s",
        OMEGA_0,
        J,
        G,
        NU_MAX,
        reduced_temperature,
    )
    logger.info("temperature in Quantity a.u.: %s", temp.as_au())
    logger.info("beta: %s", temp.to_beta())
    logger.info("thermal phonon occupation: %s", thermal_phonon_occupation(reduced_temperature))
    logger.info("periodic nearest-neighbor lattice: %s", PERIODIC)
    logger.info("model scheme: %d", scheme)
    logger.info("phonon local dimension: %d", NU_MAX)
    logger.info("initial FC site: %d", _initial_site())
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


def run_multiset(reduced_temperature):
    if N_SNAPSHOTS < 1:
        raise ValueError("N_SNAPSHOTS must be at least 1.")

    output_xlsx, output_npz = _output_paths(reduced_temperature, "multiset")
    temp = temperature_quantity(reduced_temperature)
    model = build_model(scheme=2)
    dynamics_job = MultisetChargeDiffusionDynamics(
        model=model,
        max_bonddim=MAX_BONDDIM,
        temperature=temp,
        method="thermo_field",
        initial_site=_initial_site(),
        stop_at_edge=False,
        if_startup_substeps=IF_STARTUP_SUBSTEPS,
        startup_substeps_n=STARTUP_SUBSTEPS_N,
        if_rdm=True,
    )

    _log_common(reduced_temperature, "multiset", output_xlsx, output_npz, scheme=2)
    logger.info("0th population: %s", dynamics_job.e_occupations_array[0])
    dynamics_job.evolve(evolve_dt=EVOLVE_DT, nsteps=N_SNAPSHOTS - 1)

    populations = np.asarray(dynamics_job.e_occupations_array)
    pd.DataFrame(populations).to_excel(output_xlsx, index=False, header=False)
    np.savez(
        output_npz,
        reduced_temperature=reduced_temperature,
        temperature=temp.as_au(),
        beta=temp.to_beta(),
        thermal_phonon_occupation=thermal_phonon_occupation(reduced_temperature),
        nrow=NROW,
        ncol=NCOL,
        initial_site=_initial_site(),
        time_series=np.asarray(dynamics_job.evolve_times),
        e_occupations=populations,
        S_all=_as_array(dynamics_job.S_all_array),
        S_maxbond_eachset=_as_array(dynamics_job.S_maxbond_eachset_array),
        S_maxbond=np.asarray(dynamics_job.S_maxbond_array),
        S_maxbond_normed=np.asarray(dynamics_job.S_maxbond_normed_array),
        S_maxbond_unnormed=np.asarray(dynamics_job.S_maxbond_unnormed_array),
        rdm_el=np.asarray(dynamics_job.rdm_el_array),
        S_el=np.asarray(dynamics_job.S_el_array),
    )


def run_singleset(reduced_temperature):
    if N_SNAPSHOTS < 1:
        raise ValueError("N_SNAPSHOTS must be at least 1.")

    output_xlsx, output_npz = _output_paths(reduced_temperature, "singleset")
    temp = temperature_quantity(reduced_temperature)
    model = build_model(scheme=MODEL_SCHEME)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=EVOLVE_DT)
    evolve_config.if_startup_substeps = IF_STARTUP_SUBSTEPS
    evolve_config.startup_substeps_n = STARTUP_SUBSTEPS_N
    compress_config = CompressConfig(
        CompressCriteria.fixed,
        max_bonddim=MAX_BONDDIM,
    )
    job_name = os.environ.get(
        "JOB_NAME",
        f"Holstein77_{_temperature_tag(reduced_temperature)}_{MAX_BONDDIM}bd_singleset",
    )
    dynamics_job = ChargeDiffusionDynamics(
        model,
        temperature=temp,
        evolve_config=evolve_config,
        compress_config=compress_config,
        init_electron=InitElectron.fc,
        stop_at_edge=False,
        rdm=True,
        dump_dir="./",
        job_name=job_name,
    )

    _log_common(reduced_temperature, "singleset", output_xlsx, output_npz, scheme=MODEL_SCHEME)
    logger.info("0th population: %s", dynamics_job.e_occupations_array[0])
    dynamics_job.evolve(evolve_dt=EVOLVE_DT, nsteps=N_SNAPSHOTS - 1)

    populations = np.asarray(dynamics_job.e_occupations_array)
    bond_entropy = np.asarray(dynamics_job.bond_vn_entropy_array)
    pd.DataFrame(populations).to_excel(output_xlsx, index=False, header=False)
    np.savez(
        output_npz,
        reduced_temperature=reduced_temperature,
        temperature=temp.as_au(),
        beta=temp.to_beta(),
        thermal_phonon_occupation=thermal_phonon_occupation(reduced_temperature),
        nrow=NROW,
        ncol=NCOL,
        initial_site=_initial_site(),
        time_series=np.asarray(dynamics_job.evolve_times),
        e_occupations=populations,
        r_square=np.asarray(dynamics_job.r_square_array),
        energies=np.asarray(dynamics_job.energies),
        ph_occupations=np.asarray(dynamics_job.ph_occupations_array),
        reduced_density_matrices=np.asarray(dynamics_job.reduced_density_matrices),
        k_occupations=np.asarray(dynamics_job.k_occupations_array),
        eph_vn_entropy=np.asarray(dynamics_job.eph_vn_entropy_array),
        bond_vn_entropy=bond_entropy,
        S_maxbond=np.max(bond_entropy, axis=1),
        S_el=np.asarray(dynamics_job.eph_vn_entropy_array),
        coherent_length=np.asarray(dynamics_job.coherent_length_array),
    )
