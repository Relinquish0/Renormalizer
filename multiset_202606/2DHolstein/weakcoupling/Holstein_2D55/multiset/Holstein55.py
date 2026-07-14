# -*- coding: utf-8 -*-

"""Multiset MPS charge-diffusion dynamics for the 2D 15x15 Holstein model.

This benchmark follows the 2D weak-coupling case in Kloss et al.:
g = 0.5, omega0 = J = 1, chi' = 16, nu_max = 8, dt = 0.1.

The square lattice is flattened in row-major order:
(0, 0), (0, 1), ..., (0, 14), (1, 0), ..., (14, 14).
The default initial Franck-Condon excitation is placed at the center site
(7, 7), i.e. flattened index 112.
"""

import os

import numpy as np
import pandas as pd

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.multiset import MultisetChargeDiffusionDynamics
from renormalizer.mps.backend import USE_GPU
from renormalizer.utils import Quantity
from renormalizer.utils.log import package_logger as logger


NROW = int(os.environ.get("NROW", "5"))
NCOL = int(os.environ.get("NCOL", "5"))
OMEGA_0 = float(os.environ.get("OMEGA_0", "1.0"))
J = float(os.environ.get("J", "1.0"))
G = float(os.environ.get("G", "0.5"))
NU_MAX = int(os.environ.get("NU_MAX", "8"))

MAX_BONDDIM = int(os.environ.get("MAX_BONDDIM", "16"))
EVOLVE_DT = float(os.environ.get("EVOLVE_DT", "0.1"))
N_SNAPSHOTS = int(os.environ.get("N_SNAPSHOTS", "500"))
INITIAL_SITE = os.environ.get("INITIAL_SITE")
OUTPUT_XLSX = os.environ.get(
    "OUTPUT_XLSX",
    f"H{NROW}{NCOL}_2D_m{MAX_BONDDIM}.xlsx",
)
OUTPUT_NPZ = os.environ.get(
    "OUTPUT_NPZ",
    f"H{NROW}{NCOL}_2D_m{MAX_BONDDIM}.npz",
)
IF_STARTUP_SUBSTEPS = os.environ.get("IF_STARTUP_SUBSTEPS", "1").lower() not in {
    "0",
    "false",
    "no",
}
STARTUP_SUBSTEPS_N = int(os.environ.get("STARTUP_SUBSTEPS_N", "10"))


def site_index(ix, iy, ncol):
    return ix * ncol + iy


def build_2d_j_matrix(nrow, ncol, j, periodic=False):
    j_matrix = np.zeros((nrow * ncol, nrow * ncol))

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


def build_model():
    phonon_dim = NU_MAX + 1
    lam = G**2 * OMEGA_0
    displacement = np.sqrt(2.0 * lam) / OMEGA_0

    ph = Phonon.simple_phonon(
        Quantity(OMEGA_0),
        Quantity(displacement),
        phonon_dim,
    )
    mol = Mol(Quantity(0), [ph])
    j_matrix = build_2d_j_matrix(NROW, NCOL, J, periodic=False)

    return HolsteinModel(
        [mol] * (NROW * NCOL),
        j_matrix,
        scheme=2,
    )


def main():
    if N_SNAPSHOTS < 1:
        raise ValueError("N_SNAPSHOTS must be at least 1.")

    initial_site = (
        site_index(NROW // 2, NCOL // 2, NCOL)
        if INITIAL_SITE is None
        else int(INITIAL_SITE)
    )
    model = build_model()

    dynamics_job = MultisetChargeDiffusionDynamics(
        model=model,
        max_bonddim=MAX_BONDDIM,
        initial_site=initial_site,
        stop_at_edge=False,
        if_startup_substeps=IF_STARTUP_SUBSTEPS,
        startup_substeps_n=STARTUP_SUBSTEPS_N,
    )

    logger.info("GPU enabled: %s", USE_GPU)
    logger.info("lattice: %d x %d (%d sites)", NROW, NCOL, NROW * NCOL)
    logger.info("parameters: omega0=%s, J=%s, g=%s, nu_max=%d", OMEGA_0, J, G, NU_MAX)
    logger.info("phonon local dimension: %d", NU_MAX + 1)
    logger.info("initial FC site: %d", initial_site)
    logger.info("maximum bond dimension: %d", MAX_BONDDIM)
    logger.info("evolve time step: %s", EVOLVE_DT)
    logger.info("number of stored snapshots: %d", N_SNAPSHOTS)
    logger.info(
        "startup substeps enabled: %s, count: %d",
        IF_STARTUP_SUBSTEPS,
        STARTUP_SUBSTEPS_N,
    )
    logger.info("output xlsx: %s", OUTPUT_XLSX)
    logger.info("output npz: %s", OUTPUT_NPZ)

    logger.info("0th population: %s", dynamics_job.e_occupations_array[0])
    dynamics_job.evolve(evolve_dt=EVOLVE_DT, nsteps=N_SNAPSHOTS - 1)

    populations = np.array(dynamics_job.e_occupations_array)
    pd.DataFrame(populations).to_excel(OUTPUT_XLSX, index=False, header=False)
    np.savez(
        OUTPUT_NPZ,
        time_series=np.array(dynamics_job.evolve_times),
        e_occupations=populations,
        S_maxbond=np.array(dynamics_job.S_maxbond_array),
        rdm_el=np.array(dynamics_job.rdm_el_array),
        S_el=np.array(dynamics_job.S_el_array),
    )


if __name__ == "__main__":
    main()
