# -*- coding: utf-8 -*-

import json
import logging
import os
from pathlib import Path

import numpy as np

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod,
    Quantity,
    log,
)
from renormalizer.utils.constant import cm2au


log.init_log(getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()))
logger = logging.getLogger("renormalizer.cbe_fmo")

BASE_DIR = Path(__file__).resolve().parents[1]

with open(BASE_DIR / "../example" / "fmo_sdf.json") as fin:
    sdf_values = np.array(json.load(fin))

J_MATRIX_CM = np.array([
    [310, -98, 6, -6, 7, -12, -10, 38],
    [-98, 230, 30, 7, 2, 12, 5, 8],
    [6, 30, 0, -59, -2, -10, 5, 2],
    [-6, 7, -59, 180, -65, -17, -65, -2],
    [7, 2, -2, -65, 405, 89, -6, 5],
    [-12, 11, -10, -17, 89, 320, 32, -10],
    [-10, 5, 5, -64, -6, 32, 270, -11],
    [38, 8, 2, -2, 5, -10, -11, 505],
])

N_PHONONS = int(os.environ.get("N_PHONONS", "35"))
TOTAL_HR = 0.42


def build_model():
    omegas_cm = np.linspace(2, 300, N_PHONONS)
    omegas_au = omegas_cm * cm2au
    hr_factors = np.interp(omegas_cm, sdf_values[:, 0], sdf_values[:, 1])
    hr_factors *= TOTAL_HR / hr_factors.sum()

    lams = hr_factors * omegas_au
    phonons = [
        Phonon.simplest_phonon(Quantity(o), Quantity(lam), lam=True)
        for o, lam in zip(omegas_au, lams)
    ]

    j_matrix_au = J_MATRIX_CM * cm2au
    mlist = [Mol(Quantity(j), phonons) for j in np.diag(j_matrix_au)]

    # Starts from 1 in the literature ordering.
    mol_arrangement = np.array([7, 5, 3, 1, 2, 4, 6]) - 1
    return HolsteinModel(
        list(np.array(mlist)[mol_arrangement]),
        j_matrix_au[mol_arrangement][:, mol_arrangement], scheme=2
    )


if __name__ == "__main__":
    model = build_model()

    evolve_dt = float(os.environ.get("EVOLVE_DT", "160"))
    evolve_time = float(os.environ.get("EVOLVE_TIME", "40000"))
    max_bonddim = int(os.environ.get("MAX_BONDDIM", "32"))

    expansion_method = os.environ.get("EXPANSION_METHOD", "cbe")
    cbe_max_expand_env = os.environ.get("CBE_MAX_EXPAND")
    cbe_max_expand = None if cbe_max_expand_env in [None, "", "none", "None"] else int(cbe_max_expand_env)
    cbe_Dpre_env = os.environ.get("CBE_DPRE")
    cbe_Dpre = None if cbe_Dpre_env in [None, "", "none", "None"] else int(cbe_Dpre_env)
    cbe_warmup_time = float(os.environ.get("CBE_WARMUP_TIME", "160.0"))
    cbe_warmup_substeps = int(os.environ.get("CBE_WARMUP_SUBSTEPS", "10"))
    cbe_Dmax = int(os.environ.get("CBE_DMAX", max_bonddim))

    evolve_config = EvolveConfig(
        EvolveMethod.tdvp_ps,
        guess_dt=evolve_dt,
        expansion_method=expansion_method,
        cbe_eps_pre=float(os.environ.get("CBE_EPS_PRE", "1e-4")),
        cbe_eps_final=float(os.environ.get("CBE_EPS_FINAL", "1e-6")),
        cbe_eps_trim=float(os.environ.get("CBE_EPS_TRIM", "1e-14")),
        cbe_Dmax=cbe_Dmax,
        cbe_max_expand=cbe_max_expand,
        cbe_Dpre=cbe_Dpre,
        cbe_warmup_time=cbe_warmup_time,
        cbe_warmup_substeps=cbe_warmup_substeps,
        cbe_disable_after_warmup=os.environ.get("CBE_DISABLE_AFTER_WARMUP", "1") != "0",
        cbe_lock_after_warmup=os.environ.get("CBE_LOCK_AFTER_WARMUP", "1") != "0",
        cbe_production_dt=evolve_dt,
        cbe_warmup_dt=cbe_warmup_time / cbe_warmup_substeps,
    )
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bonddim)

    ct = ChargeDiffusionDynamics(
        model,
        evolve_config=evolve_config,
        compress_config=compress_config,
        init_electron=InitElectron.fc,
    )
    ct.dump_dir = "./"
    ct.job_name = os.environ.get("JOB_NAME", f"fmo_0k_singleset_cbe_M{max_bonddim}")
    ct.stop_at_edge = False

    if expansion_method == "cbe" and cbe_warmup_time > 0 and cbe_warmup_substeps > 0:
        warmup_dt = cbe_warmup_time / cbe_warmup_substeps
        logger.info(
            "Starting CBE warm-up: warmup_time=%s warmup_substeps=%s warmup_dt=%s",
            cbe_warmup_time, cbe_warmup_substeps, warmup_dt,
        )
        ct.evolve(evolve_dt=warmup_dt, nsteps=cbe_warmup_substeps)
        logger.info("CBE warm-up finished at time %s", ct.latest_evolve_time)
        logger.info("Bond dims after warm-up: %s", ct.latest_mps.bond_dims)
        logger.info("Bond entropy after warm-up: %s", ct.latest_mps.calc_bond_entropy())
        try:
            _, svals = ct.latest_mps.copy().canonicalise().compress(temp_m_trunc=np.inf, ret_s=True)
            logger.info("Singular values after warm-up: %s", svals)
        except Exception:
            logger.exception("Failed to print singular values after warm-up")
        if evolve_config.cbe_disable_after_warmup:
            evolve_config.cbe_runtime_disabled = True
            logger.info("Switch to normal 1TDVP at time %s", ct.latest_evolve_time)
        if evolve_config.cbe_lock_after_warmup:
            compress_config.bond_dim_max_value = cbe_Dmax
            compress_config.max_dims = np.full(len(ct.latest_mps) + 1, cbe_Dmax, dtype=int)
            logger.info("Locked max bond dimension to cbe_Dmax=%s", cbe_Dmax)
        remaining_time = evolve_time - cbe_warmup_time
        if remaining_time > 0:
            ct.evolve(evolve_dt=evolve_dt, evolve_time=remaining_time)
    else:
        ct.evolve(evolve_dt=evolve_dt, evolve_time=evolve_time)
