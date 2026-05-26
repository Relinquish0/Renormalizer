# -*- coding: utf-8 -*-

import logging
import json
import os

import renormalizer.lib as reno_lib
import renormalizer.mps.mps as mps_model
from renormalizer.lib.krylov.krylov import _expm_krylov as _project_krylov
from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron
from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.utils.log import package_logger as logger

import numpy as np

log.init_log(logging.INFO)


KRYLOV_ALLCLOSE_RTOL = float(os.environ.get("KRYLOV_ALLCLOSE_RTOL", "1e-8"))
KRYLOV_ALLCLOSE_ATOL = float(os.environ.get("KRYLOV_ALLCLOSE_ATOL", "1e-10"))


def _expm_krylov_strict_allclose(Afunc, dt, vstart, block_size=50):
    if not np.iscomplex(dt):
        dt = dt.real

    vstart = xp.asarray(vstart)
    nrmv = float(xp.linalg.norm(vstart))
    assert nrmv > 0
    vstart = vstart / nrmv

    alpha = np.zeros(block_size)
    beta = np.zeros(block_size - 1)

    V = xp.empty((block_size, len(vstart)), dtype=vstart.dtype)
    V[0] = vstart
    res = None

    for j in range(len(vstart)):
        w = Afunc(V[j])
        alpha[j] = xp.vdot(w, V[j]).real

        if j == len(vstart) - 1:
            return _project_krylov(alpha[:j + 1], beta[:j], V[:j + 1, :].T, nrmv, dt), j + 1

        if len(V) == j + 1:
            V, old_V = xp.empty((len(V) + block_size, len(vstart)), dtype=vstart.dtype), V
            V[:len(old_V)] = old_V
            del old_V
            alpha = np.concatenate([alpha, np.zeros(block_size)])
            beta = np.concatenate([beta, np.zeros(block_size)])

        w -= alpha[j] * V[j] + (beta[j - 1] * V[j - 1] if j > 0 else 0)
        beta[j] = xp.linalg.norm(w)
        if beta[j] < 100 * len(vstart) * np.finfo(float).eps:
            return _project_krylov(alpha[:j + 1], beta[:j], V[:j + 1, :].T, nrmv, dt), j + 1

        if 3 < j and j % 2 == 0:
            new_res = _project_krylov(alpha[:j + 1], beta[:j], V[:j + 1].T, nrmv, dt)
            if res is not None and xp.allclose(
                res,
                new_res,
                rtol=KRYLOV_ALLCLOSE_RTOL,
                atol=KRYLOV_ALLCLOSE_ATOL,
            ):
                return new_res, j + 1
            res = new_res
        V[j + 1] = w / beta[j]


reno_lib.expm_krylov = _expm_krylov_strict_allclose
mps_model.expm_krylov = _expm_krylov_strict_allclose


with open("../../../example/fmo_sdf.json") as fin:
    # a 107*2 matrix
    sdf_values = json.load(fin)
sdf_values = np.array(sdf_values)

j_matrix_cm = np.array([[310, -98, 6, -6, 7, -12, -10, 38, ],
                        [-98, 230, 30, 7, 2, 12, 5, 8, ],
                        [6, 30, 0, -59, -2, -10, 5, 2, ],
                        [-6, 7, -59, 180, -65, -17, -65, -2, ],
                        [7, 2, -2, -65, 405, 89, -6, 5, ],
                        [-12, 11, -10, -17, 89, 320, 32, -10, ],
                        [-10, 5, 5, -64, -6, 32, 270, -11, ],
                        [38, 8, 2, -2, 5, -10, -11, 505, ], ])

N_PHONONS = 35

TOTAL_HR = 0.42

if __name__ == "__main__":

    omegas_cm = np.linspace(2, 300, N_PHONONS)
    omegas_au = omegas_cm * cm2au
    hr_factors = np.interp(omegas_cm, sdf_values[:, 0], sdf_values[:, 1])

    hr_factors *= TOTAL_HR / hr_factors.sum()

    lams = hr_factors * omegas_au
    phonons = [Phonon.simplest_phonon(Quantity(o), Quantity(l), lam=True) for o,l in zip(omegas_au, lams)]


    j_matrix_au = j_matrix_cm * cm2au

    mlist = []
    for j in np.diag(j_matrix_au):
        m = Mol(Quantity(j), phonons)
        mlist.append(m)

    # starts from 1
    mol_arangement = np.array([7, 5, 3, 1, 2, 4, 6]) - 1
    model = HolsteinModel(list(np.array(mlist)[mol_arangement]), j_matrix_au[mol_arangement][:, mol_arangement], )
    
    evolve_dt = 160
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=evolve_dt)
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=64)
    ct = ChargeDiffusionDynamics(model, temperature=Quantity(300,"K"),evolve_config=evolve_config, compress_config=compress_config, init_electron=InitElectron.fc)
    ct.dump_dir = "./"
    ct.job_name = 'fmo_300k_64bd_strictkrylov'
    ct.stop_at_edge = False
    logger.info(f"GPU enabled: {USE_GPU}")
    logger.info("Backend: %s", "CuPy" if USE_GPU else "NumPy")
    logger.info(
        "strict Krylov allclose enabled: rtol=%s, atol=%s",
        KRYLOV_ALLCLOSE_RTOL,
        KRYLOV_ALLCLOSE_ATOL,
    )
    ct.evolve(evolve_dt=evolve_dt, evolve_time=40000)