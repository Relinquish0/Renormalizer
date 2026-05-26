import os

import numpy as np
import pandas as pd

import renormalizer.lib as reno_lib
import renormalizer.mps.mps as mps_model
from renormalizer.lib.krylov.krylov import _expm_krylov as _project_krylov
from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.log import package_logger as logger


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

N = 75
omega_0 = 1.0
J = 1.0
g = 1.5
nu_max = 16
phonon_dim = nu_max + 1

lam = g**2 * omega_0
displacement = np.sqrt(2 * lam) / omega_0

ph = Phonon.simple_phonon(
    Quantity(omega_0),
    Quantity(displacement),
    phonon_dim,
)
mol = Mol(Quantity(0), [ph])
model = HolsteinModel(
    [mol] * N,
    Quantity(J),
    scheme=2,
    periodic=True,
)

max_bonddim = int(os.environ.get("MAX_BONDDIM", "64"))
evolve_dt = float(os.environ.get("EVOLVE_DT", "0.1"))
n_snapshots = int(os.environ.get("N_SNAPSHOTS", "500"))
if_startup_substeps = os.environ.get("IF_STARTUP_SUBSTEPS", "1").lower() not in {"0", "false", "no"}
startup_substeps_n = int(os.environ.get("STARTUP_SUBSTEPS_N", "10"))
output_xlsx = os.environ.get("OUTPUT_XLSX", f"Holstein75_{max_bonddim}bd_singleset.xlsx")
job_name = os.environ.get("JOB_NAME", f"Holstein75_{max_bonddim}bd_singleset_strictkrylov")

evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=evolve_dt)
evolve_config.if_startup_substeps = if_startup_substeps
evolve_config.startup_substeps_n = startup_substeps_n
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bonddim)

ct = ChargeDiffusionDynamics(
    model,
    evolve_config=evolve_config,
    compress_config=compress_config,
    init_electron=InitElectron.fc,
)
ct.job_name = job_name
ct.stop_at_edge = False

logger.info(f"GPU enabled: {USE_GPU}")
logger.info(f"Backend: {'CuPy' if USE_GPU else 'NumPy'}")
logger.info(
    "strict Krylov allclose enabled: rtol=%s, atol=%s",
    KRYLOV_ALLCLOSE_RTOL,
    KRYLOV_ALLCLOSE_ATOL,
)
logger.info("maximum bond dimension:%d, evolve time step:%s", max_bonddim, evolve_dt)
logger.info("number of stored snapshots:%d", n_snapshots)
logger.info("startup substeps enabled:%s, count:%d", if_startup_substeps, startup_substeps_n)
logger.info("output xlsx: %s", output_xlsx)

ct.evolve(evolve_dt=evolve_dt, nsteps=n_snapshots - 1)
pd.DataFrame(np.array(ct.e_occupations_array)).to_excel(output_xlsx, index=False, header=False)
