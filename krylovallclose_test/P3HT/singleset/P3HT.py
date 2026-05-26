import importlib.util
import logging
import os
from pathlib import Path

import numpy as np

import renormalizer.lib as reno_lib
import renormalizer.tn.time_evolution as tn_time_evolution
from renormalizer.lib.krylov.krylov import _expm_krylov as _project_krylov
from renormalizer.model import Op
from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.tn import BasisTree, TTNO, TTNS
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod, Quantity


logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
tn_time_evolution.expm_krylov = _expm_krylov_strict_allclose

MODEL_PATH = Path(__file__).resolve().parents[3] / "multiset_202605/P3HT:PCBM/multiset/P3HT.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location("p3ht_multiset_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_model_module()
P3HTPCBMModel = model_module.P3HTPCBMModel


def run(job_name="p3ht_ttns", max_bond_dim=32, dt_fs=1.0, total_fs=200.0):
    logger.info("GPU enabled: %s", USE_GPU)
    logger.info("Backend: %s", "CuPy" if USE_GPU else "NumPy")
    logger.info(
        "strict Krylov allclose enabled: rtol=%s, atol=%s",
        KRYLOV_ALLCLOSE_RTOL,
        KRYLOV_ALLCLOSE_ATOL,
    )
    model = P3HTPCBMModel()
    basis_tree = BasisTree.binary_mctdh(model.basis, contract_primitive=True)
    ttno = TTNO(basis_tree, model.ham_terms)
    occ_ops = [TTNO(basis_tree, Op(r"a^\dagger a", state)) for state in model.state_labels]

    init_state = model.basis[0].dof_name_map[model.le_states[0]]
    ttns = TTNS(basis_tree, condition={model.le_states[0]: init_state})
    ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bond_dim)
    ttns = expand_bond_dimension_general(ttns, ttno, ex_mps=None)

    step_au = Quantity(dt_fs, "fs").as_au()
    nsteps = int(round(total_fs / dt_fs))
    times_fs = [0.0]
    occupations = [[ttns.expectation(op) for op in occ_ops]]
    energies = [ttns.expectation(ttno)]
    bond_dims = [ttns.bond_dims]
    logger.info("step=%d time_fs=%.6f e_occupations=%s", 0, times_fs[-1], np.array2string(np.asarray(occupations[-1]), precision=8))

    for istep in range(1, nsteps + 1):
        ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=step_au)
        ttns = ttns.evolve(ttno, step_au)
        times_fs.append(times_fs[-1] + dt_fs)
        occupations.append([ttns.expectation(op) for op in occ_ops])
        energies.append(ttns.expectation(ttno))
        bond_dims.append(ttns.bond_dims)
        logger.info(
            "step=%d time_fs=%.6f e_occupations=%s",
            istep,
            times_fs[-1],
            np.array2string(np.asarray(occupations[-1]), precision=8),
        )

    occupations = np.asarray(occupations, dtype=float)
    n_le = len(model.le_states)
    out_path = Path(__file__).with_name(f"{job_name}_{max_bond_dim}.npz")
    np.savez(
        out_path,
        time_fs=np.asarray(times_fs),
        state_labels=np.asarray(model.state_labels, dtype=object),
        e_occupations=occupations,
        le_occupations=occupations[:, :n_le],
        cs_occupations=occupations[:, n_le:],
        le1_occupation=occupations[:, 0],
        energies=np.asarray(energies),
        bond_dims=np.asarray(bond_dims, dtype=object),
    )
    return out_path


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


def main():
    print(
        run(
            job_name=os.environ.get("JOB_NAME", "p3ht_ttns"),
            max_bond_dim=getenv("MAX_BONDDIM", 32, int),
            dt_fs=getenv("DT_FS", 1.0, float),
            total_fs=getenv("TOTAL_FS", 200.0, float),
        )
    )


if __name__ == "__main__":
    main()
