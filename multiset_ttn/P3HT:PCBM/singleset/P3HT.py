import importlib.util
import logging
import os
from pathlib import Path

import numpy as np

from renormalizer.model import Op
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.tn import BasisTree, TTNO, TTNS
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod, Quantity


logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


MODEL_PATH = Path(__file__).resolve().parents[3] / "multiset_202605/P3HT:PCBM/multiset/P3HT.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location("p3ht_multiset_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_model_module()
P3HTPCBMModel = model_module.P3HTPCBMModel


def build_all_electronic_fc_ttns(basis_tree, model):
    vacuum = TTNS(basis_tree)
    creation_ops = [Op(r"a^\dagger", state) for state in model.state_labels]
    return TTNO(basis_tree, creation_ops) @ vacuum


def run(job_name="p3ht_ttns", max_bond_dim=32, dt_fs=1.0, total_fs=200.0):
    model = P3HTPCBMModel()
    basis_tree = BasisTree.binary_mctdh(model.basis, contract_primitive=True)
    ttno = TTNO(basis_tree, model.ham_terms)
    occ_ops = [TTNO(basis_tree, Op(r"a^\dagger a", state)) for state in model.state_labels]

    init_state = model.basis[0].dof_name_map[model.le_states[0]]
    ttns = TTNS(basis_tree, condition={model.le_states[0]: init_state})
    ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bond_dim)
    tn_state_ex = build_all_electronic_fc_ttns(basis_tree, model)
    ttns = expand_bond_dimension_general(ttns, ttno, ex_mps=tn_state_ex)

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
