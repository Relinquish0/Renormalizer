import importlib.util
import logging
import os
import time
from pathlib import Path

import numpy as np

from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.tn import BasisTree, MsTTNO, MsTTNS, TTNS
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
N_OT = model_module.N_OT


def build_initial_ms_ttns(ms_model, basis_tree, initial_site=0):
    init = TTNS(basis_tree, condition={})
    init.compress_config = CompressConfig(CompressCriteria.fixed)
    zero = init.copy().scale(0, inplace=True)
    components = []
    for istate in range(ms_model.N_electron):
        components.append(init.copy() if istate == initial_site else zero.copy())
    return MsTTNS.from_ttns_list(components)


def run(job_name="p3ht_ms_ttn", max_bond_dim=32, dt_fs=1.0, total_fs=200.0, initial_site=0):
    model = P3HTPCBMModel()
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=Quantity(dt_fs, "fs").as_au())
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bond_dim)
    ms_model = MultisetModel(
        model,
        max_bonddim=max_bond_dim,
        evolve_config=evolve_config,
        compress_config=compress_config,
        auto_init=False,
    )

    basis_tree = BasisTree.binary_mctdh(ms_model.init_model.basis, contract_primitive=True)
    ms_ttns = build_initial_ms_ttns(ms_model, basis_tree, initial_site=initial_site)
    ms_ttns.compress_config = compress_config
    ms_ttns.evolve_config = evolve_config

    time_start = time.perf_counter()
    ms_ttno = MsTTNO.from_multiset_model(basis_tree, ms_model)
    logger.info("profile MsTTNO_build seconds=%.6f", time.perf_counter() - time_start)

    time_start = time.perf_counter()
    ms_ttns = ms_ttns.expand_bond_dimension_multiset(ms_ttno)
    logger.info("profile expand_bond_dimension_multiset seconds=%.6f", time.perf_counter() - time_start)

    step_au = Quantity(dt_fs, "fs").as_au()
    nsteps = int(round(total_fs / dt_fs))
    times_fs = [0.0]
    time_start = time.perf_counter()
    pop = ms_ttns.population()
    logger.info("profile population step=%d seconds=%.6f", 0, time.perf_counter() - time_start)
    occupations = [pop]
    bond_dims = [ms_ttns.bond_dims]
    logger.info("step=%d time_fs=%.6f e_occupations=%s", 0, times_fs[-1], np.array2string(occupations[-1], precision=8))

    for istep in range(1, nsteps + 1):
        ms_ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=step_au)
        ms_ttns = ms_ttns.evolve(ms_ttno, step_au, normalize=True)
        times_fs.append(times_fs[-1] + dt_fs)
        time_start = time.perf_counter()
        pop = ms_ttns.population()
        logger.info("profile population step=%d seconds=%.6f", istep, time.perf_counter() - time_start)
        occupations.append(pop)
        bond_dims.append(ms_ttns.bond_dims)
        logger.info(
            "step=%d time_fs=%.6f e_occupations=%s",
            istep,
            times_fs[-1],
            np.array2string(occupations[-1], precision=8),
        )

    occupations = np.asarray(occupations, dtype=float)
    out_path = Path(__file__).with_name(f"{job_name}_{max_bond_dim}.npz")
    np.savez(
        out_path,
        time_fs=np.asarray(times_fs),
        state_labels=np.asarray(model.state_labels, dtype=object),
        e_occupations=occupations,
        le_occupations=occupations[:, :N_OT],
        cs_occupations=occupations[:, N_OT:],
        le1_occupation=occupations[:, 0],
        bond_dims=np.asarray(bond_dims, dtype=object),
        active_pairs=np.asarray(ms_ttno.active_pairs_index, dtype=int),
    )
    return out_path


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


def main():
    print(
        run(
            job_name=os.environ.get("JOB_NAME", "p3ht_ms_ttn"),
            max_bond_dim=getenv("MAX_BONDDIM", 32, int),
            dt_fs=getenv("DT_FS", 1.0, float),
            total_fs=getenv("TOTAL_FS", 200.0, float),
            initial_site=getenv("INITIAL_SITE", 0, int),
        )
    )


if __name__ == "__main__":
    main()
