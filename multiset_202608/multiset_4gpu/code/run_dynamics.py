# -*- coding: utf-8 -*-

"""4xV100 multiset-MPS driver for the 2D Holstein scaling ladder.

Patch installation order matters and is enforced here:

1. ``mpi_common.configure_rank_gpu`` before anything imports the Renormalizer
   backend, because ``RENO_GPU`` is read at import time.
2. ``mpi_apply_hop_patch`` / ``mpi_krylov_patch`` -- the Krylov and Hamiltonian
   action decomposition, needed at every stage.
3. ``mpi_shard_patch`` -- alpha-row sharding, which must be installed *before*
   the ``MultisetModel`` is constructed since it intercepts the pair grouping
   that runs on the last line of ``MultisetModel.__init__``.
4. ``mpi_expand_patch`` -- only for stage 1; from stage 2 the shard patch owns
   ``expand_bond_dimension_multiset``.

Set ``RENO_MS_SHARD_STAGE`` to 1..4 to select how much is distributed; see
``mpi_shard_patch`` for what each stage covers.
"""

import json
import os
import sys
import time

from mpi4py import MPI

# The patch modules import each other by flat name, so this directory has to be
# importable however the driver was launched.  scaling/run_scaling.py sets both
# cwd and PYTHONPATH, but a bare `srun ... python .../code/run_dynamics.py` does
# not, and the resulting ModuleNotFoundError kills every rank a second after the
# job starts -- which reads like an allocation failure rather than an import one.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mpi_common as mpi_common

mpi_common.configure_rank_gpu()

COMM = mpi_common.COMM
RANK = mpi_common.RANK
SIZE = mpi_common.SIZE

import numpy as np

import mpi_apply_hop_patch as mpi_hop
import mpi_krylov_patch as mpi_krylov
import mpi_shard_patch as mpi_shard

mpi_hop.install_patch()
mpi_krylov.install_patch()
mpi_shard.install_patch()

if mpi_shard.STAGE < 2:
    import mpi_expand_patch as mpi_expand

    mpi_expand.install_patch()
else:
    mpi_expand = None

from renormalizer.mps.backend import GPU_ID as BACKEND_GPU_ID
from renormalizer.mps.backend import USE_GPU
from renormalizer.multiset import MultisetChargeDiffusionDynamics
from renormalizer.utils.log import package_logger as logger

import memory
from holstein import LIGHTWEIGHT_OBSERVABLES, HolsteinConfig, build_model

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
RUN_LABEL = os.environ.get("RUN_LABEL", "")
# SETUP_ONLY stops the run early so the host-side ceiling can be measured on a
# CPU node, without a GPU and without paying for a full TDVP sweep:
#
#   0  full run
#   1  stop after the model, MPOs and the initial state are built
#   2  also build the per-pair environment cache before stopping
#
# Level 2 exists because level 1 systematically *understates* the footprint:
# ``conj_mps`` is built lazily, so the environment cache -- the largest host
# term and the one the alpha sharding is aimed at -- is only allocated on the
# first evolution step.  Building it directly costs a small fraction of a step
# while measuring exactly the term that decides whether a lattice fits.
_SETUP_ONLY_RAW = os.environ.get("SETUP_ONLY", "0").strip().lower()
if _SETUP_ONLY_RAW in {"2", "environ", "environment"}:
    SETUP_ONLY, PROBE_ENVIRON = True, True
else:
    SETUP_ONLY = _SETUP_ONLY_RAW not in {"0", "false", "no", "", "off"}
    PROBE_ENVIRON = False


def _report_memory(phase, ms_mps=None, ms_model=None):
    """Log per-rank host and GPU high-water marks; returns rank 0's view.

    When a state is passed in, the multiset MPS and the environment cache are
    broken out as well.  Two MPS totals are reported: what this rank has
    *resident* (its own alpha rows plus the ghost halo, i.e. what its footprint
    is actually made of) and what it *owns* -- only the latter sums across ranks
    to the size of the whole multiset state without counting a halo row twice.
    """
    host = mpi_common.host_peak_gb()
    pool_used, pool_total, dev_used, dev_total = mpi_common.gpu_report()
    hosts = mpi_common.gather_all(host)
    devs = mpi_common.gather_all(dev_used)
    pools = mpi_common.gather_all(pool_total)

    state = {}
    if ms_mps is not None:
        layout = mpi_common.AlphaLayout(len(ms_mps.msmps))
        record = memory.snapshot(
            phase, ms_mps=ms_mps, ms_model=ms_model, owned_alphas=layout.owned
        )
        resident = record["mps"]
        owned = record["mps_owned"]
        state = {
            "mps_resident_gb": mpi_common.gather_all(resident["total_gb"]),
            "mps_resident_sets": mpi_common.gather_all(resident["n_sets"]),
            "mps_owned_gb": mpi_common.gather_all(owned["total_gb"]),
            "mps_owned_sets": mpi_common.gather_all(owned["n_sets"]),
            "mps_per_set_max_gb": mpi_common.gather_all(resident["per_set_max_gb"]),
            "mps_device_gb": mpi_common.gather_all(resident["device_gb"]),
        }
        environ = record.get("environ")
        if environ:
            state["environ_gb"] = mpi_common.gather_all(environ["total_gb"])
            state["environ_pairs"] = mpi_common.gather_all(environ["n_environ"])

    if RANK == 0:
        logger.info(
            "[mem/%s] host peak GiB per rank: %s (sum %.1f); "
            "GPU used GiB per rank: %s; cupy pool GiB per rank: %s",
            phase,
            ["%.1f" % v for v in hosts],
            sum(hosts),
            ["%.2f" % v for v in devs],
            ["%.2f" % v for v in pools],
        )
        if state:
            logger.info(
                "[mem/%s] multiset MPS: whole state %.2f GiB over %d sets; "
                "largest single set %.1f MiB; resident per rank %s GiB "
                "(%s sets incl. ghosts); on device per rank %s GiB",
                phase,
                sum(state["mps_owned_gb"]),
                int(sum(state["mps_owned_sets"])),
                max(state["mps_per_set_max_gb"]) * 1024,
                ["%.1f" % v for v in state["mps_resident_gb"]],
                ["%d" % v for v in state["mps_resident_sets"]],
                ["%.2f" % v for v in state["mps_device_gb"]],
            )
            if "environ_gb" in state:
                logger.info(
                    "[mem/%s] environments: %.1f GiB total over %d pairs; "
                    "per rank %s GiB",
                    phase,
                    sum(state["environ_gb"]),
                    int(sum(state["environ_pairs"])),
                    ["%.1f" % v for v in state["environ_gb"]],
                )
    out = {
        "phase": phase,
        "host_peak_gb": hosts,
        "gpu_used_gb": devs,
        "gpu_pool_gb": pools,
        "gpu_total_gb": dev_total,
    }
    out.update(state)
    return out


def main():
    cfg = HolsteinConfig()
    memory_log = []

    if RANK == 0:
        logger.info("=" * 72)
        logger.info(
            "lattice %s (%d sites), m=%d, dt=%s, steps=%d",
            cfg.tag,
            cfg.n_sites,
            cfg.max_bonddim,
            cfg.evolve_dt,
            cfg.nsteps,
        )
        logger.info(
            "ranks=%d, shard stage=%d, krylov mode=%s, krylov block=%s, GPU=%s",
            SIZE,
            mpi_shard.STAGE,
            os.environ.get("RENO_MPI_KRYLOV_MODE", "distributed"),
            os.environ.get("RENO_MS_KRYLOV_BLOCK", "24"),
            USE_GPU,
        )
        logger.info("output dir: %s", os.path.abspath(OUTPUT_DIR))
        logger.info("=" * 72)
    logger.info("rank %d/%d -> backend GPU %s", RANK, SIZE, BACKEND_GPU_ID)

    COMM.Barrier()
    t_model = time.perf_counter()
    model = build_model(cfg)
    t_model = time.perf_counter() - t_model

    t_setup = time.perf_counter()
    job = MultisetChargeDiffusionDynamics(
        model=model,
        max_bonddim=cfg.max_bonddim,
        initial_site=cfg.initial_site,
        stop_at_edge=False,
        if_rdm=False,
        observables=LIGHTWEIGHT_OBSERVABLES,
    )
    COMM.Barrier()
    t_setup = time.perf_counter() - t_setup
    memory_log.append(
        _report_memory("after-setup", ms_mps=job.latest_mps, ms_model=job.ms_model)
    )
    if RANK == 0:
        logger.info(
            "model build %.1f s, multiset setup + init + expand %.1f s",
            t_model,
            t_setup,
        )

    t_environ = 0.0
    if SETUP_ONLY and PROBE_ENVIRON:
        COMM.Barrier()
        t_environ = time.perf_counter()
        # Same entry point the sweep uses, so this measures the cache the
        # evolution would really hold -- including the alpha sharding, which
        # only gives each rank the pairs it owns.
        job.ms_model._get_or_build_environ_list(job.latest_mps)
        COMM.Barrier()
        t_environ = time.perf_counter() - t_environ
        memory_log.append(
            _report_memory(
                "after-environ", ms_mps=job.latest_mps, ms_model=job.ms_model
            )
        )
        if RANK == 0:
            logger.info("environment cache built in %.1f s", t_environ)

    if SETUP_ONLY:
        if RANK == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            stem = os.path.join(OUTPUT_DIR, f"{cfg.tag}{RUN_LABEL}_setup")
            with open(stem + "_summary.json", "w") as handle:
                json.dump(
                    {
                        "config": cfg.as_dict(),
                        "mpi_size": SIZE,
                        "shard_stage": mpi_shard.STAGE,
                        "model_seconds": t_model,
                        "setup_seconds": t_setup,
                        "environ_seconds": t_environ,
                        "memory": memory_log,
                        "status": "setup+environ" if PROBE_ENVIRON else "setup-only",
                    },
                    handle,
                    indent=2,
                )
            logger.info("setup only: wrote %s_summary.json", stem)
        return

    t_evolve = time.perf_counter()
    job.evolve(evolve_dt=cfg.evolve_dt, nsteps=cfg.nsteps)
    COMM.Barrier()
    t_evolve = time.perf_counter() - t_evolve
    memory_log.append(
        _report_memory("after-evolve", ms_mps=job.latest_mps, ms_model=job.ms_model)
    )

    wall = COMM.reduce(t_evolve, op=MPI.MAX, root=0)
    mpi_krylov.summarize_patch_usage()
    mpi_hop.summarize_patch_usage()
    if mpi_expand is not None:
        mpi_expand.summarize_patch_usage()

    if RANK == 0:
        populations = np.array(job.e_occupations_array)
        times = np.array(job.evolve_times)
        logger.info(
            "evolve wall time %.1f s for %d steps (%.1f s/step)",
            wall,
            cfg.nsteps,
            wall / max(cfg.nsteps, 1),
        )
        logger.info("final population sum: %.12f", populations[-1].sum())
        logger.info("final population max: %.6e", populations[-1].max())

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.join(OUTPUT_DIR, f"{cfg.tag}{RUN_LABEL}")
        np.savez(
            stem + ".npz",
            time_series=times,
            e_occupations=populations,
            wall_time_seconds=np.array(wall),
            setup_seconds=np.array(t_setup),
            model_seconds=np.array(t_model),
            mpi_size=np.array(SIZE),
            shard_stage=np.array(mpi_shard.STAGE),
        )
        summary = {
            "config": cfg.as_dict(),
            "mpi_size": SIZE,
            "shard_stage": mpi_shard.STAGE,
            "krylov_mode": os.environ.get("RENO_MPI_KRYLOV_MODE", "distributed"),
            "krylov_block": int(os.environ.get("RENO_MS_KRYLOV_BLOCK", "24")),
            "model_seconds": t_model,
            "setup_seconds": t_setup,
            "evolve_seconds": wall,
            "seconds_per_step": wall / max(cfg.nsteps, 1),
            "memory": memory_log,
            "status": "ok",
        }
        with open(stem + "_summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("wrote %s.npz and %s_summary.json", stem, stem)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("rank %d failed; aborting the MPI job.", RANK)
        sys.stdout.flush()
        sys.stderr.flush()
        COMM.Abort(1)
        sys.exit(1)
