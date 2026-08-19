# -*- coding: utf-8 -*-

"""4-GPU MPI multiset size-scaling probe: 10 steps, pass or fail.

The MPI monkey patches are imported from the existing benchmark directory
rather than copied, so this probe always measures the same parallel code that
``2DHolstein1515_speed`` runs in production.
"""

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_ROOT = os.path.dirname(HERE)
PATCH_DIR = os.path.join(
    os.path.dirname(CASE_ROOT), "2DHolstein1515_speed", "multiset_parallel"
)
sys.path.insert(0, CASE_ROOT)
sys.path.insert(0, PATCH_DIR)

from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def _local_rank():
    for key in ("OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "MV2_COMM_WORLD_LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return RANK


def _configure_rank_gpu():
    local_rank = _local_rank()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        count = len([item for item in visible.split(",") if item.strip()])
        gpu_id = 0 if count == 1 else local_rank % max(count, 1)
    else:
        gpu_id = local_rank
    os.environ["RENO_GPU"] = str(gpu_id)
    return gpu_id


# RENO_GPU must be set before renormalizer.mps.backend is imported.
GPU_ID = _configure_rank_gpu()

import mpi_apply_hop_patch as mpi_hop
import mpi_expand_patch as mpi_expand
import mpi_krylov_patch as mpi_krylov

mpi_hop.install_patch()
mpi_expand.install_patch()
mpi_krylov.install_patch()

import common
from renormalizer.utils.log import package_logger as logger


def main():
    tag = os.environ.get(
        "RUN_TAG",
        f"mpi{SIZE}_{common.NROW}x{common.NCOL}_m{common.MAX_BONDDIM}",
    )
    tag_rank = f"{tag}_r{RANK}"
    out_dir = os.environ.get("RESULT_DIR", os.path.join(CASE_ROOT, "results"))

    from renormalizer.mps.backend import USE_GPU
    from renormalizer.multiset import MultisetChargeDiffusionDynamics

    if RANK == 0:
        logger.info(
            "[%s] lattice %dx%d (%d sites), M=%d, nu_max=%d, GPU=%s, ranks=%d, "
            "krylov_mode=%s, nsteps=%d",
            tag, common.NROW, common.NCOL, common.NROW * common.NCOL,
            common.MAX_BONDDIM, common.NU_MAX, USE_GPU, SIZE,
            os.environ.get("RENO_MPI_KRYLOV_MODE", "distributed"), common.NSTEPS,
        )

    plog = common.PhaseLog(tag_rank, logger)
    common.install_phase_probes(plog)
    status, note, extra = "ok", "", {"mpi_size": SIZE, "rank": RANK, "gpu_id": GPU_ID}

    try:
        model = common.build_model()
        plog.mark("build_model")

        job = MultisetChargeDiffusionDynamics(
            model=model,
            max_bonddim=common.MAX_BONDDIM,
            initial_site=common.site_index(
                common.NROW // 2, common.NCOL // 2, common.NCOL
            ),
            stop_at_edge=False,
            if_startup_substeps=False,
            if_rdm=False,
            observables=common.OBSERVABLES,
        )
        plog.mark("job_ready")

        extra.update(common.template_footprint(job.ms_model))
        if common.SKIP_EXPAND:
            # S and W do not depend on bond dimension, so they are exact here.
            # The Krylov width does: every bond is still 1 without the expansion.
            extra["krylov_note"] = "bond dims still 1; krylov width not measured"
        extra.update(common.krylov_vector_gb(job.ms_model, ranks=SIZE))
        if RANK == 0:
            logger.info("[%s] static sizes: %s", tag, extra)

        if common.SETUP_ONLY or common.SKIP_EXPAND:
            extra["steps_completed"] = 0
            note = "footprint only (expansion skipped)" if common.SKIP_EXPAND else "setup only"
        else:
            budget_start = time.perf_counter()
            done = 0
            for step in range(common.NSTEPS):
                job.evolve(evolve_dt=common.EVOLVE_DT, nsteps=1)
                plog.mark(f"step_{step + 1}")
                done = step + 1
                # Every rank must leave the loop together or the next
                # collective would deadlock.
                over = time.perf_counter() - budget_start > common.STEP_BUDGET_S
                if COMM.allreduce(1 if over else 0, op=MPI.MAX):
                    note = f"stopped after {done} steps on the {common.STEP_BUDGET_S:.0f} s budget"
                    if RANK == 0:
                        logger.warning("[%s] %s", tag, note)
                    break
            extra["steps_completed"] = done
            extra["population_sum"] = float(sum(job.e_occupations_array[-1]))

    except Exception as error:  # noqa: BLE001 - the failure mode is the result
        status = type(error).__name__
        note = str(error)[:600]
        logger.error(
            "[%s] FAILED after phase %s: %s",
            tag_rank,
            plog.phases[-1]["phase"] if plog.phases else "start",
            note,
        )
        traceback.print_exc()

    common.summarize(tag_rank, plog, status, note, extra, logger, out_dir)

    # A single rank running out of memory is the answer, so report a failure
    # only after every rank has written its own record.
    failures = COMM.allreduce(0 if status == "ok" else 1, op=MPI.SUM)
    if RANK == 0 and failures:
        logger.error("[%s] %d/%d ranks failed", tag, failures, SIZE)
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        logger.exception("Rank %d died outside the instrumented region.", RANK)
        COMM.Abort(1)
        raise
    sys.exit(code)
