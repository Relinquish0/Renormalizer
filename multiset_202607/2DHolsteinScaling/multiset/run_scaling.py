# -*- coding: utf-8 -*-

"""Single-GPU multiset size-scaling probe: 10 steps, pass or fail.

Reports wall time and peak host/GPU memory for every expensive phase, so a
failure can be attributed to a specific step rather than to "the run died".
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# renormalizer.mps.backend enables the GPU whenever CuPy imports and reads
# RENO_GPU at import time, so it has to be set before renormalizer is loaded.
os.environ.setdefault("RENO_GPU", "0")

import common
from renormalizer.utils.log import package_logger as logger


def main():
    tag = os.environ.get(
        "RUN_TAG", f"single_{common.NROW}x{common.NCOL}_m{common.MAX_BONDDIM}"
    )
    out_dir = os.environ.get(
        "RESULT_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
    )

    from renormalizer.mps.backend import USE_GPU
    from renormalizer.multiset import MultisetChargeDiffusionDynamics

    logger.info(
        "[%s] lattice %dx%d (%d sites), M=%d, nu_max=%d, GPU=%s, nsteps=%d",
        tag, common.NROW, common.NCOL, common.NROW * common.NCOL,
        common.MAX_BONDDIM, common.NU_MAX, USE_GPU, common.NSTEPS,
    )

    plog = common.PhaseLog(tag, logger)
    common.install_phase_probes(plog)
    status, note, extra = "ok", "", {}

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
        extra.update(common.krylov_vector_gb(job.ms_model, ranks=1))
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
                if time.perf_counter() - budget_start > common.STEP_BUDGET_S:
                    note = f"stopped after {done} steps on the {common.STEP_BUDGET_S:.0f} s budget"
                    logger.warning("[%s] %s", tag, note)
                    break
            extra["steps_completed"] = done
            extra["population_sum"] = float(sum(job.e_occupations_array[-1]))

    except Exception as error:  # noqa: BLE001 - the failure mode is the result
        status = type(error).__name__
        note = str(error)[:600]
        logger.error("[%s] FAILED during phase after %s: %s", tag,
                     plog.phases[-1]["phase"] if plog.phases else "start", note)
        traceback.print_exc()

    common.summarize(tag, plog, status, note, extra, logger, out_dir)
    return 0 if status == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
