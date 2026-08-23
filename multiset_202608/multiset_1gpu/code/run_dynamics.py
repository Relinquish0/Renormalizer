# -*- coding: utf-8 -*-

"""Single-GPU multiset MPS driver for the 2D Holstein size ladder.

This is the one-process counterpart of ``multiset_4gpu/code/run_dynamics.py``
and follows ``multiset_202607/2DHolstein1515_speed/multiset/Holstein1515.py``:
same model, same observables, same environment-variable configuration.  The
model builder in ``holstein.py`` is a verbatim copy of the 4-GPU one, so the two
ladders differ only in how many devices they run on.

What is added here is the memory accounting.  The point of the single-GPU ladder
is to find where it stops and why, and "why" needs the state, the environments
and the device pool reported separately -- see ``memory.py``.

``RENO_GPU`` has to be set before ``renormalizer.mps.backend`` is imported, so
it is resolved at the top of this module rather than inside ``main``.
"""

import json
import os
import time

# The backend picks its device at import time from RENO_GPU.  Under SLURM,
# CUDA_VISIBLE_DEVICES already narrows the process to its one allocated device,
# so ordinal 0 is that device.  RENO_NO_GPU=1 forces the NumPy backend, which is
# how the host-side probe runs on a CPU node.
if os.environ.get("RENO_NO_GPU", "0").strip().lower() not in {"0", "false", "no", ""}:
    os.environ.pop("RENO_GPU", None)
else:
    os.environ.setdefault("RENO_GPU", "0")

import numpy as np

from renormalizer.mps.backend import USE_GPU
from renormalizer.multiset import MultisetChargeDiffusionDynamics
from renormalizer.utils.log import package_logger as logger

import memory
from holstein import LIGHTWEIGHT_OBSERVABLES, HolsteinConfig, build_model

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
RUN_LABEL = os.environ.get("RUN_LABEL", "")

# SETUP_ONLY stops the run early, so the host-side ceiling can be measured
# without a GPU and without paying for a full TDVP sweep:
#
#   0  full run
#   1  stop after the model, MPOs and the initial state are built
#   2  also build the per-pair environment cache before stopping
#
# Level 1 alone understates the footprint: conj_mps is built lazily, so the
# environment cache is not allocated until the first sweep.
_SETUP_ONLY_RAW = os.environ.get("SETUP_ONLY", "0").strip().lower()
if _SETUP_ONLY_RAW in {"2", "environ", "environment"}:
    SETUP_ONLY, PROBE_ENVIRON = True, True
else:
    SETUP_ONLY = _SETUP_ONLY_RAW not in {"0", "false", "no", "", "off"}
    PROBE_ENVIRON = False

IF_STARTUP_SUBSTEPS = os.environ.get("IF_STARTUP_SUBSTEPS", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "",
}
STARTUP_SUBSTEPS_N = int(os.environ.get("STARTUP_SUBSTEPS_N", "10"))


def _report(memory_log, phase, ms_mps=None, ms_model=None):
    record = memory.snapshot(phase, ms_mps=ms_mps, ms_model=ms_model)
    memory_log.append(record)
    logger.info(memory.format_snapshot(record))
    return record


def main():
    cfg = HolsteinConfig()
    memory_log = []

    logger.info("=" * 72)
    logger.info(
        "single-GPU multiset: lattice %s (%d sites), m=%d, nu_max=%d, dt=%s, steps=%d",
        cfg.tag,
        cfg.n_sites,
        cfg.max_bonddim,
        cfg.nu_max,
        cfg.evolve_dt,
        cfg.nsteps,
    )
    logger.info(
        "omega0=%s, J=%s, g=%s, initial site=%d, GPU=%s",
        cfg.omega_0,
        cfg.j,
        cfg.g,
        cfg.initial_site,
        USE_GPU,
    )
    logger.info("output dir: %s", os.path.abspath(OUTPUT_DIR))
    logger.info("=" * 72)

    t_model = time.perf_counter()
    model = build_model(cfg)
    t_model = time.perf_counter() - t_model

    t_setup = time.perf_counter()
    job = MultisetChargeDiffusionDynamics(
        model=model,
        max_bonddim=cfg.max_bonddim,
        initial_site=cfg.initial_site,
        stop_at_edge=False,
        if_startup_substeps=IF_STARTUP_SUBSTEPS,
        startup_substeps_n=STARTUP_SUBSTEPS_N,
        if_rdm=False,
        observables=LIGHTWEIGHT_OBSERVABLES,
    )
    t_setup = time.perf_counter() - t_setup
    _report(memory_log, "after-setup", ms_mps=job.latest_mps, ms_model=job.ms_model)
    logger.info(
        "model build %.1f s, multiset setup + init + expand %.1f s", t_model, t_setup
    )

    t_environ = 0.0
    if SETUP_ONLY and PROBE_ENVIRON:
        t_environ = time.perf_counter()
        # The same entry point the sweep uses, so this is the cache the
        # evolution would really hold, not an approximation of it.
        job.ms_model._get_or_build_environ_list(job.latest_mps)
        t_environ = time.perf_counter() - t_environ
        _report(
            memory_log, "after-environ", ms_mps=job.latest_mps, ms_model=job.ms_model
        )
        logger.info("environment cache built in %.1f s", t_environ)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SETUP_ONLY:
        stem = os.path.join(OUTPUT_DIR, f"{cfg.tag}{RUN_LABEL}_setup")
        with open(stem + "_summary.json", "w") as handle:
            json.dump(
                {
                    "config": cfg.as_dict(),
                    "n_gpu": 1,
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
    t_evolve = time.perf_counter() - t_evolve
    _report(memory_log, "after-evolve", ms_mps=job.latest_mps, ms_model=job.ms_model)

    populations = np.array(job.e_occupations_array)
    times = np.array(job.evolve_times)
    logger.info(
        "evolve wall time %.1f s for %d steps (%.1f s/step)",
        t_evolve,
        cfg.nsteps,
        t_evolve / max(cfg.nsteps, 1),
    )
    logger.info("final population sum: %.12f", populations[-1].sum())
    logger.info("final population max: %.6e", populations[-1].max())

    stem = os.path.join(OUTPUT_DIR, f"{cfg.tag}{RUN_LABEL}")
    np.savez(
        stem + ".npz",
        time_series=times,
        e_occupations=populations,
        wall_time_seconds=np.array(t_evolve),
        setup_seconds=np.array(t_setup),
        model_seconds=np.array(t_model),
        n_gpu=np.array(1),
    )
    with open(stem + "_summary.json", "w") as handle:
        json.dump(
            {
                "config": cfg.as_dict(),
                "n_gpu": 1,
                "model_seconds": t_model,
                "setup_seconds": t_setup,
                "evolve_seconds": t_evolve,
                "seconds_per_step": t_evolve / max(cfg.nsteps, 1),
                "memory": memory_log,
                "status": "ok",
            },
            handle,
            indent=2,
        )
    logger.info("wrote %s.npz and %s_summary.json", stem, stem)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("single-GPU multiset run failed")
        raise
