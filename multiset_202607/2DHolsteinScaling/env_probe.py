# -*- coding: utf-8 -*-
"""Measure how much host RAM the per-pair Environ cache holds after one step.

`Environ.write` stores every L/R tensor with `asnumpy` (renormalizer/mps/lib.py:113),
so one numpy tensor per (pair, domain, site) lives on the host.  This probe sums
those bytes and compares them with the host RSS the process actually grew by.
"""
import os, sys, json, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RENO_GPU", "0")

import common
from renormalizer.utils.log import package_logger as logger


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


def environ_bytes(ms_model):
    cache = ms_model._environ_cache
    if cache is None:
        return None
    total = 0
    tensors = 0
    shapes = {}
    for env in cache["envs"]:
        for key, arr in env._virtual_disk.items():
            total += arr.nbytes
            tensors += 1
            shapes[arr.shape] = shapes.get(arr.shape, 0) + arr.nbytes
    top = sorted(shapes.items(), key=lambda kv: -kv[1])[:4]
    return {
        "n_environ_objects": len(cache["envs"]),
        "n_tensors": tensors,
        "environ_host_gb": round(total / 1024.0**3, 3),
        "top_shapes_gb": [[list(s), round(b / 1024.0**3, 3)] for s, b in top],
    }


def mps_bytes(ms_model):
    """Host bytes held by the multiset MPS itself (N_e MPSs of L sites)."""
    total = 0
    tensors = 0
    for mps in ms_model.MsMps.msmps:
        for i in range(len(mps)):
            arr = mps[i].array
            total += getattr(arr, "nbytes", 0)
            tensors += 1
    return {"n_mps": len(ms_model.MsMps.msmps), "n_mps_tensors": tensors,
            "mps_host_gb": round(total / 1024.0**3, 3)}


def main():
    from renormalizer.multiset import MultisetChargeDiffusionDynamics
    tag = os.environ.get("RUN_TAG", f"envprobe_{common.NROW}x{common.NCOL}")
    out = {"tag": tag, "nrow": common.NROW, "ncol": common.NCOL,
           "n_sites": common.NROW * common.NCOL, "max_bonddim": common.MAX_BONDDIM}

    model = common.build_model()
    job = MultisetChargeDiffusionDynamics(
        model=model, max_bonddim=common.MAX_BONDDIM,
        initial_site=common.site_index(common.NROW // 2, common.NCOL // 2, common.NCOL),
        stop_at_edge=False, if_startup_substeps=False, if_rdm=False,
        observables=common.OBSERVABLES)
    out["host_after_setup_gb"] = round(rss(), 3)
    out.update(common.template_footprint(job.ms_model))

    job.evolve(evolve_dt=common.EVOLVE_DT, nsteps=1)
    out["host_after_step1_gb"] = round(rss(), 3)
    out["host_step1_growth_gb"] = round(out["host_after_step1_gb"] - out["host_after_setup_gb"], 3)
    out.update(environ_bytes(job.ms_model) or {})
    out.update(mps_bytes(job.ms_model))
    if out.get("environ_host_gb") is not None:
        out["environ_plus_mps_gb"] = round(out["environ_host_gb"] + out["mps_host_gb"], 3)
    if out.get("environ_host_gb") is not None and out["host_step1_growth_gb"] > 0:
        out["environ_fraction_of_growth"] = round(
            out["environ_host_gb"] / out["host_step1_growth_gb"], 3)

    logger.info("[%s] ENVPROBE %s", tag, json.dumps(out))
    d = os.environ.get("RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
