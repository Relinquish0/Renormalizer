# -*- coding: utf-8 -*-
"""How much of a TDVP step is spent moving Environ tensors across PCIe.

`Environ.write` calls `asnumpy` (GPU->host) and `Environ.read` calls `asxp`
(host->GPU) -- renormalizer/mps/lib.py:112-116.  The multiset sweep reads L and R
for *every* active pair at *every* site (multiset_model.py:530-531) and rewrites
them through `GetLR(method="System")`.  This probe counts and times those calls.
"""
import os, sys, json, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RENO_GPU", "0")

import common
from renormalizer.utils.log import package_logger as logger

STATS = {"read_calls": 0, "read_bytes": 0, "read_s": 0.0,
         "write_calls": 0, "write_bytes": 0, "write_s": 0.0}


def install_counters():
    from renormalizer.mps.lib import Environ
    import cupy
    orig_read, orig_write = Environ.read, Environ.write

    def read(self, domain, siteidx):
        t = time.perf_counter()
        out = orig_read(self, domain, siteidx)
        cupy.cuda.runtime.deviceSynchronize()
        STATS["read_s"] += time.perf_counter() - t
        STATS["read_calls"] += 1
        STATS["read_bytes"] += self._virtual_disk[(domain, siteidx)].nbytes
        return out

    def write(self, domain, siteidx, tensor):
        t = time.perf_counter()
        orig_write(self, domain, siteidx, tensor)
        cupy.cuda.runtime.deviceSynchronize()
        STATS["write_s"] += time.perf_counter() - t
        STATS["write_calls"] += 1
        STATS["write_bytes"] += self._virtual_disk[(domain, siteidx)].nbytes
        return None

    Environ.read, Environ.write = read, write


def main():
    from renormalizer.multiset import MultisetChargeDiffusionDynamics
    tag = os.environ.get("RUN_TAG", f"envtime_{common.NROW}x{common.NCOL}")
    model = common.build_model()
    job = MultisetChargeDiffusionDynamics(
        model=model, max_bonddim=common.MAX_BONDDIM,
        initial_site=common.site_index(common.NROW // 2, common.NCOL // 2, common.NCOL),
        stop_at_edge=False, if_startup_substeps=False, if_rdm=False,
        observables=common.OBSERVABLES)

    install_counters()          # only count the evolution, not the setup sweeps
    for k in STATS:
        STATS[k] = 0 if not k.endswith("_s") else 0.0

    t0 = time.perf_counter()
    job.evolve(evolve_dt=common.EVOLVE_DT, nsteps=1)
    step_s = time.perf_counter() - t0

    out = {"tag": tag, "nrow": common.NROW, "ncol": common.NCOL,
           "n_sites": common.NROW * common.NCOL, "max_bonddim": common.MAX_BONDDIM,
           "step_seconds": round(step_s, 2),
           "host_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2, 3)}
    out.update({k: (round(v, 2) if k.endswith("_s") else v) for k, v in STATS.items()})
    out["pcie_gib_moved"] = round((STATS["read_bytes"] + STATS["write_bytes"]) / 1024.0**3, 2)
    out["pcie_seconds"] = round(STATS["read_s"] + STATS["write_s"], 2)
    out["pcie_fraction_of_step"] = round(out["pcie_seconds"] / step_s, 3)
    out["effective_gib_per_s"] = round(out["pcie_gib_moved"] / max(out["pcie_seconds"], 1e-9), 2)

    logger.info("[%s] ENVTIME %s", tag, json.dumps(out))
    d = os.environ.get("RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
