# -*- coding: utf-8 -*-

"""Per-rank Krylov footprint under the production MPI patch.

Companion to ``krylov_probe.py``.  Same fake diagonal Hamiltonian, same vector
lengths, but the solver is the patched ``expm_krylov_mpi`` from
``2DHolstein1515_speed/multiset_parallel``.  This measures whether the Lanczos
basis really is sharded 1/P, without the Hamiltonian contraction, the scatter
matrices or the halo exchange mixed into the number.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = os.path.join(
    os.path.dirname(HERE), "2DHolstein1515_speed", "multiset_parallel"
)
sys.path.insert(0, HERE)
sys.path.insert(0, PATCH_DIR)

from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()


def _local_rank():
    for key in ("OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return RANK


visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible:
    count = len([item for item in visible.split(",") if item.strip()])
    os.environ["RENO_GPU"] = str(0 if count == 1 else _local_rank() % max(count, 1))
else:
    os.environ["RENO_GPU"] = str(_local_rank())

import mpi_krylov_patch as mpi_krylov

mpi_krylov.install_patch()

from renormalizer.lib import expm_krylov
from renormalizer.mps.backend import USE_GPU, xp

GB = 1024.0**3
M = int(os.environ.get("MAX_BONDDIM", "64"))
D = int(os.environ.get("NU_MAX", "8"))
BLOCK = int(os.environ.get("BLOCK_SIZE", "50"))
SIZES = [int(v) for v in os.environ.get("SIZES", "15 20 25 28").split()]
SPREAD = float(os.environ.get("SPREAD", "50.0"))


def pool_peak():
    if not USE_GPU:
        return 0.0
    return xp.get_default_memory_pool().total_bytes() / GB


def free_pool():
    if USE_GPU:
        xp.get_default_memory_pool().free_all_blocks()


def probe(n_lattice):
    n_e = n_lattice * n_lattice
    dim = M * D * M
    n = n_e * dim
    _, _, _, _, _, start, stop = mpi_krylov._local_bounds(n, n_e)

    free_pool()
    diag_local = xp.asarray(np.linspace(-SPREAD, SPREAD, n)[start:stop])
    v0 = xp.asarray(np.random.default_rng(0).standard_normal(n).astype(np.complex128))
    free_pool()
    base = pool_peak()

    # In "local" mode the patched solver hands Afunc only this rank's slice.
    afunc = lambda v: diag_local * v
    afunc._reno_vector_block_count = n_e

    out, iters = expm_krylov(afunc, -1j * 0.05, v0, block_size=BLOCK)
    peak = pool_peak()
    del out, diag_local, v0
    free_pool()

    return {
        "lattice": f"{n_lattice}x{n_lattice}",
        "ranks": NPROC,
        "n_electron": n_e,
        "vector_len": n,
        "local_len": int(stop - start),
        "local_fraction": round((stop - start) / n, 4),
        "analytic_local_basis_gb": round(BLOCK * (stop - start) * 16 / GB, 3),
        "analytic_full_basis_gb": round(BLOCK * n * 16 / GB, 3),
        "lanczos_iters": int(iters),
        "peak_before_gb": round(base, 3),
        "measured_peak_gb": round(peak, 3),
    }


def main():
    if RANK == 0:
        print(f"ranks={NPROC} mode={mpi_krylov._CTX.mode} M={M} d={D} block={BLOCK}")
    rows = []
    for n_lattice in SIZES:
        try:
            row = probe(n_lattice)
        except Exception as error:  # noqa: BLE001 - OOM is a result
            row = {"lattice": f"{n_lattice}x{n_lattice}", "ranks": NPROC,
                   "status": type(error).__name__, "note": str(error)[:200]}
            print(f"[rank {RANK}] {json.dumps(row)}", flush=True)
            rows.append(row)
            break
        print(f"[rank {RANK}] {json.dumps(row)}", flush=True)
        rows.append(row)

    out_dir = os.environ.get("RESULT_DIR", os.path.join(HERE, "results"))
    os.makedirs(out_dir, exist_ok=True)
    tag = os.environ.get("RUN_TAG", "krylov_probe_mpi")
    with open(os.path.join(out_dir, f"{tag}_r{RANK}.json"), "w", encoding="utf-8") as s:
        json.dump(rows, s, indent=2)
        s.write("\n")


if __name__ == "__main__":
    main()
