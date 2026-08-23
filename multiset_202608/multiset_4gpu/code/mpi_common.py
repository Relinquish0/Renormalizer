# -*- coding: utf-8 -*-

"""Rank/GPU bootstrap and the alpha-ownership convention shared by every patch.

Import this *before* anything that pulls in ``renormalizer.mps.backend``:
``configure_rank_gpu()`` sets ``RENO_GPU``, which the backend reads at import.

The alpha (electronic-row) partition defined here is the single global
convention used by the hop, Krylov, expansion and sharding patches.  It is the
same contiguous split ``mpi_krylov_patch._partition_counts`` produces, so a
Krylov vector sliced by rank and an alpha row owned by rank always agree.
"""

import os
import resource

import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def local_rank():
    for key in (
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return RANK


def configure_rank_gpu():
    """Pin this rank to one GPU by setting ``RENO_GPU`` before the backend loads.

    ``RENO_NO_GPU=1`` skips the pinning entirely, which is how the CPU-only
    host-memory probe keeps the backend on NumPy without each rank first failing
    to open a device that is not there.
    """
    if os.environ.get("RENO_NO_GPU", "0").strip().lower() not in {"0", "false", "no", ""}:
        os.environ.pop("RENO_GPU", None)
        return None
    lrank = local_rank()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_count = len([item for item in visible.split(",") if item.strip()])
        gpu_id = 0 if visible_count <= 1 else lrank % visible_count
    else:
        gpu_id = lrank
    os.environ["RENO_GPU"] = str(gpu_id)
    return gpu_id


# --------------------------------------------------------------------------- #
# alpha partition
# --------------------------------------------------------------------------- #
def partition_counts(n, size=None):
    size = SIZE if size is None else size
    base, rem = divmod(n, size)
    counts = np.array(
        [base + (1 if r < rem else 0) for r in range(size)], dtype=np.int64
    )
    displs = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
    return counts, displs


class AlphaLayout:
    """Contiguous alpha ranges, one per rank, plus the owner lookup."""

    def __init__(self, n_electron, size=None, rank=None):
        self.size = SIZE if size is None else size
        self.rank = RANK if rank is None else rank
        self.n_electron = int(n_electron)
        self.counts, self.displs = partition_counts(self.n_electron, self.size)
        self.start = int(self.displs[self.rank])
        self.stop = self.start + int(self.counts[self.rank])
        self.owners = np.empty(self.n_electron, dtype=np.int64)
        for owner, (count, first) in enumerate(zip(self.counts, self.displs)):
            self.owners[first : first + count] = owner

    def owns(self, alpha):
        return self.start <= alpha < self.stop

    @property
    def owned(self):
        return range(self.start, self.stop)

    def owner_of(self, alpha):
        return int(self.owners[alpha])

    def __repr__(self):
        return (
            f"AlphaLayout(rank={self.rank}/{self.size}, "
            f"alpha=[{self.start},{self.stop}) of {self.n_electron})"
        )


# --------------------------------------------------------------------------- #
# memory probes
# --------------------------------------------------------------------------- #
def host_peak_gb():
    """Peak resident set size of this rank, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)


def gpu_report():
    """(pool_used, pool_total, device_used, device_total) in GiB, or zeros."""
    try:
        import cupy
    except ImportError:
        return (0.0, 0.0, 0.0, 0.0)
    gib = 1024.0**3
    pool = cupy.get_default_memory_pool()
    try:
        free, total = cupy.cuda.runtime.memGetInfo()
        dev_used, dev_total = (total - free) / gib, total / gib
    except Exception:
        dev_used = dev_total = 0.0
    return (pool.used_bytes() / gib, pool.total_bytes() / gib, dev_used, dev_total)


def gather_max(value):
    """Max of a scalar across ranks, available on every rank."""
    if SIZE == 1:
        return float(value)
    return float(COMM.allreduce(float(value), op=MPI.MAX))


def gather_all(value):
    if SIZE == 1:
        return [float(value)]
    return [float(item) for item in COMM.allgather(float(value))]
