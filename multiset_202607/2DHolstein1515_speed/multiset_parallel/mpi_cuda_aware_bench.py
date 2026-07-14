# -*- coding: utf-8 -*-

"""Small MPI/CuPy communication benchmark for Curie GPU nodes."""

import os
import time

import numpy as np
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


def _stats(values):
    arr = np.array(values, dtype=np.float64)
    return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))


def _time_host(cp, n, dtype, niter):
    send_gpu = cp.ones(n, dtype=dtype) * (RANK + 1)
    recv_gpu = cp.empty_like(send_gpu)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(niter):
        COMM.Barrier()
        start = time.perf_counter()
        send_host = cp.asnumpy(send_gpu)
        recv_host = np.empty_like(send_host)
        COMM.Allreduce(send_host, recv_host, op=MPI.SUM)
        recv_gpu.set(recv_host)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - start)
    return times, float(cp.asnumpy(recv_gpu[:1])[0])


def _time_cuda_allreduce(cp, n, dtype, niter):
    send_gpu = cp.ones(n, dtype=dtype) * (RANK + 1)
    recv_gpu = cp.empty_like(send_gpu)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(niter):
        COMM.Barrier()
        start = time.perf_counter()
        COMM.Allreduce(send_gpu, recv_gpu, op=MPI.SUM)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - start)
    return times, float(cp.asnumpy(recv_gpu[:1])[0])


def _time_cuda_allgatherv(cp, n, dtype, niter):
    base = n // SIZE
    rem = n % SIZE
    counts = np.array([base + (1 if rank < rem else 0) for rank in range(SIZE)], dtype=np.int64)
    displs = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
    local_n = int(counts[RANK])
    send_gpu = cp.ones(local_n, dtype=dtype) * (RANK + 1)
    recv_gpu = cp.empty(n, dtype=dtype)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(niter):
        COMM.Barrier()
        start = time.perf_counter()
        COMM.Allgatherv(
            send_gpu,
            [recv_gpu, counts.astype(int).tolist(), displs.astype(int).tolist(), MPI._typedict[np.dtype(dtype).char]],
        )
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - start)
    return times, float(cp.asnumpy(recv_gpu[:1])[0])


def _report(name, nbytes, times, value, error=None):
    local = np.array([min(times), sum(times) / len(times), max(times)], dtype=np.float64) if times else np.zeros(3)
    gathered = COMM.gather(local, root=0)
    errors = COMM.gather(error, root=0)
    if RANK != 0:
        return
    if any(errors):
        print(f"{name} bytes={nbytes} status=FAILED errors={errors}", flush=True)
        return
    mins, means, maxs = zip(*gathered)
    print(
        f"{name} bytes={nbytes} value0={value:.6g} "
        f"rank_min_s={_stats(mins)} rank_mean_s={_stats(means)} rank_max_s={_stats(maxs)}",
        flush=True,
    )


def main():
    import cupy as cp

    gpu_id = _local_rank()
    cp.cuda.Device(gpu_id).use()
    dtype = np.complex128
    niter = int(os.environ.get("MPI_CUDA_BENCH_ITERS", "5"))
    sizes = [int(item) for item in os.environ.get("MPI_CUDA_BENCH_SIZES", "262144,1048576,4194304").split(",")]

    if RANK == 0:
        print(f"MPI size={SIZE} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
        print(f"dtype={np.dtype(dtype)} niter={niter} sizes={sizes}", flush=True)

    for n in sizes:
        nbytes = n * np.dtype(dtype).itemsize
        times, value = _time_host(cp, n, dtype, niter)
        _report("HOST_ALLREDUCE", nbytes, times, value)

        try:
            times, value = _time_cuda_allreduce(cp, n, dtype, niter)
            _report("CUDA_ALLREDUCE", nbytes, times, value)
        except Exception as exc:
            _report("CUDA_ALLREDUCE", nbytes, [], 0.0, error=repr(exc))

        try:
            times, value = _time_cuda_allgatherv(cp, n, dtype, niter)
            _report("CUDA_ALLGATHERV", nbytes, times, value)
        except Exception as exc:
            _report("CUDA_ALLGATHERV", nbytes, [], 0.0, error=repr(exc))


if __name__ == "__main__":
    main()
