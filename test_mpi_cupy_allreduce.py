from mpi4py import MPI
import cupy as cp
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

x = cp.ones(8, dtype=cp.float32) * (rank + 1)

# 如果 MPI CUDA-aware，这句应该直接工作（GPU buffer）
try:
    y = cp.empty_like(x)
    comm.Allreduce(x, y, op=MPI.SUM)
    cp.cuda.Stream.null.synchronize()
    if rank == 0:
        print("GPU Allreduce OK. y =", y.get())
except Exception as e:
    if rank == 0:
        print("GPU Allreduce FAILED:", repr(e))