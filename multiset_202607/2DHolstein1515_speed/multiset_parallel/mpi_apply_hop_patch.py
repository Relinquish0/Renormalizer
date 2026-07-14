# -*- coding: utf-8 -*-

"""MPI parallel _apply_hop_batched patch for this benchmark case.

This module intentionally monkey-patches Renormalizer at runtime so the
Renormalizer package itself stays unchanged.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple
import time

import numpy as np
from mpi4py import MPI

from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.multiset import multiset_model
from renormalizer.utils.log import package_logger as logger


@dataclass
class _MpiHopContext:
    comm: MPI.Comm
    rank: int
    size: int
    allreduce_mode: str
    debug_shapes: bool
    output_counts: Optional[Tuple[int, ...]] = None
    calls: int = 0
    local_pairs: int = 0
    total_pairs: int = 0
    reduce_scatter_calls: int = 0
    local_segment_calls: int = 0
    compute_seconds: float = 0.0
    comm_seconds: float = 0.0
    progress_interval: int = 0


_CTX = _MpiHopContext(
    comm=MPI.COMM_WORLD,
    rank=MPI.COMM_WORLD.Get_rank(),
    size=MPI.COMM_WORLD.Get_size(),
    allreduce_mode=os.environ.get("RENO_MPI_ALLREDUCE_MODE", "host").lower(),
    debug_shapes=os.environ.get("RENO_MPI_DEBUG_SHAPES", "0").lower() in {"1", "true", "yes"},
    progress_interval=int(os.environ.get("RENO_MPI_HOP_PROGRESS_INTERVAL", "200")),
)

_ORIGINAL_APPLY_HOP_BATCHED = multiset_model.MultisetModel._apply_hop_batched


def _pair_slice(n_pairs):
    """Return this rank's contiguous slice of a pair batch."""
    base = n_pairs // _CTX.size
    rem = n_pairs % _CTX.size
    start = _CTX.rank * base + min(_CTX.rank, rem)
    stop = start + base + (1 if _CTX.rank < rem else 0)
    return start, stop


def _allreduce_y_out(y_out):
    if _CTX.size == 1:
        return y_out

    y_out = y_out.ravel()

    if _CTX.debug_shapes:
        local_info = (int(y_out.size), str(y_out.dtype), tuple(int(v) for v in y_out.shape))
        all_info = _CTX.comm.allgather(local_info)
        if len(set(all_info)) != 1:
            raise RuntimeError(f"MPI _apply_hop_batched Allreduce shape mismatch: {all_info}")

    if _CTX.output_counts is not None:
        counts = list(_CTX.output_counts)
        if sum(counts) != int(y_out.size):
            raise RuntimeError(
                "MPI _apply_hop_batched Reduce_scatter count mismatch: "
                f"sum(counts)={sum(counts)} y_out.size={int(y_out.size)} counts={counts}"
            )
        local_count = counts[_CTX.rank]
        _CTX.reduce_scatter_calls += 1

        if _CTX.allreduce_mode == "cuda":
            recv = xp.empty(local_count, dtype=y_out.dtype)
            _CTX.comm.Reduce_scatter(y_out, recv, recvcounts=counts, op=MPI.SUM)
            return recv

        if USE_GPU:
            send = xp.asnumpy(y_out)
            recv = np.empty(local_count, dtype=send.dtype)
            _CTX.comm.Reduce_scatter(send, recv, recvcounts=counts, op=MPI.SUM)
            return xp.asarray(recv)

        recv = np.empty(local_count, dtype=y_out.dtype)
        _CTX.comm.Reduce_scatter(y_out, recv, recvcounts=counts, op=MPI.SUM)
        return recv

    if _CTX.allreduce_mode == "cuda":
        _CTX.comm.Allreduce(MPI.IN_PLACE, y_out, op=MPI.SUM)
        return y_out

    if USE_GPU:
        send = xp.asnumpy(y_out)
        recv = np.empty_like(send)
        _CTX.comm.Allreduce(send, recv, op=MPI.SUM)
        return xp.asarray(recv)

    recv = np.empty_like(y_out)
    _CTX.comm.Allreduce(y_out, recv, op=MPI.SUM)
    return recv


def _allreduce_local_segment(local_out):
    local_out = local_out.ravel()
    if _CTX.output_counts is None:
        return _allreduce_y_out(local_out)

    counts = list(_CTX.output_counts)
    local_count = counts[_CTX.rank]
    max_count = max(counts)
    _CTX.local_segment_calls += 1

    if int(local_out.size) != local_count:
        raise RuntimeError(
            "MPI _apply_hop_batched local segment size mismatch: "
            f"local_out.size={int(local_out.size)} local_count={local_count}"
        )

    if _CTX.allreduce_mode == "cuda":
        send = local_out
        if local_count != max_count:
            padded = xp.zeros(max_count, dtype=local_out.dtype)
            padded[:local_count] = local_out
            send = padded
        recv = xp.empty(max_count, dtype=local_out.dtype)
        _CTX.comm.Allreduce(send, recv, op=MPI.SUM)
        return recv[:local_count]

    send_np = xp.asnumpy(local_out) if USE_GPU else np.asarray(local_out)
    if local_count != max_count:
        padded_np = np.zeros(max_count, dtype=send_np.dtype)
        padded_np[:local_count] = send_np
        send_np = padded_np
    recv_np = np.empty(max_count, dtype=send_np.dtype)
    _CTX.comm.Allreduce(send_np, recv_np, op=MPI.SUM)
    recv_np = recv_np[:local_count]
    return xp.asarray(recv_np) if USE_GPU else recv_np


def _local_output_bounds(dim):
    if _CTX.output_counts is None:
        return None
    counts = list(_CTX.output_counts)
    start = sum(counts[:_CTX.rank])
    stop = start + counts[_CTX.rank]
    row_start = start // dim
    row_stop = (stop + dim - 1) // dim
    offset = start - row_start * dim
    return start, stop, row_start, row_stop, offset


def _apply_hop_batched_mpi(self, Y, batched_groups, dim, shape):
    if _CTX.size == 1:
        return _ORIGINAL_APPLY_HOP_BATCHED(self, Y, batched_groups, dim, shape)

    n_electron = self.N_electron
    Y = Y.reshape(n_electron, dim)
    output_bounds = _local_output_bounds(dim)
    if output_bounds is None:
        Y_out = xp.zeros((n_electron, dim), dtype=Y.dtype)
        row_start = 0
        row_stop = n_electron
        local_offset = 0
        local_count = n_electron * dim
    else:
        _, _, row_start, row_stop, local_offset = output_bounds
        local_count = _CTX.output_counts[_CTX.rank]
        Y_out = xp.zeros((row_stop - row_start, dim), dtype=Y.dtype)

    local_pairs_this_call = 0
    total_pairs_this_call = 0
    compute_start = time.perf_counter()

    for group in batched_groups:
        n_pairs_total = int(group["n_pairs"])
        total_pairs_this_call += n_pairs_total
        start, stop = _pair_slice(n_pairs_total)
        if start == stop:
            continue

        local_pairs_this_call += stop - start

        L_all = group["L"][start:stop]
        R_all = group["R"][start:stop]
        S = group["S"][:, start:stop]
        beta_idx = group["beta_idx"][start:stop]
        nsite = group["nsite"]
        n_pairs = stop - start

        Y_exp = Y[beta_idx]

        if nsite == 1:
            W_all = group["W"][start:stop]
            if len(shape) == 3:
                Y_exp = Y_exp.reshape(n_pairs, shape[0], shape[1], shape[2])
                temp = xp.einsum("ncek,nlfk->ncelf", Y_exp, R_all)
                temp2 = xp.einsum("ncelf,nbdef->ncdlb", temp, W_all)
                out = xp.einsum("ncdlb,nabc->nadl", temp2, L_all)
            elif len(shape) == 4:
                Y_exp = Y_exp.reshape(n_pairs, shape[0], shape[1], shape[2], shape[3])
                temp = xp.einsum("ncegk,nlfk->nceglf", Y_exp, R_all)
                temp2 = xp.einsum("nceglf,nbdef->ncglbd", temp, W_all)
                out = xp.einsum("ncglbd,nabc->nadgl", temp2, L_all)
            else:
                raise ValueError(f"Unsupported local tensor shape for nsite=1: {shape}")
        else:
            if len(shape) != 2:
                raise ValueError(f"Unsupported local tensor shape for nsite=0: {shape}")
            Y_exp = Y_exp.reshape(n_pairs, shape[0], shape[1])
            temp = xp.einsum("nck,nlbk->nclb", Y_exp, R_all)
            out = xp.einsum("nclb,nabc->nal", temp, L_all)

        out_flat = out.reshape(n_pairs, dim)
        if output_bounds is None:
            Y_out += xp.matmul(S, out_flat)
        else:
            Y_out += xp.matmul(S[row_start:row_stop], out_flat)

    _CTX.calls += 1
    _CTX.local_pairs += local_pairs_this_call
    _CTX.total_pairs += total_pairs_this_call
    _CTX.compute_seconds += time.perf_counter() - compute_start

    if output_bounds is None:
        reduce_input = Y_out
    else:
        reduce_input = Y_out.ravel()[local_offset : local_offset + local_count]

    comm_start = time.perf_counter()
    reduced = _allreduce_local_segment(reduce_input)
    _CTX.comm_seconds += time.perf_counter() - comm_start

    if _CTX.progress_interval and _CTX.calls % _CTX.progress_interval == 0 and _CTX.rank == 0:
        logger.info(
            "MPI _apply_hop_batched progress: calls=%d local_pairs=%d total_pairs=%d "
            "local_segment_calls=%d compute_s=%.6f comm_s=%.6f",
            _CTX.calls,
            _CTX.local_pairs,
            _CTX.total_pairs,
            _CTX.local_segment_calls,
            _CTX.compute_seconds,
            _CTX.comm_seconds,
        )
    return reduced


def install_patch():
    multiset_model.MultisetModel._apply_hop_batched = _apply_hop_batched_mpi
    if _CTX.rank == 0:
        logger.info(
            "Installed MPI _apply_hop_batched patch: size=%d mode=%s USE_GPU=%s",
            _CTX.size,
            _CTX.allreduce_mode,
            USE_GPU,
        )


def summarize_patch_usage():
    values = np.array(
        [
            _CTX.calls,
            _CTX.local_pairs,
            _CTX.total_pairs,
            _CTX.reduce_scatter_calls,
            _CTX.local_segment_calls,
            _CTX.compute_seconds,
            _CTX.comm_seconds,
        ],
        dtype=np.float64,
    )
    gathered = None
    if _CTX.rank == 0:
        gathered = np.empty((_CTX.size, 7), dtype=np.float64)
    _CTX.comm.Gather(values, gathered, root=0)
    if _CTX.rank == 0:
        logger.info(
            "MPI _apply_hop_batched usage by rank "
            "[calls, local_pairs, total_pairs, reduce_scatter_calls, "
            "local_segment_calls, compute_s, comm_s]: %s",
            gathered.tolist(),
        )


def set_output_counts(counts):
    _CTX.output_counts = None if counts is None else tuple(int(v) for v in counts)


def rank():
    return _CTX.rank


def size():
    return _CTX.size


def comm():
    return _CTX.comm
