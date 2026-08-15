# -*- coding: utf-8 -*-

"""MPI parallel ``_apply_hop_batched`` patch for the multiset benchmark.

Two distributed Krylov layouts are supported:

``reduce_scatter``
    Compatibility/correctness baseline.  Every Lanczos action first gathers
    the full input vector.  Ranks split Hamiltonian pairs, build a full output,
    and use ``Reduce_scatter`` to return one output slice per rank.

``local``
    Electronic-row decomposition.  A rank owns complete electronic rows of a
    Krylov vector, computes every Hamiltonian pair targeting those rows, and
    exchanges only remote ``beta`` rows needed by those pairs.  No full input
    gather and no output reduction is required inside the action.

The OpenMPI build on Curie is not CUDA-aware, so vector communication is
staged through contiguous NumPy buffers.  GPU-buffer collectives remain an
explicitly unsafe diagnostic option in the launcher, not a production path.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

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
    progress_interval: int
    output_counts: Optional[Tuple[int, ...]] = None
    electron_counts: Optional[Tuple[int, ...]] = None
    electron_displs: Optional[Tuple[int, ...]] = None
    vector_dim: Optional[int] = None
    distributed_mode: Optional[str] = None
    calls: int = 0
    local_pairs: int = 0
    total_pairs: int = 0
    reduce_scatter_calls: int = 0
    allreduce_calls: int = 0
    halo_exchange_calls: int = 0
    halo_rows_sent: int = 0
    halo_rows_received: int = 0
    halo_bytes_sent: int = 0
    halo_bytes_received: int = 0
    max_available_rows: int = 0
    compute_seconds: float = 0.0
    comm_seconds: float = 0.0


_CTX = _MpiHopContext(
    comm=MPI.COMM_WORLD,
    rank=MPI.COMM_WORLD.Get_rank(),
    size=MPI.COMM_WORLD.Get_size(),
    allreduce_mode=os.environ.get("RENO_MPI_ALLREDUCE_MODE", "host").lower(),
    debug_shapes=os.environ.get("RENO_MPI_DEBUG_SHAPES", "0").lower()
    in {"1", "true", "yes"},
    progress_interval=int(os.environ.get("RENO_MPI_HOP_PROGRESS_INTERVAL", "200")),
)

_ORIGINAL_APPLY_HOP_BATCHED = multiset_model.MultisetModel._apply_hop_batched


def _pair_slice(n_pairs):
    """Return this rank's contiguous Hamiltonian-pair slice."""
    base = n_pairs // _CTX.size
    rem = n_pairs % _CTX.size
    start = _CTX.rank * base + min(_CTX.rank, rem)
    stop = start + base + (1 if _CTX.rank < rem else 0)
    return start, stop


def _displacements(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if len(counts) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)


def _to_numpy(array):
    return xp.asnumpy(array) if USE_GPU else np.asarray(array)


def _host_indices(array):
    return np.asarray(_to_numpy(array), dtype=np.int64)


def _mpi_dtype_from_dtype(dtype):
    return MPI._typedict[np.dtype(dtype).char]


def _contract_group(Y_exp, L_all, R_all, W_all, nsite, shape, n_pairs):
    """Apply one already-filtered contraction group."""
    if nsite == 1:
        if len(shape) == 3:
            Y_exp = Y_exp.reshape(n_pairs, shape[0], shape[1], shape[2])
            temp = xp.einsum("ncek,nlfk->ncelf", Y_exp, R_all)
            temp2 = xp.einsum("ncelf,nbdef->ncdlb", temp, W_all)
            out = xp.einsum("ncdlb,nabc->nadl", temp2, L_all)
        elif len(shape) == 4:
            Y_exp = Y_exp.reshape(
                n_pairs, shape[0], shape[1], shape[2], shape[3]
            )
            temp = xp.einsum("ncegk,nlfk->nceglf", Y_exp, R_all)
            temp2 = xp.einsum("nceglf,nbdef->ncglbd", temp, W_all)
            out = xp.einsum("ncglbd,nabc->nadgl", temp2, L_all)
        else:
            raise ValueError(
                f"Unsupported local tensor shape for nsite=1: {shape}"
            )
    else:
        if len(shape) != 2:
            raise ValueError(
                f"Unsupported local tensor shape for nsite=0: {shape}"
            )
        Y_exp = Y_exp.reshape(n_pairs, shape[0], shape[1])
        temp = xp.einsum("nck,nlbk->nclb", Y_exp, R_all)
        out = xp.einsum("nclb,nabc->nal", temp, L_all)
    return out.reshape(n_pairs, -1)


def _reduce_full_output(y_out):
    """Sum pair shards and optionally scatter the complete output vector."""
    y_out = y_out.ravel()

    if _CTX.debug_shapes:
        local_info = (int(y_out.size), str(y_out.dtype))
        all_info = _CTX.comm.allgather(local_info)
        if len(set(all_info)) != 1:
            raise RuntimeError(
                f"MPI _apply_hop_batched reduction shape mismatch: {all_info}"
            )

    if _CTX.output_counts is not None:
        counts = list(_CTX.output_counts)
        if sum(counts) != int(y_out.size):
            raise RuntimeError(
                "MPI _apply_hop_batched Reduce_scatter count mismatch: "
                f"sum(counts)={sum(counts)} y_out.size={int(y_out.size)} "
                f"counts={counts}"
            )
        local_count = counts[_CTX.rank]
        _CTX.reduce_scatter_calls += 1

        if _CTX.allreduce_mode == "cuda":
            recv = xp.empty(local_count, dtype=y_out.dtype)
            _CTX.comm.Reduce_scatter(y_out, recv, recvcounts=counts, op=MPI.SUM)
            return recv

        send = np.ascontiguousarray(_to_numpy(y_out))
        recv = np.empty(local_count, dtype=send.dtype)
        _CTX.comm.Reduce_scatter(send, recv, recvcounts=counts, op=MPI.SUM)
        return xp.asarray(recv) if USE_GPU else recv

    _CTX.allreduce_calls += 1
    if _CTX.allreduce_mode == "cuda":
        _CTX.comm.Allreduce(MPI.IN_PLACE, y_out, op=MPI.SUM)
        return y_out

    send = np.ascontiguousarray(_to_numpy(y_out))
    recv = np.empty_like(send)
    _CTX.comm.Allreduce(send, recv, op=MPI.SUM)
    return xp.asarray(recv) if USE_GPU else recv


def _legacy_pair_sharded_action(self, Y, batched_groups, dim, shape):
    """Correct full-output + reduction baseline."""
    n_electron = self.N_electron
    Y = Y.reshape(n_electron, dim)
    Y_out = xp.zeros((n_electron, dim), dtype=Y.dtype)

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
        selector = slice(start, stop)
        n_pairs = stop - start
        L_all = group["L"][selector]
        R_all = group["R"][selector]
        W_all = group["W"][selector] if group["nsite"] == 1 else None
        S = group["S"][:, selector]
        beta_idx = group["beta_idx"][selector]
        Y_exp = Y[beta_idx]
        out_flat = _contract_group(
            Y_exp,
            L_all,
            R_all,
            W_all,
            group["nsite"],
            shape,
            n_pairs,
        )
        Y_out += xp.matmul(S, out_flat)

    _CTX.compute_seconds += time.perf_counter() - compute_start
    comm_start = time.perf_counter()
    reduced = _reduce_full_output(Y_out)
    _CTX.comm_seconds += time.perf_counter() - comm_start
    _record_action(local_pairs_this_call, total_pairs_this_call)
    return reduced


def _owner_lookup(electron_counts, electron_displs):
    n_electron = sum(electron_counts)
    owners = np.empty(n_electron, dtype=np.int64)
    for owner, (count, start) in enumerate(
        zip(electron_counts, electron_displs)
    ):
        owners[start : start + count] = owner
    return owners


def _selector_for_alpha_range(alpha_idx, start, stop):
    """Prefer a slice for the sorted active-pair layout; fall back safely."""
    if len(alpha_idx) == 0:
        return slice(0, 0), np.empty(0, dtype=np.int64)
    if np.all(alpha_idx[1:] >= alpha_idx[:-1]):
        first = int(np.searchsorted(alpha_idx, start, side="left"))
        last = int(np.searchsorted(alpha_idx, stop, side="left"))
        return slice(first, last), np.arange(first, last, dtype=np.int64)
    indices = np.flatnonzero((start <= alpha_idx) & (alpha_idx < stop))
    return xp.asarray(indices), indices


def _build_local_plan(batched_groups, n_electron):
    counts = tuple(_CTX.electron_counts)
    displs = tuple(_CTX.electron_displs)
    if sum(counts) != n_electron:
        raise RuntimeError(
            "MPI local Afunc electronic layout mismatch: "
            f"sum(electron_counts)={sum(counts)} N_electron={n_electron}"
        )

    cache_key = (counts, displs, _CTX.rank)
    if batched_groups:
        cache = batched_groups[0].setdefault("_mpi_local_plan_cache", {})
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    host_groups = []
    for group in batched_groups:
        alpha_idx = _host_indices(group["alpha_idx"])
        beta_idx = _host_indices(group["beta_idx"])
        if len(alpha_idx) != int(group["n_pairs"]) or len(beta_idx) != len(
            alpha_idx
        ):
            raise RuntimeError("Invalid MPI batched-group index metadata")
        host_groups.append((alpha_idx, beta_idx))

    needed_by_rank = []
    for target_rank in range(_CTX.size):
        alpha_start = displs[target_rank]
        alpha_stop = alpha_start + counts[target_rank]
        needed = set()
        for alpha_idx, beta_idx in host_groups:
            mask = (alpha_start <= alpha_idx) & (alpha_idx < alpha_stop)
            needed.update(int(value) for value in beta_idx[mask])
        needed_by_rank.append(needed)

    rank = _CTX.rank
    alpha_start = displs[rank]
    alpha_stop = alpha_start + counts[rank]
    owners = _owner_lookup(counts, displs)

    send_betas_by_dest = []
    for dest in range(_CTX.size):
        betas = sorted(
            beta
            for beta in needed_by_rank[dest]
            if dest != rank and owners[beta] == rank
        )
        send_betas_by_dest.append(np.asarray(betas, dtype=np.int64))

    recv_betas_by_source = []
    for source in range(_CTX.size):
        betas = sorted(
            beta
            for beta in needed_by_rank[rank]
            if source != rank and owners[beta] == source
        )
        recv_betas_by_source.append(np.asarray(betas, dtype=np.int64))

    recv_order = np.concatenate(
        [values for values in recv_betas_by_source if len(values)], axis=0
    ) if any(len(values) for values in recv_betas_by_source) else np.empty(
        0, dtype=np.int64
    )

    available_lookup = np.full(n_electron, -1, dtype=np.int64)
    available_lookup[alpha_start:alpha_stop] = np.arange(
        counts[rank], dtype=np.int64
    )
    if len(recv_order):
        available_lookup[recv_order] = counts[rank] + np.arange(
            len(recv_order), dtype=np.int64
        )

    group_plans = []
    local_pairs = 0
    for alpha_idx, beta_idx in host_groups:
        selector, host_selector = _selector_for_alpha_range(
            alpha_idx, alpha_start, alpha_stop
        )
        selected_betas = beta_idx[host_selector]
        beta_positions = available_lookup[selected_betas]
        if np.any(beta_positions < 0):
            missing = selected_betas[beta_positions < 0]
            raise RuntimeError(
                f"MPI local Afunc halo plan is missing beta rows {missing.tolist()}"
            )
        group_plans.append(
            {
                "selector": selector,
                "beta_positions": xp.asarray(beta_positions),
                "n_pairs": int(len(host_selector)),
            }
        )
        local_pairs += len(host_selector)

    send_local_positions = [
        values - alpha_start for values in send_betas_by_dest
    ]
    plan = {
        "alpha_start": alpha_start,
        "alpha_stop": alpha_stop,
        "local_alpha_count": counts[rank],
        "send_local_positions": send_local_positions,
        "recv_betas_by_source": recv_betas_by_source,
        "recv_order": recv_order,
        "group_plans": group_plans,
        "local_pairs": int(local_pairs),
        "total_pairs": int(sum(int(group["n_pairs"]) for group in batched_groups)),
    }
    if batched_groups:
        cache[cache_key] = plan
    return plan


def _exchange_halo(Y_local, plan, dim):
    # Include GPU packing and device-to-host staging in communication time.
    comm_start = time.perf_counter()
    send_row_counts = np.asarray(
        [len(values) for values in plan["send_local_positions"]],
        dtype=np.int64,
    )
    recv_row_counts = np.asarray(
        [len(values) for values in plan["recv_betas_by_source"]],
        dtype=np.int64,
    )

    send_positions = (
        np.concatenate(
            [values for values in plan["send_local_positions"] if len(values)],
            axis=0,
        )
        if np.any(send_row_counts)
        else np.empty(0, dtype=np.int64)
    )
    if len(send_positions):
        send_rows = Y_local[xp.asarray(send_positions)]
        send = np.ascontiguousarray(_to_numpy(send_rows).reshape(-1))
    else:
        send = np.empty(0, dtype=np.dtype(Y_local.dtype))

    send_counts = send_row_counts * dim
    recv_counts = recv_row_counts * dim
    send_displs = _displacements(send_counts)
    recv_displs = _displacements(recv_counts)
    recv = np.empty(int(np.sum(recv_counts)), dtype=np.dtype(Y_local.dtype))
    mpi_dtype = _mpi_dtype_from_dtype(Y_local.dtype)

    _CTX.comm.Alltoallv(
        [
            send,
            send_counts.astype(int).tolist(),
            send_displs.astype(int).tolist(),
            mpi_dtype,
        ],
        [
            recv,
            recv_counts.astype(int).tolist(),
            recv_displs.astype(int).tolist(),
            mpi_dtype,
        ],
    )
    if len(plan["recv_order"]):
        recv_rows = xp.asarray(recv.reshape(len(plan["recv_order"]), dim))
        available = xp.concatenate((Y_local, recv_rows), axis=0)
    else:
        available = Y_local
    _CTX.comm_seconds += time.perf_counter() - comm_start

    _CTX.halo_exchange_calls += 1
    _CTX.halo_rows_sent += int(np.sum(send_row_counts))
    _CTX.halo_rows_received += int(np.sum(recv_row_counts))
    _CTX.halo_bytes_sent += int(send.nbytes)
    _CTX.halo_bytes_received += int(recv.nbytes)
    _CTX.max_available_rows = max(_CTX.max_available_rows, int(len(available)))
    return available


def _local_electronic_action(self, Y, batched_groups, dim, shape):
    """Consume a local electronic-row slice and return the matching output."""
    if _CTX.electron_counts is None or _CTX.electron_displs is None:
        raise RuntimeError("MPI local Afunc was entered without an electronic layout")
    if _CTX.vector_dim != dim:
        raise RuntimeError(
            f"MPI local Afunc dim mismatch: layout={_CTX.vector_dim} action={dim}"
        )

    plan = _build_local_plan(batched_groups, self.N_electron)
    local_alpha_count = plan["local_alpha_count"]
    if int(Y.size) != local_alpha_count * dim:
        raise RuntimeError(
            "MPI local Afunc input size mismatch: "
            f"Y.size={int(Y.size)} expected={local_alpha_count * dim}"
        )
    Y_local = Y.reshape(local_alpha_count, dim)
    Y_available = _exchange_halo(Y_local, plan, dim)
    Y_out = xp.zeros((local_alpha_count, dim), dtype=Y.dtype)

    compute_start = time.perf_counter()
    alpha_start = plan["alpha_start"]
    alpha_stop = plan["alpha_stop"]
    for group, group_plan in zip(batched_groups, plan["group_plans"]):
        n_pairs = group_plan["n_pairs"]
        if n_pairs == 0:
            continue
        selector = group_plan["selector"]
        L_all = group["L"][selector]
        R_all = group["R"][selector]
        W_all = group["W"][selector] if group["nsite"] == 1 else None
        Y_exp = Y_available[group_plan["beta_positions"]]
        out_flat = _contract_group(
            Y_exp,
            L_all,
            R_all,
            W_all,
            group["nsite"],
            shape,
            n_pairs,
        )
        S_local = group["S"][alpha_start:alpha_stop, selector]
        Y_out += xp.matmul(S_local, out_flat)

    _CTX.compute_seconds += time.perf_counter() - compute_start
    _record_action(plan["local_pairs"], plan["total_pairs"])
    return Y_out.ravel()


def _record_action(local_pairs, total_pairs):
    _CTX.calls += 1
    _CTX.local_pairs += int(local_pairs)
    _CTX.total_pairs += int(total_pairs)
    if (
        _CTX.progress_interval
        and _CTX.calls % _CTX.progress_interval == 0
        and _CTX.rank == 0
    ):
        logger.info(
            "MPI _apply_hop_batched progress: calls=%d local_pairs=%d "
            "total_pairs=%d reduce_scatter_calls=%d halo_exchange_calls=%d "
            "halo_rows_sent=%d compute_s=%.6f comm_s=%.6f",
            _CTX.calls,
            _CTX.local_pairs,
            _CTX.total_pairs,
            _CTX.reduce_scatter_calls,
            _CTX.halo_exchange_calls,
            _CTX.halo_rows_sent,
            _CTX.compute_seconds,
            _CTX.comm_seconds,
        )


def _apply_hop_batched_mpi(self, Y, batched_groups, dim, shape):
    if _CTX.size == 1:
        return _ORIGINAL_APPLY_HOP_BATCHED(self, Y, batched_groups, dim, shape)
    if _CTX.distributed_mode == "local":
        return _local_electronic_action(self, Y, batched_groups, dim, shape)
    return _legacy_pair_sharded_action(self, Y, batched_groups, dim, shape)


def install_patch():
    multiset_model.MultisetModel._apply_hop_batched = _apply_hop_batched_mpi
    if _CTX.rank == 0:
        logger.info(
            "Installed MPI _apply_hop_batched patch: size=%d transport=%s "
            "USE_GPU=%s",
            _CTX.size,
            _CTX.allreduce_mode,
            USE_GPU,
        )


def set_distributed_layout(
    counts,
    mode,
    electron_counts=None,
    electron_displs=None,
    vector_dim=None,
):
    """Set action layout for the duration of one distributed Krylov solve."""
    if counts is None:
        _CTX.output_counts = None
        _CTX.electron_counts = None
        _CTX.electron_displs = None
        _CTX.vector_dim = None
        _CTX.distributed_mode = None
        return

    _CTX.output_counts = tuple(int(value) for value in counts)
    _CTX.distributed_mode = str(mode)
    _CTX.electron_counts = (
        None
        if electron_counts is None
        else tuple(int(value) for value in electron_counts)
    )
    _CTX.electron_displs = (
        None
        if electron_displs is None
        else tuple(int(value) for value in electron_displs)
    )
    _CTX.vector_dim = None if vector_dim is None else int(vector_dim)


def set_output_counts(counts):
    """Backward-compatible helper for the old Reduce_scatter path."""
    set_distributed_layout(counts, mode="reduce_scatter")


def summarize_patch_usage():
    values = np.array(
        [
            _CTX.calls,
            _CTX.local_pairs,
            _CTX.total_pairs,
            _CTX.reduce_scatter_calls,
            _CTX.allreduce_calls,
            _CTX.halo_exchange_calls,
            _CTX.halo_rows_sent,
            _CTX.halo_rows_received,
            _CTX.halo_bytes_sent,
            _CTX.halo_bytes_received,
            _CTX.max_available_rows,
            _CTX.compute_seconds,
            _CTX.comm_seconds,
        ],
        dtype=np.float64,
    )
    gathered = None
    if _CTX.rank == 0:
        gathered = np.empty((_CTX.size, len(values)), dtype=np.float64)
    _CTX.comm.Gather(values, gathered, root=0)
    if _CTX.rank == 0:
        logger.info(
            "MPI _apply_hop_batched usage by rank "
            "[calls, local_pairs, total_pairs, reduce_scatter_calls, "
            "allreduce_calls, halo_exchange_calls, halo_rows_sent, "
            "halo_rows_received, halo_bytes_sent, halo_bytes_received, "
            "max_available_rows, compute_s, comm_s]: %s",
            gathered.tolist(),
        )


def rank():
    return _CTX.rank


def size():
    return _CTX.size


def comm():
    return _CTX.comm
