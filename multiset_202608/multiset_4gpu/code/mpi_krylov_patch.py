# -*- coding: utf-8 -*-

"""MPI-synchronized Krylov solvers for the multiset benchmark.

``distributed`` preserves the original correctness baseline: Lanczos basis
vectors are sharded, but every matrix action gathers its full input and the
Hamiltonian action returns a slice through ``Reduce_scatter``.

``local`` keeps complete electronic rows on each rank and passes those local
rows directly to the Hamiltonian action.  Only sparse electronic halo rows
are exchanged by :mod:`mpi_apply_hop_patch`; a full vector is gathered once,
when a Krylov solve returns to the unchanged TDVP caller.
"""

import os
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

import renormalizer.lib as reno_lib
from renormalizer.lib.krylov import krylov as krylov_mod
from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.multiset import multiset_model
from renormalizer.utils.log import package_logger as logger

import mpi_apply_hop_patch as mpi_hop


@dataclass
class _MpiKrylovContext:
    comm: MPI.Comm
    rank: int
    size: int
    mode: str
    calls: int = 0
    distributed_calls: int = 0
    local_slice_calls: int = 0
    missing_metadata_fallbacks: int = 0
    gathered_input_vectors: int = 0
    gathered_result_vectors: int = 0
    beta_any_returns: int = 0
    converged_all_returns: int = 0
    full_space_returns: int = 0
    max_global_len: int = 0
    max_local_len: int = 0
    max_basis_rows: int = 0


_CTX = _MpiKrylovContext(
    comm=MPI.COMM_WORLD,
    rank=MPI.COMM_WORLD.Get_rank(),
    size=MPI.COMM_WORLD.Get_size(),
    mode=os.environ.get("RENO_MPI_KRYLOV_MODE", "distributed").lower(),
)


def _all_true(flag):
    if _CTX.size == 1:
        return bool(flag)
    return bool(_CTX.comm.allreduce(1 if flag else 0, op=MPI.MIN))


def _any_true(flag):
    if _CTX.size == 1:
        return bool(flag)
    return bool(_CTX.comm.allreduce(1 if flag else 0, op=MPI.MAX))


def _check_common_length(v_len):
    if _CTX.size == 1:
        return
    lengths = _CTX.comm.allgather(int(v_len))
    if len(set(lengths)) != 1:
        raise RuntimeError(
            "MPI Krylov ranks entered expm_krylov with different vector lengths: "
            f"{lengths}"
        )


def _partition_counts(n):
    base = n // _CTX.size
    rem = n % _CTX.size
    counts = np.array(
        [base + (1 if rank < rem else 0) for rank in range(_CTX.size)],
        dtype=np.int64,
    )
    displs = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
    return counts, displs


def _partition_vector(n, vector_block_count=None):
    """Partition a vector, preserving logical blocks when metadata is given."""
    if vector_block_count is None:
        counts, displs = _partition_counts(n)
        return counts, displs, None, None, None

    vector_block_count = int(vector_block_count)
    if vector_block_count < 1:
        raise ValueError("vector_block_count must be positive")
    if n % vector_block_count != 0:
        raise ValueError(
            "Krylov vector length is not divisible by vector_block_count: "
            f"length={n} blocks={vector_block_count}"
        )
    vector_dim = n // vector_block_count
    electron_counts, electron_displs = _partition_counts(vector_block_count)
    counts = electron_counts * vector_dim
    displs = electron_displs * vector_dim
    return counts, displs, electron_counts, electron_displs, vector_dim


def _local_bounds(n, vector_block_count=None):
    layout = _partition_vector(n, vector_block_count)
    counts, displs = layout[:2]
    start = int(displs[_CTX.rank])
    stop = start + int(counts[_CTX.rank])
    return layout + (start, stop)


def _to_numpy(array):
    if USE_GPU:
        return xp.asnumpy(array)
    return np.asarray(array)


def _mpi_dtype(array):
    return MPI._typedict[np.dtype(array.dtype).char]


def _allgatherv_xp(local, counts, displs):
    allreduce_mode = os.environ.get("RENO_MPI_ALLREDUCE_MODE", "host").lower()
    if USE_GPU and allreduce_mode == "cuda":
        local_gpu = xp.ascontiguousarray(local)
        full_gpu = xp.empty(int(np.sum(counts)), dtype=local_gpu.dtype)
        _CTX.comm.Allgatherv(
            local_gpu,
            [
                full_gpu,
                counts.astype(int).tolist(),
                displs.astype(int).tolist(),
                _mpi_dtype(full_gpu),
            ],
        )
        _CTX.gathered_input_vectors += 1
        return full_gpu

    local_np = np.ascontiguousarray(_to_numpy(local))
    full_np = np.empty(int(np.sum(counts)), dtype=local_np.dtype)
    _CTX.comm.Allgatherv(
        local_np,
        [full_np, counts.astype(int).tolist(), displs.astype(int).tolist(), _mpi_dtype(full_np)],
    )
    _CTX.gathered_input_vectors += 1
    return xp.asarray(full_np)


def _gather_result_xp(local, counts, displs):
    full = _allgatherv_xp(local, counts, displs)
    _CTX.gathered_input_vectors -= 1
    _CTX.gathered_result_vectors += 1
    return full


def _distributed_norm(local):
    local_sq = float(xp.vdot(local, local).real)
    return float(np.sqrt(_CTX.comm.allreduce(local_sq, op=MPI.SUM)))


def _distributed_vdot_real(left_local, right_local):
    local_value = float(xp.vdot(left_local, right_local).real)
    return float(_CTX.comm.allreduce(local_value, op=MPI.SUM))


def _expm_krylov_coeff(alpha, beta, nrmv, dt):
    try:
        w_hess, u_hess = krylov_mod.eigh_tridiagonal(alpha, beta)
    except np.linalg.LinAlgError:
        logger.warning(
            "Tridiagonal diagonalization in MPI Krylov solver failed, size:%d.",
            len(alpha),
        )
        h = np.diag(alpha) + np.diag(beta, k=-1) + np.diag(beta, k=1)
        w_hess, u_hess = np.linalg.eigh(h)
    return u_hess @ (nrmv * np.exp(dt * w_hess) * u_hess[0])


def _expm_krylov_local(alpha, beta, V_local, nrmv, dt):
    coeff = xp.asarray(_expm_krylov_coeff(alpha, beta, nrmv, dt))
    return V_local[: len(coeff)].T @ coeff


def expm_krylov_sync(
    Afunc,
    dt,
    vstart: xp.ndarray,
    block_size=50,
    vector_block_count=None,
):
    """MPI-safe version of renormalizer.lib.krylov.krylov.expm_krylov."""
    if not np.iscomplex(dt):
        dt = dt.real

    vstart = xp.asarray(vstart)
    _check_common_length(len(vstart))
    nrmv = float(xp.linalg.norm(vstart))
    assert nrmv > 0
    vstart = vstart / nrmv

    alpha = np.zeros(block_size)
    beta = np.zeros(block_size - 1)

    V = xp.empty((block_size, len(vstart)), dtype=vstart.dtype)
    V[0] = vstart
    res = None
    _CTX.calls += 1
    _CTX.max_global_len = max(_CTX.max_global_len, int(len(vstart)))
    _CTX.max_local_len = max(_CTX.max_local_len, int(len(vstart)))
    _CTX.max_basis_rows = max(_CTX.max_basis_rows, int(len(V)))

    for j in range(len(vstart)):
        w = Afunc(V[j])
        alpha[j] = xp.vdot(w, V[j]).real

        if j == len(vstart) - 1:
            _CTX.full_space_returns += 1
            return (
                krylov_mod._expm_krylov(alpha[: j + 1], beta[:j], V[: j + 1, :].T, nrmv, dt),
                j + 1,
            )

        if len(V) == j + 1:
            V, old_V = xp.empty((len(V) + block_size, len(vstart)), dtype=vstart.dtype), V
            V[: len(old_V)] = old_V
            del old_V
            alpha = np.concatenate([alpha, np.zeros(block_size)])
            beta = np.concatenate([beta, np.zeros(block_size)])
            _CTX.max_basis_rows = max(_CTX.max_basis_rows, int(len(V)))

        w -= alpha[j] * V[j] + (beta[j - 1] * V[j - 1] if j > 0 else 0)
        beta[j] = xp.linalg.norm(w)

        local_beta_small = beta[j] < 100 * len(vstart) * np.finfo(float).eps
        if _any_true(local_beta_small):
            _CTX.beta_any_returns += 1
            return (
                krylov_mod._expm_krylov(alpha[: j + 1], beta[:j], V[: j + 1, :].T, nrmv, dt),
                j + 1,
            )

        if 3 < j and j % 2 == 0:
            new_res = krylov_mod._expm_krylov(alpha[: j + 1], beta[:j], V[: j + 1].T, nrmv, dt)
            local_converged = res is not None and xp.allclose(new_res, res, rtol=1e-8, atol=1e-10)
            if _all_true(local_converged):
                _CTX.converged_all_returns += 1
                return new_res, j + 1
            res = new_res

        V[j + 1] = w / beta[j]


def expm_krylov_distributed(
    Afunc,
    dt,
    vstart: xp.ndarray,
    block_size=50,
    vector_block_count=None,
    local_afunc=False,
    local_rows=None,
):
    """Krylov solver with Lanczos basis vectors sharded across MPI ranks.

    In compatibility mode Afunc consumes a gathered full vector and returns a
    local slice.  With ``local_afunc=True`` it consumes and returns only the
    complete electronic rows owned by this rank.

    ``local_rows`` marks the sharded-MPS case: ``vstart`` already holds only this
    rank's electronic rows, so there is nothing to slice off the front and the
    result stays local instead of being gathered.  Passing it is what keeps the
    solver from ever materialising a full N_electron vector.
    """
    if not np.iscomplex(dt):
        dt = dt.real

    vstart = xp.asarray(vstart)
    vstart_is_local = local_rows is not None
    if vstart_is_local:
        if vector_block_count is None:
            raise ValueError(
                "MPI local-vstart Krylov mode requires vector_block_count metadata"
            )
        local_rows = int(local_rows)
        if local_rows <= 0 or len(vstart) % local_rows != 0:
            raise ValueError(
                "Local Krylov vector length is not divisible by local_rows: "
                f"length={len(vstart)} local_rows={local_rows}"
            )
        vector_dim = len(vstart) // local_rows
        global_len = int(vector_block_count) * vector_dim
        electron_counts, electron_displs = _partition_counts(int(vector_block_count))
        if int(electron_counts[_CTX.rank]) != local_rows:
            raise RuntimeError(
                "Krylov alpha partition disagrees with the sharded MPS layout: "
                f"krylov={int(electron_counts[_CTX.rank])} mps={local_rows}"
            )
        counts = electron_counts * vector_dim
        displs = electron_displs * vector_dim
    else:
        global_len = len(vstart)
        _check_common_length(global_len)
        (
            counts,
            displs,
            electron_counts,
            electron_displs,
            vector_dim,
            start,
            stop,
        ) = _local_bounds(global_len, vector_block_count)
    if local_afunc and electron_counts is None:
        raise ValueError(
            "MPI local Krylov mode requires vector_block_count metadata"
        )
    local_count = int(counts[_CTX.rank])

    if vstart_is_local:
        vstart_local = vstart.copy()
    else:
        vstart_local = vstart[start:stop].copy()
    del vstart

    nrmv = _distributed_norm(vstart_local)
    assert nrmv > 0
    vstart_local = vstart_local / nrmv

    alpha = np.zeros(block_size)
    beta = np.zeros(block_size - 1)
    V_local = xp.empty((block_size, local_count), dtype=vstart_local.dtype)
    V_local[0] = vstart_local
    res_local = None

    _CTX.calls += 1
    _CTX.distributed_calls += 1
    if local_afunc:
        _CTX.local_slice_calls += 1
    _CTX.max_global_len = max(_CTX.max_global_len, int(global_len))
    _CTX.max_local_len = max(_CTX.max_local_len, int(local_count))
    _CTX.max_basis_rows = max(_CTX.max_basis_rows, int(len(V_local)))
    mpi_hop.set_distributed_layout(
        counts,
        mode="local" if local_afunc else "reduce_scatter",
        electron_counts=electron_counts if local_afunc else None,
        electron_displs=electron_displs if local_afunc else None,
        vector_dim=vector_dim if local_afunc else None,
    )

    def _finish(local_result):
        # A caller that handed us local rows gets local rows back; the gather is
        # only there for the replicated-MPS callers that still expect a full
        # vector from the solver.
        if vstart_is_local:
            return local_result
        return _gather_result_xp(local_result, counts, displs)

    try:
        for j in range(global_len):
            if local_afunc:
                w_local = Afunc(V_local[j])
            else:
                v_full = _allgatherv_xp(V_local[j], counts, displs)
                w_local = Afunc(v_full)
                del v_full
            if len(w_local) != local_count:
                raise RuntimeError(
                    "Distributed Krylov expected local Afunc output length "
                    f"{local_count}, got {len(w_local)} on rank {_CTX.rank}."
                )

            alpha[j] = _distributed_vdot_real(w_local, V_local[j])

            if j == global_len - 1:
                _CTX.full_space_returns += 1
                result_local = _expm_krylov_local(
                    alpha[: j + 1], beta[:j], V_local[: j + 1], nrmv, dt
                )
                return _finish(result_local), j + 1

            if len(V_local) == j + 1:
                new_V = xp.empty((len(V_local) + block_size, local_count), dtype=V_local.dtype)
                new_V[: len(V_local)] = V_local
                V_local = new_V
                alpha = np.concatenate([alpha, np.zeros(block_size)])
                beta = np.concatenate([beta, np.zeros(block_size)])
                _CTX.max_basis_rows = max(_CTX.max_basis_rows, int(len(V_local)))

            w_local -= alpha[j] * V_local[j] + (beta[j - 1] * V_local[j - 1] if j > 0 else 0)
            beta[j] = _distributed_norm(w_local)

            if beta[j] < 100 * global_len * np.finfo(float).eps:
                _CTX.beta_any_returns += 1
                result_local = _expm_krylov_local(
                    alpha[: j + 1], beta[:j], V_local[: j + 1], nrmv, dt
                )
                return _finish(result_local), j + 1

            if 3 < j and j % 2 == 0:
                new_res_local = _expm_krylov_local(
                    alpha[: j + 1], beta[:j], V_local[: j + 1], nrmv, dt
                )
                local_converged = (
                    res_local is not None
                    and bool(xp.allclose(new_res_local, res_local, rtol=1e-8, atol=1e-10))
                )
                if _all_true(local_converged):
                    _CTX.converged_all_returns += 1
                    return _finish(new_res_local), j + 1
                res_local = new_res_local

            V_local[j + 1] = w_local / beta[j]
    finally:
        mpi_hop.set_distributed_layout(None, mode="none")


def expm_krylov_mpi(
    Afunc,
    dt,
    vstart: xp.ndarray,
    block_size=50,
    vector_block_count=None,
):
    if vector_block_count is None:
        vector_block_count = getattr(Afunc, "_reno_vector_block_count", None)
    local_rows = getattr(Afunc, "_reno_local_rows", None)
    if _CTX.size > 1 and local_rows is not None:
        # Sharded-MPS path: vstart already carries only this rank's alpha rows.
        return expm_krylov_distributed(
            Afunc,
            dt,
            vstart,
            block_size=block_size,
            vector_block_count=vector_block_count,
            local_afunc=True,
            local_rows=local_rows,
        )
    if _CTX.size > 1 and _CTX.mode in {"local", "halo", "local_slice"}:
        if vector_block_count is None:
            _CTX.missing_metadata_fallbacks += 1
            if _CTX.rank == 0 and _CTX.missing_metadata_fallbacks == 1:
                logger.warning(
                    "MPI local Krylov received an untagged Afunc; falling back "
                    "to replicated synchronized Krylov for compatibility."
                )
            return expm_krylov_sync(
                Afunc,
                dt,
                vstart,
                block_size=block_size,
            )
        return expm_krylov_distributed(
            Afunc,
            dt,
            vstart,
            block_size=block_size,
            vector_block_count=vector_block_count,
            local_afunc=True,
        )
    if _CTX.size > 1 and _CTX.mode in {"distributed", "dist", "sharded"}:
        return expm_krylov_distributed(
            Afunc,
            dt,
            vstart,
            block_size=block_size,
            vector_block_count=vector_block_count,
            local_afunc=False,
        )
    return expm_krylov_sync(
        Afunc,
        dt,
        vstart,
        block_size=block_size,
        vector_block_count=vector_block_count,
    )


def install_patch():
    krylov_mod.expm_krylov = expm_krylov_mpi
    reno_lib.expm_krylov = expm_krylov_mpi
    multiset_model.expm_krylov = expm_krylov_mpi
    if _CTX.rank == 0:
        logger.info("Installed MPI Krylov patch: size=%d mode=%s", _CTX.size, _CTX.mode)


def summarize_patch_usage():
    values = np.array(
        [
            _CTX.calls,
            _CTX.distributed_calls,
            _CTX.local_slice_calls,
            _CTX.missing_metadata_fallbacks,
            _CTX.gathered_input_vectors,
            _CTX.gathered_result_vectors,
            _CTX.beta_any_returns,
            _CTX.converged_all_returns,
            _CTX.full_space_returns,
            _CTX.max_global_len,
            _CTX.max_local_len,
            _CTX.max_basis_rows,
        ],
        dtype=np.int64,
    )
    gathered = None
    if _CTX.rank == 0:
        gathered = np.empty((_CTX.size, 12), dtype=np.int64)
    _CTX.comm.Gather(values, gathered, root=0)
    if _CTX.rank == 0:
        logger.info(
            "MPI Krylov usage by rank "
            "[calls, distributed_calls, local_slice_calls, "
            "missing_metadata_fallbacks, gathered_input_vectors, "
            "gathered_result_vectors, "
            "beta_any_returns, converged_all_returns, full_space_returns, "
            "max_global_len, max_local_len, max_basis_rows]: %s",
            gathered.tolist(),
        )
