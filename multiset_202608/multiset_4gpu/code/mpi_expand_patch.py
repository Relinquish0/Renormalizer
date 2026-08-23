# -*- coding: utf-8 -*-

"""MPI parallel initial bond-dimension expansion for this benchmark.

This runtime patch keeps the Renormalizer package unchanged. It shards the
outer electron-state loop in MultisetModel.expand_bond_dimension_multiset
across MPI ranks, then broadcasts expanded MPS components one at a time back
to every rank before TDVP evolution starts.
"""

import time
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from renormalizer.mps.lib import _sum
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.multiset import multiset_model
from renormalizer.utils.log import package_logger as logger


@dataclass
class _MpiExpandContext:
    comm: MPI.Comm
    rank: int
    size: int
    calls: int = 0
    local_alphas: int = 0
    total_alphas: int = 0
    elapsed_seconds: float = 0.0
    gather_seconds: float = 0.0


_CTX = _MpiExpandContext(
    comm=MPI.COMM_WORLD,
    rank=MPI.COMM_WORLD.Get_rank(),
    size=MPI.COMM_WORLD.Get_size(),
)

_ORIGINAL_EXPAND_MULTISET = multiset_model.MultisetModel.expand_bond_dimension_multiset


def _owned_alphas(n_electron):
    return list(range(_CTX.rank, n_electron, _CTX.size))


def _expand_alpha_with_hint(model, alpha, original_mps, coef):
    mps_alpha = original_mps[alpha]
    mps_alpha.compress_config = model.compress_config

    diag_mpo = model.MsMpo.msmpo[alpha][alpha]
    hint_mpo = diag_mpo if len(diag_mpo) > 0 else None

    cross_states = []
    for pair_id in model._active_pairs_by_alpha[alpha]:
        beta = model._active_pairs_index[pair_id][1]
        if beta == alpha:
            continue
        driven = model._active_pair_mpos[pair_id].apply(original_mps[beta])
        cross_states.append(driven)

    ex_mps = _sum(cross_states, compress=False) if cross_states else None
    if ex_mps is not None:
        ex_mps.compress_config = model.compress_config

    return expand_bond_dimension_general(
        mps_alpha,
        hint_mpo=hint_mpo,
        coef=coef,
        ex_mps=ex_mps,
    )


def _expand_alpha_no_hint(model, alpha, coef):
    return expand_bond_dimension_general(
        model.MsMps.msmps[alpha],
        hint_mpo=None,
        coef=coef,
        ex_mps=None,
    )


def _finalize_expanded(expanded):
    expanded.scale(float(abs(expanded.coeff)), inplace=True)
    expanded.coeff = 1.0
    return expanded


def _expand_bond_dimension_multiset_mpi(self, coef: float = 1e-10, use_hint: bool = True):
    if _CTX.size == 1:
        return _ORIGINAL_EXPAND_MULTISET(self, coef=coef, use_hint=use_hint)

    if getattr(self.MsMps, "electronic_ancilla", False):
        if _CTX.rank == 0:
            logger.warning(
                "MPI expand patch does not shard electronic-ancilla states; "
                "falling back to original expand_bond_dimension_multiset."
            )
        return _ORIGINAL_EXPAND_MULTISET(self, coef=coef, use_hint=use_hint)

    start_time = time.perf_counter()
    n_electron = self.N_electron
    local_alphas = _owned_alphas(n_electron)

    for alpha in range(n_electron):
        self.MsMps.msmps[alpha].compress_config = self.compress_config

    original_mps = None
    if use_hint:
        original_mps = [self.MsMps.msmps[beta].copy() for beta in range(n_electron)]

    for alpha in local_alphas:
        if use_hint:
            expanded = _expand_alpha_with_hint(self, alpha, original_mps, coef)
        else:
            expanded = _expand_alpha_no_hint(self, alpha, coef)
        self.MsMps.msmps[alpha] = _finalize_expanded(expanded)

    gather_start = time.perf_counter()
    for alpha in range(n_electron):
        owner = alpha % _CTX.size
        payload = self.MsMps.msmps[alpha] if _CTX.rank == owner else None
        self.MsMps.msmps[alpha] = _CTX.comm.bcast(payload, root=owner)
    gather_elapsed = time.perf_counter() - gather_start

    del original_mps

    elapsed = time.perf_counter() - start_time
    _CTX.calls += 1
    _CTX.local_alphas += len(local_alphas)
    _CTX.total_alphas += n_electron
    _CTX.elapsed_seconds += elapsed
    _CTX.gather_seconds += gather_elapsed

    local_stats = np.array(
        [len(local_alphas), elapsed, gather_elapsed],
        dtype=np.float64,
    )
    gathered_stats = None
    if _CTX.rank == 0:
        gathered_stats = np.empty((_CTX.size, 3), dtype=np.float64)
    _CTX.comm.Gather(local_stats, gathered_stats, root=0)

    if _CTX.rank == 0:
        logger.info(
            "MPI expand_bond_dimension_multiset completed: use_hint=%s coef=%s "
            "stats_by_rank [local_alphas, elapsed_s, gather_s]=%s",
            use_hint,
            coef,
            gathered_stats.tolist(),
        )


def install_patch():
    multiset_model.MultisetModel.expand_bond_dimension_multiset = _expand_bond_dimension_multiset_mpi
    if _CTX.rank == 0:
        logger.info(
            "Installed MPI expand_bond_dimension_multiset patch: size=%d",
            _CTX.size,
        )


def summarize_patch_usage():
    values = np.array(
        [_CTX.calls, _CTX.local_alphas, _CTX.total_alphas, _CTX.elapsed_seconds, _CTX.gather_seconds],
        dtype=np.float64,
    )
    gathered = None
    if _CTX.rank == 0:
        gathered = np.empty((_CTX.size, 5), dtype=np.float64)
    _CTX.comm.Gather(values, gathered, root=0)
    if _CTX.rank == 0:
        logger.info(
            "MPI expand usage by rank "
            "[calls, local_alphas, total_alphas, elapsed_s, gather_s]: %s",
            gathered.tolist(),
        )
