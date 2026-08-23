# -*- coding: utf-8 -*-

"""Alpha-row sharding of the multiset TDVP program.

The multiset ansatz keeps one MPS per electronic site ``alpha``, and every
object the program builds is indexed by the ``(alpha, beta)`` Hamiltonian
blocks.  ``_active_pairs_by_alpha`` is already an exact, non-overlapping
partition of those blocks by ``alpha``, which makes ``alpha`` the natural axis
to distribute over.  Four cumulative stages are selected by
``RENO_MS_SHARD_STAGE``:

1. Replicated.  The only parallelism is the Krylov/hop decomposition provided
   by :mod:`mpi_krylov_patch` and :mod:`mpi_apply_hop_patch`; host memory does
   not shard at all, so four ranks reach a *smaller* lattice than one.
2. Shard the Hamiltonian pair list by ``alpha``.  ``Environ``, the batched
   contraction templates and the reverse templates all follow the pair list, so
   this shards the single largest host term with one change.
3. Shard the state.  A rank keeps the ``alpha`` rows it owns plus the ghost
   rows its own pairs reach into (``alpha +/- 1`` and ``alpha +/- NCOL`` for a
   2D lattice), and exchanges one site tensor per ghost row per sweep step.
4. Shard the setup.  ``Model``/``Mpo`` blocks of non-owned ``alpha`` rows are
   never built, so construction time and memory shard as well.

Install before the ``MultisetModel`` is constructed: stage 4 has to intercept
``_ConstructMsMpo``, and every stage has to intercept the pair grouping that
``MultisetModel.__init__`` runs on its last line.
"""

import os
from typing import Dict, List, Optional

import numpy as np
from mpi4py import MPI

from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.mps.lib import _sum
from renormalizer.mps.matrix import asnumpy, asxp, tensordot
from renormalizer.mps.mpo import Mpo
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.multiset import multiset_model, multiset_mpo, multiset_mps
from renormalizer.multiset.multiset_model import KRYLOV_BLOCK_SIZE, MultisetModel
from renormalizer.multiset.multiset_mpo import MultisetMpo
from renormalizer.multiset.multiset_mps import (
    MultisetMps,
    _state_expectation,
    _state_inner_product,
)
from renormalizer.utils import CompressCriteria
from renormalizer.utils.log import package_logger as logger

from mpi_common import COMM, RANK, SIZE, AlphaLayout

STAGE = int(os.environ.get("RENO_MS_SHARD_STAGE", "1"))

# Tag base for the per-site ghost exchange.  Sites are tagged individually so a
# late message can never be matched against the wrong sweep position.
_TAG_SITE = 7000
_TAG_CHAIN = 11000

_ORIGINAL: Dict[str, object] = {}
_LAYOUT: Optional[AlphaLayout] = None


def _ensure_layout(n_electron) -> AlphaLayout:
    global _LAYOUT
    if _LAYOUT is None or _LAYOUT.n_electron != int(n_electron):
        _LAYOUT = AlphaLayout(int(n_electron))
    return _LAYOUT


def layout() -> Optional[AlphaLayout]:
    return _LAYOUT


def active() -> bool:
    """True when this run actually distributes anything."""
    return SIZE > 1 and STAGE >= 2


def state_is_sharded() -> bool:
    return SIZE > 1 and STAGE >= 3


# --------------------------------------------------------------------------- #
# stage 2 / 4: pair-list and setup sharding
# --------------------------------------------------------------------------- #
def _construct_msmpo_sharded(self):
    """Build MPO blocks only for the alpha rows this rank owns (stage 4)."""
    lay = _ensure_layout(self.N_electron)
    for i in range(self.N_electron):
        owned = lay.owns(i)
        for j in range(self.N_electron):
            block = self.MsModel[i][j]
            if not owned or len(block.ham_terms) == 0:
                self.msmpo[i][j] = []
            else:
                self.msmpo[i][j] = Mpo(model=block, terms=None)


def _construct_msmodel_sharded(self):
    """Build ``Model`` blocks only for the alpha rows this rank owns (stage 4)."""
    lay = _ensure_layout(self.N_electron)
    from renormalizer.model.model import Model

    for i in range(self.N_electron):
        owned = lay.owns(i)
        for j in range(self.N_electron):
            if not owned or len(self.MsOp[i][j]) == 0:
                self.MsModel[i][j] = multiset_model.EMPTY_MS_BLOCK
            else:
                self.MsModel[i][j] = Model(
                    basis=self.basis_set, ham_terms=self.MsOp[i][j]
                )
        if not owned:
            # The symbolic operators of a foreign row are dead weight once its
            # Model is not going to be built.
            self.MsOp[i] = [[] for _ in range(self.N_electron)]


def _active_mpo_select_grouping_sharded(self):
    """``MultisetModel._active_mpo_select_grouping`` restricted to owned alpha.

    Everything downstream is keyed off ``_active_pairs_index`` /
    ``_site_group_templates``, so restricting the pair list here is what makes
    ``Environ``, the batched ``W`` stacks and the reverse templates shard.
    """
    lay = _ensure_layout(self.N_electron)
    self._environ_cache = None
    active_pairs = []
    active_pair_mpos = []
    active_pairs_by_alpha = [[] for _ in range(self.N_electron)]
    mpo_group_dicts = None

    for alpha in lay.owned:
        for beta in range(self.N_electron):
            mpo = self.MsMpo.msmpo[alpha][beta]
            if len(mpo) == 0:
                continue
            pair_id = len(active_pairs)
            active_pairs.append((alpha, beta))
            active_pair_mpos.append(mpo)
            active_pairs_by_alpha[alpha].append(pair_id)

            if mpo_group_dicts is None:
                mpo_group_dicts = [dict() for _ in range(len(mpo))]

            for imps, local_mpo in enumerate(mpo):
                W = asxp(local_mpo.array)
                key = tuple(W.shape)
                group = mpo_group_dicts[imps].setdefault(
                    key,
                    {
                        "pair_ids": [],
                        "alpha_idx": [],
                        "beta_idx": [],
                        "w_tensors": [],
                    },
                )
                group["pair_ids"].append(pair_id)
                group["alpha_idx"].append(alpha)
                group["beta_idx"].append(beta)
                group["w_tensors"].append(W)

    self._active_pairs_index = active_pairs
    self._active_pair_mpos = active_pair_mpos
    self._active_pairs_by_alpha = active_pairs_by_alpha
    self._site_group_templates = []
    self._ghost_sync_plan = None
    # Tell ``mpi_apply_hop_patch`` that the pair groups it receives are already
    # an alpha-disjoint shard, so it must not slice them a second time.
    self._alpha_sharded_groups = True

    if mpo_group_dicts is None:
        raise RuntimeError(
            f"rank {RANK} owns alpha rows {lay.start}:{lay.stop} but no "
            "Hamiltonian blocks; the alpha partition is inconsistent."
        )

    for site_groups in mpo_group_dicts:
        templates = []
        for key in sorted(site_groups):
            group = site_groups[key]
            templates.append(
                {
                    "pair_ids": tuple(group["pair_ids"]),
                    "alpha_idx": xp.asarray(group["alpha_idx"], dtype=np.int64),
                    "beta_idx": xp.asarray(group["beta_idx"], dtype=np.int64),
                    "W": xp.stack(group["w_tensors"]),
                    "nsite": 1,
                    "n_pairs": len(group["pair_ids"]),
                }
            )
        self._site_group_templates.append(templates)


# --------------------------------------------------------------------------- #
# residency and the ghost exchange plan
# --------------------------------------------------------------------------- #
def resident_alphas(model) -> List[int]:
    """Owned alpha rows plus the beta rows this rank's own pairs read."""
    lay = _ensure_layout(model.N_electron)
    rows = set(lay.owned)
    for _, beta in model._active_pairs_index:
        rows.add(beta)
    return sorted(rows)


def _ghost_sync_plan(model):
    """Who sends which owned rows to whom, and in what order.

    Built once: it depends only on the alpha partition and the pair topology,
    both of which are fixed after ``_active_mpo_select_grouping``.
    """
    plan = getattr(model, "_ghost_sync_plan", None)
    if plan is not None:
        return plan

    lay = _ensure_layout(model.N_electron)
    resident = resident_alphas(model)
    ghosts = [a for a in resident if not lay.owns(a)]
    all_ghosts = COMM.allgather(ghosts) if SIZE > 1 else [ghosts]

    # Receiving order is the sender's owned order, so both sides agree without
    # sending an index list alongside every payload.
    send = []  # (dest_rank, local_row_positions)
    recv = []  # (src_rank, global_alphas)
    for peer, peer_ghosts in enumerate(all_ghosts):
        if peer == RANK:
            continue
        mine = [a - lay.start for a in peer_ghosts if lay.owns(a)]
        if mine:
            send.append((peer, xp.asarray(mine, dtype=np.int64)))
    for alpha in ghosts:
        owner = lay.owner_of(alpha)
        recv.append((owner, alpha))
    grouped_recv = {}
    for owner, alpha in recv:
        grouped_recv.setdefault(owner, []).append(alpha)
    recv = [(owner, sorted(alphas)) for owner, alphas in sorted(grouped_recv.items())]

    plan = {
        "resident": resident,
        "ghosts": ghosts,
        "send": send,
        "recv": recv,
        "owned": list(lay.owned),
    }
    model._ghost_sync_plan = plan
    if RANK == 0:
        logger.info(
            "alpha sharding: %d owned rows + %d ghost rows per rank "
            "(%.1f%% of %d)",
            len(plan["owned"]),
            len(ghosts),
            100.0 * len(resident) / model.N_electron,
            model.N_electron,
        )
    return plan


def _sync_ghost_site(ms_mps, plan, imps, block):
    """Push site ``imps`` of every owned row to the ranks holding it as a ghost.

    ``block`` is the freshly assigned tensor stack, shaped ``(n_owned, *site)``.
    Only this one site moves: ``Environ.GetLR(..., method="System")`` touches
    ``mps[siteidx]`` alone, and the neighbouring site a zero-site solve modifies
    is overwritten at the next sweep position before anything reads it.
    """
    ghost_updates = []
    if SIZE > 1 and (plan["send"] or plan["recv"]):
        site_shape = tuple(block.shape[1:])
        dtype = np.dtype(block.dtype)
        requests = []
        send_buffers = []
        for dest, rows in plan["send"]:
            buf = np.ascontiguousarray(asnumpy(block[rows]))
            send_buffers.append(buf)
            requests.append(COMM.Isend(buf, dest=dest, tag=_TAG_SITE + (imps % 512)))
        recv_buffers = []
        for src, alphas in plan["recv"]:
            buf = np.empty((len(alphas),) + site_shape, dtype=dtype)
            recv_buffers.append((alphas, buf))
            requests.append(COMM.Irecv(buf, source=src, tag=_TAG_SITE + (imps % 512)))
        MPI.Request.Waitall(requests)
        for alphas, buf in recv_buffers:
            for k, alpha in enumerate(alphas):
                ghost_updates.append((alpha, asxp(buf[k])))
        del send_buffers, recv_buffers

    for alpha, tensor in ghost_updates:
        ms_mps.msmps[alpha][imps] = tensor


def _set_ghost_qn(ms_mps, plan, qn_index, qn_value, qnidx):
    for alpha in plan["ghosts"]:
        ms_mps.msmps[alpha].qn[qn_index] = qn_value
        ms_mps.msmps[alpha].qnidx = qnidx


# --------------------------------------------------------------------------- #
# reductions over a sharded state
# --------------------------------------------------------------------------- #
def _allreduce_complex(value):
    if SIZE == 1:
        return value
    return COMM.allreduce(value, op=MPI.SUM)


def _ref_row(ms_mps):
    """Any resident chain: they all share site count, shapes, qn and direction."""
    lay = _ensure_layout(ms_mps.N_electron)
    for alpha in lay.owned:
        row = ms_mps.msmps[alpha]
        if row is not None and len(row) > 0:
            return row
    for row in ms_mps.msmps:
        if row is not None and len(row) > 0:
            return row
    raise RuntimeError(f"rank {RANK} holds no multiset rows")


def _owned_rows(ms_mps):
    lay = _ensure_layout(ms_mps.N_electron)
    return [ms_mps.msmps[a] for a in lay.owned if ms_mps.msmps[a] is not None]


def _resident_rows(ms_mps):
    return [row for row in ms_mps.msmps if row is not None]


def _ms_normalize_sharded(self, kind):
    if kind not in ("mps_only",):
        raise ValueError(f"kind={kind} is not valid.")
    total = 0.0
    for mps in _owned_rows(self):
        total += _state_inner_product(mps, mps)
    total = _allreduce_complex(total) ** 0.5
    # Ghost rows are scaled too: they must stay bit-identical to the owner copy.
    for mps in _resident_rows(self):
        mps.scale(1.0 / total, inplace=True)


def _e_occupations_sharded(self):
    occupations = np.zeros(self.N_electron, dtype=np.float64)
    lay = _ensure_layout(self.N_electron)
    for alpha in lay.owned:
        mps = self.msmps[alpha]
        if mps is not None:
            occupations[alpha] = _state_inner_product(mps, mps).real
    if SIZE > 1:
        COMM.Allreduce(MPI.IN_PLACE, occupations, op=MPI.SUM)
    return occupations


def _ph_occupations_sharded(self):
    lay = _ensure_layout(self.N_electron)
    total = None
    for alpha in lay.owned:
        mps = self.msmps[alpha]
        if mps is None:
            continue
        occ = np.asarray(mps.ph_occupations)
        total = occ.copy() if total is None else total + occ
    if total is None:
        return np.array([])
    if SIZE > 1:
        total = np.ascontiguousarray(total)
        COMM.Allreduce(MPI.IN_PLACE, total, op=MPI.SUM)
    if np.allclose(total.imag, 0):
        return total.real
    return total


def _copy_sharded(self):
    new = MultisetMps.__new__(MultisetMps)
    new.MsModel = self.MsModel
    new.N_electron = self.N_electron
    new.temperature = self.temperature
    new.init_model = self.init_model
    new.method = self.method
    new.msmps = [None if m is None else m.copy() for m in self.msmps]
    return new


def _to_complex_sharded(self):
    new = MultisetMps.__new__(MultisetMps)
    new.MsModel = self.MsModel
    new.N_electron = self.N_electron
    new.temperature = self.temperature
    new.init_model = self.init_model
    new.method = self.method
    new.msmps = [None if m is None else m.to_complex() for m in self.msmps]
    return new


def _rho_el_sharded(self):
    raise NotImplementedError(
        "rho_el needs every pair of electronic rows on one rank; it is not "
        "available with RENO_MS_SHARD_STAGE>=3."
    )


def _compute_multiset_norm_sharded(self, ms_mps):
    if getattr(ms_mps, "electronic_ancilla", False):
        return ms_mps.total_norm() ** 0.5
    total = 0.0
    for mps in _owned_rows(ms_mps):
        total += _state_inner_product(mps, mps)
    return _allreduce_complex(total) ** 0.5


def _inner_product_sharded(self):
    total = 0.0
    for mps in _owned_rows(self.MsMps):
        total += _state_inner_product(mps, mps)
    return _allreduce_complex(total)


def _hamiltonian_sharded(self):
    num = 0.0
    for pair_id, (alpha, beta) in enumerate(self._active_pairs_index):
        num += _state_expectation(
            self.MsMps.msmps[alpha],
            self.MsMps.msmps[beta],
            self._active_pair_mpos[pair_id],
        )
    den = 0.0
    for mps in _owned_rows(self.MsMps):
        den += _state_inner_product(mps, mps)
    num = _allreduce_complex(num)
    den = _allreduce_complex(den)
    return (num / den).real


# --------------------------------------------------------------------------- #
# environment cache: reference row instead of row 0
# --------------------------------------------------------------------------- #
def _cache_key(model, ms_mps):
    ref = _ref_row(ms_mps)
    token = getattr(ms_mps, "_environ_cache_token", id(ms_mps))
    return (
        token,
        len(model._active_pairs_index),
        len(ref),
        ref.to_right,
        ref.qnidx,
    )


def _has_valid_environ_cache_sharded(self, ms_mps):
    if not self._reuse_environ_cache or self._environ_cache is None:
        return False
    if self._environ_cache["cache_key"] != _cache_key(self, ms_mps):
        return False
    return len(self._environ_cache["envs"]) == len(self._active_pairs_index)


def _store_environ_cache_sharded(self, ms_mps, environ_list):
    if not hasattr(ms_mps, "_environ_cache_token"):
        ms_mps._environ_cache_token = self._environ_cache_token_counter
        self._environ_cache_token_counter += 1
    self._environ_cache = {
        "cache_key": _cache_key(self, ms_mps),
        "envs": environ_list,
    }


def _get_or_build_environ_list_sharded(self, ms_mps, conj_mps=None):
    if self._has_valid_environ_cache(ms_mps):
        return self._environ_cache["envs"]
    environ_list = self._build_environ_list(ms_mps, conj_mps)
    self._store_environ_cache(ms_mps, environ_list)
    return environ_list


def _rescale_environ_cache_sharded(self, ms_mps, scale_factor):
    if not self._has_valid_environ_cache(ms_mps):
        return
    ref = _ref_row(ms_mps)
    site_num = len(ref)
    scale_sq = (scale_factor * np.conjugate(scale_factor)).real
    if np.allclose(scale_sq, 1.0):
        return
    if ref.to_right and ref.qnidx == 0:
        domain, affected = "L", range(0, site_num - 1)
    elif (not ref.to_right) and ref.qnidx == site_num - 1:
        domain, affected = "R", range(1, site_num)
    else:
        self._invalidate_environ_cache()
        return
    for environ in self._environ_cache["envs"]:
        for siteidx in affected:
            key = (domain, siteidx)
            if key in environ._virtual_disk:
                environ._virtual_disk[key] *= scale_sq


# --------------------------------------------------------------------------- #
# bond-dimension expansion over owned alpha rows
# --------------------------------------------------------------------------- #
def _expand_bond_dimension_sharded(self, coef: float = 1e-10, use_hint: bool = True):
    if getattr(self.MsMps, "electronic_ancilla", False):
        raise NotImplementedError(
            "electronic-ancilla states are not sharded by alpha"
        )
    lay = _ensure_layout(self.N_electron)
    plan = _ghost_sync_plan(self)
    resident = set(plan["resident"])

    for mps in _resident_rows(self.MsMps):
        mps.compress_config = self.compress_config

    if use_hint:
        # Pre-expansion chains are bond dimension 1, so the copy is negligible.
        original_mps = [
            None if m is None else m.copy() for m in self.MsMps.msmps
        ]
    else:
        original_mps = None

    for alpha in lay.owned:
        if use_hint:
            mps_alpha = original_mps[alpha]
            mps_alpha.compress_config = self.compress_config
            diag_mpo = self.MsMpo.msmpo[alpha][alpha]
            hint_mpo = diag_mpo if len(diag_mpo) > 0 else None
            cross_states = []
            for pair_id in self._active_pairs_by_alpha[alpha]:
                beta = self._active_pairs_index[pair_id][1]
                if beta == alpha:
                    continue
                cross_states.append(
                    self._active_pair_mpos[pair_id].apply(original_mps[beta])
                )
            ex_mps = _sum(cross_states, compress=False) if cross_states else None
            if ex_mps is not None:
                ex_mps.compress_config = self.compress_config
            expanded = expand_bond_dimension_general(
                mps_alpha, hint_mpo=hint_mpo, coef=coef, ex_mps=ex_mps
            )
        else:
            expanded = expand_bond_dimension_general(
                self.MsMps.msmps[alpha], hint_mpo=None, coef=coef, ex_mps=None
            )
        expanded.scale(float(abs(expanded.coeff)), inplace=True)
        expanded.coeff = 1.0
        self.MsMps.msmps[alpha] = expanded

    del original_mps

    if SIZE == 1:
        return

    if not state_is_sharded():
        # Stage 2 keeps the state replicated: every rank needs every expanded row.
        for alpha in range(self.N_electron):
            owner = lay.owner_of(alpha)
            payload = self.MsMps.msmps[alpha] if RANK == owner else None
            self.MsMps.msmps[alpha] = COMM.bcast(payload, root=owner)
        return

    # Stage 3+: only ghost holders need a copy.  Lock-step over alpha, so for
    # each row the owner sends while exactly its ghost holders receive and no
    # rank waits on a message another rank has not reached yet.
    residents_by_rank = [set(rows) for rows in COMM.allgather(sorted(resident))]
    for alpha in range(self.N_electron):
        owner = lay.owner_of(alpha)
        targets = [
            peer
            for peer, rows in enumerate(residents_by_rank)
            if peer != owner and alpha in rows
        ]
        if RANK == owner:
            for peer in targets:
                COMM.send(self.MsMps.msmps[alpha], dest=peer, tag=_TAG_CHAIN)
        elif RANK in targets:
            self.MsMps.msmps[alpha] = COMM.recv(source=owner, tag=_TAG_CHAIN)
        else:
            self.MsMps.msmps[alpha] = None


# --------------------------------------------------------------------------- #
# the sharded TDVP-PS sweep
# --------------------------------------------------------------------------- #
def _ms_evolve_tdvp_ps_sharded(self, ms_mps_, ms_mpo, evolve_dt):
    from scipy import stats

    lay = _ensure_layout(self.N_electron)
    plan = _ghost_sync_plan(self)
    owned = plan["owned"]
    n_local = len(owned)

    if np.iscomplex(evolve_dt):
        ms_mps = ms_mps_.copy()
    else:
        ms_mps = ms_mps_.to_complex()
    if hasattr(ms_mps_, "_environ_cache_token"):
        ms_mps._environ_cache_token = ms_mps_._environ_cache_token
    if self.evolve_config.ivp_solver != "krylov":
        raise NotImplementedError("alpha sharding supports the krylov solver only")

    Environ_list = self._get_or_build_environ_list(ms_mps)
    ref_alpha = owned[0]
    n_pairs = len(self._active_pairs_index)
    local_steps = []

    for _ in range(2):
        for imps in ms_mps.msmps[ref_alpha].iter_idx_list(full=True):
            ref = ms_mps.msmps[ref_alpha]
            system = "L" if ref.to_right else "R"
            shape_imps = list(ref[imps].shape)
            dim = int(np.prod(shape_imps))

            l_array_ab = [environ.read("L", imps - 1) for environ in Environ_list]
            r_array_ab = [environ.read("R", imps + 1) for environ in Environ_list]
            batched_data = self._build_site_batched_data(imps, l_array_ab, r_array_ab)

            Y0 = xp.stack(
                [asxp(ms_mps.msmps[a][imps].array).reshape(dim) for a in owned]
            ).reshape(-1)
            ivp_eq = lambda Y: self._apply_hop_batched(Y, batched_data, dim, shape_imps)
            ivp_eq._reno_vector_block_count = self.N_electron
            ivp_eq._reno_local_rows = n_local
            mps_t, j = multiset_model.expm_krylov(
                ivp_eq, -1j * evolve_dt / 2, Y0, block_size=KRYLOV_BLOCK_SIZE
            )
            mps_t = mps_t.reshape((n_local,) + tuple(shape_imps))
            local_steps.append(j)

            qnbigl, qnbigr, _ = ref._get_big_qn([imps])
            max_qr_rank = None
            if self.compress_config.criteria is not CompressCriteria.threshold:
                self.compress_config.set_bonddim(len(ref.bond_dims))
                bond_idx = imps + 1 if system == "L" else imps
                max_qr_rank = self.compress_config.max_dims[bond_idx]
            u_batch, qnlset, vt_batch, qnrset = self._batched_qr_qn(
                mps_t, qnbigl, qnbigr, ref.qntot, system, max_rank=max_qr_rank
            )

            if not ref.to_right and imps != 0:
                block = vt_batch.reshape([n_local, -1] + shape_imps[1:])
                for k, alpha in enumerate(owned):
                    ms_mps.msmps[alpha][imps] = block[k]
                    ms_mps.msmps[alpha].qn[imps] = qnrset
                    ms_mps.msmps[alpha].qnidx = imps - 1
                _sync_ghost_site(ms_mps, plan, imps, block)
                _set_ghost_qn(ms_mps, plan, imps, qnrset, imps - 1)

                shapeU = list(u_batch[0].shape)
                dimU = int(np.prod(shapeU))
                r_array_u = [None] * n_pairs
                for alpha in owned:
                    mps_conj_alpha = [None] * len(ms_mps.msmps[alpha])
                    mps_conj_alpha[imps] = ms_mps.msmps[alpha][imps].conj()
                    for pair_id in self._active_pairs_by_alpha[alpha]:
                        beta = self._active_pairs_index[pair_id][1]
                        r_array_u[pair_id] = Environ_list[pair_id].GetLR(
                            "R",
                            imps,
                            ms_mps.msmps[beta],
                            self._active_pair_mpos[pair_id],
                            itensor=r_array_ab[pair_id],
                            method="System",
                            mps_conj=mps_conj_alpha,
                        )

                batched_u = self._build_reverse_batched_data(l_array_ab, r_array_u)
                U0 = u_batch.reshape(n_local, dimU).reshape(-1)
                ivp_eq_Ut = lambda Y: self._apply_hop_batched(Y, batched_u, dimU, shapeU)
                ivp_eq_Ut._reno_vector_block_count = self.N_electron
                ivp_eq_Ut._reno_local_rows = n_local
                Ut, j2 = multiset_model.expm_krylov(
                    ivp_eq_Ut, 1j * evolve_dt / 2, U0, block_size=KRYLOV_BLOCK_SIZE
                )
                local_steps.append(j2)
                Ut = Ut.reshape(n_local, dimU)
                # Site imps-1 is the next sweep position: it is overwritten there
                # before any ghost copy is read, so it needs no exchange here.
                for k, alpha in enumerate(owned):
                    ms_mps.msmps[alpha][imps - 1] = tensordot(
                        ms_mps.msmps[alpha][imps - 1].array,
                        Ut[k].reshape(shapeU),
                        axes=(-1, 0),
                    )

            elif ref.to_right and imps != len(ref) - 1:
                block = u_batch.reshape([n_local] + shape_imps[:-1] + [-1])
                for k, alpha in enumerate(owned):
                    ms_mps.msmps[alpha][imps] = block[k]
                    ms_mps.msmps[alpha].qn[imps + 1] = qnlset
                    ms_mps.msmps[alpha].qnidx = imps + 1
                _sync_ghost_site(ms_mps, plan, imps, block)
                _set_ghost_qn(ms_mps, plan, imps + 1, qnlset, imps + 1)

                shapeC = list(vt_batch[0].shape)
                dimC = int(np.prod(shapeC))
                l_array_c = [None] * n_pairs
                for alpha in owned:
                    mps_conj_alpha = [None] * len(ms_mps.msmps[alpha])
                    mps_conj_alpha[imps] = ms_mps.msmps[alpha][imps].conj()
                    for pair_id in self._active_pairs_by_alpha[alpha]:
                        beta = self._active_pairs_index[pair_id][1]
                        l_array_c[pair_id] = Environ_list[pair_id].GetLR(
                            "L",
                            imps,
                            ms_mps.msmps[beta],
                            self._active_pair_mpos[pair_id],
                            itensor=l_array_ab[pair_id],
                            method="System",
                            mps_conj=mps_conj_alpha,
                        )

                batched_c = self._build_reverse_batched_data(l_array_c, r_array_ab)
                C0 = vt_batch.reshape(n_local, dimC).reshape(-1)
                ivp_eq_Ct = lambda Y: self._apply_hop_batched(Y, batched_c, dimC, shapeC)
                ivp_eq_Ct._reno_vector_block_count = self.N_electron
                ivp_eq_Ct._reno_local_rows = n_local
                Ct, j2 = multiset_model.expm_krylov(
                    ivp_eq_Ct, 1j * evolve_dt / 2, C0, block_size=KRYLOV_BLOCK_SIZE
                )
                local_steps.append(j2)
                Ct = Ct.reshape(n_local, dimC)
                for k, alpha in enumerate(owned):
                    ms_mps.msmps[alpha][imps + 1] = tensordot(
                        Ct[k].reshape(shapeC),
                        ms_mps.msmps[alpha][imps + 1].array,
                        axes=(1, 0),
                    )

            else:
                for k, alpha in enumerate(owned):
                    ms_mps.msmps[alpha][imps] = mps_t[k]
                _sync_ghost_site(ms_mps, plan, imps, mps_t)

        for mps in _resident_rows(ms_mps):
            mps._switch_direction()

    steps_stat = stats.describe(local_steps)
    logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
    self.evolve_config.stat = steps_stat
    self._store_environ_cache(ms_mps, Environ_list)
    return ms_mps


# --------------------------------------------------------------------------- #
# job-level helpers that walk every alpha row
# --------------------------------------------------------------------------- #
def _state_bond_dims_sharded(state):
    """``_state_bond_dims`` that tolerates the rows this rank does not hold."""
    if isinstance(state, MultisetMps):
        return [None if mps is None else list(mps.bond_dims) for mps in state.msmps]
    if hasattr(state, "bond_dims"):
        return list(state.bond_dims)
    if isinstance(state, (tuple, list)):
        return [_state_bond_dims_sharded(item) for item in state]
    return None


def _init_mps_sharded(self):
    """``MultisetChargeDiffusionDynamics.init_mps`` skipping non-resident rows.

    Identical to the original apart from the two f-strings, which index every
    alpha row eagerly and would raise on the ``None`` placeholders that
    ``expand_bond_dimension_multiset`` leaves behind.
    """
    from renormalizer.utils import Quantity

    state = self._init_msmps(self.init_mp())
    self._fc_excitation(state, self.initial_site)

    self.ms_model.set_mps(state)
    energy = Quantity(self.ms_model.Hamiltonian())
    logger.info("[init] E0 = %.6f a.u.", energy.as_au())
    self._set_hamiltonian_offset(energy)

    self.ms_model.expand_bond_dimension_multiset(
        coef=1e-10, use_hint=self.use_init_hint
    )
    self.ms_model.MsMps.ms_normalize("mps_only")

    resident = [m for m in self.ms_model.MsMps.msmps if m is not None]
    logger.info(
        "[init] rank %d holds %d/%d rows, max bond dim %d",
        RANK,
        len(resident),
        self.ms_model.N_electron,
        max(max(m.bond_dims) for m in resident),
    )
    return self.ms_model.MsMps


# --------------------------------------------------------------------------- #
# installation
# --------------------------------------------------------------------------- #
def install_patch():
    """Install the patches for ``RENO_MS_SHARD_STAGE``; call before model setup."""
    if SIZE == 1 or STAGE < 2:
        if RANK == 0:
            logger.info(
                "alpha sharding disabled (stage=%d, size=%d)", STAGE, SIZE
            )
        return

    _ORIGINAL["grouping"] = MultisetModel._active_mpo_select_grouping
    _ORIGINAL["expand"] = MultisetModel.expand_bond_dimension_multiset
    _ORIGINAL["hamiltonian"] = MultisetModel.Hamiltonian
    _ORIGINAL["inner"] = MultisetModel.Inner_product

    MultisetModel._active_mpo_select_grouping = _active_mpo_select_grouping_sharded
    MultisetModel.expand_bond_dimension_multiset = _expand_bond_dimension_sharded
    MultisetModel.Hamiltonian = _hamiltonian_sharded
    MultisetModel.Inner_product = _inner_product_sharded
    MultisetModel._compute_multiset_norm = _compute_multiset_norm_sharded
    MultisetModel._has_valid_environ_cache = _has_valid_environ_cache_sharded
    MultisetModel._store_environ_cache = _store_environ_cache_sharded
    MultisetModel._get_or_build_environ_list = _get_or_build_environ_list_sharded
    MultisetModel._rescale_environ_cache_after_normalize = _rescale_environ_cache_sharded

    # Owned-rows + allreduce is identical to a full local sum while the state is
    # still replicated, so these are safe to install from stage 2 on.
    MultisetMps.ms_normalize = _ms_normalize_sharded
    MultisetMps.copy = _copy_sharded
    MultisetMps.to_complex = _to_complex_sharded
    MultisetMps.e_occupations_multiset = property(_e_occupations_sharded)
    MultisetMps.ph_occupations_multiset = property(_ph_occupations_sharded)

    if STAGE >= 3:
        from renormalizer.multiset import multiset_tdjob
        from renormalizer.multiset.multiset_tdjob import (
            MultisetChargeDiffusionDynamics,
        )

        MultisetModel._ms_evolve_tdvp_ps = _ms_evolve_tdvp_ps_sharded
        MultisetMps.rho_el = _rho_el_sharded
        multiset_tdjob._state_bond_dims = _state_bond_dims_sharded
        MultisetChargeDiffusionDynamics.init_mps = _init_mps_sharded

    if STAGE >= 4:
        MultisetModel.ConstructMsModel = _construct_msmodel_sharded
        MultisetMpo._ConstructMsMpo = _construct_msmpo_sharded

    if RANK == 0:
        sharded = ["pairs/environments"]
        if STAGE >= 3:
            sharded.append("state")
        if STAGE >= 4:
            sharded.append("setup")
        logger.info(
            "Installed alpha sharding patch: stage=%d size=%d, sharded: %s",
            STAGE,
            SIZE,
            ", ".join(sharded),
        )
