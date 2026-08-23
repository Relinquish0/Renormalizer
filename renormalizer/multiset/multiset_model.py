# -*- coding: utf-8 -*-

import logging
import os
from typing import List, Tuple

import numpy as np
from scipy import stats

from renormalizer.lib import expm_krylov
from renormalizer.model.model import Model
from renormalizer.model.op import Op
from renormalizer.mps import Mps
from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.mps.lib import Environ, _sum
from renormalizer.mps.matrix import asnumpy, asxp, tensordot
from renormalizer.mps.mpo import Mpo
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.multiset.multiset_mpo import MultisetMpo
from renormalizer.multiset.multiset_mps import (
    ElectronicAncillaMultisetMps,
    MsEvolveMethod,
    MultisetMps,
    _state_expectation,
    _state_inner_product,
)
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod, Quantity

logger = logging.getLogger(__name__)

# Lanczos basis pre-allocation for the multiset TDVP solves.  The multiset Krylov
# vector spans every electronic component at once, so this array is the largest
# single GPU allocation in the program; convergence is measured at 17 iterations
# at every lattice size tested, so 50 wastes about two thirds of it.
KRYLOV_BLOCK_SIZE = int(os.environ.get("RENO_MS_KRYLOV_BLOCK", "24"))

if USE_GPU:
    import cupyx as _cupyx


def _scatter_add_rows(target, row_idx, values):
    """``target[row_idx] += values`` with duplicate rows accumulated.

    Replaces multiplying by a dense one-hot (N_electron x n_pairs) matrix, whose
    FLOPs and footprint both grow one power of L faster than the contraction it
    was attached to.
    """
    if not USE_GPU:
        np.add.at(target, row_idx, values)
        return target
    if target.dtype.kind == "c":
        # cupyx.scatter_add has no complex kernel; a float64 view of a contiguous
        # complex128 array is an exact reinterpretation, and the row axis is
        # untouched by it.
        _cupyx.scatter_add(
            target.view(xp.float64),
            row_idx,
            xp.ascontiguousarray(values).view(xp.float64),
        )
    else:
        _cupyx.scatter_add(target, row_idx, values)
    return target


class _EmptyMsBlock(list):
    """Stand-in for an ``(alpha, beta)`` block that carries no Hamiltonian terms.

    Behaves like the empty list the block matrix is initialised with, and also
    answers ``.ham_terms`` so the consumers that reach for it (``MultisetMpo``,
    ``multiset_spectra``) need no special case.
    """

    __slots__ = ()

    @property
    def ham_terms(self):
        return []


EMPTY_MS_BLOCK = _EmptyMsBlock()


class MultisetModel:
    def __init__(
        self,
        model: Model,
        max_bonddim,
        temperature: Quantity = Quantity(0, "K"),
        method: str = "thermo_field",
        evolve_config: EvolveConfig = None,
        compress_config: CompressConfig = None,
        auto_init: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.method = method
        if evolve_config is None:
            self.evolve_config: EvolveConfig = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps)
        else:
            self.evolve_config = evolve_config
        if compress_config is None:
            self.compress_config: CompressConfig = CompressConfig(
                CompressCriteria.fixed, max_bonddim=max_bonddim
            )
        else:
            self.compress_config = compress_config
        self.N_electron = self.model.n_edofs
        self.electron_index = {dof: idx for idx, dof in enumerate(self.model.e_dofs)}
        self.basis_set = [
            item for item in self.model.basis if not item.is_electron
        ]

        self.MsModel = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
        self.MsOp = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]

        self.SplitHamTerm()
        self.ConstructInitModel()
        self.ConstructMsModel()

        self.MsMpo = MultisetMpo(self.MsModel, self.N_electron)
        self._active_pairs_index: List[Tuple[int, int]] = []
        self._active_pair_mpos: List[Mpo] = []
        self._active_pairs_by_alpha: List[List[int]] = [[] for _ in range(self.N_electron)]
        
        self._site_group_templates = []
        self._qr_qn_plan_cache = {}
        self._reuse_environ_cache = True
        self._environ_cache = None
        # Constant shift H -> H - offset*I, applied inside the batched hop instead
        # of being baked into N_electron^2 rebuilt MPOs (see set_energy_offset).
        self._energy_offset = 0.0
        self._environ_cache_token_counter = 0
        self._active_mpo_select_grouping()
        self.MsMps = None
        if auto_init:
            logger.info(
                "MultisetModel no longer auto-initialises MsMps. "
                "Initial states should be prepared explicitly by a MultisetTdJob subclass."
            )

    def SplitHamTerm(self):
        for ham_term in self.model.ham_terms:
            transition = self._get_electron_transition(ham_term)
            if transition is None:
                for alpha in range(self.N_electron):
                    self.MsOp[alpha][alpha].append(self._reset_all_MsOp(ham_term))
                continue

            alpha, beta = transition
            self.MsOp[alpha][beta].append(self._reset_all_MsOp(ham_term))

    def _get_electron_transition(self, op: Op):
        electron_ops = []
        for dof, symbol in zip(op.dofs, op.split_symbol):
            if dof in self.model.e_dofs:
                electron_ops.append((dof, symbol))

        if len(electron_ops) == 0:
            return None
        if len(electron_ops) != 2:
            raise ValueError(f"Unsupported electronic operator structure in multiset conversion: {op}")

        (dof1, symbol1), (dof2, symbol2) = electron_ops
        if {symbol1, symbol2} != {r"a^\dagger", "a"}:
            raise ValueError(
                f"Unknown electron operator symbols in multiset conversion: {symbol1}, {symbol2}"
            )

        if symbol1 == r"a^\dagger":
            return self.electron_index[dof1], self.electron_index[dof2]
        return self.electron_index[dof2], self.electron_index[dof1]

    def _reset_all_MsOp(self, op: Op):
        new_op = Op.product([op])
        new_split_symbol = []
        new_qn_list = []
        new_dofs = []

        for symbol, qn, dof in zip(new_op.split_symbol, new_op.qn_list, new_op.dofs):
            if dof in self.model.e_dofs:
                continue
            new_split_symbol.append(symbol)
            new_qn_list.append(qn)
            new_dofs.append(dof)

        if len(new_split_symbol) == 0:
            if len(self.basis_set) == 0:
                raise ValueError("No vibrational basis left after removing electronic DoFs.")
            identity_dof = self.basis_set[0].dofs[0]
            return Op("I", identity_dof, new_op.factor, qn=0)

        new_op.symbol = " ".join(new_split_symbol)
        new_op.split_symbol = new_split_symbol
        new_op.qn_list = new_qn_list
        new_op.dofs = new_dofs
        return new_op

    def ConstructMsModel(self):
        # Only a thin band of the N_electron x N_electron block matrix carries
        # operators (L diagonal + 4 hopping neighbours per site).  An empty
        # ``Model`` still builds the full per-basis lookup tables: 102 KB and
        # 0.5 ms at 729 phonon sites, i.e. 52 GiB and 266 s of pure waste for the
        # 531441 blocks of a 27x27 lattice.  Empty blocks keep the ``[]`` they
        # were initialised with, wrapped so ``.ham_terms`` still answers.
        for i in range(self.N_electron):
            for j in range(self.N_electron):
                if len(self.MsOp[i][j]) == 0:
                    self.MsModel[i][j] = EMPTY_MS_BLOCK
                else:
                    self.MsModel[i][j] = Model(basis=self.basis_set, ham_terms=self.MsOp[i][j])

    def ConstructInitModel(self):
        init_terms = []
        for op in self.model.ham_terms:
            if len(op.dofs) == 1:
                init_terms.append(self._reset_all_MsOp(op))
        self.init_model = Model(basis=self.basis_set, ham_terms=init_terms)

    def _active_mpo_select_grouping(self):
        self._environ_cache = None
        active_pairs = []
        active_pair_mpos = []
        active_pairs_by_alpha = [[] for _ in range(self.N_electron)]
        mpo_group_dicts = None

        for alpha in range(self.N_electron):
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
                    active_mpos_group = mpo_group_dicts[imps].setdefault(
                        key,
                        {
                            "pair_ids": [],
                            "alpha_idx": [],
                            "beta_idx": [],
                            "w_tensors": [],
                        },
                    )
                    active_mpos_group["pair_ids"].append(pair_id)
                    active_mpos_group["alpha_idx"].append(alpha)
                    active_mpos_group["beta_idx"].append(beta)
                    active_mpos_group["w_tensors"].append(W)

        self._active_pairs_index = active_pairs
        self._active_pair_mpos = active_pair_mpos
        self._active_pairs_by_alpha = active_pairs_by_alpha
        self._site_group_templates = []

        if mpo_group_dicts is None:
            return

        for site_groups in mpo_group_dicts:
            templates = []
            for key in sorted(site_groups):
                active_mpos_group = site_groups[key]
                templates.append(
                    {
                        "pair_ids": tuple(active_mpos_group["pair_ids"]),
                        "alpha_idx": xp.asarray(active_mpos_group["alpha_idx"], dtype=np.int64),
                        "beta_idx": xp.asarray(active_mpos_group["beta_idx"], dtype=np.int64),
                        "W": xp.stack(active_mpos_group["w_tensors"]),
                        "nsite": 1,
                        "n_pairs": len(active_mpos_group["pair_ids"]),
                    }
                )
            self._site_group_templates.append(templates)

    def set_energy_offset(self, energy):
        """Shift the Hamiltonian by ``-energy * I`` without touching the MPOs.

        In the projected TDVP equations the environments are built from isometries,
        so a constant shift of ``H`` is exactly a constant shift of every one-site
        and zero-site effective Hamiltonian.  Applying it as a scalar in
        ``_apply_hop_batched`` is therefore identical to passing ``offset=`` to
        every diagonal ``Mpo``, and avoids rebuilding N_electron^2 MPOs plus a
        second pass of ``_active_mpo_select_grouping``.
        """
        if isinstance(energy, Quantity):
            energy = energy.as_au()
        self._energy_offset = float(energy)

    def _invalidate_environ_cache(self):
        self._environ_cache = None

    def _has_valid_environ_cache(self, ms_mps: MultisetMps) -> bool:
        if not self._reuse_environ_cache or self._environ_cache is None:
            return False
        ref_mps = ms_mps.msmps[0]
        state_token = getattr(ms_mps, "_environ_cache_token", id(ms_mps))
        current_cache_key = (
            state_token,
            len(self._active_pairs_index),
            len(ref_mps),
            ref_mps.to_right,
            ref_mps.qnidx,
        )
        if self._environ_cache["cache_key"] != current_cache_key:
            return False
        if len(self._environ_cache["envs"]) != len(self._active_pairs_index):
            return False
        return True

    def _build_environ_list(self, ms_mps: MultisetMps, conj_mps=None):
        """Build one ``Environ`` per active pair, holding at most one conjugate MPS.

        ``Environ`` needs the conjugate of the *bra* (alpha) chain.  Materialising
        all ``N_electron`` conjugates at once costs a second full copy of the
        multiset state; building them one alpha row at a time costs one chain.
        """
        environ_list = [None] * len(self._active_pairs_index)
        for alpha in range(self.N_electron):
            pair_ids = self._active_pairs_by_alpha[alpha]
            if not pair_ids:
                continue
            if conj_mps is not None:
                conj_alpha = conj_mps[alpha]
            else:
                conj_alpha = ms_mps.msmps[alpha].conj()
            for pair_id in pair_ids:
                beta = self._active_pairs_index[pair_id][1]
                environ_list[pair_id] = Environ(
                    ms_mps.msmps[beta],
                    self._active_pair_mpos[pair_id],
                    mps_conj=conj_alpha,
                )
            if conj_mps is None:
                del conj_alpha
        if any(environ is None for environ in environ_list):
            raise RuntimeError(
                "_active_pairs_by_alpha does not cover every active pair"
            )
        return environ_list

    def _get_or_build_environ_list(self, ms_mps: MultisetMps, conj_mps=None):
        if self._has_valid_environ_cache(ms_mps):
            return self._environ_cache["envs"]
        environ_list = self._build_environ_list(ms_mps, conj_mps)
        ref_mps = ms_mps.msmps[0]
        state_token = getattr(ms_mps, "_environ_cache_token", id(ms_mps))
        self._environ_cache = {
            "cache_key": (
                state_token,
                len(self._active_pairs_index),
                len(ref_mps),
                ref_mps.to_right,
                ref_mps.qnidx,
            ),
            "envs": environ_list,
        }
        return environ_list

    def _store_environ_cache(self, ms_mps: MultisetMps, environ_list):
        if not hasattr(ms_mps, "_environ_cache_token"):
            ms_mps._environ_cache_token = self._environ_cache_token_counter
            self._environ_cache_token_counter += 1
        ref_mps = ms_mps.msmps[0]
        state_token = getattr(ms_mps, "_environ_cache_token", id(ms_mps))
        self._environ_cache = {
            "cache_key": (
                state_token,
                len(self._active_pairs_index),
                len(ref_mps),
                ref_mps.to_right,
                ref_mps.qnidx,
            ),
            "envs": environ_list,
        }

    def _compute_multiset_norm(self, ms_mps: MultisetMps):
        if getattr(ms_mps, "electronic_ancilla", False):
            return ms_mps.total_norm() ** 0.5
        total_tn_coeff = 0.0
        for alpha in range(ms_mps.N_electron):
            total_tn_coeff += _state_inner_product(ms_mps.msmps[alpha], ms_mps.msmps[alpha])
        return total_tn_coeff**0.5

    def _rescale_environ_cache_after_normalize(self, ms_mps: MultisetMps, scale_factor):
        if not self._has_valid_environ_cache(ms_mps):
            return

        ref_mps = ms_mps.msmps[0]
        site_num = len(ref_mps)
        scale_sq = (scale_factor * np.conjugate(scale_factor)).real
        if np.allclose(scale_sq, 1.0):
            return

        if ref_mps.to_right and ref_mps.qnidx == 0:
            domain = "L"
            affected_indices = range(0, site_num - 1)
        elif (not ref_mps.to_right) and ref_mps.qnidx == site_num - 1:
            domain = "R"
            affected_indices = range(1, site_num)
        else:
            self._invalidate_environ_cache()
            return

        for environ in self._environ_cache["envs"]:
            for siteidx in affected_indices:
                key = (domain, siteidx)
                if key in environ._virtual_disk:
                    environ._virtual_disk[key] *= scale_sq

    def _get_qr_qn_plan(self, qnbigl, qnbigr, qntot):
        cache_key = (
            qnbigl.tobytes(),
            qnbigr.tobytes(),
            np.asarray(qntot).tobytes(),
        )
        if cache_key in self._qr_qn_plan_cache:
            return self._qr_qn_plan_cache[cache_key]

        qntot = np.asarray(qntot)
        qn_size = len(qntot)
        localqnl = qnbigl.reshape(-1, qn_size)
        localqnr = qnbigr.reshape(-1, qn_size)

        seen = set()
        ordered_qn = []
        for qn in localqnl:
            qn_tuple = tuple(qn)
            if qn_tuple not in seen:
                seen.add(qn_tuple)
                ordered_qn.append(qn_tuple)

        plan = []
        for nl in ordered_qn:
            nl_array = np.asarray(nl)
            nr = qntot - nl_array
            rset = np.where(np.all(localqnr == nr, axis=-1))[0]
            if len(rset) == 0:
                continue
            lset = np.where(np.all(localqnl == nl_array, axis=-1))[0]
            plan.append((lset, rset, nl, nr))

        if len(plan) == 0:
            raise ValueError("Invalid quantum number")

        self._qr_qn_plan_cache[cache_key] = plan
        return plan

    def _batched_qr_qn(self, coef_batch, qnbigl, qnbigr, qntot, system, max_rank=None):
        assert system in ["L", "R"]

        batch_size = coef_batch.shape[0]
        left_dim = int(np.prod(qnbigl.shape[:-1]))
        right_dim = int(np.prod(qnbigr.shape[:-1]))
        coef_matrix = coef_batch.reshape(batch_size, left_dim, right_dim)

        u_blocks = []
        vt_blocks = []
        qnl_list = []
        qnr_list = []

        for lset, rset, nl, nr in self._get_qr_qn_plan(qnbigl, qnbigr, qntot):
            block = coef_matrix[:, lset][:, :, rset]

            if system == "L":
                u_block, vt_block = xp.linalg.qr(block, mode="reduced")
            else:
                q_t, r_t = xp.linalg.qr(xp.swapaxes(block, -1, -2), mode="reduced")
                u_block = xp.swapaxes(r_t, -1, -2)
                vt_block = xp.swapaxes(q_t, -1, -2)

            kdim = u_block.shape[-1]
            u_full = xp.zeros((batch_size, left_dim, kdim), dtype=coef_matrix.dtype)
            vt_full = xp.zeros((batch_size, kdim, right_dim), dtype=coef_matrix.dtype)
            u_full[:, lset, :] = u_block
            vt_full[:, :, rset] = vt_block

            u_blocks.append(u_full)
            vt_blocks.append(vt_full)
            qnl_list.extend([nl] * kdim)
            qnr_list.extend([nr.copy() for _ in range(kdim)])

        u_batch = xp.concatenate(u_blocks, axis=-1)
        vt_batch = xp.concatenate(vt_blocks, axis=1)
        if max_rank is not None:
            max_rank = max(1, int(max_rank))
            u_batch = u_batch[:, :, :max_rank]
            vt_batch = vt_batch[:, :max_rank, :]
            qnl_list = qnl_list[:max_rank]
            qnr_list = qnr_list[:max_rank]

        return u_batch, qnl_list, vt_batch, qnr_list

    def _build_site_batched_data(self, imps, l_tensors, r_tensors):
        batched_groups = []
        for template in self._site_group_templates[imps]:
            pair_ids = template["pair_ids"]
            batched_groups.append(
                {
                    "L": xp.stack([l_tensors[pair_id] for pair_id in pair_ids]),
                    "R": xp.stack([r_tensors[pair_id] for pair_id in pair_ids]),
                    "W": template["W"],
                    "alpha_idx": template["alpha_idx"],
                    "beta_idx": template["beta_idx"],
                    "nsite": template["nsite"],
                    "n_pairs": template["n_pairs"],
                }
            )
        return batched_groups

    def _build_reverse_batched_data(self, l_tensors, r_tensors):
        groups = {}
        for pair_id, l_tensor in enumerate(l_tensors):
            if l_tensor is None:
                continue
            key = (l_tensor.shape[1],)
            active_mpos_group = groups.setdefault(
                key,
                {
                    "L": [],
                    "R": [],
                    "alpha_idx": [],
                    "beta_idx": [],
                },
            )
            active_mpos_group["L"].append(l_tensor)
            active_mpos_group["R"].append(r_tensors[pair_id])
            active_mpos_group["alpha_idx"].append(self._active_pairs_index[pair_id][0])
            active_mpos_group["beta_idx"].append(self._active_pairs_index[pair_id][1])

        batched_groups = []
        for key in sorted(groups):
            active_mpos_group = groups[key]
            batched_groups.append(
                {
                    "L": xp.stack(active_mpos_group["L"]),
                    "R": xp.stack(active_mpos_group["R"]),
                    "W": None,
                    "alpha_idx": xp.asarray(active_mpos_group["alpha_idx"], dtype=np.int64),
                    "beta_idx": xp.asarray(active_mpos_group["beta_idx"], dtype=np.int64),
                    "nsite": 0,
                    "n_pairs": len(active_mpos_group["alpha_idx"]),
                }
            )
        return batched_groups

    def reset_mps(self, init_mp=None, msmps=None):
        self.MsMps = MultisetMps(
            self.MsModel,
            self.N_electron,
            temperature=self.temperature,
            init_model=self.init_model,
            method=self.method,
            init_mp=init_mp,
            msmps=msmps,
        )
        self.MsMps.ms_normalize("mps_only")
        self._invalidate_environ_cache()
        return self.MsMps

    def set_mps(self, ms_mps: MultisetMps):
        self.MsMps = ms_mps
        return self.MsMps

    def evolve_state(self, ms_mps: MultisetMps, evolve_dt, normalize=True) -> MultisetMps:
        method = {
            MsEvolveMethod.ms_evolve_tdvp_ps: self._ms_evolve_tdvp_ps,
            EvolveMethod.tdvp_ps: self._ms_evolve_tdvp_ps,
        }[self.evolve_config.method]

        if getattr(ms_mps, "electronic_ancilla", False):
            new_msmps = ms_mps.copy()
            for ancilla in range(ms_mps.n_anc):
                ancilla_state = ms_mps.ancilla_set_state(ancilla)
                evolved_ancilla_state = method(ms_mps_=ancilla_state, ms_mpo=self.MsMpo, evolve_dt=evolve_dt)
                for alpha in range(ms_mps.N_electron):
                    new_msmps.set_electron_ancilla_state(alpha, ancilla, evolved_ancilla_state.msmps[alpha])
        else:
            new_msmps = method(ms_mps_=ms_mps, ms_mpo=self.MsMpo, evolve_dt=evolve_dt)
        if normalize:
            norm = self._compute_multiset_norm(new_msmps)
            new_msmps.ms_normalize("mps_only")
            self._rescale_environ_cache_after_normalize(new_msmps, 1.0 / norm)
        return new_msmps

    def evolve(self, evolve_dt, normalize=True):
        self.MsMps = self.evolve_state(self.MsMps, evolve_dt, normalize=normalize)

    def _ms_evolve_tdvp_ps(self, ms_mps_: MultisetMps, ms_mpo: MultisetMpo, evolve_dt) -> "Mps":
        if np.iscomplex(evolve_dt):
            ms_mps = ms_mps_.copy()
            if hasattr(ms_mps_, "_environ_cache_token"):
                ms_mps._environ_cache_token = ms_mps_._environ_cache_token
            if self.evolve_config.ivp_solver != "krylov":
                evolve_dt = -evolve_dt.imag
                coef = -1
        else:
            ms_mps = ms_mps_.to_complex()
            if hasattr(ms_mps_, "_environ_cache_token"):
                ms_mps._environ_cache_token = ms_mps_._environ_cache_token
            if self.evolve_config.ivp_solver != "krylov":
                coef = 1j

        # The full N_electron conjugate copy is only consumed on a cache miss, and
        # even then only one alpha row at a time (see _build_environ_list).
        Environ_list = self._get_or_build_environ_list(ms_mps)

        local_steps = []
        for i in range(2):
            for imps in ms_mps.msmps[0].iter_idx_list(full=True):
                system = "L" if ms_mps.msmps[0].to_right else "R"
                shape_imps = list(ms_mps.msmps[0][imps].shape)
                dim = int(np.prod(shape_imps))

                l_array_ab = [environ.read("L", imps - 1) for environ in Environ_list]
                r_array_ab = [environ.read("R", imps + 1) for environ in Environ_list]
                batched_data = self._build_site_batched_data(imps, l_array_ab, r_array_ab)
                Y0 = xp.stack(
                    [asxp(ms_mps.msmps[a][imps].array).reshape(dim) for a in range(self.N_electron)]
                ).reshape(-1)
                ivp_eq = lambda Y: self._apply_hop_batched(Y, batched_data, dim, shape_imps)
                ivp_eq._reno_vector_block_count = self.N_electron
                if self.evolve_config.ivp_solver == "krylov":
                    mps_t, j = expm_krylov(ivp_eq, -1j * evolve_dt / 2, Y0, block_size=KRYLOV_BLOCK_SIZE)

                mps_t = mps_t.reshape((self.N_electron,) + tuple(shape_imps))
                local_steps.append(j)

                qnbigl, qnbigr, _ = ms_mps.msmps[0]._get_big_qn([imps])
                max_qr_rank = None
                if self.compress_config.criteria is not CompressCriteria.threshold:
                    self.compress_config.set_bonddim(len(ms_mps.msmps[0].bond_dims))
                    bond_idx = imps + 1 if system == "L" else imps
                    max_qr_rank = self.compress_config.max_dims[bond_idx]
                u_batch, qnlset, vt_batch, qnrset = self._batched_qr_qn(
                    mps_t,
                    qnbigl,
                    qnbigr,
                    ms_mps.msmps[0].qntot,
                    system,
                    max_rank=max_qr_rank,
                )

                if not ms_mps.msmps[0].to_right and imps != 0:
                    for alpha in range(self.N_electron):
                        ms_mps.msmps[alpha][imps] = vt_batch[alpha].reshape([-1] + shape_imps[1:])
                        ms_mps.msmps[alpha].qn[imps] = qnrset
                        ms_mps.msmps[alpha].qnidx = imps - 1

                    shapeU = list(u_batch[0].shape)
                    dimU = int(np.prod(shapeU))
                    r_array_u = [None for _ in range(len(self._active_pairs_index))]
                    for alpha in range(self.N_electron):
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
                    U0 = u_batch.reshape(self.N_electron, dimU).reshape(-1)

                    if self.evolve_config.ivp_solver == "krylov":
                        ivp_eq_Ut = lambda Y: self._apply_hop_batched(Y, batched_u, dimU, shapeU)
                        ivp_eq_Ut._reno_vector_block_count = self.N_electron
                        Ut, j2 = expm_krylov(ivp_eq_Ut, 1j * evolve_dt / 2, U0, block_size=KRYLOV_BLOCK_SIZE)

                    local_steps.append(j2)
                    Ut = Ut.reshape(self.N_electron, dimU)

                    for alpha in range(self.N_electron):
                        ms_mps.msmps[alpha][imps - 1] = tensordot(
                            ms_mps.msmps[alpha][imps - 1].array,
                            Ut[alpha].reshape(shapeU),
                            axes=(-1, 0),
                        )

                elif ms_mps.msmps[0].to_right and imps != len(ms_mps.msmps[0]) - 1:
                    for alpha in range(self.N_electron):
                        ms_mps.msmps[alpha][imps] = u_batch[alpha].reshape(shape_imps[:-1] + [-1])
                        ms_mps.msmps[alpha].qn[imps + 1] = qnlset
                        ms_mps.msmps[alpha].qnidx = imps + 1

                    shapeC = list(vt_batch[0].shape)
                    dimC = int(np.prod(shapeC))

                    l_array_c = [None for _ in range(len(self._active_pairs_index))]
                    for alpha in range(self.N_electron):
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
                    C0 = vt_batch.reshape(self.N_electron, dimC).reshape(-1)
                    ivp_eq_Ct = lambda Y: self._apply_hop_batched(Y, batched_c, dimC, shapeC)
                    ivp_eq_Ct._reno_vector_block_count = self.N_electron
                    if self.evolve_config.ivp_solver == "krylov":
                        Ct, j2 = expm_krylov(ivp_eq_Ct, 1j * evolve_dt / 2, C0, block_size=KRYLOV_BLOCK_SIZE)

                    local_steps.append(j2)
                    Ct = Ct.reshape(self.N_electron, dimC)

                    for alpha in range(self.N_electron):
                        ms_mps.msmps[alpha][imps + 1] = tensordot(
                            Ct[alpha].reshape(shapeC),
                            ms_mps.msmps[alpha][imps + 1].array,
                            axes=(1, 0),
                        )

                else:
                    for alpha in range(self.N_electron):
                        ms_mps.msmps[alpha][imps] = mps_t[alpha]
            for alpha in range(self.N_electron):
                ms_mps.msmps[alpha]._switch_direction()
        steps_stat = stats.describe(local_steps)
        logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
        self.evolve_config.stat = steps_stat
        self._store_environ_cache(ms_mps, Environ_list)

        return ms_mps

    def expand_bond_dimension_multiset(self, coef: float = 1e-10, use_hint: bool = True):
        if getattr(self.MsMps, "electronic_ancilla", False):
            source_state = self.MsMps
            expanded_state = source_state.copy()
            for ancilla in range(source_state.n_anc):
                ancilla_state = source_state.ancilla_set_state(ancilla)
                self.MsMps = ancilla_state
                self.expand_bond_dimension_multiset(coef=coef, use_hint=use_hint)
                for alpha in range(ancilla_state.N_electron):
                    expanded_state.set_electron_ancilla_state(alpha, ancilla, self.MsMps.msmps[alpha])
            self.MsMps = expanded_state
            return

        for alpha in range(self.N_electron):
            self.MsMps.msmps[alpha].compress_config = self.compress_config

        if not use_hint:
            for alpha in range(self.N_electron):
                expanded = expand_bond_dimension_general(
                    self.MsMps.msmps[alpha],
                    hint_mpo=None,
                    coef=coef,
                    ex_mps=None,
                )
                expanded.scale(float(abs(expanded.coeff)), inplace=True)
                expanded.coeff = 1.0
                self.MsMps.msmps[alpha] = expanded
            return

        original_mps = [self.MsMps.msmps[beta].copy() for beta in range(self.N_electron)]

        for alpha in range(self.N_electron):
            mps_alpha = original_mps[alpha]
            mps_alpha.compress_config = self.compress_config

            diag_mpo = self.MsMpo.msmpo[alpha][alpha]
            hint_mpo = diag_mpo if len(diag_mpo) > 0 else None

            cross_states = []
            for pair_id in self._active_pairs_by_alpha[alpha]:
                beta = self._active_pairs_index[pair_id][1]
                if beta == alpha:
                    continue
                driven = self._active_pair_mpos[pair_id].apply(original_mps[beta])
                cross_states.append(driven)

            ex_mps = _sum(cross_states, compress=False) if cross_states else None
            if ex_mps is not None:
                ex_mps.compress_config = self.compress_config

            expanded = expand_bond_dimension_general(
                mps_alpha,
                hint_mpo=hint_mpo,
                coef=coef,
                ex_mps=ex_mps,
            )

            expanded.scale(float(abs(expanded.coeff)), inplace=True)
            expanded.coeff = 1.0
            self.MsMps.msmps[alpha] = expanded
            
    def population(self):
        return self.MsMps.e_occupations_multiset

    def popultation(self):
        return self.population()

    def Hamiltonian(self):
        num = 0.0
        if getattr(self.MsMps, "electronic_ancilla", False):
            for ancilla in range(self.MsMps.n_anc):
                for pair_id, (alpha, beta) in enumerate(self._active_pairs_index):
                    num += _state_expectation(
                        self.MsMps.get_electron_ancilla_state(alpha, ancilla),
                        self.MsMps.get_electron_ancilla_state(beta, ancilla),
                        self._active_pair_mpos[pair_id],
                    )
            den = self.MsMps.total_norm()
        else:
            for pair_id, (alpha, beta) in enumerate(self._active_pairs_index):
                num += _state_expectation(
                    self.MsMps.msmps[alpha],
                    self.MsMps.msmps[beta],
                    self._active_pair_mpos[pair_id],
                )
            den = sum(_state_inner_product(mps, mps) for mps in self.MsMps.msmps)
        return (num / den).real

    def rho_el(self):
        return self.MsMps.rho_el()

    def decoherence_metrics(self):
        rho = self.rho_el()

        tr = np.trace(rho).real
        purity = np.trace(rho @ rho).real

        return tr, purity, rho

    def Inner_product(self) -> "float":
        if getattr(self.MsMps, "electronic_ancilla", False):
            return self.MsMps.total_norm()
        return sum(_state_inner_product(mps, mps) for mps in self.MsMps.msmps)

    def _apply_hop_batched(self, Y, batched_groups, dim, shape):
        N = self.N_electron
        Y = Y.reshape(N, dim)
        Y_out = xp.zeros((N, dim), dtype=Y.dtype)

        for group in batched_groups:
            L_all = group["L"]
            R_all = group["R"]
            beta_idx = group["beta_idx"]
            nsite = group["nsite"]
            n_pairs = group["n_pairs"]

            Y_exp = Y[beta_idx]

            if nsite == 1:
                W_all = group["W"]
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
            _scatter_add_rows(Y_out, group["alpha_idx"], out_flat)
        if self._energy_offset:
            Y_out -= self._energy_offset * Y
        return Y_out.ravel()
