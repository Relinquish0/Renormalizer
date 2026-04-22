# -*- coding: utf-8 -*-

import logging
from typing import List, Tuple

import numpy as np
from scipy import stats

from renormalizer.lib import expm_krylov
from renormalizer.model.model import Model
from renormalizer.model.op import Op
from renormalizer.mps import Mps
from renormalizer.mps.backend import xp
from renormalizer.mps.lib import Environ, _sum
from renormalizer.mps.matrix import asxp, tensordot
from renormalizer.mps.mpo import Mpo
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.multiset.multiset_mpo import MultisetMpo
from renormalizer.multiset.multiset_mps import MsEvolveMethod, MultisetMps
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, Quantity

logger = logging.getLogger(__name__)


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
        self.basis_set = [
            item for item in self.model.basis if type(item).__name__ != "BasisSimpleElectron"
        ]

        self.MsModel = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
        self.MsOp = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]

        self.SplitHamTerm()
        self.ConstructInitModel()
        self.ConstructMsModel()

        self.MsMpo = MultisetMpo(self.MsModel, self.N_electron)
        self._active_pairs: List[Tuple[int, int]] = []
        self._active_alpha: List[int] = []
        self._active_beta: List[int] = []
        self._active_pair_mpos: List[Mpo] = []
        self._active_pairs_by_alpha: List[List[int]] = [[] for _ in range(self.N_electron)]
        self._site_group_templates = []
        self._qr_qn_plan_cache = {}
        self._refresh_mpo_cache()
        self.MsMps = None
        if auto_init:
            logger.info(
                "MultisetModel no longer auto-initialises MsMps. "
                "Initial states should be prepared explicitly by a MultisetTdJob subclass."
            )

        self._matvec_calls = 0
        self._ivp_calls = 0

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
            return dof1, dof2
        return dof2, dof1

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
            new_op.symbol = "I"
            new_op.split_symbol = ["I"]
            new_op.qn_list = [np.array([0])]
            new_op.dofs = [tuple([0, 0])]
            return new_op

        new_op.symbol = " ".join(new_split_symbol)
        new_op.split_symbol = new_split_symbol
        new_op.qn_list = new_qn_list
        new_op.dofs = new_dofs
        return new_op

    def ConstructMsModel(self):
        for i in range(self.N_electron):
            for j in range(self.N_electron):
                self.MsModel[i][j] = Model(basis=self.basis_set, ham_terms=self.MsOp[i][j])

    def ConstructInitModel(self):
        init_terms = []
        for op in self.model.ham_terms:
            if len(op.dofs) == 1:
                init_terms.append(self._reset_all_MsOp(op))
        self.init_model = Model(basis=self.basis_set, ham_terms=init_terms)

    def _refresh_mpo_cache(self):
        active_pairs = []
        active_pair_mpos = []
        active_pairs_by_alpha = [[] for _ in range(self.N_electron)]
        site_group_dicts = None

        for alpha in range(self.N_electron):
            for beta in range(self.N_electron):
                mpo = self.MsMpo.msmpo[alpha][beta]
                if len(mpo) == 0:
                    continue
                pair_id = len(active_pairs)
                active_pairs.append((alpha, beta))
                active_pair_mpos.append(mpo)
                active_pairs_by_alpha[alpha].append(pair_id)

                if site_group_dicts is None:
                    site_group_dicts = [dict() for _ in range(len(mpo))]

                for imps, local_mpo in enumerate(mpo):
                    W = asxp(local_mpo.array)
                    key = tuple(W.shape)
                    bucket = site_group_dicts[imps].setdefault(
                        key,
                        {
                            "pair_ids": [],
                            "alpha_idx": [],
                            "beta_idx": [],
                            "w_tensors": [],
                        },
                    )
                    bucket["pair_ids"].append(pair_id)
                    bucket["alpha_idx"].append(alpha)
                    bucket["beta_idx"].append(beta)
                    bucket["w_tensors"].append(W)

        self._active_pairs = active_pairs
        self._active_pair_mpos = active_pair_mpos
        self._active_alpha = [alpha for alpha, _ in active_pairs]
        self._active_beta = [beta for _, beta in active_pairs]
        self._active_pairs_by_alpha = active_pairs_by_alpha
        self._site_group_templates = []

        if site_group_dicts is None:
            return

        for site_groups in site_group_dicts:
            templates = []
            for key in sorted(site_groups):
                bucket = site_groups[key]
                templates.append(
                    {
                        "pair_ids": tuple(bucket["pair_ids"]),
                        "alpha_idx": xp.asarray(bucket["alpha_idx"], dtype=np.int64),
                        "beta_idx": xp.asarray(bucket["beta_idx"], dtype=np.int64),
                        "S": self._build_scatter_matrix(
                            bucket["alpha_idx"], bucket["w_tensors"][0].real.dtype
                        ),
                        "W": xp.stack(bucket["w_tensors"]),
                        "nsite": 1,
                        "n_pairs": len(bucket["pair_ids"]),
                    }
                )
            self._site_group_templates.append(templates)

    def _build_scatter_matrix(self, alpha_idx, dtype):
        scatter = xp.zeros((self.N_electron, len(alpha_idx)), dtype=dtype)
        for i, alpha in enumerate(alpha_idx):
            scatter[alpha, i] = 1.0
        return scatter

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
                    "S": template["S"],
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
            bucket = groups.setdefault(
                key,
                {
                    "L": [],
                    "R": [],
                    "alpha_idx": [],
                    "beta_idx": [],
                },
            )
            bucket["L"].append(l_tensor)
            bucket["R"].append(r_tensors[pair_id])
            bucket["alpha_idx"].append(self._active_alpha[pair_id])
            bucket["beta_idx"].append(self._active_beta[pair_id])

        batched_groups = []
        for key in sorted(groups):
            bucket = groups[key]
            batched_groups.append(
                {
                    "L": xp.stack(bucket["L"]),
                    "R": xp.stack(bucket["R"]),
                    "S": self._build_scatter_matrix(bucket["alpha_idx"], bucket["L"][0].real.dtype),
                    "W": None,
                    "alpha_idx": xp.asarray(bucket["alpha_idx"], dtype=np.int64),
                    "beta_idx": xp.asarray(bucket["beta_idx"], dtype=np.int64),
                    "nsite": 0,
                    "n_pairs": len(bucket["alpha_idx"]),
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
        return self.MsMps

    def set_mps(self, ms_mps: MultisetMps):
        self.MsMps = ms_mps
        return self.MsMps

    def evolve_state(self, ms_mps: MultisetMps, evolve_dt, normalize=True) -> MultisetMps:
        method = {MsEvolveMethod.ms_evolve_tdvp_ps: self._ms_evolve_tdvp_ps}[self.evolve_config.method]

        new_msmps = method(ms_mps_=ms_mps, ms_mpo=self.MsMpo, evolve_dt=evolve_dt)
        if normalize:
            new_msmps.ms_normalize("mps_only")
        return new_msmps

    def evolve(self, evolve_dt, normalize=True):
        self.MsMps = self.evolve_state(self.MsMps, evolve_dt, normalize=normalize)

    def _ms_evolve_tdvp_ps(self, ms_mps_: MultisetMps, ms_mpo: MultisetMpo, evolve_dt) -> "Mps":
        if np.iscomplex(evolve_dt):
            ms_mps = ms_mps_.copy()
            if self.evolve_config.ivp_solver != "krylov":
                evolve_dt = -evolve_dt.imag
                coef = -1
        else:
            ms_mps = ms_mps_.to_complex()
            if self.evolve_config.ivp_solver != "krylov":
                coef = 1j

        conj_mps = [mps_alpha.conj() for mps_alpha in ms_mps.msmps]
        Environ_list = [
            Environ(
                ms_mps.msmps[self._active_beta[pair_id]],
                self._active_pair_mpos[pair_id],
                mps_conj=conj_mps[self._active_alpha[pair_id]],
            )
            for pair_id in range(len(self._active_pairs))
        ]

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
                if self.evolve_config.ivp_solver == "krylov":
                    mps_t, j = expm_krylov(ivp_eq, -1j * evolve_dt / 2, Y0)
                    self._ivp_calls += 1

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
                    r_array_u = [None for _ in range(len(self._active_pairs))]
                    for alpha in range(self.N_electron):
                        mps_conj_alpha = [None] * len(ms_mps.msmps[alpha])
                        mps_conj_alpha[imps] = ms_mps.msmps[alpha][imps].conj()
                        for pair_id in self._active_pairs_by_alpha[alpha]:
                            beta = self._active_beta[pair_id]
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
                        Ut, j2 = expm_krylov(ivp_eq_Ut, 1j * evolve_dt / 2, U0)
                        self._ivp_calls += 1

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

                    l_array_c = [None for _ in range(len(self._active_pairs))]
                    for alpha in range(self.N_electron):
                        mps_conj_alpha = [None] * len(ms_mps.msmps[alpha])
                        mps_conj_alpha[imps] = ms_mps.msmps[alpha][imps].conj()
                        for pair_id in self._active_pairs_by_alpha[alpha]:
                            beta = self._active_beta[pair_id]
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
                    if self.evolve_config.ivp_solver == "krylov":
                        Ct, j2 = expm_krylov(ivp_eq_Ct, 1j * evolve_dt / 2, C0)
                        self._ivp_calls += 1

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

        return ms_mps

    def expand_bond_dimension_multiset(self, coef: float = 1e-10, use_hint: bool = True):
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
                beta = self._active_beta[pair_id]
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
        bras = [self.MsMps.msmps[a].conj() for a in range(self.N_electron)]
        for pair_id, (alpha, beta) in enumerate(self._active_pairs):
            num += self.MsMps.msmps[beta].expectation(
                mpo=self._active_pair_mpos[pair_id],
                self_conj=bras[alpha],
            )
        den = sum(bras[a].dot(self.MsMps.msmps[a]) for a in range(self.N_electron))
        return (num / den).real

    def rho_el(self):
        return self.MsMps.rho_el()

    def decoherence_metrics(self):
        rho = self.rho_el()

        tr = np.trace(rho).real
        purity = np.trace(rho @ rho).real

        return tr, purity, rho

    def Inner_product(self) -> "float":
        return self.MsMps.total_mps().conj().dot(self.MsMps.total_mps())

    def _apply_hop_batched(self, Y, batched_groups, dim, shape):
        N = self.N_electron
        Y = Y.reshape(N, dim)
        Y_out = xp.zeros((N, dim), dtype=Y.dtype)

        for group in batched_groups:
            L_all = group["L"]
            R_all = group["R"]
            S = group["S"]
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
            Y_out += xp.matmul(S, out_flat)

        self._matvec_calls += 1
        return Y_out.ravel()
