# -*- coding: utf-8 -*-
# MultisetModel with batched MPS implementation

import logging
from typing import List
from enum import Enum
import numpy as np
from scipy import stats

from renormalizer.model.model import Model
from renormalizer.model.basis import BasisSimpleElectron
from renormalizer.model.op import Op
from renormalizer.utils import Quantity

from renormalizer.mps.mpo import Mpo
from renormalizer.mps import Mps
from renormalizer.mps.multiset_mps import MultisetMps
from renormalizer.mps.multiset_mpo import MultisetMpo
from renormalizer.mps.lib import Environ
from renormalizer.lib import expm_krylov
from renormalizer.mps.matrix import tensordot, asxp, asnumpy
from renormalizer.mps import svd_qn
from renormalizer.utils import CompressConfig, EvolveConfig
from renormalizer.mps.backend import xp

logger = logging.getLogger(__name__)


class MsEvolveMethod(Enum):
    ms_evolve_tdvp_ps = "TDVP PS one-site with multiset-mps ansatz"


class MultisetModel:
    def __init__(self, model: Model, max_bonddim: int):
        self.model = model
        self.evolve_config = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps)
        self.compress_config = CompressConfig(max_bonddim=max_bonddim)

        # Determine number of electronic states
        self.N_electron = self.model.ham_terms[-1].dofs[0] + 1
        self.basis_set = [item for item in self.model.basis
                         if type(item).__name__ != 'BasisSimpleElectron']

        # Build model grid for H^{α,β}
        self.MsModel = [[[] for _ in range(self.N_electron)]
                       for _ in range(self.N_electron)]
        self.MsOp = [[[] for _ in range(self.N_electron)]
                    for _ in range(self.N_electron)]

        self.SplitHamTerm()
        self.ConstructMsModel()

        # Initialize batched MPS and MPO
        self.MsMpo = MultisetMpo(self.MsModel, self.N_electron)
        self.MsMps = self._init_batched_mps()
        self.MsMps.normalize_multiset("mps_only")
        self.cdd_init_mps()

        # Performance counters
        self._matvec_calls = 0
        self._ivp_calls = 0

    def _init_batched_mps(self) -> MultisetMps:
        """Initialize batched MPS from Hartree product states."""
        ms_mps = MultisetMps(self.N_electron, self.MsModel[0][0])

        # Create N individual MPS
        temp_mps_list = []
        for alpha in range(self.N_electron):
            mps_alpha = Mps.hartree_product_state(model=self.MsModel[alpha][alpha])
            temp_mps_list.append(mps_alpha)

        # Stack into batched tensors [N, bond_L, phys, bond_R]
        n_sites = len(temp_mps_list[0])
        for site_idx in range(n_sites):
            tensors = [temp_mps_list[alpha][site_idx].array
                      for alpha in range(self.N_electron)]
            batched_tensor = np.stack(tensors, axis=0)
            ms_mps.append(batched_tensor)

        # Set quantum numbers (use first MPS as reference)
        ms_mps.qnidx = temp_mps_list[0].qnidx
        ms_mps.qntot = temp_mps_list[0].qntot
        ms_mps.to_right = temp_mps_list[0].to_right

        # Batched QN: stack QN arrays for all N states
        ms_mps.qn_batched = []
        for site_idx in range(len(temp_mps_list[0].qn)):
            qn_list = [temp_mps_list[alpha].qn[site_idx]
                      for alpha in range(self.N_electron)]
            ms_mps.qn_batched.append(np.stack(qn_list, axis=0))

        return ms_mps

    def SplitHamTerm(self):
        """Split Hamiltonian into H^{α,β} components."""
        for term in self.model.ham_terms:
            if len(term.dofs) == 2:
                self.MsOp[term.dofs[0]][term.dofs[1]].append(
                    self._reset_all_MsOp(term))
            elif len(term.dofs) == 1:
                for alpha in range(self.N_electron):
                    self.MsOp[alpha][alpha].append(
                        self._reset_all_MsOp(term))
            else:  # len(term.dofs) == 3
                self.MsOp[term.dofs[0]][term.dofs[1]].append(
                    self._reset_all_MsOp(term))

    def _reset_all_MsOp(self, op: Op) -> Op:
        """Convert operator a^dagger a to I for multiset formalism."""
        new_op = Op.product([op])
        new_split_symbol = []
        new_qn_list = []
        new_dofs = []

        i = 0
        while i < len(new_op.split_symbol):
            if (i + 1 < len(new_op.split_symbol) and
                new_op.split_symbol[i] == 'a^\\dagger' and
                new_op.split_symbol[i+1] == 'a'):

                if len(new_op.split_symbol) > i+2:
                    new_op.symbol = new_op.symbol.replace("a^\dagger a ", "")
                else:
                    new_op.symbol = new_op.symbol.replace("a^\dagger a", "I")
                    new_split_symbol.append('I')
                    new_qn_list.append(np.array([0]))
                    new_dofs.append(tuple([0,0]))
                i += 2
            else:
                new_split_symbol.append(new_op.split_symbol[i])
                new_qn_list.append(new_op.qn_list[i])
                new_dofs.append(new_op.dofs[i])
                i += 1

        new_op.split_symbol = new_split_symbol
        new_op.qn_list = new_qn_list
        new_op.dofs = new_dofs
        return new_op

    def ConstructMsModel(self):
        """Construct model grid from split operators."""
        for i in range(self.N_electron):
            for j in range(self.N_electron):
                self.MsModel[i][j] = Model(basis=self.basis_set,
                                           ham_terms=self.MsOp[i][j])

    def fc_excitation(self, alpha: int):
        """Franck-Condon excitation on site alpha."""
        # Get individual MPS for each state
        mps_list = []
        for beta in range(self.N_electron):
            mps_beta = self.MsMps.get_single_set(beta)
            if beta != alpha:
                mps_beta.scale(1e-10, inplace=True)
            mps_list.append(mps_beta)

        # Re-stack into batched MPS
        n_sites = len(mps_list[0])
        new_ms_mps = MultisetMps(self.N_electron, self.MsModel[0][0])
        for site_idx in range(n_sites):
            tensors = [mps_list[beta][site_idx].array
                      for beta in range(self.N_electron)]
            batched_tensor = np.stack(tensors, axis=0)
            new_ms_mps.append(batched_tensor)

        # Copy metadata
        new_ms_mps.qnidx = mps_list[0].qnidx
        new_ms_mps.qntot = mps_list[0].qntot
        new_ms_mps.to_right = mps_list[0].to_right
        new_ms_mps.qn_batched = []
        for site_idx in range(len(mps_list[0].qn)):
            qn_list = [mps_list[beta].qn[site_idx]
                      for beta in range(self.N_electron)]
            new_ms_mps.qn_batched.append(np.stack(qn_list, axis=0))

        self.MsMps = new_ms_mps
        # NOTE: Do NOT normalize here - it would undo the FC excitation scaling!

    def cdd_init_mps(self):
        """Initialize MPS with bond dimension expansion and FC excitation."""
        # Expand bond dimensions for all states
        expanded_mps_list = []
        for alpha in range(self.N_electron):
            mps_alpha = self.MsMps.get_single_set(alpha)
            mps_alpha.compress_config = self.compress_config
            mps_alpha = mps_alpha.expand_bond_dimension()
            # Normalize each MPS to norm 1.0 before FC excitation
            mps_alpha.normalize("mps_only")
            expanded_mps_list.append(mps_alpha)

        # Re-stack into batched MPS
        n_sites = len(expanded_mps_list[0])
        new_ms_mps = MultisetMps(self.N_electron, self.MsModel[0][0])
        for site_idx in range(n_sites):
            tensors = [expanded_mps_list[alpha][site_idx].array
                      for alpha in range(self.N_electron)]
            batched_tensor = np.stack(tensors, axis=0)
            new_ms_mps.append(batched_tensor)

        # Copy metadata
        new_ms_mps.qnidx = expanded_mps_list[0].qnidx
        new_ms_mps.qntot = expanded_mps_list[0].qntot
        new_ms_mps.to_right = expanded_mps_list[0].to_right
        new_ms_mps.qn_batched = []
        for site_idx in range(len(expanded_mps_list[0].qn)):
            qn_list = [expanded_mps_list[alpha].qn[site_idx]
                      for alpha in range(self.N_electron)]
            new_ms_mps.qn_batched.append(np.stack(qn_list, axis=0))

        self.MsMps = new_ms_mps

        # FC excitation on middle site (now all states start with norm 1.0)
        self.fc_excitation(self.N_electron // 2)
        energy = Quantity(self.Hamiltonian())

        # Set MPO offsets
        for alpha in range(self.N_electron):
            for beta in range(self.N_electron):
                if len(self.MsModel[alpha][beta].ham_terms) == 0:
                    self.MsMpo.msmpo[alpha][beta] = []
                elif alpha == beta:
                    self.MsMpo.msmpo[alpha][beta] = Mpo(
                        model=self.MsModel[alpha][beta], terms=None, offset=energy)
                else:
                    self.MsMpo.msmpo[alpha][beta] = Mpo(
                        model=self.MsModel[alpha][beta], terms=None, offset=Quantity(0))

        # Note: Canonicalization removed - it undoes the FC excitation scaling
        # The original code has a comment "seem to make no sense" about this step

    def popultation(self):
        """Electronic populations via batched dot products."""
        return self.MsMps.batched_dot(self.MsMps).real.tolist()

    def Hamiltonian(self):
        """Energy expectation <Ψ|H|Ψ> / <Ψ|Ψ>."""
        num = 0.0
        for alpha in range(self.N_electron):
            mps_alpha_conj = self.MsMps.get_single_set(alpha).conj()
            for beta in range(self.N_electron):
                if len(self.MsMpo.msmpo[alpha][beta]) == 0:
                    continue
                mps_beta = self.MsMps.get_single_set(beta)
                num += mps_beta.expectation(
                    mpo=self.MsMpo.msmpo[alpha][beta],
                    self_conj=mps_alpha_conj
                )

        den = sum(self.MsMps.batched_dot(self.MsMps))
        return (num / den).real

    def evolve(self, evolve_dt, normalize=True):
        """Time evolution using TDVP-PS."""
        method = {
            MsEvolveMethod.ms_evolve_tdvp_ps: self._ms_evolve_tdvp_ps
        }[self.evolve_config.method]

        new_msmps = method(ms_mps_=self.MsMps, ms_mpo=self.MsMpo,
                          evolve_dt=evolve_dt)
        self.MsMps = new_msmps
        self.MsMps.normalize_multiset("mps_only")

    def _ms_evolve_tdvp_ps(self, ms_mps_: MultisetMps, ms_mpo: MultisetMpo,
                          evolve_dt) -> MultisetMps:
        """TDVP projector splitting evolution (one-site)."""
        # Handle complex/real time
        if np.iscomplex(evolve_dt):
            ms_mps = ms_mps_.copy()
        else:
            ms_mps = ms_mps_.to_complex()

        # Extract individual MPS for Environ construction
        mps_list = [ms_mps.get_single_set(alpha) for alpha in range(self.N_electron)]

        # Construct N² Environ matrices
        Environ_list = [[[] for _ in range(self.N_electron)]
                       for _ in range(self.N_electron)]
        for alpha in range(self.N_electron):
            for beta in range(self.N_electron):
                Environ_list[alpha][beta] = Environ(
                    mps_list[beta], ms_mpo.msmpo[alpha][beta],
                    mps_conj=mps_list[alpha].conj())

        local_steps = []

        # Sweep for 2 rounds
        for i in range(2):
            for imps in mps_list[0].iter_idx_list(full=True):
                system = "L" if mps_list[0].to_right else "R"
                shape_imps = list(mps_list[0][imps].shape)
                dim = int(np.prod(shape_imps))

                # Construct L, R, W arrays for all (α,β) pairs
                l_array_ab = [[None] * self.N_electron for _ in range(self.N_electron)]
                r_array_ab = [[None] * self.N_electron for _ in range(self.N_electron)]
                w_array_ab = [[None] * self.N_electron for _ in range(self.N_electron)]

                for alpha in range(self.N_electron):
                    for beta in range(self.N_electron):
                        if len(ms_mpo.msmpo[alpha][beta]) == 0:
                            continue
                        l_array_ab[alpha][beta] = Environ_list[alpha][beta].read("L", imps - 1)
                        r_array_ab[alpha][beta] = Environ_list[alpha][beta].read("R", imps + 1)
                        w_array_ab[alpha][beta] = asxp(ms_mpo.msmpo[alpha][beta][imps].array)

                # Build batched data
                batched_data = self._build_batched_data(l_array_ab, r_array_ab, w_array_ab)

                # Stack Y0 from all states
                Y0 = xp.concatenate([asxp(mps_list[a][imps].ravel().array)
                                    for a in range(self.N_electron)])

                # Evolve
                ivp_eq = lambda Y: self._apply_hop_batched(Y, batched_data, dim, shape_imps)
                mps_t, j = expm_krylov(ivp_eq, -1j * evolve_dt / 2, Y0)
                self._ivp_calls += 1
                local_steps.append(j)

                mps_t = mps_t.reshape(self.N_electron, dim)

                # SVD for each MPS
                qnbigl, qnbigr, _ = mps_list[0]._get_big_qn([imps])
                u_list = []
                vt_list = []
                for alpha in range(self.N_electron):
                    u, qnlset, v, qnrset = svd_qn.svd_qn(
                        asnumpy(mps_t[alpha]), qnbigl, qnbigr,
                        mps_list[0].qntot, QR=True, system=system,
                        full_matrices=False)
                    u_list.append(asxp(u))
                    vt_list.append(asxp(v.T))

                # Apply SVD results
                if not mps_list[0].to_right and imps != 0:
                    for alpha in range(self.N_electron):
                        mps_list[alpha][imps] = vt_list[alpha].reshape([-1] + shape_imps[1:])
                        mps_list[alpha].qn[imps] = qnrset
                        mps_list[alpha].qnidx = imps-1

                    # Reverse U evolution
                    shapeU = list(u_list[0].shape)
                    dimU = int(np.prod(shapeU))

                    r_array_u = [[None] * self.N_electron for _ in range(self.N_electron)]
                    for alpha in range(self.N_electron):
                        mps_conj_alpha = [None] * len(mps_list[alpha])
                        mps_conj_alpha[imps] = mps_list[alpha][imps].conj()
                        for beta in range(self.N_electron):
                            if len(ms_mpo.msmpo[alpha][beta]) == 0:
                                continue
                            r_array_u[alpha][beta] = Environ_list[alpha][beta].GetLR(
                                "R", imps, mps_list[beta], ms_mpo.msmpo[alpha][beta],
                                itensor=r_array_ab[alpha][beta], method="System",
                                mps_conj=mps_conj_alpha)

                    batched_u = self._build_batched_data(l_array_ab, r_array_u)
                    U0 = xp.concatenate([u_list[alpha].ravel() for alpha in range(self.N_electron)])

                    ivp_eq_Ut = lambda Y: self._apply_hop_batched(Y, batched_u, dimU, shapeU)
                    Ut, j2 = expm_krylov(ivp_eq_Ut, 1j * evolve_dt / 2, U0)
                    self._ivp_calls += 1
                    local_steps.append(j2)
                    Ut = Ut.reshape(self.N_electron, dimU)

                    for alpha in range(self.N_electron):
                        mps_list[alpha][imps - 1] = tensordot(
                            mps_list[alpha][imps - 1].array,
                            Ut[alpha].reshape(shapeU), axes=(-1, 0))

                elif mps_list[0].to_right and imps != len(mps_list[0]) - 1:
                    for alpha in range(self.N_electron):
                        mps_list[alpha][imps] = u_list[alpha].reshape(shape_imps[:-1] + [-1])
                        mps_list[alpha].qn[imps + 1] = qnlset
                        mps_list[alpha].qnidx = imps+1

                    # Reverse C evolution
                    shapeC = list(vt_list[0].shape)
                    dimC = int(np.prod(shapeC))

                    l_array_c = [[None] * self.N_electron for _ in range(self.N_electron)]
                    for alpha in range(self.N_electron):
                        mps_conj_alpha = [None] * len(mps_list[alpha])
                        mps_conj_alpha[imps] = mps_list[alpha][imps].conj()
                        for beta in range(self.N_electron):
                            if len(ms_mpo.msmpo[alpha][beta]) == 0:
                                continue
                            l_array_c[alpha][beta] = Environ_list[alpha][beta].GetLR(
                                "L", imps, mps_list[beta], ms_mpo.msmpo[alpha][beta],
                                itensor=l_array_ab[alpha][beta], method="System",
                                mps_conj=mps_conj_alpha)

                    batched_c = self._build_batched_data(l_array_c, r_array_ab)
                    C0 = xp.concatenate([vt_list[alpha].ravel() for alpha in range(self.N_electron)])

                    ivp_eq_Ct = lambda Y: self._apply_hop_batched(Y, batched_c, dimC, shapeC)
                    Ct, j2 = expm_krylov(ivp_eq_Ct, 1j * evolve_dt / 2, C0)
                    self._ivp_calls += 1
                    local_steps.append(j2)
                    Ct = Ct.reshape(self.N_electron, dimC)

                    for alpha in range(self.N_electron):
                        mps_list[alpha][imps + 1] = tensordot(
                            Ct[alpha].reshape(shapeC),
                            mps_list[alpha][imps + 1].array, axes=(1, 0))

                else:
                    for alpha in range(self.N_electron):
                        mps_list[alpha][imps] = mps_t[alpha].reshape(shape_imps)

            for alpha in range(self.N_electron):
                mps_list[alpha]._switch_direction()

        steps_stat = stats.describe(local_steps)
        logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
        self.evolve_config.stat = steps_stat

        # Re-stack into batched MPS
        n_sites = len(mps_list[0])
        new_ms_mps = MultisetMps(self.N_electron, self.MsModel[0][0])
        for site_idx in range(n_sites):
            tensors = [mps_list[alpha][site_idx].array
                      for alpha in range(self.N_electron)]
            batched_tensor = np.stack(tensors, axis=0)
            new_ms_mps.append(batched_tensor)

        new_ms_mps.qnidx = mps_list[0].qnidx
        new_ms_mps.qntot = mps_list[0].qntot
        new_ms_mps.to_right = mps_list[0].to_right
        new_ms_mps.qn_batched = []
        for site_idx in range(len(mps_list[0].qn)):
            qn_list = [mps_list[alpha].qn[site_idx]
                      for alpha in range(self.N_electron)]
            new_ms_mps.qn_batched.append(np.stack(qn_list, axis=0))

        return new_ms_mps

    def _build_batched_data(self, l_arrays, r_arrays, w_arrays=None):
        """Build batched tensors grouped by operator shape."""
        N = self.N_electron
        nsite = 1 if w_arrays is not None else 0

        # Group pairs by tensor shapes
        groups = {}
        for alpha in range(N):
            for beta in range(N):
                if nsite == 1:
                    W = w_arrays[alpha][beta]
                    if W is None:
                        continue
                    key = (W.shape[0], W.shape[3])
                else:
                    L = l_arrays[alpha][beta]
                    if L is None:
                        continue
                    key = (L.shape[1],)

                if key not in groups:
                    groups[key] = []
                groups[key].append((alpha, beta))

        batched_groups = []
        if not groups:
            return batched_groups

        # Find dtype
        dtype = None
        for alpha in range(N):
            for beta in range(N):
                if l_arrays[alpha][beta] is not None:
                    dtype = l_arrays[alpha][beta].dtype
                    break
            if dtype is not None:
                break

        for key, pairs in groups.items():
            n_pairs = len(pairs)

            L_stack = xp.stack([l_arrays[a][b] for a, b in pairs])
            R_stack = xp.stack([r_arrays[a][b] for a, b in pairs])

            W_stack = None
            if nsite == 1:
                W_stack = xp.stack([w_arrays[a][b] for a, b in pairs])

            beta_idx = xp.array([b for _, b in pairs], dtype=xp.int64)

            # Scatter matrix
            S = xp.zeros((N, n_pairs), dtype=dtype)
            for i, (alpha, _) in enumerate(pairs):
                S[alpha, i] = 1.0

            batched_groups.append({
                'L': L_stack, 'R': R_stack, 'W': W_stack,
                'S': S, 'beta_idx': beta_idx, 'nsite': nsite,
                'n_pairs': n_pairs,
            })

        return batched_groups

    def _apply_hop_batched(self, Y, batched_groups, dim, shape):
        """Apply Hamiltonian using batched einsum."""
        N = self.N_electron
        Y = Y.reshape(N, dim)
        Y_out = xp.zeros((N, dim), dtype=Y.dtype)

        for group in batched_groups:
            L_all = group['L']
            R_all = group['R']
            S = group['S']
            beta_idx = group['beta_idx']
            nsite = group['nsite']
            n_pairs = group['n_pairs']

            Y_exp = Y[beta_idx]

            if nsite == 1:
                W_all = group['W']
                Y_exp = Y_exp.reshape(n_pairs, shape[0], shape[1], shape[2])

                temp = xp.einsum('ncek,nlfk->ncelf', Y_exp, R_all)
                temp2 = xp.einsum('ncelf,nbdef->ncdlb', temp, W_all)
                out = xp.einsum('ncdlb,nabc->nadl', temp2, L_all)
            else:
                Y_exp = Y_exp.reshape(n_pairs, shape[0], shape[1])

                temp = xp.einsum('nck,nlbk->nclb', Y_exp, R_all)
                out = xp.einsum('nclb,nabc->nal', temp, L_all)

            out_flat = out.reshape(n_pairs, dim)
            Y_out += xp.matmul(S, out_flat)

        self._matvec_calls += 1
        return Y_out.ravel()
