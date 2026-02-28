# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
from typing import List, Union, Dict, Callable
from collections import Counter
from enum import Enum
import numpy as np
from scipy import stats

from renormalizer.model.model import Model
from renormalizer.model.basis import BasisSet, BasisSimpleElectron, BasisMultiElectronVac, BasisHalfSpin, BasisSHO
from renormalizer.model.mol import Mol, Phonon
from renormalizer.model.op import Op, OpSum
from renormalizer.utils import Quantity, cached_property

from renormalizer.mps.mpo import Mpo
from renormalizer.mps import Mps
from renormalizer.mps.mps import adaptive_tdvp
from renormalizer.mps.lib import Environ, _sum
from renormalizer.lib import solve_ivp, expm_krylov
from renormalizer.mps.matrix import (
    multi_tensor_contract,
    ones,
    tensordot,
    Matrix,
    asnumpy,
    asxp)
from renormalizer.mps import svd_qn

from renormalizer.utils import (
    OptimizeConfig,
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod
)

from renormalizer.mps.backend import xp

logger = logging.getLogger(__name__)

class MsEvolveMethod(Enum):
    ms_evolve_tdvp_ps = "TDVP PS one-site with multiset-mps ansatz"

class MultisetMpo:
    """
    Docstring for MultisetMpo
    """
    def __init__(self, msmodel, N_electron: int):
        self.MsModel = msmodel
        self.N_electron = N_electron
        self.msmpo = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
        self._ConstructMsMpo()

    def _ConstructMsMpo(self):
        for i in range(self.N_electron):
            for j in range(self.N_electron):
                self.msmpo[i][j] = Mpo(model=self.MsModel[i][j],terms=None)

    def total_mpo(self) -> "Mpo":
        mpos = []
        for alpha in range(self.N_electron):
            mpos.append(_sum(mps_list=self.msmpo[alpha],compress=False,temp_m_trunc=None))
        return _sum(mps_list=mpos,compress=False,temp_m_trunc=None)

class MultisetMps:
    """
    Docstring for MultisetMps
    """
    def __init__(self, msmodel, N_electron: int):
        self.MsModel = msmodel
        self.N_electron = N_electron
        self.msmps = [[] for _ in range(self.N_electron)] 
        self._ConstructMsMps()

    def _ConstructMsMps(self):
        for i in range(self.N_electron):
            self.msmps[i]=Mps.hartree_product_state(model=self.MsModel[i][i])
    
    def copy(self) -> "MultisetMps":
        """  
        Create a deep copy of an entire MultisetMps object  
        """   
        new = MultisetMps.__new__(MultisetMps)
        new.MsModel = self.MsModel
        new.N_electron = self.N_electron
        new.msmps = [m.copy() for m in self.msmps]   
        return new

    def to_complex(self) -> "MultisetMps":
        """  
        Create a deep copy of a complex MultisetMps object  
        """   
        new = MultisetMps.__new__(MultisetMps)
        new.MsModel = self.MsModel
        new.N_electron = self.N_electron
        new.msmps = [m.to_complex() for m in self.msmps] 
        return new

    def total_mps(self) -> "Mps":
        return _sum(mps_list=self.msmps,compress=False,temp_m_trunc=None)
    
    def ms_normalize(self, kind):
        r''' normalize the wavefunction

        Parameters
        ----------
        kind: str
            "mps_only": the mps part is normalized and coeff is not modified;
            "mps_norm_to_coeff": the mps part is normalized and the norm is multiplied to coeff;
            "mps_and_coeff": both mps and coeff is normalized

        Returns
        -------
        ``self`` is overwritten.
        '''

        total_tn_coeff = 0

        for alpha in range(self.N_electron):
            total_tn_coeff += self.msmps[alpha].conj().dot(self.msmps[alpha])

        total_tn_coeff = total_tn_coeff ** 0.5

        if kind in ["mps_only"]:
            for alpha in range(self.N_electron):
                self.msmps[alpha].scale(1.0 / total_tn_coeff, inplace=True)

        # to do. I don't know how to distribute coeff and total_tn_coeff

        # elif kind in ["mps_and_coeff"]:
        #     for alpha in range(self.N_electron):
        #         self.msmps[alpha].scale(1.0 / total_tn_coeff, inplace=True)
        #         self.msmps[alpha].coeff = self.msmps[alpha].coeff / np.linalg.norm(self.msmps[alpha].coeff)
        # elif kind in ["mps_norm_to_coeff"]:
        #     new_coeff = tn.coeff * tn_norm
        else:
            raise ValueError(f"kind={kind} is not valid.")

class MultisetModel:
    def __init__(self, model: Model, max_bonddim): 
        self.model = model
        self.evolve_config: EvolveConfig = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps)
        self.compress_config: CompressConfig  = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bonddim)
        self.N_electron = self.model.ham_terms[-1].dofs[0] + 1 # This may consult bug!!! 
        self.basis_set = [item for item in self.model.basis if type(item).__name__ != 'BasisSimpleElectron'] # casting the electron terms

        self.MsModel = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
        self.MsOp = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]

        # for i in range(len(self.model.ham_terms)):
        #     print(self.model.ham_terms[i])

        self.SplitHamTerm()

        # for i in range(self.N_electron):
        #     for j in range(self.N_electron):
        #         print("="*20,i,j,"="*20)
        #         for k in range(len(self.MsOp[i][j])):
        #             print(self.MsOp[i][j][k])

        self.ConstructMsModel()

        self.MsMpo = MultisetMpo(self.MsModel,self.N_electron)
        self.MsMps = MultisetMps(self.MsModel,self.N_electron)
        self.MsMps.ms_normalize("mps_only")
        self.cdd_init_mps()

        self._matvec_calls = 0
        self._ivp_calls = 0

    def SplitHamTerm(self):
        # Convert Hamiltonian from H to H^{\alpha,\beta} and storage in self.MsOp
        for i in range(len(self.model.ham_terms)):
            if len(self.model.ham_terms[i].dofs) == 2:             
                self.MsOp[self.model.ham_terms[i].dofs[0]][self.model.ham_terms[i].dofs[1]].append(self._reset_all_MsOp(self.model.ham_terms[i]))
            elif len(self.model.ham_terms[i].dofs) == 1:
                for alpha in range(self.N_electron):
                    self.MsOp[alpha][alpha].append(self._reset_all_MsOp(self.model.ham_terms[i]))             
                # self.MsOp[self.model.ham_terms[i].dofs[0][0]][self.model.ham_terms[i].dofs[0][0]].append(self._reset_all_MsOp(self.model.ham_terms[i]))
            else: # len(self.model.ham_terms[i].dofs) == 3
                self.MsOp[self.model.ham_terms[i].dofs[0]][self.model.ham_terms[i].dofs[1]].append(self._reset_all_MsOp(self.model.ham_terms[i]))

    def _reset_all_MsOp(self, op:Op):
        # 1. Convert operator "a^\dagger a " to "I" and change the property of each Op
        # 2. Since only BasisSHO survives, the dofs change, consulting the split_symbol/qn_list change.
        new_op = Op.product([op])
        new_split_symbol = []
        new_qn_list = []
        new_dofs = []

        i = 0
        while i < len(new_op.split_symbol):
            # search 'a^\\dagger', 'a' 
            if (new_op.split_symbol[i] == 'a^\\dagger' and new_op.split_symbol[i+1] == 'a'):

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
        for i in range(self.N_electron):
            for j in range(self.N_electron):
                self.MsModel[i][j] = Model(basis=self.basis_set, ham_terms=self.MsOp[i][j])

    def evolve(self, evolve_dt, normalize=True):

        method = {
            MsEvolveMethod.ms_evolve_tdvp_ps: self._ms_evolve_tdvp_ps
        }[self.evolve_config.method]

        new_msmps = method(ms_mps_=self.MsMps,ms_mpo=self.MsMpo, evolve_dt=evolve_dt)
        self.MsMps = new_msmps
        self.MsMps.ms_normalize("mps_only")
        # if normalize:
        #     if np.iscomplex(evolve_dt):
        #         self.MsMps.ms_normalize("mps_and_coeff")
        #     else:
        #         self.MsMps.ms_normalize("mps_only")

    # @adaptive_tdvp
    def _ms_evolve_tdvp_ps(self, ms_mps_:MultisetMps, ms_mpo:MultisetMpo, evolve_dt) -> "Mps":
        # PhysRevB.94.165116
        # TDVP projector splitting
        # one-site
        if np.iscomplex(evolve_dt):
            ms_mps = ms_mps_.copy()
            if self.evolve_config.ivp_solver != "krylov":
                evolve_dt = -evolve_dt.imag
                # used in calculating derivatives
                coef = -1
        else:
            ms_mps = ms_mps_.to_complex()
            if self.evolve_config.ivp_solver != "krylov":
                coef = 1j

        # construct the environment matrix list.
        # almost half is not used. Not a big deal.
        Environ_list = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
        for alpha in range(self.N_electron):
            for beta in range(self.N_electron):
                Environ_list[alpha][beta] = Environ(ms_mps.msmps[beta], ms_mpo.msmpo[alpha][beta], mps_conj=ms_mps.msmps[alpha].conj())

        # statistics for debug output
        local_steps = []
        # sweep for 2 rounds
        for i in range(2):
            for imps in ms_mps.msmps[0].iter_idx_list(full=True): # All mps in msmps are same             
                    system = "L" if ms_mps.msmps[0].to_right else "R"
                    shape_imps = list(ms_mps.msmps[0][imps].shape)
                    dim = int(np.prod(shape_imps))
                    
                    # Construt the sum of efficient Hamiltonian
                    l_array_ab = [[None for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                    r_array_ab = [[None for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                    w_array_ab = [[None for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                    for alpha in range(self.N_electron):
                        for beta in range(self.N_electron):
                            l_array_ab[alpha][beta] = Environ_list[alpha][beta].read("L", imps - 1)
                            r_array_ab[alpha][beta] = Environ_list[alpha][beta].read("R", imps + 1)
                            w_array_ab[alpha][beta] = asxp(ms_mpo.msmpo[alpha][beta][imps].array)

                    batched_data = self._build_batched_data(l_array_ab, r_array_ab, w_array_ab)
                    Y0 = xp.concatenate([asxp(ms_mps.msmps[a][imps].ravel().array) for a in range(self.N_electron)])
                    # Construct the partial differential equation of Multiset TDVP
                    ivp_eq = lambda Y: self._apply_hop_batched(Y, batched_data, dim, shape_imps)
                    if self.evolve_config.ivp_solver == "krylov":
                        mps_t, j = expm_krylov(
                            ivp_eq,
                            -1j * evolve_dt / 2,
                            Y0)
                        self._ivp_calls += 1
                    # This part has not been changed yet.
                    # else:
                    #     sol = solve_ivp(
                    #         lambda t, y: sum(hop_beta(y.reshape(shape)).ravel() for hop_beta in hop_list)/coef, # In this line "y:" is different from origin code
                    #         (0, evolve_dt/2),
                    #         mps_alpha[imps].ravel().array,
                    #         method=self.evolve_config.ivp_solver,
                    #         rtol=self.evolve_config.ivp_rtol,
                    #         atol=self.evolve_config.ivp_atol,
                    #     )
                    #     mps_t, j = sol.y, sol.nfev

                    mps_t = mps_t.reshape(self.N_electron, dim)
                    local_steps.append(j)

                    # SVD decomposition for each mps_alpha
                    qnbigl, qnbigr, _ = ms_mps.msmps[0]._get_big_qn([imps])
                    u_list = [[] for _ in range(self.N_electron)]
                    vt_list = [[] for _ in range(self.N_electron)]
                    for alpha in range(self.N_electron):
                        u, qnlset, v, qnrset = svd_qn.svd_qn(
                            asnumpy(mps_t[alpha]),
                            qnbigl,
                            qnbigr,
                            ms_mps.msmps[0].qntot,
                            QR=True,
                            system=system,
                            full_matrices=False,
                        )
                        u_list[alpha] = asxp(u)
                        vt_list[alpha] = asxp(v.T)

                    if not ms_mps.msmps[0].to_right and imps != 0:
                        for alpha in range(self.N_electron):
                            ms_mps.msmps[alpha][imps] = vt_list[alpha].reshape([-1] + shape_imps[1:])
                            ms_mps.msmps[alpha].qn[imps] = qnrset
                            ms_mps.msmps[alpha].qnidx = imps-1

                        shapeU = list(u_list[0].shape)
                        dimU = int(np.prod(shapeU))
                        # Construct reverse U environment tensors
                        r_array_u = [[None for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                        for alpha in range(self.N_electron):
                            mps_conj_alpha = [None] * len(ms_mps.msmps[alpha])
                            mps_conj_alpha[imps] = ms_mps.msmps[alpha][imps].conj()
                            for beta in range(self.N_electron):
                                r_array_u[alpha][beta] = Environ_list[alpha][beta].GetLR(
                                    "R", imps, ms_mps.msmps[beta], ms_mpo.msmpo[alpha][beta], itensor=r_array_ab[alpha][beta], method="System",
                                    mps_conj=mps_conj_alpha
                                )

                        batched_u = self._build_batched_data(l_array_ab, r_array_u)
                        U0 = xp.concatenate([u_list[alpha].ravel() for alpha in range(self.N_electron)])

                        if self.evolve_config.ivp_solver == "krylov":
                            ivp_eq_Ut = lambda Y: self._apply_hop_batched(Y, batched_u, dimU, shapeU)
                            Ut, j2 = expm_krylov(
                                ivp_eq_Ut,
                                1j * evolve_dt / 2,
                                U0
                            )
                            self._ivp_calls += 1
                        # This part has not been changed yet.
                        # else:
                        #     sol = solve_ivp(
                        #         lambda t, y: sum(hop_u_beta(y.reshape(shape_u)).ravel() for hop_u_beta in hop_u_list)/ -coef,
                        #         (0, evolve_dt/2),
                        #         u.ravel(),
                        #         method=self.evolve_config.ivp_solver,
                        #         rtol=self.evolve_config.ivp_rtol,
                        #         atol=self.evolve_config.ivp_atol,
                        #     )
                        #     mps_t, j = sol.y, sol.nfev

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
                            ms_mps.msmps[alpha][imps] = u_list[alpha].reshape(shape_imps[:-1] + [-1])
                            ms_mps.msmps[alpha].qn[imps + 1] = qnlset
                            ms_mps.msmps[alpha].qnidx = imps+1

                        shapeC = list(vt_list[0].shape)
                        dimC = int(np.prod(shapeC))

                        # Construct reverse C environment tensors
                        l_array_c = [[None for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                        for alpha in range(self.N_electron):
                            mps_conj_alpha = [None] * len(ms_mps.msmps[alpha])
                            mps_conj_alpha[imps] = ms_mps.msmps[alpha][imps].conj()
                            for beta in range(self.N_electron):
                                l_array_c[alpha][beta] = Environ_list[alpha][beta].GetLR(
                                    "L", imps, ms_mps.msmps[beta], ms_mpo.msmpo[alpha][beta], itensor=l_array_ab[alpha][beta], method="System",
                                    mps_conj=mps_conj_alpha
                                )

                        batched_c = self._build_batched_data(l_array_c, r_array_ab)
                        C0 = xp.concatenate([vt_list[alpha].ravel() for alpha in range(self.N_electron)])
                        ivp_eq_Ct = lambda Y: self._apply_hop_batched(Y, batched_c, dimC, shapeC)
                        if self.evolve_config.ivp_solver == "krylov":
                            Ct, j2 = expm_krylov(
                                ivp_eq_Ct,
                                1j * evolve_dt / 2,
                                C0
                            )
                            self._ivp_calls += 1
                        # else:
                        #     sol = solve_ivp(
                        #         lambda t, y: sum(hop_svt_beta(y.reshape(shape_svt)).ravel() for hop_svt_beta in hop_svt_list) / -coef,
                        #         (0, evolve_dt/2),
                        #         vt.ravel(),
                        #         method=self.evolve_config.ivp_solver,
                        #         rtol=self.evolve_config.ivp_rtol,
                        #         atol=self.evolve_config.ivp_atol,
                        #     )
                        #     mps_t, j = sol.y, sol.nfev

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
                            ms_mps.msmps[alpha][imps] = mps_t[alpha].reshape(shape_imps)
            for alpha in range(self.N_electron):
                ms_mps.msmps[alpha]._switch_direction()
        steps_stat = stats.describe(local_steps)
        logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
        self.evolve_config.stat = steps_stat
        
        return ms_mps

    def fc_excitation(self,alpha:int):
        '''
        Franck-Condon excitation on the alpha site, the multiset-formalism
        '''

        for beta in range(self.N_electron):
            if beta != alpha: 
                self.MsMps.msmps[beta].scale(1e-10,inplace=True)

                # set the mps[beta] as 0 directly. But this method would consult the assert in other code.
                # for i in range(len(self.MsMps.msmps[beta])):    
                #     self.MsMps.msmps[beta][i].array = np.zeros(shape=self.MsMps.msmps[beta][i].shape, dtype=self.MsMps.msmps[beta][i].dtype) 
        self.MsMps.ms_normalize("mps_only")

    def cdd_init_mps(self):
        '''
        This is a copy from ChargeDiffusionDynamics.init_mps

        In ChargeDiffusionDynamics.init_mps: excitation/creat electron -> set Mpo's offset -> expand bond dimension -> cononicalise
        In cdd_init_mps: expand bond dimension -> excitation -> offset
        '''

        for alpha in range(self.N_electron):
            self.MsMps.msmps[alpha].compress_config = self.compress_config
            self.MsMps.msmps[alpha] = self.MsMps.msmps[alpha].expand_bond_dimension() # Now is random expanded   

        self.fc_excitation(self.N_electron // 2)      
        energy = Quantity(self.Hamiltonian())

        for alpha in range(self.N_electron):
            for beta in range(self.N_electron):
                # tentative_mpo = self.MsMpo.msmpo[alpha][beta]

                if alpha == beta:
                    self.MsMpo.msmpo[alpha][beta] = Mpo(model = self.MsModel[alpha][beta], terms = None, offset = energy)
                else:
                    self.MsMpo.msmpo[alpha][beta] = Mpo(model = self.MsModel[alpha][beta], terms = None, offset = Quantity(0))

        for alpha in range(self.N_electron):
            self.MsMps.msmps[alpha].canonicalise() # seem to make no sense  


    def popultation(self):
        # strategy: by computing <\Psi^\alpha|\Psi^\alpha>
        bras = [self.MsMps.msmps[a].conj() for a in range(self.N_electron)]
        return [bras[a].dot(self.MsMps.msmps[a]).real for a in range(self.N_electron)]

    # def Hamiltonian(self) -> "float":
    #     return self.MsMps.total_mps().expectation(self.MsMpo.total_mpo())

    def Hamiltonian(self):
        # <Psi|H|Psi> / <Psi|Psi>
        num = 0.0
        for alpha in range(self.N_electron):
            for beta in range(self.N_electron):
                num += self.MsMps.msmps[beta].expectation(
                    mpo=self.MsMpo.msmpo[alpha][beta],
                    self_conj=self.MsMps.msmps[alpha].conj()
                )
        den = sum(self.MsMps.msmps[a].conj().dot(self.MsMps.msmps[a]) for a in range(self.N_electron))
        return (num / den).real

    def rho_el(self):
        """Electronic reduced density matrix rho_{ab} = <Psi^a|Psi^b>."""
        Ne = self.N_electron
        rho = np.zeros((Ne, Ne), dtype=np.complex128)
        for a in range(Ne):
            bra = self.MsMps.msmps[a].conj()
            for b in range(Ne):
                rho[a, b] = bra.dot(self.MsMps.msmps[b])
        return rho

    def decoherence_metrics(self):
        """
        Return (trace, purity, entropy, rho_el).
        purity = Tr(rho^2), entropy = -Tr(rho log rho) (natural log).
        """
        rho = self.rho_el()

        tr = np.trace(rho).real
        purity = np.trace(rho @ rho).real

        return tr, purity, rho

    def Inner_product(self) -> "float":
        return self.MsMps.total_mps().conj().dot(self.MsMps.total_mps())

    def add_environ_tensors(self, environ1, environ2):  
        """  
        Parameters  
        ----------  
        environ1: Environ 
        environ2: Environ 

        Returns  
        -------  
        Environ  
        """  
        # 创建新的 Environ 对象  
        result = Environ.__new__(Environ)  
        result._virtual_disk = {}  
        result.sentinel = environ1.sentinel  
        
        # 获取所有键的并集  
        all_keys = set(environ1._virtual_disk.keys()) | set(environ2._virtual_disk.keys())  
        
        # 对应相加张量  
        for key in all_keys:  
            if key in environ1._virtual_disk and key in environ2._virtual_disk:  
                # 两个都有，相加  
                tensor1 = environ1.read(key[0], key[1])  
                tensor2 = environ2.read(key[0], key[1])  
                result.write(key[0], key[1], tensor1 + tensor2)  
            elif key in environ1._virtual_disk:  
                # 只有 environ1 有  
                tensor = environ1.read(key[0], key[1])  
                result.write(key[0], key[1], tensor.copy())  
            else:  
                # 只有 environ2 有  
                tensor = environ2.read(key[0], key[1])  
                result.write(key[0], key[1], tensor.copy())  
        
        return result
    
    def _build_batched_data(self, l_arrays, r_arrays, w_arrays=None):
        """Build batched tensors grouped by operator shape (no padding).

        Groups all N² (alpha, beta) pairs by their MPO bond dimensions,
        then stacks tensors within each group. Since all tensors in a group
        have identical shapes, no zero-padding is needed.

        For FMO (N=7): diagonal pairs (M=2, 7 pairs) and off-diagonal
        pairs (M=1, 42 pairs) form 2 groups.

        Args:
            l_arrays: N×N nested list of L tensors (cupy arrays on GPU)
            r_arrays: N×N nested list of R tensors (cupy arrays on GPU)
            w_arrays: N×N nested list of W tensors for nsite=1, None for nsite=0

        Returns:
            list of group dicts for _apply_hop_batched
        """
        N = self.N_electron
        nsite = 1 if w_arrays is not None else 0

        # Group pairs by tensor shapes (MPO bond dimensions)
        groups = {}
        for alpha in range(N):
            for beta in range(N):
                if nsite == 1:
                    W = w_arrays[alpha][beta]
                    key = (W.shape[0], W.shape[3])  # (M_left, M_right)
                else:
                    key = (l_arrays[alpha][beta].shape[1],)  # (M,)
                if key not in groups:
                    groups[key] = []
                groups[key].append((alpha, beta))

        # For each group, stack tensors (no padding needed)
        batched_groups = []
        dtype = l_arrays[0][0].dtype
        for key, pairs in groups.items():
            n_pairs = len(pairs)

            L_stack = xp.stack([l_arrays[a][b] for a, b in pairs])
            R_stack = xp.stack([r_arrays[a][b] for a, b in pairs])

            W_stack = None
            if nsite == 1:
                W_stack = xp.stack([w_arrays[a][b] for a, b in pairs])

            beta_idx = xp.array([b for _, b in pairs], dtype=xp.int64)

            # Per-group scatter matrix: (N, n_pairs)
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
        """Apply hop operator using shape-grouped batched einsum.

        Iterates over groups of (alpha, beta) pairs that share the same
        MPO bond dimensions. Within each group, uses batched einsum
        (no padding). Accumulates results across groups.

        For nsite=1 (main site): 3 einsums per group
            "abc, bdef, lfk, cek -> adl" decomposed as:
            Step 1: Y⊗R  →  "ncek, nlfk -> ncelf"  (contract over k)
            Step 2: ⊗W   →  "ncelf, nbdef -> ncdlb" (contract over e, f)
            Step 3: ⊗L   →  "ncdlb, nabc -> nadl"   (contract over c, b)

        For nsite=0 (reverse U/C): 2 einsums per group
            "abc, lbk, ck -> al" decomposed as:
            Step 1: Y⊗R  →  "nck, nlbk -> nclb"     (contract over k)
            Step 2: ⊗L   →  "nclb, nabc -> nal"      (contract over c, b)
        """
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

            # Expand Y: Y_exp[i] = Y[beta_of_pair_i]
            Y_exp = Y[beta_idx]  # (n_pairs, dim)

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

            # Scatter: accumulate results to alpha
            out_flat = out.reshape(n_pairs, dim)
            Y_out += xp.matmul(S, out_flat)  # (N, dim)

        self._matvec_calls += 1
        return Y_out.ravel()

