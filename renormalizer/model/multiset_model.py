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
from renormalizer.mps.hop_expr import hop_expr

from renormalizer.utils import (
    OptimizeConfig,
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod
)

from renormalizer.mps.backend import xp
from concurrent.futures import ThreadPoolExecutor

import concurrent.futures

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
                    l_array_ab = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]                    
                    r_array_ab = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                    hop_list = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                    for alpha in range(self.N_electron):
                        for beta in range(self.N_electron):
                            l_array_ab[alpha][beta] = Environ_list[alpha][beta].read("L", imps - 1)
                            r_array_ab[alpha][beta] = Environ_list[alpha][beta].read("R", imps + 1)
                            hop_list[alpha][beta] = hop_expr(l_array_ab[alpha][beta], r_array_ab[alpha][beta], [asxp(ms_mpo.msmpo[alpha][beta][imps].array)], shape_imps)

                    Y0 = xp.concatenate([asxp(ms_mps.msmps[a][imps].ravel().array) for a in range(self.N_electron)])

                    # Construct the partial differential equation of Multiset TDVP
                    if self.evolve_config.ivp_solver == "krylov":
                        mps_t, j = expm_krylov(
                            lambda Y: self._apply_hop_list(Y, hop_list, dim, shape_imps),
                            -1j * evolve_dt / 2,
                            Y0)
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
                        # Construct hop_u list
                        hop_u_list = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                        for alpha in range(self.N_electron):
                            mps_conj_alpha = [None] * len(ms_mps.msmps[alpha])
                            mps_conj_alpha[imps] = ms_mps.msmps[alpha][imps].conj()
                            for beta in range(self.N_electron):
                                r_array = Environ_list[alpha][beta].GetLR(
                                    "R", imps, ms_mps.msmps[beta], ms_mpo.msmpo[alpha][beta], itensor=r_array_ab[alpha][beta], method="System", 
                                    mps_conj=mps_conj_alpha
                                )
                                # reverse update u site
                                hop_u_list[alpha][beta] = hop_expr(l_array_ab[alpha][beta], r_array, [], shapeU)

                        U0 = xp.concatenate([u_list[alpha].ravel() for alpha in range(self.N_electron)])

                        if self.evolve_config.ivp_solver == "krylov":
                            Ut, j2 = expm_krylov(
                                lambda Y: self._apply_hop_list(Y, hop_u_list, dimU, shapeU),
                                1j * evolve_dt / 2,
                                U0
                            )
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

                        # Construct hop_svt list
                        hop_svt_list = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                        for alpha in range(self.N_electron):
                            mps_conj_alpha = [None] * len(ms_mps.msmps[alpha])
                            mps_conj_alpha[imps] = ms_mps.msmps[alpha][imps].conj()
                            for beta in range(self.N_electron):
                                l_array = (Environ_list[alpha][beta].GetLR(
                                    "L", imps, ms_mps.msmps[beta], ms_mpo.msmpo[alpha][beta], itensor=l_array_ab[alpha][beta], method="System", 
                                    mps_conj=mps_conj_alpha
                                ))
                                # reverse update svt site
                                hop_svt_list[alpha][beta] = hop_expr(l_array, r_array_ab[alpha][beta], [], shapeC)
                        
                        C0 = xp.concatenate([vt_list[alpha].ravel() for alpha in range(self.N_electron)])

                        if self.evolve_config.ivp_solver == "krylov":
                            Ct, j2 = expm_krylov(
                                lambda Y: self._apply_hop_list(Y, hop_svt_list, dimC, shapeC),
                                1j * evolve_dt / 2,
                                C0
                            )
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
        # steps_stat = stats.describe(local_steps)
        # logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
        # self.evolve_config.stat = steps_stat
        
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
    
    def _apply_hop_list(self, Y, hop_list, dim, shape):
        # 预先切分 Y，避免在循环中重复 reshape
        Y = Y.reshape(self.N_electron, dim)
        Y_out = xp.zeros((self.N_electron, dim), dtype=Y.dtype)

        for alpha in range(self.N_electron):  
            res_alpha = xp.zeros(dim, dtype=Y.dtype)

            for beta in range(self.N_electron):
                # 提取 Y 的第 beta 个分量
                y_beta = Y[beta].reshape(shape)
                
                # 执行算符作用 (这一步是在 GPU 上进行的矩阵乘法)
                # op_list[alpha][beta] 内部应当使用的是 cupy.tensordot 或类似操作
                op_result = hop_list[alpha][beta](y_beta)
                
                # 累加结果
                res_alpha += op_result.ravel()
            
            # 将计算好的 alpha 分量存入输出数组
            Y_out[alpha] = res_alpha

        return Y_out.ravel()

    def _apply_hop_list_gpu(self, Y, hop_list, dim, shape):
        # --- 0. 初始化多卡缓存 (挂载在 self 上，随实例永久存在) ---
        # 结构: self._gpu_cache[gpu_id][(alpha, beta)] = gpu_matrix
        if not hasattr(self, '_gpu_cache'):
            self._gpu_cache = {} 
        
        # 1. 预处理数据
        Y_split = Y.reshape(self.N_electron, dim)
        num_gpus = 4
        
        # 任务分配：例如 10 个电子分给 4 张卡 -> [ [0,1,2], [3,4,5], [6,7], [8,9] ]
        tasks = [[] for _ in range(num_gpus)]
        for i in range(self.N_electron):
            tasks[i % num_gpus].append(i)

        # 2. 定义 Worker (运行在各自的 GPU 线程中)
        def worker(gpu_id, alpha_indices):
            with xp.cuda.Device(gpu_id):
                # 确保当前 GPU 的缓存容器存在
                if gpu_id not in self._gpu_cache:
                    self._gpu_cache[gpu_id] = {}
                local_cache = self._gpu_cache[gpu_id]

                # A. 将状态向量 Y 广播到当前 GPU (相对较小，每次拷贝)
                # stream 用于掩盖传输延迟（可选优化，视 Y 大小而定）
                local_Y = xp.asarray(Y_split)
                
                local_results = []
                
                for alpha in alpha_indices:
                    # 预分配一个列表存储 beta 求和项
                    # 此时我们可以利用矩阵乘法的性质：Sum(Op_beta @ Y_beta)
                    # 如果显存允许，甚至可以合并成一次大矩阵乘法，但这里为了兼容性保持循环
                    
                    beta_accum = None # 用于累加结果
                    
                    for beta in range(self.N_electron):
                        # --- B. 获取/加载算符 (核心优化) ---
                        cache_key = (alpha, beta)
                        if cache_key in local_cache:
                            op = local_cache[cache_key]
                        else:
                            # 第一次运行：从主存/GPU0 拷贝到当前 GPU 并缓存
                            raw_op = hop_list[alpha][beta]
                            # 假设 raw_op 是 cupy/numpy 数组。如果是对象，需取其 .data
                            op = xp.asarray(raw_op) 
                            local_cache[cache_key] = op
                        
                        # --- C. 计算 ---
                        # 准备右侧向量
                        y_in = local_Y[beta].reshape(shape)
                        
                        # 执行运算 (op 是 50MB 矩阵，这一步是计算密集型)
                        # 假设 op 是矩阵，使用 @ 乘法；如果是函数则调用
                        res = op @ y_in if not callable(op) else op(y_in)
                        
                        # 展平结果
                        res_flat = res.ravel()
                        
                        # --- D. 累加 (避免最后 stack 的内存峰值) ---
                        if beta_accum is None:
                            beta_accum = res_flat
                        else:
                            beta_accum += res_flat
                    
                    local_results.append(beta_accum)
                
                return local_results

        # 3. 并行执行
        # 使用 ThreadPoolExecutor 因为主要耗时在 GPU 计算和数据传输，GIL 不是瓶颈
        results_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_ids = {
                executor.submit(worker, i, tasks[i]): tasks[i] 
                for i in range(num_gpus) if tasks[i]
            }
            
            for future in concurrent.futures.as_completed(future_to_ids):
                idxs = future_to_ids[future]
                try:
                    gpu_res = future.result()
                    for i, alpha in enumerate(idxs):
                        results_map[alpha] = gpu_res[i]
                except Exception as e:
                    # 方便调试异步异常
                    print(f"Error on GPU task: {e}")
                    raise e

        # 4. 结果拉回并拼接
        # 注意：这里会发生 Device -> Host -> Device (或 P2P) 的传输
        # 为了速度，统一拉回主设备 (Device 0)
        final_list = [xp.asarray(results_map[alpha], device=0) for alpha in range(self.N_electron)]
        
        return xp.concatenate(final_list)