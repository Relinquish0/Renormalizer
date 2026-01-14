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
    def __init__(self, model: Model): 
        self.model = model
        self.evolve_config: EvolveConfig = EvolveConfig(method=MsEvolveMethod.ms_evolve_tdvp_ps)
        self.compress_config: CompressConfig  = CompressConfig(CompressCriteria.fixed, max_bonddim=32)
        self.N_electron = self.model.ham_terms[-1].dofs[0] + 1 # This may consult bug!!! 
        self.basis_set = [item for item in self.model.basis if type(item).__name__ != 'BasisSimpleElectron'] # casting the electron terms

        self.MsModel = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
        self.MsOp = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
        self.SplitHamTerm()
                    
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
                self.MsOp[self.model.ham_terms[i].dofs[0][0]][self.model.ham_terms[i].dofs[0][0]].append(self._reset_all_MsOp(self.model.ham_terms[i]))
            else: # len(self.model.ham_terms[i].dofs) == 3
                self.MsOp[self.model.ham_terms[i].dofs[0]][self.model.ham_terms[i].dofs[1]].append(self._reset_all_MsOp(self.model.ham_terms[i]))

    def ConstructMsModel(self):
        for i in range(self.N_electron):
            for j in range(self.N_electron):
                self.MsModel[i][j] = Model(basis=self.basis_set, ham_terms=self.MsOp[i][j])

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
                if isinstance(new_op.dofs[i], tuple):
                    new_dofs.append((0,)+new_op.dofs[i][1:])
                else:
                    new_dofs.append(new_op.dofs[i])
                i += 1
            
        new_op.split_symbol = new_split_symbol
        new_op.qn_list = new_qn_list
        new_op.dofs = new_dofs
        return new_op          

    def evolve(self, evolve_dt, normalize=True):

        method = {
            MsEvolveMethod.ms_evolve_tdvp_ps: self._ms_evolve_tdvp_ps
        }[self.evolve_config.method]

        mps_alpha_next = []

        new_mps = method(ms_mps_=self.MsMps,ms_mpo=self.MsMpo, evolve_dt=evolve_dt)
        mps_alpha_next.append(new_mps)
        self.MsMps.msmps= mps_alpha_next

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

                    # Construt the sum of efficient Hamiltonian
                    hop_list = [[[] for _ in range(self.N_electron)] for _ in range(self.N_electron)]
                    for alpha in range(self.N_electron):
                        for beta in range(self.N_electron):
                            l_array = Environ_list[alpha][beta].read("L", imps - 1)
                            r_array = Environ_list[alpha][beta].read("R", imps + 1)

                            shape = list(ms_mps.msmps[alpha][imps].shape)
                            hop_list[alpha][beta] = hop_expr(l_array, r_array, [asxp(ms_mpo.msmpo[alpha][beta][imps].array)], shape)
                    print('*'*150)
                    print(hop_list[0][0])
                    print('*'*150)
                    # Construct the partial differential equation of Multiset TDVP
                    if self.evolve_config.ivp_solver == "krylov":
                        print(ms_mps.msmps[0][imps].ravel().array.shape)
                        mps_t, j = expm_krylov(
                            lambda y: sum(hop_beta(y.reshape(shape)).ravel() for hop_beta in hop_list),
                            -1j * evolve_dt / 2, mps_alpha[imps].ravel().array
                        )
                    else:
                        sol = solve_ivp(
                            lambda t, y: sum(hop_beta(y.reshape(shape)).ravel() for hop_beta in hop_list)/coef, # In this line "y:" is different from origin code
                            (0, evolve_dt/2),
                            mps_alpha[imps].ravel().array,
                            method=self.evolve_config.ivp_solver,
                            rtol=self.evolve_config.ivp_rtol,
                            atol=self.evolve_config.ivp_atol,
                        )
                        mps_t, j = sol.y, sol.nfev

                    local_steps.append(j)
                    mps_t = mps_t.reshape(shape)

                    qnbigl, qnbigr, _ = mps_alpha._get_big_qn([imps])
                    u, qnlset, v, qnrset = svd_qn.svd_qn(
                        asnumpy(mps_t),
                        qnbigl,
                        qnbigr,
                        mps_alpha.qntot,
                        QR=True,
                        system=system,
                        full_matrices=False,
                    )
                    vt = v.T

                    if not mps_alpha.to_right and imps != 0:
                        mps_alpha[imps] = vt.reshape([-1] + shape[1:])
                        mps_alpha.qn[imps] = qnrset
                        mps_alpha.qnidx = imps-1

                        shape_u = u.shape

                        # Construct hop_u list
                        hop_u_list = []
                        for beta in range(self.N_electron):
                            l_array = Environ_beta_list[beta].read("L", imps - 1)
                            r_array = Environ_beta_list[beta].read("R", imps + 1)
                            r_array = (Environ_beta_list[beta].GetLR(
                                "R", imps, ms_mps_beta.msmps[beta], ms_mpo.msmpo[alpha][beta], itensor=r_array, method="System", mps_conj=mps_alpha.conj()
                            ))
                            # reverse update u site
                            hop_u_list.append(hop_expr(l_array, r_array, [], shape_u))

                        if self.evolve_config.ivp_solver == "krylov":
                            mps_t, j = expm_krylov(
                                lambda y: sum(hop_u_beta(y.reshape(shape_u)).ravel() for hop_u_beta in hop_u_list),
                                1j * evolve_dt / 2, u.ravel()
                            )
                        else:
                            sol = solve_ivp(
                                lambda t, y: sum(hop_u_beta(y.reshape(shape_u)).ravel() for hop_u_beta in hop_u_list)/ -coef,
                                (0, evolve_dt/2),
                                u.ravel(),
                                method=self.evolve_config.ivp_solver,
                                rtol=self.evolve_config.ivp_rtol,
                                atol=self.evolve_config.ivp_atol,
                            )
                            mps_t, j = sol.y, sol.nfev

                        local_steps.append(j)
                        mps_t = mps_t.reshape(shape_u)

                        mps_alpha[imps - 1] = tensordot(mps_alpha[imps - 1].array, mps_t, axes=(-1, 0),)

                    elif mps_alpha.to_right and imps != len(mps_alpha) - 1:
                        mps_alpha[imps] = u.reshape(shape[:-1] + [-1])
                        mps_alpha.qn[imps + 1] = qnlset
                        mps_alpha.qnidx = imps+1
                        shape_svt = vt.shape

                        # Construct hop_svt list
                        hop_svt_list = []
                        for beta in range(self.N_electron):
                            l_array = Environ_beta_list[beta].read("L", imps - 1)
                            r_array = Environ_beta_list[beta].read("R", imps + 1)
                            l_array = (Environ_beta_list[beta].GetLR(
                                "L", imps, ms_mps_beta.msmps[beta], ms_mpo.msmpo[alpha][beta], itensor=l_array, method="System", mps_conj=mps_alpha.conj()
                            ))
                            # reverse update svt site
                            hop_svt_list.append(hop_expr(l_array, r_array, [], shape_svt))

                        if self.evolve_config.ivp_solver == "krylov":
                            mps_t, j = expm_krylov(
                                lambda y: sum(hop_svt_beta(y.reshape(shape_svt)).ravel() for hop_svt_beta in hop_svt_list),
                                1j * evolve_dt / 2, vt.ravel()
                            )
                        else:
                            sol = solve_ivp(
                                lambda t, y: sum(hop_svt_beta(y.reshape(shape_svt)).ravel() for hop_svt_beta in hop_svt_list) / -coef,
                                (0, evolve_dt/2),
                                vt.ravel(),
                                method=self.evolve_config.ivp_solver,
                                rtol=self.evolve_config.ivp_rtol,
                                atol=self.evolve_config.ivp_atol,
                            )
                            mps_t, j = sol.y, sol.nfev

                        local_steps.append(j)
                        mps_t = mps_t.reshape(shape_svt)

                        mps_alpha[imps + 1] = tensordot(mps_t, mps_alpha[imps + 1].array, axes=(1, 0),)

                    else:
                        mps_alpha[imps] = mps_t
            mps_alpha._switch_direction()

        steps_stat = stats.describe(local_steps)
        logger.debug(f"TDVP-PS Krylov space: {steps_stat}")
        mps_alpha.evolve_config.stat = steps_stat

        return mps_alpha

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

        for alpha in range(self.N_electron):
            for beta in range(self.N_electron):
                tentative_mpo = self.MsMpo.msmpo[alpha][beta]
                energy = Quantity(self.MsMps.msmps[beta].expectation(mpo = tentative_mpo, self_conj = self.MsMps.msmps[alpha]))
                self.MsMpo.msmpo[alpha][beta] = Mpo(model = self.MsModel[alpha][beta], terms = None, offset = Quantity(0))

        for alpha in range(self.N_electron):
            self.MsMps.msmps[alpha].canonicalise() # seem to make no sense  


    def popultation(self):
        # strategy: by computing <\Psi^\alpha|\Psi^\alpha>
        population = []
        for alpha in range(self.N_electron):
            population.append(self.MsMps.msmps[alpha].conj().dot(self.MsMps.msmps[alpha]).real)
        return population

    def Hamiltonian(self) -> "float":
        return self.MsMps.total_mps().expectation(self.MsMpo.total_mpo())
    
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