# -*- coding: utf-8 -*-
# Author: Jinjun Zeng 
from mpi4py import MPI
import numpy as np
from renormalizer.mps.backend import xp

if xp != np:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # 获取可见的 GPU 数量
    num_gpus = xp.cuda.runtime.getDeviceCount()

    # 轮询分配：Rank 0 -> GPU 0, Rank 1 -> GPU 1, ..., Rank 4 -> GPU 0
    device_id = rank % num_gpus
    xp.cuda.Device(device_id).use()

    print(f"Rank {rank} is using GPU {device_id}")

import logging
from typing import List, Union, Dict, Callable
from collections import Counter
from enum import Enum


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


from concurrent.futures import ThreadPoolExecutor

from joblib import Parallel, delayed

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
    def _ms_evolve_tdvp_ps(self, ms_mps_:MultisetMps, ms_mpo:MultisetMpo, evolve_dt):
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
            mpsconj = ms_mps.msmps[alpha].conj()
            for beta in range(self.N_electron):
                Environ_list[alpha][beta] = Environ(ms_mps.msmps[beta], ms_mpo.msmpo[alpha][beta], mps_conj=mpsconj)

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
                            lambda Y: self._apply_block_operator_streams(Y, hop_list, dim, shape_imps),
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
                                lambda Y: self._apply_block_operator_streams(Y, hop_u_list, dimU, shapeU),
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
                                lambda Y: self._apply_block_operator_streams(Y, hop_svt_list, dimC, shapeC),
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

    def _apply_block_operator(self, Y, op_list, block_dim, block_shape):
        """
        GPU 优化版本: 
        1. 移除 Joblib 多线程 (避免 CUDA 上下文竞争)
        2. 使用 CuPy 的异步特性
        3. 显式循环累加，避免 Python sum() 产生过多的中间临时数组
        """
        # 确保输入 Y 已经被 reshape 为 (N_electron, block_dim)
        Y_reshaped = Y.reshape(self.N_electron, block_dim)
        
        # 预分配输出数组，避免多次 concatenate 带来的显存重新分配开销
        # Y 和 Y_out 都在 GPU 上
        Y_out = xp.zeros((self.N_electron, block_dim), dtype=Y.dtype)

        # 循环 alpha (行)
        for alpha in range(self.N_electron):
            # 在 GPU 上初始化累加器
            # 注意：如果 block_shape 和 block_dim 不一致，需要注意 reshape
            # 这里假设 op_list 返回的结果 ravel 后长度为 block_dim
            res_alpha = xp.zeros(block_dim, dtype=Y.dtype)
            
            # 循环 beta (列) 进行收缩
            for beta in range(self.N_electron):
                # 提取 Y 的第 beta 个分量
                y_beta = Y_reshaped[beta].reshape(block_shape)
                
                # 执行算符作用 (这一步是在 GPU 上进行的矩阵乘法)
                # op_list[alpha][beta] 内部应当使用的是 cupy.tensordot 或类似操作
                op_result = op_list[alpha][beta](y_beta)
                
                # 累加结果
                res_alpha += op_result.ravel()
            
            # 将计算好的 alpha 分量存入输出数组
            Y_out[alpha] = res_alpha

        # 展平返回，保持与 expm_krylov 接口一致
        return Y_out.ravel()



    def _apply_block_operator_joblib(self, Y, op_list, block_dim, block_shape):
        """
        多线程优化版本 (适用于 0.1s 级别的短任务)
        """
        # 1. 预处理
        Y_reshaped = Y.reshape(self.N_electron, block_dim)

        # 2. 定义任务
        def compute_row(alpha):
            # 这里的 sum 是串行的，但每次 op_list 调用内部的 opt_einsum 是耗时的
            # 在多线程下，7 个 alpha 会同时进行
            return sum(
                op_list[alpha][beta](
                    Y_reshaped[beta].reshape(block_shape)
                ).ravel()
                for beta in range(self.N_electron)
            )

        # 3. 并行执行
        # 关键修改：prefer='threads'
        # 没有任何启动开销，共享内存，速度极快
        results = Parallel(n_jobs=self.N_electron, prefer='threads')(
            delayed(compute_row)(alpha) for alpha in range(self.N_electron)
        )
        
        # 4. 拼接
        return xp.concatenate(results)

    def _apply_block_operator_mpi4py(self, Y, op_list, block_dim, block_shape):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # 确保输入 Y 在所有进程上是一致的，并且在当前设备的显存中
        # Y shape: (N_electron * block_dim)
        
        # 定义任务分配：简单的静态分配
        # 例如 7 个电子，7 个进程，rank 0 处理 alpha=0, rank 1 处理 alpha=1...
        # 如果进程数不能整除 N_electron，这里使用 Allgatherv 逻辑会更通用
        
        # 1. 计算当前 Rank 负责的 alpha 范围
        # 使用 numpy.array_split 均匀分配任务索引
        all_alphas = np.arange(self.N_electron)
        my_alphas = np.array_split(all_alphas, size)[rank]
        
        # 2. 本地计算 (只计算属于 my_alphas 的部分)
        # 预分配本地结果数组 (GPU)
        local_size = len(my_alphas) * block_dim
        Y_local_gpu = xp.zeros(local_size, dtype=Y.dtype)
        
        Y_reshaped = Y.reshape(self.N_electron, block_dim)

        # 循环当前进程负责的 alpha
        for i, alpha in enumerate(my_alphas):
            res_alpha = xp.zeros(block_dim, dtype=Y.dtype)
            
            # 收缩 beta (这一步依然需要所有的 Y，所以 Y 是全量的)
            for beta in range(self.N_electron):
                y_beta = Y_reshaped[beta].reshape(block_shape)
                # 累加 H_ab * Y_b
                res_alpha += op_list[alpha][beta](y_beta).ravel()
            
            # 填入本地结果片段
            Y_local_gpu[i * block_dim : (i + 1) * block_dim] = res_alpha

        # 3. MPI 通信 (Gather 汇总结果)
        # 为了保证兼容性，先将数据移回 CPU
        Y_local_cpu = asnumpy(Y_local_gpu)
        
        # 准备接收缓冲区
        if rank == 0:
            # 只有 rank 0 需要分配完整 buffer (如果是 Gather)
            # 但我们需要 Allgather，因为 Krylov 算法要求每个进程都有完整的下一步向量
            pass 

        # 计算每个进程的数据量，用于 Allgatherv
        # counts: 每个进程贡献的元素数量
        counts = [len(np.array_split(all_alphas, size)[r]) * block_dim for r in range(size)]
        # displs: 偏移量
        displs = [sum(counts[:r]) for r in range(size)]
        
        # 全局 CPU 缓冲区
        Y_global_cpu = np.empty(self.N_electron * block_dim, dtype=Y.dtype)
        
        # 执行通信：将分散计算的结果拼成完整向量，并分发给所有进程
        comm.Allgatherv([Y_local_cpu, MPI.DOUBLE_COMPLEX], 
                        [Y_global_cpu, counts, displs, MPI.DOUBLE_COMPLEX])
        
        # 4. 将汇总后的完整向量拷回 GPU
        return asxp(Y_global_cpu)

    def _apply_block_operator_streams(self, Y, op_list, block_dim, block_shape):
        """
        基于 CUDA Streams 的并行版本
        替代 Joblib，利用 GPU 自身的并发能力处理多行计算
        """
        # 1. 预处理
        Y_reshaped = Y.reshape(self.N_electron, block_dim)
        
        # 预分配输出显存 (在默认流上分配)
        Y_out = xp.zeros((self.N_electron, block_dim), dtype=Y.dtype)
        
        # 2. 创建 CUDA 流池
        # 为每一行 (alpha) 创建一个独立的流
        streams = [xp.cuda.Stream() for _ in range(self.N_electron)]
        
        # 3. 并发派发任务
        # 注意：这里的 Python for 循环依然是串行的，但是派发给 GPU 的指令是异步的
        # Python 会极快地跑完这个循环，把任务塞进 7 个不同的 GPU 队列中
        for alpha in range(self.N_electron):
            with streams[alpha]:
                # 在当前流 (stream[alpha]) 中执行累加
                # 注意：我们需要一个临时的累加变量，避免直接写入 Y_out 导致潜在的竞争（虽然写不同行是安全的，但显式分开更保险）
                
                # 获取当前 alpha 行需要的 beta 输入
                # 这一步没有计算，只是切片，非常快
                y_betas = [Y_reshaped[beta].reshape(block_shape) for beta in range(self.N_electron)]
                
                # 执行核心计算
                # 这里的 sum 会调用一系列 cupy 内核
                res_alpha = sum(
                    op_list[alpha][beta](y_betas[beta]).ravel()
                    for beta in range(self.N_electron)
                )
                
                # 将结果写入输出数组
                # 注意：Y_out 的内存是共享的，但不同的流写入不同的行 (alpha)，互不冲突
                Y_out[alpha] = res_alpha

        # 4. 同步
        # 等待所有流完成工作，确保 Y_out 数据已就绪
        # 这一步是必须的，否则返回时 GPU 可能还没算完
        for s in streams:
            s.synchronize()

        # 5. 返回
        return Y_out.ravel()