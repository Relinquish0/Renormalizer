# -*- coding: utf-8 -*-

import numpy as np

from renormalizer.model.model import Model
from renormalizer.mps.lib import _sum
from renormalizer.mps.mpo import Mpo
from renormalizer.multiset.multiset_mps import MultisetMps
from renormalizer.utils import CompressConfig


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
                if len(self.MsModel[i][j].ham_terms) == 0:
                    self.msmpo[i][j] = []
                else:
                    self.msmpo[i][j] = Mpo(model=self.MsModel[i][j], terms=None)

    def total_mpo(self) -> "Mpo":
        mpos = []
        for alpha in range(self.N_electron):
            mpos.append(_sum(mps_list=self.msmpo[alpha], compress=False, temp_m_trunc=None))
        return _sum(mps_list=mpos, compress=False, temp_m_trunc=None)


class MultisetBlockMpo(MultisetMpo):
    """
    A block operator in multiset representation:

        O = sum_{alpha,beta} |alpha><beta| \\otimes O^{alpha,beta}

    The local operator in each block is represented as an MPO acting on the
    phonon-only basis used by ``MultisetModel``.
    """

    def __init__(
        self,
        basis_set,
        ms_terms,
        N_electron: int,
        compress_config: CompressConfig = None,
    ):
        self.MsTerms = ms_terms
        self.compress_config = compress_config
        msmodel = [
            [Model(basis=basis_set, ham_terms=ms_terms[a][b]) for b in range(N_electron)]
            for a in range(N_electron)
        ]
        super().__init__(msmodel, N_electron)
        self._active_pairs = [
            (alpha, beta)
            for alpha in range(self.N_electron)
            for beta in range(self.N_electron)
            if len(self.msmpo[alpha][beta]) != 0
        ]

    @property
    def has_terms(self):
        return len(self._active_pairs) != 0

    def apply(self, ms_state: MultisetMps) -> MultisetMps:
        new_state = MultisetMps.__new__(MultisetMps)
        new_state.MsModel = ms_state.MsModel
        new_state.N_electron = ms_state.N_electron
        new_state.temperature = ms_state.temperature
        new_state.init_model = ms_state.init_model
        new_state.method = ms_state.method
        new_state.msmps = []

        zero_template = ms_state.msmps[0].copy()
        zero_template.scale(0.0, inplace=True)
        zero_template.coeff = 1.0
        if self.compress_config is not None:
            zero_template.compress_config = self.compress_config

        for alpha in range(self.N_electron):
            contributions = []
            for beta in range(self.N_electron):
                mpo = self.msmpo[alpha][beta]
                if len(mpo) == 0:
                    continue
                source = ms_state.msmps[beta]
                if self.compress_config is not None:
                    source.compress_config = self.compress_config
                applied = mpo.contract(source)
                applied.normalize("mps_norm_to_coeff")
                if self.compress_config is not None:
                    applied.compress_config = self.compress_config
                contributions.append(applied)

            if len(contributions) == 0:
                new_state.msmps.append(zero_template.copy())
            elif len(contributions) == 1:
                new_state.msmps.append(contributions[0])
            else:
                combined = _sum(contributions, compress=True, temp_m_trunc=None)
                if self.compress_config is not None:
                    combined.compress_config = self.compress_config
                new_state.msmps.append(combined)
        return new_state

    def matrix_element(self, bra_state: MultisetMps, ket_state: MultisetMps) -> complex:
        total = 0j
        bra_conj = [state.conj() for state in bra_state.msmps]
        for alpha, beta in self._active_pairs:
            mpo = self.msmpo[alpha][beta]
            bra = bra_state.msmps[alpha]
            ket = ket_state.msmps[beta]
            total += (
                ket.expectation(mpo, self_conj=bra_conj[alpha])
                * np.conjugate(bra.coeff)
                * ket.coeff
            )
        return complex(total)
