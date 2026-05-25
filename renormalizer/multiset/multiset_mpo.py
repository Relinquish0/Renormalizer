# -*- coding: utf-8 -*-

from renormalizer.mps.lib import _sum
from renormalizer.mps.mpo import Mpo


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
