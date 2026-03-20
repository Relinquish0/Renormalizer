# -*- coding: utf-8 -*-

from renormalizer.mps.backend import USE_GPU, xp
import logging
import json
from renormalizer import Model, Mps, Mpo, optimize_mps

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity
from renormalizer.utils.constant import cm2au

import numpy as np

with open("/curie-home/zengjj/Renormalizer/example/fmo_sdf.json") as fin:
    # a 107*2 matrix
    sdf_values = json.load(fin)
sdf_values = np.array(sdf_values)

j_matrix_cm = np.array([[1, 800],
                        [800, 100],])

N_PHONONS = 2

TOTAL_HR = 0.42

if __name__ == "__main__":

    omegas_cm = np.linspace(2, 300, N_PHONONS)
    omegas_au = omegas_cm * cm2au
    hr_factors = np.interp(omegas_cm, sdf_values[:, 0], sdf_values[:, 1])

    hr_factors *= TOTAL_HR / hr_factors.sum()

    lams = hr_factors * omegas_au
    phonons = [Phonon.simplest_phonon(Quantity(o), Quantity(l), lam=True) for o,l in zip(omegas_au, lams)]


    j_matrix_au = j_matrix_cm * cm2au

    mlist = []
    for j in np.diag(j_matrix_au):
        m = Mol(Quantity(j), phonons)
        mlist.append(m)

    # starts from 1
    mol_arangement = np.array([1]) - 1
    model = HolsteinModel(list(np.array(mlist)[mol_arangement]), j_matrix_au[mol_arangement][:, mol_arangement], )

    mps = Mps.random(model, qntot=0, m_max=100)
    print(mps)
    for i in range(len(mps)):
        print(i)
        
        print(mps[i].shape)

