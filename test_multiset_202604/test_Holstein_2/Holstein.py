# -*- coding: utf-8 -*-

from renormalizer.mps.backend import USE_GPU, xp
import logging
import json
from renormalizer import Model, Mps, Mpo, optimize_mps

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron
from renormalizer.multiset import MultisetChargeDiffusionDynamics

import numpy as np

from renormalizer.multiset import MultisetModel

from renormalizer.utils.log import package_logger as logger
import sys

import pandas as pd
from datetime import datetime

with open("../../example/fmo_sdf.json") as fin:
    # a 107*2 matrix
    sdf_values = json.load(fin)
sdf_values = np.array(sdf_values)

j_matrix_cm = np.array([[1, 800],
                        [800, 100],])

N_PHONONS = 3

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
    mol_arangement = np.array([1,2]) - 1
    model = HolsteinModel(list(np.array(mlist)[mol_arangement]), j_matrix_au[mol_arangement][:, mol_arangement], )
    
    max_bonddim = 4
    evolve_dt = 160
    n_snapshots = 10
    dynamics_job = MultisetChargeDiffusionDynamics(
        model=model,
        max_bonddim=max_bonddim,
        temperature=Quantity(0, "K"),
        stop_at_edge=False,
        dump_dir = "./",
        job_name = 'Holstein2'
    )

    from renormalizer.mps.backend import USE_GPU, xp  

    logger.info(f"GPU enabled: {USE_GPU}")  
    logger.info(f"Backend: {'CuPy' if USE_GPU else 'NumPy'}")
    logger.info("maximum bond dimension:%d, evolve time step:%d", max_bonddim, evolve_dt)
    logger.info("number of stored snapshots:%d", n_snapshots)

    logger.info("0th population: %s", dynamics_job.e_occupations_array[0])
    dynamics_job.evolve(evolve_dt=evolve_dt, nsteps=n_snapshots - 1)

    populations = np.array(dynamics_job.e_occupations_array)

    pd.DataFrame(populations).to_excel(datetime.now().strftime("%Y-%m-%d-%H%M_FMO") + str(max_bonddim) +'bd_' + str(evolve_dt) + "t.xlsx",
                                index=False, header=False)    
    

