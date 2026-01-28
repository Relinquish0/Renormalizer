# -*- coding: utf-8 -*-

import logging
import json

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron

from renormalizer.model.multiset_model import MultisetModel
import numpy as np

log.init_log(logging.INFO)
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)

import pandas as pd
from datetime import datetime

with open("/curie-home/zengjj/Renormalizer/example/fmo_sdf.json") as fin:
    # a 107*2 matrix
    sdf_values = json.load(fin)
sdf_values = np.array(sdf_values)

j_matrix_cm = np.array([[310, -98, 6, -6, 7, -12, -10, 38, ],
                        [-98, 230, 30, 7, 2, 12, 5, 8, ],
                        [6, 30, 0, -59, -2, -10, 5, 2, ],
                        [-6, 7, -59, 180, -65, -17, -65, -2, ],
                        [7, 2, -2, -65, 405, 89, -6, 5, ],
                        [-12, 11, -10, -17, 89, 320, 32, -10, ],
                        [-10, 5, 5, -64, -6, 32, 270, -11, ],
                        [38, 8, 2, -2, 5, -10, -11, 505, ], ])

N_PHONONS = 35

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
    mol_arangement = np.array([7, 5, 3, 1, 2, 4, 6]) - 1
    model = HolsteinModel(list(np.array(mlist)[mol_arangement]), j_matrix_au[mol_arangement][:, mol_arangement], )

    max_bonddim = 4
    evolve_dt = 160
    logger.info("maximum bond dimension:%d, evolve time step:%d", max_bonddim, evolve_dt)
    multisetmodel = MultisetModel(model, max_bonddim=max_bonddim)

    from renormalizer.mps.backend import USE_GPU, xp  
    logger.info(f"GPU enabled: {USE_GPU}")  
    logger.info(f"Backend: {'CuPy' if USE_GPU else 'NumPy'}")

    populations = []
    for i in range(10):
        population = multisetmodel.popultation()
        logger.info("%dth population: %s", i, population)
        populations.append(population)
        multisetmodel.evolve(evolve_dt=evolve_dt)

    # pd.DataFrame(populations).to_excel(datetime.now().strftime("%Y-%m-%d-%H%M_FMO") + str(max_bonddim) +'bd_' + str(evolve_dt) + "t.xlsx",
    #                         index=False, header=False)    


    '''
    evolve_dt = 160j
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=evolve_dt)
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=32)
    ct = ChargeDiffusionDynamics(model, evolve_config=evolve_config, compress_config=compress_config, init_electron=InitElectron.fc)

    ct.dump_dir = "./"
    ct.job_name = 'fmo'
    ct.stop_at_edge = False
    ct.evolve(evolve_dt=evolve_dt, evolve_time=40000)
    '''
    

