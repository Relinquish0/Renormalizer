# -*- coding: utf-8 -*-

import logging
import json
from renormalizer import Model, Mps, Mpo, optimize_mps

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron

import numpy as np

from renormalizer.model.multiset_model import MultisetModel

log.init_log(logging.INFO)
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)

with open("/curie-home/zengjj/Renormalizer/example/fmo_sdf.json") as fin:
    # a 107*2 matrix
    sdf_values = json.load(fin)
sdf_values = np.array(sdf_values)

j_matrix_cm = np.array([[10, 8],
                        [2, 1],])

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
    mol_arangement = np.array([1,2]) - 1
    model = HolsteinModel(list(np.array(mlist)[mol_arangement]), j_matrix_au[mol_arangement][:, mol_arangement], )

    multisetmodel = MultisetModel(model=model, max_bonddim=4)
    for i in range(10):
        logger.info("%dth Inner product: %s", i, multisetmodel.popultation())
        multisetmodel.evolve(160)
    '''

    evolve_dt = 160
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=evolve_dt)
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=32)


    ct = ChargeDiffusionDynamics(model, evolve_config=evolve_config, compress_config=compress_config, init_electron=InitElectron.fc)
    ct.dump_dir = "./"
    ct.job_name = 'Holstein'
    ct.stop_at_edge = False
    ct.evolve(evolve_dt=evolve_dt, evolve_time=1600)
    ''' 



