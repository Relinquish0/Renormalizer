# -*- coding: utf-8 -*-

"""2D 5x5 Holstein model built from a custom electronic coupling matrix.

The lattice is flattened in row-major order:
(0, 0), (0, 1), ..., (0, 4), (1, 0), ..., (4, 4)

With this convention, the central site is index 12, which matches the
default initial site used by ChargeDiffusionDynamics for a 25-site model.
"""

import numpy as np

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod, log
from renormalizer.utils.constant import cm2au
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron

from renormalizer.multiset import MultisetChargeDiffusionDynamics

import numpy as np
import pandas as pd
from renormalizer.utils.log import package_logger as logger
from datetime import datetime


# Parameter setup
N = 5          # 5x5 square lattice, total number of sites = 25
G = 0.5        # electron-phonon coupling constant g
W0 = 1.0       # phonon frequency
J = 0.1        # nearest-neighbour hopping integral
NBOSE = 8      # phonon Fock space truncation
TMAX = 40      # total propagation time
DT = 0.2       # time step


def site_index(ix, iy, ncol):
    return ix * ncol + iy


def build_2d_j_matrix(nrow, ncol, j, periodic=True):
    j_matrix = np.zeros((nrow * ncol, nrow * ncol))

    for ix in range(nrow):
        for iy in range(ncol):
            current = site_index(ix, iy, ncol)

            if periodic or ix + 1 < nrow:
                neighbour = site_index((ix + 1) % nrow, iy, ncol)
                if neighbour != current:
                    j_matrix[current, neighbour] = j
                    j_matrix[neighbour, current] = j

            if periodic or iy + 1 < ncol:
                neighbour = site_index(ix, (iy + 1) % ncol, ncol)
                if neighbour != current:
                    j_matrix[current, neighbour] = j
                    j_matrix[neighbour, current] = j

    return j_matrix


# Step 1: build the phonon mode
lam = G ** 2 * W0
ph = Phonon.simplest_phonon(
    Quantity(W0),
    Quantity(lam),
    lam=True,
    max_pdim=NBOSE,
)

# Step 2: build the molecule on each lattice site
mol = Mol(Quantity(0), [ph])

# Step 3: assemble the 2D HolsteinModel with the same periodic 2D lattice used in pyttn
j_matrix = build_2d_j_matrix(N, N, J, periodic=True)
model = HolsteinModel(
    [mol] * (N * N),
    j_matrix,
    scheme=2,
)

max_bonddim = 4
evolve_dt = DT
num_steps = int(round(TMAX / evolve_dt))
if not np.isclose(num_steps * evolve_dt, TMAX):
    raise ValueError('TMAX must be an integer multiple of DT.')
n_snapshots = num_steps + 1
dynamics_job = MultisetChargeDiffusionDynamics(
    model=model,
    max_bonddim=max_bonddim,
    stop_at_edge=False,
)

from renormalizer.mps.backend import USE_GPU, xp  

logger.info(f"GPU enabled: {USE_GPU}")  
logger.info(f"Backend: {'CuPy' if USE_GPU else 'NumPy'}")
logger.info("maximum bond dimension:%d, evolve time step:%d", max_bonddim, evolve_dt)
logger.info("number of stored snapshots:%d", n_snapshots)

logger.info("0th population: %s", dynamics_job.e_occupations_array[0])
dynamics_job.evolve(evolve_dt=evolve_dt, nsteps=num_steps)
populations = np.array(dynamics_job.e_occupations_array)
pd.DataFrame(populations).to_excel(datetime.now().strftime("%Y-%m-%d-%H%M_FMO") + str(max_bonddim) +'bd_' + str(evolve_dt) + "t.xlsx",
                            index=False, header=False)    
