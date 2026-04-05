# -*- coding: utf-8 -*-

import numpy as np

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron


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

max_bonddim_ = 4
evolve_dt = 0.2

evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=evolve_dt)
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bonddim_)
ct = ChargeDiffusionDynamics(model, evolve_config=evolve_config, compress_config=compress_config, init_electron=InitElectron.fc)
ct.dump_dir = "./"
ct.job_name = '2DHolstein25_chi{}'.format(max_bonddim_)
ct.stop_at_edge = False
ct.evolve(evolve_dt=evolve_dt, evolve_time=40)
