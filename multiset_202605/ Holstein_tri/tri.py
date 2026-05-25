import numpy as np

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import (
    Quantity,
    EvolveConfig,
    CompressConfig,
    CompressCriteria,
    EvolveMethod,
)
from renormalizer.transport import ChargeDiffusionDynamics, InitElectron


# Parameters from the figure
N = 3
OMEGA_0 = 1.0
J = -1.0
G = 1.0
PHONON_DIM = 2
DUMP_DIR = "./"
JOB_NAME = "HolsteinTri3"


def build_triangle_j_matrix(j):
    j_matrix = np.zeros((N, N))
    edges = [(0, 1), (0, 2), (1, 2)]
    for i, j_site in edges:
        j_matrix[i, j_site] = j
        j_matrix[j_site, i] = j
    return j_matrix


# Step 1: build the phonon mode
displacement = G * np.sqrt(2.0 / OMEGA_0)
ph = Phonon.simple_phonon(
    Quantity(OMEGA_0),
    Quantity(displacement),
    PHONON_DIM,
)

# Step 2: build the molecule on each site
mol = Mol(Quantity(0), [ph])

# Step 3: assemble the triangular Holstein model
j_matrix = build_triangle_j_matrix(J)
model = HolsteinModel(
    [mol] * N,
    j_matrix,
    scheme=2,
)

evolve_dt = 0.1
evolve_time = 10
max_bonddim = 16

evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=evolve_dt)
evolve_config.if_startup_substeps = True
evolve_config.startup_substeps_n = 10

compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bonddim)

ct = ChargeDiffusionDynamics(
    model,
    evolve_config=evolve_config,
    compress_config=compress_config,
    init_electron=InitElectron.fc,
    dump_dir=DUMP_DIR,
    job_name=JOB_NAME,
)
ct.stop_at_edge = False
ct.evolve(evolve_dt=evolve_dt, evolve_time=evolve_time)
