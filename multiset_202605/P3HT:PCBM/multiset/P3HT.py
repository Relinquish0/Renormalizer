import logging
import os
from pathlib import Path

import numpy as np

from renormalizer.model import BasisMultiElectronVac, BasisSHO, Model, Op
from renormalizer.multiset.multiset_tdjob import MultisetChargeDiffusionDynamics
from renormalizer.utils import EvolveConfig, EvolveMethod, Quantity, log


log.init_log(logging.INFO)

N_OT = 13
N_MODE = 8
NBAS_OFFSET = 18
EPS_LE = 100.0
J_LE = 100.0
T_CS = -120.0
LAMBDA_LE1_CS1 = -200.0
W_R = 10.0
G_R_DIAG = 30.0 / np.sqrt(2.0)
G_R_OFFDIAG = -10.0 / np.sqrt(2.0)

EPS_CS = np.array([0.0, 33.6, 47.4, 56.0, 61.8, 65.7, 68.4, 70.0, 70.9, 71.2, 71.1, 70.5, 69.5])
W_F = np.array([200.025, 184.269, 177.853, 141.11, 93.952, 79.933, 55.892, 33.264])
W_OT = np.array([401.283, 397.773, 182.714, 178.531, 134.550, 111.848, 42.621, 18.316])
G_F = np.array([45.246, 65.701, -40.280, -17.511, 28.026, -13.629, -23.732, 9.86])
G_OT_CS = np.array([7.017, -0.077, -67.849, 57.668, -40.145, 11.68, -10.784, -12.309])
G_OT_LE = np.array([4.035, 2.921, -129.712, 46.885, -32.908, 36.591, -20.211, -7.77])


def nbas_from_gw(g_mev, w_mev, offset=NBAS_OFFSET):
    return int(np.ceil((abs(g_mev) / w_mev) ** 2 + 3.0 * abs(g_mev) / w_mev + offset))


class P3HTPCBMModel(Model):
    def __init__(self):
        self.le_states = [f"LE{i}" for i in range(1, N_OT + 1)]
        self.cs_states = [f"CS{i}" for i in range(1, N_OT + 1)]
        self.state_labels = self.le_states + self.cs_states
        self.r_mode = "R"
        self.f_modes = [f"F{i}" for i in range(1, N_MODE + 1)]
        self.ot_modes = [[f"OT{i}_m{j}" for j in range(1, N_MODE + 1)] for i in range(1, N_OT + 1)]

        basis = [BasisMultiElectronVac(self.state_labels)]
        basis.append(BasisSHO(self.r_mode, Quantity(W_R, "meV").as_au(), nbas_from_gw(G_R_OFFDIAG, W_R)))
        for i in range(N_MODE):
            basis.append(BasisSHO(self.f_modes[i], Quantity(W_F[i], "meV").as_au(), nbas_from_gw(G_F[i], W_F[i])))
        for i in range(N_OT):
            for j in range(N_MODE):
                gmax = max(abs(G_OT_CS[j]), abs(G_OT_LE[j]))
                basis.append(BasisSHO(self.ot_modes[i][j], Quantity(W_OT[j], "meV").as_au(), nbas_from_gw(gmax, W_OT[j])))

        ham_terms = []

        for i, state in enumerate(self.le_states):
            ham_terms.append(Op(r"a^\dagger a", state, Quantity(EPS_LE, "meV")))
            if i < N_OT - 1:
                ham_terms.append(Op(r"a^\dagger a", [self.le_states[i], self.le_states[i + 1]], Quantity(J_LE, "meV")))
                ham_terms.append(Op(r"a^\dagger a", [self.le_states[i + 1], self.le_states[i]], Quantity(J_LE, "meV")))

        for i, state in enumerate(self.cs_states):
            ham_terms.append(Op(r"a^\dagger a", state, Quantity(EPS_CS[i], "meV")))
            if i < N_OT - 1:
                ham_terms.append(Op(r"a^\dagger a", [self.cs_states[i], self.cs_states[i + 1]], Quantity(T_CS, "meV")))
                ham_terms.append(Op(r"a^\dagger a", [self.cs_states[i + 1], self.cs_states[i]], Quantity(T_CS, "meV")))

        ham_terms.append(Op(r"a^\dagger a", [self.le_states[0], self.cs_states[0]], Quantity(LAMBDA_LE1_CS1, "meV")))
        ham_terms.append(Op(r"a^\dagger a", [self.cs_states[0], self.le_states[0]], Quantity(LAMBDA_LE1_CS1, "meV")))

        ham_terms.append(Op(r"b^\dagger b", self.r_mode, Quantity(W_R, "meV")))
        for i in range(N_MODE):
            ham_terms.append(Op(r"b^\dagger b", self.f_modes[i], Quantity(W_F[i], "meV")))
        for i in range(N_OT):
            for j in range(N_MODE):
                ham_terms.append(Op(r"b^\dagger b", self.ot_modes[i][j], Quantity(W_OT[j], "meV")))

        ham_terms.append(
            Op(r"a^\dagger a", self.cs_states[0]) * Op(r"b^\dagger + b", self.r_mode, Quantity(G_R_DIAG, "meV"))
        )
        ham_terms.append(
            Op(r"a^\dagger a", [self.le_states[0], self.cs_states[0]])
            * Op(r"b^\dagger + b", self.r_mode, Quantity(G_R_OFFDIAG, "meV"))
        )
        ham_terms.append(
            Op(r"a^\dagger a", [self.cs_states[0], self.le_states[0]])
            * Op(r"b^\dagger + b", self.r_mode, Quantity(G_R_OFFDIAG, "meV"))
        )

        for i in range(N_MODE):
            f_op = Op(r"b^\dagger + b", self.f_modes[i], Quantity(G_F[i], "meV"))
            for state in self.cs_states:
                ham_terms.append(Op(r"a^\dagger a", state) * f_op)

        for i in range(N_OT):
            for j in range(N_MODE):
                ot_op_cs = Op(r"b^\dagger + b", self.ot_modes[i][j], Quantity(G_OT_CS[j], "meV"))
                ot_op_le = Op(r"b^\dagger + b", self.ot_modes[i][j], Quantity(G_OT_LE[j], "meV"))
                ham_terms.append(Op(r"a^\dagger a", self.cs_states[i]) * ot_op_cs)
                ham_terms.append(Op(r"a^\dagger a", self.le_states[i]) * ot_op_le)

        super().__init__(basis, ham_terms)
        self.mol_num = len(self.state_labels)


def run(job_name="p3ht_multiset", max_bond_dim=64, dt_fs=1.0, total_fs=200.0):
    model = P3HTPCBMModel()
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=Quantity(dt_fs, "fs").as_au())
    job = MultisetChargeDiffusionDynamics(
        model=model,
        max_bonddim=max_bond_dim,
        temperature=Quantity(0, "K"),
        evolve_config=evolve_config,
        initial_site=0,
        stop_at_edge=False,
        dump_dir=str(Path(__file__).parent),
        job_name=job_name + "_" + str(max_bond_dim),
        observables={"ph_occupations": False},
    )
    nsteps = int(round(total_fs / dt_fs))
    job.evolve(evolve_dt=Quantity(dt_fs, "fs").as_au(), nsteps=nsteps)

    e_occ = np.array(job.e_occupations_array)
    out_path = Path(__file__).with_name(f"{job_name}.npz")
    np.savez(
        out_path,
        time_fs=np.array(job.evolve_times) * Quantity(1, "a.u.").as_unit("fs").value,
        state_labels=np.array(model.state_labels, dtype=object),
        e_occupations=e_occ,
        le_occupations=e_occ[:, :N_OT],
        cs_occupations=e_occ[:, N_OT:],
        le1_occupation=e_occ[:, 0],
        ph_occupations=np.array(job.ph_occupations_array),
        energies=np.array(job.energies),
    )
    return out_path


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


def main():
    out_path = run(
        job_name=os.environ.get("JOB_NAME", "p3ht_multiset"),
        max_bond_dim=getenv("MAX_BONDDIM", 32, int),
        dt_fs=getenv("DT_FS", 1.0, float),
        total_fs=getenv("TOTAL_FS", 200.0, float),
    )
    print(out_path)


if __name__ == "__main__":
    main()
