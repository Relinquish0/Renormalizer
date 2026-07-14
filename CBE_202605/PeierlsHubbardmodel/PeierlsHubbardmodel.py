import logging
import os
from functools import partial
from pathlib import Path

import numpy as np

from renormalizer import Model, Mpo, Mps
from renormalizer.model.basis import BasisSet, BasisSHO
from renormalizer.model.h_qc import generate_ladder_operator, simplify_op
from renormalizer.model.op import Op, OpSum
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod, log


log.init_log(getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()))
logger = logging.getLogger("renormalizer.peierls_hubbard")


class BasisPeierlsHubbardSite(BasisSet):
    multi_dof = True

    def __init__(self, up, down, ph, omega, nph_max):
        self.up = up
        self.down = down
        self.ph = ph
        self.omega = float(omega)
        self.nph_max = int(nph_max)
        self.nphonon = self.nph_max + 1
        self._phonon_basis = BasisSHO(ph, self.omega, self.nphonon)
        sigmaqn = []
        for nup in range(2):
            for ndown in range(2):
                for _ in range(self.nphonon):
                    sigmaqn.append([nup, ndown])
        super().__init__((up, down, ph), 4 * self.nphonon, sigmaqn)

    def local_index(self, nup, ndown, nph):
        return ((int(nup) * 2 + int(ndown)) * self.nphonon) + int(nph)

    @staticmethod
    def _spin_mat(symbol):
        if symbol == "I":
            return np.eye(2)
        if symbol == "Z":
            return np.diag([1.0, -1.0])
        if symbol in ["+", "sigma_+"]:
            return np.diag([1.0], k=1)
        if symbol in ["-", "sigma_-"]:
            return np.diag([1.0], k=-1)
        raise ValueError(f"op_symbol:{symbol} is not supported")

    def _elem_mat(self, symbol, dof):
        if symbol == "I":
            return np.eye(self.nbas)
        if dof == self.up:
            mat = np.kron(np.kron(self._spin_mat(symbol), np.eye(2)), np.eye(self.nphonon))
        elif dof == self.down:
            mat = np.kron(np.kron(np.eye(2), self._spin_mat(symbol)), np.eye(self.nphonon))
        elif dof == self.ph:
            mat = np.kron(np.eye(4), self._phonon_basis.op_mat(Op(symbol, self.ph)))
        else:
            raise ValueError(f"Unknown DoF name {dof} in {self}.")
        return mat

    def op_mat(self, op: Op):
        if not isinstance(op, Op):
            op = Op(op, self.dof[0])
        mat = np.eye(self.nbas)
        for symbol, dof in zip(op.split_symbol, op.dofs):
            mat = mat @ self._elem_mat(symbol, dof)
        return mat * op.factor

    def copy(self, new_dof):
        return self.__class__(new_dof[0], new_dof[1], new_dof[2], self.omega, self.nph_max)


class PeierlsHubbardModel:
    def __init__(self, nsites=100, U=10.0, omega=3.0, g=1.0, t=1.0, nph_max=8):
        self.nsites = int(nsites)
        self.U = float(U)
        self.omega = float(omega)
        self.g = float(g)
        self.t = float(t)
        self.nph_max = int(nph_max)
        self.norbs = 2 * self.nsites
        self.a_ops, self.adag_ops = generate_ladder_operator(self.norbs)
        self.process_op = partial(simplify_op, norbs=self.norbs, conserve_qn=True)
        self.model = self._build_model()

    @staticmethod
    def up(site):
        return 2 * site

    @staticmethod
    def down(site):
        return 2 * site + 1

    @staticmethod
    def ph(site):
        return ("ph", site)

    def cdag_c(self, p, q):
        return self.process_op(self.adag_ops[p] * self.a_ops[q])

    def _build_model(self):
        basis = []
        for site in range(self.nsites):
            basis.append(BasisPeierlsHubbardSite(self.up(site), self.down(site), self.ph(site), self.omega, self.nph_max))

        ham = []
        for site in range(self.nsites):
            nup = self.cdag_c(self.up(site), self.up(site))
            ndn = self.cdag_c(self.down(site), self.down(site))
            ham.append((nup * ndn) * self.U)
            ham.append(Op(r"b^\dagger b", self.ph(site), self.omega))

        for site in range(self.nsites - 1):
            xdiff = Op(r"b^\dagger+b", self.ph(site), self.g) + Op(r"b^\dagger+b", self.ph(site + 1), -self.g)
            for spin in (0, 1):
                left = 2 * site + spin
                right = 2 * (site + 1) + spin
                hop = self.cdag_c(left, right) + self.cdag_c(right, left)
                ham.extend((hop * -self.t).simplify())
                ham.extend((hop * xdiff).simplify())

        return Model(basis, OpSum(ham).simplify())

    def wavepacket_creation(self, spin, center, momentum, width):
        x = np.arange(self.nsites) - (self.nsites - 1) / 2
        weights = np.exp(-((x - center) / width) ** 2) * np.exp(-1j * momentum * x)
        weights = weights / np.linalg.norm(weights)
        terms = []
        for site, weight in enumerate(weights):
            orb = self.up(site) if spin == "up" else self.down(site)
            terms.append(self.process_op(self.adag_ops[orb]) * weight)
        return OpSum(terms).simplify()

    def initial_state(self, x0=20.0, k=np.pi / 2, width=4.0):
        mps = Mps.hartree_product_state(self.model, {})
        cup = Mpo(self.model, self.wavepacket_creation("up", x0, k, width))
        cdown = Mpo(self.model, self.wavepacket_creation("down", -x0, -k, width))
        mps = cup.apply(mps, canonicalise=True)
        mps = cdown.apply(mps, canonicalise=True)
        mps.normalize("mps_and_coeff")
        return mps

    def observables(self):
        sz_ops = []
        for site in range(self.nsites):
            nup = self.cdag_c(self.up(site), self.up(site))
            ndn = self.cdag_c(self.down(site), self.down(site))
            sz_ops.append(Mpo(self.model, 0.5 * (nup - ndn)))
        nph_ops = [Mpo(self.model, Op(r"b^\dagger b", self.ph(site))) for site in range(self.nsites)]
        return sz_ops, nph_ops


def save_npz(path, data):
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in data.items()})


def main():
    nsites = int(os.environ.get("NSITES", "100"))
    dt = float(os.environ.get("EVOLVE_DT", "0.05"))
    evolve_time = float(os.environ.get("EVOLVE_TIME", "40"))
    save_every = int(os.environ.get("SAVE_EVERY", "10"))
    max_bonddim = int(os.environ.get("MAX_BONDDIM", "500"))
    out_dir = Path(os.environ.get("OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    phm = PeierlsHubbardModel(
        nsites=nsites,
        U=float(os.environ.get("U", "10")),
        omega=float(os.environ.get("OMEGA_PH", "3")),
        g=float(os.environ.get("G", "1")),
        t=float(os.environ.get("HOPPING_T", "1")),
        nph_max=int(os.environ.get("NPH_MAX", "8")),
    )
    mpo = Mpo(phm.model)
    mps = phm.initial_state(
        x0=float(os.environ.get("X0", str(nsites / 4))),
        k=float(os.environ.get("K", str(np.pi / 2))),
        width=float(os.environ.get("WIDTH", "4")),
    )

    mps.evolve_config = EvolveConfig(
        EvolveMethod.tdvp_ps,
        guess_dt=dt,
        expansion_method="cbe",
        cbe_Dmax=max_bonddim,
        cbe_eps_pre=float(os.environ.get("CBE_EPS_PRE", "1e-4")),
        cbe_eps_final=float(os.environ.get("CBE_EPS_FINAL", "1e-6")),
        cbe_eps_trim=float(os.environ.get("CBE_EPS_TRIM", "1e-12")),
        cbe_max_expand=None if os.environ.get("CBE_MAX_EXPAND", "") == "" else int(os.environ["CBE_MAX_EXPAND"]),
        cbe_disable_after_warmup=False,
        cbe_lock_after_warmup=False,
        cbe_warmup_time=evolve_time,
        cbe_warmup_substeps=max(1, round(evolve_time / dt)),
    )
    mps.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bonddim)
    sz_ops, nph_ops = phm.observables()

    data = {
        "x": np.arange(nsites) - (nsites - 1) / 2,
        "physical_site_count": nsites,
        "mps_site_count": phm.model.nsite,
        "site_basis_dim": phm.model.pbond_list[0],
        "time": [],
        "sz": [],
        "nph": [],
        "energy": [],
        "bond_dims": [],
        "bond_entropy": [],
        "max_discarded_weight": [],
        "sum_discarded_weight": [],
    }
    nsteps = round(evolve_time / dt)
    job_name = os.environ.get("JOB_NAME", f"peierls_hubbard_g1_L{nsites}_M{max_bonddim}")
    out_path = out_dir / f"{job_name}.npz"

    for step in range(nsteps + 1):
        if step % save_every == 0:
            stats = getattr(mps.evolve_config, "cbe_last_stats", None) or {}
            data["time"].append(step * dt)
            data["sz"].append(mps.expectations(sz_ops))
            data["nph"].append(mps.expectations(nph_ops))
            data["energy"].append(mps.expectation(mpo))
            data["bond_dims"].append(np.asarray(mps.bond_dims, dtype=int))
            data["bond_entropy"].append(mps.calc_bond_entropy())
            data["max_discarded_weight"].append(stats.get("max_discarded_weight", 0.0) or 0.0)
            data["sum_discarded_weight"].append(stats.get("sum_discarded_weight", 0.0) or 0.0)
            save_npz(out_path, data)
            logger.info("saved step=%s time=%s maxD=%s energy=%s", step, step * dt, max(mps.bond_dims), data["energy"][-1])
        if step < nsteps:
            mps.evolve_config.current_time = step * dt
            mps.evolve_config.step_index = step
            mps = mps.evolve(mpo, dt)
            mps.evolve_config.current_time = (step + 1) * dt
            mps.evolve_config.step_index = step + 1


if __name__ == "__main__":
    main()
