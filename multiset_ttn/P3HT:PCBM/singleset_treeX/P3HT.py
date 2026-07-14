import importlib.util
import logging
import os
import time
from pathlib import Path

import numpy as np

from renormalizer.model import BasisMultiElectronVac, BasisSHO, Model, Op
from renormalizer.model.basis import BasisDummy
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.tn import BasisTree, TTNO, TTNS, TreeNodeBasis, print_as_tree
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod, Quantity


logging.basicConfig(level=logging.INFO, format="%(asctime)s[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


MODEL_PATH = Path(__file__).resolve().parents[3] / "multiset_202605/P3HT:PCBM/multiset/P3HT.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location("p3ht_multiset_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_model_module()
N_OT = model_module.N_OT
N_MODE = model_module.N_MODE
NBAS_OFFSET = model_module.NBAS_OFFSET
EPS_LE = model_module.EPS_LE
J_LE = model_module.J_LE
T_CS = model_module.T_CS
LAMBDA_LE1_CS1 = model_module.LAMBDA_LE1_CS1
W_R = model_module.W_R
G_R_DIAG = model_module.G_R_DIAG
G_R_OFFDIAG = model_module.G_R_OFFDIAG
EPS_CS = model_module.EPS_CS
W_F = model_module.W_F
W_OT = model_module.W_OT
G_F = model_module.G_F
G_OT_CS = model_module.G_OT_CS
G_OT_LE = model_module.G_OT_LE
nbas_from_gw = model_module.nbas_from_gw


class P3HTPCBMTreeXModel(Model):
    def __init__(self):
        self.le_states = [f"LE{i}" for i in range(1, N_OT + 1)]
        self.cs_states = [f"CS{i}" for i in range(1, N_OT + 1)]
        self.state_labels = self.le_states + self.cs_states
        self.r_mode = "R"
        self.f_modes = [f"F{i}" for i in range(1, N_MODE + 1)]
        self.ot_modes = [[f"OT{i}_m{j}" for j in range(1, N_MODE + 1)] for i in range(1, N_OT + 1)]

        basis = []
        self.electron_bases = []
        for le_state, cs_state in zip(self.le_states, self.cs_states):
            e_basis = BasisMultiElectronVac([le_state, cs_state])
            basis.append(e_basis)
            self.electron_bases.append(e_basis)
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


def _dof_names(basis):
    names = []
    for dof in basis.dofs:
        if isinstance(dof, tuple) and len(dof) == 1:
            names.append(dof[0])
        elif isinstance(dof, tuple):
            names.extend(dof)
        else:
            names.append(dof)
    return names


def _leaf(basis):
    return TreeNodeBasis([basis])


def _dummy(label):
    return TreeNodeBasis([BasisDummy(label)])


def _balanced_subtree(label, children):
    if len(children) == 1:
        return children[0]
    node = _dummy(label)
    if len(children) <= 2:
        node.add_child(children)
        return node
    mid = (len(children) + 1) // 2
    node.add_child(_balanced_subtree((label, "L"), children[:mid]))
    node.add_child(_balanced_subtree((label, "R"), children[mid:]))
    return node


def build_treeX_basis_tree(basis_list):
    basis_by_name = {}
    for basis in basis_list:
        for name in _dof_names(basis):
            basis_by_name[name] = basis

    def require(name):
        try:
            return basis_by_name[name]
        except KeyError as exc:
            raise KeyError(f"Missing basis for TreeX dof {name!r}") from exc

    def ot_subtree(i):
        e_leaf = _leaf(require(f"LE{i}"))
        mode_leaves = [_leaf(require(f"OT{i}_m{j}")) for j in range(1, N_MODE + 1)]
        mode_tree = _balanced_subtree(("TreeX", f"OT{i}", "modes"), mode_leaves)
        node = _dummy(("TreeX", f"OT{i}"))
        node.add_child([e_leaf, mode_tree])
        return node

    r_leaf = _leaf(require("R"))
    f_subtree = _balanced_subtree(("TreeX", "F modes"), [_leaf(require(f"F{i}")) for i in range(1, N_MODE + 1)])

    ot1_with_r = _dummy(("TreeX", "OT1 and R"))
    ot1_with_r.add_child([ot_subtree(1), r_leaf])

    child1 = _dummy(("TreeX", "OT1-2 and R"))
    child1.add_child([ot1_with_r, ot_subtree(2)])

    child2 = _dummy(("TreeX", "OT3-6"))
    child2.add_child([
        _balanced_subtree(("TreeX", "OT3-4"), [ot_subtree(3), ot_subtree(4)]),
        _balanced_subtree(("TreeX", "OT5-6"), [ot_subtree(5), ot_subtree(6)]),
    ])

    ot7_13 = _balanced_subtree(("TreeX", "OT7-13"), [ot_subtree(i) for i in range(7, N_OT + 1)])
    child3 = _dummy(("TreeX", "OT7-13 and F"))
    child3.add_child([ot7_13, f_subtree])

    root = _dummy(("TreeX", "Root"))
    root.add_child([child1, child2, child3])
    return BasisTree(root)


def _format_float(value):
    return f"{float(value):.6g}"


def _tree_node_texts(basis_tree, entropy):
    return [f"{bnode.dofs} {_format_float(ent)}" for bnode, ent in zip(basis_tree.node_list, entropy)]


def calc_observables(ttns, occ_ops, ttno, istep):
    time_start = time.perf_counter()
    pop = np.asarray([ttns.expectation(op) for op in occ_ops], dtype=float)
    energy = ttns.expectation(ttno)
    bond_entropy = np.asarray(ttns.calc_bond_entropy(), dtype=float)
    logger.info("profile observables step=%d seconds=%.6f", istep, time.perf_counter() - time_start)
    return pop, energy, bond_entropy


def log_bond_entropy_tree(basis_tree, bond_entropy, istep, time_fs, final=False):
    tag = "final" if final else f"step={istep}"
    logger.info("bond entropy tree dump %s time_fs=%.6f", tag, time_fs)
    print_as_tree(_tree_node_texts(basis_tree, bond_entropy), basis_tree.adj_matrix, print_function=logger.info)


def log_tree_shapes(basis_tree, ttns):
    logger.info("TreeX singleset basis tree. Each OT subtree contains its local electronic basis and local OT modes.")
    basis_tree.print(print_function=logger.info)
    logger.info("TreeX singleset TTNS shape tree:")
    ttns.print_shape(full=True, print_function=logger.info)


def build_all_electronic_fc_ttns(basis_tree, model):
    vacuum = TTNS(basis_tree)
    creation_ops = [Op(r"a^\dagger", state) for state in model.state_labels]
    return TTNO(basis_tree, creation_ops) @ vacuum


def run(
    job_name="p3ht_ttns_treeX_bond_entropy",
    max_bond_dim=32,
    dt_fs=1.0,
    total_fs=200.0,
    initial_site=0,
    entropy_log_interval=20,
):
    model = P3HTPCBMTreeXModel()
    basis_tree = build_treeX_basis_tree(model.basis)
    ttno = TTNO(basis_tree, model.ham_terms)
    occ_ops = [TTNO(basis_tree, Op(r"a^\dagger a", state)) for state in model.state_labels]

    init_state = model.electron_bases[initial_site].dof_name_map[model.le_states[initial_site]]
    ttns = TTNS(basis_tree, condition={model.le_states[initial_site]: init_state})
    ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bond_dim)
    ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=Quantity(dt_fs, "fs").as_au())
    log_tree_shapes(basis_tree, ttns)

    if bool(int(os.environ.get("PRINT_ONLY", "0"))):
        return Path(__file__).with_name(f"{job_name}_{max_bond_dim}_print_only.npz")

    time_start = time.perf_counter()
    tn_state_ex = build_all_electronic_fc_ttns(basis_tree, model)
    ttns = expand_bond_dimension_general(ttns, ttno, ex_mps=tn_state_ex)
    logger.info("profile expand_bond_dimension_general seconds=%.6f", time.perf_counter() - time_start)
    logger.info("TreeX singleset TTNS shape tree after expansion:")
    ttns.print_shape(full=True, print_function=logger.info)

    step_au = Quantity(dt_fs, "fs").as_au()
    nsteps = int(round(total_fs / dt_fs))
    times_fs = [0.0]
    pop, energy, bond_entropy = calc_observables(ttns, occ_ops, ttno, 0)
    occupations = [pop]
    energies = [energy]
    S_bond_entropy_steps = [bond_entropy]
    S_maxbonds = [float(np.max(bond_entropy))]
    bond_dims = [ttns.bond_dims]
    logger.info(
        "step=%d time_fs=%.6f e_occupations=%s S_maxbond=%.12g",
        0,
        times_fs[-1],
        np.array2string(occupations[-1], precision=8),
        S_maxbonds[-1],
    )
    if entropy_log_interval > 0:
        log_bond_entropy_tree(basis_tree, bond_entropy, 0, times_fs[-1])

    for istep in range(1, nsteps + 1):
        ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=step_au)
        ttns = ttns.evolve(ttno, step_au)
        times_fs.append(times_fs[-1] + dt_fs)
        pop, energy, bond_entropy = calc_observables(ttns, occ_ops, ttno, istep)
        occupations.append(pop)
        energies.append(energy)
        S_bond_entropy_steps.append(bond_entropy)
        S_maxbonds.append(float(np.max(bond_entropy)))
        bond_dims.append(ttns.bond_dims)
        logger.info(
            "step=%d time_fs=%.6f e_occupations=%s S_maxbond=%.12g",
            istep,
            times_fs[-1],
            np.array2string(occupations[-1], precision=8),
            S_maxbonds[-1],
        )
        if entropy_log_interval > 0 and istep % entropy_log_interval == 0:
            log_bond_entropy_tree(basis_tree, bond_entropy, istep, times_fs[-1])

    log_bond_entropy_tree(basis_tree, bond_entropy, nsteps, times_fs[-1], final=True)

    occupations = np.asarray(occupations, dtype=float)
    S_bond_entropy_steps = np.asarray(S_bond_entropy_steps, dtype=float)
    n_le = len(model.le_states)
    out_path = Path(__file__).with_name(f"{job_name}_{max_bond_dim}.npz")
    np.savez(
        out_path,
        time_fs=np.asarray(times_fs),
        state_labels=np.asarray(model.state_labels, dtype=object),
        dof_strs=np.asarray([str(node.dofs) for node in basis_tree.node_list], dtype=object),
        adj_matrix=basis_tree.adj_matrix,
        e_occupations=occupations,
        le_occupations=occupations[:, :n_le],
        cs_occupations=occupations[:, n_le:],
        le1_occupation=occupations[:, 0],
        energies=np.asarray(energies),
        S_bond_entropy=S_bond_entropy_steps,
        S_maxbond=np.asarray(S_maxbonds, dtype=float),
        bond_dims=np.asarray(bond_dims, dtype=object),
    )
    return out_path


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


def main():
    print(
        run(
            job_name=os.environ.get("JOB_NAME", "p3ht_ttns_treeX_bond_entropy"),
            max_bond_dim=getenv("MAX_BONDDIM", 32, int),
            dt_fs=getenv("DT_FS", 1.0, float),
            total_fs=getenv("TOTAL_FS", 200.0, float),
            initial_site=getenv("INITIAL_SITE", 0, int),
            entropy_log_interval=getenv("ENTROPY_LOG_INTERVAL", 20, int),
        )
    )


if __name__ == "__main__":
    main()
