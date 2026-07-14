import importlib.util
import logging
import os
import time
from pathlib import Path

import numpy as np

from renormalizer.model import Op
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
P3HTPCBMModel = model_module.P3HTPCBMModel
N_OT = model_module.N_OT
N_MODE = model_module.N_MODE


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


def _leaf(basis, bond_dim=None):
    return TreeNodeBasis([basis], bond_dim)


def _dummy(label, bond_dim=None):
    return TreeNodeBasis([BasisDummy(label)], bond_dim)


def _perfect_binary(nodes, dummy_label, bond_dims=None):
    assert len(nodes) == 4
    if bond_dims is None:
        bond_dims = [None, None]
    node1 = _dummy((dummy_label, 1), bond_dims[0])
    node2 = _dummy((dummy_label, 2), bond_dims[1])
    node3 = _dummy((dummy_label, 3), bond_dims[1])
    node1.add_child([node2, node3])
    node2.add_child(nodes[:2])
    node3.add_child(nodes[2:])
    return node1


def build_tree_basis_tree(basis_list, bond_dim):
    """Build the run.py tree5 topology with local OT near-leaf bond dimensions."""

    basis_by_name = {}
    for basis in basis_list:
        for name in _dof_names(basis):
            basis_by_name[name] = basis

    def require(name):
        try:
            return basis_by_name[name]
        except KeyError as exc:
            raise KeyError(f"Missing basis for Tree dof {name!r}") from exc

    bond_dim2 = max(bond_dim // 8, 16)
    logger.info("Tree target bond_dim=%d OT near-leaf bond_dim=%d", bond_dim, bond_dim2)

    node_f_r = _dummy("F/R", bond_dim)
    node_f_list = [_leaf(require(f"F{i}"), bond_dim) for i in range(1, N_MODE + 1)]
    tree_f1 = _perfect_binary(node_f_list[:4], "F1 to F4", [bond_dim, bond_dim])
    tree_f2 = _perfect_binary(node_f_list[4:], "F5 to F8", [bond_dim, bond_dim])
    f_root = _dummy("Vib F", bond_dim)
    f_root.add_child([tree_f1, tree_f2])
    node_f_r.add_child([f_root, _leaf(require("R"), bond_dim)])

    basis_ot_by_mol = [[require(f"OT{i}_m{j}") for j in range(1, N_MODE + 1)] for i in range(1, N_OT + 1)]

    ot_low_freq_subtree_list = []
    for i, basis_ot in enumerate(basis_ot_by_mol, start=1):
        low_freq_mol_root = _dummy(f"OT{i} low", bond_dim if i not in [11, 12, 13] else bond_dim2)
        low_freq_mol_root.add_child([
            _dummy(f"OT{i} 1-3 low", bond_dim2),
            _dummy(f"OT{i} 4-6 low", bond_dim2),
        ])
        low_freq_mol_root.children[1].add_child([_leaf(basis_ot[2 + j], bond_dim2) for j in range(3)])
        low_freq_mol_root.children[0].add_child([_leaf(basis_ot[5 + j], bond_dim2) for j in range(3)])
        ot_low_freq_subtree_list.append(low_freq_mol_root)

    ot_low_freq_1_2 = _dummy("OT 1-2 low", bond_dim)
    ot_low_freq_1_2.add_child(ot_low_freq_subtree_list[:2])
    ot_low_freq_3_4 = _dummy("OT 3-4 low", bond_dim)
    ot_low_freq_3_4.add_child(ot_low_freq_subtree_list[2:4])
    ot_low_freq_5_6 = _dummy("OT 5-6 low", bond_dim)
    ot_low_freq_5_6.add_child(ot_low_freq_subtree_list[4:6])
    ot_low_freq_7_8 = _dummy("OT 7-8 low", bond_dim)
    ot_low_freq_7_8.add_child(ot_low_freq_subtree_list[6:8])

    ot_low_freq_1_4 = _dummy("OT 1-4 low", bond_dim)
    ot_low_freq_1_4.add_child([ot_low_freq_1_2, ot_low_freq_3_4])
    ot_low_freq_5_8 = _dummy("OT 5-8 low", bond_dim)
    ot_low_freq_5_8.add_child([ot_low_freq_5_6, ot_low_freq_7_8])
    ot_low_freq_9_10 = _dummy("OT 9-10 low", bond_dim)
    ot_low_freq_9_10.add_child(ot_low_freq_subtree_list[8:10])
    ot_low_freq_10_13 = _dummy("OT 10-13 low", bond_dim2)
    ot_low_freq_10_13.add_child(ot_low_freq_subtree_list[10:13])

    ot_low_freq_1_8 = _dummy("OT 1-8 low", bond_dim)
    ot_low_freq_1_8.add_child([ot_low_freq_1_4, ot_low_freq_5_8])
    ot_low_freq_9_13 = _dummy("OT 9-13 low", bond_dim)
    ot_low_freq_9_13.add_child([ot_low_freq_9_10, ot_low_freq_10_13])
    ot_low_freq = _dummy("OT low", bond_dim)
    ot_low_freq.add_child([ot_low_freq_1_8, ot_low_freq_9_13])

    ot_high_freq_subtree_list = []
    for i in range(5):
        nodes = [_leaf(basis, bond_dim) for basis in basis_ot_by_mol[2 * i][:2] + basis_ot_by_mol[2 * i + 1][:2]]
        subtree_root = _perfect_binary(nodes, f"OT {2 * i + 1}-{2 * i + 2} high", [bond_dim, bond_dim])
        ot_high_freq_subtree_list.append(subtree_root)
    ot_high_freq_1_8 = _perfect_binary(ot_high_freq_subtree_list[:4], "OT 1-8 high", [bond_dim, bond_dim])

    ot_high_freq_10_13 = _dummy("OT 11-13 high", bond_dim2)
    ot_high_freq_10_13.add_child([
        _dummy("OT 11 high", bond_dim2),
        _dummy("OT 12 high", bond_dim2),
        _dummy("OT 13 high", bond_dim2),
    ])
    for i in range(3):
        ot_high_freq_10_13.children[i].add_child([
            _leaf(basis_ot_by_mol[10 + i][0], bond_dim2),
            _leaf(basis_ot_by_mol[10 + i][1], bond_dim2),
        ])

    ot_high_freq_9_13 = _dummy("OT 9-13 high", bond_dim)
    ot_high_freq_9_13.add_child([ot_high_freq_subtree_list[4], ot_high_freq_10_13])
    ot_high_freq = _dummy("OT high", bond_dim)
    ot_high_freq.add_child([ot_high_freq_1_8, ot_high_freq_9_13])

    node_ot = _dummy("OT", bond_dim)
    node_ot.add_child([ot_low_freq, ot_high_freq])

    root = _dummy("root", 1)
    root.add_child(node_f_r)
    root.add_child(_leaf(require("LE1"), bond_dim))
    root.add_child(node_ot)
    return BasisTree(root)


def tree_compress_config(basis_tree):
    target_dims = np.asarray(basis_tree.bond_dims, dtype=int)
    config = CompressConfig(CompressCriteria.fixed, max_bonddim=int(np.max(target_dims)))
    config.max_dims = target_dims
    config.bond_dim_max_value = int(np.max(target_dims))
    return config


def build_fc_ttns(basis_tree, model, initial_site):
    vacuum = TTNS(basis_tree)
    excite = TTNO(basis_tree, Op(r"a^\dagger", model.le_states[initial_site]))
    return excite @ vacuum


def build_all_electronic_fc_ttns(basis_tree, model):
    vacuum = TTNS(basis_tree)
    creation_ops = [Op(r"a^\dagger", state) for state in model.state_labels]
    return TTNO(basis_tree, creation_ops) @ vacuum


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
    logger.info("Tree singleset basis tree matched to run.py tree5 with local OT near-leaf bond dimensions.")
    basis_tree.print(print_function=logger.info)
    logger.info("Tree singleset TTNS shape tree:")
    ttns.print_shape(full=True, print_function=logger.info)


def run(
    job_name="p3ht_ttns_tree_bond_entropy",
    max_bond_dim=64,
    dt_fs=1.0,
    total_fs=200.0,
    initial_site=0,
    entropy_log_interval=20,
):
    model = P3HTPCBMModel()
    basis_tree = build_tree_basis_tree(model.basis, max_bond_dim)
    logger.info("Building Hamiltonian TTNO.")
    time_start = time.perf_counter()
    ttno = TTNO(basis_tree, model.ham_terms, algo="Hopcroft-Karp")
    logger.info("profile build Hamiltonian TTNO seconds=%.6f", time.perf_counter() - time_start)
    logger.info("Building occupation TTNOs.")
    time_start = time.perf_counter()
    occ_ops = [TTNO(basis_tree, Op(r"a^\dagger a", state)) for state in model.state_labels]
    logger.info("profile build occupation TTNOs seconds=%.6f", time.perf_counter() - time_start)

    logger.info("Building FC initial TTNS by applying a^dagger to the vacuum state.")
    time_start = time.perf_counter()
    ttns = build_fc_ttns(basis_tree, model, initial_site)
    logger.info("profile build FC TTNS seconds=%.6f", time.perf_counter() - time_start)
    ttns.compress_config = tree_compress_config(basis_tree)
    ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=Quantity(dt_fs, "fs").as_au())
    log_tree_shapes(basis_tree, ttns)

    if bool(int(os.environ.get("PRINT_ONLY", "0"))):
        return Path(__file__).with_name(f"{job_name}_{max_bond_dim}_print_only.npz")

    logger.info("Expanding TTNS bond dimensions with all-electronic FC directions.")
    time_start = time.perf_counter()
    tn_state_ex = build_all_electronic_fc_ttns(basis_tree, model)
    ttns = expand_bond_dimension_general(ttns, ttno, ex_mps=tn_state_ex)
    logger.info("profile expand_bond_dimension_general seconds=%.6f", time.perf_counter() - time_start)
    logger.info("Tree singleset TTNS shape tree after expansion:")
    ttns.print_shape(full=True, print_function=logger.info)

    step_au = Quantity(dt_fs, "fs").as_au()
    nsteps = int(round(total_fs / dt_fs))
    times_fs = [0.0]
    pop, energy, bond_entropy = calc_observables(ttns, occ_ops, ttno, 0)
    occupations = [pop]
    energies = [energy]
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
    n_le = len(model.le_states)
    out_path = Path(__file__).with_name(f"{job_name}_{max_bond_dim}.npz")
    np.savez(
        out_path,
        time_fs=np.asarray(times_fs),
        state_labels=np.asarray(model.state_labels, dtype=object),
        dof_strs=np.asarray([str(node.dofs) for node in basis_tree.node_list], dtype=object),
        adj_matrix=basis_tree.adj_matrix,
        target_bond_dims=np.asarray(basis_tree.bond_dims, dtype=int),
        e_occupations=occupations,
        le_occupations=occupations[:, :n_le],
        cs_occupations=occupations[:, n_le:],
        le1_occupation=occupations[:, 0],
        energies=np.asarray(energies),
        S_maxbond=np.asarray(S_maxbonds, dtype=float),
        bond_dims=np.asarray(bond_dims, dtype=object),
    )
    return out_path


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


def main():
    print(
        run(
            job_name=os.environ.get("JOB_NAME", "p3ht_ttns_tree_bond_entropy"),
            max_bond_dim=getenv("MAX_BONDDIM", 64, int),
            dt_fs=getenv("DT_FS", 1.0, float),
            total_fs=getenv("TOTAL_FS", 200.0, float),
            initial_site=getenv("INITIAL_SITE", 0, int),
            entropy_log_interval=getenv("ENTROPY_LOG_INTERVAL", 20, int),
        )
    )


if __name__ == "__main__":
    main()
