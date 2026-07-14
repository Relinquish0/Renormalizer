import importlib.util
import logging
import os
import time
from pathlib import Path

import numpy as np

from renormalizer.model.basis import BasisDummy
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.tn import BasisTree, MsTTNO, MsTTNS, TTNS, TreeNodeBasis, print_as_tree
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


def _dof_name(basis):
    dof = basis.dofs[0]
    if isinstance(dof, tuple) and len(dof) == 1:
        return dof[0]
    return dof


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
    """Build a multiset-compatible TreeX-like vibrational basis tree.

    The original TreeX groups each OT molecule's electronic states with its local
    OT vibrations.  In this multiset implementation electronic states are stored
    in the leading set dimension, so this tree keeps only vibrational basis nodes:
    OT local vibrational subtrees, the R mode, and the F modes.
    """

    basis_by_name = {_dof_name(basis): basis for basis in basis_list}

    def require(name):
        try:
            return basis_by_name[name]
        except KeyError as exc:
            raise KeyError(f"Missing basis for TreeX dof {name!r}") from exc

    def ot_subtree(i):
        leaves = [_leaf(require(f"OT{i}_m{j}")) for j in range(1, N_MODE + 1)]
        return _balanced_subtree(("TreeX", f"OT{i}", "modes"), leaves)

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


def build_initial_ms_ttns(ms_model, basis_tree, initial_site=0):
    init = TTNS(basis_tree, condition={})
    init.compress_config = CompressConfig(CompressCriteria.fixed)
    zero = init.copy().scale(0, inplace=True)
    components = []
    for istate in range(ms_model.N_electron):
        components.append(init.copy() if istate == initial_site else zero.copy())
    return MsTTNS.from_ttns_list(components)


def calc_observables(ms_ttns, istep):
    time_start = time.perf_counter()
    pop = ms_ttns.population()
    S_cond, S_raw = ms_ttns.calc_bond_entropy_cond_raw_allset(populations=pop)
    S_all = S_cond
    S_all_unnormed = S_raw
    if S_cond.size == 0 or S_cond.shape[1] == 0:
        S_maxbond_normed = 0.0
    else:
        S_maxbond_normed = float(np.max(S_cond))
    if S_raw.size == 0 or S_raw.shape[1] == 0:
        S_maxbond_unnormed = 0.0
    else:
        S_maxbond_unnormed = float(np.max(S_raw))
    logger.info("profile observables step=%d seconds=%.6f", istep, time.perf_counter() - time_start)
    return pop, S_all, S_maxbond_normed, S_all_unnormed, S_maxbond_unnormed, S_cond, S_raw


def _format_float(value):
    return f"{float(value):.6g}"


def _tree_node_texts(basis_tree, entropy):
    node_texts = []
    for bnode, ent in zip(basis_tree.node_list, entropy):
        node_texts.append(f"{bnode.dofs} {_format_float(ent)}")
    return node_texts


def log_allset_bond_entropy_trees(basis_tree, S_all_unnormed, istep, time_fs, final=False):
    tag = "final" if final else f"step={istep}"
    logger.info(
        "bond entropy tree dump S_unnormed %s time_fs=%.6f nset=%d",
        tag,
        time_fs,
        len(S_all_unnormed),
    )
    for iset, entropy in enumerate(S_all_unnormed):
        logger.info("bond entropy tree S_unnormed %s set=%d", tag, iset)
        print_as_tree(_tree_node_texts(basis_tree, entropy), basis_tree.adj_matrix, print_function=logger.info)


def log_tree_shapes(basis_tree, ms_ttns):
    logger.info("TreeX basis tree. Electronic states are represented by the multiset set dimension, not tree leaves.")
    basis_tree.print(print_function=logger.info)
    logger.info("TreeX TTNS shape tree for set 0:")
    ms_ttns.to_ttns_list()[0].print_shape(full=True, print_function=logger.info)
    logger.info("TreeX multiset tensor shapes include leading nset dimension:")
    for inode, node in enumerate(ms_ttns.node_list):
        logger.info("node=%d shape=%s", inode, node.tensor.shape)


def run(
    job_name="p3ht_ms_ttn_treeX_bond_entropy",
    max_bond_dim=32,
    dt_fs=1.0,
    total_fs=200.0,
    initial_site=0,
    entropy_log_interval=20,
):
    model = P3HTPCBMModel()
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=Quantity(dt_fs, "fs").as_au())
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=max_bond_dim)
    ms_model = MultisetModel(
        model,
        max_bonddim=max_bond_dim,
        evolve_config=evolve_config,
        compress_config=compress_config,
        auto_init=False,
    )

    basis_tree = build_treeX_basis_tree(ms_model.init_model.basis)
    ms_ttns = build_initial_ms_ttns(ms_model, basis_tree, initial_site=initial_site)
    ms_ttns.compress_config = compress_config
    ms_ttns.evolve_config = evolve_config
    log_tree_shapes(basis_tree, ms_ttns)

    if bool(int(os.environ.get("PRINT_ONLY", "0"))):
        return Path(__file__).with_name(f"{job_name}_{max_bond_dim}_print_only.npz")

    time_start = time.perf_counter()
    ms_ttno = MsTTNO.from_multiset_model(basis_tree, ms_model)
    logger.info("profile MsTTNO_build seconds=%.6f", time.perf_counter() - time_start)

    time_start = time.perf_counter()
    ms_ttns = ms_ttns.expand_bond_dimension_multiset(ms_ttno)
    logger.info("profile expand_bond_dimension_multiset seconds=%.6f", time.perf_counter() - time_start)
    logger.info("TreeX TTNS shape tree after expansion for set 0:")
    ms_ttns.to_ttns_list()[0].print_shape(full=True, print_function=logger.info)

    step_au = Quantity(dt_fs, "fs").as_au()
    nsteps = int(round(total_fs / dt_fs))
    times_fs = [0.0]
    pop, S_all, S_maxbond_normed, S_all_unnormed, S_maxbond_unnormed, S_cond, S_raw = calc_observables(ms_ttns, 0)
    occupations = [pop]
    S_all_steps = [S_all]
    S_all_unnormed_steps = [S_all_unnormed]
    S_cond_steps = [S_cond]
    S_raw_steps = [S_raw]
    S_maxbond_normeds = [S_maxbond_normed]
    S_maxbond_unnormeds = [S_maxbond_unnormed]
    bond_dims = [ms_ttns.bond_dims]
    logger.info(
        "step=%d time_fs=%.6f e_occupations=%s S_maxbond_normed=%.12g S_maxbond_unnormed=%.12g",
        0,
        times_fs[-1],
        np.array2string(occupations[-1], precision=8),
        S_maxbond_normeds[-1],
        S_maxbond_unnormeds[-1],
    )
    if entropy_log_interval > 0:
        log_allset_bond_entropy_trees(basis_tree, S_all_unnormed, 0, times_fs[-1])

    for istep in range(1, nsteps + 1):
        ms_ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, guess_dt=step_au)
        ms_ttns = ms_ttns.evolve(ms_ttno, step_au, normalize=True)
        times_fs.append(times_fs[-1] + dt_fs)
        pop, S_all, S_maxbond_normed, S_all_unnormed, S_maxbond_unnormed, S_cond, S_raw = calc_observables(ms_ttns, istep)
        occupations.append(pop)
        S_all_steps.append(S_all)
        S_all_unnormed_steps.append(S_all_unnormed)
        S_cond_steps.append(S_cond)
        S_raw_steps.append(S_raw)
        S_maxbond_normeds.append(S_maxbond_normed)
        S_maxbond_unnormeds.append(S_maxbond_unnormed)
        bond_dims.append(ms_ttns.bond_dims)
        logger.info(
            "step=%d time_fs=%.6f e_occupations=%s S_maxbond_normed=%.12g S_maxbond_unnormed=%.12g",
            istep,
            times_fs[-1],
            np.array2string(occupations[-1], precision=8),
            S_maxbond_normeds[-1],
            S_maxbond_unnormeds[-1],
        )
        if entropy_log_interval > 0 and istep % entropy_log_interval == 0:
            log_allset_bond_entropy_trees(basis_tree, S_all_unnormed, istep, times_fs[-1])

    log_allset_bond_entropy_trees(basis_tree, S_all_unnormed_steps[-1], nsteps, times_fs[-1], final=True)

    occupations = np.asarray(occupations, dtype=float)
    S_all_steps = np.asarray(S_all_steps, dtype=float)
    S_all_unnormed_steps = np.asarray(S_all_unnormed_steps, dtype=float)
    S_cond_steps = np.asarray(S_cond_steps, dtype=float)
    S_raw_steps = np.asarray(S_raw_steps, dtype=float)
    S_maxbond_normeds = np.asarray(S_maxbond_normeds, dtype=float)
    S_maxbond_unnormeds = np.asarray(S_maxbond_unnormeds, dtype=float)
    out_path = Path(__file__).with_name(f"{job_name}_{max_bond_dim}.npz")
    np.savez(
        out_path,
        time_fs=np.asarray(times_fs),
        state_labels=np.asarray(model.state_labels, dtype=object),
        dof_strs=np.asarray([str(node.dofs) for node in basis_tree.node_list], dtype=object),
        adj_matrix=basis_tree.adj_matrix,
        e_occupations=occupations,
        le_occupations=occupations[:, :N_OT],
        cs_occupations=occupations[:, N_OT:],
        le1_occupation=occupations[:, 0],
        S_all=S_all_steps,
        S_all_normed=S_all_steps,
        S_all_unnormed=S_all_unnormed_steps,
        S_cond=S_cond_steps,
        S_raw=S_raw_steps,
        S_maxbond=S_maxbond_normeds,
        S_maxbond_normed=S_maxbond_normeds,
        S_maxbond_unnormed=S_maxbond_unnormeds,
        bond_dims=np.asarray(bond_dims, dtype=object),
        active_pairs=np.asarray(ms_ttno.active_pairs_index, dtype=int),
    )
    return out_path


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


def main():
    print(
        run(
            job_name=os.environ.get("JOB_NAME", "p3ht_ms_ttn_treeX_bond_entropy"),
            max_bond_dim=getenv("MAX_BONDDIM", 32, int),
            dt_fs=getenv("DT_FS", 1.0, float),
            total_fs=getenv("TOTAL_FS", 200.0, float),
            initial_site=getenv("INITIAL_SITE", 0, int),
            entropy_log_interval=getenv("ENTROPY_LOG_INTERVAL", 20, int),
        )
    )


if __name__ == "__main__":
    main()
