import sys
from typing import Union, List

from renormalizer import BasisMultiElectron, BasisSHO, Quantity, Op, Model, Mps, Mpo, BasisDummy, BasisSimpleElectron, \
    BasisMultiElectronVac
from renormalizer.tn import TTNS, TTNO, TreeNodeBasis, BasisTree
from renormalizer.mps.mps import expand_bond_dimension_general
from renormalizer.tn.node import TreeNode
from renormalizer.utils.configs import EvolveConfig, CompressConfig, EvolveMethod, CompressCriteria
from renormalizer.utils.log import package_logger as logger
from renormalizer.mps.backend import np

if __debug__:
    M_str = "008"
    dt = 4
    # 0 for MPS, 1 for MCTDH1, 2 for MCTDH2
    tree_type = 5
    final_time = 20
    # the number of phonon basis offset
    N_B_STR = "14"
else:
    M_str = sys.argv[1]
    dt = int(sys.argv[2])
    N_B_STR = sys.argv[3]
    tree_type = int(sys.argv[4])
    final_time = 200

M = int(M_str)
N_B = int(N_B_STR)

logger.info(f"M: {M}, dt: {dt}, nb: {N_B}, tree_type: {tree_type}")

# the number of OTs
N_OT = 13

# the number of effective modes
N_MODE = 8

# qn for a^\dagger a
if tree_type in [0, 1, 5, 6, 7]:
    # a single site for electronic DOFs
    QN = [0, 0]
else:
    QN = [1, -1]


def calculate_nbas(g, omega, n_b):
    return int(round((g / omega) ** 2 + 3 * g / omega + n_b))


def get_basis_and_ham_e_and_r():
    # electronic basis and ham terms

    basis_e = BasisMultiElectron([f"LE{j}" for j in range(N_OT)] + [f"CS{j}" for j in range(N_OT)],
                                 sigmaqn=[0] * 2 * N_OT)

    basis_e2 = [BasisMultiElectronVac([f"LE{j}", f"CS{j}"]) for j in range(N_OT)]

    epsilon_le = Quantity(100, "meV").as_au()
    epsilon_cs_values = [0.0,
                         33.6,
                         47.4,
                         56.0,
                         61.8,
                         65.7,
                         68.4,
                         70.0,
                         70.9,
                         71.2,
                         71.1,
                         70.5,
                         69.5]
    epsilon_cs = [Quantity(v, "meV").as_au() for v in epsilon_cs_values]
    J = Quantity(100, "meV").as_au()
    t = Quantity(-120, "meV").as_au()
    lambda_ = Quantity(-200, "meV").as_au()

    # Eq. S6
    ham_e = []
    for i in range(N_OT):
        le_dof = f"LE{i}"
        cs_dof = f"CS{i}"
        ham_e.append(Op(r"a^\dagger a", [le_dof, le_dof], qn=QN, factor=epsilon_le))
        ham_e.append(Op(r"a^\dagger a", [cs_dof, cs_dof], qn=QN, factor=epsilon_cs[i]))
        if i != N_OT - 1:
            le_dof2 = f"LE{i + 1}"
            cs_dof2 = f"CS{i + 1}"
            ham_e.append(Op(r"a^\dagger a", [le_dof, le_dof2], qn=QN, factor=J))
            ham_e.append(Op(r"a^\dagger a", [le_dof2, le_dof], qn=QN, factor=J))
            ham_e.append(Op(r"a^\dagger a", [cs_dof, cs_dof2], qn=QN, factor=t))
            ham_e.append(Op(r"a^\dagger a", [cs_dof2, cs_dof], qn=QN, factor=t))
    ham_e.append(Op(r"a^\dagger a", ["LE0", "CS0"], qn=QN, factor=lambda_))
    ham_e.append(Op(r"a^\dagger a", ["CS0", "LE0"], qn=QN, factor=lambda_))

    # The R mode. the data should be the same as model A

    omega_r = Quantity(10, "meV").as_au()
    g_r_cs_cs = Quantity(30 / np.sqrt(2), "meV").as_au()
    g_r_le_cs = Quantity(-10 / np.sqrt(2), "meV").as_au()
    nbas_r = calculate_nbas(np.max(np.abs([g_r_cs_cs, g_r_le_cs])), omega_r, N_B)
    basis_r = BasisSHO("R", omega=omega_r, nbas=nbas_r)

    # Eq. S4, the third term, and Eq. S9
    ham_r = [
        Op(r"b^\dagger b", dof="R", factor=omega_r),
        g_r_cs_cs * Op(r"a^\dagger a", ["CS0", "CS0"], qn=QN) * Op(r"b^\dagger+b", dof="R"),
        g_r_le_cs * Op(r"a^\dagger a", ["CS0", "LE0"], qn=QN) * Op(r"b^\dagger+b", dof="R"),
        g_r_le_cs * Op(r"a^\dagger a", ["LE0", "CS0"], qn=QN) * Op(r"b^\dagger+b", dof="R"),
    ]

    return basis_e, basis_e2, basis_r, ham_e, ham_r


def get_basis_and_ham_f():
    # the F modes
    omega_f_values = [200.025,
                      184.269,
                      177.853,
                      141.11,
                      93.952,
                      79.933,
                      55.892,
                      33.264]
    omega_f = [Quantity(v, "meV").as_au() for v in omega_f_values]
    g_f_values = [45.246,
                  65.701,
                  -40.280,
                  -17.511,
                  28.026,
                  -13.629,
                  -23.732,
                  9.86]
    g_f = [Quantity(v, "meV").as_au() for v in g_f_values]

    basis_f = []
    for i in range(N_MODE):
        nbas = calculate_nbas(g_f[i], omega_f[i], N_B)
        basis_f.append(BasisSHO(f"F{i}", omega_f[i], nbas))

    ham_f = []
    for i in range(N_MODE):
        # Eq. S4, the first term
        ham_f.append(Op(r"b^\dagger b", dof=f"F{i}", factor=omega_f[i]))
        # Eq. S7
        for j in range(N_OT):
            ham_f.append(
                g_f[i] * Op(r"a^\dagger a", [f"CS{j}", f"CS{j}"], qn=QN)
                * Op(r"b^\dagger+b", dof=f"F{i}")
            )
    return basis_f, ham_f


def get_basis_and_ham_ot():
    # ot vibrations
    omega_ot_values = [
        401.283,
        397.773,
        182.714,
        178.531,
        134.550,
        111.848,
        42.621,
        18.316]
    omega_ot = [Quantity(v, "meV").as_au() for v in omega_ot_values]
    g_ot_cs_values = [
        7.017,
        -0.077,
        -67.849,
        57.668,
        -40.145,
        11.68,
        -10.784,
        -12.309]
    g_ot_cs = [Quantity(v, "meV").as_au() for v in g_ot_cs_values]
    g_ot_le_values = [
        4.035,
        2.921,
        -129.712,
        46.885,
        -32.908,
        36.591,
        -20.211,
        -7.77,
    ]
    g_ot_le = [Quantity(v, "meV").as_au() for v in g_ot_le_values]

    basis_ot_high_freq = []
    basis_ot_low_freq = []
    basis_ot_by_mol = [[] for _ in range(N_OT)]
    threshold = Quantity(250, "meV").as_au()

    for n in range(N_OT):
        for l in range(N_MODE):
            nbas = calculate_nbas(np.max(np.abs([g_ot_cs[l], g_ot_le[l]])), omega_ot[l], N_B)
            b = BasisSHO((n, l), omega_ot[l], nbas)
            basis_ot_by_mol[n].append(b)
            if omega_ot[l] < threshold:
                basis_ot_low_freq.append(b)
            else:
                basis_ot_high_freq.append(b)

    ham_ot = []
    for n in range(N_OT):
        op_cs = Op(r"a^\dagger a", [f"CS{n}", f"CS{n}"], qn=QN)
        op_le = Op(r"a^\dagger a", [f"LE{n}", f"LE{n}"], qn=QN)
        for l in range(N_MODE):
            # Eq. S4, the second term
            ham_ot.append(Op(r"b^\dagger b", dof=(n, l), factor=omega_ot[l]))
            # Eq. S8
            op_vib = Op(r"b^\dagger+b", dof=(n, l))
            ham_ot.extend([
                g_ot_cs[l] * op_cs * op_vib,
                g_ot_le[l] * op_le * op_vib
            ])

    return basis_ot_low_freq, basis_ot_high_freq, basis_ot_by_mol, ham_ot


def main():
    basis_e, basis_e2, basis_r, ham_e, ham_r = get_basis_and_ham_e_and_r()
    basis_f, ham_f = get_basis_and_ham_f()
    basis_ot_low_freq, basis_ot_high_freq, basis_ot_by_mol, ham_ot = get_basis_and_ham_ot()
    hamiltonian = ham_e + ham_r + ham_f + ham_ot

    tn_state: Union[Mps, TTNS]
    h_tn_op: Union[Mpo, TTNO]

    if tree_type in [0, 6, 7, 8, 9]:
        # MPS
        basis_list = [basis_e] + [basis_r] + basis_f
        if tree_type in [0, 9]:
            for b in basis_ot_by_mol:
                basis_list = basis_list + b
        elif tree_type == 6:
            for b in basis_ot_by_mol[1:]:
                basis_list = basis_list + b
            basis_list = basis_list + basis_ot_by_mol[0]
        elif tree_type == 7:
            for b in basis_ot_by_mol[::-1]:
                basis_list = basis_list + b
        else:
            assert tree_type == 8
            mol_seq = np.array([12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13]) - 1
            basis_list = basis_f.copy()
            for i in mol_seq:
                if i == 0:
                    basis_list.append(basis_r)
                basis_list.append(basis_e2[i])
                basis_list.extend(basis_ot_by_mol[i])
            for b in basis_list:
                logger.info(b)

            logger.info([i for i, b in enumerate(basis_list) if b in basis_e2])
        model = Model(basis_list, hamiltonian)
        tn_state = Mps.hartree_product_state(model, condition={})
        if tree_type in [8]:
            tn_state = Mpo(model, Op(r"a^\dagger", ["LE0"])) @ tn_state

        h_tn_op = Mpo(model)
        logger.info(f"MPO bond dim: {h_tn_op.bond_dims}")
        n_le_tn_op = Mpo(model, Op(r"a^\dagger a", ["LE0", "LE0"]))
        n_r_tn_op = Mpo(model, Op(r"b^\dagger b", ["R", "R"]))
    else:
        if tree_type == 1:
            node_f_r = TreeNodeBasis(BasisDummy("F/R"))
            tree_f = BasisTree.binary_mctdh(basis_f, contract_primitive=True, dummy_label="F")
            node_f_r.add_children([tree_f.root, TreeNodeBasis(basis_r)])

            tree_ot_low = BasisTree.binary_mctdh(basis_ot_low_freq, contract_primitive=True, dummy_label="OT low")
            tree_ot_high = BasisTree.binary_mctdh(basis_ot_high_freq, contract_primitive=True, dummy_label="OT high")
            node_ot = TreeNodeBasis(BasisDummy("OT"))
            node_ot.add_children([tree_ot_low.root, tree_ot_high.root])

            root = TreeNodeBasis(BasisDummy("root"))
            root.add_child(node_f_r)
            root.add_child(TreeNodeBasis(basis_e))
            root.add_child(node_ot)
            basis_tree = BasisTree(root)
        elif tree_type == 2:
            node_f_r = TreeNodeBasis(BasisDummy("F/R"))
            tree_f = BasisTree.binary_mctdh(basis_f, contract_primitive=True, dummy_label="F")
            node_f_r.add_children([tree_f.root, TreeNodeBasis(basis_r)])

            ot_subtree_roots = []
            for i in range(N_OT):
                subtree = BasisTree.binary_mctdh(basis_ot_by_mol[i], contract_primitive=True, dummy_label=f"OT {i} vib")
                subtree_root = TreeNodeBasis(BasisDummy(f"OT {i}"))
                subtree_root.add_children([TreeNodeBasis(basis_e2[i]), subtree.root])
                ot_subtree_roots.append(subtree_root)

            # OT 1-4, expect strong entanglement
            node1 = perfect_binary(ot_subtree_roots[:4], dummy_label="OT 1-4")

            # OT 9-13, expect the smallest entanglement
            node2 = perfect_binary(ot_subtree_roots[-4:], dummy_label="OT 10-13")

            # OT 7-9
            node3 = TreeNodeBasis(BasisDummy("OT 7-9"))
            node3.add_child(ot_subtree_roots[6])
            node4 = TreeNodeBasis(BasisDummy("OT 8-9"))
            node4.add_children([ot_subtree_roots[7], ot_subtree_roots[8]])
            node3.add_child(node4)

            # OT 5-13
            node5 = perfect_binary([ot_subtree_roots[4], ot_subtree_roots[5], node3, node2],
                                   dummy_label="OT 5-13")

            root = TreeNodeBasis(BasisDummy("root"))
            root.add_children([node_f_r, node1, node5])
            basis_tree = BasisTree(root)
        elif tree_type == 3:

            ot_subtree_roots = []
            for i in range(N_OT):
                vib_subtree = BasisTree.binary_mctdh(basis_ot_by_mol[i], contract_primitive=True, dummy_label=f"OT {i} vib")
                subtree_root = TreeNodeBasis(BasisDummy(f"OT {i}"))
                if i == 0:
                    e_and_r = TreeNodeBasis(BasisDummy(f"OT {i} e and R"))
                    e_and_r.add_children([TreeNodeBasis(basis_e2[i]), TreeNodeBasis(basis_r)])
                    subtree_root.add_children([e_and_r, vib_subtree.root])
                else:
                    subtree_root.add_children([TreeNodeBasis(basis_e2[i]), vib_subtree.root])
                ot_subtree_roots.append(subtree_root)

            subtree1 = TreeNodeBasis(BasisDummy("OT 1-2"))
            subtree1.add_children([ot_subtree_roots[0], ot_subtree_roots[1]])

            tree_f = BasisTree.binary_mctdh(basis_f, contract_primitive=True, dummy_label="F")
            subtree2 = perfect_binary(ot_subtree_roots[2:5] + [tree_f.root], dummy_label="OT 3-5 and F")

            subtree3 = TreeNodeBasis(BasisDummy("OT 6-13"))
            subtree31 = perfect_binary(ot_subtree_roots[5:9], "OT 6-9")
            subtree32 = perfect_binary(ot_subtree_roots[9:], "OT 10-13")
            subtree3.add_children([subtree31, subtree32])

            root = TreeNodeBasis(BasisDummy("root"))
            root.add_children([subtree1, subtree2, subtree3])
            basis_tree = BasisTree(root)

        elif tree_type == 4:

            ot_subtree_roots = []
            for i in range(N_OT):
                vib_subtree = BasisTree.binary_mctdh(basis_ot_by_mol[i], contract_primitive=True, dummy_label=f"OT {i} vib")
                subtree_root = TreeNodeBasis(BasisDummy(f"OT {i}"))
                if i == 0:
                    e_and_r = TreeNodeBasis(BasisDummy(f"OT {i} e and R"))
                    e_and_r.add_children([TreeNodeBasis(basis_e2[i]), TreeNodeBasis(basis_r)])
                    subtree_root.add_children([e_and_r, vib_subtree.root])
                else:
                    subtree_root.add_children([TreeNodeBasis(basis_e2[i]), vib_subtree.root])
                ot_subtree_roots.append(subtree_root)

            subtree1 = TreeNodeBasis(BasisDummy("OT 1-2"))
            subtree1.add_children([ot_subtree_roots[0], ot_subtree_roots[1]])

            subtree2 = perfect_binary(ot_subtree_roots[2:6], dummy_label="OT 3-6")

            subtree3 = TreeNodeBasis(BasisDummy("OT 7-13 and F"))
            subtree_ot_7_to_13 = TreeNodeBasis(BasisDummy("OT 7-13"))
            subtree_ot789 = TreeNodeBasis(BasisDummy("OT 7-9"))
            subtree_ot89 = TreeNodeBasis(BasisDummy("OT 8-9"))
            subtree_ot89.add_children([ot_subtree_roots[7], ot_subtree_roots[8]])
            subtree_ot789.add_children([ot_subtree_roots[6], subtree_ot89])
            subtree_ot_7_to_13.add_child(subtree_ot789)
            subtree_ot_7_to_13.add_child(perfect_binary(ot_subtree_roots[9:], "OT 10-13"))
            subtree3.add_child(subtree_ot_7_to_13)
            tree_f = BasisTree.binary_mctdh(basis_f, contract_primitive=True, dummy_label="F")
            subtree3.add_child(tree_f.root)

            root = TreeNodeBasis(BasisDummy("root"))
            root.add_children([subtree1, subtree2, subtree3])
            basis_tree = BasisTree(root)
        elif tree_type == 5:
            basis_tree = build_tree5(basis_f, basis_r, basis_ot_by_mol, basis_e, M)
        else:
            assert False

        basis_tree.print(logger.info)
        for i, node in enumerate(basis_tree.node_list):
            logger.info(f"{i} {node}")

        dof_strs = []
        for n in basis_tree.node_list:
            assert len(n.basis_sets) == 1
            dof_strs.append(str(n.basis_sets[0].dofs))
        logger.info(dof_strs)

        tn_state = TTNS(basis_tree)
        if tree_type in [2, 3, 4]:
            tn_state = TTNO(basis_tree, Op(r"a^\dagger", ["LE0"])) @ tn_state
        h_tn_op = TTNO(basis_tree, hamiltonian, algo="Hopcroft-Karp")
        h_tn_op.print_shape(print_function=logger.info)

        n_le_tn_op = TTNO(basis_tree, Op(r"a^\dagger a", ["LE0", "LE0"]))
        n_r_tn_op = TTNO(basis_tree, Op(r"b^\dagger b", ["R", "R"]))

    tn_state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    if tree_type == 5:
        tn_state.compress_config = CompressConfig(CompressCriteria.fixed)
        tn_state.compress_config.max_dims = basis_tree.bond_dims
    else:
        tn_state.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=M)
    le_list = [tn_state.expectation(n_le_tn_op)]
    n_r_list = [tn_state.expectation(n_r_tn_op)]
    singular_values_list = [tn_state.calc_bond_singular_values()]
    entropy_list = [tn_state.calc_bond_entropy(singular_values_list[-1])]

    rdm_dict = tn_state.calc_1site_rdm()
    # first index: site idx
    # second index: time
    # them RDM indices. Use a dict to prevent rugged array error when saving
    rdms_dict = {}
    for i in range(len(rdm_dict)):
        rdms_dict[f"rdm{i}"] = [rdm_dict[i]]

    if M != 1:
        if tree_type in [0, 6, 7, 9]:
            tn_state = tn_state.expand_bond_dimension(h_tn_op, include_ex=False)
        if tree_type in [8]:
            tn_state = tn_state.expand_bond_dimension(h_tn_op, include_ex=True)
        elif tree_type in [1, 5]:
            tn_state = expand_bond_dimension_general(tn_state, h_tn_op)
            tn_state.print_shape(print_function=logger.info)
        elif tree_type in [2, 3, 4]:
            tn_state_ex = TTNS(basis_tree)
            ex_ops = ([Op(r"a^\dagger", f"LE{i}") for i in range(N_OT)]
                      + [Op(r"a^\dagger", f"CS{i}") for i in range(N_OT)])
            tn_state_ex = TTNO(basis_tree, ex_ops) @ tn_state_ex
            tn_state = expand_bond_dimension_general(tn_state, h_tn_op, ex_mps=tn_state_ex)
            tn_state.print_shape(print_function=logger.info)
        else:
            assert False

    time_step_value = dt
    steps = int((final_time // time_step_value) + 1)
    time_step = Quantity(time_step_value, "fs").as_au()
    logger.info(f"time_step_value: {time_step_value}")
    logger.info(f"time_step: {time_step}")

    for i in range(steps):
        logger.info(i)

        tn_state = tn_state.evolve(h_tn_op, time_step)

        n_le = tn_state.expectation(n_le_tn_op)
        logger.info(f"n_le: {n_le}")
        le_list.append(n_le)

        n_r = tn_state.expectation(n_r_tn_op)
        logger.info(f"n_r: {n_r}")
        n_r_list.append(n_r)

        singular_values = tn_state.calc_bond_singular_values()
        singular_values_list.append(singular_values)

        entropy = tn_state.calc_bond_entropy(singular_values)
        logger.info(entropy)
        entropy_list.append(entropy)

        rdm_dict = tn_state.calc_1site_rdm()
        for j in range(len(rdm_dict)):
            rdms_dict[f"rdm{j}"].append(rdm_dict[j])

    logger.info(le_list)
    t_list = np.arange(len(le_list)) * time_step_value

    if tree_type == 0:
        model_str = "mps"
    elif tree_type == 1:
        model_str = "tree1"
    elif tree_type == 2:
        model_str = "tree2"
    elif tree_type == 3:
        model_str = "tree3"
    elif tree_type == 4:
        model_str = "tree4"
    elif tree_type == 5:
        model_str = "tree5"
    elif tree_type == 6:
        model_str = "mps2"
    elif tree_type == 7:
        model_str = "mps3"
    elif tree_type == 8:
        model_str = "mps4"
    elif tree_type == 9:
        model_str = "mps5"
    else:
        assert False

    # Find the maximum number of columns across all arrays
    max_cols = max(arr.shape[1] for arr in singular_values_list)

    # Pad each array with zeros to match the maximum number of columns
    singular_values_array = np.array([
        np.pad(arr, ((0, 0), (0, max_cols - arr.shape[1])))
        for arr in singular_values_list
    ])

    data_to_save = {
        "n_le": le_list,
        "n_r": n_r_list,
        "singular_values": singular_values_array,
        "entropy_array": entropy_list,
        "t_list": t_list,
    }
    data_to_save.update(rdms_dict)

    data_to_save["dof_num"] = len(rdms_dict)


    if tree_type not in  [0, 5, 6, 7, 8, 9]:
        data_to_save["dof_strs"] = dof_strs
        data_to_save["adj_matrix"] = basis_tree.adj_matrix

    np.savez(f"./Results/{model_str}_M{M_str}_dt{dt}_nb{N_B}.npz", **data_to_save)


def build_tree5(basis_f, basis_r, basis_ot_by_mol, basis_e, bond_dim):
    bond_dim2 = max(bond_dim // 8, 16)
    node_f_r = TreeNodeBasis(BasisDummy("F/R"), bond_dim)
    # tree_f = BasisTree.binary_mctdh(basis_f, contract_primitive=True, dummy_label="F")
    node_f_list = [TreeNodeBasis(b, bond_dim) for b in basis_f]
    tree_f1 = perfect_binary(node_f_list[:4], "F1 to F4", [bond_dim, bond_dim])
    tree_f2 = perfect_binary(node_f_list[4:], "F5 to F8", [bond_dim, bond_dim])
    f_root = TreeNodeBasis(BasisDummy("Vib F"), bond_dim).add_children([tree_f1, tree_f2])
    node_f_r.add_children([f_root, TreeNodeBasis(basis_r, bond_dim)])

    ot_low_freq_subtree_list = []
    for i, basis_ot in enumerate(basis_ot_by_mol):
        low_freq_mol_root = TreeNodeBasis(BasisDummy(f"OT{i} low"), bond_dim if i not in [10, 11, 12] else bond_dim2)
        low_freq_mol_root.add_children([
            TreeNodeBasis(BasisDummy(f"OT{i} 1-3 low"), bond_dim2),
            TreeNodeBasis(BasisDummy(f"OT{i} 4-6 low"), bond_dim2),
        ])
        low_freq_mol_root.children[1].add_children(
            [TreeNodeBasis(basis_ot[2 + j], bond_dim2) for j in range(3)]  # from 3 to 7
        )
        low_freq_mol_root.children[0].add_children(
            [TreeNodeBasis(basis_ot[5 + j], bond_dim2) for j in range(3)]
        )
        ot_low_freq_subtree_list.append(low_freq_mol_root)

    ot_low_freq_1_2 = TreeNodeBasis(BasisDummy("OT 1-2 low"), bond_dim)
    ot_low_freq_1_2.add_children(ot_low_freq_subtree_list[:2])

    ot_low_freq_3_4 = TreeNodeBasis(BasisDummy("OT 3-4 low"), bond_dim)
    ot_low_freq_3_4.add_children(ot_low_freq_subtree_list[2:4])

    ot_low_freq_5_6 = TreeNodeBasis(BasisDummy("OT 5-6 low"), bond_dim)
    ot_low_freq_5_6.add_children(ot_low_freq_subtree_list[4:6])

    ot_low_freq_7_8 = TreeNodeBasis(BasisDummy("OT 7-8 low"), bond_dim)
    ot_low_freq_7_8.add_children(ot_low_freq_subtree_list[6:8])

    ot_low_freq_1_4 = TreeNodeBasis(BasisDummy("OT 1-4 low"), bond_dim)
    ot_low_freq_1_4.add_children([ot_low_freq_1_2, ot_low_freq_3_4])

    ot_low_freq_5_8 = TreeNodeBasis(BasisDummy("OT 5-8 low"), bond_dim)
    ot_low_freq_5_8.add_children([ot_low_freq_5_6, ot_low_freq_7_8])

    ot_low_freq_9_10 = TreeNodeBasis(BasisDummy("OT 9-10 low"), bond_dim)
    ot_low_freq_9_10.add_children(ot_low_freq_subtree_list[8:10])

    ot_low_freq_10_13 = TreeNodeBasis(BasisDummy("OT 10-13 low"), bond_dim2)
    ot_low_freq_10_13.add_children(ot_low_freq_subtree_list[10:13])

    ot_low_freq_1_8 = TreeNodeBasis(BasisDummy("OT 1-8 low"), bond_dim)
    ot_low_freq_1_8.add_children([ot_low_freq_1_4, ot_low_freq_5_8])

    ot_low_freq_9_13 = TreeNodeBasis(BasisDummy("OT 9-13 low"), bond_dim)
    ot_low_freq_9_13.add_children([ot_low_freq_9_10, ot_low_freq_10_13])

    ot_low_freq = TreeNodeBasis(BasisDummy("OT low"), bond_dim)
    ot_low_freq.add_children([ot_low_freq_1_8, ot_low_freq_9_13])

    ot_high_freq_subtree_list = []
    for i in range(5):
        nodes = [TreeNodeBasis(b, bond_dim) for b in basis_ot_by_mol[2 * i][:2] + basis_ot_by_mol[2 * i + 1][:2]]
        subtree_root = perfect_binary(nodes, f"OT {2 * i + 1}-{2 * i + 2} high", [bond_dim, bond_dim])
        ot_high_freq_subtree_list.append(subtree_root)
    ot_high_freq_1_8 = perfect_binary(ot_high_freq_subtree_list[:4], "OT 1-8 high", [bond_dim, bond_dim])

    ot_high_freq_10_13 = TreeNodeBasis(BasisDummy("OT 11-13 high"), bond_dim2)
    ot_high_freq_10_13.add_children([
        TreeNodeBasis(BasisDummy("OT 11 high"), bond_dim2),
        TreeNodeBasis(BasisDummy("OT 12 high"), bond_dim2),
        TreeNodeBasis(BasisDummy("OT 13 high"), bond_dim2),
    ])

    for i in range(3):
        ot_high_freq_10_13.children[i].add_children([
            TreeNodeBasis(basis_ot_by_mol[10 + i][0], bond_dim2),
            TreeNodeBasis(basis_ot_by_mol[10 + i][1], bond_dim2)
        ])

    ot_high_freq_9_13 = TreeNodeBasis(BasisDummy("OT 9-13 high"), bond_dim)
    ot_high_freq_9_13.add_children([ot_high_freq_subtree_list[4], ot_high_freq_10_13])

    ot_high_freq = TreeNodeBasis(BasisDummy("OT high"), bond_dim)
    ot_high_freq.add_children([ot_high_freq_1_8, ot_high_freq_9_13])

    node_ot = TreeNodeBasis(BasisDummy("OT"), bond_dim)
    node_ot.add_children([ot_low_freq, ot_high_freq])

    root = TreeNodeBasis(BasisDummy("root"), 1)
    root.add_child(node_f_r)
    root.add_child(TreeNodeBasis(basis_e, bond_dim))
    root.add_child(node_ot)
    return BasisTree(root)


def expand_bond_dimension(mps, hint_mpo=None, coef=1e-10, ex_mps=None):
    """
    expand bond dimension as required in compress_config. works for both mps and ttns
    """
    from renormalizer.mps.lib import compressed_sum
    # expander m target
    m_target = mps.compress_config.bond_dim_max_value - mps.bond_dims_mean
    # will be restored at exit
    mps.compress_config.bond_dim_max_value = m_target
    if mps.compress_config.criteria is not CompressCriteria.fixed:
        logger.warning("Setting compress criteria to fixed")
        mps.compress_config.criteria = CompressCriteria.fixed
    logger.debug(f"target for expander: {m_target}")
    if hint_mpo is None:
        expander = mps.__class__.random(mps.model, 1, m_target)
    else:
        # fill states related to `hint_mpo`
        logger.debug(
            f"average bond dimension of hint mpo: {hint_mpo.bond_dims_mean}"
        )
        if ex_mps is None:
            lastone = mps
        else:
            # in case of localized `self`
            lastone = mps + ex_mps
        expander_list = []
        cumulated_m = 0
        while True:
            lastone.compress_config.criteria = CompressCriteria.fixed
            expander_list.append(lastone)
            expander = compressed_sum(expander_list)
            if cumulated_m == expander.bond_dims_mean:
                # probably a small system, the required bond dimension can't be reached
                break
            cumulated_m = expander.bond_dims_mean
            logger.debug(
                f"cumulated bond dimension: {cumulated_m}. lastone bond dimension: {lastone.bond_dims}"
            )
            if m_target < cumulated_m:
                break
            lastone = (hint_mpo @ lastone).normalize("mps_and_coeff")
            lastone = lastone.canonicalise().compress(np.min(lastone.bond_dims[1:]))
    logger.debug(f"expander bond dimension: {expander.bond_dims}")
    mps.compress_config.bond_dim_max_value += mps.bond_dims_mean
    return (mps + expander.scale(coef * mps.norm, inplace=True)).canonicalise().canonicalise().normalize(
        "mps_norm_to_coeff")


def perfect_binary(nodes: List[TreeNode], dummy_label, bond_dims=None) -> TreeNode:
    assert len(nodes) == 4
    if bond_dims is None:
        bond_dims = [None, None]
    node1 = TreeNodeBasis(BasisDummy((dummy_label, 1)), bond_dims[0])
    node2 = TreeNodeBasis(BasisDummy((dummy_label, 2)), bond_dims[1])
    node3 = TreeNodeBasis(BasisDummy((dummy_label, 3)), bond_dims[1])
    node1.add_children([node2, node3])
    node2.add_children(nodes[:2])
    node3.add_children(nodes[2:])
    return node1


if __name__ == "__main__":
    main()

