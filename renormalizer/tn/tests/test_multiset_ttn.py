import numpy as onp

from renormalizer import BasisMultiElectronVac, BasisSHO, Model, Mps, Op
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mps import MultisetMps
from renormalizer.tn import BasisTree, MsTTNO, MsTTNS, TTNO, TTNS
from renormalizer.tn.multiset_hop_expr import _scatter_pair_output
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod


def _mps_to_linear_ttns(mps, basis_tree):
    ttns = TTNS(basis_tree)
    for i in range(len(mps)):
        node = ttns.node_list[::-1][i]
        node.tensor = mps[i].array
        node.qn = mps.qn[i + 1]
        if i == 0:
            node.tensor = node.tensor[0, ...]
    ttns.check_shape()
    ttns.check_canonical()
    return ttns


def _coupled_two_state_model():
    basis = [BasisMultiElectronVac(["e0", "e1"]), BasisSHO("v", 0.01, 2)]
    terms = [
        Op(r"a^\dagger a", "e0", 0.0),
        Op(r"a^\dagger a", "e1", 0.02),
        Op(r"a^\dagger a", ["e0", "e1"], 0.005),
        Op(r"a^\dagger a", ["e1", "e0"], 0.005),
        Op(r"b^\dagger b", "v", 0.01),
    ]
    return Model(basis, terms)


def test_complex_pair_scatter_without_atomic_add():
    output = onp.zeros((3, 2), dtype=complex)
    pair_output = onp.asarray([[1 + 2j, 3j], [2 - 1j, 4], [5j, 6 - 2j]])
    scatter_groups = (
        (0, onp.asarray([0, 1], dtype=onp.int64)),
        (2, onp.asarray([2], dtype=onp.int64)),
    )

    _scatter_pair_output(output, pair_output, scatter_groups)

    onp.testing.assert_allclose(output[0], pair_output[0] + pair_output[1])
    onp.testing.assert_allclose(output[1], 0)
    onp.testing.assert_allclose(output[2], pair_output[2])


def test_linear_tree_matches_multiset_mps_one_step():
    model = _coupled_two_state_model()
    ms_model = MultisetModel(
        model,
        max_bonddim=4,
        evolve_config=EvolveConfig(EvolveMethod.tdvp_ps),
        auto_init=False,
    )
    init = Mps.hartree_product_state(ms_model.init_model, {})
    zero = init.copy().scale(0, inplace=True)
    ms_mps = MultisetMps(
        ms_model.MsModel,
        ms_model.N_electron,
        init_model=ms_model.init_model,
        msmps=[init, zero],
    )

    dt = 0.1
    ms_mps_evolved = ms_model.evolve_state(ms_mps, dt, normalize=False)

    basis_tree = BasisTree.linear(ms_model.init_model.basis[::-1])
    ms_ttns = MsTTNS.from_ttns_list([_mps_to_linear_ttns(mps, basis_tree) for mps in ms_mps.msmps])
    ms_ttno = MsTTNO.from_multiset_model(basis_tree, ms_model)
    ms_ttns_evolved = ms_ttns.evolve(ms_ttno, dt, normalize=False)

    assert ms_ttno.active_pairs_index == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for iset, mps in enumerate(ms_mps_evolved.msmps):
        onp.testing.assert_allclose(
            ms_ttns_evolved.todense(iset, order=basis_tree.basis_list).ravel(),
            mps.todense(),
            atol=1e-12,
        )


def test_diagonal_blocks_match_independent_ttns():
    basis_list = [BasisSHO("v0", 0.01, 2), BasisSHO("v1", 0.02, 2)]
    model0 = Model(basis_list, [Op(r"b^\dagger b", "v0", 0.01), Op(r"b^\dagger b", "v1", 0.02)])
    model1 = Model(basis_list, [Op(r"b^\dagger b", "v0", 0.03), Op(r"b^\dagger b", "v1", 0.04)])
    zero = Model(basis_list, [])
    basis_tree = BasisTree.linear(basis_list)

    ttns0 = TTNS.random(basis_tree, 0, 2, 1.0).canonicalise()
    ttns1 = TTNS.random(basis_tree, 0, 2, 1.0).canonicalise()
    ms_ttns = MsTTNS.from_ttns_list([ttns0, ttns1])
    ms_ttno = MsTTNO(basis_tree, [[model0, zero], [zero, model1]], nset=2)

    dt = 0.05
    ms_evolved = ms_ttns.evolve(ms_ttno, dt, normalize=False)

    ttns0.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    ttns1.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    ref0 = ttns0.evolve(TTNO(basis_tree, model0.ham_terms), dt, normalize=False)
    ref1 = ttns1.evolve(TTNO(basis_tree, model1.ham_terms), dt, normalize=False)

    onp.testing.assert_allclose(ms_evolved.todense(0).ravel(), ref0.todense().ravel(), atol=1e-12)
    onp.testing.assert_allclose(ms_evolved.todense(1).ravel(), ref1.todense().ravel(), atol=1e-12)


def test_shape_canonical_and_population_sanity():
    model = _coupled_two_state_model()
    ms_model = MultisetModel(model, max_bonddim=4, auto_init=False)
    init = Mps.hartree_product_state(ms_model.init_model, {})
    zero = init.copy().scale(0, inplace=True)
    basis_tree = BasisTree.linear(ms_model.init_model.basis[::-1])
    ms_ttns = MsTTNS.from_ttns_list([
        _mps_to_linear_ttns(init, basis_tree),
        _mps_to_linear_ttns(zero, basis_tree),
    ])
    ms_ttno = MsTTNO.from_multiset_model(basis_tree, ms_model)

    evolved = ms_ttns.evolve(ms_ttno, 0.1, normalize=True)
    evolved.check_shape()
    evolved.check_canonical()

    assert len(evolved.ttns_list) == 2
    assert len(evolved.layout_groups) == 1
    for component in evolved.ttns_list:
        for node, bnode in zip(component.node_list, basis_tree.node_list):
            # The electronic-set index is no longer stored in every TTNS node.
            assert node.tensor.ndim == len(node.children) + bnode.n_sets + 1
            assert node.qn.shape == (node.tensor.shape[-1], bnode.qn_size)

    # Components must be independently owned rather than views into one
    # leading-set-axis tensor.
    assert evolved.ttns_list[0] is not evolved.ttns_list[1]
    for node0, node1 in zip(evolved.ttns_list[0].node_list, evolved.ttns_list[1].node_list):
        assert not onp.shares_memory(node0.tensor, node1.tensor)

    pop = evolved.population()
    assert pop.shape == (2,)
    onp.testing.assert_allclose(pop.sum(), 1.0, atol=1e-12)



def test_population_handles_mctdh_dummy_nodes_without_dense():
    basis_list = [BasisSHO("v0", 0.01, 2), BasisSHO("v1", 0.02, 2), BasisSHO("v2", 0.03, 2)]
    basis_tree = BasisTree.binary_mctdh(basis_list, contract_primitive=True)
    ttns0 = TTNS(basis_tree, condition={})
    ttns1 = ttns0.copy().scale(0, inplace=True)
    ms_ttns = MsTTNS.from_ttns_list([ttns0, ttns1])

    pop = ms_ttns.population()
    assert pop.shape == (2,)
    onp.testing.assert_allclose(pop, [1.0, 0.0], atol=1e-12)

    dense = ms_ttns.todense(0)
    assert dense.shape == (2, 2, 2)


def test_print_shape_prints_every_selected_component():
    basis_list = [BasisSHO("v0", 0.01, 2), BasisSHO("v1", 0.02, 3)]
    basis_tree = BasisTree.linear(basis_list)
    component = TTNS(basis_tree, condition={})
    ms_ttns = MsTTNS.from_ttns_list([component, component])

    lines = []
    ms_ttns.print_shape(
        full=True,
        print_function=lines.append,
        component_indices=[0, 1],
        component_labels=["LE1", "CS1"],
    )

    assert lines.count("MsTTNS component LE1 (local index 0)") == 1
    assert lines.count("MsTTNS component CS1 (local index 1)") == 1
    # Both two-node TTNS trees are rendered, rather than only component zero.
    assert sum("(" in line and ")" in line for line in lines) >= 4

    with onp.testing.assert_raises(ValueError):
        ms_ttns.print_shape(component_indices=[0, 1], component_labels=["only-one"])

def test_expand_bond_dimension_multiset_uses_diag_and_cross_hints():
    basis_list = [BasisSHO("v0", 0.01, 2), BasisSHO("v1", 0.02, 2)]
    basis_tree = BasisTree.linear(basis_list)
    model0 = Model(basis_list, [Op(r"b^\dagger b", "v0", 0.01), Op(r"b^\dagger b", "v1", 0.02)])
    model1 = Model(basis_list, [Op(r"b^\dagger b", "v0", 0.02), Op(r"b^\dagger b", "v1", 0.01)])
    offdiag = Model(basis_list, [Op(r"b^\dagger+b", "v0", 0.001)])

    ttns0 = TTNS(basis_tree, condition={})
    ttns1 = ttns0.copy().scale(0, inplace=True)
    ms_ttns = MsTTNS.from_ttns_list([ttns0, ttns1])
    ms_ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=3)
    ms_ttno = MsTTNO(basis_tree, [[model0, offdiag], [offdiag, model1]], nset=2)

    expanded = ms_ttns.expand_bond_dimension_multiset(ms_ttno)

    expanded.check_shape()
    expanded.check_canonical()
    assert max(expanded.bond_dims) > 1
    assert expanded.population()[1] > 0


def test_expand_zero_component_before_active_component_uses_template():
    basis_list = [BasisSHO("v0", 0.01, 2), BasisSHO("v1", 0.02, 2)]
    basis_tree = BasisTree.linear(basis_list)
    diagonal = Model(basis_list, [Op(r"b^\dagger b", "v0", 0.01)])
    zero_model = Model(basis_list, [])
    active = TTNS(basis_tree, condition={})
    zero_state = active.copy().scale(0, inplace=True)
    state = MsTTNS.from_ttns_list([zero_state, active])
    state.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=3)
    operator = MsTTNO(
        basis_tree,
        [[diagonal, zero_model], [zero_model, diagonal]],
        nset=2,
    )

    expanded = state.expand_bond_dimension_multiset(operator)

    expanded.check_shape()
    expanded.check_canonical()
    assert max(expanded.component_bond_dims[0]) == 2
    assert expanded.population()[0] > 0


def test_pad_component_reaches_common_binary_tree_layout():
    basis_list = [BasisSHO(f"v{i}", 0.01, 4) for i in range(8)]
    basis_tree = BasisTree.binary_mctdh(basis_list, contract_primitive=True)
    exact_dims = TTNS(basis_tree, condition={}).bond_dims_exact
    target_dims = onp.asarray([min(int(dim), 16) for dim in exact_dims])
    rank_deficient_dims = onp.minimum(target_dims, 6)
    component = TTNS.random(basis_tree, 0, rank_deficient_dims)

    padded = MsTTNS._pad_component_bond_dimension(component, target_dims)

    onp.testing.assert_array_equal(padded.bond_dims, target_dims)
    assert sorted(set(component.bond_dims)) == [1, 4, 6]
    assert sorted(set(padded.bond_dims)) == [1, 4, 16]


def test_krylov_limits_preserve_nonuniform_tree_profile():
    # P3HT Tree at chi=128 has a root entry of zero, 15-dimensional terminal
    # expansion edges, and 127-dimensional central expansion edges.  A scalar
    # cap must never promote the terminal edges to the central-edge limit.
    target = onp.asarray([0, 15, 127, 15, 127])

    full = MsTTNS._krylov_bond_limits(target)
    propagated = MsTTNS._krylov_bond_limits(target, scalar_cap=64)

    onp.testing.assert_array_equal(full, [1, 15, 127, 15, 127])
    onp.testing.assert_array_equal(propagated, [1, 15, 64, 15, 64])



def test_mps_compatible_onehop_expansion_preserves_seeded_distant_row():
    basis = [
        BasisMultiElectronVac(["e0", "e1", "e2"]),
        BasisSHO("v", 0.01, 2),
        BasisSHO("w", 0.02, 2),
    ]
    terms = [
        Op(r"a^\dagger a", "e0", 0.0),
        Op(r"a^\dagger a", "e1", 0.01),
        Op(r"a^\dagger a", "e2", 0.02),
        Op(r"a^\dagger a", ["e0", "e1"], 0.005),
        Op(r"a^\dagger a", ["e1", "e0"], 0.005),
        Op(r"a^\dagger a", ["e1", "e2"], 0.006),
        Op(r"a^\dagger a", ["e2", "e1"], 0.006),
        Op(r"b^\dagger b", "v", 0.01),
        Op(r"b^\dagger b", "w", 0.02),
    ]
    model = Model(basis, terms)
    ms_model = MultisetModel(model, max_bonddim=4, auto_init=False)
    init = Mps.hartree_product_state(ms_model.init_model, {})
    seed = init.copy().scale(1e-10, inplace=True)
    basis_tree = BasisTree.linear(ms_model.init_model.basis[::-1])
    ms_ttns = MsTTNS.from_ttns_list([
        _mps_to_linear_ttns(init, basis_tree),
        _mps_to_linear_ttns(seed, basis_tree),
        _mps_to_linear_ttns(seed, basis_tree),
    ])
    ms_ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=4)
    ms_ttno = MsTTNO.from_multiset_model(basis_tree, ms_model)

    expanded = ms_ttns.expand_bond_dimension_multiset(ms_ttno)
    populations = expanded.population()

    expanded.check_shape()
    expanded.check_canonical()
    assert populations[1] > 0
    assert populations[2] > 0
    # The distant component has no direct connection to the active row.  Its
    # nonzero state therefore comes from the same epsilon vacuum seed used by
    # multiset-MPS, not from a hidden multi-round graph propagation.
    onp.testing.assert_allclose(populations[2], 1e-20, rtol=1e-6, atol=1e-26)
