import numpy as onp

from renormalizer import BasisMultiElectronVac, BasisSHO, Model, Mps, Op
from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.multiset.multiset_mps import MultisetMps
from renormalizer.tn import BasisTree, MsTTNO, MsTTNS, TTNO, TTNS
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

    for node, bnode in zip(evolved.node_list, basis_tree.node_list):
        assert node.tensor.shape[0] == 2
        assert node.qn.shape == (node.tensor.shape[-1], bnode.qn_size)

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



def test_multiround_cross_hints_reach_second_neighbor():
    basis = [BasisMultiElectronVac(["e0", "e1", "e2"]), BasisSHO("v", 0.01, 2)]
    terms = [
        Op(r"a^\dagger a", "e0", 0.0),
        Op(r"a^\dagger a", "e1", 0.01),
        Op(r"a^\dagger a", "e2", 0.02),
        Op(r"a^\dagger a", ["e0", "e1"], 0.005),
        Op(r"a^\dagger a", ["e1", "e0"], 0.005),
        Op(r"a^\dagger a", ["e1", "e2"], 0.006),
        Op(r"a^\dagger a", ["e2", "e1"], 0.006),
        Op(r"b^\dagger b", "v", 0.01),
    ]
    model = Model(basis, terms)
    ms_model = MultisetModel(model, max_bonddim=4, auto_init=False)
    init = Mps.hartree_product_state(ms_model.init_model, {})
    zero = init.copy().scale(0, inplace=True)
    basis_tree = BasisTree.linear(ms_model.init_model.basis[::-1])
    ms_ttns = MsTTNS.from_ttns_list([
        _mps_to_linear_ttns(init, basis_tree),
        _mps_to_linear_ttns(zero, basis_tree),
        _mps_to_linear_ttns(zero, basis_tree),
    ])
    ms_ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=4)
    ms_ttno = MsTTNO.from_multiset_model(basis_tree, ms_model)

    hints = ms_ttns._build_multiround_cross_hints(ms_ttns.to_ttns_list(), ms_ttno, graph_rounds=2)

    assert hints[1]
    assert hints[2]
    second_neighbor_hint = MsTTNS._sum_ttns(hints[2], compress=False)
    assert second_neighbor_hint.norm > 0
