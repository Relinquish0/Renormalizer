import os

from renormalizer.multiset.multiset_model import MultisetModel
from renormalizer.tn import BasisTree, MsTTNO
from renormalizer.utils import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod, Quantity

from P3HT import P3HTPCBMModel, build_initial_ms_ttns


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


def main():
    max_bond_dim = getenv("MAX_BONDDIM", 32, int)
    dt_fs = getenv("DT_FS", 1.0, float)
    initial_site = getenv("INITIAL_SITE", 0, int)
    expand = getenv("EXPAND", 0, int)

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

    basis_tree = BasisTree.binary_mctdh(ms_model.init_model.basis, contract_primitive=True)
    ms_ttns = build_initial_ms_ttns(ms_model, basis_tree, initial_site=initial_site)
    ms_ttns.compress_config = compress_config
    ms_ttns.evolve_config = evolve_config

    if expand:
        ms_ttno = MsTTNO.from_multiset_model(basis_tree, ms_model)
        ms_ttns = ms_ttns.expand_bond_dimension_multiset(ms_ttno)

    print(f"nset = {ms_ttns.nset}")
    print(f"tree nodes = {ms_ttns.size}")
    print(f"expanded = {bool(expand)}")
    print("Each set shares the same TTNS topology. Shape tree for set 0:")
    ms_ttns.to_ttns_list()[0].print_shape(full=True)
    print("Multiset node tensor shapes, including leading nset dimension:")
    for inode, node in enumerate(ms_ttns.node_list):
        print(f"{inode}: {node.tensor.shape}")


if __name__ == "__main__":
    main()
