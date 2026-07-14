import numpy as np

from renormalizer import Mpo, Mps
from renormalizer.model.op import Op

from CBE_202605.PeierlsHubbardmodel.PeierlsHubbardmodel import (
    BasisPeierlsHubbardSite,
    PeierlsHubbardModel,
)


def test_grouped_basis_has_one_mps_site_per_lattice_site():
    nsites = 4
    nph_max = 2
    phm = PeierlsHubbardModel(nsites=nsites, nph_max=nph_max)

    assert phm.model.nsite == nsites
    assert phm.model.pbond_list == [4 * (nph_max + 1)] * nsites
    assert all(isinstance(basis, BasisPeierlsHubbardSite) for basis in phm.model.basis)


def test_grouped_basis_local_operators_and_vacuum_state():
    basis = BasisPeierlsHubbardSite(up=("up", 0), down=("down", 0), ph=("ph", 0), omega=3.0, nph_max=2)

    assert basis.nbas == 12
    np.testing.assert_array_equal(basis.sigmaqn[0], [0, 0])
    np.testing.assert_array_equal(basis.sigmaqn[basis.local_index(1, 0, 0)], [1, 0])
    np.testing.assert_array_equal(basis.sigmaqn[basis.local_index(0, 1, 0)], [0, 1])
    np.testing.assert_allclose(np.diag(basis.op_mat(Op(r"b^\dagger b", ("ph", 0))))[:3], [0, 1, 2])

    phm = PeierlsHubbardModel(nsites=3, nph_max=2)
    mps = Mps.hartree_product_state(phm.model, {})
    assert mps.site_num == 3
    assert mps.bond_dims == [1, 1, 1, 1]
    np.testing.assert_array_equal(mps.qntot, [0, 0])
    Mpo(phm.model)
