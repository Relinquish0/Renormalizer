# -*- coding: utf-8 -*-

import numpy as np

from renormalizer.model import Model, Op
from renormalizer.model.basis import BasisSHO
from renormalizer.mps import Mps, Mpo
from renormalizer.mps.lib import Environ
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.matrix import asnumpy, asxp, tensordot
from renormalizer.mps.svd_qn import add_outer
from renormalizer.utils import EvolveConfig, EvolveMethod
from renormalizer.mps.cbe import (
    _array,
    _candidate_bond_qns,
    _hpsi_qn_block,
    _qn_mask,
    cbe_expand_left_to_right,
    cbe_expand_right_to_left,
    cbe_shrewd_selection_left_to_right,
    cbe_shrewd_selection_right_to_left,
)


def _two_site_model():
    basis = [BasisSHO("v0", 0.1, 3), BasisSHO("v1", 0.2, 3)]
    terms = [Op(r"b^\dagger+b", "v0") * Op(r"b^\dagger+b", "v1")]
    return Model(basis, terms)


def _mps_mpo_env():
    model = _two_site_model()
    mps = Mps.hartree_product_state(model, {"v0": 0, "v1": 0})
    mpo = Mpo(model)
    environ = Environ(mps, mpo)
    return mps, mpo, environ


def test_factorized_qn_block_matches_full_two_site_hpsi():
    mps, mpo, environ = _mps_mpo_env()
    bond_idx = 0
    l_array = _array(environ.read("L", bond_idx - 1))
    r_array = _array(environ.read("R", bond_idx + 2))
    A_l = _array(mps[bond_idx])
    B_r = _array(mps[bond_idx + 1])
    W_l = _array(mpo[bond_idx])
    W_r = _array(mpo[bond_idx + 1])

    theta = asnumpy(tensordot(A_l, B_r, axes=1))
    hop = hop_expr(l_array, r_array, [asxp(W_l), asxp(W_r)], theta.shape)
    hmat = asnumpy(hop(asxp(theta))).reshape(
        A_l.shape[0] * A_l.shape[1], B_r.shape[1] * B_r.shape[2]
    )
    q_left = add_outer(
        np.array(mps.qn[bond_idx]), np.array(mps._get_sigmaqn(bond_idx))
    ).reshape(-1, mps.model.qn_size)
    q_right = add_outer(
        np.array(mps._get_sigmaqn(bond_idx + 1)), np.array(mps.qn[bond_idx + 2])
    ).reshape(-1, mps.model.qn_size)

    for qn in _candidate_bond_qns(mps, q_left, q_right):
        lidx = np.where(_qn_mask(q_left, qn))[0]
        ridx = np.where(_qn_mask(q_right, mps.qntot - qn))[0]
        block = _hpsi_qn_block(A_l, B_r, W_l, W_r, l_array, r_array, lidx, ridx)
        assert np.allclose(block, hmat[np.ix_(lidx, ridx)])



def test_cbe_shrewd_selection_right_to_left_and_expand():
    mps, mpo, environ = _mps_mpo_env()
    bond_idx = 0
    left = mps[bond_idx].array
    center = mps[bond_idx + 1].array
    l_array = environ.read("L", bond_idx - 1)
    r_array = environ.read("R", bond_idx + 2)

    result = cbe_shrewd_selection_right_to_left(
        mps, mpo, bond_idx, l_array, r_array, cbe_Dmax=3,
        cbe_eps_pre=1e-12, cbe_eps_final=1e-12
    )

    assert result.D_expand > 0
    assert result.tensor.shape[:2] == left.shape[:2]
    assert result.tensor.shape[2] == result.D_expand
    assert np.linalg.norm(left.reshape(-1, left.shape[-1]).conj().T @ result.tensor.reshape(-1, result.D_expand)) < 1e-10

    expanded = cbe_expand_right_to_left(left, center, result.tensor)
    assert expanded.A_ex.shape[-1] == left.shape[-1] + result.D_expand
    assert expanded.C_ex.shape[0] == center.shape[0] + result.D_expand
    assert expanded.orthogonality_error < 1e-10
    assert expanded.isometry_error < 1e-10
    assert expanded.wavefunction_error < 1e-10


def test_cbe_shrewd_selection_left_to_right_and_expand():
    mps, mpo, environ = _mps_mpo_env()
    bond_idx = 0
    center = mps[bond_idx].array
    right = mps[bond_idx + 1].array
    l_array = environ.read("L", bond_idx - 1)
    r_array = environ.read("R", bond_idx + 2)

    result = cbe_shrewd_selection_left_to_right(
        mps, mpo, bond_idx, l_array, r_array, cbe_Dmax=3,
        cbe_eps_pre=1e-12, cbe_eps_final=1e-12
    )

    assert result.D_expand > 0
    assert result.tensor.shape[1:] == right.shape[1:]
    assert result.tensor.shape[0] == result.D_expand
    assert np.linalg.norm(result.tensor.reshape(result.D_expand, -1) @ right.reshape(right.shape[0], -1).conj().T) < 1e-10

    expanded = cbe_expand_left_to_right(center, right, result.tensor)
    assert expanded.C_ex.shape[-1] == center.shape[-1] + result.D_expand
    assert expanded.B_ex.shape[0] == right.shape[0] + result.D_expand
    assert expanded.orthogonality_error < 1e-10
    assert expanded.isometry_error < 1e-10
    assert expanded.wavefunction_error < 1e-10


def test_cbe_zero_expand_keeps_local_tensors_unchanged():
    mps, mpo, environ = _mps_mpo_env()
    bond_idx = 0
    left = mps[bond_idx].array
    center = mps[bond_idx + 1].array
    l_array = environ.read("L", bond_idx - 1)
    r_array = environ.read("R", bond_idx + 2)

    result = cbe_shrewd_selection_right_to_left(
        mps, mpo, bond_idx, l_array, r_array, cbe_Dmax=1,
        cbe_eps_pre=1e-12, cbe_eps_final=1e-12, cbe_max_expand=0
    )

    assert result.D_expand == 0
    expanded = cbe_expand_right_to_left(left, center, result.tensor)
    assert expanded.A_ex.shape == left.shape
    assert expanded.C_ex.shape == center.shape
    assert np.allclose(expanded.A_ex, left)
    assert np.allclose(expanded.C_ex, center)



def test_tdvp_ps_cbe_warmup_calls_and_none_does_not_call():
    model = _two_site_model()
    mpo = Mpo(model)

    cbe_mps = Mps.hartree_product_state(model, {"v0": 0, "v1": 0})
    cbe_mps.evolve_config = EvolveConfig(
        EvolveMethod.tdvp_ps,
        expansion_method="cbe", cbe_Dmax=3, cbe_max_expand=1, cbe_warmup_time=1.0,
    )
    cbe_mps = cbe_mps.evolve(mpo, 0.01)
    stats = cbe_mps.evolve_config.cbe_last_stats
    assert stats["cbe_active"] is True
    assert stats["num_calls"] > 0
    assert stats["num_expanded_bonds"] > 0
    assert max(cbe_mps.bond_dims) <= 3

    none_mps = Mps.hartree_product_state(model, {"v0": 0, "v1": 0})
    none_mps.evolve_config = EvolveConfig(
        EvolveMethod.tdvp_ps,
        expansion_method="none",
    )
    none_mps = none_mps.evolve(mpo, 0.01)
    stats = none_mps.evolve_config.cbe_last_stats
    assert stats["cbe_active"] is False
    assert stats["num_calls"] == 0
