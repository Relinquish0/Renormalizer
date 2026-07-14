# -*- coding: utf-8 -*-
"""Local controlled-bond-expansion helpers for one-site TDVP.

This module is intentionally local-only: it does not mutate an MPS and does not
drive time evolution. The TDVP sweep can call these helpers immediately before a
one-site local evolution, then perform the evolution in the expanded space.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import scipy.linalg
import scipy.sparse.linalg

from renormalizer.mps.backend import np
from renormalizer.mps.lib import contract_one_site
from renormalizer.mps.svd_qn import add_outer
from renormalizer.mps.matrix import asnumpy, tensordot


logger = logging.getLogger(__name__)


@dataclass
class CBESelectionResult:
    tensor: np.ndarray
    singular_values: np.ndarray
    D_expand: int
    debug_info: Dict
    qn: Optional[np.ndarray] = None


@dataclass
class CBEExpansionResult:
    A_ex: Optional[np.ndarray] = None
    B_ex: Optional[np.ndarray] = None
    C_ex: Optional[np.ndarray] = None
    L_ex: Optional[np.ndarray] = None
    R_ex: Optional[np.ndarray] = None
    orthogonality_error: float = 0.0
    isometry_error: float = 0.0
    wavefunction_error: float = 0.0
    debug_info: Optional[Dict] = None


def _array(tensor):
    return asnumpy(tensor.array if hasattr(tensor, "array") else tensor)


def _select_count(singular_values, eps, max_expand=None, eps_trim=1e-12):
    if max_expand is not None:
        max_expand = int(max_expand)
        if max_expand <= 0:
            return 0
    if len(singular_values) == 0:
        return 0
    s0 = float(singular_values[0])
    if s0 <= eps_trim:
        return 0
    keep = int(np.count_nonzero((singular_values > eps * s0) & (singular_values > eps_trim)))
    if max_expand is not None:
        keep = min(keep, max_expand)
    return keep


def _orthonormal_columns(mat, eps_trim=1e-12):
    if mat.size == 0 or mat.shape[1] == 0:
        return mat[:, :0]
    q, r = scipy.linalg.qr(mat, mode="economic")
    keep = np.abs(np.diag(r)) > eps_trim
    return q[:, keep]


def _orthonormal_rows(mat, eps_trim=1e-12):
    return _orthonormal_columns(mat.conj().T, eps_trim).conj().T



def _hpsi_qn_factors(A_l, B_r, W_l, W_r, l_array, r_array, lidx, ridx):
    """Return factor matrices for one qn block of the two-site H|psi>.

    The dense two-site TDVP contraction would form
    ``Htheta[a, d, g, l]`` for all output virtual/physical indices.  CBE only
    needs selected qn-compatible rows ``(a, d)`` and columns ``(g, l)``, so this
    helper contracts the left and right halves separately.  The qn block is
    represented as ``left_mat @ right_mat.T`` and can be decomposed without
    explicitly forming a large dense block.
    """
    if len(lidx) == 0 or len(ridx) == 0:
        empty = np.zeros((0, 0), dtype=A_l.dtype)
        return empty, empty

    phys_l = A_l.shape[1]
    right_bond = B_r.shape[2]

    row_left = np.asarray(lidx // phys_l, dtype=int)
    row_phys = np.asarray(lidx % phys_l, dtype=int)
    col_phys = np.asarray(ridx // right_bond, dtype=int)
    col_right = np.asarray(ridx % right_bond, dtype=int)

    L_rows = l_array[row_left, :, :]
    Wl_rows = np.take(W_l, row_phys, axis=1).transpose(1, 0, 2, 3)
    left_proj = np.einsum(
        "rbc,rbef,cem->rfm", L_rows, Wl_rows, A_l, optimize=True
    )

    R_cols = r_array[col_right, :, :]
    Wr_cols = np.take(W_r, col_phys, axis=1).transpose(1, 0, 2, 3)
    right_proj = np.einsum(
        "nfhj,njk,mhk->nfm", Wr_cols, R_cols, B_r, optimize=True
    )

    left_mat = left_proj.reshape(len(lidx), -1)
    right_mat = right_proj.reshape(len(ridx), -1)
    return left_mat, right_mat


def _hpsi_qn_block(A_l, B_r, W_l, W_r, l_array, r_array, lidx, ridx):
    """Return one qn block of the two-site H|psi> matrix."""
    left_mat, right_mat = _hpsi_qn_factors(
        A_l, B_r, W_l, W_r, l_array, r_array, lidx, ridx
    )
    if left_mat.size == 0 or right_mat.size == 0:
        return np.zeros((len(lidx), len(ridx)), dtype=A_l.dtype)
    return left_mat @ right_mat.T


def _project_left_factor(left_mat, basis):
    if basis.shape[1] == 0 or left_mat.size == 0:
        return left_mat
    return left_mat - basis @ (basis.conj().T @ left_mat)


def _project_right_factor(right_mat, basis):
    if basis.shape[0] == 0 or right_mat.size == 0:
        return right_mat
    return right_mat - basis.conj().T @ (basis @ right_mat)


def _paper_preselection_dim(D_before, W_l, W_r, max_new, cbe_Dpre=None):
    if cbe_Dpre is not None:
        dim = max(int(cbe_Dpre), 0)
    else:
        mpo_mid = max(int(W_l.shape[-1]), int(W_r.shape[0]), 1)
        dim = max(int(np.ceil(max(int(D_before), 1) / mpo_mid)), 1)
    if max_new is not None:
        dim = min(dim, max(int(max_new), 0))
    return dim


def _factorized_top_svd(left_mat, right_mat, max_rank, eps_trim=1e-12):
    """Leading SVD of ``left_mat @ right_mat.T`` without forming it."""
    m = left_mat.shape[0]
    n = right_mat.shape[0]
    inner = left_mat.shape[1] if left_mat.ndim == 2 else 0
    rank_bound = min(m, n, inner)
    if max_rank is not None:
        rank_bound = min(rank_bound, int(max_rank))
    if rank_bound <= 0:
        return (
            np.zeros((m, 0), dtype=left_mat.dtype),
            np.array([], dtype=float),
            np.zeros((0, n), dtype=right_mat.dtype),
        )

    exact_rank = min(m, n, inner)
    if exact_rank <= 64 or rank_bound >= exact_rank - 1:
        qx, rx = scipy.linalg.qr(left_mat, mode="economic")
        qz, rz = scipy.linalg.qr(right_mat.conj(), mode="economic")
        small = rx @ rz.conj().T
        us, s, vhs = scipy.linalg.svd(small, full_matrices=False)
        keep = min(rank_bound, len(s))
        s = s[:keep]
        u = qx @ us[:, :keep]
        vh = vhs[:keep, :] @ qz.conj().T
    else:
        k = max(1, rank_bound)

        def matvec(v):
            return left_mat @ (right_mat.T @ v)

        def rmatvec(v):
            return right_mat.conj() @ (left_mat.conj().T @ v)

        op = scipy.sparse.linalg.LinearOperator(
            (m, n), matvec=matvec, rmatvec=rmatvec,
            dtype=np.result_type(left_mat.dtype, right_mat.dtype),
        )
        try:
            u, s, vh = scipy.sparse.linalg.svds(op, k=k, which="LM")
            order = np.argsort(s)[::-1]
            s = s[order]
            u = u[:, order]
            vh = vh[order, :]
        except Exception:
            logger.debug("CBE factorized svds failed; falling back to dense QR SVD", exc_info=True)
            qx, rx = scipy.linalg.qr(left_mat, mode="economic")
            qz, rz = scipy.linalg.qr(right_mat.conj(), mode="economic")
            small = rx @ rz.conj().T
            us, s, vhs = scipy.linalg.svd(small, full_matrices=False)
            keep = min(rank_bound, len(s))
            s = s[:keep]
            u = qx @ us[:, :keep]
            vh = vhs[:keep, :] @ qz.conj().T

    keep_nonzero = s > eps_trim
    return u[:, keep_nonzero], s[keep_nonzero], vh[keep_nonzero, :]



def _project_rows_against_basis(rows, basis):
    if basis.shape[0] == 0 or rows.size == 0:
        return rows
    return rows - (rows @ basis.conj().T) @ basis


def _algorithm1_right_to_left_block(
    left_factor, right_factor, aq, bq, D_before, W_l, W_r, remaining,
    cbe_Dpre, cbe_eps_pre, cbe_eps_final, cbe_eps_trim,
):
    """Paper-style shrewd selection in one qn block for A_tr.

    The block is never formed densely.  We first find the right environment
    response, preselect a small left complement A_pr, then perform final
    selection only in span(A_pr).
    """
    pre_dim = _paper_preselection_dim(D_before, W_l, W_r, remaining, cbe_Dpre)
    if pre_dim <= 0:
        return np.zeros((left_factor.shape[0], 0), dtype=left_factor.dtype), np.array([]), np.array([]), 0

    right_orth = _project_right_factor(right_factor, bq)
    if np.linalg.norm(right_orth) <= cbe_eps_trim:
        return np.zeros((left_factor.shape[0], 0), dtype=left_factor.dtype), np.array([]), np.array([]), 0

    _, s_right, vh_right = _factorized_top_svd(left_factor, right_orth, pre_dim, cbe_eps_trim)
    keep_right = _select_count(s_right, cbe_eps_pre, pre_dim, cbe_eps_trim)
    if keep_right == 0:
        return np.zeros((left_factor.shape[0], 0), dtype=left_factor.dtype), s_right, np.array([]), 0

    v_pre = vh_right[:keep_right].conj().T
    left_probe = left_factor @ (right_orth.T @ v_pre)
    left_probe = _project_left_factor(left_probe, aq)
    if np.linalg.norm(left_probe) <= cbe_eps_trim:
        return np.zeros((left_factor.shape[0], 0), dtype=left_factor.dtype), s_right, np.array([]), 0

    u_pre, s_pre, _ = scipy.linalg.svd(left_probe, full_matrices=False)
    keep_pre = _select_count(s_pre, cbe_eps_pre, pre_dim, cbe_eps_trim)
    if keep_pre == 0:
        return np.zeros((left_factor.shape[0], 0), dtype=left_factor.dtype), s_pre, np.array([]), 0

    A_pr = _orthonormal_columns(u_pre[:, :keep_pre], cbe_eps_trim)
    if A_pr.shape[1] == 0:
        return np.zeros((left_factor.shape[0], 0), dtype=left_factor.dtype), s_pre, np.array([]), 0

    final_rank = remaining if remaining is not None else A_pr.shape[1]
    small_left = A_pr.conj().T @ left_factor
    u_final, s_final, _ = _factorized_top_svd(small_left, right_orth, final_rank, cbe_eps_trim)
    keep_final = _select_count(s_final, cbe_eps_final, final_rank, cbe_eps_trim)
    if keep_final == 0:
        return np.zeros((left_factor.shape[0], 0), dtype=left_factor.dtype), s_pre, s_final, A_pr.shape[1]
    return A_pr @ u_final[:, :keep_final], s_pre, s_final[:keep_final], A_pr.shape[1]


def _algorithm1_left_to_right_block(
    left_factor, right_factor, aq, bq, D_before, W_l, W_r, remaining,
    cbe_Dpre, cbe_eps_pre, cbe_eps_final, cbe_eps_trim,
):
    """Paper-style shrewd selection in one qn block for B_tr."""
    pre_dim = _paper_preselection_dim(D_before, W_l, W_r, remaining, cbe_Dpre)
    if pre_dim <= 0:
        return np.zeros((0, right_factor.shape[0]), dtype=right_factor.dtype), np.array([]), np.array([]), 0

    left_orth = _project_left_factor(left_factor, aq)
    if np.linalg.norm(left_orth) <= cbe_eps_trim:
        return np.zeros((0, right_factor.shape[0]), dtype=right_factor.dtype), np.array([]), np.array([]), 0

    _, s_left, vh_left = _factorized_top_svd(left_orth, right_factor, pre_dim, cbe_eps_trim)
    keep_pre = _select_count(s_left, cbe_eps_pre, pre_dim, cbe_eps_trim)
    if keep_pre == 0:
        return np.zeros((0, right_factor.shape[0]), dtype=right_factor.dtype), s_left, np.array([]), 0

    B_pr = _project_rows_against_basis(vh_left[:keep_pre], bq)
    B_pr = _orthonormal_rows(B_pr, cbe_eps_trim)
    if B_pr.shape[0] == 0:
        return np.zeros((0, right_factor.shape[0]), dtype=right_factor.dtype), s_left, np.array([]), 0

    final_rank = remaining if remaining is not None else B_pr.shape[0]
    small_right = right_factor.T @ B_pr.conj().T
    _, s_final, vh_final = _factorized_top_svd(left_orth, small_right.T, final_rank, cbe_eps_trim)
    keep_final = _select_count(s_final, cbe_eps_final, final_rank, cbe_eps_trim)
    if keep_final == 0:
        return np.zeros((0, right_factor.shape[0]), dtype=right_factor.dtype), s_left, s_final, B_pr.shape[0]
    return vh_final[:keep_final] @ B_pr, s_left, s_final[:keep_final], B_pr.shape[0]


def _max_new_dim(D_before, cbe_Dmax=None, cbe_max_expand=None):
    if cbe_Dmax is None:
        available = None
    else:
        available = max(int(cbe_Dmax) - int(D_before), 0)
    if cbe_max_expand is None:
        return available
    if available is None:
        return int(cbe_max_expand)
    return min(available, int(cbe_max_expand))



def _qn_key(qn):
    return tuple(np.asarray(qn, dtype=int).tolist())


def _qn_mask(qn_array, qn):
    key = _qn_key(qn)
    return np.array([_qn_key(item) == key for item in qn_array])


def _candidate_bond_qns(mps, q_left, q_right):
    qns = []
    seen = set()
    for qn in q_left:
        qr = mps.qntot - qn
        key = _qn_key(qn)
        if key in seen or not np.any(_qn_mask(q_right, qr)):
            continue
        seen.add(key)
        qns.append(np.array(qn, dtype=int))
    return qns


def _select_qn_block_columns(raw_block, lidx, left_dim, qn, basis, accepted,
        accepted_qn, eps_trim):
    vec = np.zeros(left_dim, dtype=raw_block.dtype)
    vec[lidx] = raw_block
    for ibasis in range(basis.shape[1]):
        vec = vec - basis[:, ibasis] * np.vdot(basis[:, ibasis], vec)
    for basis_vec, basis_qn in zip(accepted, accepted_qn):
        if _qn_key(basis_qn) == _qn_key(qn):
            vec = vec - basis_vec * np.vdot(basis_vec, vec)
    norm = np.linalg.norm(vec)
    if norm <= eps_trim:
        return None
    return vec / norm


def _select_qn_block_rows(raw_block, ridx, right_dim, qn, basis, accepted,
        accepted_qn, eps_trim):
    vec = np.zeros(right_dim, dtype=raw_block.dtype)
    vec[ridx] = raw_block
    for ibasis in range(basis.shape[0]):
        vec = vec - np.vdot(vec, basis[ibasis]) * basis[ibasis]
    for basis_vec, basis_qn in zip(accepted, accepted_qn):
        if _qn_key(basis_qn) == _qn_key(qn):
            vec = vec - np.vdot(vec, basis_vec) * basis_vec
    norm = np.linalg.norm(vec)
    if norm <= eps_trim:
        return None
    return vec / norm


def cbe_shrewd_selection_right_to_left(
    mps,
    mpo,
    bond_idx: int,
    l_array,
    r_array,
    cbe_Dmax: int = None,
    cbe_eps_pre: float = 1e-4,
    cbe_eps_final: float = 1e-6,
    cbe_eps_trim: float = 1e-12,
    cbe_max_expand: int = None,
    cbe_Dpre: int = None,
) -> CBESelectionResult:
    """Select left-site complement vectors for expanding bond ``bond_idx``.

    Selection is performed independently inside each legal virtual-qn block.
    This makes every returned vector carry a known bond quantum number instead
    of assigning qn labels after a dense SVD.
    """
    A_l = _array(mps[bond_idx])
    B_r = _array(mps[bond_idx + 1])
    W_l = _array(mpo[bond_idx])
    W_r = _array(mpo[bond_idx + 1])
    l_array = _array(l_array)
    r_array = _array(r_array)
    left_dim = A_l.shape[0] * A_l.shape[1]
    right_dim = B_r.shape[1] * B_r.shape[2]
    D_before = A_l.shape[2]
    max_new = _max_new_dim(D_before, cbe_Dmax, cbe_max_expand)

    A_basis_raw = A_l.reshape(left_dim, D_before)
    B_basis_raw = B_r.reshape(B_r.shape[0], right_dim)
    A_basis = _orthonormal_columns(A_basis_raw, cbe_eps_trim)
    q_left = add_outer(np.array(mps.qn[bond_idx]), np.array(mps._get_sigmaqn(bond_idx))).reshape(-1, mps.model.qn_size)
    q_right = add_outer(np.array(mps._get_sigmaqn(bond_idx + 1)), np.array(mps.qn[bond_idx + 2])).reshape(-1, mps.model.qn_size)
    bond_qn = np.array(mps.qn[bond_idx + 1])

    accepted = []
    accepted_qn = []
    s_all = []
    s_kept = []
    D_pre_total = 0
    remaining = max_new

    for qn in _candidate_bond_qns(mps, q_left, q_right):
        if remaining is not None and remaining <= 0:
            break
        lmask = _qn_mask(q_left, qn)
        rmask = _qn_mask(q_right, mps.qntot - qn)
        lidx = np.where(lmask)[0]
        ridx = np.where(rmask)[0]
        if len(lidx) == 0 or len(ridx) == 0:
            continue
        left_factor, right_factor = _hpsi_qn_factors(
            A_l, B_r, W_l, W_r, l_array, r_array, lidx, ridx
        )
        aq = A_basis_raw[np.ix_(lidx, _qn_mask(bond_qn, qn))]
        bq = B_basis_raw[np.ix_(_qn_mask(bond_qn, qn), ridx)]
        aq = _orthonormal_columns(aq, cbe_eps_trim)
        bq = _orthonormal_rows(bq, cbe_eps_trim)
        U_block, S_pre_block, S_block, D_pre_block = _algorithm1_right_to_left_block(
            left_factor, right_factor, aq, bq, D_before, W_l, W_r, remaining,
            cbe_Dpre, cbe_eps_pre, cbe_eps_final, cbe_eps_trim,
        )
        s_all.extend(S_pre_block.tolist())
        D_pre_total += D_pre_block
        for ivec in range(U_block.shape[1]):
            vec = _select_qn_block_columns(
                U_block[:, ivec], lidx, left_dim, qn, A_basis,
                accepted, accepted_qn, cbe_eps_trim,
            )
            if vec is None:
                continue
            accepted.append(vec)
            accepted_qn.append(qn)
            s_kept.append(S_block[ivec])
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break

    if not accepted:
        A_tr = np.zeros(A_l.shape[:2] + (0,), dtype=A_l.dtype)
        debug = {
            "reason": "qn_block_selection_zero",
            "D_before": D_before,
            "D_pre": 0,
            "S_pre": np.array(s_all),
            "S_final": np.array([]),
            "selected_qn": [],
        }
        return CBESelectionResult(A_tr, np.array([]), 0, debug, np.zeros((0, mps.model.qn_size), dtype=int))

    A_tr_mat = np.stack(accepted, axis=1)
    D_expand = A_tr_mat.shape[1]
    S_final = np.array(s_kept)
    A_tr = A_tr_mat.reshape(A_l.shape[0], A_l.shape[1], D_expand)
    qn_new = np.array(accepted_qn, dtype=int)

    orth_error = float(np.linalg.norm(A_basis.conj().T @ A_tr_mat)) if D_expand else 0.0
    debug = {
        "reason": "selected" if D_expand else "final_selection_zero",
        "D_before": D_before,
        "D_pre": int(D_pre_total),
        "S_pre": np.array(s_all),
        "S_final": S_final,
        "orthogonality_error": orth_error,
        "selected_qn": qn_new.tolist(),
    }
    logger.debug("CBE shrewd selection direction=right-to-left bond_index=%s D_before=%s D_expand=%s selected_qn=%s largest_final_singular_values=%s orthogonality_error=%s reason=%s", bond_idx, D_before, D_expand, qn_new.tolist(), S_final[: min(5, len(S_final))], orth_error, debug["reason"])
    return CBESelectionResult(A_tr, S_final[:D_expand], D_expand, debug, qn_new)


def cbe_shrewd_selection_left_to_right(
    mps,
    mpo,
    bond_idx: int,
    l_array,
    r_array,
    cbe_Dmax: int = None,
    cbe_eps_pre: float = 1e-4,
    cbe_eps_final: float = 1e-6,
    cbe_eps_trim: float = 1e-12,
    cbe_max_expand: int = None,
    cbe_Dpre: int = None,
) -> CBESelectionResult:
    """Select right-site complement vectors for expanding bond ``bond_idx``."""
    A_l = _array(mps[bond_idx])
    B_r = _array(mps[bond_idx + 1])
    W_l = _array(mpo[bond_idx])
    W_r = _array(mpo[bond_idx + 1])
    l_array = _array(l_array)
    r_array = _array(r_array)
    left_dim = A_l.shape[0] * A_l.shape[1]
    right_dim = B_r.shape[1] * B_r.shape[2]
    D_before = B_r.shape[0]
    max_new = _max_new_dim(D_before, cbe_Dmax, cbe_max_expand)

    A_basis_raw = A_l.reshape(left_dim, A_l.shape[2])
    B_basis_raw = B_r.reshape(D_before, right_dim)
    B_basis = _orthonormal_rows(B_basis_raw, cbe_eps_trim)
    q_left = add_outer(np.array(mps.qn[bond_idx]), np.array(mps._get_sigmaqn(bond_idx))).reshape(-1, mps.model.qn_size)
    q_right = add_outer(np.array(mps._get_sigmaqn(bond_idx + 1)), np.array(mps.qn[bond_idx + 2])).reshape(-1, mps.model.qn_size)
    bond_qn = np.array(mps.qn[bond_idx + 1])

    accepted = []
    accepted_qn = []
    s_all = []
    s_kept = []
    D_pre_total = 0
    remaining = max_new

    for qn in _candidate_bond_qns(mps, q_left, q_right):
        if remaining is not None and remaining <= 0:
            break
        lmask = _qn_mask(q_left, qn)
        rmask = _qn_mask(q_right, mps.qntot - qn)
        lidx = np.where(lmask)[0]
        ridx = np.where(rmask)[0]
        if len(lidx) == 0 or len(ridx) == 0:
            continue
        left_factor, right_factor = _hpsi_qn_factors(
            A_l, B_r, W_l, W_r, l_array, r_array, lidx, ridx
        )
        aq = A_basis_raw[np.ix_(lidx, _qn_mask(bond_qn, qn))]
        bq = B_basis_raw[np.ix_(_qn_mask(bond_qn, qn), ridx)]
        aq = _orthonormal_columns(aq, cbe_eps_trim)
        bq = _orthonormal_rows(bq, cbe_eps_trim)
        Vh_block, S_pre_block, S_block, D_pre_block = _algorithm1_left_to_right_block(
            left_factor, right_factor, aq, bq, D_before, W_l, W_r, remaining,
            cbe_Dpre, cbe_eps_pre, cbe_eps_final, cbe_eps_trim,
        )
        s_all.extend(S_pre_block.tolist())
        D_pre_total += D_pre_block
        for ivec in range(Vh_block.shape[0]):
            vec = _select_qn_block_rows(
                Vh_block[ivec], ridx, right_dim, qn, B_basis,
                accepted, accepted_qn, cbe_eps_trim,
            )
            if vec is None:
                continue
            accepted.append(vec)
            accepted_qn.append(qn)
            s_kept.append(S_block[ivec])
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break

    if not accepted:
        B_tr = np.zeros((0,) + B_r.shape[1:], dtype=B_r.dtype)
        debug = {
            "reason": "qn_block_selection_zero",
            "D_before": D_before,
            "D_pre": 0,
            "S_pre": np.array(s_all),
            "S_final": np.array([]),
            "selected_qn": [],
        }
        return CBESelectionResult(B_tr, np.array([]), 0, debug, np.zeros((0, mps.model.qn_size), dtype=int))

    B_tr_mat = np.stack(accepted, axis=0)
    D_expand = B_tr_mat.shape[0]
    S_final = np.array(s_kept)
    B_tr = B_tr_mat.reshape(D_expand, B_r.shape[1], B_r.shape[2])
    qn_new = np.array(accepted_qn, dtype=int)

    orth_error = float(np.linalg.norm(B_tr_mat @ B_basis.conj().T)) if D_expand else 0.0
    debug = {
        "reason": "selected" if D_expand else "final_selection_zero",
        "D_before": D_before,
        "D_pre": int(D_pre_total),
        "S_pre": np.array(s_all),
        "S_final": S_final,
        "orthogonality_error": orth_error,
        "selected_qn": qn_new.tolist(),
    }
    logger.debug("CBE shrewd selection direction=left-to-right bond_index=%s D_before=%s D_expand=%s selected_qn=%s largest_final_singular_values=%s orthogonality_error=%s reason=%s", bond_idx, D_before, D_expand, qn_new.tolist(), S_final[: min(5, len(S_final))], orth_error, debug["reason"])
    return CBESelectionResult(B_tr, S_final[:D_expand], D_expand, debug, qn_new)


def cbe_expand_right_to_left(A_l, C_right, A_tr, l_array=None, W_l=None, bond_idx=None) -> CBEExpansionResult:
    A_l = _array(A_l)
    C_right = _array(C_right)
    A_tr = _array(A_tr)
    D_expand = A_tr.shape[-1]
    if D_expand == 0:
        return CBEExpansionResult(
            A_ex=A_l.copy(),
            C_ex=C_right.copy(),
            orthogonality_error=0.0,
            isometry_error=float(np.linalg.norm(A_l.reshape(-1, A_l.shape[-1]).conj().T @ A_l.reshape(-1, A_l.shape[-1]) - np.eye(A_l.shape[-1]))),
            wavefunction_error=0.0,
            debug_info={"D_expand": 0},
        )

    A_ex = np.concatenate([A_l, A_tr], axis=2)
    zero = np.zeros((D_expand,) + C_right.shape[1:], dtype=C_right.dtype)
    C_ex = np.concatenate([C_right, zero], axis=0)
    A_mat = A_l.reshape(-1, A_l.shape[-1])
    A_tr_mat = A_tr.reshape(-1, D_expand)
    A_ex_mat = A_ex.reshape(-1, A_ex.shape[-1])
    AC = asnumpy(tensordot(A_l, C_right, axes=1))
    AC_ex = asnumpy(tensordot(A_ex, C_ex, axes=1))
    L_ex = None if l_array is None or W_l is None else contract_one_site(l_array, A_ex, W_l, "L")
    orth_error = float(np.linalg.norm(A_mat.conj().T @ A_tr_mat))
    iso_error = float(np.linalg.norm(A_ex_mat.conj().T @ A_ex_mat - np.eye(A_ex_mat.shape[1])))
    wf_error = float(np.linalg.norm(AC_ex - AC))
    logger.debug(
        "CBE local expansion direction=right-to-left D_before=%s D_expand=%s "
        "bond_index=%s D_after_expand=%s orthogonality_error=%s isometry_error=%s "
        "wavefunction_error=%s",
        A_l.shape[-1], D_expand, bond_idx, A_ex.shape[-1], orth_error, iso_error, wf_error,
    )
    return CBEExpansionResult(A_ex=A_ex, C_ex=C_ex, L_ex=L_ex,
        orthogonality_error=orth_error, isometry_error=iso_error,
        wavefunction_error=wf_error, debug_info={"D_expand": D_expand})


def cbe_expand_left_to_right(C_left, B_right, B_tr, r_array=None, W_right=None, bond_idx=None) -> CBEExpansionResult:
    C_left = _array(C_left)
    B_right = _array(B_right)
    B_tr = _array(B_tr)
    D_expand = B_tr.shape[0]
    if D_expand == 0:
        B_mat = B_right.reshape(B_right.shape[0], -1)
        return CBEExpansionResult(
            B_ex=B_right.copy(),
            C_ex=C_left.copy(),
            orthogonality_error=0.0,
            isometry_error=float(np.linalg.norm(B_mat @ B_mat.conj().T - np.eye(B_mat.shape[0]))),
            wavefunction_error=0.0,
            debug_info={"D_expand": 0},
        )

    B_ex = np.concatenate([B_right, B_tr], axis=0)
    zero = np.zeros(C_left.shape[:-1] + (D_expand,), dtype=C_left.dtype)
    C_ex = np.concatenate([C_left, zero], axis=2)
    B_mat = B_right.reshape(B_right.shape[0], -1)
    B_tr_mat = B_tr.reshape(D_expand, -1)
    B_ex_mat = B_ex.reshape(B_ex.shape[0], -1)
    CB = asnumpy(tensordot(C_left, B_right, axes=1))
    CB_ex = asnumpy(tensordot(C_ex, B_ex, axes=1))
    R_ex = None if r_array is None or W_right is None else contract_one_site(r_array, B_ex, W_right, "R")
    orth_error = float(np.linalg.norm(B_tr_mat @ B_mat.conj().T))
    iso_error = float(np.linalg.norm(B_ex_mat @ B_ex_mat.conj().T - np.eye(B_ex_mat.shape[0])))
    wf_error = float(np.linalg.norm(CB_ex - CB))
    logger.debug(
        "CBE local expansion direction=left-to-right D_before=%s D_expand=%s "
        "bond_index=%s D_after_expand=%s orthogonality_error=%s isometry_error=%s "
        "wavefunction_error=%s",
        B_right.shape[0], D_expand, bond_idx, B_ex.shape[0], orth_error, iso_error, wf_error,
    )
    return CBEExpansionResult(B_ex=B_ex, C_ex=C_ex, R_ex=R_ex,
        orthogonality_error=orth_error, isometry_error=iso_error,
        wavefunction_error=wf_error, debug_info={"D_expand": D_expand})
