from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple, Union

import opt_einsum as oe

from renormalizer import Model, Op
from renormalizer.model.basis import BasisDummy, BasisSet
from renormalizer.mps.backend import np, backend, xp
from renormalizer.mps.matrix import asnumpy, asxp_oe_args, tensordot
from renormalizer.mps.oe_contract_wrap import oe_contract
from renormalizer.mps.svd_qn import add_outer, get_qn_mask
from renormalizer.tn.node import TreeNodeTensor, copy_connection
from renormalizer.tn.tree import TTNO, TTNS, get_skip_pidx
from renormalizer.tn.treebase import BasisTree, Tree
from renormalizer.utils import calc_vn_entropy_dm
from renormalizer.utils.configs import CompressConfig, CompressCriteria, EvolveConfig, EvolveMethod


logger = logging.getLogger(__name__)


class MsTreeNodeTensor(TreeNodeTensor):
    """
    Tree tensor node with a leading multiset index.

    The tensor shape is
    ``[nset, child1, child2, ..., physical1, physical2, ..., parent]``.
    The virtual-bond quantum number ``qn`` is shared by all sets.
    """

    def check_canonical(self, atol=None, assertion=True):
        if atol is None:
            atol = backend.canonical_atol
        tensor = self.tensor.reshape(self.tensor.shape[0], -1, self.tensor.shape[-1])
        s = np.einsum("nli,nlj->nij", tensor.conj(), tensor)
        eye = np.eye(tensor.shape[-1])
        res = np.allclose(s, eye, atol=atol)
        if assertion:
            assert res
        return res


class MsTTNBase(Tree):
    def __init__(self, basis: BasisTree, root: MsTreeNodeTensor):
        self.basis = basis
        super().__init__(root)
        self.tn2bn = {tn: bn for tn, bn in zip(self.node_list, self.basis.node_list)}
        self.tn2dofs = {tn: bn.dofs for tn, bn in self.tn2bn.items()}

    @property
    def pbond_dims(self):
        return self.basis.pbond_dims


class MsTTNS(MsTTNBase):
    def __init__(
        self,
        basis: BasisTree,
        nset: int = None,
        condition: Dict = None,
        root: MsTreeNodeTensor = None,
        ttns_list: Sequence[TTNS] = None,
    ):
        if ttns_list is not None:
            if root is not None or nset is not None:
                raise ValueError("`ttns_list` can not be combined with `root` or `nset`.")
            root = self._root_from_ttns_list(basis, ttns_list)
            nset = len(ttns_list)
        elif root is None:
            if nset is None:
                raise ValueError("`nset` is required when no root or ttns_list is provided.")
            ref = TTNS(basis, condition=condition)
            root = self._root_from_ttns_list(basis, [ref.copy() for _ in range(nset)])
        else:
            if nset is None:
                nset = root.tensor.shape[0]

        self.nset = int(nset)
        super().__init__(basis, root)
        self.coeff = 1
        self.compress_config = CompressConfig(CompressCriteria.fixed)
        self.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, force_ovlp=False)
        self._population_cache = None
        self.check_shape()

    @staticmethod
    def _root_from_ttns_list(basis: BasisTree, ttns_list: Sequence[TTNS]) -> MsTreeNodeTensor:
        if len(ttns_list) == 0:
            raise ValueError("`ttns_list` must not be empty.")
        ref = ttns_list[0]
        if ref.basis is not basis:
            if len(ref.basis.node_list) != len(basis.node_list):
                raise ValueError("Basis tree incompatible with the TTNS list.")
        nodes = []
        for inode, ref_node in enumerate(ref.node_list):
            tensors = []
            for ttns in ttns_list:
                node = ttns.node_list[inode]
                if node.shape != ref_node.shape:
                    raise ValueError("All TTNS components must have identical node shapes.")
                np.testing.assert_allclose(node.qn, ref_node.qn)
                tensors.append(node.tensor)
            nodes.append(MsTreeNodeTensor(np.stack(tensors, axis=0), ref_node.qn.copy()))
        return copy_connection(ref.node_list, nodes)

    @classmethod
    def from_ttns_list(cls, ttns_list: Sequence[TTNS]) -> "MsTTNS":
        return cls(ttns_list[0].basis, ttns_list=ttns_list)

    def to_ttns_list(self) -> List[TTNS]:
        components = []
        for iset in range(self.nset):
            new = TTNS(self.basis)
            new.coeff = self.coeff
            new.compress_config = self.compress_config.copy()
            new.evolve_config = self.evolve_config.copy()
            for dst, src in zip(new.node_list, self.node_list):
                dst.tensor = src.tensor[iset].copy()
                dst.qn = src.qn.copy()
            components.append(new)
        return components

    def metacopy(self):
        nodes = []
        for node in self.node_list:
            shape = (self.nset,) + tuple(node.shape[1:])
            nodes.append(MsTreeNodeTensor(np.zeros(shape, dtype=node.tensor.dtype), node.qn.copy()))
        root = copy_connection(self.node_list, nodes)
        new = self.__class__(self.basis, nset=self.nset, root=root)
        new.coeff = self.coeff
        new.compress_config = self.compress_config.copy()
        new.evolve_config = self.evolve_config.copy()
        return new

    def copy(self):
        new = self.metacopy()
        for dst, src in zip(new.node_list, self.node_list):
            dst.tensor = src.tensor.copy()
            dst.qn = src.qn.copy()
        new._population_cache = None if self._population_cache is None else self._population_cache.copy()
        return new

    def to_complex(self, inplace: bool = False):
        new = self if inplace else self.metacopy()
        for dst, src in zip(new.node_list, self.node_list):
            dst.tensor = np.asarray(src.tensor, dtype=backend.complex_dtype)
            dst.qn = src.qn.copy()
        return new

    @property
    def qntot(self):
        return self.root.qn[0]

    @property
    def bond_dims(self):
        return [node.tensor.shape[-1] for node in self.node_list]

    @property
    def bond_dims_exact(self) -> np.ndarray:
        # same estimate as TTNS.bond_dims_exact; the leading multiset axis is
        # not a tensor-network bond and should not enter this local Hilbert size.
        with np.errstate(over="ignore"):
            bond_dims_exact = [None] * len(self)
            for node in self.postorder_list():
                node_idx = self.node_idx[node]
                local_dim = float(np.prod(self.pbond_dims[node_idx]))
                for child in node.children:
                    child_idx = self.node_idx[child]
                    local_dim *= bond_dims_exact[child_idx]
                bond_dims_exact[node_idx] = local_dim
            bond_dims_exact[self.node_idx[self.root]] = 1
            return np.asarray(bond_dims_exact)

    def check_shape(self):
        for snode, bnode in zip(self.node_list, self.basis.node_list):
            assert snode.tensor.shape[0] == self.nset
            assert snode.tensor.ndim == len(snode.children) + bnode.n_sets + 2
            assert snode.qn.shape[0] == snode.tensor.shape[-1]
            assert snode.qn.shape[1] == bnode.qn_size
            for i, b in enumerate(bnode.basis_sets):
                assert snode.shape[1 + len(snode.children) + i] == b.nbas

    def check_canonical(self, atol=None):
        for node in self.node_list[1:]:
            node.check_canonical(atol)
        return True

    def get_qnmat(self, node: MsTreeNodeTensor, include_parent: bool = False):
        qnbigl = np.zeros(self.basis.qn_size, dtype=int)
        for child in node.children:
            qnbigl = add_outer(qnbigl, child.qn)
        for b in self.tn2bn[node].basis_sets:
            qnbigl = add_outer(qnbigl, b.sigmaqn)
        if not include_parent:
            qnbigr = self.qntot - node.qn
            qnmat = add_outer(qnbigl, qnbigr)
            return qnbigl, qnbigr, qnmat

        qnbigr = np.zeros(self.basis.qn_size, dtype=int)
        assert node.parent is not None
        for child in node.parent.children:
            if child is node:
                continue
            qnbigr = add_outer(qnbigr, child.qn)
        for b in self.tn2bn[node.parent].basis_sets:
            qnbigr = add_outer(qnbigr, b.sigmaqn)
        qnbigr = add_outer(qnbigr, self.qntot - node.parent.qn)
        qnmat = add_outer(qnbigl, qnbigr)
        return qnbigl, qnbigr, qnmat

    def get_qnmask(self, node, include_parent=False):
        qnmat = self.get_qnmat(node, include_parent)[-1]
        return get_qn_mask(qnmat, self.qntot)

    def _get_qr_qn_plan(self, qnbigl, qnbigr):
        qntot = np.asarray(self.qntot)
        qn_size = len(qntot)
        localqnl = qnbigl.reshape(-1, qn_size)
        localqnr = qnbigr.reshape(-1, qn_size)

        seen = set()
        ordered_qn = []
        for qn in localqnl:
            qn_tuple = tuple(qn)
            if qn_tuple not in seen:
                seen.add(qn_tuple)
                ordered_qn.append(qn_tuple)

        plan = []
        for nl in ordered_qn:
            nl_array = np.asarray(nl)
            nr = qntot - nl_array
            lset = np.where(np.all(localqnl == nl_array, axis=-1))[0]
            rset = np.where(np.all(localqnr == nr, axis=-1))[0]
            if len(lset) == 0 or len(rset) == 0:
                continue
            plan.append((lset, rset, nl, tuple(nr.tolist())))
        if not plan:
            raise ValueError("Invalid quantum number")
        return plan

    def _batched_qr_qn(self, coef_batch, qnbigl, qnbigr, system: str):
        assert system in ["L", "R"]
        batch_size = coef_batch.shape[0]
        left_dim = int(np.prod(qnbigl.shape[:-1]))
        right_dim = int(np.prod(qnbigr.shape[:-1]))
        coef_matrix = xp.asarray(coef_batch.reshape(batch_size, left_dim, right_dim))

        u_blocks = []
        v_blocks = []
        qnl_list = []
        qnr_list = []
        for lset, rset, nl, nr in self._get_qr_qn_plan(qnbigl, qnbigr):
            block = coef_matrix[:, lset][:, :, rset]
            if system == "L":
                q_block, r_block = xp.linalg.qr(block, mode="reduced")
                u_block = q_block
                v_block = xp.swapaxes(r_block, -1, -2)
            else:
                q_t, r_t = xp.linalg.qr(xp.swapaxes(block, -1, -2), mode="reduced")
                u_block = xp.swapaxes(r_t, -1, -2)
                v_block = q_t

            kdim = u_block.shape[-1]
            u_full = xp.zeros((batch_size, left_dim, kdim), dtype=coef_matrix.dtype)
            v_full = xp.zeros((batch_size, right_dim, kdim), dtype=coef_matrix.dtype)
            u_full[:, lset, :] = u_block
            v_full[:, rset, :] = v_block
            u_blocks.append(u_full)
            v_blocks.append(v_full)
            qnl_list.extend([nl] * kdim)
            qnr_list.extend([nr] * kdim)

        u_batch = xp.concatenate(u_blocks, axis=-1)
        v_batch = xp.concatenate(v_blocks, axis=-1)
        return asnumpy(u_batch), np.asarray(qnl_list, dtype=int), asnumpy(v_batch), np.asarray(qnr_list, dtype=int)

    def decompose_to_parent(self, node: MsTreeNodeTensor):
        assert node.parent is not None
        qnbigl, qnbigr, _ = self.get_qnmat(node, include_parent=False)
        tensor = node.tensor.reshape(self.nset, -1, node.shape[-1])
        u, qnlnew, v, qnrnew = self._batched_qr_qn(tensor, qnbigl, qnbigr, system="L")
        node.tensor = u.reshape((self.nset,) + tuple(node.shape[1:-1]) + (u.shape[-1],))
        node.qn = qnlnew
        return v

    def merge_to_parent(self, node: MsTreeNodeTensor, v):
        parent = node.parent
        axis = 1 + node.idx_as_child
        parent.tensor = np.moveaxis(parent.tensor, axis, -1)
        merged = np.einsum("n...r,nrk->n...k", parent.tensor, v)
        parent.tensor = np.moveaxis(merged, -1, axis)

    def push_cano_to_parent(self, node: MsTreeNodeTensor):
        v = self.decompose_to_parent(node)
        self.merge_to_parent(node, v)

    def _moveaxis_for_child(self, node: MsTreeNodeTensor, ichild: int):
        qnbigl = np.zeros(self.basis.qn_size, dtype=int)
        for child in node.children:
            if child is node.children[ichild]:
                continue
            qnbigl = add_outer(qnbigl, child.qn)
        for b in self.tn2bn[node].basis_sets:
            qnbigl = add_outer(qnbigl, b.sigmaqn)
        qnbigl = add_outer(qnbigl, self.qntot - node.qn)
        qnbigr = node.children[ichild].qn
        tensor = np.moveaxis(node.tensor, 1 + ichild, -1)
        shape = list(tensor.shape)
        tensor = tensor.reshape(self.nset, -1, node.shape[1 + ichild])
        return qnbigl, qnbigr, tensor, shape

    def decompose_to_child(self, node: MsTreeNodeTensor, ichild: int):
        qnbigl, qnbigr, tensor, shape = self._moveaxis_for_child(node, ichild)
        u, qnl, v, qnr = self._batched_qr_qn(tensor, qnbigl, qnbigr, system="L")
        shape[-1] = u.shape[-1]
        node.tensor = np.moveaxis(u.reshape(shape), -1, 1 + ichild)
        node.children[ichild].qn = qnr
        return v

    def merge_to_child(self, node: MsTreeNodeTensor, ichild: int, v):
        child = node.children[ichild]
        child.tensor = np.einsum("n...r,nrk->n...k", child.tensor, v)

    def push_cano_to_child(self, node: MsTreeNodeTensor, ichild: int):
        v = self.decompose_to_child(node, ichild)
        self.merge_to_child(node, ichild, v)

    def canonicalise(self):
        for node in self.postorder_list()[:-1]:
            self.push_cano_to_parent(node)
        self._population_cache = None
        return self

    @staticmethod
    def _sum_ttns(ttns_list: Sequence[TTNS], compress=False, temp_m_trunc=None):
        if len(ttns_list) == 0:
            return None
        res = ttns_list[0].copy()
        for ttns in ttns_list[1:]:
            res = res + ttns
        if compress:
            res.canonicalise().compress(temp_m_trunc)
        return res

    @staticmethod
    def _expand_component_bond_dimension(ttns: TTNS, hint_ttno: TTNO = None, coef: float = 1e-10, ex_ttns: TTNS = None):
        ttns = ttns.copy()
        if len(ttns.node_list) == 1:
            return ttns
        ttns.compress_config.set_bonddim(len(ttns.bond_dims))
        m_target = np.minimum(
            np.asarray(ttns.compress_config.max_dims) - np.asarray(ttns.bond_dims),
            ttns.bond_dims_exact,
        ).astype(int)
        if np.all(m_target <= 0):
            return ttns

        if hint_ttno is None:
            expander = TTNS.random(ttns.basis, ttns.qntot, m_target)
        else:
            logger.debug(f"bond dimension of multiset TTN hint TTNO: {hint_ttno.bond_dims}")
            lastone = ttns if ex_ttns is None else ttns + ex_ttns
            if lastone.norm == 0:
                expander = TTNS.random(ttns.basis, ttns.qntot, m_target)
            else:
                expander_list = []
                expander_dims = np.zeros_like(m_target)
                while True:
                    nextone = hint_ttno.apply(lastone)
                    if nextone.norm == 0:
                        logger.warning("TTN hint application produced a zero expander; falling back to random expansion")
                        expander = TTNS.random(ttns.basis, ttns.qntot, m_target)
                        break
                    lastone = nextone.normalize("ttns_and_coeff")
                    lastone = lastone.canonicalise().compress(np.max(m_target))
                    expander_list.append(lastone)
                    expander = MsTTNS._sum_ttns(expander_list, compress=True, temp_m_trunc=m_target)

                    if np.all(np.asarray(expander.bond_dims) >= m_target):
                        break

                    if np.all(np.asarray(expander.bond_dims) == expander_dims):
                        logger.warning("TTN expander does not increase anymore. The expand target is too high")
                        m_target2 = np.max(m_target - np.asarray(expander_dims))
                        expander2 = hint_ttno.apply(lastone).canonicalise().compress(np.maximum(m_target2, 1))
                        expander = expander + expander2
                        break

                    expander_dims = np.asarray(expander.bond_dims)
                    temp_m_trunc = int(np.max(m_target) / np.max(hint_ttno.bond_dims)) + 1
                    lastone = lastone.canonicalise().compress(temp_m_trunc)

        ref_norm = ttns.norm
        if ref_norm == 0 and ex_ttns is not None:
            ref_norm = ex_ttns.norm
        if ref_norm == 0:
            ref_norm = 1.0
        expanded = ttns + expander.scale(coef * ref_norm, inplace=False)
        expanded = expanded.canonicalise().compress(ttns.compress_config.max_dims)
        if expanded.ttns_norm != 0:
            expanded.normalize("ttns_norm_to_coeff")
        return expanded

    def _prepare_graph_hint_state(self, ttns: TTNS, m_trunc=None):
        if ttns is None or ttns.norm == 0:
            return None
        hint = ttns.copy()
        hint.compress_config = self.compress_config.copy()
        hint.compress_config.set_bonddim(len(hint.bond_dims))
        hint.evolve_config = self.evolve_config.copy()
        if m_trunc is None:
            m_trunc = hint.compress_config.max_dims
        hint = hint.canonicalise()
        if len(hint.node_list) > 1:
            hint = hint.compress(m_trunc)
        if hint.ttns_norm != 0:
            hint.normalize("ttns_norm_to_coeff")
            hint.scale(float(abs(hint.coeff)), inplace=True)
            hint.coeff = 1.0
        return hint

    def _build_multiround_cross_hints(self, original_ttns, ms_ttno: "MsTTNO", graph_rounds: int = None):
        if graph_rounds is None:
            graph_rounds = max(self.nset - 1, 0)

        pairs_by_beta = [[] for _ in range(self.nset)]
        for pair_id, (alpha, beta) in enumerate(ms_ttno.active_pairs_index):
            if alpha == beta:
                continue
            pairs_by_beta[beta].append((alpha, ms_ttno.active_pair_ttnos[pair_id]))

        graph_compress_config = self.compress_config.copy()
        graph_compress_config.set_bonddim(len(original_ttns[0].bond_dims))
        graph_max_dims = graph_compress_config.max_dims

        hints_by_alpha = [[] for _ in range(self.nset)]
        frontier = [self._prepare_graph_hint_state(ttns, graph_max_dims) for ttns in original_ttns]
        processed_sources = set()

        for iround in range(graph_rounds):
            source_ids = [
                beta for beta, source in enumerate(frontier)
                if source is not None and beta not in processed_sources
            ]
            if not source_ids:
                break

            next_candidates = [[] for _ in range(self.nset)]
            for beta in source_ids:
                processed_sources.add(beta)
                source = frontier[beta]
                for alpha, pair_ttno in pairs_by_beta[beta]:
                    if alpha in processed_sources:
                        continue
                    driven = pair_ttno.apply(source)
                    driven = self._prepare_graph_hint_state(driven, graph_max_dims)
                    if driven is None:
                        continue
                    hints_by_alpha[alpha].append(driven)
                    next_candidates[alpha].append(driven)

            frontier = [None for _ in range(self.nset)]
            n_next = 0
            for alpha, states in enumerate(next_candidates):
                if not states:
                    continue
                combined = self._sum_ttns(
                    states, compress=len(original_ttns[0].node_list) > 1, temp_m_trunc=graph_max_dims
                )
                combined = self._prepare_graph_hint_state(combined, graph_max_dims)
                if combined is None:
                    continue
                frontier[alpha] = combined
                n_next += 1
            logger.debug("multiset TTN cross-hint graph round %s produced %s frontier states", iround + 1, n_next)
            if n_next == 0:
                break

        return hints_by_alpha

    def expand_bond_dimension_multiset(
        self, ms_ttno: "MsTTNO", coef: float = 1e-10, use_hint: bool = True, graph_rounds: int = None
    ):
        if graph_rounds is not None:
            logger.warning(
                "graph_rounds is ignored by direct multiset TTNS bond expansion; "
                "expansion now follows MultisetModel.expand_bond_dimension_multiset."
            )

        original_ttns = self.to_ttns_list()
        for ttns in original_ttns:
            ttns.compress_config = self.compress_config.copy()
            ttns.evolve_config = self.evolve_config.copy()

        expanded_components = []
        for alpha in range(self.nset):
            ttns_alpha = original_ttns[alpha]
            ttns_alpha.compress_config = self.compress_config.copy()

            if not use_hint:
                expanded = self._expand_component_bond_dimension(ttns_alpha, hint_ttno=None, coef=coef, ex_ttns=None)
            else:
                diag_ttno = None
                cross_states = []
                for pair_id in ms_ttno.active_pairs_by_alpha[alpha]:
                    pair_alpha, beta = ms_ttno.active_pairs_index[pair_id]
                    assert pair_alpha == alpha
                    pair_ttno = ms_ttno.active_pair_ttnos[pair_id]
                    if beta == alpha:
                        diag_ttno = pair_ttno
                    else:
                        driven = pair_ttno.apply(original_ttns[beta])
                        driven.compress_config = self.compress_config.copy()
                        driven.evolve_config = self.evolve_config.copy()
                        cross_states.append(driven)

                ex_ttns = self._sum_ttns(cross_states, compress=False) if cross_states else None
                if ex_ttns is not None:
                    ex_ttns.compress_config = self.compress_config.copy()
                    ex_ttns.evolve_config = self.evolve_config.copy()

                expanded = self._expand_component_bond_dimension(
                    ttns_alpha,
                    hint_ttno=diag_ttno,
                    coef=coef,
                    ex_ttns=ex_ttns,
                )

            expanded.scale(float(abs(expanded.coeff)), inplace=True)
            expanded.coeff = 1.0
            expanded.compress_config = self.compress_config.copy()
            expanded.evolve_config = self.evolve_config.copy()
            expanded_components.append(expanded)

        new = MsTTNS.from_ttns_list(expanded_components)
        new.compress_config = self.compress_config.copy()
        new.evolve_config = self.evolve_config.copy()
        return new

    def get_node_indices(
        self, node: MsTreeNodeTensor, conj: bool = False, include_parent: bool = False, ttno: "MsTTNO" = None
    ) -> List[Tuple]:
        if include_parent:
            snode_indices = self.get_node_indices(node, conj, ttno=ttno)
            parent_indices = self.get_node_indices(node.parent, conj, ttno=ttno)
            indices = snode_indices + parent_indices
            shared_bond = snode_indices[-1]
            for _ in range(2):
                indices.remove(shared_bond)
            return indices

        _id = str(id(self)) + ("_conj" if conj else "")
        skip_pidx = [] if conj or ttno is None else get_skip_pidx(node, self, ttno)
        all_dofs = self.tn2dofs[node]
        indices = []
        for child in node.children:
            indices.append((_id, str(all_dofs), str(self.tn2dofs[child])))
        for i, dofs in enumerate(all_dofs):
            indices.append(("up" if conj or i in skip_pidx else "down", str(dofs)))
        if node.parent is None:
            indices.append((_id, "root", str(all_dofs)))
        else:
            indices.append((_id, str(self.tn2dofs[node.parent]), str(all_dofs)))
        assert len(indices) == node.tensor.ndim - 1
        return indices

    def to_contract_args(self, iset: int, conj: bool = False):
        args = []
        for node in self.node_list:
            tensor = node.tensor[iset]
            if conj:
                tensor = tensor.conj()
            indices = self.get_node_indices(node, conj)
            indices = [indices[i] for i, s in enumerate(tensor.shape) if s != 1]
            tensor = tensor.squeeze()
            args.extend([tensor, indices])
        return args

    def todense(self, iset: int, order: List[BasisSet] = None):
        args = self.to_contract_args(iset)
        if order is None:
            order = self.basis.basis_list
        output_indices = []
        for basis in order:
            if isinstance(basis, BasisDummy):
                continue
            output_indices.append(("down", str(basis.dofs)))
        args.append(output_indices)
        return asnumpy(oe_contract(*asxp_oe_args(args)))

    def _norm_node_indices(self, node: MsTreeNodeTensor, conj: bool = False):
        if not conj:
            _id = str(id(self)) + "_norm"
        else:
            _id = str(id(self)) + "_norm_conj"
        all_dofs = self.tn2dofs[node]
        indices = []
        for child in node.children:
            indices.append((_id, str(all_dofs), str(self.tn2dofs[child])))
        for dofs in all_dofs:
            indices.append(("norm_phys", str(dofs)))
        if node.parent is None:
            indices.append((_id, "root", str(all_dofs)))
        else:
            indices.append((_id, str(self.tn2dofs[node.parent]), str(all_dofs)))
        assert len(indices) == node.tensor.ndim - 1
        return indices

    def _norm_parent_indices(self, node: MsTreeNodeTensor, conj: bool = False):
        indices = self._norm_node_indices(node, conj=conj)
        return indices[-1]

    def population(self):
        if getattr(self, "_population_cache", None) is not None:
            return self._population_cache.copy()

        batch_idx = ("ms_set_norm", str(id(self)))
        env_children = {node: [] for node in self.node_list}
        root_env = None
        for node in self.postorder_list():
            args = []
            for ichild, child_env in enumerate(env_children[node]):
                child = node.children[ichild]
                args.extend([
                    child_env,
                    [
                        batch_idx,
                        self._norm_parent_indices(child, conj=True),
                        self._norm_parent_indices(child, conj=False),
                    ],
                ])
            args.extend([node.tensor.conj(), [batch_idx] + self._norm_node_indices(node, conj=True)])
            args.extend([node.tensor, [batch_idx] + self._norm_node_indices(node, conj=False)])
            output_indices = [
                batch_idx,
                self._norm_parent_indices(node, conj=True),
                self._norm_parent_indices(node, conj=False),
            ]
            args.append(output_indices)
            env = asnumpy(oe_contract(*asxp_oe_args(args)))
            if node.parent is None:
                root_env = env
            else:
                env_children[node.parent].append(env)

        populations = root_env.reshape(self.nset, -1).sum(axis=1).real
        self._population_cache = np.asarray(populations)
        return self._population_cache.copy()

    def rdm_el(self):
        bra_idx = ("ms_bra_set", str(id(self)))
        ket_idx = ("ms_ket_set", str(id(self)))
        env_children = {node: [] for node in self.node_list}
        root_env = None
        for node in self.postorder_list():
            args = []
            for ichild, child_env in enumerate(env_children[node]):
                child = node.children[ichild]
                args.extend([
                    child_env,
                    [
                        bra_idx,
                        ket_idx,
                        self._norm_parent_indices(child, conj=True),
                        self._norm_parent_indices(child, conj=False),
                    ],
                ])
            args.extend([node.tensor.conj(), [bra_idx] + self._norm_node_indices(node, conj=True)])
            args.extend([node.tensor, [ket_idx] + self._norm_node_indices(node, conj=False)])
            output_indices = [
                bra_idx,
                ket_idx,
                self._norm_parent_indices(node, conj=True),
                self._norm_parent_indices(node, conj=False),
            ]
            args.append(output_indices)
            env = asnumpy(oe_contract(*asxp_oe_args(args)))
            if node.parent is None:
                root_env = env
            else:
                env_children[node.parent].append(env)

        overlaps = root_env.reshape(self.nset, self.nset, -1).sum(axis=2)
        return np.asarray(overlaps.T)

    def calc_electronic_entropy(self, rdm_el=None):
        if rdm_el is None:
            rdm_el = self.rdm_el()
        rdm_el = (rdm_el + rdm_el.conj().T) / 2
        return calc_vn_entropy_dm(rdm_el)

    def calc_bond_entropy_cond_raw_allset(self, populations=None, atol: float = 1e-14):
        if populations is None:
            populations = self.population()

        components = self.to_ttns_list()
        S_cond_all = []
        S_raw_all = []
        for population, component in zip(populations, components):
            population = max(float(np.real(population)), 0.0)
            S_cond = np.zeros(len(self.node_list), dtype=float)
            if population > 0.0 and component.root.children:
                s_array = np.asarray(component.calc_bond_singular_values(), dtype=float)
                for inode, sigma in enumerate(s_array):
                    if self.node_list[inode].parent is None:
                        continue
                    weights = np.square(np.abs(sigma))
                    q = weights / population
                    q = q[q > atol]
                    if q.size == 0:
                        continue
                    entropy = -float(np.sum(q * np.log(q)))
                    S_cond[inode] = 0.0 if abs(entropy) < atol else entropy

            if population == 0.0:
                S_raw = np.zeros_like(S_cond, dtype=float)
            else:
                S_raw = population * S_cond - population * np.log(population)
            S_cond_all.append(S_cond)
            S_raw_all.append(S_raw)
        return np.asarray(S_cond_all, dtype=float), np.asarray(S_raw_all, dtype=float)

    def calc_bond_entropy_allset(self):
        S_cond, _ = self.calc_bond_entropy_cond_raw_allset()
        return S_cond

    def calc_bond_entropy_unnormed_allset(self, S_all_normed=None, populations=None):
        if S_all_normed is None:
            S_all_normed = self.calc_bond_entropy_allset()
        if populations is None:
            populations = self.population()
        S_all_unnormed = []
        for entropy, population in zip(S_all_normed, populations):
            entropy = np.asarray(entropy, dtype=float)
            population = max(float(np.real(population)), 0.0)
            if population == 0.0:
                S_all_unnormed.append(np.zeros_like(entropy, dtype=float))
            else:
                S_all_unnormed.append(population * entropy - population * np.log(population))
        return np.asarray(S_all_unnormed, dtype=float)

    def calc_bond_entropy_summary(self, include_unnormed: bool = False):
        populations = self.population()
        S_all = self.calc_bond_entropy_allset()
        if S_all.size == 0 or S_all.shape[1] == 0:
            S_maxbond_eachset = np.zeros(self.nset, dtype=float)
        else:
            S_maxbond_eachset = np.max(S_all, axis=1)
        S_maxbond = float(np.max(S_maxbond_eachset)) if len(S_maxbond_eachset) > 0 else 0.0
        if not include_unnormed:
            return S_all, S_maxbond_eachset, S_maxbond

        S_all_unnormed = self.calc_bond_entropy_unnormed_allset(S_all, populations)
        if S_all_unnormed.size == 0 or S_all_unnormed.shape[1] == 0:
            S_maxbond_eachset_unnormed = np.zeros(self.nset, dtype=float)
        else:
            S_maxbond_eachset_unnormed = np.max(S_all_unnormed, axis=1)
        S_maxbond_unnormed = (
            float(np.max(S_maxbond_eachset_unnormed)) if len(S_maxbond_eachset_unnormed) > 0 else 0.0
        )
        return S_all, S_maxbond_eachset, S_maxbond, S_all_unnormed, S_maxbond_eachset_unnormed, S_maxbond_unnormed

    def calc_bond_entropy(self):
        S_all = self.calc_bond_entropy_allset()
        if S_all.size == 0 or S_all.shape[1] == 0:
            return np.zeros(len(self.node_list), dtype=float)
        return np.max(S_all, axis=0)

    def ms_normalize(self, kind="ttns_only"):
        if kind != "ttns_only":
            raise ValueError(f"Unsupported normalization kind for MsTTNS: {kind}")
        populations = self.population()
        norm2 = float(np.sum(populations))
        norm = float(np.sqrt(norm2))
        self.root.tensor /= norm
        self._population_cache = populations / norm2
        return self

    def evolve(self, ms_ttno: "MsTTNO", tau: Union[complex, float], normalize: bool = True):
        from renormalizer.tn.multiset_time_evolution import evolve_ms_tdvp_ps

        imag_time = np.iscomplex(tau)
        if imag_time:
            coeff = 1
            tau = tau.imag
            state = self
        else:
            coeff = -1j
            state = self.to_complex()
        new_state = evolve_ms_tdvp_ps(state, ms_ttno, coeff, tau)
        if normalize:
            new_state.ms_normalize("ttns_only")
        return new_state


class MsTTNO(MsTTNBase):
    def __init__(self, basis: BasisTree, ms_models, nset: int = None, algo: str = "Hopcroft-Karp"):
        self.ms_models = ms_models
        if nset is None:
            nset = len(ms_models)
        self.nset = int(nset)
        self.active_pairs_index: List[Tuple[int, int]] = []
        self.active_pair_ttnos: List[TTNO] = []
        self.active_pairs_by_alpha: List[List[int]] = [[] for _ in range(self.nset)]

        for alpha in range(self.nset):
            for beta in range(self.nset):
                terms = self._terms_from_entry(ms_models[alpha][beta])
                if len(terms) == 0:
                    continue
                pair_id = len(self.active_pairs_index)
                self.active_pairs_index.append((alpha, beta))
                self.active_pair_ttnos.append(TTNO(basis, terms, algo=algo))
                self.active_pairs_by_alpha[alpha].append(pair_id)

        if len(self.active_pair_ttnos) == 0:
            raise ValueError("MsTTNO requires at least one active Hamiltonian block.")

        nodes = []
        ref = self.active_pair_ttnos[0]
        for node in ref.node_list:
            nodes.append(MsTreeNodeTensor(np.expand_dims(node.tensor, axis=0), node.qn.copy()))
        root = copy_connection(ref.node_list, nodes)
        super().__init__(basis, root)

        self.pair_node_tensors = [
            [onode.tensor for onode in ttno.node_list] for ttno in self.active_pair_ttnos
        ]
        self.node_groups = self._build_node_groups()
        self.hop_expr_cache = {}

    @staticmethod
    def _terms_from_entry(entry):
        if isinstance(entry, Model):
            return entry.ham_terms
        if isinstance(entry, Op):
            return [entry]
        if entry is None:
            return []
        return list(entry)

    @classmethod
    def from_multiset_model(cls, basis: BasisTree, ms_model) -> "MsTTNO":
        return cls(basis, ms_model.MsModel, nset=ms_model.N_electron)

    def _build_pair_to_alpha_matrix(self, alpha_idx, dtype):
        scatter = xp.zeros((self.nset, len(alpha_idx)), dtype=dtype)
        for i, alpha in enumerate(alpha_idx):
            scatter[alpha, i] = 1.0
        return scatter

    def _build_node_groups(self):
        all_groups = []
        for inode in range(len(self.active_pair_ttnos[0])):
            groups = {}
            for pair_id, (alpha, beta) in enumerate(self.active_pairs_index):
                W = self.pair_node_tensors[pair_id][inode]
                group = groups.setdefault(
                    tuple(W.shape),
                    {"pair_ids": [], "alpha_idx": [], "beta_idx": [], "W": []},
                )
                group["pair_ids"].append(pair_id)
                group["alpha_idx"].append(alpha)
                group["beta_idx"].append(beta)
                group["W"].append(W)
            node_groups = []
            for key in sorted(groups):
                group = groups[key]
                node_groups.append(
                    {
                        "pair_ids": tuple(group["pair_ids"]),
                        "alpha_idx": xp.asarray(group["alpha_idx"], dtype=np.int64),
                        "beta_idx": xp.asarray(group["beta_idx"], dtype=np.int64),
                        "S": self._build_pair_to_alpha_matrix(group["alpha_idx"], group["W"][0].real.dtype),
                        "W": xp.stack([xp.asarray(w) for w in group["W"]]),
                        "n_pairs": len(group["pair_ids"]),
                    }
                )
            all_groups.append(node_groups)
        return all_groups

    def get_node_indices(self, node: MsTreeNodeTensor, prefix_up="up", prefix_down="down") -> List:
        idx = self.node_idx[node]
        ref_node = self.active_pair_ttnos[0].node_list[idx]
        return self.active_pair_ttnos[0].get_node_indices(ref_node, prefix_up, prefix_down)

    def get_pair_node_indices(self, pair_id: int, inode: int, prefix_up="up", prefix_down="down") -> List:
        onode = self.active_pair_ttnos[pair_id].node_list[inode]
        return self.active_pair_ttnos[pair_id].get_node_indices(onode, prefix_up, prefix_down)


class MsTTNEnviron:
    """
    Active-pair environment storage for multiset TTN.

    Environments are stored only for nonzero Hamiltonian blocks listed in
    ``MsTTNO.active_pairs_index``.  They are not expanded to dense
    ``[nset, nset]`` blocks.
    """

    def __init__(self, ms_ttns: MsTTNS, ms_ttno: MsTTNO, build_environ=True):
        self.basis_ttns = ms_ttns.basis
        self.basis_ttno = ms_ttno.basis
        self.n_pairs = len(ms_ttno.active_pairs_index)
        self.tn2dofs_ttns = {tn: bn.dofs for tn, bn in zip(ms_ttns.node_list, self.basis_ttns.node_list)}
        self.tn2dofs_ttno = {
            tn: bn.dofs for tn, bn in zip(ms_ttns.node_list, self.basis_ttno.node_list)
        }
        self.parent_envs = [
            [None for _ in range(ms_ttns.size)] for _ in range(self.n_pairs)
        ]
        self.children_envs = [
            [[None for _ in node.children] for node in ms_ttns.node_list] for _ in range(self.n_pairs)
        ]
        root_idx = ms_ttns.node_idx[ms_ttns.root]
        for pair_id in range(self.n_pairs):
            op_root_dim = ms_ttno.pair_node_tensors[pair_id][root_idx].shape[-1]
            self.parent_envs[pair_id][root_idx] = np.ones((1, op_root_dim, 1), dtype=backend.real_dtype)
        if build_environ:
            self.build_children_environ(ms_ttns, ms_ttno)
            self.build_parent_environ(ms_ttns, ms_ttno)

    def build_children_environ(self, ms_ttns: MsTTNS, ms_ttno: MsTTNO):
        for snode in ms_ttns.postorder_list():
            self.build_children_environ_node(snode, ms_ttns, ms_ttno)

    def build_parent_environ(self, ms_ttns: MsTTNS, ms_ttno: MsTTNO):
        for snode in ms_ttns.node_list:
            for ichild in range(len(snode.children)):
                self.build_parent_environ_node(snode, ichild, ms_ttns, ms_ttno)

    def _batch_axis(self, inode: int, igroup: int, direction: str):
        return ("ms_env_batch", inode, igroup, direction)

    def _op_batch_id(self, ms_ttno: MsTTNO):
        return str(id(ms_ttno)) + "_env_batched"

    def _batched_child_indices(self, snode: MsTreeNodeTensor, i: int, ms_ttns: MsTTNS, ms_ttno: MsTTNO, batch):
        child = snode.children[i]
        dofs_ttns = self.tn2dofs_ttns[snode]
        dofs_child_ttns = self.tn2dofs_ttns[child]
        dofs_ttno = self.tn2dofs_ttno[snode]
        dofs_child_ttno = self.tn2dofs_ttno[child]
        return [
            batch,
            (str(id(ms_ttns)) + "_conj", str(dofs_ttns), str(dofs_child_ttns)),
            (self._op_batch_id(ms_ttno), str(dofs_ttno), str(dofs_child_ttno)),
            (str(id(ms_ttns)), str(dofs_ttns), str(dofs_child_ttns)),
        ]

    def _batched_parent_indices(self, snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, batch):
        dofs_ttns = self.tn2dofs_ttns[snode]
        dofs_ttno = self.tn2dofs_ttno[snode]
        if snode.parent is not None:
            dofs_parent_ttns = self.tn2dofs_ttns[snode.parent]
            dofs_parent_ttno = self.tn2dofs_ttno[snode.parent]
        else:
            dofs_parent_ttns = dofs_parent_ttno = "root"
        return [
            batch,
            (str(id(ms_ttns)) + "_conj", str(dofs_parent_ttns), str(dofs_ttns)),
            (self._op_batch_id(ms_ttno), str(dofs_parent_ttno), str(dofs_ttno)),
            (str(id(ms_ttns)), str(dofs_parent_ttns), str(dofs_ttns)),
        ]

    def _batched_op_indices(self, snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, batch):
        inode = ms_ttns.node_idx[snode]
        onode = ms_ttno.node_list[inode]
        all_dofs = ms_ttno.tn2dofs[onode]
        indices = [batch]
        for child in snode.children:
            ochild = ms_ttno.node_list[ms_ttns.node_idx[child]]
            indices.append((self._op_batch_id(ms_ttno), str(all_dofs), str(ms_ttno.tn2dofs[ochild])))
        for dofs in all_dofs:
            indices.append(("up", str(dofs)))
            indices.append(("down", str(dofs)))
        if snode.parent is None:
            indices.append((self._op_batch_id(ms_ttno), "root", str(all_dofs)))
        else:
            oparent = ms_ttno.node_list[ms_ttns.node_idx[snode.parent]]
            indices.append((self._op_batch_id(ms_ttno), str(ms_ttno.tn2dofs[oparent]), str(all_dofs)))
        return indices

    @staticmethod
    def _group_indices(group):
        return [int(i) for i in asnumpy(group["alpha_idx"])], [int(i) for i in asnumpy(group["beta_idx"])]

    @staticmethod
    def _stack_pair_tensors(pair_ids, getter):
        return xp.stack([xp.asarray(getter(pair_id)) for pair_id in pair_ids])

    def build_children_environ_node(self, snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO):
        if snode.parent is None:
            return
        inode = ms_ttns.node_idx[snode]
        parent_idx = ms_ttns.node_idx[snode.parent]
        ichild = snode.idx_as_child
        for igroup, group in enumerate(ms_ttno.node_groups[inode]):
            pair_ids = group["pair_ids"]
            alpha_idx, beta_idx = self._group_indices(group)
            batch = self._batch_axis(inode, igroup, "children")
            args = []
            for i in range(len(snode.children)):
                child_env = self._stack_pair_tensors(pair_ids, lambda pair_id, i=i: self.children_envs[pair_id][inode][i])
                args.extend([child_env, self._batched_child_indices(snode, i, ms_ttns, ms_ttno, batch)])
            args.extend([xp.asarray(snode.tensor[alpha_idx]).conj(), [batch] + ms_ttns.get_node_indices(snode, conj=True)])
            args.extend([group["W"], self._batched_op_indices(snode, ms_ttns, ms_ttno, batch)])
            args.extend([xp.asarray(snode.tensor[beta_idx]), [batch] + ms_ttns.get_node_indices(snode, ttno=ms_ttno)])
            args.append(self._batched_parent_indices(snode, ms_ttns, ms_ttno, batch))
            res = asnumpy(oe_contract(*asxp_oe_args(args)))
            for ibatch, pair_id in enumerate(pair_ids):
                self.children_envs[pair_id][parent_idx][ichild] = res[ibatch]

    def build_parent_environ_node(self, snode: MsTreeNodeTensor, ichild: int, ms_ttns: MsTTNS, ms_ttno: MsTTNO):
        inode = ms_ttns.node_idx[snode]
        child = snode.children[ichild]
        child_idx = ms_ttns.node_idx[child]
        for igroup, group in enumerate(ms_ttno.node_groups[inode]):
            pair_ids = group["pair_ids"]
            alpha_idx, beta_idx = self._group_indices(group)
            batch = self._batch_axis(inode, igroup, "parent")
            args = []
            for j in range(len(snode.children)):
                if j == ichild:
                    continue
                child_env = self._stack_pair_tensors(pair_ids, lambda pair_id, j=j: self.children_envs[pair_id][inode][j])
                args.extend([child_env, self._batched_child_indices(snode, j, ms_ttns, ms_ttno, batch)])
            parent_env = self._stack_pair_tensors(pair_ids, lambda pair_id: self.parent_envs[pair_id][inode])
            args.extend([parent_env, self._batched_parent_indices(snode, ms_ttns, ms_ttno, batch)])
            args.extend([xp.asarray(snode.tensor[alpha_idx]).conj(), [batch] + ms_ttns.get_node_indices(snode, conj=True)])
            args.extend([group["W"], self._batched_op_indices(snode, ms_ttns, ms_ttno, batch)])
            args.extend([xp.asarray(snode.tensor[beta_idx]), [batch] + ms_ttns.get_node_indices(snode, ttno=ms_ttno)])
            args.append(self._batched_child_indices(snode, ichild, ms_ttns, ms_ttno, batch))
            res = asnumpy(oe_contract(*asxp_oe_args(args)))
            for ibatch, pair_id in enumerate(pair_ids):
                self.parent_envs[pair_id][child_idx] = res[ibatch]

    def get_child_indices(self, snode: MsTreeNodeTensor, i: int, ms_ttns: MsTTNS, ms_ttno: MsTTNO, pair_id: int):
        child = snode.children[i]
        dofs_ttns = self.tn2dofs_ttns[snode]
        dofs_child_ttns = self.tn2dofs_ttns[child]
        dofs_ttno = self.tn2dofs_ttno[snode]
        dofs_child_ttno = self.tn2dofs_ttno[child]
        return [
            (str(id(ms_ttns)) + "_conj", str(dofs_ttns), str(dofs_child_ttns)),
            (str(id(ms_ttno.active_pair_ttnos[pair_id])), str(dofs_ttno), str(dofs_child_ttno)),
            (str(id(ms_ttns)), str(dofs_ttns), str(dofs_child_ttns)),
        ]

    def get_parent_indices(self, snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, pair_id: int):
        dofs_ttns = self.tn2dofs_ttns[snode]
        dofs_ttno = self.tn2dofs_ttno[snode]
        if snode.parent is not None:
            dofs_parent_ttns = self.tn2dofs_ttns[snode.parent]
            dofs_parent_ttno = self.tn2dofs_ttno[snode.parent]
        else:
            dofs_parent_ttns = dofs_parent_ttno = "root"
        return [
            (str(id(ms_ttns)) + "_conj", str(dofs_parent_ttns), str(dofs_ttns)),
            (str(id(ms_ttno.active_pair_ttnos[pair_id])), str(dofs_parent_ttno), str(dofs_ttno)),
            (str(id(ms_ttns)), str(dofs_parent_ttns), str(dofs_ttns)),
        ]

