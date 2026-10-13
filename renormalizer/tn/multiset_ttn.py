from __future__ import annotations

import logging
import os
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
    """A multiset wavefunction stored as independent :class:`TTNS` objects.

    The old implementation put the electronic-set index in front of every
    node tensor.  That layout was convenient for batched contractions, but it
    made component-wise expansion require two complete copies of the state.
    ``MsTTNS`` now owns a list of ordinary TTNS objects and only packs the
    tensor of the node currently being processed.  This mirrors
    ``MultisetMps`` and keeps the public multiset API unchanged.

    All components coupled in one TDVP step currently need compatible node
    shapes and quantum-number sectors.  ``layout_groups`` exposes the groups
    explicitly so a future adaptive-rank implementation can evolve compatible
    groups separately without changing the storage model again.
    """

    def __init__(
        self,
        basis: BasisTree,
        nset: int = None,
        condition: Dict = None,
        root: MsTreeNodeTensor = None,
        ttns_list: Sequence[TTNS] = None,
        _copy_components: bool = True,
    ):
        if ttns_list is not None:
            if root is not None or nset is not None:
                raise ValueError("`ttns_list` can not be combined with `root` or `nset`.")
            components = [ttns.copy() for ttns in ttns_list] if _copy_components else list(ttns_list)
        elif root is None:
            if nset is None:
                raise ValueError("`nset` is required when no root or ttns_list is provided.")
            ref = TTNS(basis, condition=condition)
            components = [ref] + [ref.copy() for _ in range(int(nset) - 1)]
        else:
            if nset is None:
                nset = root.tensor.shape[0]
            components = self._components_from_stacked_root(basis, root, int(nset))

        if not components:
            raise ValueError("MsTTNS requires at least one TTNS component.")
        self.ttns_list = components
        # Alias consistent with MultisetMps and useful for downstream code.
        self.msttns = self.ttns_list
        self.nset = len(components)
        super().__init__(basis, components[0].root)
        self.coeff = 1
        self.compress_config = CompressConfig(CompressCriteria.fixed)
        self.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, force_ovlp=False)
        self._population_cache = None
        self.check_shape()

    @staticmethod
    def _components_from_stacked_root(basis: BasisTree, root: MsTreeNodeTensor, nset: int) -> List[TTNS]:
        """Compatibility reader for the former leading-set-axis layout."""
        stacked_nodes = Tree(root).node_list
        components = []
        for iset in range(nset):
            component = TTNS(basis)
            for dst, src in zip(component.node_list, stacked_nodes):
                dst.tensor = src.tensor[iset].copy()
                dst.qn = src.qn.copy()
            components.append(component)
        return components

    @classmethod
    def from_ttns_list(cls, ttns_list: Sequence[TTNS]) -> "MsTTNS":
        return cls(ttns_list[0].basis, ttns_list=ttns_list)

    @classmethod
    def _from_owned_components(cls, ttns_list: Sequence[TTNS]) -> "MsTTNS":
        """Build from freshly-created components without another deep copy."""
        return cls(ttns_list[0].basis, ttns_list=ttns_list, _copy_components=False)

    def to_ttns_list(self, copy: bool = True) -> List[TTNS]:
        if copy:
            return [component.copy() for component in self.ttns_list]
        return list(self.ttns_list)

    def print_shape(
        self,
        full: bool = False,
        print_function=None,
        component_indices: Sequence[int] = None,
        component_labels: Sequence = None,
    ):
        """Print the tensor-tree shape of every selected TTNS component.

        ``MsTTNS`` stores electronic components as independent ``TTNS``
        objects, so delegating to only ``self.root`` would silently describe
        component zero and hide a possible layout mismatch.  This method uses
        :meth:`TTNS.print_shape` for each selected component and labels every
        tree explicitly.

        Parameters
        ----------
        full
            Print each complete node-tensor shape.  If false, print only the
            parent-bond dimension, matching :meth:`TTNS.print_shape`.
        print_function
            Callback for one output line.  Defaults to :func:`print`.
        component_indices
            Local component indices to print.  Defaults to all components.
        component_labels
            Optional display labels corresponding to ``component_indices``.
            This is useful when a distributed caller stores only a shard but
            wants to show global electronic-state indices.
        """
        if print_function is None:
            print_function = print
        if component_indices is None:
            component_indices = list(range(self.nset))
        else:
            component_indices = [int(index) for index in component_indices]
        for index in component_indices:
            if index < 0 or index >= self.nset:
                raise IndexError(
                    f"component index {index} outside MsTTNS with {self.nset} components"
                )

        if component_labels is None:
            component_labels = component_indices
        else:
            component_labels = list(component_labels)
            if len(component_labels) != len(component_indices):
                raise ValueError(
                    "component_labels must have the same length as component_indices"
                )

        for index, label in zip(component_indices, component_labels):
            print_function(f"MsTTNS component {label} (local index {index})")
            self.ttns_list[index].print_shape(
                full=full, print_function=print_function
            )

    def _inode(self, node_or_inode) -> int:
        if isinstance(node_or_inode, int):
            return node_or_inode
        return self.node_idx[node_or_inode]

    def component_node(self, iset: int, node_or_inode) -> TreeNodeTensor:
        return self.ttns_list[iset].node_list[self._inode(node_or_inode)]

    def node_tensor_batch(self, node_or_inode, indices=None, device: bool = False):
        """Pack one node across compatible TTNS components.

        Packing is intentionally node-local: no second copy of the full tree is
        ever formed.  ``indices`` is used by active-pair contractions to avoid
        packing components that are not needed by the current Hamiltonian
        group.
        """
        inode = self._inode(node_or_inode)
        if indices is None:
            indices = range(self.nset)
        tensors = [self.ttns_list[int(iset)].node_list[inode].tensor for iset in indices]
        stack_module = xp if device else np
        return stack_module.stack([stack_module.asarray(tensor) for tensor in tensors])

    def set_node_tensor_batch(self, node_or_inode, tensors, indices=None):
        inode = self._inode(node_or_inode)
        if indices is None:
            indices = range(self.nset)
        tensors = asnumpy(tensors)
        for ibatch, iset in enumerate(indices):
            self.ttns_list[int(iset)].node_list[inode].tensor = tensors[ibatch]
        self._population_cache = None

    @property
    def layout_groups(self):
        """Component indices grouped by complete tree tensor layout."""
        groups = {}
        for iset, component in enumerate(self.ttns_list):
            signature = tuple((node.shape, node.qn.shape, node.qn.tobytes()) for node in component.node_list)
            groups.setdefault(signature, []).append(iset)
        return tuple(tuple(indices) for indices in groups.values())

    def _require_common_layout(self):
        groups = self.layout_groups
        if len(groups) != 1:
            raise ValueError(
                "Coupled multiset TDVP currently requires one compatible TTNS layout group; "
                f"found component groups with sizes {[len(group) for group in groups]}."
            )
        return groups[0]

    def metacopy(self):
        components = []
        for component in self.ttns_list:
            nodes = [
                TreeNodeTensor(np.zeros_like(node.tensor), node.qn.copy())
                for node in component.node_list
            ]
            root = copy_connection(component.node_list, nodes)
            new_component = TTNS(self.basis, root=root)
            new_component.coeff = component.coeff
            new_component.compress_config = component.compress_config.copy()
            new_component.evolve_config = component.evolve_config.copy()
            components.append(new_component)
        new = self.__class__._from_owned_components(components)
        new.coeff = self.coeff
        new.compress_config = self.compress_config.copy()
        new.evolve_config = self.evolve_config.copy()
        return new

    def copy(self):
        new = self.__class__._from_owned_components([component.copy() for component in self.ttns_list])
        new.coeff = self.coeff
        new.compress_config = self.compress_config.copy()
        new.evolve_config = self.evolve_config.copy()
        new._population_cache = None if self._population_cache is None else self._population_cache.copy()
        return new

    def to_complex(self, inplace: bool = False):
        new = self if inplace else self.copy()
        for component in new.ttns_list:
            component.to_complex(inplace=True)
        return new

    @property
    def qntot(self):
        return self.root.qn[0]

    @property
    def bond_dims(self):
        return np.max(np.asarray(self.component_bond_dims, dtype=int), axis=0).tolist()

    @property
    def component_bond_dims(self):
        return [component.bond_dims for component in self.ttns_list]

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
        for component in self.ttns_list:
            component.check_shape()
            if len(component.node_list) != len(self.basis.node_list):
                raise ValueError("TTNS component has an incompatible basis-tree topology.")
        return True

    def check_canonical(self, atol=None):
        for component in self.ttns_list:
            component.check_canonical(atol)
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
        self._require_common_layout()
        inode = self._inode(node)
        qnbigl, qnbigr, _ = self.get_qnmat(node, include_parent=False)
        tensor_batch = self.node_tensor_batch(inode)
        tensor = tensor_batch.reshape(self.nset, -1, node.shape[-1])
        u, qnlnew, v, qnrnew = self._batched_qr_qn(tensor, qnbigl, qnbigr, system="L")
        new_shape = tuple(node.shape[:-1]) + (u.shape[-1],)
        self.set_node_tensor_batch(inode, u.reshape((self.nset,) + new_shape))
        for component in self.ttns_list:
            component.node_list[inode].qn = qnlnew
        return v

    def merge_to_parent(self, node: MsTreeNodeTensor, v):
        parent = node.parent
        parent_inode = self._inode(parent)
        # axis 0 is the temporary component batch.
        axis = 1 + node.idx_as_child
        parent_batch = np.moveaxis(self.node_tensor_batch(parent_inode), axis, -1)
        merged = np.einsum("n...r,nrk->n...k", parent_batch, v)
        self.set_node_tensor_batch(parent_inode, np.moveaxis(merged, -1, axis))

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
        tensor = np.moveaxis(self.node_tensor_batch(node), 1 + ichild, -1)
        shape = list(tensor.shape)
        tensor = tensor.reshape(self.nset, -1, node.shape[ichild])
        return qnbigl, qnbigr, tensor, shape

    def decompose_to_child(self, node: MsTreeNodeTensor, ichild: int):
        self._require_common_layout()
        qnbigl, qnbigr, tensor, shape = self._moveaxis_for_child(node, ichild)
        u, qnl, v, qnr = self._batched_qr_qn(tensor, qnbigl, qnbigr, system="L")
        shape[-1] = u.shape[-1]
        self.set_node_tensor_batch(node, np.moveaxis(u.reshape(shape), -1, 1 + ichild))
        child_inode = self._inode(node.children[ichild])
        for component in self.ttns_list:
            component.node_list[child_inode].qn = qnr
        return v

    def merge_to_child(self, node: MsTreeNodeTensor, ichild: int, v):
        child = node.children[ichild]
        child_batch = self.node_tensor_batch(child)
        self.set_node_tensor_batch(child, np.einsum("n...r,nrk->n...k", child_batch, v))

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
        """Sum TTNS terms without constructing a high-arity direct sum.

        An MPS can normally accumulate a small batch before compression because
        every site has only two virtual legs.  At a ternary TTNS node, however,
        adding ``k`` terms enlarges all four virtual legs before canonicalisation
        and the temporary tensor scales as ``k**4``.  Compress after every
        binary addition so the peak contains at most one compressed accumulator
        and one new term.  Since fixed-rank truncation is nonlinear, callers
        that need an unbiased final subspace should give this accumulator a
        modest working-rank margin and compress to the requested rank only
        after all terms have been accumulated.
        """
        if len(ttns_list) == 0:
            return None
        sum_mode = os.environ.get(
            "RENO_MS_TTN_SUM_MODE", "sequential"
        ).strip().lower()
        if sum_mode not in {"sequential", "batch"}:
            raise ValueError(
                "RENO_MS_TTN_SUM_MODE must be 'sequential' or 'batch', "
                f"got {sum_mode!r}"
            )
        res = ttns_list[0].copy()
        for ttns in ttns_list[1:]:
            res = res + ttns
            if compress and sum_mode == "sequential":
                res = res.canonicalise().compress(temp_m_trunc)
        if compress and (sum_mode == "batch" or len(ttns_list) == 1):
            res.canonicalise().compress(temp_m_trunc)
        return res

    @staticmethod
    def _krylov_bond_limits(m_target, scalar_cap=None):
        """Return strictly positive, per-edge limits for a Krylov TTNS.

        ``m_target`` is deliberately nonuniform for published Tree ansatzes
        (for example 15 on terminal ternary branches and 127 on central bonds
        at chi=128).  Replacing it with ``max(m_target)`` is harmless for an
        MPS only when every bond has the same limit, but can create a
        prohibitively large high-arity TTNS tensor.  The root entry is unused
        by ``TTNS.compress``; setting it, and any saturated edge, to one keeps
        the profile a valid TTNS rank vector.
        """
        limits = np.maximum(np.asarray(m_target, dtype=int), 1)
        if scalar_cap is not None:
            limits = np.minimum(limits, max(int(scalar_cap), 1))
        return limits

    @staticmethod
    def _krylov_working_limits(m_target, bond_dims_exact):
        """Return a memory-safe oversampled rank profile for Krylov buildup.

        The published P3HT ``Tree`` has terminal ternary branches whose final
        virtual dimension is only 16 while central bonds can be 128.  Clipping
        *every intermediate Krylov vector* to the terminal final rank makes the
        sequential sum strongly order dependent and can remove the directions
        that carry population to OT11--13.  Conversely, using the scalar
        maximum (the historical MPS-style implementation) lets a ternary node
        grow to 128 on all legs and creates a tens-of-GiB contraction.

        Oversample only small requested ranks.  Large/central ranks retain
        their requested limit, so this restores a useful pre-truncation space
        on the terminal branches without recreating the chi**4 OOM.  The
        expander is compressed back to ``m_target`` before it is attached to
        the physical state, hence the final TTNS layout is unchanged.
        """
        raw_target = np.asarray(m_target, dtype=int)
        target = np.maximum(raw_target, 1)
        # ``bond_dims_exact`` is intentionally an object array: for a large
        # subtree its combinatorial Hilbert-space dimension can exceed int64.
        # Working ranks are small, so cap them element by element without ever
        # narrowing the exact Python integers to a C long.
        exact = np.asarray(bond_dims_exact, dtype=object)
        factor = max(
            1.0,
            float(os.environ.get("RENO_MS_TTN_KRYLOV_OVERSAMPLE", "1")),
        )
        small_cap = max(
            1,
            int(os.environ.get("RENO_MS_TTN_KRYLOV_OVERSAMPLE_CAP", "32")),
        )
        if factor == 1.0:
            return np.minimum(target, exact)
        oversampled = np.ceil(target * factor).astype(int)
        working = np.where(
            (raw_target > 0) & (target < small_cap),
            np.minimum(oversampled, small_cap),
            target,
        )
        return np.asarray(
            [
                min(int(work), max(int(exact_dim), 1))
                for work, exact_dim in zip(
                    np.maximum(working, target), exact
                )
            ],
            dtype=int,
        )

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
        krylov_target = MsTTNS._krylov_bond_limits(m_target)
        krylov_working_target = MsTTNS._krylov_working_limits(
            m_target, ttns.bond_dims_exact
        )

        if hint_ttno is None:
            expander = TTNS.random(ttns.basis, ttns.qntot, m_target)
        else:
            logger.debug(f"bond dimension of multiset TTN hint TTNO: {hint_ttno.bond_dims}")
            lastone = ttns if ex_ttns is None else ttns + ex_ttns
            if lastone.norm == 0:
                # A zero electronic component has no physical Krylov hint.  It
                # can be created directly at the requested final layout; the
                # former zero + random(m_target) + compress route needlessly
                # built and decomposed a larger intermediate tree.
                final_dims = np.minimum(
                    np.asarray(ttns.compress_config.max_dims),
                    ttns.bond_dims_exact,
                ).astype(int)
                expanded = TTNS.random(ttns.basis, ttns.qntot, final_dims)
                return expanded.scale(coef, inplace=True)
            else:
                # Match the successful multiset-MPS enrichment: accumulate
                # H_diag^n |psi> until the requested expansion rank is filled.
                # A single application followed by an unrelated random
                # complement is not sufficient for a one-site TDVP manifold,
                # especially for a 100+ mode nuclear wavefunction.
                expander_list = []
                expander_dims = np.zeros_like(m_target)
                while True:
                    nextone = hint_ttno.apply(lastone)
                    if nextone.norm == 0:
                        logger.warning(
                            "TTN hint application produced a zero expander; "
                            "falling back to random expansion"
                        )
                        expander = TTNS.random(ttns.basis, ttns.qntot, m_target)
                        break

                    lastone = nextone.normalize("ttns_and_coeff")
                    # Keep every edge below a per-edge working target.  Small
                    # terminal Tree bonds may be modestly oversampled so the
                    # sequential sum does not discard late Krylov directions;
                    # central bonds never exceed their requested final rank.
                    lastone = lastone.canonicalise().compress(
                        krylov_working_target
                    )
                    expander_list.append(lastone)
                    expander = MsTTNS._sum_ttns(
                        expander_list,
                        compress=True,
                        temp_m_trunc=krylov_working_target,
                    )

                    current_dims = np.asarray(expander.bond_dims)
                    if np.all(current_dims >= m_target):
                        break
                    if np.array_equal(current_dims, expander_dims):
                        logger.warning(
                            "TTN Krylov expander no longer increases; "
                            "adding the final MS-MPS-style Krylov hint"
                        )
                        missing_limits = MsTTNS._krylov_bond_limits(
                            m_target - expander_dims
                        )
                        expander2 = hint_ttno.apply(lastone)
                        if expander2.norm != 0:
                            expander2 = expander2.canonicalise().compress(
                                missing_limits
                            )
                            expander = MsTTNS._sum_ttns(
                                [expander, expander2],
                                compress=True,
                                temp_m_trunc=krylov_working_target,
                            )
                        break

                    expander_dims = current_dims
                    temp_m_trunc = int(
                        np.max(m_target) / max(np.max(hint_ttno.bond_dims), 1)
                    ) + 1
                    propagation_limits = MsTTNS._krylov_bond_limits(
                        m_target, scalar_cap=temp_m_trunc
                    )
                    lastone = lastone.canonicalise().compress(
                        propagation_limits
                    )

                # Select the final requested basis from the oversampled union
                # only once, after all useful Krylov directions are present.
                if not np.array_equal(
                    np.asarray(expander.bond_dims, dtype=int), krylov_target
                ):
                    expander = expander.canonicalise().compress(krylov_target)

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

    @staticmethod
    def _pad_component_bond_dimension(ttns: TTNS, target_dims, coef: float = 1e-10):
        """Fill rank-deficient bonds so independently stored components align.

        Fixed-rank compression is an upper bound: a physically sparse hint can
        still return fewer singular vectors than requested.  Node-local batch
        operations require identical tensor layouts, so add a full-layout
        random complement and compress back to the requested per-node profile.
        """
        target_dims = np.asarray(target_dims, dtype=int)
        current_dims = np.asarray(ttns.bond_dims, dtype=int)
        if np.array_equal(current_dims, target_dims):
            return ttns
        if np.any(current_dims > target_dims):
            ttns = ttns.canonicalise().compress(target_dims)
            current_dims = np.asarray(ttns.bond_dims, dtype=int)
        # A standalone tree with rank-one leaf bonds cannot supply a large
        # missing internal rank: its parent ranks are bounded by its children.
        # Use a full target-layout complement, then truncate the direct sum.
        padding = TTNS.random(ttns.basis, ttns.qntot, target_dims)
        ref_norm = ttns.norm if ttns.norm != 0 else 1.0
        padded = ttns + padding.scale(coef * ref_norm, inplace=True)
        padded = padded.canonicalise().compress(target_dims)
        if not np.array_equal(np.asarray(padded.bond_dims, dtype=int), target_dims):
            raise RuntimeError(
                "Could not pad independent TTNS component to common target layout: "
                f"current={current_dims.tolist()} target={target_dims.tolist()} "
                f"result={padded.bond_dims}"
            )
        return padded

    def expand_bond_dimension_multiset(
        self, ms_ttno: "MsTTNO", coef: float = 1e-10, use_hint: bool = True
    ):
        """Expand each electronic TTNS with the multiset-MPS recipe.

        For row ``alpha``, form the one-hop physical seed
        ``psi_alpha + sum_{beta != alpha} H[alpha,beta] psi_beta`` and then
        repeatedly apply the diagonal block ``H[alpha,alpha]``.  Localized
        initial conditions should seed every inactive electronic component
        with a tiny copy of the same vibrational state, as multiset-MPS does,
        rather than with an exact zero.
        """
        # Components are already independent TTNS objects.  Keep read-only
        # references here: _expand_component_bond_dimension copies the target
        # component before modifying it, while TTNO.apply is non-mutating.
        original_ttns = self.to_ttns_list(copy=False)
        self.compress_config.set_bonddim(len(original_ttns[0].bond_dims))
        for ttns in original_ttns:
            ttns.compress_config = self.compress_config.copy()
            ttns.evolve_config = self.evolve_config.copy()

        expanded_components = []
        zero_component_template = None
        for alpha in range(self.nset):
            ttns_alpha = original_ttns[alpha]
            ttns_alpha.compress_config = self.compress_config.copy()
            ttns_alpha.compress_config.set_bonddim(len(ttns_alpha.bond_dims))

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
                        cross_states.append(pair_ttno.apply(original_ttns[beta]))

                ex_ttns = self._sum_ttns(cross_states, compress=False) if cross_states else None
                if ex_ttns is not None:
                    ex_ttns.compress_config = self.compress_config.copy()
                    ex_ttns.evolve_config = self.evolve_config.copy()
                    if ex_ttns.norm == 0:
                        ex_ttns = None

                if ex_ttns is None and ttns_alpha.norm == 0:
                    # Most electronic rows of a localized Holstein initial
                    # state are identical zeros.  Generate the expensive
                    # binary-tree random basis once, then deep-copy it so the
                    # all TTNS components remain independently owned.
                    if zero_component_template is None:
                        final_dims = np.minimum(
                            np.asarray(ttns_alpha.compress_config.max_dims),
                            ttns_alpha.bond_dims_exact,
                        ).astype(int)
                        zero_component_template = TTNS.random(
                            ttns_alpha.basis,
                            ttns_alpha.qntot,
                            final_dims,
                        )
                    expanded = zero_component_template.copy().scale(coef, inplace=True)
                else:
                    expanded = self._expand_component_bond_dimension(
                        ttns_alpha,
                        hint_ttno=diag_ttno,
                        coef=coef,
                        ex_ttns=ex_ttns,
                    )

            expanded.scale(float(abs(expanded.coeff)), inplace=True)
            expanded.coeff = 1.0
            final_dims = np.minimum(
                np.asarray(self.compress_config.max_dims),
                expanded.bond_dims_exact,
            ).astype(int)
            if not np.array_equal(np.asarray(expanded.bond_dims), final_dims):
                logger.info(
                    "padding multiset TTNS component %d from max/mean %.0f/%.2f "
                    "to common max/mean %.0f/%.2f",
                    alpha,
                    np.max(expanded.bond_dims),
                    np.mean(expanded.bond_dims),
                    np.max(final_dims),
                    np.mean(final_dims),
                )
                expanded = self._pad_component_bond_dimension(expanded, final_dims, coef=coef)
            expanded.compress_config = self.compress_config.copy()
            expanded.evolve_config = self.evolve_config.copy()
            expanded_components.append(expanded)

        # Ownership transfer avoids the former list -> fully stacked state
        # copy, which doubled peak host memory at the end of expansion.
        new = MsTTNS._from_owned_components(expanded_components)
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
        assert len(indices) == node.tensor.ndim
        return indices

    def to_contract_args(self, iset: int, conj: bool = False):
        if not 0 <= iset < self.nset:
            raise IndexError(iset)
        args = []
        for node in self.ttns_list[iset].node_list:
            tensor = node.tensor
            if conj:
                tensor = tensor.conj()
            indices = self.get_node_indices(self.node_list[self.ttns_list[iset].node_idx[node]], conj)
            indices = [indices[i] for i, s in enumerate(tensor.shape) if s != 1]
            tensor = tensor.squeeze()
            args.extend([tensor, indices])
        return args

    def todense(self, iset: int, order: List[BasisSet] = None):
        # TTNS.todense includes BasisDummy entries in its default output list;
        # retain the multiset behaviour of exposing physical modes only.
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
        assert len(indices) == node.tensor.ndim
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
            node_batch = self.node_tensor_batch(node)
            args.extend([node_batch.conj(), [batch_idx] + self._norm_node_indices(node, conj=True)])
            args.extend([node_batch, [batch_idx] + self._norm_node_indices(node, conj=False)])
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
            node_batch = self.node_tensor_batch(node)
            args.extend([node_batch.conj(), [bra_idx] + self._norm_node_indices(node, conj=True)])
            args.extend([node_batch, [ket_idx] + self._norm_node_indices(node, conj=False)])
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

        components = self.to_ttns_list(copy=False)
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
        if norm == 0:
            raise ValueError("Can not normalize a zero multiset TTNS state.")
        for component in self.ttns_list:
            component.root.tensor /= norm
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
        self.pair_group_size = max(1, int(os.environ.get("RENO_MS_TTN_PAIR_GROUP_SIZE", "32")))
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
                for start in range(0, len(group["pair_ids"]), self.pair_group_size):
                    stop = start + self.pair_group_size
                    pair_ids = group["pair_ids"][start:stop]
                    alpha_idx = group["alpha_idx"][start:stop]
                    beta_idx = group["beta_idx"][start:stop]
                    tensors = group["W"][start:stop]
                    scatter_groups = []
                    for alpha in dict.fromkeys(alpha_idx):
                        positions = [i for i, target in enumerate(alpha_idx) if target == alpha]
                        scatter_groups.append(
                            (int(alpha), xp.asarray(positions, dtype=np.int64))
                        )
                    node_groups.append(
                        {
                            "pair_ids": tuple(pair_ids),
                            "alpha_idx": xp.asarray(alpha_idx, dtype=np.int64),
                            "beta_idx": xp.asarray(beta_idx, dtype=np.int64),
                            "scatter_groups": tuple(scatter_groups),
                            # Kept for callers written against the old group
                            # schema.  The hop path now scatters directly by
                            # alpha_idx and does not materialize nset x npair
                            # products for every group.
                            "S": self._build_pair_to_alpha_matrix(alpha_idx, tensors[0].real.dtype),
                            "W": xp.stack([xp.asarray(w) for w in tensors]),
                            "n_pairs": len(pair_ids),
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

    def release(self):
        """Drop every tensor and tree reference held by this sweep cache.

        A multiset environment can own many large NumPy arrays.  Relying on
        function-local reference counting is fragile here because the TTNS
        nodes themselves form parent/children reference cycles.  Production
        drivers call this method in a ``finally`` block at the end of every
        TDVP step so no environment from an earlier step can remain reachable
        during the next one.
        """
        self.parent_envs.clear()
        self.children_envs.clear()
        self.tn2dofs_ttns.clear()
        self.tn2dofs_ttno.clear()
        self.basis_ttns = None
        self.basis_ttno = None
        self.n_pairs = 0

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
            bra_batch = ms_ttns.node_tensor_batch(inode, alpha_idx, device=True)
            ket_batch = ms_ttns.node_tensor_batch(inode, beta_idx, device=True)
            args.extend([bra_batch.conj(), [batch] + ms_ttns.get_node_indices(snode, conj=True)])
            args.extend([group["W"], self._batched_op_indices(snode, ms_ttns, ms_ttno, batch)])
            args.extend([ket_batch, [batch] + ms_ttns.get_node_indices(snode, ttno=ms_ttno)])
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
            bra_batch = ms_ttns.node_tensor_batch(inode, alpha_idx, device=True)
            ket_batch = ms_ttns.node_tensor_batch(inode, beta_idx, device=True)
            args.extend([bra_batch.conj(), [batch] + ms_ttns.get_node_indices(snode, conj=True)])
            args.extend([group["W"], self._batched_op_indices(snode, ms_ttns, ms_ttno, batch)])
            args.extend([ket_batch, [batch] + ms_ttns.get_node_indices(snode, ttno=ms_ttno)])
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
