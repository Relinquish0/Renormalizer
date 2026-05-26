import opt_einsum as oe

from renormalizer.mps.backend import np, xp
from renormalizer.mps.matrix import asxp
from renormalizer.mps.oe_contract_wrap import oe_contract_expression
from renormalizer.tn.multiset_ttn import MsTreeNodeTensor, MsTTNEnviron, MsTTNO, MsTTNS


def _batch_axis(inode, igroup, kind):
    return ("ms_pair_batch", inode, igroup, kind)


def _op_id(ms_ttno: MsTTNO):
    return str(id(ms_ttno)) + "_batched"


def _batched_child_indices(snode: MsTreeNodeTensor, i: int, ms_ttns: MsTTNS, ms_ttno: MsTTNO, batch):
    child = snode.children[i]
    dofs_ttns = ms_ttns.tn2dofs[snode]
    dofs_child_ttns = ms_ttns.tn2dofs[child]
    dofs_ttno = ms_ttno.tn2dofs[ms_ttno.node_list[ms_ttns.node_idx[snode]]]
    dofs_child_ttno = ms_ttno.tn2dofs[ms_ttno.node_list[ms_ttns.node_idx[child]]]
    return [
        batch,
        (str(id(ms_ttns)) + "_conj", str(dofs_ttns), str(dofs_child_ttns)),
        (_op_id(ms_ttno), str(dofs_ttno), str(dofs_child_ttno)),
        (str(id(ms_ttns)), str(dofs_ttns), str(dofs_child_ttns)),
    ]


def _batched_parent_indices(snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, batch):
    dofs_ttns = ms_ttns.tn2dofs[snode]
    dofs_ttno = ms_ttno.tn2dofs[ms_ttno.node_list[ms_ttns.node_idx[snode]]]
    if snode.parent is not None:
        dofs_parent_ttns = ms_ttns.tn2dofs[snode.parent]
        dofs_parent_ttno = ms_ttno.tn2dofs[ms_ttno.node_list[ms_ttns.node_idx[snode.parent]]]
    else:
        dofs_parent_ttns = dofs_parent_ttno = "root"
    return [
        batch,
        (str(id(ms_ttns)) + "_conj", str(dofs_parent_ttns), str(dofs_ttns)),
        (_op_id(ms_ttno), str(dofs_parent_ttno), str(dofs_ttno)),
        (str(id(ms_ttns)), str(dofs_parent_ttns), str(dofs_ttns)),
    ]


def _batched_op_indices(snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, batch):
    inode = ms_ttns.node_idx[snode]
    onode = ms_ttno.node_list[inode]
    all_dofs = ms_ttno.tn2dofs[onode]
    indices = [batch]
    for child in snode.children:
        ochild = ms_ttno.node_list[ms_ttns.node_idx[child]]
        indices.append((_op_id(ms_ttno), str(all_dofs), str(ms_ttno.tn2dofs[ochild])))
    for dofs in all_dofs:
        indices.append(("up", str(dofs)))
        indices.append(("down", str(dofs)))
    if snode.parent is None:
        indices.append((_op_id(ms_ttno), "root", str(all_dofs)))
    else:
        oparent = ms_ttno.node_list[ms_ttns.node_idx[snode.parent]]
        indices.append((_op_id(ms_ttno), str(ms_ttno.tn2dofs[oparent]), str(all_dofs)))
    return indices


def _contract_expression(args, x_shape, x_indices, y_indices):
    args_fake = args.copy()
    args_fake.extend([np.empty(x_shape), x_indices])
    args_fake.append(y_indices)
    indices, tensors = oe.parser.convert_interleaved_input(args_fake)
    expr_args = [asxp(t) for t in tensors[:-1]] + [x_shape]
    return oe_contract_expression(
        indices,
        *expr_args,
        constants=list(range(len(tensors)))[:-1],
    )


def _stack_pair_tensors(pair_ids, getter):
    return xp.stack([xp.asarray(getter(pair_id)) for pair_id in pair_ids])


def _build_1site_specs(snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, ms_ttne: MsTTNEnviron):
    inode = ms_ttns.node_idx[snode]
    shape = snode.shape[1:]
    specs = []
    for igroup, group in enumerate(ms_ttno.node_groups[inode]):
        pair_ids = group["pair_ids"]
        batch = _batch_axis(inode, igroup, "site")
        args = []
        for i in range(len(snode.children)):
            env = _stack_pair_tensors(pair_ids, lambda pair_id, i=i: ms_ttne.children_envs[pair_id][inode][i])
            args.extend([env, _batched_child_indices(snode, i, ms_ttns, ms_ttno, batch)])
        parent_env = _stack_pair_tensors(pair_ids, lambda pair_id: ms_ttne.parent_envs[pair_id][inode])
        args.extend([parent_env, _batched_parent_indices(snode, ms_ttns, ms_ttno, batch)])
        args.extend([group["W"], _batched_op_indices(snode, ms_ttns, ms_ttno, batch)])

        input_indices = [batch] + ms_ttns.get_node_indices(snode, ttno=ms_ttno)
        output_indices = [batch] + ms_ttns.get_node_indices(snode, conj=True)
        expr = _contract_expression(args, (len(pair_ids),) + tuple(shape), input_indices, output_indices)
        specs.append(
            {
                "expr": expr,
                "S": group["S"],
                "beta_idx": group["beta_idx"],
                "n_pairs": len(pair_ids),
            }
        )
    return specs


def _build_0site_specs(snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, ms_ttne: MsTTNEnviron):
    inode = ms_ttns.node_idx[snode]
    parent = snode.parent
    parent_idx = ms_ttns.node_idx[parent]
    ichild = snode.idx_as_child
    child_shape = ms_ttne.children_envs[0][parent_idx][ichild].shape[0]
    parent_shape = ms_ttne.parent_envs[0][inode].shape[0]
    shape = (child_shape, parent_shape)
    specs = []
    for igroup, group in enumerate(ms_ttno.node_groups[inode]):
        pair_ids = group["pair_ids"]
        batch = _batch_axis(inode, igroup, "bond")

        child_indices = _batched_child_indices(parent, ichild, ms_ttns, ms_ttno, batch)
        child_output = child_indices[1]
        child_input = child_indices[3]
        child_env = _stack_pair_tensors(pair_ids, lambda pair_id: ms_ttne.children_envs[pair_id][parent_idx][ichild])

        parent_indices = _batched_parent_indices(snode, ms_ttns, ms_ttno, batch)
        parent_output = tuple(list(parent_indices[1]) + ["hop0_conj"])
        parent_input = tuple(list(parent_indices[3]) + ["hop0"])
        parent_indices[1] = parent_output
        parent_indices[3] = parent_input
        parent_env = _stack_pair_tensors(pair_ids, lambda pair_id: ms_ttne.parent_envs[pair_id][inode])

        args = [
            child_env,
            child_indices,
            parent_env,
            parent_indices,
        ]
        input_indices = [batch, child_input, parent_input]
        output_indices = [batch, child_output, parent_output]
        expr = _contract_expression(args, (len(pair_ids),) + shape, input_indices, output_indices)
        specs.append(
            {
                "expr": expr,
                "S": group["S"],
                "beta_idx": group["beta_idx"],
                "n_pairs": len(pair_ids),
            }
        )
    return specs, shape


def ms_hop_expr1(snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, ms_ttne: MsTTNEnviron):
    shape = snode.shape[1:]
    dim = int(np.prod(shape))
    specs = _build_1site_specs(snode, ms_ttns, ms_ttno, ms_ttne)

    def hop(y):
        y = xp.asarray(y).reshape(ms_ttns.nset, *shape)
        y_out = xp.zeros((ms_ttns.nset,) + tuple(shape), dtype=y.dtype)
        for spec in specs:
            y_exp = y[spec["beta_idx"]]
            out = spec["expr"](y_exp)
            y_out += xp.matmul(spec["S"], out.reshape(spec["n_pairs"], dim)).reshape((ms_ttns.nset,) + tuple(shape))
        return y_out.reshape(ms_ttns.nset * dim)

    return hop


def ms_hop_expr0(snode: MsTreeNodeTensor, ms_ttns: MsTTNS, ms_ttno: MsTTNO, ms_ttne: MsTTNEnviron):
    if snode.parent is None:
        raise ValueError("0-site hop is undefined at the root.")
    specs, shape = _build_0site_specs(snode, ms_ttns, ms_ttno, ms_ttne)
    dim = int(np.prod(shape))

    def hop(y):
        y = xp.asarray(y).reshape(ms_ttns.nset, *shape)
        y_out = xp.zeros((ms_ttns.nset,) + shape, dtype=y.dtype)
        for spec in specs:
            y_exp = y[spec["beta_idx"]]
            out = spec["expr"](y_exp)
            y_out += xp.matmul(spec["S"], out.reshape(spec["n_pairs"], dim)).reshape((ms_ttns.nset,) + shape)
        return y_out.reshape(ms_ttns.nset * dim)

    return hop
