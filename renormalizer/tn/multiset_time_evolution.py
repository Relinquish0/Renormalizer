import logging
import time
from typing import List, Tuple, Union

from scipy import stats

from renormalizer.lib import expm_krylov
from renormalizer.mps.backend import np
from renormalizer.mps.matrix import asxp
from renormalizer.tn.multiset_hop_expr import ms_hop_expr0, ms_hop_expr1
from renormalizer.tn.multiset_ttn import MsTreeNodeTensor, MsTTNEnviron, MsTTNO, MsTTNS


logger = logging.getLogger(__name__)


def _node_debug_label(snode: MsTreeNodeTensor, ms_ttns: MsTTNS):
    inode = ms_ttns.node_idx[snode]
    dofs = ms_ttns.tn2dofs[snode]
    return inode, dofs


def evolve_ms_tdvp_ps(ms_ttns: MsTTNS, ms_ttno: MsTTNO, coeff: Union[complex, float], tau: float):
    ms_ttns.check_canonical()

    time_start = time.perf_counter()
    ms_ttne = MsTTNEnviron(ms_ttns, ms_ttno)
    logger.info("MS-TDVP-PS profile MsTTNEnviron_init seconds=%.6f", time.perf_counter() - time_start)

    time_start = time.perf_counter()
    local_steps1 = _tdvp_ps_forward(ms_ttns, ms_ttno, ms_ttne, coeff, tau / 2)
    logger.info("MS-TDVP-PS profile forward_sweep seconds=%.6f", time.perf_counter() - time_start)

    time_start = time.perf_counter()
    local_steps2 = _tdvp_ps_backward(ms_ttns, ms_ttno, ms_ttne, coeff, tau / 2)
    logger.info("MS-TDVP-PS profile backward_sweep seconds=%.6f", time.perf_counter() - time_start)

    if local_steps1 or local_steps2:
        steps_stat = stats.describe(local_steps1 + local_steps2)
        logger.debug(f"MS-TDVP-PS Krylov space: {steps_stat}")
        ms_ttns.evolve_config.stat = steps_stat
    return ms_ttns


def _new_sweep_profile():
    return {
        "evolve_1site": 0.0,
        "evolve_0site": 0.0,
        "decompose": 0.0,
        "merge": 0.0,
        "push_cano": 0.0,
        "children_env_update": 0.0,
        "parent_env_update": 0.0,
    }


def _add_profile_time(profile, key, time_start):
    profile[key] += time.perf_counter() - time_start


def _log_sweep_detail(name, profile, local_steps):
    total = sum(profile.values())
    logger.info(
        "MS-TDVP-PS profile %s detail total_accounted=%.6f evolve_1site=%.6f evolve_0site=%.6f "
        "decompose=%.6f merge=%.6f push_cano=%.6f children_env_update=%.6f parent_env_update=%.6f "
        "nlocal=%d",
        name,
        total,
        profile["evolve_1site"],
        profile["evolve_0site"],
        profile["decompose"],
        profile["merge"],
        profile["push_cano"],
        profile["children_env_update"],
        profile["parent_env_update"],
        len(local_steps),
    )


def _tdvp_ps_forward(
    ms_ttns: MsTTNS, ms_ttno: MsTTNO, ms_ttne: MsTTNEnviron, coeff: Union[complex, float], tau: float
) -> List[int]:
    local_steps: List[int] = []
    profile = _new_sweep_profile()
    stack: List[Tuple[MsTreeNodeTensor, int]] = [(ms_ttns.root, -1)]
    while stack:
        snode, ichild = stack[-1]
        if (not snode.children) or (ichild == len(snode.children) - 1):
            time_part = time.perf_counter()
            ms, j = evolve_1site(snode, ms_ttns, ms_ttno, ms_ttne, coeff, tau)
            _add_profile_time(profile, "evolve_1site", time_part)
            snode.tensor = ms.reshape(snode.shape)
            local_steps.append(j)

            if snode.parent is None:
                stack.pop()
                continue

            time_part = time.perf_counter()
            ms = ms_ttns.decompose_to_parent(snode)
            _add_profile_time(profile, "decompose", time_part)

            time_part = time.perf_counter()
            ms_ttne.build_children_environ_node(snode, ms_ttns, ms_ttno)
            _add_profile_time(profile, "children_env_update", time_part)

            shape = (ms_ttns.nset, ms.shape[2], ms.shape[1])
            time_part = time.perf_counter()
            ms_t, j = evolve_0site(np.swapaxes(ms, 1, 2), snode, ms_ttns, ms_ttno, ms_ttne, coeff, -tau)
            _add_profile_time(profile, "evolve_0site", time_part)

            time_part = time.perf_counter()
            ms_ttns.merge_to_parent(snode, np.swapaxes(ms_t.reshape(shape), 1, 2))
            _add_profile_time(profile, "merge", time_part)
            local_steps.append(j)

            stack.pop()
            continue

        ichild += 1
        child = snode.children[ichild]
        time_part = time.perf_counter()
        ms_ttns.push_cano_to_child(snode, ichild)
        _add_profile_time(profile, "push_cano", time_part)

        time_part = time.perf_counter()
        ms_ttne.build_parent_environ_node(snode, ichild, ms_ttns, ms_ttno)
        _add_profile_time(profile, "parent_env_update", time_part)
        stack[-1] = (snode, ichild)
        stack.append((child, -1))

    _log_sweep_detail("forward_sweep", profile, local_steps)
    return local_steps


def _tdvp_ps_backward(
    ms_ttns: MsTTNS, ms_ttno: MsTTNO, ms_ttne: MsTTNEnviron, coeff: Union[complex, float], tau: float
) -> List[int]:
    local_steps: List[int] = []
    profile = _new_sweep_profile()
    stack: List[Tuple[MsTreeNodeTensor, int]] = [(ms_ttns.root, -1)]
    while stack:
        snode, ichild = stack[-1]
        if ichild == -1:
            time_part = time.perf_counter()
            ms, j = evolve_1site(snode, ms_ttns, ms_ttno, ms_ttne, coeff, tau)
            _add_profile_time(profile, "evolve_1site", time_part)
            snode.tensor = ms.reshape(snode.shape)
            local_steps.append(j)
        if ichild == len(snode.children) - 1:
            if snode is not ms_ttns.root:
                time_part = time.perf_counter()
                ms_ttns.push_cano_to_parent(snode)
                _add_profile_time(profile, "push_cano", time_part)

                time_part = time.perf_counter()
                ms_ttne.build_children_environ_node(snode, ms_ttns, ms_ttno)
                _add_profile_time(profile, "children_env_update", time_part)
            stack.pop()
            continue

        ichild += 1
        child = snode.children[ichild]
        time_part = time.perf_counter()
        ms = ms_ttns.decompose_to_child(snode, ichild)
        _add_profile_time(profile, "decompose", time_part)

        time_part = time.perf_counter()
        ms_ttne.build_parent_environ_node(snode, ichild, ms_ttns, ms_ttno)
        _add_profile_time(profile, "parent_env_update", time_part)

        shape = ms.shape
        time_part = time.perf_counter()
        ms, j = evolve_0site(ms, child, ms_ttns, ms_ttno, ms_ttne, coeff, -tau)
        _add_profile_time(profile, "evolve_0site", time_part)

        time_part = time.perf_counter()
        ms_ttns.merge_to_child(snode, ichild, ms.reshape(shape))
        _add_profile_time(profile, "merge", time_part)
        local_steps.append(j)
        stack[-1] = (snode, ichild)
        stack.append((child, -1))

    _log_sweep_detail("backward_sweep", profile, local_steps)
    return local_steps


def evolve_1site(
    snode: MsTreeNodeTensor,
    ms_ttns: MsTTNS,
    ms_ttno: MsTTNO,
    ms_ttne: MsTTNEnviron,
    coeff: Union[complex, float],
    tau: float,
):
    ms = snode.tensor
    hop = ms_hop_expr1(snode, ms_ttns, ms_ttno, ms_ttne)
    ms_t, j = expm_krylov(lambda y: hop(y), coeff * tau, asxp(ms).ravel())
    inode, dofs = _node_debug_label(snode, ms_ttns)
    logger.debug(
        "MS-TDVP-PS Krylov 1-site inode=%s dofs=%s shape=%s tau=%s coeff=%s steps=%s",
        inode,
        dofs,
        ms.shape,
        tau,
        coeff,
        j,
    )
    return ms_t, j


def evolve_0site(
    ms: np.ndarray,
    snode: MsTreeNodeTensor,
    ms_ttns: MsTTNS,
    ms_ttno: MsTTNO,
    ms_ttne: MsTTNEnviron,
    coeff: Union[complex, float],
    tau: float,
):
    hop = ms_hop_expr0(snode, ms_ttns, ms_ttno, ms_ttne)
    ms_t, j = expm_krylov(lambda y: hop(y), coeff * tau, asxp(ms).ravel())
    inode, dofs = _node_debug_label(snode, ms_ttns)
    logger.debug(
        "MS-TDVP-PS Krylov 0-site inode=%s dofs=%s shape=%s tau=%s coeff=%s steps=%s",
        inode,
        dofs,
        ms.shape,
        tau,
        coeff,
        j,
    )
    return ms_t, j

