# -*- coding: utf-8 -*-

"""Shared model builder and resource probes for the multiset size-scaling test.

The lattice/Hamiltonian is identical to ``2DHolstein1515_speed`` (2D Holstein,
omega0 = J = 1, g = 0.5, nu_max = 8, open boundaries, dt = 0.1); only the
lattice size and the number of steps change.  The point of this harness is not
speed but the answer to "does it still run, and if not, which phase died".
"""

import json
import os
import resource
import time

import numpy as np


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


# ----------------------------------------------------------------- parameters

NROW = env_int("NROW", 15)
NCOL = env_int("NCOL", 15)
OMEGA_0 = env_float("OMEGA_0", 1.0)
J = env_float("J", 1.0)
G = env_float("G", 0.5)
NU_MAX = env_int("NU_MAX", 8)
MAX_BONDDIM = env_int("MAX_BONDDIM", 64)
EVOLVE_DT = env_float("EVOLVE_DT", 0.1)
NSTEPS = env_int("NSTEPS", 10)
# Stop early once the evolution has burned this many seconds.  Peak memory is
# reached during the first step, so a truncated run still answers "does it fit".
STEP_BUDGET_S = env_float("STEP_BUDGET_S", 7200.0)
# Skip the evolution entirely and only map the setup phases.
SETUP_ONLY = os.environ.get("SETUP_ONLY", "0").lower() in {"1", "true", "yes"}
# Footprint-only: skip expand_bond_dimension_multiset (which is O(L^2) and
# dominates setup) so the exact template/Krylov footprints can be read off
# cheaply.  Bond dimensions stay at 1, so no evolution step is meaningful.
SKIP_EXPAND = os.environ.get("SKIP_EXPAND", "0").lower() in {"1", "true", "yes"}

OBSERVABLES = {
    "energy": False,
    "r_square": False,
    "e_occupations": True,
    "ph_occupations": False,
    "S_all": False,
    "S_maxbond_eachset": False,
    "S_maxbond": False,
    "S_maxbond_normed": False,
    "S_maxbond_unnormed": False,
    "rho": False,
    "coherent_length": False,
    "trace": False,
    "purity": False,
}


def site_index(ix, iy, ncol):
    return ix * ncol + iy


def build_j_matrix(nrow, ncol, j):
    n = nrow * ncol
    j_matrix = np.zeros((n, n))
    for ix in range(nrow):
        for iy in range(ncol):
            here = site_index(ix, iy, ncol)
            if ix + 1 < nrow:
                there = site_index(ix + 1, iy, ncol)
                j_matrix[here, there] = j_matrix[there, here] = j
            if iy + 1 < ncol:
                there = site_index(ix, iy + 1, ncol)
                j_matrix[here, there] = j_matrix[there, here] = j
    return j_matrix


def build_model():
    from renormalizer.model import HolsteinModel, Mol, Phonon
    from renormalizer.utils import Quantity

    displacement = np.sqrt(2.0 * G**2 * OMEGA_0) / OMEGA_0
    ph = Phonon.simple_phonon(Quantity(OMEGA_0), Quantity(displacement), NU_MAX)
    mol = Mol(Quantity(0), [ph])
    return HolsteinModel(
        [mol] * (NROW * NCOL), build_j_matrix(NROW, NCOL, J), scheme=2
    )


# --------------------------------------------------------------- probes

def host_peak_gb():
    """Peak resident set size of this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


def gpu_report():
    """(pool_used_GiB, pool_highwater_GiB, device_used_GiB, device_total_GiB)."""
    try:
        from renormalizer.mps.backend import USE_GPU

        if not USE_GPU:
            return None
        import cupy

        pool = cupy.get_default_memory_pool()
        free, total = cupy.cuda.runtime.memGetInfo()
        return (
            pool.used_bytes() / 1024.0**3,
            pool.total_bytes() / 1024.0**3,
            (total - free) / 1024.0**3,
            total / 1024.0**3,
        )
    except Exception:
        return None


class PhaseLog:
    """Records wall time and peak memory for each named phase."""

    def __init__(self, tag, logger):
        self.tag = tag
        self.logger = logger
        self.phases = []
        self._t0 = time.perf_counter()

    def mark(self, name, extra=None):
        now = time.perf_counter()
        gpu = gpu_report()
        entry = {
            "phase": name,
            "seconds": now - self._t0,
            "host_peak_gb": host_peak_gb(),
            "gpu_pool_used_gb": None if gpu is None else round(gpu[0], 3),
            "gpu_pool_peak_gb": None if gpu is None else round(gpu[1], 3),
            "gpu_dev_used_gb": None if gpu is None else round(gpu[2], 3),
            "gpu_dev_total_gb": None if gpu is None else round(gpu[3], 3),
        }
        if extra:
            entry.update(extra)
        self.phases.append(entry)
        self.logger.info(
            "[%s] PHASE %-24s %9.2f s | host_peak %7.2f GiB | gpu_pool %6.2f/%6.2f GiB"
            " | gpu_dev %6.2f/%6.2f GiB%s",
            self.tag,
            name,
            entry["seconds"],
            entry["host_peak_gb"],
            entry["gpu_pool_used_gb"] or 0.0,
            entry["gpu_pool_peak_gb"] or 0.0,
            entry["gpu_dev_used_gb"] or 0.0,
            entry["gpu_dev_total_gb"] or 0.0,
            "" if not extra else " | " + json.dumps(extra),
        )
        self._t0 = now
        return entry


def install_phase_probes(plog):
    """Wrap the four expensive setup routines so each is timed separately.

    These are the phases the scaling study needs to distinguish:
    ``ConstructMsModel`` (N_e^2 Model objects), ``_ConstructMsMpo``
    (N_e^2 Mpo builds), ``_active_mpo_select_grouping`` (stacked W and the
    dense pair->alpha scatter matrices, both on the GPU), and
    ``expand_bond_dimension_multiset`` (initial bond-dimension inflation).
    """
    from renormalizer.multiset.multiset_model import MultisetModel
    from renormalizer.multiset.multiset_mpo import MultisetMpo

    def wrap(owner, name, label):
        original = getattr(owner, name)

        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            plog.mark(label)
            return result

        setattr(owner, name, wrapped)

    if SKIP_EXPAND:
        def _no_expansion(self, coef=1e-10, use_hint=True):
            return None

        MultisetModel.expand_bond_dimension_multiset = _no_expansion

    wrap(MultisetModel, "ConstructMsModel", "ConstructMsModel")
    wrap(MultisetMpo, "_ConstructMsMpo", "ConstructMsMpo")
    wrap(MultisetModel, "_active_mpo_select_grouping", "select_grouping")
    wrap(MultisetModel, "expand_bond_dimension_multiset", "expand_bond_dim")


def template_footprint(ms_model):
    """Static GPU footprint of the batched-contraction templates."""
    import numpy as _np

    s_bytes = w_bytes = 0
    n_pairs = len(ms_model._active_pairs_index)
    for site_templates in ms_model._site_group_templates:
        for template in site_templates:
            s_bytes += int(_np.prod(template["S"].shape)) * template["S"].dtype.itemsize
            w_bytes += int(_np.prod(template["W"].shape)) * template["W"].dtype.itemsize
    return {
        "n_active_pairs": n_pairs,
        "scatter_S_gb": round(s_bytes / 1024.0**3, 3),
        "stacked_W_gb": round(w_bytes / 1024.0**3, 3),
    }


def krylov_vector_gb(ms_model, block_size=50, ranks=1):
    """Bytes the single-site Krylov basis needs at the widest site."""
    n_e = ms_model.N_electron
    dims = []
    for mps in [ms_model.MsMps.msmps[0]]:
        for site in range(len(mps)):
            dims.append(int(_prod(mps[site].shape)))
    dim = max(dims)
    per_vector = n_e * dim * 16 / 1024.0**3
    return {
        "widest_site_dim": dim,
        "krylov_vector_gb": round(per_vector, 4),
        "krylov_basis_gb": round(per_vector * block_size / max(ranks, 1), 3),
    }


def _prod(shape):
    out = 1
    for value in shape:
        out *= int(value)
    return out


def summarize(tag, plog, status, note, extra, logger, out_dir):
    record = {
        "tag": tag,
        "nrow": NROW,
        "ncol": NCOL,
        "n_sites": NROW * NCOL,
        "max_bonddim": MAX_BONDDIM,
        "nu_max": NU_MAX,
        "nsteps": NSTEPS,
        "status": status,
        "note": note,
        "host_peak_gb": round(host_peak_gb(), 3),
        "phases": plog.phases,
    }
    record.update(extra or {})
    gpu = gpu_report()
    if gpu is not None:
        record["gpu_pool_peak_gb"] = round(gpu[1], 3)
        record["gpu_dev_total_gb"] = round(gpu[3], 3)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    logger.info("[%s] RESULT status=%s note=%s", tag, status, note)
    logger.info("[%s] wrote %s", tag, path)
    return record
