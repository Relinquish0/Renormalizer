# -*- coding: utf-8 -*-

"""Isolated measurement of the Krylov solver's GPU footprint.

The single-site TDVP vector in the multiset method has length
``N_electron * M * d * M`` (multiset_model.py:533-535), and
``renormalizer/lib/krylov/krylov.py:56`` allocates the whole Lanczos basis
``V = (block_size, len(vstart))`` complex128 up front, before the first
iteration.  This probe drives the real ``expm_krylov`` with a cheap diagonal
Afunc at exactly those vector lengths, so the measured peak is the solver's own
cost with no Hamiltonian contraction mixed in.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RENO_GPU", "0")

from renormalizer.lib import expm_krylov
from renormalizer.mps.backend import USE_GPU, xp

GB = 1024.0**3
M = int(os.environ.get("MAX_BONDDIM", "64"))
D = int(os.environ.get("NU_MAX", "8"))
BLOCK = int(os.environ.get("BLOCK_SIZE", "50"))
SIZES = [int(v) for v in os.environ.get("SIZES", "8 10 12 15 18 20").split()]
# Spectral width of the fake Hamiltonian.  A wide spectrum delays Lanczos
# convergence, which is the worst case for the retained-basis width.
SPREAD = float(os.environ.get("SPREAD", "50.0"))


def pool():
    if not USE_GPU:
        return 0.0, 0.0
    p = xp.get_default_memory_pool()
    return p.used_bytes() / GB, p.total_bytes() / GB


def probe(n_lattice):
    n_e = n_lattice * n_lattice
    dim = M * D * M
    n = n_e * dim
    analytic = BLOCK * n * 16 / GB

    if USE_GPU:
        xp.get_default_memory_pool().free_all_blocks()
    base_used, _ = pool()

    diag = xp.asarray(np.linspace(-SPREAD, SPREAD, n))
    v0 = xp.asarray(np.random.default_rng(0).standard_normal(n).astype(np.complex128))

    def afunc(v):
        return diag * v

    if USE_GPU:
        xp.get_default_memory_pool().free_all_blocks()
    pre_used, _ = pool()

    out, iters = expm_krylov(afunc, -1j * 0.05, v0, block_size=BLOCK)
    used, peak = pool()
    del out, diag, v0
    if USE_GPU:
        xp.get_default_memory_pool().free_all_blocks()

    return {
        "lattice": f"{n_lattice}x{n_lattice}",
        "n_electron": n_e,
        "site_dim": dim,
        "vector_len": n,
        "one_vector_gb": round(n * 16 / GB, 4),
        "analytic_basis_gb": round(analytic, 3),
        "lanczos_iters": int(iters),
        "inputs_gb": round(pre_used - base_used, 3),
        "measured_peak_gb": round(peak, 3),
        "solver_only_gb": round(peak - (pre_used - base_used), 3),
        "overhead_vs_basis": round((peak - (pre_used - base_used)) / analytic, 3),
    }


def main():
    print(f"USE_GPU={USE_GPU} M={M} d={D} block_size={BLOCK} spread={SPREAD}")
    rows = []
    for n_lattice in SIZES:
        try:
            row = probe(n_lattice)
        except Exception as error:  # noqa: BLE001 - OOM is a result
            row = {"lattice": f"{n_lattice}x{n_lattice}",
                   "status": type(error).__name__, "note": str(error)[:200]}
            print(json.dumps(row))
            rows.append(row)
            break
        print(json.dumps(row))
        rows.append(row)

    out_dir = os.environ.get("RESULT_DIR", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.environ.get("RUN_TAG", "krylov_probe") + ".json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
