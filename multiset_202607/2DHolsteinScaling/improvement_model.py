# -*- coding: utf-8 -*-
"""What each proposed fix buys on a 1 TB / 4xV100 node.

GPU terms are the calibrated model in footprint_model.py.  Host terms are
calibrated on the measured 4-rank V100 run (job 128435) plus the env_probe
measurement of the Environ cache:

    environ  = C_ENV * n_pairs * n_sites        (C_ENV from envprobe 10x10)
    mps      = C_MPS * L * n_sites * M*d*M      (one complex copy of the multiset MPS)
    setup    = measured expansion-phase peak, extrapolated as L^2.1

The multiset sweep holds the environ cache, the working complex MPS *and* a full
conjugate copy of it (multiset_model.py:520), so the plateau is
setup + environ + 2*mps.  Verified against 115.8 GiB/rank at 15x15.
"""
import argparse

import footprint_model as fm

GPU_BUDGET = 31.38          # V100 32 GB minus the CUDA context
HOST_BUDGET = 1024.0        # SLURM mem=1T

C_ENV = 1.442e-4            # GiB per (pair, site); envprobe 10x10: 6.633 / (460*100)
                            # 6x6/8x8/10x10 give 1.378/1.423/1.442e-4 -- converging from below
                            # as the edge sites (bond < M) become a smaller share
C_MPS = 4.70e-4             # GiB per (alpha, site); envprobe 10x10 measures the MPS
                            # list directly: 4.700 / (100*100).  A saturated (64,9,64)
                            # complex128 site would be 5.49e-4, so QN blocking plus the
                            # tapered edges leave the bonds ~14% under M.
C_SETUP = 16.48 / 225 ** 2.1   # GiB, calibrated on the 15x15 expansion peak
SETUP_EXP = 2.1


def host_terms(n, ranks=4, shard_setup=False, shard_env=False,
               shard_mps=False, drop_conj=False):
    lattice, n_pairs, _ = fm.geometry(n)
    n_sites = lattice
    setup = C_SETUP * lattice ** SETUP_EXP
    environ = C_ENV * n_pairs * n_sites
    mps = C_MPS * lattice * n_sites
    n_mps_copies = 1.0 if drop_conj else 2.0

    if shard_setup:
        setup /= ranks
    if shard_env:
        environ /= ranks
    if shard_mps:
        # a rank owns N/P electronic rows plus one ghost row on each side
        mps *= (1.0 / ranks + 2.0 / n)
    return {"setup": setup, "environ": environ, "mps": n_mps_copies * mps,
            "total": setup + environ + n_mps_copies * mps}


SCENARIOS = [
    ("S0 baseline (today)",            dict()),
    ("S1 +lazy conj_mps",              dict(drop_conj=True)),
    ("S2 +shard Environ by alpha",     dict(drop_conj=True, shard_env=True)),
    ("S3 +shard MPS by alpha",         dict(drop_conj=True, shard_env=True, shard_mps=True)),
    ("S4 +shard setup (MsModel/Mpo)",  dict(drop_conj=True, shard_env=True, shard_mps=True,
                                            shard_setup=True)),
]


def gpu_per_rank(n, ranks, drop_S=False, block_size=fm.BLOCK_SIZE):
    f = fm.footprint(n, ranks=ranks, block_size=block_size)
    total = f["total_gb"]
    if drop_S:
        total -= f["scatter_s_gb"]
    return total


def max_lattice(fn, lo=6, hi=48):
    best = None
    for n in range(lo, hi + 1):
        if fn(n):
            best = n
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--sizes", type=int, nargs="*", default=[15, 20, 25, 28])
    args = ap.parse_args()
    P = args.ranks

    print(f"host budget {HOST_BUDGET:.0f} GiB total, GPU budget {GPU_BUDGET:.2f} GiB/rank, "
          f"{P} ranks\n")
    print("=== host per rank (GiB) and total, by scenario ===")
    head = "scenario".ljust(32) + "".join(f"{n}x{n:<8}" for n in args.sizes)
    print(head)
    for name, kw in SCENARIOS:
        cells = []
        for n in args.sizes:
            t = host_terms(n, ranks=P, **kw)["total"]
            cells.append(f"{t:6.1f}/{t * P:7.0f}")
        print(name.ljust(32) + " ".join(cells))

    print("\n=== largest lattice that fits, by scenario ===")
    print("scenario".ljust(32) + "host-bound  GPU-bound(with S)  GPU-bound(no S, bs=24)  overall")
    for name, kw in SCENARIOS:
        n_host = max_lattice(lambda n: host_terms(n, ranks=P, **kw)["total"] * P <= HOST_BUDGET)
        n_gpu = max_lattice(lambda n: gpu_per_rank(n, P) <= GPU_BUDGET)
        n_gpu2 = max_lattice(lambda n: gpu_per_rank(n, P, drop_S=True, block_size=24) <= GPU_BUDGET)
        print(name.ljust(32) + f"{n_host}x{n_host}".ljust(12)
              + f"{n_gpu}x{n_gpu}".ljust(19) + f"{n_gpu2}x{n_gpu2}".ljust(24)
              + f"{min(n_host, n_gpu2)}x{min(n_host, n_gpu2)}")

    print("\n=== host breakdown, 15x15, 4 ranks (GiB/rank) ===")
    for name, kw in SCENARIOS:
        t = host_terms(15, ranks=P, **kw)
        print(name.ljust(32) + f"setup {t['setup']:6.1f}  environ {t['environ']:6.1f}  "
                               f"mps {t['mps']:6.1f}  total {t['total']:6.1f}")


if __name__ == "__main__":
    main()
