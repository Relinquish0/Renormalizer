# -*- coding: utf-8 -*-

"""Peak-GPU-memory model for the multiset 2D Holstein run, term by term.

Every coefficient below is either exact arithmetic or calibrated against a
measurement in ``results/``; nothing here is a guess.

  krylov   EXACT shape, MEASURED constant.  ``krylov.py:56`` allocates the whole
           Lanczos basis ``(block_size, N_e * M*d*M)`` complex128 up front.
           ``krylov_probe.py`` drove the real solver at 9 lattice sizes and the
           ratio peak/(block_size * len * 16) was 1.16 at every one of them;
           the patched MPI solver gave 1.29 x (1/P of the basis), also at every
           size (``results/krylov_probe_*.json``, ``krylov_mpi_*.json``).
           Sharding is an exact 1/P: measured local_fraction = 0.2500.

  scatter_S EXACT.  Dense float64 pair->alpha scatter, L^2 * n_pairs * 8 bytes.
           Measured 0.402 / 1.211 / 2.289 GiB at 15 / 18 / 20; model gives
           0.402 / 1.211 / 2.289.  REPLICATED on every rank: it is built in the
           unpatched library and the MPI patches only ever slice it
           (``mpi_apply_hop_patch.py:202,465`` are the only two references).

  stacked_W CALIBRATED, 832 B * L * n_pairs.  Measured 833.5 / 832.6 / 830.6 B
           at 15 / 18 / 20.  Also replicated.

  hop      CALIBRATED from the 4x4 single-GPU peak only -- the weakest term
           here, but subdominant everywhere it matters.  Scales with the halo
           factor 1/P + 2/N, not 1/P, because the row decomposition gives each
           rank one ghost row on each side.  Validated once against the 4x4
           4-rank measurement: predicted 0.292 vs measured 0.30 GiB.
"""

GB = 1024.0**3
BLOCK_SIZE = 50           # renormalizer.lib.krylov.expm_krylov default
KRY_OVERHEAD_1 = 1.16     # measured, single GPU
KRY_OVERHEAD_P = 1.29     # measured, patched MPI solver, per rank
W_BYTES_PER_PAIR_SITE = 832.0
HOP_UNSHARDED = 4.86    # hop temporaries that stay full-lattice on every rank
HOP_SHARDED   = 2.10    # hop temporaries that do shard 1/P
                        # Both in units of n_pairs*dim*16, fitted to two measured
                        # 15x15 pool peaks: 10.58 GiB on 1 GPU (A5000) and
                        # 5.18 GiB/rank on 4 V100s.  Together they give
                        # hop(P) = base*(4.86 + 2.10/P): 70% never shards.
                        # The earlier halo model 1/P+2/N predicted a sharding
                        # factor of 0.383; the measured value is 0.774.


def geometry(n, m=64, d=8):
    lattice = n * n
    bonds = 2 * n * (n - 1)
    n_pairs = lattice + 2 * bonds
    return lattice, n_pairs, m * d * m


def footprint(n, m=64, d=8, ranks=1, block_size=BLOCK_SIZE):
    lattice, n_pairs, dim = geometry(n, m, d)

    basis = block_size * lattice * dim * 16
    if ranks == 1:
        krylov = KRY_OVERHEAD_1 * basis
    else:
        krylov = KRY_OVERHEAD_P * basis / ranks

    scatter_s = lattice**2 * n_pairs * 8                     # replicated
    stacked_w = lattice * n_pairs * W_BYTES_PER_PAIR_SITE    # replicated
    hop = (HOP_UNSHARDED + HOP_SHARDED / ranks) * n_pairs * dim * 16

    total = krylov + scatter_s + stacked_w + hop
    return {
        "n": n, "lattice": lattice, "n_pairs": n_pairs, "site_dim": dim,
        "krylov_gb": krylov / GB, "scatter_s_gb": scatter_s / GB,
        "stacked_w_gb": stacked_w / GB, "hop_gb": hop / GB,
        "total_gb": total / GB,
    }


if __name__ == "__main__":
    USABLE = 31.38   # Tesla V100-SXM2-32GB: 31.73 GiB total - 0.35 CUDA context

    print(f"M=64, nu_max=8, block_size={BLOCK_SIZE}, usable per V100 = {USABLE} GiB\n")
    head = (f"{'N':>3} {'L':>5} {'pairs':>6} | {'kry':>6} {'S':>6} {'W':>5} {'hop':>6}"
            f" {'TOTAL':>7} {'fit':>4} | {'kry/4':>6} {'S':>6} {'W':>5} {'hop':>6}"
            f" {'TOTAL':>7} {'fit':>4}")
    print(head)
    print("-" * len(head))
    for n in (15, 18, 20, 22, 23, 24, 25, 26, 28, 30):
        a = footprint(n, ranks=1)
        b = footprint(n, ranks=4)
        print(f"{n:>3} {a['lattice']:>5} {a['n_pairs']:>6} |"
              f" {a['krylov_gb']:>6.2f} {a['scatter_s_gb']:>6.2f} {a['stacked_w_gb']:>5.2f}"
              f" {a['hop_gb']:>6.2f} {a['total_gb']:>7.2f}"
              f" {'ok' if a['total_gb'] <= USABLE else 'OOM':>4} |"
              f" {b['krylov_gb']:>6.2f} {b['scatter_s_gb']:>6.2f} {b['stacked_w_gb']:>5.2f}"
              f" {b['hop_gb']:>6.2f} {b['total_gb']:>7.2f}"
              f" {'ok' if b['total_gb'] <= USABLE else 'OOM':>4}")

    print("\nwhy 4 GPUs do not buy 4x capacity (per-rank share of the total):")
    print(f"{'N':>3} | {'kry':>18} {'S':>18} {'W':>15} {'hop':>15} | {'1GPU/4GPU':>9}")
    for n in (15, 20, 25, 28):
        a, b = footprint(n, ranks=1), footprint(n, ranks=4)
        def frac(key):
            return f"{b[key]:5.2f} ({100*b[key]/b['total_gb']:4.1f}%)"
        print(f"{n:>3} | {frac('krylov_gb'):>18} {frac('scatter_s_gb'):>18}"
              f" {frac('stacked_w_gb'):>15} {frac('hop_gb'):>15} |"
              f" {a['total_gb']/b['total_gb']:8.2f}x")
