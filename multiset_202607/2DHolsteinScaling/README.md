# 2D Holstein multiset — size-scaling probe (M = 64, T = 0 K)

Answers one question: **how large a 2D Holstein lattice can the multiset code
still run at virtual bond dimension 64**, on 1 V100 and on 4 V100s, and which
phase runs out of memory first.

Hamiltonian and parameters are identical to `../2DHolstein1515_speed`
(omega0 = J = 1, g = 0.5, nu_max = 8, open boundaries, dt = 0.1, T = 0);
only `NROW`/`NCOL` change. Ten evolution steps is the pass criterion —
throughput is not measured here.

## Layout

| path | role |
|---|---|
| `common.py` | model builder + per-phase time/memory probes (shared by both parts) |
| `footprint_model.py` | closed-form GPU/host footprint, fitted to the measured small sizes |
| `multiset/run_scaling.py` | single-GPU probe |
| `multiset/submit.slurm` | single-GPU size ladder |
| `multiset_parallel/run_scaling.py` | 4-GPU MPI probe |
| `multiset_parallel/submit.slurm` | 4-GPU size ladder |
| `probe_setup_ladder.sh` | setup-only ladder, no queue needed (uses local device 0) |
| `results/` | one JSON per (size, rank) plus the SLURM logs |

`multiset_parallel/run_scaling.py` imports the three MPI monkey patches from
`../2DHolstein1515_speed/multiset_parallel/` instead of copying them, so this
probe always measures the same parallel code as the production run.

## Running

```bash
# single GPU
sbatch --export=ALL,SIZES="15 18 20",NSTEPS=10,STEP_BUDGET_S=2400 multiset/submit.slurm
# 4 GPUs
sbatch --export=ALL,SIZES="15 20 25",NSTEPS=10,STEP_BUDGET_S=2400 multiset_parallel/submit.slurm
# projected footprint, no GPU needed
python footprint_model.py --sizes 15 20 25 28 --ranks 1 4
```

Knobs (all environment variables): `NROW`, `NCOL`, `MAX_BONDDIM`, `NU_MAX`,
`EVOLVE_DT`, `NSTEPS`, `STEP_BUDGET_S` (stop early once the evolution has burned
this many seconds — peak memory is already reached in step 1), `SETUP_ONLY`
(skip the evolution and map the setup phases only).

## Phases reported

`build_model` -> `ConstructMsModel` (N_e^2 `Model` objects) -> `ConstructMsMpo`
(N_e^2 `Mpo` builds) -> `select_grouping` (stacked `W` and the dense
pair->alpha scatter matrices, both on the GPU; **called twice**, the second time
after the Hamiltonian offset is applied) -> `expand_bond_dim` -> `job_ready`
-> `step_1` ... `step_10`.

Each phase logs its own wall time, peak host RSS, CuPy pool usage/high-water,
and whole-device usage, so a failure lands on a named phase.

## Validated apparatus (2026-08-18)

Both probes were smoke-tested end-to-end at 4x4, M=64, on an RTX A5000:

| probe | expand_bond_dim | step_1 | step_2 | peak GPU pool |
|---|---|---|---|---|
| `multiset/run_scaling.py` (1 GPU)          | 10.00 s | 8.11 s  | 6.34 s  | 0.65 GB |
| `multiset_parallel/run_scaling.py` (4 rank)| 2.84 s  | 17.75 s | 14.07 s | 0.30 GB/rank |

The 4-rank run really does exercise the distributed path: the expansion is
3.5x faster (`mpi_expand_patch` distributes it), while the steps are slower
because 4 ranks were sharing one GPU in this smoke test.

## Memory model

`footprint_model.py` holds the closed-form model.  Exact terms come from the
array shapes the library allocates; two coefficients are calibrated against
measured probes.  Key structural findings:

* The widest single-site dimension is `M*d*M = 64*8*64 = 32768`, **independent
  of L**.  The Krylov basis is therefore `50 * L * 32768 * 16 B = L * 0.0244 GB`.
* `mpi_krylov_patch._partition_vector` shards `V_local` by electronic block
  (`vector_block_count = N_electron`) and in `local` mode never allgathers the
  full vector, so the Krylov basis is **exactly 1/P**.
* `mpi_apply_hop_patch` only *slices* `group["S"]` (lines 202, 465); it never
  rebuilds the templates.  The unpatched library therefore builds the full
  dense scatter `S` (`L**2 * n_pairs * 8 B`, i.e. **O(L^3)**) and stacked `W`
  on **every rank** -- both are replicated, not sharded.
* The hop working set follows the electronic-row decomposition: a rank owns
  `N/P` rows plus one ghost row on each side, so it touches `1/P + 2/N` of the
  lattice.

That sharding model reproduces the measured 4x4 pair to within 3%:
predicted `0.391/4 + 0.259*(1/4 + 2/4) = 0.292` GB vs measured `0.30` GB.

Run `python footprint_model.py` for the projection table.

## Measured results (2026-08-18)

All numbers below were measured on one RTX A5000 (24564 MiB) unless marked
"model".  The V100 budget used for the ceilings is 31.38 GiB usable
(31.73 total minus a 0.35 GiB CUDA context).

### Why the Krylov solver is the memory bottleneck

`renormalizer/lib/krylov/krylov.py:56` allocates the *entire* Lanczos basis
`V = xp.empty((block_size, len(vstart)))` before the first iteration, and the
basis cannot be freed because `_expm_krylov` projects back through it
(`V @ u_hess`, krylov.py:25).  In multiset the vector spans every electronic
component at once (multiset_model.py:533-539), so
`len(vstart) = N_electron * M*d*M`, i.e. N_e times an ordinary MPS site vector.
`block_size` is 50 at all three call sites (539, 589, 631) and is never
overridden, yet `krylov_probe.py` measures convergence at **17 iterations at
every lattice size tested** -- about two thirds of the largest allocation in
the program is never touched.

### Krylov footprint, single GPU (krylov_probe.py)

site_dim = M*d*M = 32768 and lanczos_iters = 17 at every size.

| lattice | vector_len  | one vector | 50-vector basis | measured peak | peak/basis |
|---------|-------------|------------|-----------------|---------------|------------|
| 15x15   |  7,372,800  | 0.110 GiB  |  5.493 GiB      |  6.372 GiB    | 1.16       |
| 20x20   | 13,107,200  | 0.195 GiB  |  9.766 GiB      | 11.328 GiB    | 1.16       |
| 25x25   | 20,480,000  | 0.305 GiB  | 15.259 GiB      | 17.700 GiB    | 1.16       |
| 28x28   | 25,690,112  | 0.383 GiB  | 19.141 GiB      | **22.203 GiB**| 1.16       |

The constant 1.16 is ~6 extra full-length vectors (w, res, new_res, the
allclose temporaries, the normalised start, the Afunc output).

### Krylov footprint per rank, production MPI patch (krylov_probe_mpi.py)

| lattice | local_fraction | local basis | measured peak | peak/basis |
|---------|----------------|-------------|---------------|------------|
| 15x15   | 0.2533         | 1.392 GiB   |  1.792 GiB    | 1.29       |
| 20x20   | 0.2500         | 2.441 GiB   |  3.149 GiB    | 1.29       |
| 25x25   | 0.2500         | 3.833 GiB   |  4.942 GiB    | 1.29       |
| 28x28   | 0.2500         | 4.785 GiB   | **6.173 GiB** | 1.29       |

The basis shards exactly 1/P.  28x28 drops 22.20 -> 6.17 GiB per rank = 3.60x,
not 4x, because `mpi_krylov_patch.py:296` materialises the whole `vstart` on
every rank before slicing it (`vstart[start:stop].copy()`).

### Templates are replicated, not sharded (run_mpi_footprint.sh, 4 ranks, 15x15)

| quantity        | 1 rank | rank 0 | rank 1 | rank 2 | rank 3 |
|-----------------|--------|--------|--------|--------|--------|
| n_active_pairs  | 1065   | 1065   | 1065   | 1065   | 1065   |
| scatter_S_gb    | 0.402  | 0.402  | 0.402  | 0.402  | 0.402  |
| stacked_W_gb    | 0.186  | 0.186  | 0.186  | 0.186  | 0.186  |

Byte-for-byte identical: neither template is divided among ranks.  `S` costs
`L^2 * n_pairs * 8` bytes -- O(L^3) -- and the only patch references to it
(mpi_apply_hop_patch.py:202 and :465) merely slice it, so `group["S"]` stays
resident in full on every rank.

### Templates: exact sizes vs closed form (SKIP_EXPAND ladder)

| lattice | n_pairs | S measured | S = L^2*pairs*8 | W measured | W = L*pairs*832 |
|---------|---------|------------|-----------------|------------|-----------------|
| 15x15   | 1065    | 0.402      | -0.07%          | 0.186      | -0.17%          |
| 18x18   | 1548    | 1.211      | -0.02%          | 0.389      | -0.09%          |
| 20x20   | 1920    | 2.289      | -0.01%          | 0.594      | +0.18%          |
| 22x22   | 2332    | 4.070      | +0.00%          | 0.872      | +0.30%          |
| 25x25   | 3025    | 8.804      | -0.00%          | 1.459      | +0.41%          |

S is exact to 0.07% over a 22x range, so the 17.44 GiB at 28x28 is a closed-form
evaluation rather than an extrapolation.  W drifts ~0.4% high by 25x25 but is
under 8% of the budget, so that error is immaterial.

### Why 4xV100 is not 4x GPU memory (model, fitted to the measurements below)

Per-rank share of the total, and the effective GPU-capacity gain over 1 GPU:

```
  N |                kry                  S               W             hop | 1GPU/4GPU
 15 |       1.77 (34.3%)       0.40 ( 7.8%)    0.19 ( 3.6%)    2.80 (54.3%) |     2.05x
 20 |       3.15 (28.4%)       2.29 (20.7%)    0.60 ( 5.4%)    5.05 (45.6%) |     1.87x
 25 |       4.92 (21.3%)       8.80 (38.0%)    1.46 ( 6.3%)    7.95 (34.4%) |     1.65x
 28 |       6.17 (17.2%)      17.44 (48.5%)    2.31 ( 6.4%)   10.01 (27.9%) |     1.53x
```

Three reasons, each measured: the Krylov basis shards 1/P correctly; `S` and `W`
are fully replicated and `S` grows as O(L^3), becoming the largest per-rank term
from ~25x25; and 70% of the hop working set never shards (see the two-point fit
below).  The gain shrinks as the lattice grows.  Note this is the gain in *GPU*
capacity only -- the host side moves the other way, see the combined ceilings.

### First real V100 step (job 128435, 4 ranks, 15x15)

`gpu_pool 0.59/5.18 GiB`, `gpu_dev 5.58/31.73`, `host_peak 101.06 GiB`, 766.78 s.
The peak is stable across steps (5.18 GiB at steps 1-4), so a truncated run
answers "does it fit", and all four ranks agree to within 2% -- the row
decomposition is balanced.  The run also confirmed on real hardware after a real
expansion: `widest_site_dim = 32768` (= M*d*M), `krylov_basis_gb = 1.373`
(= 5.493/4, so 1/P sharding holds), and `scatter_S_gb`/`stacked_W_gb` =
0.402/0.186 at rank 0 (replication reproduces on V100).  Expansion took 365 s
and peaked at 0.79 GiB of pool -- a time cost, not a memory wall.  At 20x20
expansion took 1187 s, consistent with O(L^2) time (3.25x vs 3.16x expected).

### hop: refitted on two measured points (supersedes the halo model)

Two 15x15 pool peaks pin the hop term without ambiguity:

| ranks | measured pool peak | minus kry/S/W | => hop      |
|-------|--------------------|---------------|-------------|
| 1     | 10.578 GiB (A5000) | 6.372/0.402/0.186 | 3.618 GiB |
| 4     |  5.180 GiB (V100)  | 1.792/0.402/0.186 | 2.800 GiB |

The measured sharding factor is **0.774**, not the 0.383 that `1/P + 2/N`
predicted.  Fitting `hop = a + b/P` gives `a = 2.527` and `b = 1.091` GiB at
15x15, i.e. in units of `n_pairs*dim*16`:

    hop(P) = base * (4.86 + 2.10 / P)

**70% of the hop working set never shards at all.**  An earlier revision of this
file raised HOP_TEMPS to 14.05 from the 4-rank point alone; that was a
misattribution -- the coefficient is ~6.96 (close to the original 7.7) and the
error was entirely in the halo model.  The two-point fit reproduces both
measurements exactly (10.58 and 5.16).

### Combined GPU + host ceilings on a 1 TB / 4xV100 node

Host memory does **not** shard: 101.04 GiB on 1 rank vs 101.06 GiB *per rank*
on 4 (ratio 1.000), so P ranks need P times the total host RAM.  It grows as
L^2.18.  Applying both limits at once:

```
        GPU/rank (31.4)   host total (1024)   verdict
 1 GPU  23x23  30.0            651            max, next blocked by GPU
        24x24  33.9            784            GPU OOM
 4 GPU  18x18   8.2            895            max, next blocked by host
        19x19   9.6           1133            HOST OOM
        20x20  11.1           1417            HOST OOM (observed, job 128435)
```

**4 GPUs reach a smaller lattice than 1 GPU: 18x18 vs 23x23.**  At the 4-GPU
ceiling the cards are only 26% full (8.2 of 31.4 GiB).  The parallel code is
starved by host RAM long before GPU memory matters, and because host memory
replicates per rank, adding ranks makes that constraint strictly worse.

This is the sharpest form of "why doesn't 4xV100 give 4x memory": it gives
1.65-2.05x more *GPU* memory while demanding 4x more *host* memory, and host is
what runs out first, so the net effect on maximum size is negative.

Fix priority implied by these numbers, in order:
1. Cut host memory (the binding constraint; 22x the GPU footprint, unsharded).
2. Shard the replicated `S` (O(L^3), 38% of per-rank GPU at 25x25).
3. Shard the 70% of hop temporaries that currently stay full-lattice.
4. `block_size` 50 -> 20 (cheap, but only helps once 1-3 are done).

## Host memory is the real wall (job 128435, measured 2026-08-18)

The 20x20 4-rank case did **not** die of GPU memory.  It was killed by the
Linux OOM killer with the GPU at 3.51 / 31.73 GiB:

```
slurmstepd: error: Detected 1 oom_kill event in StepId=128435.1.
srun: error: curie-gpu003: task 1: Out Of Memory
```

Host peak per rank by phase:

| phase            | 15x15  | 20x20        |
|------------------|--------|--------------|
| ConstructMsMpo   |   3.31 |  13.29       |
| select_grouping  |   5.39 |  19.82       |
| expand_bond_dim  |  16.48 |  57.68       |
| step_1           | 101.06 |  OOM-killed  |
| plateau          | 115.83 |  --          |

At 15x15 the host footprint is **115.83 GiB/rank against 5.18 GiB on the GPU --
a factor of 22**.  Host memory scales as L^2.10, fitted on three measured expansion-phase peaks
(16.48 / 57.68 / 141.29 GiB per rank at L = 225 / 400 / 625).  The pairwise
exponents drift downward -- 2.177 then 2.007 -- trending to **L^2 = N_e^2**,
which is the signature of `ConstructMsModel` / `_ConstructMsMpo` building N_e^2
host-side Model and Mpo objects.  That favours those over the
`RENO_MPI_ALLREDUCE_MODE=host` staging hypothesis, whose buffers would scale
with the vector (L), not L^2.  The 25x25 expansion peak of 141.29 GiB came in
4.5% under the 148 GiB predicted from the two-point fit.  Projecting the 15x15 step plateau:

| lattice | GiB/rank | 4 ranks | vs mem=1T |
|---------|----------|---------|-----------|
| 15x15   |    115.8 |   463   | fits      |
| 18x18   |    256.2 |  1025   | at the limit |
| 20x20   |    405.4 |  1622   | OOM (observed) |
| 25x25   |   1071.3 |  4285   | OOM       |
| 28x28   |   1754.8 |  7019   | OOM       |

**Host-bound ceiling on 4 ranks at mem=1T is 18x18**, far below the
27x27 the GPU model allows.  Raising the allocation to 2 TB buys ~21x21 and
4 TB buys ~25x25 -- the L^2.18 growth means memory is a poor lever.

This supersedes the GPU-derived ceilings as a statement about what will
actually run: the GPU budget is real but is never the first thing to fail on
this cluster.  The GPU findings (Krylov 22.20 GiB at 28x28 single / 6.17 per
rank, 1/P basis sharding, replicated O(L^3) `S`) are unaffected -- they explain
GPU memory, which simply is not the binding constraint at P=4.

Not yet diagnosed: where the 22x host-to-GPU ratio comes from.  Two candidates
worth profiling before any further scaling work -- `RENO_MPI_ALLREDUCE_MODE=host`
stages every collective through host buffers (mandatory on this OpenMPI build),
and `ConstructMsModel` / `_ConstructMsMpo` build N_e^2 host-side Model and Mpo
objects (24.27 GiB/rank at 25x25 before anything else runs).

## Where the host memory actually goes (env_probe.py, 2026-08-19)

The 22x host-to-GPU ratio flagged above is now resolved.  `Environ.write` stores
every L/R tensor with `asnumpy` (renormalizer/mps/lib.py:115), and multiset builds
one `Environ` per active pair (multiset_model.py:246-253), so the whole L/R cache
is host-resident.  On top of that `multiset_model.py:520` builds a full conjugate
copy of all N_e MPSs on every step, and keeps it alive for the whole step, even
though it is only consumed on a cache miss (the cache survives across steps via
`_rescale_environ_cache_after_normalize`).

| lattice | n_pairs | sites | Environ cache | per (pair,site) | share of step-1 growth |
|---------|---------|-------|---------------|-----------------|------------------------|
| 6x6     | 156     | 36    | 0.774 GiB     | 1.378e-4 GiB    | 39.9% |
| 8x8     | 288     | 64    | 2.622 GiB     | 1.423e-4 GiB    | 41.0% |
| 10x10   | 460     | 100   | 6.633 GiB     | 1.442e-4 GiB    | 41.2% |

So `environ = 1.44e-4 * n_pairs * n_sites` GiB, i.e. O(L^2), replicated per rank.
The coefficient rises 1.378 -> 1.423 -> 1.442 and the increments halve each step,
converging from below as the tapered edge sites (bond < M) become a smaller share;
1.44e-4 is the value used in the projections.  Every tensor is one of two shapes,
(64,1,64) and (64,2,64) -- 64%/36% of the bytes at 10x10 -- i.e. two environment
tensors (L and R) per pair per site, at the MPO bond widths.

The 10x10 probe also measures the MPS list directly: 4.700 GiB for 100 MPSs x 100
sites = **4.70e-4 GiB per (alpha, site)**.  A saturated `(64,9,64)` complex128 site
would be 5.49e-4, so QN blocking plus the tapered edges leave the bonds ~14% under
M.  Applying both coefficients at 15x15 gives setup 16.5 + environ 34.6 + 2 x MPS
47.6 = **98.6 GiB/rank against the measured 101.06** -- 2.4% low, with the residual
in the transient Krylov/hop host buffers the model does not carry.
`improvement_model.py` projects what each fix buys; `4V100_瓶颈与改进方案.md` is the
full write-up.

## `S` is a one-hot matrix doing a scatter-add

`_build_pair_to_alpha_matrix` (multiset_model.py:219) builds a dense one-hot
(N_e x n_pairs) matrix and `multiset_model.py:797` applies it as
`Y_out += xp.matmul(S, out_flat)`.  The same information is already in
`alpha_idx` (`:207`).  Microbenchmark on an A5000, M=64 d=9 w=2:

| N  | einsum chain | `S @ out` | `scatter_add` | matmul share of hop | speedup |
|----|--------------|-----------|---------------|---------------------|---------|
| 10 | 0.139 s      | 0.041 s   | 0.0014 s      | 22.5%               | 28.8x   |
| 15 | 0.328 s      | 0.201 s   | 0.0035 s      | 38.0%               | 58.0x   |
| 20 | 0.582 s      | 0.592 s   | 0.0062 s      | 50.4%               | 95.4x   |

Its FLOPs go as `N_e * n_pairs * dim` (O(L^2)) while the real einsums go as O(L),
so by 20x20 half the hop time is spent multiplying by a permutation-like matrix.
Removing it also removes the 17.44 GiB `S` at 28x28 -- 49% of per-rank GPU.

## The environment cache is a per-call-latency problem, not a bandwidth one (env_timing_probe.py, 2026-08-19)

`Environ.read`/`write` were monkey-patched with `deviceSynchronize()`-bracketed
timers, installed *after* setup so only the evolution is counted:

| lattice | read/write calls | bytes/call | us/call | read | write | PCIe s | step s | share |
|---------|------------------|------------|---------|------|-------|--------|--------|-------|
| 6x6     | 22,464 / 22,464  | 72.4 KiB   | 161     | 64   | 258   |  7.23  |  51.46 | 14.0% |
| 8x8     | 73,728 / 73,728  | 74.6 KiB   | 131     | 70   | 191   | 19.27  | 191.55 | 10.1% |

Call count is exactly `n_pairs * n_sites * 8` (4 reads + 4 writes per pair-site).
A 72 KiB transfer needs ~6 us on PCIe 3.0, so **~95% of the 131-161 us is per-call
overhead** -- the Python call, the fresh pageable host allocation `asnumpy` makes
every time, and the implicit sync.  Effective bandwidth is 0.43-0.54 GiB/s.  Writes
cost 3-4x reads for the same bytes because `asnumpy` takes the pageable path while
`asxp` lands in the CuPy pool.

**The share falls with size, so extrapolate carefully.**  Fitted against
`pairs*sites`, step time goes as ^1.106 but PCIe only as ^0.825 (the per-call
overhead amortises), giving 14.0% -> 10.1% -> ~7.7% at 15x15.  In absolute terms
15x15 costs **1.92 M calls, 136 GiB, ~251 s per rank per step** (environments are
not sharded, so every rank pays in full).  That is 7.7% of the projected 3268 s
A5000 step -- but 33% of the *measured* 766.78 s 4-rank V100 step, because per-call
latency is host/driver-bound and does not shrink when the GPU gets 16x faster at
FP64.  The faster the card, the more this dominates.  Caveat: 131 us/call was
measured on the local A5000 host; the cluster's PCIe topology was not measured, so
the 33% is the upper side of the estimate.
