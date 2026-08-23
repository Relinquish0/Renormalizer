# 4×V100 multiset MPS — 2D Holstein scaling

Implementation of the four-stage plan in
`multiset_202607/2DHolsteinScaling/4V100_瓶颈与改进方案.md`, whose goal is to get
the 4-GPU multiset code from the 18×18 it reached before to the S4 target of
27×27 at bond dimension 64.

## Layout

```
code/                      the reusable 4-GPU machinery
  holstein.py              2D Holstein model builder + run configuration
  mpi_common.py            rank→GPU pinning and the global α-ownership convention
  mpi_apply_hop_patch.py   distributed Hamiltonian action (pair shard / α shard + halo)
  mpi_krylov_patch.py      distributed Lanczos, including the already-local-vstart mode
  mpi_shard_patch.py       the stage 2–4 α-sharding of environments, state and setup
  mpi_expand_patch.py      bond-dimension expansion, stage 1 only
  run_dynamics.py          the driver: one lattice, one MPI job
scaling/
  run_scaling.py           runs code/run_dynamics.py once per lattice size
  submit.slurm             4V100 batch job, --mem=1500G
  submit_setup_cpu.slurm   GPU-free host-memory probe, for use while the GPUs are busy
  result/<NxN>/            per-size logs, populations (.npz) and summary (.json)
```

## Result

The ceiling is **29x29 (841 sites)** at m = 64, and it is set by host memory.
Measured over 5 steps:

| | 27x27 | 29x29 |
|---|---|---|
| host per rank (peak) | 286.1 GiB | 377.8 GiB |
| host total | 1098.6 GiB (73%) | **1453.6 GiB (96.9% of 1500)** |
| device per GPU | 11.36 GiB | 13.27 GiB |
| device total | 45.2 GiB | 52.76 GiB (42% of 126.9) |
| setup | 4389 s | 6100 s |
| per step | 3536 s | 4750 s |

At 29x29 the state breaks down as (measured, `SETUP_ONLY=0` with the state
instrumentation in `memory.py`):

| term | size |
|---|---|
| whole multiset MPS, one copy | 343.8 GiB over 841 sets |
| largest single set | 418.6 MiB |
| MPS resident per rank | 109.6 GiB (268 sets: 211 owned + ghost halo) |
| environments | 504.3 GiB total over 4089 pairs, 127.5 GiB per rank |
| setup (MPOs, templates) | 68.8 GiB per rank |

Nothing of the state lives on the device: the MPS and environment tensors stay
as NumPy arrays on the host whatever the backend is, and the GPU only holds the
transient contraction working set. That is why the host is 97% full while the
V100s are 42% full, and why 30x30 is out of reach on this node.

## Stages

`RENO_MS_SHARD_STAGE` selects how much is distributed. Every stage is a superset
of the one below it, and all four reproduce the single-process result.

| stage | what is distributed | what is still replicated |
|---|---|---|
| 1 | Hamiltonian pairs, Krylov vectors | environments, MPOs, MPS, `Model` setup |
| 2 | + environments and pair groups, by α | MPS, `Model` setup |
| 3 | + the MPS itself, by α with ghost rows | `Model`/`Mpo` setup |
| 4 | + `Model`/`Mpo` construction, by α | — |

The α (electronic-row) partition is contiguous and identical in every patch —
see `mpi_common.AlphaLayout`. Because a 2D nearest-neighbour coupling reaches
α ± 1 and α ± NCOL, a rank needs a halo of NCOL rows on each side of its own
range; those ghost rows are refreshed one site at a time during the TDVP sweep.

## Library changes this depends on

In `renormalizer/` (they are correctness-neutral; a 3×3 check reproduces the
previous results to 7e-14):

- `multiset_model.py`: empty `(α, β)` blocks are no longer real `Model` objects
  (`EMPTY_MS_BLOCK`) — at 27×27 that alone saves ≈52 GiB and ≈266 s of setup;
  lazy `conj_mps` construction; `scatter_add` instead of a one-hot matmul;
  a scalar `_energy_offset` instead of rebuilding N²ₑ MPOs.
- `lib/krylov/krylov.py`: bounded Krylov basis growth.

## Running

```bash
sbatch scaling/submit.slurm                                  # default ladder at stage 4
sbatch --export=ALL,SIZES="27",NSTEPS=100 scaling/submit.slurm
```

### Measuring host memory without a GPU

The 4V100 partition is all-or-nothing, so when another job holds even one of its
GPUs there is no way to run there at all. `SETUP_ONLY` lets the host-side
question — *does this lattice fit in RAM?* — be answered on an idle CPU node
meanwhile:

| `SETUP_ONLY` | stops after | measures |
|---|---|---|
| `0` | nothing (full run) | everything |
| `1` | model, MPOs, initial state | setup only |
| `2` | + the per-pair environment cache | setup + the dominant host term |

Level 1 alone is misleading. `conj_mps` is built lazily, so the environment
cache is not allocated until the first sweep — and that cache is both the
largest host term and the one the α-sharding exists to divide. Level 2 builds it
through the same `_get_or_build_environ_list` the sweep uses, so the sharding
applies, at a small fraction of a TDVP step (a single sweep at 15×15 on NumPy
does not finish in 45 minutes; the cache build takes minutes).

```bash
sbatch --export=ALL,SIZES="27 26 25",STAGE=4,SETUP_ONLY=2,STOP_ON_SUCCESS=1 \
  scaling/submit_setup_cpu.slurm
```

`STOP_ON_SUCCESS=1` pairs with a *descending* size list: the largest lattice is
tried first, so a run that works answers the question immediately instead of
climbing through sizes that were never in doubt.

Locally (one GPU, oversubscribed — for correctness only, not for timing):

```bash
cd code
NROW=3 NCOL=3 NU_MAX=4 MAX_BONDDIM=16 NSTEPS=3 \
  RENO_MS_SHARD_STAGE=4 OUTPUT_DIR=/tmp/out \
  mpirun -n 4 --oversubscribe python -u run_dynamics.py
```
