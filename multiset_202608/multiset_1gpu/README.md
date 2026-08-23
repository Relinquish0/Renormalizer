# Single-V100 multiset MPS — 2D Holstein scaling

The one-GPU counterpart of `../multiset_4gpu`, built to answer a single
question: with one V100 instead of four, how large a 2D Holstein lattice can the
multiset ansatz still evolve at bond dimension 64, and what runs out first?

The physics is identical to the 4-GPU ladder — `code/holstein.py` is a verbatim
copy — so any difference between the two ladders is the parallelism, not the
model. The driver follows
`multiset_202607/2DHolstein1515_speed/multiset/Holstein1515.py`: same
environment-variable configuration, same lightweight observable set.

```
ω₀ = J = 1,  g = 0.5,  ν_max = 8,  open boundaries,  T = 0 K
m = 64,  dt = 0.1,  initial excitation at the lattice centre
```

## Layout

```
code/
  holstein.py        2D Holstein model builder + run configuration (shared with multiset_4gpu)
  memory.py          host / device / per-set accounting
  run_dynamics.py    the driver: one lattice, one process, one GPU
scaling/
  run_scaling.py     runs code/run_dynamics.py once per lattice size
  submit.slurm       1×V100 batch job, --mem=1500G
  result/<NxN>/      per-size logs, populations (.npz) and summary (.json)
plot/
```

## Where the memory actually lives

This is worth stating up front because it is measured, not assumed, and it is
counter-intuitive: **with `USE_GPU` true, the MPS tensors and the environment
cache are still NumPy arrays on the host.** The device only ever holds the
transient working set — the current site's contractions and the Krylov basis.

`code/memory.py` therefore reports three things separately, and classifies every
array by where it really is rather than by which backend is active:

| term | reported as | why it matters |
|---|---|---|
| process high-water mark | `host_peak_gb` | what the 1500 GiB node limit applies to |
| the multiset state | `mps.total_gb`, plus per-set mean/max over the N_e sets | the ansatz is N_e separate MPSs; the per-set number is the unit that scales |
| the per-pair environments | `environ.total_gb` | normally the largest single term |
| device | `gpu_device_used_gb`, `gpu_pool_total_gb` | the 32 GiB ceiling |

The environment cache is built lazily (`conj_mps` is only materialised on the
first sweep), so a measurement taken right after setup reports no environments
at all. `SETUP_ONLY=2` builds it explicitly for host-side probing:

| `SETUP_ONLY` | stops after | measures |
|---|---|---|
| `0` | nothing (full run) | everything |
| `1` | model, MPOs, initial state | setup only |
| `2` | + the per-pair environment cache | setup + the dominant host term |

## Result

The ceiling is **21x21 (441 sites)**, and it is set by the 32 GiB device, not by
the host. Measured (m = 64, nu_max = 8, 2 steps per rung):

| lattice | host peak | MPS total | per set | environments | device |
|---|---|---|---|---|---|
| 15x15 | 89.3 GiB | 24.31 GiB / 225 sets | 110.6 MiB | 34.96 GiB / 1065 pairs | 10.91 GiB |
| 18x18 | 188.8 GiB | 50.67 GiB / 324 sets | 160.1 MiB | 73.36 GiB / 1548 pairs | 15.85 GiB |
| 20x20 | 289.8 GiB | 77.39 GiB / 400 sets | 198.1 MiB | 112.44 GiB / 1920 pairs | 27.46 GiB |
| **21x21** | **353.0 GiB** | **94.15 GiB / 441 sets** | **218.6 MiB** | **136.99 GiB / 2121 pairs** | **30.38 / 31.73 GiB** |
| 22x22 | — | — | — | — | needs 35.4 GB: **OOM** |

22x22 gets through setup on 72.7 GiB of host and then dies in the first sweep:
`OutOfMemoryError: Out of memory allocating 10,150,215,680 bytes (allocated so
far: 25,283,090,432 bytes)` -- 35.4 GB against the V100's 31.7 GB.

The host never came close: 353 GiB of 1500 GiB at the ceiling, 24%. Every extra
GiB of host RAM on this node buys nothing for a single GPU.

For comparison, 4 GPUs reach **29x29 (841 sites)** -- 1.91x the sites -- and
there the binding constraint is reversed: 1453.6 GiB of host (96.9% of the node)
against 13.27 GiB per device (42%).

## Running

```bash
sbatch scaling/submit.slurm                                    # default ladder
sbatch --export=ALL,SIZES="15 18 20",NSTEPS=5 scaling/submit.slurm
```

`--qos=normal` is what makes a one-GPU allocation possible on this partition:
the `4gpu` QOS is all-four-or-nothing (a 3-GPU job is cancelled at start),
while `normal` caps the request at `gres/gpu=1`.

The ladder climbs and stops at the first size that fails, because with an
ascending list that failure *is* the answer. `run_scaling.py` tags the cause as
`gpu_oom` or `host_oom` from the log, which is the distinction the whole
exercise is about.

Locally, without SLURM:

```bash
cd code
NROW=5 NCOL=5 NU_MAX=4 MAX_BONDDIM=16 NSTEPS=1 OUTPUT_DIR=/tmp/out \
  python -u run_dynamics.py
```
