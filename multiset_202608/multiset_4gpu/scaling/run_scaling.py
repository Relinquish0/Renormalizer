# -*- coding: utf-8 -*-

"""Size-ladder driver: run ``code/run_dynamics.py`` once per lattice size.

Each size gets its own MPI job so that a rank running out of host or GPU memory
takes down only that rung of the ladder.  Results land in
``scaling/result/<NxN>/`` next to the log the SLURM script tees there.

Usage (inside an allocation, srun available):

    python scaling/run_scaling.py --sizes 15 18 20 22 25 27 --stage 4

Anything not passed on the command line is inherited from the environment, so
the SLURM script can set MAX_BONDDIM/NU_MAX/NSTEPS once for the whole ladder.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE_DIR = os.path.join(ROOT, "code")
RESULT_ROOT = os.path.join(HERE, "result")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=os.environ.get("SIZES", "15 18 20 22 25 27").split(),
        help="square lattice edge lengths, e.g. 15 18 20; 'NxM' is also accepted",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=int(os.environ.get("RENO_MS_SHARD_STAGE", "4")),
        choices=(1, 2, 3, 4),
        help="how much of the state is distributed (see code/mpi_shard_patch.py)",
    )
    parser.add_argument("--ntasks", type=int, default=int(os.environ.get("NTASKS", "4")))
    parser.add_argument("--nsteps", type=int, default=int(os.environ.get("NSTEPS", "5")))
    parser.add_argument(
        "--max-bonddim", type=int, default=int(os.environ.get("MAX_BONDDIM", "64"))
    )
    parser.add_argument("--nu-max", type=int, default=int(os.environ.get("NU_MAX", "8")))
    parser.add_argument(
        "--evolve-dt", type=float, default=float(os.environ.get("EVOLVE_DT", "0.1"))
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("PER_SIZE_TIMEOUT", "14400")),
        help="wall-clock budget per size, in seconds",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=os.environ.get("STOP_ON_FAILURE", "1") not in {"0", "false", ""},
        help="stop the ladder at the first size that fails (the usual intent: "
        "the first failure *is* the answer)",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        default=os.environ.get("STOP_ON_SUCCESS", "0") not in {"0", "false", ""},
        help="stop at the first size that succeeds.  Pair this with a descending "
        "size list to find the ceiling from above: the biggest lattice is tried "
        "first, so a run that works answers the question immediately instead of "
        "spending hours climbing through sizes that were never in doubt.",
    )
    parser.add_argument("--label", default=os.environ.get("RUN_LABEL", ""))
    return parser.parse_args(argv)


def normalise_size(token):
    token = str(token).lower()
    if "x" in token:
        nrow, ncol = token.split("x", 1)
        return int(nrow), int(ncol)
    return int(token), int(token)


def has_gpu_allocation():
    """True when SLURM actually gave this job GPUs.

    The CPU-partition host-memory probe runs the very same driver, and asking
    srun for --gpus-per-task there fails the step outright with "Invalid generic
    resource (gres) specification".
    """
    if os.environ.get("RENO_NO_GPU", "0").strip().lower() not in {"0", "false", "no", ""}:
        return False
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_JOB_GPUS", "GPU_DEVICE_ORDINAL"):
        value = os.environ.get(key)
        if value and value.strip() not in {"0", ""}:
            return True
    gres = os.environ.get("SLURM_JOB_GRES", "")
    return "gpu" in gres


def launchers(ntasks):
    """Launch commands to try in order.

    Individual compute nodes turn up with a broken PMIx plugin ("can not load
    PMIx library"), which kills the step before Python ever starts.  That is a
    property of the node, not of the run, so fall back to pmi2 and then to a
    plain mpirun instead of losing the rung.
    """
    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        options = []
        for mpi_type in ("pmix", "pmi2"):
            cmd = ["srun", f"--mpi={mpi_type}", f"--ntasks={ntasks}", "--cpu-bind=cores"]
            if has_gpu_allocation():
                cmd[3:3] = ["--gpus-per-task=1", "--gpu-bind=single:1"]
            options.append(cmd)
        options.append(["mpirun", "-n", str(ntasks)])
        return options
    return [["mpirun", "-n", str(ntasks), "--oversubscribe"]]


LAUNCH_FAILURE_MARKERS = (
    "can not load PMIx library",
    "Unable to load any plugin",
    "Invalid MPI type",
    "Cannot create context for mpi",
    "Unable to create step",
)


def launch_failed_before_start(log_path, offset=0):
    """True when srun rejected the step itself, so a different launcher may work.

    ``offset`` skips whatever earlier attempts already wrote, so a stale marker
    from attempt 1 cannot make attempt 2 look like a launcher problem.
    """
    try:
        with open(log_path, errors="replace") as handle:
            handle.seek(offset)
            text = handle.read(20000)
    except OSError:
        return False
    return any(marker in text for marker in LAUNCH_FAILURE_MARKERS)


def run_one(nrow, ncol, args):
    tag = f"{nrow}x{ncol}"
    outdir = os.path.join(RESULT_ROOT, tag)
    os.makedirs(outdir, exist_ok=True)

    env = dict(os.environ)
    env.update(
        NROW=str(nrow),
        NCOL=str(ncol),
        NSTEPS=str(args.nsteps),
        MAX_BONDDIM=str(args.max_bonddim),
        NU_MAX=str(args.nu_max),
        EVOLVE_DT=str(args.evolve_dt),
        RENO_MS_SHARD_STAGE=str(args.stage),
        OUTPUT_DIR=outdir,
        RUN_LABEL=args.label,
        PYTHONUNBUFFERED="1",
        PYTHONPATH=CODE_DIR + os.pathsep + env.get("PYTHONPATH", ""),
    )

    driver = [sys.executable, "-u", os.path.join(CODE_DIR, "run_dynamics.py")]
    log_path = os.path.join(outdir, f"run{args.label}.log")
    print(f"=== {tag} ({nrow * ncol} sites), stage {args.stage} -> {log_path}", flush=True)

    started = time.time()
    options = launchers(args.ntasks)
    for attempt, prefix in enumerate(options):
        cmd = prefix + driver
        mode = "w" if attempt == 0 else "a"
        offset = os.path.getsize(log_path) if attempt else 0
        with open(log_path, mode) as log:
            log.write(f"# {' '.join(cmd)}\n")
            log.flush()
            try:
                code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.timeout, cwd=CODE_DIR)
                status = "ok" if code == 0 else f"exit {code}"
            except subprocess.TimeoutExpired:
                code, status = 124, f"timeout after {args.timeout}s"
        if code == 0 or attempt == len(options) - 1:
            break
        if not launch_failed_before_start(log_path, offset):
            break
        print(f"    launcher {prefix[0]} {prefix[1]} could not start the step; retrying",
              flush=True)
    elapsed = time.time() - started
    print(f"=== {tag}: {status} in {elapsed:.0f}s", flush=True)
    return {
        "tag": tag,
        "nrow": nrow,
        "ncol": ncol,
        "n_sites": nrow * ncol,
        "exit_code": code,
        "status": status,
        "seconds": elapsed,
        "log": log_path,
        "failure_reason": None if code == 0 else classify_failure(log_path),
    }


OOM_MARKERS = (
    ("gpu_oom", ("cudaErrorMemoryAllocation", "OutOfMemoryError", "out of memory")),
    ("host_oom", ("MemoryError", "Cannot allocate memory", "oom-kill", "Killed")),
    ("mpi_abort", ("MPI_ABORT", "PMIX", "srun: error")),
)


def classify_failure(log_path):
    """Best-effort tag for why a rung failed, from the tail of its log."""
    try:
        with open(log_path, errors="replace") as handle:
            tail = handle.read()[-200000:]
    except OSError:
        return "unreadable log"
    for name, markers in OOM_MARKERS:
        if any(marker in tail for marker in markers):
            return name
    return "unknown"


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    results = []
    for token in args.sizes:
        nrow, ncol = normalise_size(token)
        record = run_one(nrow, ncol, args)
        results.append(record)
        with open(os.path.join(RESULT_ROOT, f"ladder{args.label}.json"), "w") as handle:
            json.dump({"stage": args.stage, "ntasks": args.ntasks,
                       "max_bonddim": args.max_bonddim, "nu_max": args.nu_max,
                       "nsteps": args.nsteps, "runs": results}, handle, indent=2)
        if record["exit_code"] == 0 and args.stop_on_success:
            print(f"stopping ladder: {record['tag']} succeeded", flush=True)
            break
        if record["exit_code"] != 0 and args.stop_on_failure and not args.stop_on_success:
            print(f"stopping ladder: {record['tag']} failed ({record['failure_reason']})",
                  flush=True)
            break

    ok = [r for r in results if r["exit_code"] == 0]
    print("\n=== ladder summary ===", flush=True)
    for record in results:
        print(f"  {record['tag']:>7}  {record['n_sites']:>5} sites  "
              f"{record['status']:<24} {record['seconds']:7.0f}s", flush=True)
    if ok:
        best = max(ok, key=lambda r: r["n_sites"])
        print(f"largest completed lattice: {best['tag']} ({best['n_sites']} sites)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
