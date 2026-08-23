# -*- coding: utf-8 -*-

"""Size-ladder driver: run ``code/run_dynamics.py`` once per lattice size.

The single-GPU counterpart of ``multiset_4gpu/scaling/run_scaling.py``, minus
the MPI launcher: one process, one device, one lattice per subprocess.  Giving
each size its own process matters here, because the interesting outcome is the
one that dies -- a rung that exhausts the 32 GiB device or the host must not
take the rest of the ladder with it.

Usage (inside an allocation holding one GPU):

    python scaling/run_scaling.py --sizes 15 18 20 22 24 25 26 27

Anything not passed on the command line is inherited from the environment, so
the SLURM script can set MAX_BONDDIM/NU_MAX/NSTEPS once for the whole ladder.
"""

import argparse
import json
import os
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
        default=os.environ.get("SIZES", "15 18 20 22 24 25 26 27").split(),
        help="square lattice edge lengths, e.g. 15 18 20; 'NxM' is also accepted",
    )
    parser.add_argument("--nsteps", type=int, default=int(os.environ.get("NSTEPS", "2")))
    parser.add_argument(
        "--max-bonddim", type=int, default=int(os.environ.get("MAX_BONDDIM", "64"))
    )
    parser.add_argument("--nu-max", type=int, default=int(os.environ.get("NU_MAX", "8")))
    parser.add_argument(
        "--evolve-dt", type=float, default=float(os.environ.get("EVOLVE_DT", "0.1"))
    )
    parser.add_argument(
        "--setup-only",
        default=os.environ.get("SETUP_ONLY", "0"),
        help="0 full run, 1 stop after setup, 2 also build the environment cache",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("PER_SIZE_TIMEOUT", "21600")),
        help="wall-clock budget per size, in seconds",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=os.environ.get("STOP_ON_FAILURE", "1") not in {"0", "false", ""},
        help="stop the ladder at the first size that fails.  With an ascending "
        "size list the first failure is the answer, so continuing past it only "
        "burns the allocation on sizes that cannot fit either.",
    )
    parser.add_argument("--label", default=os.environ.get("RUN_LABEL", ""))
    return parser.parse_args(argv)


def normalise_size(token):
    token = str(token).lower()
    if "x" in token:
        nrow, ncol = token.split("x", 1)
        return int(nrow), int(ncol)
    return int(token), int(token)


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
        SETUP_ONLY=str(args.setup_only),
        OUTPUT_DIR=outdir,
        RUN_LABEL=args.label,
        PYTHONUNBUFFERED="1",
        PYTHONPATH=CODE_DIR + os.pathsep + env.get("PYTHONPATH", ""),
    )

    cmd = [sys.executable, "-u", os.path.join(CODE_DIR, "run_dynamics.py")]
    log_path = os.path.join(outdir, f"run{args.label}.log")
    print(f"=== {tag} ({nrow * ncol} sites), 1 GPU -> {log_path}", flush=True)

    started = time.time()
    with open(log_path, "w") as log:
        log.write(f"# {' '.join(cmd)}\n")
        log.flush()
        try:
            code = subprocess.call(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                cwd=CODE_DIR,
            )
            status = "ok" if code == 0 else f"exit {code}"
        except subprocess.TimeoutExpired:
            code, status = 124, f"timeout after {args.timeout}s"
    elapsed = time.time() - started
    print(f"=== {tag}: {status} in {elapsed:.0f}s", flush=True)

    record = {
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
    record.update(harvest_memory(outdir, tag, args.label))
    return record


OOM_MARKERS = (
    (
        "gpu_oom",
        (
            "cudaErrorMemoryAllocation",
            "OutOfMemoryError",
            "cupy.cuda.memory.OutOfMemoryError",
            "out of memory",
        ),
    ),
    ("host_oom", ("MemoryError", "Cannot allocate memory", "oom-kill", "Killed")),
)


def classify_failure(log_path):
    """Best-effort tag for why a rung failed, from the tail of its log.

    Device and host exhaustion are checked in that order: a CuPy allocation
    failure raises OutOfMemoryError, while the host running out shows up as a
    MemoryError or as the kernel's OOM killer, and only the second one means the
    1500 GiB request was the limit.
    """
    try:
        with open(log_path, errors="replace") as handle:
            tail = handle.read()[-200000:]
    except OSError:
        return "unreadable log"
    for name, markers in OOM_MARKERS:
        if any(marker in tail for marker in markers):
            return name
    return "unknown"


def harvest_memory(outdir, tag, label):
    """Lift the headline memory numbers out of a rung's summary, if it wrote one.

    Keeping them in the ladder file means the whole scaling curve can be read
    without opening eight per-size JSONs.
    """
    for name in (f"{tag}{label}_summary.json", f"{tag}{label}_setup_summary.json"):
        path = os.path.join(outdir, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                summary = json.load(handle)
        except (OSError, ValueError):
            continue
        records = summary.get("memory") or []
        if not records:
            continue
        last = records[-1]
        out = {
            "host_peak_gb": last.get("host_peak_gb"),
            "gpu_device_used_gb": last.get("gpu_device_used_gb"),
            "gpu_pool_total_gb": last.get("gpu_pool_total_gb"),
            "setup_seconds": summary.get("setup_seconds"),
            "seconds_per_step": summary.get("seconds_per_step"),
        }
        if last.get("mps"):
            out["mps_total_gb"] = last["mps"]["total_gb"]
            out["mps_per_set_mean_gb"] = last["mps"]["per_set_mean_gb"]
            out["mps_per_set_max_gb"] = last["mps"]["per_set_max_gb"]
        if last.get("environ"):
            out["environ_total_gb"] = last["environ"]["total_gb"]
        return out
    return {}


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    results = []
    for token in args.sizes:
        nrow, ncol = normalise_size(token)
        record = run_one(nrow, ncol, args)
        results.append(record)
        with open(os.path.join(RESULT_ROOT, f"ladder{args.label}.json"), "w") as handle:
            json.dump(
                {
                    "n_gpu": 1,
                    "max_bonddim": args.max_bonddim,
                    "nu_max": args.nu_max,
                    "nsteps": args.nsteps,
                    "setup_only": args.setup_only,
                    "runs": results,
                },
                handle,
                indent=2,
            )
        if record["exit_code"] != 0 and args.stop_on_failure:
            print(
                f"stopping ladder: {record['tag']} failed ({record['failure_reason']})",
                flush=True,
            )
            break

    print("\n=== ladder summary ===", flush=True)
    for record in results:
        extra = ""
        if record.get("host_peak_gb") is not None:
            extra = (
                f"  host {record['host_peak_gb']:6.1f} GiB"
                f"  GPU {record.get('gpu_device_used_gb', 0):5.2f} GiB"
            )
        print(
            f"  {record['tag']:>7}  {record['n_sites']:>5} sites  "
            f"{record['status']:<24} {record['seconds']:7.0f}s{extra}",
            flush=True,
        )
    ok = [r for r in results if r["exit_code"] == 0]
    if ok:
        best = max(ok, key=lambda r: r["n_sites"])
        print(
            f"largest completed lattice: {best['tag']} ({best['n_sites']} sites)",
            flush=True,
        )
    else:
        print("no lattice completed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
