#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run one complete single-GPU multiset calculation per Slurm task.

This is deliberately an outer, coarse-grained parallel entry point.  It does
not import mpi4py and does not install any of the MPI monkey patches in this
directory.  After selecting one parameter tuple for the current task, it runs
the sibling ``multiset/Holstein1515.py`` script unchanged.

``INITIAL_SITES`` is required.  Every comma-separated ``*_VALUES`` variable
that is present must contain exactly one value per task; scalar variables such
as ``G`` remain available for parameters shared by all tasks.
"""

import json
import math
import os
from pathlib import Path
import re
import runpy


LIST_PARAMETERS = {
    "NROW_VALUES": ("NROW", int),
    "NCOL_VALUES": ("NCOL", int),
    "OMEGA_0_VALUES": ("OMEGA_0", float),
    "J_VALUES": ("J", float),
    "G_VALUES": ("G", float),
    "NU_MAX_VALUES": ("NU_MAX", int),
    "MAX_BONDDIM_VALUES": ("MAX_BONDDIM", int),
    "EVOLVE_DT_VALUES": ("EVOLVE_DT", float),
    "N_SNAPSHOTS_VALUES": ("N_SNAPSHOTS", int),
    "STARTUP_SUBSTEPS_N_VALUES": ("STARTUP_SUBSTEPS_N", int),
}


def _parse_int_env(name, default=None):
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc


def _split_exact(name, task_count, required=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        if required:
            raise ValueError(
                f"{name} is required and must contain exactly {task_count} "
                "comma-separated values"
            )
        return None

    values = [value.strip() for value in raw_value.split(",")]
    if any(value == "" for value in values):
        raise ValueError(f"{name} contains an empty comma-separated value: {raw_value!r}")
    if len(values) != task_count:
        raise ValueError(
            f"{name} must contain exactly {task_count} values (one per task); "
            f"got {len(values)}: {raw_value!r}"
        )
    return values


def _select_value(list_name, scalar_name, converter, rank, task_count, required=False):
    values = _split_exact(list_name, task_count, required=required)
    if values is None:
        return None
    raw_value = values[rank]
    try:
        value = converter(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{list_name}[{rank}] must be a valid {converter.__name__}; got {raw_value!r}"
        ) from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{list_name}[{rank}] must be finite; got {raw_value!r}")
    os.environ[scalar_name] = str(value)
    return value


def _scalar_int(name, default):
    raw_value = os.environ.get(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw_value!r}") from exc


def _scalar_float(name, default):
    raw_value = os.environ.get(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float; got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {raw_value!r}")
    return value


def _safe_label(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return label or "run"


def _configure_gpu(rank):
    local_rank = _parse_int_env("SLURM_LOCALID", default=rank)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_devices = [device.strip() for device in visible.split(",") if device.strip()]
        if not visible_devices:
            raise ValueError("CUDA_VISIBLE_DEVICES does not contain a usable device")
        # Slurm normally exposes one logical device per task.  If the site
        # exposes all allocated GPUs instead, use the unique local task index.
        if len(visible_devices) == 1:
            gpu_id = 0
        elif local_rank < len(visible_devices):
            gpu_id = local_rank
        else:
            raise ValueError(
                f"cannot assign local task {local_rank} a unique GPU from "
                f"CUDA_VISIBLE_DEVICES={visible!r}"
            )
    else:
        gpu_id = local_rank
    os.environ["RENO_GPU"] = str(gpu_id)
    return local_rank, gpu_id, visible


def _validate_selected_parameters(initial_site):
    nrow = _scalar_int("NROW", 15)
    ncol = _scalar_int("NCOL", 15)
    nu_max = _scalar_int("NU_MAX", 8)
    max_bonddim = _scalar_int("MAX_BONDDIM", 16)
    n_snapshots = _scalar_int("N_SNAPSHOTS", 500)
    startup_substeps_n = _scalar_int("STARTUP_SUBSTEPS_N", 10)
    omega_0 = _scalar_float("OMEGA_0", 1.0)
    _scalar_float("J", 1.0)
    _scalar_float("G", 0.5)
    evolve_dt = _scalar_float("EVOLVE_DT", 0.1)

    for name, value in (
        ("NROW", nrow),
        ("NCOL", ncol),
        ("NU_MAX", nu_max),
        ("MAX_BONDDIM", max_bonddim),
        ("N_SNAPSHOTS", n_snapshots),
        ("STARTUP_SUBSTEPS_N", startup_substeps_n),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1; got {value}")
    if omega_0 <= 0:
        raise ValueError(f"OMEGA_0 must be positive; got {omega_0}")
    if evolve_dt == 0 and n_snapshots > 1:
        raise ValueError("EVOLVE_DT must be non-zero when N_SNAPSHOTS is greater than 1")

    site_count = nrow * ncol
    if not 0 <= initial_site < site_count:
        raise ValueError(
            f"INITIAL_SITES selected site {initial_site} for this task, but the valid "
            f"range for a {nrow}x{ncol} lattice is 0..{site_count - 1}"
        )
    return nrow, ncol, max_bonddim


def configure_worker():
    rank = _parse_int_env("SLURM_PROCID", default=_parse_int_env("INDEPENDENT_RANK", 0))
    task_count = _parse_int_env(
        "SLURM_NTASKS", default=_parse_int_env("INDEPENDENT_SIZE", 1)
    )
    if task_count < 1:
        raise ValueError(f"task count must be positive; got {task_count}")
    if not 0 <= rank < task_count:
        raise ValueError(f"task rank {rank} is outside the valid range 0..{task_count - 1}")

    initial_site = _select_value(
        "INITIAL_SITES", "INITIAL_SITE", int, rank, task_count, required=True
    )
    for list_name, (scalar_name, converter) in LIST_PARAMETERS.items():
        _select_value(list_name, scalar_name, converter, rank, task_count)

    labels = _split_exact("RUN_LABELS", task_count)
    label = _safe_label(labels[rank] if labels is not None else f"rank{rank:02d}")
    nrow, ncol, max_bonddim = _validate_selected_parameters(initial_site)
    local_rank, gpu_id, visible = _configure_gpu(rank)

    job_id = _safe_label(os.environ.get("SLURM_JOB_ID", "local"))
    output_dir = Path(
        os.environ.get(
            "INDEPENDENT_OUTPUT_DIR",
            str(Path(__file__).resolve().parent / "independent_results" / job_id),
        )
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"job{job_id}_r{rank:02d}_{label}"
    output_stem = (
        f"H{nrow}{ncol}_m{max_bonddim}_{run_id}_site{initial_site}"
    )
    os.environ["RUN_ID"] = run_id
    os.environ["OUTPUT_NPZ"] = str(output_dir / f"{output_stem}.npz")
    os.environ["OUTPUT_XLSX"] = str(output_dir / f"{output_stem}.xlsx")

    selected = {
        "rank": rank,
        "task_count": task_count,
        "local_rank": local_rank,
        "run_id": run_id,
        "cuda_visible_devices": visible,
        "reno_gpu": gpu_id,
        "initial_site": initial_site,
        "nrow": nrow,
        "ncol": ncol,
        "omega_0": _scalar_float("OMEGA_0", 1.0),
        "j": _scalar_float("J", 1.0),
        "g": _scalar_float("G", 0.5),
        "nu_max": _scalar_int("NU_MAX", 8),
        "max_bonddim": max_bonddim,
        "evolve_dt": _scalar_float("EVOLVE_DT", 0.1),
        "n_snapshots": _scalar_int("N_SNAPSHOTS", 500),
        "startup_substeps": os.environ.get("IF_STARTUP_SUBSTEPS", "1"),
        "startup_substeps_n": _scalar_int("STARTUP_SUBSTEPS_N", 10),
        "output_npz": os.environ["OUTPUT_NPZ"],
        "output_xlsx": os.environ["OUTPUT_XLSX"],
    }
    selected["metadata_json"] = str(Path(selected["output_npz"]).with_suffix(".json"))
    return selected


def main():
    selected = configure_worker()
    with open(selected["metadata_json"], "w", encoding="utf-8") as stream:
        json.dump(selected, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("Independent multiset worker configuration:", flush=True)
    print(json.dumps(selected, indent=2, sort_keys=True), flush=True)

    if os.environ.get("INDEPENDENT_DRY_RUN", "0").lower() in {"1", "true", "yes"}:
        print("INDEPENDENT_DRY_RUN enabled; single-GPU calculation was not started.", flush=True)
        return

    single_gpu_script = Path(__file__).resolve().parent.parent / "multiset" / "Holstein1515.py"
    if not single_gpu_script.is_file():
        raise FileNotFoundError(f"single-GPU driver not found: {single_gpu_script}")
    print(f"Running unpatched single-GPU driver: {single_gpu_script}", flush=True)
    runpy.run_path(str(single_gpu_script), run_name="__main__")


if __name__ == "__main__":
    main()
