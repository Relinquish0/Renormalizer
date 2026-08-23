# -*- coding: utf-8 -*-

"""Memory accounting for the single-GPU multiset runs.

The single-GPU question is not just "how big can it get" but "what is holding
the memory", so this module reports the three terms separately:

* the process high-water mark, which is what the 1500 GiB node limit applies to;
* the multiset state itself, one ``Mps`` per electronic site, reported per set
  because that is the unit the ansatz is built from;
* the per-pair ``Environ`` cache, which is normally the largest single term.

Every array is classified by where it actually lives.  Measured on this build,
both the MPS tensors and the environment tensors stay as NumPy arrays on the
host even when ``USE_GPU`` is true -- the device only ever holds the transient
working set of the current site contraction and the Krylov basis.  That is why
the 4-GPU 27x27 run sat at 286 GiB of host per rank against 11.4 GiB of device,
and it is the single most important fact for predicting what fits.  The split is
computed rather than assumed, so if that ever changes the numbers will say so.
"""

import resource

GIB = 1024.0**3


def host_peak_gb():
    """Peak resident set size of this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)


def host_rss_gb():
    """Current resident set size, in GiB (0.0 where /proc is unavailable)."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0**2)
    except OSError:
        pass
    return 0.0


def gpu_report():
    """(pool_used, pool_total, device_used, device_total) in GiB, or zeros."""
    try:
        import cupy
    except ImportError:
        return (0.0, 0.0, 0.0, 0.0)
    pool = cupy.get_default_memory_pool()
    try:
        free, total = cupy.cuda.runtime.memGetInfo()
        dev_used, dev_total = (total - free) / GIB, total / GIB
    except Exception:
        dev_used = dev_total = 0.0
    return (pool.used_bytes() / GIB, pool.total_bytes() / GIB, dev_used, dev_total)


def _underlying(obj):
    """The raw ndarray behind a renormalizer ``Matrix`` wrapper, if any.

    Classifying the wrapper instead of what it wraps would report every MPS
    tensor as host-resident whatever the backend is doing, which is exactly the
    mistake this split exists to avoid.
    """
    return getattr(obj, "array", obj)


def _is_device_array(array):
    return type(_underlying(array)).__module__.split(".")[0] == "cupy"


def _split_bytes(arrays):
    """Sum array bytes, split into (device, host)."""
    device = host = 0
    for array in arrays:
        nbytes = int(getattr(_underlying(array), "nbytes", 0))
        if _is_device_array(array):
            device += nbytes
        else:
            host += nbytes
    return device, host


def mps_breakdown(ms_mps, alphas=None):
    """Per-set size of a ``MultisetMps``.

    ``msmps`` holds one ``Mps`` per electronic site; they are not all the same
    size, so the maximum matters as much as the mean when predicting the next
    lattice up.

    Under the 4-GPU alpha sharding the list keeps its full length but holds
    ``None`` for rows this rank does not carry, and the rows it does carry are
    its own plus a ghost halo.  Pass ``alphas`` (e.g. the owned range) to count
    each row exactly once when summing across ranks; leave it ``None`` to get
    what this process actually has resident, which is the number its own memory
    footprint is made of.
    """
    per_set = []
    device_total = host_total = 0
    indices = range(len(ms_mps.msmps)) if alphas is None else alphas
    for alpha in indices:
        mps = ms_mps.msmps[alpha]
        if mps is None:
            continue
        device, host = _split_bytes(list(mps))
        per_set.append((device + host) / GIB)
        device_total += device
        host_total += host
    n_sets = len(per_set)
    return {
        "n_sets": n_sets,
        "total_gb": (device_total + host_total) / GIB,
        "device_gb": device_total / GIB,
        "host_gb": host_total / GIB,
        "per_set_mean_gb": (sum(per_set) / n_sets) if n_sets else 0.0,
        "per_set_max_gb": max(per_set) if per_set else 0.0,
        "per_set_min_gb": min(per_set) if per_set else 0.0,
    }


def environ_breakdown(ms_model):
    """Size of the cached per-pair environments, if one has been built yet.

    Returns ``None`` before the first sweep: ``conj_mps`` is built lazily, so
    asking right after setup would report zero and read as "the environments are
    free" rather than "they do not exist yet".
    """
    cache = getattr(ms_model, "_environ_cache", None)
    if not cache or not cache.get("envs"):
        return None
    device_total = host_total = 0
    n_environ = 0
    for environ in cache["envs"]:
        if environ is None:
            continue
        n_environ += 1
        device, host = _split_bytes(environ._virtual_disk.values())
        device_total += device
        host_total += host
    return {
        "n_environ": n_environ,
        "total_gb": (device_total + host_total) / GIB,
        "device_gb": device_total / GIB,
        "host_gb": host_total / GIB,
        "per_pair_mean_gb": (
            (device_total + host_total) / GIB / n_environ if n_environ else 0.0
        ),
    }


def snapshot(phase, ms_mps=None, ms_model=None, owned_alphas=None):
    """One labelled record of every term, ready to be logged and serialised."""
    pool_used, pool_total, dev_used, dev_total = gpu_report()
    record = {
        "phase": phase,
        "host_peak_gb": host_peak_gb(),
        "host_rss_gb": host_rss_gb(),
        "gpu_pool_used_gb": pool_used,
        "gpu_pool_total_gb": pool_total,
        "gpu_device_used_gb": dev_used,
        "gpu_device_total_gb": dev_total,
    }
    if ms_mps is not None:
        record["mps"] = mps_breakdown(ms_mps)
        if owned_alphas is not None:
            # Resident rows include the ghost halo, which belongs to another
            # rank; the owned-only figure is the one that sums to the whole
            # multiset state without counting a row twice.
            record["mps_owned"] = mps_breakdown(ms_mps, alphas=owned_alphas)
    if ms_model is not None:
        environ = environ_breakdown(ms_model)
        if environ is not None:
            record["environ"] = environ
    return record


def format_snapshot(record):
    """One-line summary for the log."""
    parts = [
        f"host peak {record['host_peak_gb']:.1f} GiB",
        f"(rss {record['host_rss_gb']:.1f})",
        f"GPU {record['gpu_device_used_gb']:.2f}/{record['gpu_device_total_gb']:.2f} GiB",
        f"pool {record['gpu_pool_total_gb']:.2f} GiB",
    ]
    mps = record.get("mps")
    if mps:
        parts.append(
            f"MPS {mps['total_gb']:.2f} GiB over {mps['n_sets']} sets "
            f"(mean {mps['per_set_mean_gb'] * 1024:.1f} MiB, "
            f"max {mps['per_set_max_gb'] * 1024:.1f} MiB, "
            f"{mps['device_gb']:.2f} on device / {mps['host_gb']:.2f} on host)"
        )
    environ = record.get("environ")
    if environ:
        parts.append(
            f"environ {environ['total_gb']:.2f} GiB over {environ['n_environ']} pairs "
            f"({environ['device_gb']:.2f} on device / {environ['host_gb']:.2f} on host)"
        )
    return f"[mem/{record['phase']}] " + "; ".join(parts)
