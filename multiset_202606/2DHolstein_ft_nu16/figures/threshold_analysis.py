"""RMSD convergence and failure-time analysis for the 2D Holstein data.

The module deliberately has no dependency on Renormalizer itself; the saved NPZ
files contain everything needed for the analysis.  ``t_epsilon`` is the first
linearly interpolated time at which the absolute difference between two 2D
RMSD curves exceeds ``epsilon``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TEMPERATURES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
FAIL_EPSILONS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


@dataclass(frozen=True)
class Comparison:
    method: str
    chi: int
    reference_chi: int
    reference_method: str = "multiset"


# All curves use the most accurate available result, multiset chi=32, as the
# common physical reference.  This also lets the sweep test different chi and
# epsilon choices without changing the target solution.
COMPARISONS = (
    Comparison("singleset", 16, 32),
    Comparison("singleset", 32, 32),
    Comparison("singleset", 64, 32),
    Comparison("multiset", 8, 32),
    Comparison("multiset", 16, 32),
)


def result_path(base_dir: Path, temperature: float, method: str, chi: int) -> Path:
    tag = f"{temperature:g}"
    return (
        Path(base_dir)
        / f"Holstein77_T{tag}"
        / method
        / f"Holstein77_T{tag}_{chi}bd_{method}.npz"
    )


def rmsd_2d(occupations: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    """Return sqrt(<r^2>) about the geometrical lattice centre."""
    occupations = np.asarray(occupations, dtype=float)
    if occupations.ndim != 2 or occupations.shape[1] != nrow * ncol:
        raise ValueError(
            f"expected occupations with shape (n_time, {nrow * ncol}), "
            f"got {occupations.shape}"
        )

    norm = occupations.sum(axis=1, keepdims=True)
    if np.any(np.abs(norm) <= 1e-14):
        raise ValueError("occupation normalization is zero at one or more times")
    density = occupations / norm

    x, y = np.indices((nrow, ncol), dtype=float)
    radius_squared = (
        (x - (nrow - 1) / 2) ** 2 + (y - (ncol - 1) / 2) ** 2
    ).reshape(-1)
    mean_radius_squared = density @ radius_squared
    # Guard only against harmless round-off immediately below zero.
    return np.sqrt(np.maximum(mean_radius_squared, 0.0))


def load_rmsd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        time = np.asarray(data["time_series"], dtype=float)
        occupations = np.asarray(data["e_occupations"], dtype=float)
        nrow, ncol = int(data["nrow"]), int(data["ncol"])
    curve = rmsd_2d(occupations, nrow, ncol)
    n = min(len(time), len(curve))
    return time[:n], curve[:n]


def aligned_error(
    time_a: np.ndarray,
    curve_a: np.ndarray,
    time_b: np.ndarray,
    curve_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute curve error on the overlapping time grid of curve A."""
    time_a = np.asarray(time_a, dtype=float)
    curve_a = np.asarray(curve_a, dtype=float)
    time_b = np.asarray(time_b, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    if any(array.ndim != 1 for array in (time_a, curve_a, time_b, curve_b)):
        raise ValueError("time axes and RMSD curves must all be one-dimensional")
    if len(time_a) != len(curve_a) or len(time_b) != len(curve_b):
        raise ValueError("each time axis must have the same length as its curve")
    if np.any(np.diff(time_a) <= 0) or np.any(np.diff(time_b) <= 0):
        raise ValueError("time axes must be strictly increasing")

    overlap = (time_a >= time_b[0]) & (time_a <= time_b[-1])
    time = time_a[overlap]
    if not len(time):
        raise ValueError("the two time axes do not overlap")
    reference = np.interp(time, time_b, curve_b)
    return time, np.abs(curve_a[overlap] - reference)


def first_threshold_crossing(
    time: np.ndarray, error: np.ndarray, epsilon: float
) -> tuple[float, bool]:
    """Return first interpolated ``error > epsilon`` time and censor flag.

    If no crossing is observed, the final sampled time is returned and
    ``censored`` is True.  This makes the finite observation window explicit.
    """
    time = np.asarray(time, dtype=float)
    error = np.asarray(error, dtype=float)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if time.ndim != 1 or error.ndim != 1 or len(time) != len(error) or not len(time):
        raise ValueError("time and error must be non-empty 1D arrays of equal length")

    indices = np.flatnonzero(error > epsilon)
    if not len(indices):
        return float(time[-1]), True
    index = int(indices[0])
    if index == 0:
        return float(time[0]), False

    t0, t1 = time[index - 1], time[index]
    e0, e1 = error[index - 1], error[index]
    if np.isclose(e1, e0):
        return float(t1), False
    fraction = np.clip((epsilon - e0) / (e1 - e0), 0.0, 1.0)
    return float(t0 + fraction * (t1 - t0)), False


def comparison_error(
    base_dir: Path, temperature: float, comparison: Comparison
) -> tuple[np.ndarray, np.ndarray]:
    time, curve = load_rmsd(
        result_path(base_dir, temperature, comparison.method, comparison.chi)
    )
    reference_time, reference_curve = load_rmsd(
        result_path(
            base_dir,
            temperature,
            comparison.reference_method,
            comparison.reference_chi,
        )
    )
    return aligned_error(time, curve, reference_time, reference_curve)


def threshold_sweep(
    base_dir: Path,
    temperatures: Iterable[float] = TEMPERATURES,
    comparisons: Iterable[Comparison] = COMPARISONS,
    epsilons: Iterable[float] = FAIL_EPSILONS,
) -> pd.DataFrame:
    rows = []
    for comparison in comparisons:
        for temperature in temperatures:
            time, error = comparison_error(base_dir, temperature, comparison)
            for epsilon in epsilons:
                t_epsilon, censored = first_threshold_crossing(time, error, epsilon)
                rows.append(
                    {
                        "temperature": temperature,
                        "method": comparison.method,
                        "chi": comparison.chi,
                        "reference_method": comparison.reference_method,
                        "reference_chi": comparison.reference_chi,
                        "epsilon": epsilon,
                        "t_epsilon": t_epsilon,
                        "censored": censored,
                        "max_error": float(np.max(error)),
                        "t_max": float(time[-1]),
                    }
                )
    return pd.DataFrame(rows)


def convergence_audit(
    base_dir: Path,
    fail_epsilon: float,
    convergence_epsilon: float,
    temperatures: Iterable[float] = TEMPERATURES,
) -> pd.DataFrame:
    """Audit the common-reference method-separation criterion.

    At the interpolated singleset-16 failure time relative to multiset-32, the
    function evaluates the multiset-16 error against the same reference.  A
    threshold pair is accepted only if multiset remains below
    ``convergence_epsilon`` for every temperature.
    """
    rows = []
    singleset = Comparison("singleset", 16, 32)
    multiset = Comparison("multiset", 16, 32)
    for temperature in temperatures:
        single_time, single_error = comparison_error(
            base_dir, temperature, singleset
        )
        multi_time, multi_error = comparison_error(base_dir, temperature, multiset)
        t_epsilon, censored = first_threshold_crossing(
            single_time, single_error, fail_epsilon
        )
        multiset_error_at_failure = float(
            np.interp(t_epsilon, multi_time, multi_error)
        )
        rows.append(
            {
                "temperature": temperature,
                "fail_epsilon": fail_epsilon,
                "convergence_epsilon": convergence_epsilon,
                "singleset_t_epsilon": t_epsilon,
                "multiset_error_at_singleset_t_epsilon": multiset_error_at_failure,
                "multiset_converged": (
                    not censored
                    and multiset_error_at_failure < convergence_epsilon
                ),
            }
        )
    return pd.DataFrame(rows)


def interval_error_audit(
    base_dir: Path,
    comparison: Comparison,
    epsilon: float,
    t_start: float,
    t_stop: float,
    temperatures: Iterable[float] = TEMPERATURES,
) -> pd.DataFrame:
    """Audit whether a comparison stays within epsilon on a time interval."""
    if t_stop < t_start:
        raise ValueError("t_stop must not be smaller than t_start")
    rows = []
    for temperature in temperatures:
        time, error = comparison_error(base_dir, temperature, comparison)
        mask = (time >= t_start) & (time <= t_stop)
        interval_time, interval_error = time[mask], error[mask]
        if not len(interval_time):
            raise ValueError(
                f"no sampled times in [{t_start}, {t_stop}] at T*={temperature:g}"
            )
        maximum_index = int(np.argmax(interval_error))
        crossing, censored = first_threshold_crossing(
            interval_time, interval_error, epsilon
        )
        rows.append(
            {
                "temperature": temperature,
                "method": comparison.method,
                "chi": comparison.chi,
                "reference_method": comparison.reference_method,
                "reference_chi": comparison.reference_chi,
                "epsilon": epsilon,
                "t_start": t_start,
                "t_stop": t_stop,
                "max_error": float(interval_error[maximum_index]),
                "time_of_max_error": float(interval_time[maximum_index]),
                "first_crossing": np.nan if censored else crossing,
                "within_tolerance": bool(np.all(interval_error <= epsilon)),
            }
        )
    return pd.DataFrame(rows)
