# -*- coding: utf-8 -*-
"""
Test script for profiling _apply_block_operator performance
Created to benchmark different optimization strategies for the operator application
"""

import logging
import json
import numpy as np
import time
from datetime import datetime

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.utils import Quantity, log
from renormalizer.utils.constant import cm2au
from renormalizer.model.multiset_model import MultisetModel
from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.utils.log import package_logger as logger

# Setup logging
logger.setLevel(logging.INFO)

# Load FMO spectral density function
with open("/curie-home/zengjj/Renormalizer/example/fmo_sdf.json") as fin:
    sdf_values = json.load(fin)
sdf_values = np.array(sdf_values)

# FMO coupling matrix (cm^-1)
j_matrix_cm = np.array([
    [310, -98, 6, -6, 7, -12, -10, 38],
    [-98, 230, 30, 7, 2, 12, 5, 8],
    [6, 30, 0, -59, -2, -10, 5, 2],
    [-6, 7, -59, 180, -65, -17, -65, -2],
    [7, 2, -2, -65, 405, 89, -6, 5],
    [-12, 11, -10, -17, 89, 320, 32, -10],
    [-10, 5, 5, -64, -6, 32, 270, -11],
    [38, 8, 2, -2, 5, -10, -11, 505],
])

# Parameters
N_PHONONS = 35
TOTAL_HR = 0.42

def setup_fmo_model(max_bonddim=64):
    """Setup FMO model similar to fmo.py"""

    omegas_cm = np.linspace(2, 300, N_PHONONS)
    omegas_au = omegas_cm * cm2au
    hr_factors = np.interp(omegas_cm, sdf_values[:, 0], sdf_values[:, 1])
    hr_factors *= TOTAL_HR / hr_factors.sum()

    lams = hr_factors * omegas_au
    phonons = [Phonon.simplest_phonon(Quantity(o), Quantity(l), lam=True)
               for o, l in zip(omegas_au, lams)]

    j_matrix_au = j_matrix_cm * cm2au

    mlist = []
    for j in np.diag(j_matrix_au):
        m = Mol(Quantity(j), phonons)
        mlist.append(m)

    # Site arrangement
    mol_arangement = np.array([7, 5, 3, 1, 2, 4, 6]) - 1
    model = HolsteinModel(
        list(np.array(mlist)[mol_arangement]),
        j_matrix_au[mol_arangement][:, mol_arangement]
    )

    multisetmodel = MultisetModel(model, max_bonddim=max_bonddim)

    return multisetmodel


def extract_test_data(multisetmodel, evolve_dt=160):
    """
    Evolve the model a bit and extract data needed to test _apply_block_operator
    Returns: Y, op_list, block_dim, block_shape, N_electron
    """
    from renormalizer.mps.lib import Environ
    from renormalizer.mps.hop_expr import hop_expr
    from renormalizer.mps.matrix import asxp

    # Evolve once to get into a realistic state
    multisetmodel.evolve(evolve_dt=evolve_dt)

    ms_mps = multisetmodel.MsMps.to_complex()
    ms_mpo = multisetmodel.MsMpo
    N_electron = multisetmodel.N_electron

    # Construct environment matrices (similar to _ms_evolve_tdvp_ps)
    Environ_list = [[[] for _ in range(N_electron)] for _ in range(N_electron)]
    for alpha in range(N_electron):
        mpsconj = ms_mps.msmps[alpha].conj()
        for beta in range(N_electron):
            Environ_list[alpha][beta] = Environ(
                ms_mps.msmps[beta],
                ms_mpo.msmpo[alpha][beta],
                mps_conj=mpsconj
            )

    # Pick a site in the middle
    imps = len(ms_mps.msmps[0]) // 2
    shape_imps = list(ms_mps.msmps[0][imps].shape)
    dim = int(np.prod(shape_imps))

    # Construct hop_list (operator list)
    l_array_ab = [[[] for _ in range(N_electron)] for _ in range(N_electron)]
    r_array_ab = [[[] for _ in range(N_electron)] for _ in range(N_electron)]
    hop_list = [[[] for _ in range(N_electron)] for _ in range(N_electron)]

    for alpha in range(N_electron):
        for beta in range(N_electron):
            l_array_ab[alpha][beta] = Environ_list[alpha][beta].read("L", imps - 1)
            r_array_ab[alpha][beta] = Environ_list[alpha][beta].read("R", imps + 1)
            hop_list[alpha][beta] = hop_expr(
                l_array_ab[alpha][beta],
                r_array_ab[alpha][beta],
                [asxp(ms_mpo.msmpo[alpha][beta][imps].array)],
                shape_imps
            )

    # Construct Y0 (input vector)
    Y0 = xp.concatenate([asxp(ms_mps.msmps[a][imps].ravel().array)
                         for a in range(N_electron)])

    return Y0, hop_list, dim, shape_imps, N_electron


def benchmark_apply_block_operator(multisetmodel, Y, op_list, block_dim, block_shape, n_iterations=100):
    """
    Benchmark the _apply_block_operator methods

    Parameters
    ----------
    multisetmodel : MultisetModel
        The model instance
    Y : array
        Input vector
    op_list : list
        Operator list
    block_dim : int
        Block dimension
    block_shape : list
        Block shape
    n_iterations : int
        Number of iterations for timing

    Returns
    -------
    dict : timing results
    """

    logger.info("="*60)
    logger.info("Starting benchmark of _apply_block_operator")
    logger.info(f"GPU enabled: {USE_GPU}")
    logger.info(f"Backend: {'CuPy' if USE_GPU else 'NumPy'}")
    logger.info(f"N_electron: {multisetmodel.N_electron}")
    logger.info(f"Block dimension: {block_dim}")
    logger.info(f"Block shape: {block_shape}")
    logger.info(f"Y shape: {Y.shape}")
    logger.info(f"Y dtype: {Y.dtype}")
    logger.info(f"Iterations: {n_iterations}")
    logger.info("="*60)

    results = {}

    # Test 1: Original _apply_block_operator
    logger.info("\n[Test 1] Original _apply_block_operator")

    # Warmup
    for _ in range(3):
        _ = multisetmodel._apply_block_operator(Y, op_list, block_dim, block_shape)

    if USE_GPU:
        xp.cuda.Device().synchronize()

    start_time = time.perf_counter()
    for _ in range(n_iterations):
        result1 = multisetmodel._apply_block_operator(Y, op_list, block_dim, block_shape)

    if USE_GPU:
        xp.cuda.Device().synchronize()

    end_time = time.perf_counter()
    time_original = (end_time - start_time) / n_iterations

    logger.info(f"Average time per call: {time_original*1000:.3f} ms")
    logger.info(f"Total time for {n_iterations} iterations: {end_time - start_time:.3f} s")
    results['original'] = {
        'avg_time_ms': time_original * 1000,
        'total_time_s': end_time - start_time
    }

    # Test 2: Optimized _apply_block_operator_cupy (if GPU available)
    if USE_GPU:
        logger.info("\n[Test 2] Optimized _apply_block_operator_cupy")

        # Warmup
        for _ in range(3):
            _ = multisetmodel._apply_block_operator_cupy(Y, op_list, block_dim, block_shape)

        xp.cuda.Device().synchronize()

        start_time = time.perf_counter()
        for _ in range(n_iterations):
            result2 = multisetmodel._apply_block_operator_cupy(Y, op_list, block_dim, block_shape)

        xp.cuda.Device().synchronize()

        end_time = time.perf_counter()
        time_cupy = (end_time - start_time) / n_iterations

        logger.info(f"Average time per call: {time_cupy*1000:.3f} ms")
        logger.info(f"Total time for {n_iterations} iterations: {end_time - start_time:.3f} s")
        results['cupy_optimized'] = {
            'avg_time_ms': time_cupy * 1000,
            'total_time_s': end_time - start_time
        }

        # Compare results
        logger.info("\n[Comparison]")
        speedup = time_original / time_cupy
        logger.info(f"Speedup (cupy_optimized vs original): {speedup:.2f}x")
        logger.info(f"Time difference: {(time_original - time_cupy)*1000:.3f} ms per call")

        # Check numerical accuracy
        diff = xp.linalg.norm(result1 - result2) / xp.linalg.norm(result1)
        logger.info(f"Relative difference between methods: {diff:.2e}")

        results['speedup'] = speedup
        results['relative_diff'] = float(diff)
    else:
        logger.info("\n[Test 2] Skipped (GPU not available)")

    return results


def profile_single_call(multisetmodel, Y, op_list, block_dim, block_shape):
    """
    Profile a single call to understand the bottleneck
    """
    import cProfile
    import pstats
    from io import StringIO

    logger.info("\n" + "="*60)
    logger.info("Profiling single call to _apply_block_operator")
    logger.info("="*60)

    profiler = cProfile.Profile()
    profiler.enable()

    result = multisetmodel._apply_block_operator(Y, op_list, block_dim, block_shape)

    profiler.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    logger.info("\n" + s.getvalue())

    return result


def main():
    """Main test function"""

    # Setup
    max_bonddim = 64
    evolve_dt = 160
    n_iterations = 10  # Number of iterations for benchmarking

    logger.info("Setting up FMO model...")
    multisetmodel = setup_fmo_model(max_bonddim=max_bonddim)

    logger.info("\nExtracting test data...")
    Y, op_list, block_dim, block_shape, N_electron = extract_test_data(
        multisetmodel,
        evolve_dt=evolve_dt
    )

    # Benchmark
    logger.info("\nRunning benchmark...")
    results = benchmark_apply_block_operator(
        multisetmodel,
        Y,
        op_list,
        block_dim,
        block_shape,
        n_iterations=n_iterations
    )

    # Profile (optional, can be commented out)
    # logger.info("\nRunning profiler...")
    # profile_single_call(multisetmodel, Y, op_list, block_dim, block_shape)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*60)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU: {USE_GPU}")
    logger.info(f"Max bond dimension: {max_bonddim}")
    logger.info(f"N_electron: {N_electron}")
    logger.info(f"Block dimension: {block_dim}")
    logger.info(f"\nResults:")
    for method, timing in results.items():
        if isinstance(timing, dict):
            logger.info(f"  {method}: {timing['avg_time_ms']:.3f} ms/call")

    if 'speedup' in results:
        logger.info(f"\nSpeedup: {results['speedup']:.2f}x")

    logger.info("="*60)

    return results


if __name__ == "__main__":
    results = main()
