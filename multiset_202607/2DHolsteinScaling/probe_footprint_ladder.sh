#!/bin/bash
# Footprint-only ladder: builds the model and both grouping passes but skips
# expand_bond_dimension_multiset (the O(L^2) setup cost), so the exact scatter-S
# and stacked-W GPU footprints can be measured at large L in minutes.
# Bond dimensions stay at 1, so no step timing or Krylov width is produced.
set -o pipefail
source /software/envs/anaconda3.env
conda activate reno
export PYTHONUNBUFFERED=1 SKIP_EXPAND=1 MAX_BONDDIM=64 NU_MAX=8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export RESULT_DIR=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results
cd /curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling
for N in "$@"; do
  export NROW=$N NCOL=$N RUN_TAG="footprint_${N}x${N}_m64"
  echo "=================== ${N}x${N} ==================="
  timeout "${PER_SIZE_TIMEOUT:-5400}" nice -n 10 python multiset/run_scaling.py 2>&1 \
    | grep -E "PHASE|RESULT|static sizes|FAILED|Error" || echo "TIMEOUT/FAIL at ${N}x${N}"
done
echo "ladder finished"
