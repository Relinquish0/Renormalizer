#!/bin/bash
# Setup-phase scaling probe.  NOTE: renormalizer/mps/backend.py enables the GPU
# whenever CuPy imports, so this runs on whatever device the host exposes.
set -o pipefail
source /software/envs/anaconda3.env
conda activate reno
export PYTHONUNBUFFERED=1 SETUP_ONLY=1 MAX_BONDDIM=64 NU_MAX=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export RESULT_DIR=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results
for N in "$@"; do
  export NROW=$N NCOL=$N RUN_TAG="cpusetup_${N}x${N}_m64"
  echo "=================== ${N}x${N} ==================="
  timeout "${PER_SIZE_TIMEOUT:-1800}" python multiset/run_scaling.py 2>&1 \
    | grep -E "PHASE|RESULT|static sizes|FAILED|Error|error" || echo "TIMEOUT/FAIL at ${N}x${N}"
done
