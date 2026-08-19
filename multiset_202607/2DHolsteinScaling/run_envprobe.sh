#!/bin/bash
set -o pipefail
source /software/envs/anaconda3.env
conda activate reno
export PYTHONUNBUFFERED=1 MAX_BONDDIM=64 NU_MAX=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export RESULT_DIR=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results
cd /curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling
for N in "$@"; do
  export NROW=$N NCOL=$N RUN_TAG="envprobe_${N}x${N}_m64"
  echo "=== ${N}x${N} ==="
  timeout 5400 nice -n 10 python env_probe.py 2>&1 | grep -E "ENVPROBE|Error|Traceback" || echo "FAIL ${N}x${N}"
done
