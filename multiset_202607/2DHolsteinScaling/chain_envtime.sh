#!/bin/bash
# wait for the env_probe ladder to release the GPU, then time the Environ traffic
while pgrep -u zengjj -f 'env_probe.py' >/dev/null 2>&1; do sleep 30; done
sleep 20
source /software/envs/anaconda3.env
conda activate reno
export PYTHONUNBUFFERED=1 MAX_BONDDIM=64 NU_MAX=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export RESULT_DIR=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results
cd /curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling
for N in 6 8; do
  export NROW=$N NCOL=$N RUN_TAG="envtime_${N}x${N}_m64"
  echo "=== envtime ${N}x${N} ==="
  timeout 5400 nice -n 10 python env_timing_probe.py 2>&1 | grep -E "ENVTIME|Error|Traceback" || echo "FAIL ${N}x${N}"
done
