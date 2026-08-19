#!/bin/bash
# Calibrate HOP_TEMPS from a real evolution step instead of the 4x4 point.
# Runs the full pipeline (expansion + 2 steps) at 15x15; the peak is reached
# during step 1, so 2 steps is enough.  HOP_TEMPS is then
#   (step_peak - krylov_measured - S - W) / (n_pairs * dim * 16)
set -o pipefail
source /software/envs/anaconda3.env
conda activate reno
export PYTHONUNBUFFERED=1 MAX_BONDDIM=64 NU_MAX=8 NSTEPS=2 STEP_BUDGET_S=3600
export NROW=15 NCOL=15 RUN_TAG=hopcal_15x15_m64
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export RESULT_DIR=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results
timeout 7200 nice -n 10 python multiset/run_scaling.py 2>&1 \
  | grep -E "PHASE|RESULT|static sizes|FAILED|Error"
