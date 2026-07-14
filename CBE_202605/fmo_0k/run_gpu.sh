#!/bin/bash
#SBATCH --job-name=fmo_cbe_32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal
#SBATCH --output=fmo_cbe_32.log
#SBATCH --error=fmo_cbe_32.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1

export MAX_BONDDIM="${MAX_BONDDIM:-32}"
export EVOLVE_DT="${EVOLVE_DT:-160}"
export EVOLVE_TIME="${EVOLVE_TIME:-40000}"
export JOB_NAME="${JOB_NAME:-fmo_0k_singleset_cbe_M${MAX_BONDDIM}}"
export CBE_WARMUP_TIME="${CBE_WARMUP_TIME:-160.0}"
export CBE_WARMUP_SUBSTEPS="${CBE_WARMUP_SUBSTEPS:-10}"
# Optional CBE controls:
# export CBE_MAX_EXPAND=4
# export CBE_EPS_PRE=1e-4
# export CBE_EPS_FINAL=1e-6
export CBE_EPS_TRIM="${CBE_EPS_TRIM:-1e-20}"
export CBE_DMAX="${CBE_DMAX:-${MAX_BONDDIM}}"

echo "=== Environment Information ==="
which python
python --version
nvcc --version
echo "==============================="

echo "Starting CBE FMO single-set run"
python fmo.py
echo "Job completed with exit code: $?"
echo "Ending"
