#!/bin/bash
#SBATCH --job-name=H31_T0_ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal
#SBATCH --output=Holstein31_T0_multiset.log
#SBATCH --error=Holstein31_T0_multiset.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export G="${G:-1.0}"
export NU_MAX="${NU_MAX:-16}"
export MAX_BONDDIM="${MAX_BONDDIM:-16}"
export EVOLVE_DT="${EVOLVE_DT:-0.1}"
export N_SNAPSHOTS="${N_SNAPSHOTS:-500}"
export IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
export STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"

echo "=== Environment Information ==="
which python
python --version
nvcc --version
echo "T_STAR=0"
echo "G=${G}"
echo "NU_MAX=${NU_MAX}"
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "EVOLVE_DT=${EVOLVE_DT}"
echo "N_SNAPSHOTS=${N_SNAPSHOTS}"
echo "==============================="

echo "Starting"
python Holstein31.py
echo "Job completed with exit code: $?"
echo "Ending"
