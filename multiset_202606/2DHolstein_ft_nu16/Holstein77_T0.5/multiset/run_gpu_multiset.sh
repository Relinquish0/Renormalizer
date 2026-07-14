#!/bin/bash
#SBATCH --job-name=H77_2D_ft_ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4V100
#SBATCH --qos=normal
#SBATCH --output=Holstein77_multiset.log
#SBATCH --error=Holstein77_multiset.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export NROW="${NROW:-7}"
export NCOL="${NCOL:-7}"
export G="${G:-1.0}"
export NU_MAX="${NU_MAX:-16}"
export MAX_BONDDIM="${MAX_BONDDIM:-8}"
export EVOLVE_DT="${EVOLVE_DT:-0.1}"
export N_SNAPSHOTS="${N_SNAPSHOTS:-251}"
export IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
export STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"

echo "=== Environment Information ==="
echo "PWD=$(pwd)"
ls -la Holstein77.py
which python
python --version
nvcc --version
echo "NROW=${NROW}"
echo "NCOL=${NCOL}"
echo "G=${G}"
echo "NU_MAX=${NU_MAX}"
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "EVOLVE_DT=${EVOLVE_DT}"
echo "N_SNAPSHOTS=${N_SNAPSHOTS}"
echo "==============================="

echo "Starting"
python Holstein77.py
echo "Job completed with exit code: $?"
echo "Ending"
