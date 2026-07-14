#!/bin/bash
#SBATCH --job-name=P3HT_ss_tree_m32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=512G
#SBATCH -p 4A100
#SBATCH --qos=normal
#SBATCH --output=P3HT_singleset_tree_m32.log
#SBATCH --error=P3HT_singleset_tree_m32.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1

export JOB_NAME="${JOB_NAME:-p3ht_ttns_tree_bond_entropy}"
export MAX_BONDDIM="${MAX_BONDDIM:-32}"
export DT_FS="${DT_FS:-1.0}"
export TOTAL_FS="${TOTAL_FS:-200.0}"
export INITIAL_SITE="${INITIAL_SITE:-0}"
export PRINT_ONLY="${PRINT_ONLY:-0}"
export ENTROPY_LOG_INTERVAL="${ENTROPY_LOG_INTERVAL:-20}"


echo "=== Environment Information ==="
which python
python --version
nvcc --version
pwd
echo "JOB_NAME=${JOB_NAME}"
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "DT_FS=${DT_FS}"
echo "TOTAL_FS=${TOTAL_FS}"
echo "INITIAL_SITE=${INITIAL_SITE}"
echo "PRINT_ONLY=${PRINT_ONLY}"
echo "ENTROPY_LOG_INTERVAL=${ENTROPY_LOG_INTERVAL}"
echo "==============================="

echo "Starting"
python P3HT.py
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit
