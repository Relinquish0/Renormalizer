#!/bin/bash
#SBATCH --job-name=P3HT_singleset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4V100
#SBATCH --qos=normal
#SBATCH --output=P3HT_singleset.log
#SBATCH --error=P3HT_singleset.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export JOB_NAME="${JOB_NAME:-p3ht_singleset}"
export MAX_BONDDIM="${MAX_BONDDIM:-32}"
export DT_FS="${DT_FS:-1.0}"
export TOTAL_FS="${TOTAL_FS:-200.0}"

WORK_DIR="${SLURM_SUBMIT_DIR:-/curie-home/zengjj/Renormalizer/test_multiset_202605/P3HT:PCBM/singleset}"
cd "${WORK_DIR}" || exit 1

echo "=== Environment Information ==="

which python
python --version
nvcc --version
pwd
echo "JOB_NAME=${JOB_NAME}"
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "DT_FS=${DT_FS}"
echo "TOTAL_FS=${TOTAL_FS}"
echo "==============================="

echo "Starting"
python P3HT.py
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit
