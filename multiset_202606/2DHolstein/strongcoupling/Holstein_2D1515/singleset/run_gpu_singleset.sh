#!/bin/bash
#SBATCH --job-name=H1515_2D_s64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal
#SBATCH --output=H1515_2D_s64.log
#SBATCH --error=H1515_2D_s64.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export MAX_BONDDIM="${MAX_BONDDIM:-64}"
export EVOLVE_DT="${EVOLVE_DT:-0.1}"
export N_SNAPSHOTS="${N_SNAPSHOTS:-500}"
export MODEL_SCHEME="${MODEL_SCHEME:-4}"
export IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
export STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"
export OUTPUT_XLSX="${OUTPUT_XLSX:-H1515_2D_s${MAX_BONDDIM}.xlsx}"
export OUTPUT_NPZ="${OUTPUT_NPZ:-H1515_2D_s${MAX_BONDDIM}.npz}"

# WORKDIR="/curie-home/zengjj/Renormalizer/multiset_202606/Holstein_2D1111/singleset"
# cd "${WORKDIR}" || exit 1

echo "=== Environment Information ==="
echo "WORKDIR=${WORKDIR}"
echo "PWD=$(pwd)"
ls -la Holstein1111.py
which python
python --version
nvcc --version
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "EVOLVE_DT=${EVOLVE_DT}"
echo "N_SNAPSHOTS=${N_SNAPSHOTS}"
echo "MODEL_SCHEME=${MODEL_SCHEME}"
echo "IF_STARTUP_SUBSTEPS=${IF_STARTUP_SUBSTEPS}"
echo "STARTUP_SUBSTEPS_N=${STARTUP_SUBSTEPS_N}"
echo "OUTPUT_XLSX=${OUTPUT_XLSX}"
echo "OUTPUT_NPZ=${OUTPUT_NPZ}"
echo "==============================="

echo "Starting"
python Holstein1515.py
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit
