#!/bin/bash
#SBATCH --job-name=H55_2D_m32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal
#SBATCH --output=H55_2D_m32.log
#SBATCH --error=H55_2D_m32.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export MAX_BONDDIM="${MAX_BONDDIM:-32}"
export EVOLVE_DT="${EVOLVE_DT:-0.1}"
export N_SNAPSHOTS="${N_SNAPSHOTS:-500}"
export IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
export STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"
export OUTPUT_XLSX="${OUTPUT_XLSX:-H55_2D_m${MAX_BONDDIM}.xlsx}"
export OUTPUT_NPZ="${OUTPUT_NPZ:-H55_2D_m${MAX_BONDDIM}.npz}"

# cd "$(dirname "$0")"

echo "=== Environment Information ==="
which python
python --version
nvcc --version
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "EVOLVE_DT=${EVOLVE_DT}"
echo "N_SNAPSHOTS=${N_SNAPSHOTS}"
echo "IF_STARTUP_SUBSTEPS=${IF_STARTUP_SUBSTEPS}"
echo "STARTUP_SUBSTEPS_N=${STARTUP_SUBSTEPS_N}"
echo "OUTPUT_XLSX=${OUTPUT_XLSX}"
echo "OUTPUT_NPZ=${OUTPUT_NPZ}"
echo "==============================="

echo "Starting"
python Holstein55.py
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit
