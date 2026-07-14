#!/bin/bash
#SBATCH --job-name=G2_H77_2D_m16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal
#SBATCH --output=G2_H77_2D_m16.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202606/2DHolstein_nu16/G2/Holstein_2D77/multiset
#SBATCH --error=G2_H77_2D_m16.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export G="2.0"
export NU_MAX="16"
export MAX_BONDDIM="${MAX_BONDDIM:-16}"
export EVOLVE_DT="${EVOLVE_DT:-0.1}"
export N_SNAPSHOTS="${N_SNAPSHOTS:-500}"
export IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
export STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"
export OUTPUT_XLSX="${OUTPUT_XLSX:-H77_2D_m${MAX_BONDDIM}.xlsx}"
export OUTPUT_NPZ="${OUTPUT_NPZ:-H77_2D_m${MAX_BONDDIM}.npz}"

echo "=== Environment Information ==="
echo "PWD=$(pwd)"
ls -la Holstein77.py
which python
python --version
nvcc --version
echo "G=${G}"
echo "NU_MAX=${NU_MAX}"
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "EVOLVE_DT=${EVOLVE_DT}"
echo "N_SNAPSHOTS=${N_SNAPSHOTS}"
echo "IF_STARTUP_SUBSTEPS=${IF_STARTUP_SUBSTEPS}"
echo "STARTUP_SUBSTEPS_N=${STARTUP_SUBSTEPS_N}"
echo "OUTPUT_XLSX=${OUTPUT_XLSX}"
echo "OUTPUT_NPZ=${OUTPUT_NPZ}"
echo "==============================="

echo "Starting"
python Holstein77.py
status=$?
echo "Job completed with exit code: $status"

echo "Ending"
echo "==============================="

exit $status
