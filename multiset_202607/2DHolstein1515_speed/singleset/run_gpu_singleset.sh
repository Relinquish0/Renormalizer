#!/bin/bash
#SBATCH --job-name=G0.5_H1515_2D_s16_speed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4V100
#SBATCH --qos=normal
#SBATCH --output=G0.5_H1515_2D_s16_speed.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolstein1515_speed/singleset
#SBATCH --error=G0.5_H1515_2D_s16_speed.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export NROW="${NROW:-15}"
export NCOL="${NCOL:-15}"
export OMEGA_0="${OMEGA_0:-1.0}"
export J="${J:-1.0}"
export G="${G:-0.5}"
export NU_MAX="${NU_MAX:-8}"
export MAX_BONDDIM="${MAX_BONDDIM:-16}"
export EVOLVE_DT="${EVOLVE_DT:-0.1}"
export N_SNAPSHOTS="${N_SNAPSHOTS:-500}"
export MODEL_SCHEME="${MODEL_SCHEME:-4}"
export IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
export STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"
export OUTPUT_XLSX="${OUTPUT_XLSX:-H1515_2D_s${MAX_BONDDIM}_speed.xlsx}"
export OUTPUT_NPZ="${OUTPUT_NPZ:-H1515_2D_s${MAX_BONDDIM}_speed.npz}"

echo "=== Environment Information ==="
echo "PWD=$(pwd)"
ls -la Holstein1515.py
which python
python --version
nvcc --version
echo "NROW=${NROW}"
echo "NCOL=${NCOL}"
echo "OMEGA_0=${OMEGA_0}"
echo "J=${J}"
echo "G=${G}"
echo "NU_MAX=${NU_MAX}"
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
status=$?
echo "Job completed with exit code: $status"

echo "Ending"
echo "==============================="

exit $status
