#!/bin/bash
#SBATCH --job-name=Dimer_emi_offset_s32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4V100
#SBATCH --qos=normal
#SBATCH --output=Dimer_emi_offset_s32.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_spectra/pbi_ft_offset/dimer/emi/singleset
#SBATCH --error=Dimer_emi_offset_s32.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export MODEL_NAME="${MODEL_NAME:-dimer}"
export OFFSET_MODE="${OFFSET_MODE:-EEX0}"
export MAX_BONDDIM="${MAX_BONDDIM:-32}"
export EVOLVE_DT="${EVOLVE_DT:-20}"
export NSTEPS="${NSTEPS:-5000}"
export INSTEPS="${INSTEPS:-50}"
export TEMPERATURE_K="${TEMPERATURE_K:-298}"

echo "=== Environment Information ==="
echo "PWD=$(pwd)"
ls -la PBI_dimer_offset.py
which python
python --version
nvcc --version
echo "MODEL_NAME=${MODEL_NAME}"
echo "OFFSET_MODE=${OFFSET_MODE}"
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "EVOLVE_DT=${EVOLVE_DT}"
echo "NSTEPS=${NSTEPS}"
echo "INSTEPS=${INSTEPS}"
echo "TEMPERATURE_K=${TEMPERATURE_K}"
echo "OUTPUT_NPZ=Dimer_emi_s${MAX_BONDDIM}_offset_${OFFSET_MODE}.npz"
echo "==============================="

echo "Starting"
python PBI_dimer_offset.py
status=$?
echo "Job completed with exit code: $status"

echo "Ending"
echo "==============================="

exit $status
