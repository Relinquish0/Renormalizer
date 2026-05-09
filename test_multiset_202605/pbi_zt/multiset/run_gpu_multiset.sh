#!/bin/bash
#SBATCH --job-name=dimer_emi
#SBATCH --nodes=1
#SBATCH --ntasks=1        # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=1   # Specify the GPUs-per-node
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=dimer_emi.log
#SBATCH --error=dimer_emi.log

export TYPE_="${TYPE_:-dimer}"
export SPECTRATYPE="${SPECTRATYPE:-emi}"
export SPECTRA_TAG="${SPECTRA_TAG:-zt}"
export MAX_BONDDIM="${MAX_BONDDIM:-16}"
export EVOLVE_DT="${EVOLVE_DT:-20}"
export NSTEPS="${NSTEPS:-5000}"

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1

# 显示环境信息用于调试
echo "=== Environment Information ==="

which python
python --version
nvcc --version
echo "TYPE_=$TYPE_"
echo "SPECTRATYPE=$SPECTRATYPE"
echo "SPECTRA_TAG=$SPECTRA_TAG"
echo "MAX_BONDDIM=$MAX_BONDDIM"
echo "EVOLVE_DT=$EVOLVE_DT"
echo "NSTEPS=$NSTEPS"

echo "==============================="

# 运行主要的Python任务
echo "Starting"
python PBI.py
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit
