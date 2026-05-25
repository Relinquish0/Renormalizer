#!/bin/bash
#SBATCH --job-name=H75_M32Krylov
#SBATCH --nodes=1
#SBATCH --ntasks=1        # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=1   # Specify the GPUs-per-node
#SBATCH -p 4V100
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=H75_M32Krylov.log
#SBATCH --error=H75_M32Krylov.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export MAX_BONDDIM="${MAX_BONDDIM:-32}"
export EVOLVE_DT="${EVOLVE_DT:-0.1}"
export N_SNAPSHOTS="${N_SNAPSHOTS:-500}"
export IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-0}"
export STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"
export OUTPUT_XLSX="${OUTPUT_XLSX:-Holstein75_${MAX_BONDDIM}bd_multiset.xlsx}"

# 显示环境信息用于调试
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

echo "==============================="

# 运行主要的Python任务
echo "Starting"
python Holstein75_multiset.py
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit
