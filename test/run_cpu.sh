#!/bin/bash
#SBATCH --job-name=fmo_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16         # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=fmo_out_cpu.log
#SBATCH --error=fmo_err_cpu.log


# your job script
source $HOME/.bashrc
source /software/envs/anaconda3.env

conda activate renormalizer
export PYTHONUNBUFFERED=1

echo "=== Environment Information ==="
which python
python --version
nvcc --version
echo "==============================="

echo "Starting"
python fmo_cpu.py
echo "Job completed with exit code: $?"
echo "Ending"
echo "==============================="
# Must explicitly exit
exit