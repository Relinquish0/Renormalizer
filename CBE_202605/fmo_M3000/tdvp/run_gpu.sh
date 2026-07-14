#!/bin/bash
#SBATCH --job-name=fmo_M3000
#SBATCH --nodes=1
#SBATCH --ntasks=1        # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=1   # Specify the GPUs-per-node
#SBATCH --mem=1024G
#SBATCH -p 4V100
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=fmo_M3000.log
#SBATCH --error=fmo_M3000.log

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

echo "==============================="

# 运行主要的Python任务
echo "Starting"
python fmo.py \
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit