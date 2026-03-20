#!/bin/bash
#SBATCH --job-name=Holstein75_256bd
#SBATCH --nodes=1
#SBATCH --ntasks=1        # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=1   # Specify the GPUs-per-node
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=Holstein75_256bd.log
#SBATCH --error=Holstein75_256bd.log

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
python Holstein75.py \
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit