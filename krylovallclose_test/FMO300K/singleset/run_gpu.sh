#!/bin/bash
#SBATCH --job-name=fmo_64strictkrylov
#SBATCH --nodes=1
#SBATCH --ntasks=1        # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=1   # Specify the GPUs-per-node
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=fmo_64strictkrylov.log
#SBATCH --error=fmo_64strictkrylov.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1
export KRYLOV_ALLCLOSE_RTOL="${KRYLOV_ALLCLOSE_RTOL:-1e-8}"
export KRYLOV_ALLCLOSE_ATOL="${KRYLOV_ALLCLOSE_ATOL:-1e-10}"

# 显示环境信息用于调试
echo "=== Environment Information ==="

which python
python --version
nvcc --version
echo "KRYLOV_ALLCLOSE_RTOL=${KRYLOV_ALLCLOSE_RTOL}"
echo "KRYLOV_ALLCLOSE_ATOL=${KRYLOV_ALLCLOSE_ATOL}"

echo "==============================="

# 运行主要的Python任务
echo "Starting"
python fmo.py
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="

exit