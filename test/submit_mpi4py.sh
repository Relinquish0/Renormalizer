#!/bin/bash
#SBATCH --job-name=fmo_mpi4py
#SBATCH --nodes=1
#SBATCH --ntasks=1         # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=1   # Specify the GPUs-per-node
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=fmo_out_mpi4py.log
#SBATCH --error=fmo_err_mpi4py.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
# 如果有 cuda-aware openmpi，加载它会大大加速 gpu 通信
# module load openmpi/4.1.x-cuda 

conda activate renormalizer

# 环境变量设置
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 调试信息
echo "Running on node: $(hostname)"
nvidia-smi

# 运行命令
# 这里的 python -u 后面不要加反斜杠，除非你换行了
# mpirun 会自动分发 ntasks (7个)
mpirun python -u fmo.py