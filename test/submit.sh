#!/bin/bash
#SBATCH --job-name=fmo_4bd_160t_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1         # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=1   # Specify the GPUs-per-node
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=fmo_out_4bd_160t_GPU.txt
#SBATCH --error=fmo_err_4bd_160t_GPU.txt

# Below are executing commands
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &

# your job script
source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4

conda activate renormalizer
export PYTHONUNBUFFERED=1
# 显示环境信息用于调试
echo "=== Environment Information ==="
which python
python --version
nvcc --version
echo "==============================="

# 运行主要的Python任务
echo "Starting"
python /curie-home/zengjj/Renormalizer/test/fmo.py \ 
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="
# Must explicitly exit
exit
