#!/bin/bash
#SBATCH --job-name=fmo_cpu_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=16         # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --qos=normal          # Depending on your needs
#SBATCH --output=fmo_out_cpu_benchmark.log
#SBATCH --error=fmo_err_cpu_benchmark.log

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
python /curie-home/zengjj/Renormalizer/example/fmo.py
echo "Job completed with exit code: $?"

echo "Ending"
echo "==============================="
# Must explicitly exit
exit