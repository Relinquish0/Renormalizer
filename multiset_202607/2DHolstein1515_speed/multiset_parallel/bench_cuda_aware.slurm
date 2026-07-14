#!/bin/bash
#SBATCH --job-name=mpi_cuda_aware_bench
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH -p 4V100
#SBATCH --qos=4gpu
#SBATCH --output=mpi_cuda_aware_bench_%j.log
#SBATCH --error=mpi_cuda_aware_bench_%j.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolstein1515_speed/multiset_parallel

set -euo pipefail

source /software/modules/init/bash
module use /software/modules/modulefiles/devtools
module use /software/modules/modulefiles/apps
module load openmpi/4.1.8-gnu-13.3.0
module load cuda/12.4

set +u
source /software/envs/anaconda3.env
conda activate reno
set -u

export PYTHONUNBUFFERED=1
export OMPI_MCA_btl="^openib"
export OMPI_MCA_btl_vader_single_copy_mechanism="none"

echo "=== MPI CUDA-aware benchmark ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
which python
python --version
python - <<'PY'
import cupy
import mpi4py
from mpi4py import MPI
print("cupy", cupy.__version__)
print("mpi4py", mpi4py.__version__)
print("MPI", MPI.Get_library_version().splitlines()[0])
PY

srun --mpi=pmix -n "${SLURM_NTASKS}" python mpi_cuda_aware_bench.py

date
