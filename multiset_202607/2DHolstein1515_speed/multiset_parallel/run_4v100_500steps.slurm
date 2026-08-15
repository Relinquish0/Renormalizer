#!/bin/bash
#SBATCH --job-name=H1515_m64_local_500
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=1T
#SBATCH --time=5-00:00:00
#SBATCH --partition=4V100
#SBATCH --qos=4gpu
#SBATCH --output=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolstein1515_speed/multiset_parallel_runs/H1515_m64_local_500steps_%j.log
#SBATCH --error=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolstein1515_speed/multiset_parallel_runs/H1515_m64_local_500steps_%j.log
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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUPY_ACCELERATORS=cutensor
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Production local-vector-slice path.  Host mode is required by the Curie
# OpenMPI build; direct CUDA-buffer collectives are not reliable here.
export RENO_MPI_KRYLOV_MODE=local
export RENO_MPI_ALLREDUCE_MODE=host
export RENO_MPI_DEBUG_SHAPES=0
export RENO_MPI_HOP_PROGRESS_INTERVAL=0
export OMPI_MCA_btl="^openib"
export OMPI_MCA_btl_vader_single_copy_mechanism=none

export NROW=15
export NCOL=15
export OMEGA_0=1.0
export J=1.0
export G=0.5
export NU_MAX=8
export MAX_BONDDIM=64
export EVOLVE_DT=0.1

# Holstein1515.py calls evolve(nsteps=N_SNAPSHOTS - 1), so 501 stored
# snapshots are required for exactly 500 evolution steps.
export N_SNAPSHOTS=501
export IF_STARTUP_SUBSTEPS=0
export STARTUP_SUBSTEPS_N=10
export INITIAL_SITE=112

run_root=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolstein1515_speed/multiset_parallel_runs
run_dir="${run_root}/${SLURM_JOB_ID}"
mkdir -p "${run_dir}"
export OUTPUT_XLSX="${run_dir}/H1515_2D_m64_local_500steps_${SLURM_JOB_ID}.xlsx"
export OUTPUT_NPZ="${run_dir}/H1515_2D_m64_local_500steps_${SLURM_JOB_ID}.npz"

echo "=== 4xV100 local-slice production run ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "NROW=${NROW} NCOL=${NCOL} NU_MAX=${NU_MAX} MAX_BONDDIM=${MAX_BONDDIM}"
echo "EVOLVE_DT=${EVOLVE_DT} N_SNAPSHOTS=${N_SNAPSHOTS} EVOLVE_STEPS=500"
echo "RENO_MPI_KRYLOV_MODE=${RENO_MPI_KRYLOV_MODE}"
echo "OUTPUT_NPZ=${OUTPUT_NPZ}"
echo "OUTPUT_XLSX=${OUTPUT_XLSX}"
python --version

set +e
srun --mpi=pmix \
    --ntasks=4 \
    --cpus-per-task=4 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    python Holstein1515.py
status=$?
set -e

echo "srun exit code: ${status}"
date
exit "${status}"
