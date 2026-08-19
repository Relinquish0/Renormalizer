#!/bin/bash
#SBATCH --job-name=ms_scale_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=1T
#SBATCH --time=1-00:00:00
#SBATCH --partition=4V100
#SBATCH --qos=4gpu
#SBATCH --output=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results/scale_4gpu_%j.log
#SBATCH --error=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results/scale_4gpu_%j.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling

# 4-GPU MPI multiset size ladder.  Override on submission, e.g.
#   sbatch --export=ALL,SIZES="20 22 25" multiset_parallel/submit.slurm

set -o pipefail

source /software/modules/init/bash
module use /software/modules/modulefiles/devtools
module use /software/modules/modulefiles/apps
module load openmpi/4.1.8-gnu-13.3.0
module load cuda/12.4
source /software/envs/anaconda3.env
conda activate reno

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUPY_ACCELERATORS=cutensor
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# Same transport settings as the production 15x15 run: this OpenMPI build is
# not CUDA-aware, so collectives are staged through host buffers.
export RENO_MPI_KRYLOV_MODE=${RENO_MPI_KRYLOV_MODE:-local}
export RENO_MPI_ALLREDUCE_MODE=${RENO_MPI_ALLREDUCE_MODE:-host}
export RENO_MPI_DEBUG_SHAPES=0
export RENO_MPI_HOP_PROGRESS_INTERVAL=0
export OMPI_MCA_btl="^openib"
export OMPI_MCA_btl_vader_single_copy_mechanism=none

export MAX_BONDDIM=${MAX_BONDDIM:-64}
export NU_MAX=${NU_MAX:-8}
export EVOLVE_DT=${EVOLVE_DT:-0.1}
export NSTEPS=${NSTEPS:-10}
export STEP_BUDGET_S=${STEP_BUDGET_S:-3600}
export SETUP_ONLY=${SETUP_ONLY:-0}
SIZES=${SIZES:-"15 18 20 22 25"}
PER_SIZE_TIMEOUT=${PER_SIZE_TIMEOUT:-14400}
export RESULT_DIR=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results

nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "SIZES=${SIZES} M=${MAX_BONDDIM} NSTEPS=${NSTEPS} SETUP_ONLY=${SETUP_ONLY}"

for N in ${SIZES}; do
    export NROW=${N} NCOL=${N}
    export RUN_TAG="mpi4_${N}x${N}_m${MAX_BONDDIM}$([ "${SETUP_ONLY}" = "1" ] && echo _setup)"
    echo "================= 4 GPU ${N}x${N} ($((N*N)) sites) ================="
    date
    timeout "${PER_SIZE_TIMEOUT}" srun --mpi=pmix \
        --ntasks=4 --cpus-per-task=4 \
        --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores \
        python multiset_parallel/run_scaling.py
    echo "exit=$? for ${N}x${N}"
    date
done
echo "ladder finished"
