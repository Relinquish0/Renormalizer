#!/bin/bash
#SBATCH --job-name=ms4gpu_scale
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=1500G
#SBATCH --time=1-00:00:00
#SBATCH --partition=4V100
#SBATCH --qos=4gpu
#SBATCH --output=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_4gpu/scaling/result/ladder_%j.log
#SBATCH --error=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_4gpu/scaling/result/ladder_%j.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_4gpu

# 4xV100 multiset size ladder.  --mem=1500G is the node maximum (the partition
# advertises 1536000 MB = exactly 1500 GiB); the host side is the binding
# constraint below stage 4, so this is deliberately pinned to the ceiling.
#
#   sbatch scaling/submit.slurm                       # default ladder, stage 4
#   sbatch --export=ALL,SIZES="25 27",STAGE=4 scaling/submit.slurm
#   sbatch --export=ALL,NSTEPS=100,SIZES="27" scaling/submit.slurm

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

# This OpenMPI build is not CUDA-aware, so every collective is staged through
# host buffers.
export RENO_MPI_ALLREDUCE_MODE=${RENO_MPI_ALLREDUCE_MODE:-host}
export RENO_MPI_DEBUG_SHAPES=0
export RENO_MPI_HOP_PROGRESS_INTERVAL=0
export OMPI_MCA_btl="^openib"
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# From stage 3 the solver auto-routes to the local-slice path (the sharded sweep
# tags its Afunc with ``_reno_local_rows``), so this only selects the behaviour
# for stages 1-2, where the MPS is still replicated.
export STAGE=${STAGE:-4}
export RENO_MPI_KRYLOV_MODE=${RENO_MPI_KRYLOV_MODE:-distributed}

# Krylov block size: 24 keeps the batched Lanczos inside the 32 GiB V100.
export RENO_MS_KRYLOV_BLOCK=${RENO_MS_KRYLOV_BLOCK:-24}
export MAX_BONDDIM=${MAX_BONDDIM:-64}
export NU_MAX=${NU_MAX:-8}
export EVOLVE_DT=${EVOLVE_DT:-0.1}
export NSTEPS=${NSTEPS:-5}
export SIZES=${SIZES:-"15 18 20 22 24 25 26 27"}
export PER_SIZE_TIMEOUT=${PER_SIZE_TIMEOUT:-14400}
# Follow the allocation rather than the #SBATCH default, so a partial-node
# backfill run (e.g. 3 GPUs while another job holds the fourth) launches the
# right number of ranks instead of oversubscribing the devices it was given.
export NTASKS=${SLURM_NTASKS:-4}

free -g | head -2
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "SIZES=${SIZES} STAGE=${STAGE} M=${MAX_BONDDIM} NSTEPS=${NSTEPS} MEM=${SLURM_MEM_PER_NODE:-?}"
date

EXTRA=""
[ "${STOP_ON_SUCCESS:-0}" != "0" ] && EXTRA="--stop-on-success"
python -u scaling/run_scaling.py --sizes ${SIZES} --stage ${STAGE} ${EXTRA}

date
echo "ladder finished"
