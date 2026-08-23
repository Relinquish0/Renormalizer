#!/bin/bash
#SBATCH --job-name=ms1gpu_scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=1500G
#SBATCH --time=1-00:00:00
#SBATCH --partition=4V100
#SBATCH --qos=normal
#SBATCH --output=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_1gpu/scaling/result/ladder_%j.log
#SBATCH --error=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_1gpu/scaling/result/ladder_%j.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_1gpu

# Single-V100 multiset size ladder.
#
# --qos=normal is what allows a *partial* GPU allocation on this partition: the
# 4gpu QOS is all-four-or-nothing (a 3-GPU job is cancelled at start), while
# normal caps the request at gres/gpu=1, which is exactly this job.
#
# --mem=1500G is the node maximum, so the host side is given every chance and
# whatever stops the ladder is a real limit rather than a stingy request.
#
#   sbatch scaling/submit.slurm
#   sbatch --export=ALL,SIZES="15 18 20",NSTEPS=5 scaling/submit.slurm

set -o pipefail

source /software/modules/init/bash
module use /software/modules/modulefiles/devtools
module use /software/modules/modulefiles/apps
module load cuda/12.4
source /software/envs/anaconda3.env
conda activate reno

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUPY_ACCELERATORS=cutensor
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

# Krylov block size: the same 24 the 4-GPU runs use, so the two ladders are
# comparing lattice size rather than solver settings.
export RENO_MS_KRYLOV_BLOCK=${RENO_MS_KRYLOV_BLOCK:-24}
export MAX_BONDDIM=${MAX_BONDDIM:-64}
export NU_MAX=${NU_MAX:-8}
export EVOLVE_DT=${EVOLVE_DT:-0.1}
export NSTEPS=${NSTEPS:-2}
export SETUP_ONLY=${SETUP_ONLY:-0}
export SIZES=${SIZES:-"15 18 20 22 24 25 26 27"}
export PER_SIZE_TIMEOUT=${PER_SIZE_TIMEOUT:-21600}

free -g | head -2
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "SIZES=${SIZES} M=${MAX_BONDDIM} NU_MAX=${NU_MAX} NSTEPS=${NSTEPS} SETUP_ONLY=${SETUP_ONLY} MEM=${SLURM_MEM_PER_NODE:-?}"
date

python -u scaling/run_scaling.py --sizes ${SIZES}

date
echo "ladder finished"
