#!/bin/bash
#SBATCH --job-name=ms_scale_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --partition=4V100
#SBATCH --qos=4gpu
#SBATCH --output=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results/scale_1gpu_%j.log
#SBATCH --error=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results/scale_1gpu_%j.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling

# Single-GPU multiset size ladder.  Override on submission, e.g.
#   sbatch --export=ALL,SIZES="20 22 25",SETUP_ONLY=1 multiset/submit.slurm

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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

export MAX_BONDDIM=${MAX_BONDDIM:-64}
export NU_MAX=${NU_MAX:-8}
export EVOLVE_DT=${EVOLVE_DT:-0.1}
export NSTEPS=${NSTEPS:-10}
export STEP_BUDGET_S=${STEP_BUDGET_S:-3600}
export SETUP_ONLY=${SETUP_ONLY:-0}
SIZES=${SIZES:-"15 18 20 22 25"}
PER_SIZE_TIMEOUT=${PER_SIZE_TIMEOUT:-14400}
export RESULT_DIR=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "SIZES=${SIZES} M=${MAX_BONDDIM} NSTEPS=${NSTEPS} SETUP_ONLY=${SETUP_ONLY}"

for N in ${SIZES}; do
    export NROW=${N} NCOL=${N}
    export RUN_TAG="single_${N}x${N}_m${MAX_BONDDIM}$([ "${SETUP_ONLY}" = "1" ] && echo _setup)"
    echo "================= 1 GPU ${N}x${N} ($((N*N)) sites) ================="
    date
    timeout "${PER_SIZE_TIMEOUT}" python multiset/run_scaling.py
    echo "exit=$? for ${N}x${N}"
    date
done
echo "ladder finished"
