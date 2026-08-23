#!/bin/bash
#SBATCH --job-name=ms_setupmem
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=08:00:00
#SBATCH --partition=CPU
#SBATCH --output=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_4gpu/scaling/result/setupmem_%j.log
#SBATCH --error=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_4gpu/scaling/result/setupmem_%j.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202608/multiset_4gpu

# Host-memory probe: build the model, MPOs, environments and initial state, then
# stop.  It needs no GPU, so it can measure the host-side ceiling (which is what
# binds below stage 4) on an idle CPU node while the 4V100 is occupied.
#
#   sbatch --export=ALL,SIZES="15 18 20",STAGE=1 scaling/submit_setup_cpu.slurm

set -o pipefail

source /software/modules/init/bash
module use /software/modules/modulefiles/devtools
module use /software/modules/modulefiles/apps
module load openmpi/4.1.8-gnu-13.3.0
source /software/envs/anaconda3.env
conda activate reno

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export RENO_MPI_ALLREDUCE_MODE=host
export RENO_MPI_DEBUG_SHAPES=0
export RENO_MPI_HOP_PROGRESS_INTERVAL=0
export OMPI_MCA_btl="^openib"
export OMPI_MCA_btl_vader_single_copy_mechanism=none
# No GPU on this partition: force the NumPy backend rather than letting each
# rank probe a device that is not there.
export RENO_NO_GPU=1
export CUDA_VISIBLE_DEVICES=""

# SETUP_ONLY=1 stops after the model/MPO/initial-state build.  That understates
# the real footprint: conj_mps is lazy, so the environment cache -- the dominant
# host term, and the one the alpha sharding is aimed at -- is not allocated
# until the first sweep.  SETUP_ONLY=2 builds that cache directly, which costs a
# fraction of a TDVP step and measures the term that decides what fits.
# SETUP_ONLY=0 runs the real evolution (only worth it on a GPU).
export SETUP_ONLY=${SETUP_ONLY:-1}
export STAGE=${STAGE:-4}
export MAX_BONDDIM=${MAX_BONDDIM:-64}
export NU_MAX=${NU_MAX:-8}
export NSTEPS=${NSTEPS:-1}
export SIZES=${SIZES:-"15 18 20 22 24 25 27"}
export PER_SIZE_TIMEOUT=${PER_SIZE_TIMEOUT:-5400}
export NTASKS=4
case "${SETUP_ONLY}" in
  1) _phase=setup ;;
  2) _phase=environ ;;
  *) _phase=step ;;
esac
export RUN_LABEL=${RUN_LABEL:-"_stage${STAGE}_${_phase}"}

free -g | head -2
echo "host memory probe: SIZES=${SIZES} STAGE=${STAGE} M=${MAX_BONDDIM} SETUP_ONLY=${SETUP_ONLY} (${_phase})"
date
EXTRA=""
[ "${STOP_ON_SUCCESS:-0}" != "0" ] && EXTRA="--stop-on-success"
python -u scaling/run_scaling.py --sizes ${SIZES} --stage ${STAGE} --label "${RUN_LABEL}" ${EXTRA}
date
