#!/bin/bash
# 4-rank footprint measurement: are the scatter-S / stacked-W templates
# replicated per rank, or sharded?  Expansion is skipped, so this only builds
# the model and the grouping templates.
set -o pipefail
source /software/modules/init/bash
module use /software/modules/modulefiles/devtools
module load openmpi/4.1.8-gnu-13.3.0
source /software/envs/anaconda3.env
conda activate reno
export PYTHONUNBUFFERED=1 SKIP_EXPAND=1 MAX_BONDDIM=64 NU_MAX=8
export RENO_MPI_KRYLOV_MODE=local RENO_MPI_ALLREDUCE_MODE=host
export OMPI_MCA_btl="^openib" OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export RESULT_DIR=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling/results
cd /curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling
for N in "$@"; do
  export NROW=$N NCOL=$N RUN_TAG="mpifootprint_${N}x${N}_m64"
  echo "=================== 4 rank ${N}x${N} ==================="
  nice -n 10 mpirun -n 4 --oversubscribe python multiset_parallel/run_scaling.py 2>&1 \
    | grep -E "static sizes|RESULT|FAILED" || echo "FAIL at ${N}x${N}"
done
echo "mpi footprint ladder finished"
