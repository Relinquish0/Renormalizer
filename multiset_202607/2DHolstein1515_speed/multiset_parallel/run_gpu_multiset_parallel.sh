#!/bin/bash
#SBATCH --job-name=H1515_m16_p4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=1024G
#SBATCH -p 4V100
#SBATCH --qos=4gpu
#SBATCH --output=H1515_m16_parallel_%j.log
#SBATCH --error=H1515_m16_parallel_%j.log
#SBATCH --chdir=/curie-home/zengjj/Renormalizer/multiset_202607/2DHolstein1515_speed/multiset_parallel

set -euo pipefail

echo "=== Environment Information ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PWD=$(pwd)"
ls -la Holstein1515.py mpi_apply_hop_patch.py mpi_expand_patch.py mpi_krylov_patch.py

source /software/modules/init/bash
module use /software/modules/modulefiles/devtools
module use /software/modules/modulefiles/apps
module load openmpi/4.1.8-gnu-13.3.0
module load cuda/12.4

set +u
source /software/envs/anaconda3.env
conda activate reno
set -u

which python
python --version
python - <<'PY'
import mpi4py
from mpi4py import MPI
print("mpi4py", mpi4py.__version__)
print("MPI", MPI.Get_library_version().splitlines()[0])
PY
nvcc --version

export PYTHONUNBUFFERED=1
export CUPY_ACCELERATORS=cutensor
export RENO_MPI_KRYLOV_MODE="${RENO_MPI_KRYLOV_MODE:-distributed}"
export RENO_MPI_ALLREDUCE_MODE="${RENO_MPI_ALLREDUCE_MODE:-host}"
export RENO_MPI_DEBUG_SHAPES="${RENO_MPI_DEBUG_SHAPES:-0}"
export RENO_MPI_HOP_PROGRESS_INTERVAL="${RENO_MPI_HOP_PROGRESS_INTERVAL:-200}"

if [ "${RENO_MPI_ALLREDUCE_MODE}" = "cuda" ] && [ "${RENO_MPI_ALLOW_UNSAFE_CUDA_MPI:-0}" != "1" ]; then
    echo "RENO_MPI_ALLREDUCE_MODE=cuda requested, but Curie OpenMPI GPU-buffer collectives segfault in mpi_cuda_aware_bench."
    echo "Falling back to RENO_MPI_ALLREDUCE_MODE=host. Set RENO_MPI_ALLOW_UNSAFE_CUDA_MPI=1 only for explicit crash testing."
    export RENO_MPI_ALLREDUCE_MODE="host"
fi

export OMPI_MCA_btl="^openib"
export OMPI_MCA_btl_vader_single_copy_mechanism="none"

export NROW="${NROW:-15}"
export NCOL="${NCOL:-15}"
export OMEGA_0="${OMEGA_0:-1.0}"
export J="${J:-1.0}"
export G="${G:-0.5}"
export NU_MAX="${NU_MAX:-8}"
export MAX_BONDDIM="${MAX_BONDDIM:-16}"
export EVOLVE_DT="${EVOLVE_DT:-0.1}"
export N_SNAPSHOTS="${N_SNAPSHOTS:-11}"
export IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-0}"
export STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"
export OUTPUT_XLSX="${OUTPUT_XLSX:-H1515_2D_m${MAX_BONDDIM}_speed_parallel.xlsx}"
export OUTPUT_NPZ="${OUTPUT_NPZ:-H1515_2D_m${MAX_BONDDIM}_speed_parallel.npz}"

echo "NROW=${NROW}"
echo "NCOL=${NCOL}"
echo "OMEGA_0=${OMEGA_0}"
echo "J=${J}"
echo "G=${G}"
echo "NU_MAX=${NU_MAX}"
echo "MAX_BONDDIM=${MAX_BONDDIM}"
echo "EVOLVE_DT=${EVOLVE_DT}"
echo "N_SNAPSHOTS=${N_SNAPSHOTS}"
echo "IF_STARTUP_SUBSTEPS=${IF_STARTUP_SUBSTEPS}"
echo "STARTUP_SUBSTEPS_N=${STARTUP_SUBSTEPS_N}"
echo "RENO_MPI_KRYLOV_MODE=${RENO_MPI_KRYLOV_MODE}"
echo "RENO_MPI_ALLREDUCE_MODE=${RENO_MPI_ALLREDUCE_MODE}"
echo "RENO_MPI_DEBUG_SHAPES=${RENO_MPI_DEBUG_SHAPES}"
echo "RENO_MPI_HOP_PROGRESS_INTERVAL=${RENO_MPI_HOP_PROGRESS_INTERVAL}"
echo "OUTPUT_XLSX=${OUTPUT_XLSX}"
echo "OUTPUT_NPZ=${OUTPUT_NPZ}"
echo "==============================="

echo "Starting MPI run"
if [ "${MPI_LAUNCHER:-srun}" = "mpiexec" ]; then
    mpiexec -n "${SLURM_NTASKS}" python Holstein1515.py
else
    srun --mpi=pmix -n "${SLURM_NTASKS}" python Holstein1515.py
fi
status=$?
echo "Job completed with exit code: ${status}"

echo "Ending"
date
echo "==============================="

exit "${status}"
