#!/bin/bash
#SBATCH --job-name=ph_cbe_g1_M500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -p 4A100,4V100
#SBATCH --qos=normal
#SBATCH --output=ph_cbe_g1_M500.log
#SBATCH --error=ph_cbe_g1_M500.log

source $HOME/.bashrc
source /software/envs/anaconda3.env

module load cuda/12.4
export CUPY_ACCELERATORS=cutensor

conda activate reno
export PYTHONUNBUFFERED=1

export NSITES="${NSITES:-100}"
export U="${U:-10}"
export OMEGA_PH="${OMEGA_PH:-3}"
export G="${G:-1}"
export HOPPING_T="${HOPPING_T:-1}"
export NPH_MAX="${NPH_MAX:-8}"
export WIDTH="${WIDTH:-4}"
export X0="${X0:-25}"
export K="${K:-1.5707963267948966}"
export EVOLVE_DT="${EVOLVE_DT:-0.05}"
export EVOLVE_TIME="${EVOLVE_TIME:-40}"
export SAVE_EVERY="${SAVE_EVERY:-10}"
export MAX_BONDDIM="${MAX_BONDDIM:-500}"
export CBE_EPS_PRE="${CBE_EPS_PRE:-1e-4}"
export CBE_EPS_FINAL="${CBE_EPS_FINAL:-1e-6}"
export CBE_EPS_TRIM="${CBE_EPS_TRIM:-1e-12}"
export OUT_DIR="${OUT_DIR:-results}"
export JOB_NAME="${JOB_NAME:-peierls_hubbard_g1_L${NSITES}_M${MAX_BONDDIM}}"

echo "=== Environment Information ==="
which python
python --version
nvcc --version
echo "==============================="
echo "Starting Peierls-Hubbard CBE-TDVP run"
python PeierlsHubbardmodel.py
echo "Job completed with exit code: $?"
echo "Ending"
