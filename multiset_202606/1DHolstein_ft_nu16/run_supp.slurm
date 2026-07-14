#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Supplementary jobs:
# 1. Submit the zero-temperature control group.
# 2. Rerun OOM finite-temperature multiset 64bd jobs: T*=0.25, 0.5, 2, 4.

T0_BOND_DIMS="${T0_BOND_DIMS:-16 32 64}"
T0_MODES="${T0_MODES:-m s}"
RERUN_TEMPERATURES="${RERUN_TEMPERATURES:-0.25 0.5 2 4}"
RERUN_BOND_DIMS="${RERUN_BOND_DIMS:-64}"
RERUN_MODES="${RERUN_MODES:-m}"

N="${N:-31}"
OMEGA_0="${OMEGA_0:-1.0}"
J="${J:-1.0}"
G="${G:-1.0}"
NU_MAX="${NU_MAX:-16}"
MODEL_SCHEME="${MODEL_SCHEME:-4}"
PERIODIC="${PERIODIC:-0}"

EVOLVE_DT="${EVOLVE_DT:-0.1}"
N_SNAPSHOTS="${N_SNAPSHOTS:-500}"
IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"

SUPP_MEM="${SUPP_MEM:-1024G}"
DRY_RUN="${DRY_RUN:-0}"

submit_job() {
    local temperature="$1"
    local bond_dim="$2"
    local mode="$3"
    local reason="$4"

    local temp_tag="${temperature%.0}"
    local temp_dir="${SCRIPT_DIR}/Holstein31_T${temp_tag}"
    local mode_tag
    local mode_label
    local workdir
    local script

    case "${mode}" in
        m|multiset)
            mode_tag="m"
            mode_label="multiset"
            workdir="${temp_dir}/multiset"
            script="${workdir}/run_gpu_multiset.sh"
            ;;
        s|singleset)
            mode_tag="s"
            mode_label="singleset"
            workdir="${temp_dir}/singleset"
            script="${workdir}/run_gpu_singleset.sh"
            ;;
        *)
            echo "Unknown mode: ${mode}. Use m/s or multiset/singleset." >&2
            exit 1
            ;;
    esac

    if [[ ! -d "${workdir}" ]]; then
        echo "Missing work directory: ${workdir}" >&2
        exit 1
    fi
    if [[ ! -f "${script}" ]]; then
        echo "Missing run script: ${script}" >&2
        exit 1
    fi

    local job_name="H31_T${temp_tag}_${mode_tag}${bond_dim}_${reason}"
    local log_path="${workdir}/${job_name}.log"
    local output_xlsx="Holstein31_T${temp_tag}_${bond_dim}bd_${mode_label}.xlsx"
    local output_npz="Holstein31_T${temp_tag}_${bond_dim}bd_${mode_label}.npz"
    local export_vars="ALL,N=${N},OMEGA_0=${OMEGA_0},J=${J},G=${G},NU_MAX=${NU_MAX},MODEL_SCHEME=${MODEL_SCHEME},PERIODIC=${PERIODIC},MAX_BONDDIM=${bond_dim},EVOLVE_DT=${EVOLVE_DT},N_SNAPSHOTS=${N_SNAPSHOTS},IF_STARTUP_SUBSTEPS=${IF_STARTUP_SUBSTEPS},STARTUP_SUBSTEPS_N=${STARTUP_SUBSTEPS_N},OUTPUT_XLSX=${output_xlsx},OUTPUT_NPZ=${output_npz},JOB_NAME=H31_T${temp_tag}_${mode_tag}${bond_dim}"

    local cmd=(
        sbatch
        --job-name="${job_name}"
        --mem="${SUPP_MEM}"
        --output="${log_path}"
        --error="${log_path}"
        --chdir="${workdir}"
        --export="${export_vars}"
        "${script}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
}

for bond_dim in ${T0_BOND_DIMS}; do
    for mode in ${T0_MODES}; do
        submit_job "0" "${bond_dim}" "${mode}" "suppT0"
    done
done

for temperature in ${RERUN_TEMPERATURES}; do
    for bond_dim in ${RERUN_BOND_DIMS}; do
        for mode in ${RERUN_MODES}; do
            submit_job "${temperature}" "${bond_dim}" "${mode}" "rerun"
        done
    done
done
