#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEMPERATURES="${TEMPERATURES:-0 0.25 0.5 1 2 4}"
SINGLESET_BOND_DIMS="${SINGLESET_BOND_DIMS:-16 32 64}"
MULTISET_BOND_DIMS="${MULTISET_BOND_DIMS:-8 16 32}"
MODES="${MODES:-m s}"

NROW="${NROW:-7}"
NCOL="${NCOL:-7}"
OMEGA_0="${OMEGA_0:-1.0}"
J="${J:-1.0}"
G="${G:-1.0}"
NU_MAX="${NU_MAX:-16}"
MODEL_SCHEME="${MODEL_SCHEME:-4}"
PERIODIC="${PERIODIC:-0}"

EVOLVE_DT="${EVOLVE_DT:-0.1}"
N_SNAPSHOTS="${N_SNAPSHOTS:-251}"
IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"

DRY_RUN="${DRY_RUN:-0}"

for temperature in ${TEMPERATURES}; do
    temp_tag="${temperature%.0}"
    temp_dir="${SCRIPT_DIR}/Holstein77_T${temp_tag}"

    if [[ ! -d "${temp_dir}" ]]; then
        echo "Missing temperature directory: ${temp_dir}" >&2
        exit 1
    fi

    for mode in ${MODES}; do
        case "${mode}" in
            m|multiset)
                mode_tag="m"
                mode_label="multiset"
                bond_dims="${MULTISET_BOND_DIMS}"
                workdir="${temp_dir}/multiset"
                script="${workdir}/run_gpu_multiset.sh"
                ;;
            s|singleset)
                mode_tag="s"
                mode_label="singleset"
                bond_dims="${SINGLESET_BOND_DIMS}"
                workdir="${temp_dir}/singleset"
                script="${workdir}/run_gpu_singleset.sh"
                ;;
            *)
                echo "Unknown mode: ${mode}. Use m/s or multiset/singleset." >&2
                exit 1
                ;;
        esac

        for bond_dim in ${bond_dims}; do
            job_name="H77_T${temp_tag}_${mode_tag}${bond_dim}"
            log_path="${workdir}/${job_name}.log"
            output_xlsx="Holstein77_T${temp_tag}_${bond_dim}bd_${mode_label}.xlsx"
            output_npz="Holstein77_T${temp_tag}_${bond_dim}bd_${mode_label}.npz"
            export_vars="ALL,NROW=${NROW},NCOL=${NCOL},OMEGA_0=${OMEGA_0},J=${J},G=${G},NU_MAX=${NU_MAX},MODEL_SCHEME=${MODEL_SCHEME},PERIODIC=${PERIODIC},MAX_BONDDIM=${bond_dim},EVOLVE_DT=${EVOLVE_DT},N_SNAPSHOTS=${N_SNAPSHOTS},IF_STARTUP_SUBSTEPS=${IF_STARTUP_SUBSTEPS},STARTUP_SUBSTEPS_N=${STARTUP_SUBSTEPS_N},OUTPUT_XLSX=${output_xlsx},OUTPUT_NPZ=${output_npz},JOB_NAME=${job_name}"

            cmd=(
                sbatch
                --partition=4V100
                --job-name="${job_name}"
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
        done
    done
done
