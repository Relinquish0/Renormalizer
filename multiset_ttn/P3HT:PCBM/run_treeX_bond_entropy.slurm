#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BOND_DIMS="${BOND_DIMS:-16 32}"
MODES="${MODES:-s m}"

DT_FS="${DT_FS:-1.0}"
TOTAL_FS="${TOTAL_FS:-200.0}"
INITIAL_SITE="${INITIAL_SITE:-0}"
PRINT_ONLY="${PRINT_ONLY:-0}"
ENTROPY_LOG_INTERVAL="${ENTROPY_LOG_INTERVAL:-20}"

DRY_RUN="${DRY_RUN:-0}"

for bond_dim in ${BOND_DIMS}; do
    for mode in ${MODES}; do
        case "${mode}" in
            s|singleset)
                mode_tag="ss"
                workdir="${SCRIPT_DIR}/singleset_treeX"
                script="${workdir}/run_gpu_singleset_treeX.sh"
                job_name="P3HT_ss_treeX_m${bond_dim}"
                log_path="${workdir}/P3HT_singleset_treeX_m${bond_dim}.log"
                job_output_name="p3ht_ttns_treeX_bond_entropy"
                export_vars="ALL,MAX_BONDDIM=${bond_dim},JOB_NAME=${job_output_name},DT_FS=${DT_FS},TOTAL_FS=${TOTAL_FS},INITIAL_SITE=${INITIAL_SITE},PRINT_ONLY=${PRINT_ONLY},ENTROPY_LOG_INTERVAL=${ENTROPY_LOG_INTERVAL}"
                ;;
            m|multiset)
                mode_tag="ms"
                workdir="${SCRIPT_DIR}/multiset_treeX_bond_entropy"
                script="${workdir}/run_gpu_multiset_treeX_bond_entropy.sh"
                job_name="P3HT_ms_treeX_BE_m${bond_dim}"
                log_path="${workdir}/P3HT_treeX_bond_entropy_m${bond_dim}.log"
                job_output_name="p3ht_ms_ttn_treeX_bond_entropy"
                export_vars="ALL,MAX_BONDDIM=${bond_dim},JOB_NAME=${job_output_name},DT_FS=${DT_FS},TOTAL_FS=${TOTAL_FS},INITIAL_SITE=${INITIAL_SITE},PRINT_ONLY=${PRINT_ONLY},ENTROPY_LOG_INTERVAL=${ENTROPY_LOG_INTERVAL},WORK_DIR=${workdir}"
                ;;
            *)
                echo "Unknown mode: ${mode}. Use s/singleset or m/multiset." >&2
                exit 1
                ;;
        esac

        if [[ ! -d "${workdir}" ]]; then
            echo "Missing workdir for ${mode_tag}: ${workdir}" >&2
            exit 1
        fi
        if [[ ! -f "${script}" ]]; then
            echo "Missing script for ${mode_tag}: ${script}" >&2
            exit 1
        fi

        cmd=(
            sbatch
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
