#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALPHAS="${ALPHAS:-0.5 1 2}"
BOND_DIMS="${BOND_DIMS:-16 32 64}"
MODES="${MODES:-m s}"

N="${N:-31}"
OMEGA_0="${OMEGA_0:-1.0}"
J="${J:-1.0}"
G="${G:-1.0}"
NU_MAX="${NU_MAX:-16}"
MODEL_SCHEME="${MODEL_SCHEME:-4}"

EVOLVE_DT="${EVOLVE_DT:-0.1}"
N_SNAPSHOTS="${N_SNAPSHOTS:-500}"
IF_STARTUP_SUBSTEPS="${IF_STARTUP_SUBSTEPS:-1}"
STARTUP_SUBSTEPS_N="${STARTUP_SUBSTEPS_N:-10}"
LONG_RANGE_PERIODIC="${LONG_RANGE_PERIODIC:-0}"

DRY_RUN="${DRY_RUN:-0}"

for alpha in ${ALPHAS}; do
    alpha_tag="${alpha%.0}"
    alpha_dir="${SCRIPT_DIR}/Holstein31_alpha${alpha_tag}"

    if [[ ! -d "${alpha_dir}" ]]; then
        echo "Missing alpha directory: ${alpha_dir}" >&2
        exit 1
    fi

    for bond_dim in ${BOND_DIMS}; do
        for mode in ${MODES}; do
            case "${mode}" in
                m|multiset)
                    mode_tag="m"
                    workdir="${alpha_dir}/multiset"
                    script="${workdir}/run_gpu_multiset.sh"
                    ;;
                s|singleset)
                    mode_tag="s"
                    workdir="${alpha_dir}/singleset"
                    script="${workdir}/run_gpu_singleset.sh"
                    ;;
                *)
                    echo "Unknown mode: ${mode}. Use m/s or multiset/singleset." >&2
                    exit 1
                    ;;
            esac

            job_name="H31_a${alpha_tag}_${mode_tag}${bond_dim}"
            log_path="${workdir}/${job_name}.log"
            export_vars="ALL,N=${N},OMEGA_0=${OMEGA_0},J=${J},G=${G},NU_MAX=${NU_MAX},MODEL_SCHEME=${MODEL_SCHEME},MAX_BONDDIM=${bond_dim},EVOLVE_DT=${EVOLVE_DT},N_SNAPSHOTS=${N_SNAPSHOTS},IF_STARTUP_SUBSTEPS=${IF_STARTUP_SUBSTEPS},STARTUP_SUBSTEPS_N=${STARTUP_SUBSTEPS_N},LONG_RANGE_PERIODIC=${LONG_RANGE_PERIODIC}"

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
done
