#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=(dimer trimer hexamer)
BOND_DIMS=(32 64)

display_name() {
    case "$1" in
        dimer) echo "Dimer" ;;
        trimer) echo "Trimer" ;;
        hexamer) echo "Hexamer" ;;
        *) echo "Unknown model: $1" >&2; return 1 ;;
    esac
}

submit_one() {
    local model="$1"
    local bond_dim="$2"
    local display
    local job_tag
    local job_dir
    local script

    display="$(display_name "$model")"
    job_tag="${display}_emi_ma${bond_dim}"
    job_dir="${ROOT_DIR}/${model}"
    script="${job_dir}/run_gpu_ancilla.sh"

    if [[ ! -f "$script" ]]; then
        echo "Missing script: $script" >&2
        return 1
    fi

    echo "Submitting ${job_tag}"
    sbatch \
        --job-name="${job_tag}" \
        --output="${job_dir}/${job_tag}.log" \
        --error="${job_dir}/${job_tag}.log" \
        --export=ALL,MAX_BONDDIM="${bond_dim}",THERMAL_INIT_METHOD=imaginary_time_propagate \
        "$script"
}

for model in "${MODELS[@]}"; do
    for bond_dim in "${BOND_DIMS[@]}"; do
        submit_one "$model" "$bond_dim"
    done
done
