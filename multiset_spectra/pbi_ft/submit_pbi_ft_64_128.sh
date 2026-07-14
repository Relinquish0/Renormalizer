#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=(dimer trimer hexamer)
BOND_DIMS=(64 128)
SPECTRA_TYPES=(abs emi)
METHODS=(singleset multiset)

display_name() {
    case "$1" in
        dimer) echo "Dimer" ;;
        trimer) echo "Trimer" ;;
        hexamer) echo "Hexamer" ;;
        *) echo "Unknown model: $1" >&2; return 1 ;;
    esac
}

method_suffix() {
    case "$1" in
        singleset) echo "s" ;;
        multiset) echo "m" ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

submit_one() {
    local model="$1"
    local method="$2"
    local bond_dim="$3"
    local spectra_type="$4"
    local display
    local suffix
    local job_tag
    local job_dir
    local script

    display="$(display_name "$model")"
    suffix="$(method_suffix "$method")"
    job_tag="${display}_${spectra_type}_${suffix}${bond_dim}"
    job_dir="${ROOT_DIR}/${model}/${spectra_type}/${method}"
    script="${job_dir}/run_gpu_${method}.sh"

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
    for method in "${METHODS[@]}"; do
        for bond_dim in "${BOND_DIMS[@]}"; do
            for spectra_type in "${SPECTRA_TYPES[@]}"; do
                submit_one "$model" "$method" "$bond_dim" "$spectra_type"
            done
        done
    done
done
