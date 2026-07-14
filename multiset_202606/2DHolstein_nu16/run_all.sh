#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOND_DIMS="${BOND_DIMS:-16 32 64}"
DRY_RUN="${DRY_RUN:-0}"

COUPLINGS=(G0.25 G0.5 G1 G2 G4)
METHODS=(multiset singleset)

for coupling in "${COUPLINGS[@]}"; do
    if [[ "$coupling" == "G2" ]]; then
        SIZES=(33 55 77 99)
    else
        SIZES=(55 77 99)
    fi

    for size in "${SIZES[@]}"; do
        for method in "${METHODS[@]}"; do
            if [[ "$method" == "multiset" ]]; then
                prefix="m"
            else
                prefix="s"
            fi
            script="$ROOT_DIR/$coupling/Holstein_2D$size/$method/run_gpu_$method.sh"
            if [[ ! -f "$script" ]]; then
                echo "Skip missing script: $script"
                continue
            fi

            for bond_dim in $BOND_DIMS; do
                workdir="$(dirname "$script")"
                output_xlsx="H${size}_2D_${prefix}${bond_dim}.xlsx"
                output_npz="H${size}_2D_${prefix}${bond_dim}.npz"
                log_file="${coupling}_H${size}_2D_${prefix}${bond_dim}.log"
                job_name="${coupling}_H${size}_${prefix}${bond_dim}"
                cmd=(
                    sbatch
                    --job-name="$job_name"
                    --output="$workdir/$log_file"
                    --error="$workdir/$log_file"
                    --chdir="$workdir"
                    --export="ALL,MAX_BONDDIM=$bond_dim,OUTPUT_XLSX=$output_xlsx,OUTPUT_NPZ=$output_npz"
                    "$script"
                )

                if [[ "$DRY_RUN" == "1" ]]; then
                    printf '%q ' "${cmd[@]}"
                    printf '\n'
                else
                    "${cmd[@]}"
                fi
            done
        done
    done
done
