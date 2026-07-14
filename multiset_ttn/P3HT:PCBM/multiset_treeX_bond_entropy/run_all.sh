#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOND_DIMS="${BOND_DIMS:-16 32}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT="$ROOT_DIR/run_gpu_multiset_treeX_bond_entropy.sh"
if [[ ! -f "$SCRIPT" ]]; then
    echo "Missing script: $SCRIPT" >&2
    exit 1
fi

for bond_dim in $BOND_DIMS; do
    log_file="P3HT_treeX_bond_entropy_m${bond_dim}.log"
    job_name="P3HT_treeX_BE_m${bond_dim}"
    cmd=(
        sbatch
        --job-name="$job_name"
        --output="$ROOT_DIR/$log_file"
        --error="$ROOT_DIR/$log_file"
        --chdir="$ROOT_DIR"
        --export="ALL,MAX_BONDDIM=$bond_dim,JOB_NAME=p3ht_ms_ttn_treeX_bond_entropy,WORK_DIR=$ROOT_DIR"
        "$SCRIPT"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${cmd[@]}"
        printf '%s\n' ''
    else
        "${cmd[@]}"
    fi
done
