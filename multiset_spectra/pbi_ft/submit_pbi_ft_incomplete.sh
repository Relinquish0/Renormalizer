#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/curie-home/zengjj/.conda/envs/reno/bin/python}"

MODELS=(dimer trimer hexamer)
BOND_DIMS=(64 128)
SPECTRA_TYPES=(abs emi)
METHODS=(singleset multiset)

EVOLVE_DT="${EVOLVE_DT:-20}"
NSTEPS="${NSTEPS:-5000}"
INSTEPS="${INSTEPS:-50}"
TEMPERATURE_K="${TEMPERATURE_K:-298}"
COMPRESS_THRESHOLD="${COMPRESS_THRESHOLD:-1e-8}"
EXPAND="${EXPAND:-1}"
THERMAL_INIT_METHOD="${THERMAL_INIT_METHOD:-imaginary_time_propagate}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ "${1:-}" == "-n" ]]; then
    DRY_RUN=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

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

is_complete() {
    local output_npz="$1"
    local expected_len="$2"
    local expected_final_time="$3"

    "$PYTHON_BIN" - "$output_npz" "$expected_len" "$expected_final_time" <<'PY'
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
expected_len = int(sys.argv[2])
expected_final_time = float(sys.argv[3])

if not path.exists():
    print("missing")
    sys.exit(1)

try:
    data = np.load(path, allow_pickle=True)
except Exception as exc:
    print(f"unreadable: {exc.__class__.__name__}: {exc}")
    sys.exit(1)

if "time_series" in data:
    time_series = np.asarray(data["time_series"])
elif "time series" in data:
    time_series = np.asarray(data["time series"])
else:
    print("missing time_series")
    sys.exit(1)

if "autocorr" not in data:
    print("missing autocorr")
    sys.exit(1)

autocorr = np.asarray(data["autocorr"])
if time_series.size < expected_len:
    print(f"short time_series: {time_series.size} < {expected_len}")
    sys.exit(1)
if autocorr.size < expected_len:
    print(f"short autocorr: {autocorr.size} < {expected_len}")
    sys.exit(1)
if not np.isclose(float(time_series[expected_len - 1]), expected_final_time):
    print(
        "wrong final time: "
        f"{float(time_series[expected_len - 1])} != {expected_final_time}"
    )
    sys.exit(1)
if not np.all(np.isfinite(time_series[:expected_len])):
    print("non-finite time_series")
    sys.exit(1)
if not np.all(np.isfinite(autocorr[:expected_len])):
    print("non-finite autocorr")
    sys.exit(1)

print("complete")
sys.exit(0)
PY
}

job_is_queued() {
    local job_name="$1"

    if ! command -v squeue >/dev/null 2>&1; then
        return 1
    fi

    [[ -n "$(squeue -h -n "$job_name" 2>/dev/null || true)" ]]
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
    local output_npz
    local expected_len
    local expected_final_time
    local status

    display="$(display_name "$model")"
    suffix="$(method_suffix "$method")"
    job_tag="${display}_${spectra_type}_${suffix}${bond_dim}"
    job_dir="${ROOT_DIR}/${model}/${spectra_type}/${method}"
    script="${job_dir}/run_gpu_${method}.sh"
    output_npz="${job_dir}/${job_tag}.npz"
    expected_len=$((NSTEPS + 1))
    expected_final_time="$("$PYTHON_BIN" -c "print(float(${EVOLVE_DT}) * int(${NSTEPS}))")"

    if [[ ! -f "$script" ]]; then
        echo "Missing script: $script" >&2
        return 1
    fi

    if status="$(is_complete "$output_npz" "$expected_len" "$expected_final_time")"; then
        echo "Complete: ${job_tag}"
        return 0
    fi

    echo "Incomplete: ${job_tag} (${status})"

    if job_is_queued "$job_tag"; then
        echo "Already queued/running: ${job_tag}"
        return 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "Dry run: would submit ${job_tag}"
        return 0
    fi

    echo "Submitting ${job_tag}"
    sbatch \
        --job-name="${job_tag}" \
        --output="${job_dir}/${job_tag}.log" \
        --error="${job_dir}/${job_tag}.log" \
        --export=ALL,MAX_BONDDIM="${bond_dim}",EVOLVE_DT="${EVOLVE_DT}",NSTEPS="${NSTEPS}",INSTEPS="${INSTEPS}",TEMPERATURE_K="${TEMPERATURE_K}",COMPRESS_THRESHOLD="${COMPRESS_THRESHOLD}",EXPAND="${EXPAND}",THERMAL_INIT_METHOD="${THERMAL_INIT_METHOD}" \
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
