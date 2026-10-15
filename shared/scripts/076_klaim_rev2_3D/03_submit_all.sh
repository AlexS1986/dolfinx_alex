#!/bin/bash
# Submit job_script.sh of every case folder. Usage: ./03_submit_all.sh [--dry-run] [--max N] [--match TEXT]
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi
BASE_DIR="${HPC_SCRATCH}/${PROJECT_NAME}"
DRY_RUN=0
MAX_SUBMISSIONS=""
MATCH_TEXT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --max) MAX_SUBMISSIONS="$2"; shift 2 ;;
        --match) MATCH_TEXT="$2"; shift 2 ;;
        -h|--help) echo "Usage: $0 [--dry-run] [--max N] [--match TEXT]"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

submitted=0
for folder_path in "${BASE_DIR}"/*/; do
    [ -d "$folder_path" ] || continue
    folder_name=$(basename "$folder_path")
    if [ -n "$MATCH_TEXT" ] && [[ "$folder_name" != *"$MATCH_TEXT"* ]]; then
        continue
    fi
    job_script="${folder_path}job_script.sh"
    if [ ! -f "$job_script" ]; then
        echo "missing job script: $job_script"
        continue
    fi
    if [ -n "$MAX_SUBMISSIONS" ] && [ "$submitted" -ge "$MAX_SUBMISSIONS" ]; then
        break
    fi
    if ls "${folder_path}"/*.out.* >/dev/null 2>&1; then
        echo "already has SLURM output, skipping (delete *.out.* to resubmit): $folder_name"
        continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would submit: $job_script"
    else
        echo "submitting: $job_script"
        sbatch "$job_script"
    fi
    submitted=$((submitted + 1))
done
echo "$submitted jobs $([ "$DRY_RUN" -eq 1 ] && echo 'would be ')submitted."
