#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKING_DIR=$(basename "$SCRIPT_DIR")
CAMPAIGN_TAG="260526_beta005_var"
SIMULATION_GLOB="simulation_*_CAMPAIGN${CAMPAIGN_TAG}_*"
JOB_SCRIPT_NAME="job_script_260526_beta005_var.sh"
DRY_RUN=0
MAX_SUBMISSIONS=""

usage() {
    echo "Usage: $0 [--dry-run] [--max N]"
    echo "  --dry-run   Print jobs without calling sbatch."
    echo "  --max N     Submit at most N jobs."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --max)
            MAX_SUBMISSIONS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

BASE_DIR="${HPC_SCRATCH}/${WORKING_DIR}"
submitted=0
missing=0

for folder_path in "${BASE_DIR}"/${SIMULATION_GLOB}; do
    if [ ! -d "$folder_path" ]; then
        continue
    fi

    job_script_path="${folder_path}/${JOB_SCRIPT_NAME}"
    if [ ! -f "$job_script_path" ]; then
        echo "Missing job script: ${job_script_path}"
        missing=$((missing + 1))
        continue
    fi

    if [ -n "$MAX_SUBMISSIONS" ] && [ "$submitted" -ge "$MAX_SUBMISSIONS" ]; then
        break
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "Would submit: ${job_script_path}"
    else
        echo "Submitting: ${job_script_path}"
        sbatch "$job_script_path"
    fi
    submitted=$((submitted + 1))
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run complete: ${submitted} jobs would be submitted, ${missing} folders missing job scripts."
else
    echo "Submitted ${submitted} jobs, ${missing} folders missing job scripts."
fi
