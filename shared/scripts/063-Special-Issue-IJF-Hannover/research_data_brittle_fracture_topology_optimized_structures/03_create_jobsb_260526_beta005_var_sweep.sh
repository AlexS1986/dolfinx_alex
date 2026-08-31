#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKING_DIR=$(basename "$SCRIPT_DIR")
CAMPAIGN_TAG="260526_beta005_var"
SIMULATION_GLOB="simulation_*_CAMPAIGN${CAMPAIGN_TAG}_*"
JOB_SCRIPT_NAME="job_script_260526_beta005_var.sh"
MEMORY_VALUE="${MEMORY_VALUE:-4000}"
PROCESSOR_NUMBER="${PROCESSOR_NUMBER:-6}"
TIME="${TIME:-10080}"

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

BASE_DIR="${HPC_SCRATCH}/${WORKING_DIR}"

for folder_path in "${BASE_DIR}"/${SIMULATION_GLOB}; do
    if [ ! -d "$folder_path" ]; then
        continue
    fi

    job_script_path="${folder_path}/${JOB_SCRIPT_NAME}"
    if [ ! -f "$job_script_path" ]; then
        echo "Skipping $(basename "$folder_path"): missing ${JOB_SCRIPT_NAME}"
        continue
    fi

    sed -i -e "s|{MEMORY_VALUE}|${MEMORY_VALUE}|g" \
           -e "s|{PROCESSOR_NUMBER}|${PROCESSOR_NUMBER}|g" \
           -e "s|{TIME}|${TIME}|g" \
           "$job_script_path"

    echo "Updated resources for $(basename "$folder_path")"
done
