#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
working_dir=$(basename "$SCRIPT_DIR")

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

BASE_DIR="${HPC_SCRATCH}/${working_dir}"
MEMORY_VALUE="${MEMORY_VALUE:-4000}"
PROCESSOR_NUMBER="${PROCESSOR_NUMBER:-6}"
TIME="${TIME:-10080}"

for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ ! -d "$folder_path" ]; then
        continue
    fi

    job_script_path="${folder_path}/job_script_260504.sh"
    if [ ! -f "$job_script_path" ]; then
        echo "Skipping $(basename "$folder_path"): missing job_script_260504.sh"
        continue
    fi

    sed -i -e "s|{MEMORY_VALUE}|${MEMORY_VALUE}|g" \
           -e "s|{PROCESSOR_NUMBER}|${PROCESSOR_NUMBER}|g" \
           -e "s|{TIME}|${TIME}|g" \
           "$job_script_path"

    echo "Updated resources for $(basename "$folder_path")"
done
