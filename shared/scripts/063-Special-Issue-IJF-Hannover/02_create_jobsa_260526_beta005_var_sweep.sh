#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKING_DIR=$(basename "$SCRIPT_DIR")
CAMPAIGN_TAG="260526_beta005_var"
SIMULATION_GLOB="simulation_*_CAMPAIGN${CAMPAIGN_TAG}_*"
JOB_SCRIPT_NAME="job_script_260526_beta005_var.sh"
JOB_TEMPLATE_PATH="${SCRIPT_DIR}/00_jobs/job_template_260504_sweep.sh"

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

BASE_DIR="${HPC_SCRATCH}/${WORKING_DIR}"

read_parameter() {
    local file=$1
    local key=$2
    sed -n "s|^${key}=||p" "$file" | tail -n 1
}

for folder_path in "${BASE_DIR}"/${SIMULATION_GLOB}; do
    if [ ! -d "$folder_path" ]; then
        continue
    fi

    folder_name=$(basename "$folder_path")
    params_file="${folder_path}/run_parameters.txt"
    if [ ! -f "$params_file" ]; then
        echo "Skipping ${folder_name}: missing run_parameters.txt"
        continue
    fi

    split=$(read_parameter "$params_file" "split")
    epsilon=$(read_parameter "$params_file" "epsilon")
    input_root_name=$(read_parameter "$params_file" "input_root_name")
    epsilon_job_label=${epsilon//./}
    job_name="063_${CAMPAIGN_TAG}_${split}_eps${epsilon_job_label}"

    sed -e "s|{FOLDER_NAME}|${folder_name}|g" \
        -e "s|{JOB_NAME}|${job_name}|g" \
        -e "s|{SPLIT}|${split}|g" \
        -e "s|{EPSILON}|${epsilon}|g" \
        -e "s|{INPUT_ROOT_NAME}|${input_root_name}|g" \
        "$JOB_TEMPLATE_PATH" > "${folder_path}/${JOB_SCRIPT_NAME}"

    echo "Created ${folder_path}/${JOB_SCRIPT_NAME}"
done
