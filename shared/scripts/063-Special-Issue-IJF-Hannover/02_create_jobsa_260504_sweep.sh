#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
working_dir=$(basename "$SCRIPT_DIR")

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

BASE_DIR="${HPC_SCRATCH}/${working_dir}"
JOB_TEMPLATE_PATH="${SCRIPT_DIR}/00_jobs/job_template_260504_sweep.sh"

if [ ! -f "$JOB_TEMPLATE_PATH" ]; then
    echo "Error: job template not found: $JOB_TEMPLATE_PATH"
    exit 1
fi

extract_token() {
    local folder_name=$1
    local token=$2
    sed -n "s|.*${token}\\([^_]*\\).*|\\1|p" <<< "$folder_name"
}

epsilon_from_label() {
    sed 's/_/./g; s/^m/-/' <<< "$1"
}

read_parameter() {
    local file=$1
    local key=$2
    sed -n "s|^${key}=||p" "$file" | tail -n 1
}

generate_job_script() {
    local folder_name=$1
    local split=$2
    local epsilon=$3
    local input_root_name=$4
    local epsilon_job_label=${epsilon//./}
    local job_name="063_${split}_eps${epsilon_job_label}"

    sed -e "s|{FOLDER_NAME}|${folder_name}|g" \
        -e "s|{JOB_NAME}|${job_name}|g" \
        -e "s|{SPLIT}|${split}|g" \
        -e "s|{EPSILON}|${epsilon}|g" \
        -e "s|{INPUT_ROOT_NAME}|${input_root_name}|g" \
        "${JOB_TEMPLATE_PATH}" > "${BASE_DIR}/${folder_name}/job_script_260504.sh"
}

for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ ! -d "$folder_path" ]; then
        continue
    fi

    folder_name=$(basename "$folder_path")
    params_file="${folder_path}/run_parameters.txt"

    if [ -f "$params_file" ]; then
        split=$(read_parameter "$params_file" "split")
        epsilon=$(read_parameter "$params_file" "epsilon")
        input_root_name=$(read_parameter "$params_file" "input_root_name")
    else
        split=$(extract_token "$folder_name" "SPLIT")
        epsilon=$(epsilon_from_label "$(extract_token "$folder_name" "EPS")")
        input_root_name="260504_input"
    fi

    if [ -z "$split" ] || [ -z "$epsilon" ]; then
        echo "Skipping ${folder_name}: could not extract split/epsilon."
        continue
    fi

    generate_job_script "$folder_name" "$split" "$epsilon" "$input_root_name"
    echo "Created job script for $folder_name"
done
