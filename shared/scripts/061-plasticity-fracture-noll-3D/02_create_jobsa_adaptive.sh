#!/bin/bash

# Define the base directory where the simulation folders are located
# Get the current directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Get the name of the folder in which the bash script is located
working_dir=$(basename "$SCRIPT_DIR")

# Ensure HPC_SCRATCH is defined
if [ -z "$HPC_SCRATCH" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

# Create the base working directory if it doesn't exist
BASE_DIR="${HPC_SCRATCH}/${working_dir}"

# Define the directory where the job template is located
JOB_TEMPLATE_DIR="./00_jobs"
JOB_TEMPLATE_PATH="${JOB_TEMPLATE_DIR}/job_template_adaptive.sh"

# Function to extract WSTEG parameter from folder name
extract_wsteg() {
    local folder_name=$1
    local wsteg_value=${folder_name#*WSTEG}
    echo "$wsteg_value"
}

# Function to generate a job script for a given simulation folder
generate_job_script() {
    local folder_name=$1
    local job_name=$2
    local wsteg_value=$3

    # Fixed values for the placeholders in job_script.sh
    local nholes=6
    local dhole=1.0

    # NEW: two separate NL parameters
    local nl_effective=5
    local nl_fracture=5

    local mesh_file="mesh_fracture_adaptive.xdmf"
    local lam_micro_param=1.0
    local mue_micro_param=1.0
    local gc_micro_param=1.0
    local eps_param=0.1
    local element_order=1

    sed -e "s|{FOLDER_NAME}|${folder_name}|g" \
        -e "s|{JOB_NAME}|${job_name}|g" \
        -e "s|{WSTEG}|${wsteg_value}|g" \
        -e "s|{NHOLES}|${nholes}|g" \
        -e "s|{DHOLE}|${dhole}|g" \
        -e "s|{NL_EFFECTIVE}|${nl_effective}|g" \
        -e "s|{NL_FRACTURE}|${nl_fracture}|g" \
        -e "s|{MESH_FILE}|${mesh_file}|g" \
        -e "s|{LAM_MICRO_PARAM}|${lam_micro_param}|g" \
        -e "s|{MUE_MICRO_PARAM}|${mue_micro_param}|g" \
        -e "s|{GC_MICRO_PARAM}|${gc_micro_param}|g" \
        -e "s|{EPS_PARAM}|${eps_param}|g" \
        -e "s|{ELEMENT_ORDER}|${element_order}|g" \
        "${JOB_TEMPLATE_PATH}" > "${BASE_DIR}/${folder_name}/job_script_adaptive.sh"
}

# Iterate over each simulation folder in the base directory
for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ -d "${folder_path}" ]; then
        folder_name=$(basename "${folder_path}")
        wsteg_value=$(extract_wsteg "${folder_name}")
        job_name="wsteg_job"

        generate_job_script "${folder_name}" "${job_name}" "${wsteg_value}"
    fi
done












