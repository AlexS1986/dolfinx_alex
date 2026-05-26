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

# Function to extract parameter-study tokens from folder names
extract_token() {
    local folder_name=$1
    local token=$2
    sed -n "s|.*${token}\\([^_]*\\).*|\\1|p" <<< "${folder_name}"
}

multiply() {
    awk "BEGIN {print $1 * $2}"
}

# Function to generate a job script for a given simulation folder
generate_job_script() {
    local folder_name=$1
    local job_name=$2
    local wsteg_value=$3

    # Fixed values for the placeholders in job_script.sh
    local nholes=6
    local dhole=1.0
    local e0=0.02
    local e1=0.6
    local mesh_file="mesh_fracture_adaptive.xdmf"
    local lam_matrix_param=1.0
    local mue_matrix_param=1.0
    local gc_matrix_param=1.0
    local sig_y_matrix_param=1.0
    local hard_matrix_param=0.2222222
    local stiffness_scale=$4
    local gc_scale=$5
    local lam_inclusion_param
    local mue_inclusion_param
    local gc_inclusion_param
    local sig_y_inclusion_param
    local hard_inclusion_param=${hard_matrix_param}
    lam_inclusion_param=$(multiply "${lam_matrix_param}" "${stiffness_scale}")
    mue_inclusion_param=$(multiply "${mue_matrix_param}" "${stiffness_scale}")
    gc_inclusion_param=$(multiply "${gc_matrix_param}" "${gc_scale}")
    # Keep the yield stress fixed; only stiffness and fracture resistance vary.
    sig_y_inclusion_param=${sig_y_matrix_param}
    local eps_param=0.1
    local element_order=1

    # Read the template and replace placeholders
    sed -e "s|{FOLDER_NAME}|${folder_name}|g" \
        -e "s|{JOB_NAME}|${job_name}|g" \
        -e "s|{WSTEG}|${wsteg_value}|g" \
        -e "s|{NHOLES}|${nholes}|g" \
        -e "s|{DHOLE}|${dhole}|g" \
        -e "s|{E0}|${e0}|g" \
        -e "s|{E1}|${e1}|g" \
        -e "s|{MESH_FILE}|${mesh_file}|g" \
        -e "s|{LAM_MATRIX_PARAM}|${lam_matrix_param}|g" \
        -e "s|{MUE_MATRIX_PARAM}|${mue_matrix_param}|g" \
        -e "s|{GC_MATRIX_PARAM}|${gc_matrix_param}|g" \
        -e "s|{SIG_Y_MATRIX_PARAM}|${sig_y_matrix_param}|g" \
        -e "s|{HARD_MATRIX_PARAM}|${hard_matrix_param}|g" \
        -e "s|{LAM_INCLUSION_PARAM}|${lam_inclusion_param}|g" \
        -e "s|{MUE_INCLUSION_PARAM}|${mue_inclusion_param}|g" \
        -e "s|{GC_INCLUSION_PARAM}|${gc_inclusion_param}|g" \
        -e "s|{SIG_Y_INCLUSION_PARAM}|${sig_y_inclusion_param}|g" \
        -e "s|{HARD_INCLUSION_PARAM}|${hard_inclusion_param}|g" \
        -e "s|{EPS_PARAM}|${eps_param}|g" \
        -e "s|{ELEMENT_ORDER}|${element_order}|g" \
        "${JOB_TEMPLATE_PATH}" > "${BASE_DIR}/${folder_name}/job_script_adaptive.sh"
}

# Iterate over each simulation folder in the base directory
for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ -d "${folder_path}" ]; then
        folder_name=$(basename "${folder_path}")

        # Extract parameter-study values from folder name
        wsteg_value=$(extract_token "${folder_name}" "WSTEG")
        stiffness_scale=$(extract_token "${folder_name}" "KINC")
        gc_scale=$(extract_token "${folder_name}" "GCINC")

        if [ -z "${wsteg_value}" ] || [ -z "${stiffness_scale}" ] || [ -z "${gc_scale}" ]; then
            echo "Skipping ${folder_name}: could not extract WSTEG/KINC/GCINC values."
            continue
        fi

        job_name="inc_K${stiffness_scale}_G${gc_scale}"

        # Call generate_job_script with the folder name and WSTEG value
        generate_job_script "${folder_name}" "${job_name}" "${wsteg_value}" "${stiffness_scale}" "${gc_scale}"
    fi
done








