#!/bin/bash

# Define specific values for WSTEG (array of values to vary)
WSTEG_VALUES=(0.25 0.375 0.4 0.6 0.75 1.0 2.0 3.0 4.0)  # Example WSTEG values
INCLUSION_STIFFNESS_SCALES=(0.5 1.0 1.5)
INCLUSION_GC_SCALES=(0.5 1.0 1.5)

# Define the template folder
TEMPLATE_FOLDER="000_template"

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
BASE_WORKING_DIR="${HPC_SCRATCH}/${working_dir}"
mkdir -p "$BASE_WORKING_DIR"

# Function to replicate the template folder for each WSTEG value
replicate_folder() {
    local wsteg_value=$1
    local stiffness_scale=$2
    local gc_scale=$3

    # Create a unique folder name that includes all parameter-study values
    current_time=$(date +%Y%m%d_%H%M%S)
    folder_name="simulation_${current_time}_WSTEG${wsteg_value}_KINC${stiffness_scale}_GCINC${gc_scale}"

    # Create the new directory
    mkdir -p "${BASE_WORKING_DIR}/${folder_name}"
    
    # Create the scratch/as12vapa directory inside the new simulation folder
    #mkdir -p "${BASE_WORKING_DIR}/${folder_name}/scratch/as12vapa"

    # Copy the contents of the template folder to the new directory
    rsync -av --exclude='000_template' "${SCRIPT_DIR}/${TEMPLATE_FOLDER}/" "${BASE_WORKING_DIR}/${folder_name}/"
}

# Iterate over all WSTEG values
for wsteg_value in "${WSTEG_VALUES[@]}"; do
    for stiffness_scale in "${INCLUSION_STIFFNESS_SCALES[@]}"; do
        for gc_scale in "${INCLUSION_GC_SCALES[@]}"; do
            replicate_folder "$wsteg_value" "$stiffness_scale" "$gc_scale"
        done
    done
done








