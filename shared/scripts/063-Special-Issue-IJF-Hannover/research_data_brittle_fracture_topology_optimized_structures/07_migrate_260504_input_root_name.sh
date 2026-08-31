#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
working_dir=$(basename "$SCRIPT_DIR")
OLD_INPUT_ROOT_NAME="${OLD_INPUT_ROOT_NAME:-260504_input}"
NEW_INPUT_ROOT_NAME="${NEW_INPUT_ROOT_NAME:-260504_dcb_beta_phi_a_rho_var_min_max}"

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

BASE_DIR="${HPC_SCRATCH}/${working_dir}"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: simulation directory does not exist: $BASE_DIR"
    exit 1
fi

migrated=0

for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ ! -d "$folder_path" ]; then
        continue
    fi

    old_root="${folder_path}/resources/${OLD_INPUT_ROOT_NAME}"
    new_root="${folder_path}/resources/${NEW_INPUT_ROOT_NAME}"
    params_file="${folder_path}/run_parameters.txt"

    if [ -d "$old_root" ] && [ ! -d "$new_root" ]; then
        mv "$old_root" "$new_root"
        migrated=$((migrated + 1))
        echo "Renamed input root in $(basename "$folder_path")"
    fi

    if [ -f "$params_file" ]; then
        sed -i "s|^input_root_name=.*|input_root_name=${NEW_INPUT_ROOT_NAME}|" "$params_file"
    fi
done

echo "Migrated $migrated folders."
echo "Run ./02_create_jobsa_260504_sweep.sh and ./03_create_jobsb_260504_sweep.sh to regenerate job scripts."
