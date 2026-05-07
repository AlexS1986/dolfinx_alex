#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
working_dir=$(basename "$SCRIPT_DIR")

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

BASE_DIR="${HPC_SCRATCH}/${working_dir}"

copy_required_inputs() {
    local source_root=$1
    local destination_root=$2

    if [ ! -d "$source_root" ]; then
        echo "Missing source root: $source_root"
        return 1
    fi

    mkdir -p "$destination_root"

    while IFS= read -r source_file; do
        local relative_path="${source_file#$source_root/}"
        local target_file="$destination_root/$relative_path"
        mkdir -p "$(dirname "$target_file")"
        cp -p "$source_file" "$target_file"
    done < <(
        find "$source_root" -type f \
            \( -name 'README.txt' \
            -o -name 'params.txt' \
            -o -name 'active_cells_mapping' \
            -o -name 'cell_data.csv' \
            -o -name 'connectivity.csv' \
            -o -name 'node_coords.csv' \
            -o -name 'points_data.csv' \
            -o -name 'mesh.xdmf' \
            -o -name 'mesh.h5' \)
    )
}

read_parameter() {
    local file=$1
    local key=$2
    sed -n "s|^${key}=||p" "$file" | tail -n 1
}

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: simulation directory does not exist: $BASE_DIR"
    exit 1
fi

repaired=0
failed=0

for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ ! -d "$folder_path" ]; then
        continue
    fi

    params_file="${folder_path}/run_parameters.txt"
    if [ ! -f "$params_file" ]; then
        echo "Skipping $(basename "$folder_path"): missing run_parameters.txt"
        failed=$((failed + 1))
        continue
    fi

    data_source_root=$(read_parameter "$params_file" "data_source_root")
    input_root_name=$(read_parameter "$params_file" "input_root_name")
    destination_root="${folder_path}/resources/${input_root_name}"

    echo "Repairing inputs for $(basename "$folder_path")"
    if copy_required_inputs "$data_source_root" "$destination_root"; then
        repaired=$((repaired + 1))
    else
        failed=$((failed + 1))
    fi
done

echo "Repaired $repaired simulation folders, $failed failures."
