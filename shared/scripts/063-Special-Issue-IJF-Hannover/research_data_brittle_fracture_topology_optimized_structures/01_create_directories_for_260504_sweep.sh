#!/bin/bash
set -euo pipefail

SPLIT_VALUES=(spectral volumetric)
EPSILON_VALUES=(0.015 0.03 0.045 0.060)
TEMPLATE_FOLDER="000_template"
INPUT_ROOT_NAME="${INPUT_ROOT_NAME:-260504_dcb_beta_phi_a_rho_var_min_max}"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
working_dir=$(basename "$SCRIPT_DIR")
DATA_SOURCE_ROOT="${DATA_SOURCE_ROOT:-${SCRIPT_DIR}/resources/260504_dcb_beta_phi_a_rho_var_min_max}"

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

if [ ! -d "$DATA_SOURCE_ROOT" ]; then
    echo "Error: DATA_SOURCE_ROOT does not exist: $DATA_SOURCE_ROOT"
    exit 1
fi

BASE_WORKING_DIR="${HPC_SCRATCH}/${working_dir}"
mkdir -p "$BASE_WORKING_DIR"

epsilon_label() {
    sed 's/\./_/g; s/-/m/g' <<< "$1"
}

copy_input_data() {
    local destination=$1
    mkdir -p "$destination"

    while IFS= read -r source_file; do
        local relative_path="${source_file#$DATA_SOURCE_ROOT/}"
        local target_file="$destination/$relative_path"
        mkdir -p "$(dirname "$target_file")"
        cp -p "$source_file" "$target_file"
    done < <(
        find "$DATA_SOURCE_ROOT" -type f \
            \( -name 'active_cells_mapping' \
            -o -name '*.csv' \
            -o -name 'mesh.xdmf' \
            -o -name 'mesh.h5' \)
    )
}

replicate_folder() {
    local split=$1
    local epsilon=$2
    local eps_label
    eps_label=$(epsilon_label "$epsilon")

    local current_time
    current_time=$(date +%Y%m%d_%H%M%S)
    local folder_name="simulation_${current_time}_SPLIT${split}_EPS${eps_label}"
    local target_dir="${BASE_WORKING_DIR}/${folder_name}"

    mkdir -p "$target_dir/resources"
    cp -a "${SCRIPT_DIR}/${TEMPLATE_FOLDER}/." "$target_dir/"
    copy_input_data "$target_dir/resources/$INPUT_ROOT_NAME"

    cat > "$target_dir/run_parameters.txt" <<EOF
split=${split}
epsilon=${epsilon}
input_root_name=${INPUT_ROOT_NAME}
data_source_root=${DATA_SOURCE_ROOT}
EOF

    echo "Created $target_dir"
}

for split in "${SPLIT_VALUES[@]}"; do
    for epsilon in "${EPSILON_VALUES[@]}"; do
        replicate_folder "$split" "$epsilon"
    done
done
