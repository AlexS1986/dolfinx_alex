#!/bin/bash
set -euo pipefail

read -r -a SPLIT_VALUES <<< "${SPLITS:-spectral volumetric}"
read -r -a EPSILON_VALUES <<< "${EPSILONS:-0.015 0.03 0.045 0.060}"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKING_DIR=$(basename "$SCRIPT_DIR")
TEMPLATE_FOLDER="${SCRIPT_DIR}/000_template"
RESOURCE_ROOT="${SCRIPT_DIR}/resources/260526_a_6_rho_0_3-0_6_beta_phi_0_05_var_min_max"
DATA_SOURCE_ROOT="${RESOURCE_ROOT}/cluster_input_var"
INPUT_ROOT_NAME="260526_a_6_rho_0_3-0_6_beta_phi_0_05_var"
CAMPAIGN_TAG="260526_beta005_var"

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

prepare_var_input() {
    local rho=$1
    local source="${RESOURCE_ROOT}/rho_${rho}/var"
    local destination="${DATA_SOURCE_ROOT}/beta_phi_0_05_a_6_rho_0_3-0_6_var/beta_0_05_a_6_rho_${rho}_var"

    mkdir -p "$destination"
    for required_file in active_cells_mapping cell_data.csv connectivity.csv mesh.h5 mesh.xdmf node_coords.csv points_data.csv; do
        if [ ! -f "${source}/${required_file}" ]; then
            echo "Error: missing input file: ${source}/${required_file}"
            exit 1
        fi
        cp -p "${source}/${required_file}" "${destination}/${required_file}"
    done
}

for rho in 0_3 0_6; do
    prepare_var_input "$rho"
done

BASE_WORKING_DIR="${HPC_SCRATCH}/${WORKING_DIR}"
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

create_simulation_folder() {
    local split=$1
    local epsilon=$2
    local timestamp
    local eps_label

    timestamp=$(date +%Y%m%d_%H%M%S)
    eps_label=$(epsilon_label "$epsilon")
    local folder_name="simulation_${timestamp}_CAMPAIGN${CAMPAIGN_TAG}_SPLIT${split}_EPS${eps_label}"
    local target_dir="${BASE_WORKING_DIR}/${folder_name}"

    mkdir -p "${target_dir}/resources"
    cp -a "${TEMPLATE_FOLDER}/." "$target_dir/"
    copy_input_data "${target_dir}/resources/${INPUT_ROOT_NAME}"

    cat > "${target_dir}/run_parameters.txt" <<EOF
split=${split}
epsilon=${epsilon}
input_root_name=${INPUT_ROOT_NAME}
data_source_root=${DATA_SOURCE_ROOT}
campaign_tag=${CAMPAIGN_TAG}
EOF

    echo "Created ${target_dir}"
}

for split in "${SPLIT_VALUES[@]}"; do
    for epsilon in "${EPSILON_VALUES[@]}"; do
        create_simulation_folder "$split" "$epsilon"
    done
done
