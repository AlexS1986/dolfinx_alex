#!/bin/bash
# Mobility side study (Reviewer 1): one simulation folder per (split, epsilon, mobility, leaf).
# Only the listed leaf folders are copied, because the job template runs every
# mesh.xdmf leaf found below the input root.
#
#   SPLITS      default "spectral"
#   EPSILONS    default "0.03"             (beta_s in mm, as in the publication)
#   MOBILITIES  default "1000 10000"       (reference M = 100 exists in results/)
#   LEAVES      default: rho = 0.3, E_var (primary case); relative to INPUT_ROOT_NAME
#
# Example including the optional cases:
#   LEAVES="beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_3_var
#           beta_phi_0_001-0_01_a_3-6_rho_0_3_min/beta_0_01_a_6_rho_0_3_min
#           beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_6_var" \
#   bash 01_create_directories_for_260929_mobility_study.sh
set -euo pipefail

read -r -a SPLIT_VALUES <<< "${SPLITS:-spectral}"
read -r -a EPSILON_VALUES <<< "${EPSILONS:-0.03}"
read -r -a MOBILITY_VALUES <<< "${MOBILITIES:-1000 10000}"
read -r -a LEAF_VALUES <<< "${LEAVES:-beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_3_var}"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKING_DIR=$(basename "$SCRIPT_DIR")
TEMPLATE_FOLDER="${SCRIPT_DIR}/000_template"
INPUT_ROOT_NAME="260504_dcb_beta_phi_a_rho_var_min_max"
DATA_SOURCE_ROOT="${SCRIPT_DIR}/resources/${INPUT_ROOT_NAME}"
CAMPAIGN_TAG="260929_mobility"

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

BASE_WORKING_DIR="${HPC_SCRATCH}/${WORKING_DIR}"
mkdir -p "$BASE_WORKING_DIR"

float_label() {
    sed 's/\./_/g; s/-/m/g' <<< "$1"
}

copy_leaf_input() {
    local leaf=$1
    local destination=$2
    local source="${DATA_SOURCE_ROOT}/${leaf}"

    mkdir -p "$destination"
    for required_file in active_cells_mapping cell_data.csv connectivity.csv mesh.h5 mesh.xdmf node_coords.csv points_data.csv; do
        if [ ! -f "${source}/${required_file}" ]; then
            echo "Error: missing input file: ${source}/${required_file}"
            exit 1
        fi
        cp -p "${source}/${required_file}" "${destination}/${required_file}"
    done
}

create_simulation_folder() {
    local split=$1
    local epsilon=$2
    local mobility=$3
    local leaf=$4
    local timestamp
    local leaf_name

    timestamp=$(date +%Y%m%d_%H%M%S)
    leaf_name=$(basename "$leaf")
    local folder_name="simulation_${timestamp}_CAMPAIGN${CAMPAIGN_TAG}_SPLIT${split}_EPS$(float_label "$epsilon")_M$(float_label "$mobility")_${leaf_name}"
    local target_dir="${BASE_WORKING_DIR}/${folder_name}"

    mkdir -p "${target_dir}/resources"
    cp -a "${TEMPLATE_FOLDER}/." "$target_dir/"
    rm -rf "${target_dir}/__pycache__"
    copy_leaf_input "$leaf" "${target_dir}/resources/${INPUT_ROOT_NAME}/${leaf}"

    cat > "${target_dir}/run_parameters.txt" <<EOF
split=${split}
epsilon=${epsilon}
mobility=${mobility}
leaf=${leaf}
input_root_name=${INPUT_ROOT_NAME}
data_source_root=${DATA_SOURCE_ROOT}
campaign_tag=${CAMPAIGN_TAG}
EOF

    echo "Created ${target_dir}"
}

for split in "${SPLIT_VALUES[@]}"; do
    for epsilon in "${EPSILON_VALUES[@]}"; do
        for mobility in "${MOBILITY_VALUES[@]}"; do
            for leaf in "${LEAF_VALUES[@]}"; do
                create_simulation_folder "$split" "$epsilon" "$mobility" "$leaf"
            done
        done
    done
done
