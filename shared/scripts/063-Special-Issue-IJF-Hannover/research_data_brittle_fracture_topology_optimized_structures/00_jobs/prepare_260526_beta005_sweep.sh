#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
RESOURCE_ROOT="${PROJECT_DIR}/resources/260526_a_6_rho_0_3-0_6_beta_phi_0_05_var_min_max"
CLUSTER_INPUT_ROOT="${RESOURCE_ROOT}/cluster_input_var"
INPUT_ROOT_NAME="260526_a_6_rho_0_3-0_6_beta_phi_0_05_var"
CAMPAIGN_TAG="260526_beta005_var"

prepare_leaf() {
    local rho=$1
    local source="${RESOURCE_ROOT}/rho_${rho}/var"
    local group="${CLUSTER_INPUT_ROOT}/beta_phi_0_05_a_6_rho_0_3-0_6_var"
    local destination="${group}/beta_0_05_a_6_rho_${rho}_var"

    if [ ! -d "$source" ]; then
        echo "Error: missing source mesh folder: $source"
        exit 1
    fi

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
    prepare_leaf "$rho"
done

echo "Prepared cluster input leaves below ${CLUSTER_INPUT_ROOT}"
echo "Only the two varying-material rho cases are included; raw rho_*/var inputs remain unchanged."

if [ "${1:-}" = "--prepare-only" ]; then
    exit 0
fi

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined. Use --prepare-only off-cluster."
    exit 1
fi

SPLITS="${SPLITS:-spectral volumetric}" \
EPSILONS="${EPSILONS:-0.015 0.03 0.045 0.060}" \
    "${PROJECT_DIR}/01_create_directories_for_260526_beta005_var_sweep.sh"

"${PROJECT_DIR}/02_create_jobsa_260526_beta005_var_sweep.sh"
MEMORY_VALUE="${MEMORY_VALUE:-4000}" \
PROCESSOR_NUMBER="${PROCESSOR_NUMBER:-6}" \
TIME="${TIME:-10080}" \
    "${PROJECT_DIR}/03_create_jobsb_260526_beta005_var_sweep.sh"

echo "Created the ${CAMPAIGN_TAG} cluster jobs."
echo "Review submission with:"
echo "${PROJECT_DIR}/04_submit_all_260526_beta005_var_jobs.sh --dry-run"
echo "Submit with:"
echo "${PROJECT_DIR}/04_submit_all_260526_beta005_var_jobs.sh"
