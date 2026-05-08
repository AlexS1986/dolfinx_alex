#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT="${ROOT:-${SCRIPT_DIR}/resources/260504_dcb_beta_phi_a_rho_var_min_max}"
NP="${NP:-4}"
SPLITS="${SPLITS:-spectral volumetric}"
EPSILON="${EPSILON:-0.015}"

mapfile -t LEAF_FOLDERS < <(
    find "${ROOT}" -type f -name mesh.xdmf \
        -exec dirname {} \; | sort
)

if [ "${#LEAF_FOLDERS[@]}" -eq 0 ]; then
    echo "[ERROR] No mesh.xdmf leaf folders found below ${ROOT}"
    exit 1
fi

echo "[INFO] Found ${#LEAF_FOLDERS[@]} mesh leaf folders below ${ROOT}"

LEAF_INDEX=0
for FOLDER in "${LEAF_FOLDERS[@]}"; do
    LEAF_INDEX=$((LEAF_INDEX + 1))
    MAPPING="${FOLDER}/active_cells_mapping"
    if [ ! -f "${MAPPING}" ]; then
        printf "# cell_data_X active_cells_to_be_meshed\n1 1\n" > "${MAPPING}"
    fi

    echo "[INFO] Preparing Dolfinx mesh in ${FOLDER}"
    python3 "${SCRIPT_DIR}/04_mesh2dlfxmesh.py" "${FOLDER}" 1

    for SPLIT in ${SPLITS}; do
        echo "[INFO] Running leaf ${LEAF_INDEX}/${#LEAF_FOLDERS[@]} with split=${SPLIT}, NP=${NP}, epsilon=${EPSILON}"
        mpirun -np "${NP}" \
            python3 "${SCRIPT_DIR}/01_phasefield_dcb_260504_folder.py" "${FOLDER}" "${LEAF_INDEX}" auto "${SPLIT}" --epsilon "${EPSILON}" "$@" \
            </dev/null
    done
done
