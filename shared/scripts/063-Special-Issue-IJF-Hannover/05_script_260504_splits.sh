#!/bin/bash
set -euo pipefail

ROOT="/home/scripts/063-Special-Issue-IJF-Hannover/resources/260504_dcb_beta_phi_a_rho_var_min_max"
NP="${NP:-4}"
SPLITS="${SPLITS:-spectral volumetric}"

while IFS= read -r FOLDER; do
    MAPPING="${FOLDER}/active_cells_mapping"
    if [ ! -f "${MAPPING}" ]; then
        printf "# cell_data_X active_cells_to_be_meshed\n1 1\n" > "${MAPPING}"
    fi

    echo "[INFO] Preparing Dolfinx mesh in ${FOLDER}"
    python3 04_mesh2dlfxmesh.py "${FOLDER}" 1
done < <(
    find "${ROOT}" -mindepth 3 -maxdepth 3 -type f -name cell_data.csv \
        -exec dirname {} \; | sort
)

for SPLIT in ${SPLITS}; do
    echo "[INFO] Running 260504 DCB batch with split=${SPLIT}, NP=${NP}"
    mpirun -np "${NP}" python3 01_phasefield_dcb_260504_folder.py "${ROOT}" auto "${SPLIT}" "$@"
done
