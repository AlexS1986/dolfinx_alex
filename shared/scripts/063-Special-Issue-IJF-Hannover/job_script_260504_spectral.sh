#!/bin/bash
#SBATCH -J dcb260504_sweep
#SBATCH -A p0023647
#SBATCH -t 10080
#SBATCH --mem-per-cpu=4000
#SBATCH -n 6
#SBATCH -N 1
#SBATCH -e /home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover/slurm_logs/%x.err.%j
#SBATCH -o /home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover/slurm_logs/%x.out.%j
#SBATCH --mail-type=END,FAIL
#SBATCH -C i01

set -euo pipefail

# ==========================================
# Host paths
# ==========================================

HOST_WORKDIR="/home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover"
CONTAINER_WORKDIR="/home/scripts/063-Special-Issue-IJF-Hannover"
HOST_ROOT="$HOST_WORKDIR/resources/260504_dcb_beta_phi_a_rho_var_min_max"
CONTAINER_ROOT="$CONTAINER_WORKDIR/resources/260504_dcb_beta_phi_a_rho_var_min_max"
LOGDIR="$HOST_WORKDIR/slurm_logs"

# Container setup
CONTAINER="$HOME/dolfinx_alex/alex-dolfinx.sif"
BINDPATH="$HOME/dolfinx_alex/shared:/home"

NP="${SLURM_NTASKS:-6}"
SPLITS="${SPLITS:-spectral volumetric}"
EPSILONS="${EPSILONS:-0.015 0.03 0.045 0.060}"

mkdir -p "$LOGDIR"

cd "$HOST_WORKDIR" || { echo "Failed to enter working directory"; exit 1; }

echo "========================================="
echo "Job started at $(date)"
echo "Running in $HOST_WORKDIR"
echo "Dataset root: $CONTAINER_ROOT"
echo "Splits: $SPLITS"
echo "Epsilons: $EPSILONS"
echo "MPI tasks: $NP"
echo "========================================="

# ==========================================
# Prepare meshes for every leaf case
# ==========================================

while IFS= read -r HOST_FOLDER; do
    RELATIVE_FOLDER="${HOST_FOLDER#$HOST_WORKDIR/}"
    CONTAINER_FOLDER="$CONTAINER_WORKDIR/$RELATIVE_FOLDER"
    MAPPING="$HOST_FOLDER/active_cells_mapping"

    if [ ! -f "$MAPPING" ]; then
        printf "# cell_data_X active_cells_to_be_meshed\n1 1\n" > "$MAPPING"
    fi

    echo "-----------------------------------------"
    echo "Preparing Dolfinx mesh for: $CONTAINER_FOLDER"
    echo "Started at $(date)"
    echo "-----------------------------------------"

    srun -n 1 apptainer exec \
        --bind "$BINDPATH" \
        "$CONTAINER" \
        python3 "$CONTAINER_WORKDIR/04_mesh2dlfxmesh.py" "$CONTAINER_FOLDER" 1
done < <(
    find "$HOST_ROOT" -mindepth 3 -maxdepth 3 -type f -name cell_data.csv \
        -exec dirname {} \; | sort
)

# ==========================================
# Run all 260504 cases for every split and epsilon
# ==========================================

for SPLIT in $SPLITS; do
    for EPSILON in $EPSILONS; do
        echo "========================================="
        echo "Running 260504 DCB batch with split=$SPLIT, epsilon=$EPSILON"
        echo "Started at $(date)"
        echo "========================================="

        srun -n "$NP" apptainer exec \
            --bind "$BINDPATH" \
            "$CONTAINER" \
            python3 "$CONTAINER_WORKDIR/01_phasefield_dcb_260504_folder.py" "$CONTAINER_ROOT" auto "$SPLIT" --epsilon "$EPSILON"
    done
done

echo "========================================="
echo "Job finished at $(date)"
echo "========================================="

exit 0
