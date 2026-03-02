#!/bin/bash
#SBATCH -J dcb_phasefield_job
#SBATCH -A p0023647
#SBATCH -t 1440
#SBATCH --mem-per-cpu=6000
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -e /home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover/slurm_logs/%x.err.%j
#SBATCH -o /home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover/slurm_logs/%x.out.%j
#SBATCH --mail-type=END
#SBATCH -C i01

# ==========================================
# Set working directory
# ==========================================

WORKDIR="/home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover"
LOGDIR="$WORKDIR/slurm_logs"

mkdir -p "$LOGDIR"

cd "$WORKDIR" || { echo "Failed to enter working directory"; exit 1; }

echo "========================================="
echo "Job started at $(date)"
echo "Running in $(pwd)"
echo "========================================="

# ==========================================
# Function to run one dataset
# ==========================================

run_case () {
    local FOLDER=$1
    local MODE=$2

    echo "-----------------------------------------"
    echo "Processing folder: $FOLDER"
    echo "Mode: $MODE"
    echo "Started at $(date)"
    echo "-----------------------------------------"

    # Mesh conversion
    srun -n 1 python3 04_mesh2dlfxmesh.py "$FOLDER"
    if [ $? -ne 0 ]; then
        echo "Mesh conversion failed!"
        exit 1
    fi

    # Phasefield
    srun -n 4 python3 01_phasefield_dcb_whole_folder.py "$FOLDER" "$MODE" 1 20
    if [ $? -ne 0 ]; then
        echo "Phasefield simulation failed!"
        exit 1
    fi

    echo "Finished mode $MODE at $(date)"
}

# ==========================================
# Run cases
# ==========================================

# --- E_var ---
FOLDER="$WORKDIR/resources/dcb_var_bcpos_E_var/export"
run_case "$FOLDER" "vary"
run_case "$FOLDER" "fromfile"

# --- E_max ---
FOLDER="$WORKDIR/resources/dcb_var_bcpos_E_max/export"
run_case "$FOLDER" "max"

# --- E_min ---
FOLDER="$WORKDIR/resources/dcb_var_bcpos_E_min/export"
run_case "$FOLDER" "min"

echo "========================================="
echo "Job finished at $(date)"
echo "========================================="

exit 0