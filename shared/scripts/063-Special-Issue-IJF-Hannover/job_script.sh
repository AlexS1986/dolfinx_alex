#!/bin/bash
#SBATCH -J dcb_phasefield_job
#SBATCH -A p0023647
#SBATCH -t 10080
#SBATCH --mem-per-cpu=4000
#SBATCH -n 6
#SBATCH -N 1
#SBATCH -e /home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover/slurm_logs/%x.err.%j
#SBATCH -o /home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover/slurm_logs/%x.out.%j
#SBATCH --mail-type=END
#SBATCH -C i01

# ==========================================
# Host paths
# ==========================================

HOST_WORKDIR="/home/as12vapa/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover"
LOGDIR="$HOST_WORKDIR/slurm_logs"

# Container setup
CONTAINER="$HOME/dolfinx_alex/alex-dolfinx.sif"
BINDPATH="$HOME/dolfinx_alex/shared:/home"

mkdir -p "$LOGDIR"

cd "$HOST_WORKDIR" || { echo "Failed to enter working directory"; exit 1; }

echo "========================================="
echo "Job started at $(date)"
echo "Running in $HOST_WORKDIR"
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

    # Mesh conversion (serial)
    srun -n 1 apptainer exec \
        --bind $BINDPATH \
        $CONTAINER \
        python3 /home/scripts/063-Special-Issue-IJF-Hannover/04_mesh2dlfxmesh.py "$FOLDER"

    if [ $? -ne 0 ]; then
        echo "Mesh conversion failed!"
        exit 1
    fi

    # Phasefield simulation (MPI parallel)
    srun -n 6 apptainer exec \
        --bind $BINDPATH \
        $CONTAINER \
        python3 /home/scripts/063-Special-Issue-IJF-Hannover/01_phasefield_dcb_whole_folder.py "$FOLDER" "$MODE" 1 20

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
FOLDER="/home/scripts/063-Special-Issue-IJF-Hannover/resources/dcb_var_bcpos_E_var/export"
run_case "$FOLDER" "vary"
run_case "$FOLDER" "fromfile"

# --- E_max ---
FOLDER="/home/scripts/063-Special-Issue-IJF-Hannover/resources/dcb_var_bcpos_E_max/export"
run_case "$FOLDER" "max"

# --- E_min ---
FOLDER="/home/scripts/063-Special-Issue-IJF-Hannover/resources/dcb_var_bcpos_E_min/export"
run_case "$FOLDER" "min"

echo "========================================="
echo "Job finished at $(date)"
echo "========================================="

exit 0