#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH -A p0023647
#SBATCH -t {TIME}
#SBATCH --mem-per-cpu={MEMORY_VALUE}
#SBATCH -n {PROCESSOR_NUMBER}
#SBATCH -N 1
#SBATCH -e /work/scratch/as12vapa/061-plasticity-fracture-noll-3D/{FOLDER_NAME}/%x.err.%j
#SBATCH -o /work/scratch/as12vapa/061-plasticity-fracture-noll-3D/{FOLDER_NAME}/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01

# --------------------------------------------------
# Working directory settings
# --------------------------------------------------

working_folder_name="{FOLDER_NAME}"
working_directory="$HPC_SCRATCH/061-plasticity-fracture-noll-3D/$working_folder_name"

# --------------------------------------------------
# Parameters
# --------------------------------------------------

# Effective stiffness mesh parameter
NL_EFFECTIVE={NL_EFFECTIVE}

# Fracture mesh parameter
NL_FRACTURE={NL_FRACTURE}

NHOLES={NHOLES}
WSTEG={WSTEG}
DHOLE={DHOLE}

MESH_FILE="{MESH_FILE}"

LAM_MICRO_PARAM={LAM_MICRO_PARAM}
MUE_MICRO_PARAM={MUE_MICRO_PARAM}
GC_MICRO_PARAM={GC_MICRO_PARAM}

EPS_PARAM={EPS_PARAM}
ELEMENT_ORDER={ELEMENT_ORDER}

LCRACK=$(awk "BEGIN {print $WSTEG + $DHOLE}")

cd $HPC_SCRATCH

# --------------------------------------------------
# Effective stiffness mesh generation
# --------------------------------------------------
echo "[$(date)] Starting effective stiffness mesh generation..."
srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work \
    $HOME/dolfinx_alex/alex-dolfinx.sif python3 \
    $working_directory/mesh_effective_stiffness.py \
    --dhole "$DHOLE" \
    --wsteg "$WSTEG" \
    --NL "$NL_EFFECTIVE"
echo "[$(date)] Effective stiffness mesh generation completed."

# --------------------------------------------------
# Effective stiffness simulation
# --------------------------------------------------
echo "[$(date)] Starting effective stiffness simulation..."
srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work \
    $HOME/dolfinx_alex/alex-dolfinx.sif python3 \
    $working_directory/run_effective_stiffness.py \
    --lam_micro_param "$LAM_MICRO_PARAM" \
    --mue_micro_param "$MUE_MICRO_PARAM"
echo "[$(date)] Effective stiffness simulation completed."

# --------------------------------------------------
# Fracture mesh generation (adaptive)
# --------------------------------------------------
echo "[$(date)] Starting fracture mesh generation (adaptive)..."
srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work \
    $HOME/dolfinx_alex/alex-dolfinx.sif python3 \
    $working_directory/mesh_fracture_adaptive.py \
    --Nholes "$NHOLES" \
    --dhole "$DHOLE" \
    --wsteg "$WSTEG" \
    --NL "$NL_FRACTURE"
echo "[$(date)] Fracture mesh generation completed."

# --------------------------------------------------
# Mesh info extraction
# --------------------------------------------------
echo "[$(date)] Extracting mesh info..."
srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work \
    $HOME/dolfinx_alex/alex-dolfinx.sif python3 \
    $working_directory/get_mesh_info.py \
    --mesh_file "$MESH_FILE"
echo "[$(date)] Mesh info extraction completed."

# --------------------------------------------------
# Final full simulation
# --------------------------------------------------
echo "[$(date)] Starting final full simulation..."
srun -n {PROCESSOR_NUMBER} apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work \
    $HOME/dolfinx_alex/alex-dolfinx.sif python3 \
    $working_directory/run_simulation.py \
    --mesh_file "$MESH_FILE" \
    --in_crack_length "$LCRACK" \
    --lam_micro_param "$LAM_MICRO_PARAM" \
    --mue_micro_param "$MUE_MICRO_PARAM" \
    --gc_micro_param "$GC_MICRO_PARAM" \
    --eps_param "$EPS_PARAM" \
    --element









