#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH -A p0023647
#SBATCH -t {TIME}
#SBATCH --mem-per-cpu={MEMORY_VALUE}
#SBATCH -n {PROCESSOR_NUMBER}
#SBATCH -N 1
#SBATCH -e /work/scratch/as12vapa/063-Special-Issue-IJF-Hannover/{FOLDER_NAME}/%x.err.%j
#SBATCH -o /work/scratch/as12vapa/063-Special-Issue-IJF-Hannover/{FOLDER_NAME}/%x.out.%j
#SBATCH --mail-type=END,FAIL
#SBATCH -C i01

set -euo pipefail

working_folder_name="{FOLDER_NAME}"
working_directory="$HPC_SCRATCH/063-Special-Issue-IJF-Hannover/$working_folder_name"
container="$HOME/dolfinx_alex/alex-dolfinx.sif"
bindpath="$HOME/dolfinx_alex/shared:/home,$working_directory:/work"
input_root_host="$working_directory/resources/{INPUT_ROOT_NAME}"
input_root_container="/work/resources/{INPUT_ROOT_NAME}"
split="{SPLIT}"
epsilon="{EPSILON}"
processor_number="{PROCESSOR_NUMBER}"

echo "========================================="
echo "Job started at $(date)"
echo "Working directory: $working_directory"
echo "Input root: $input_root_host"
echo "Split: $split"
echo "Epsilon: $epsilon"
echo "MPI tasks: $processor_number"
echo "========================================="

if [ ! -d "$working_directory" ]; then
    echo "Missing working directory: $working_directory"
    exit 1
fi

if [ ! -d "$input_root_host" ]; then
    echo "Missing input root: $input_root_host"
    exit 1
fi

cd "$working_directory"

while IFS= read -r host_folder; do
    relative_folder="${host_folder#$working_directory/}"
    container_folder="/work/$relative_folder"
    mapping="$host_folder/active_cells_mapping"

    if [ ! -f "$mapping" ]; then
        printf "# cell_data_X active_cells_to_be_meshed\n1 1\n" > "$mapping"
    fi

    echo "Preparing Dolfinx mesh for: $container_folder"
    srun -n 1 apptainer exec \
        --bind "$bindpath" \
        "$container" \
        python3 /work/04_mesh2dlfxmesh.py "$container_folder" 1
done < <(
    find "$input_root_host" -mindepth 3 -maxdepth 3 -type f -name cell_data.csv \
        -exec dirname {} \; | sort
)

echo "========================================="
echo "Running phase-field simulation"
echo "Started at $(date)"
echo "========================================="

srun -n "$processor_number" apptainer exec \
    --bind "$bindpath" \
    "$container" \
    python3 /work/01_phasefield_dcb_260504_folder.py "$input_root_container" auto "$split" --epsilon "$epsilon"

exitcode=$?
echo "Job finished at $(date) with exit code $exitcode"
exit $exitcode
