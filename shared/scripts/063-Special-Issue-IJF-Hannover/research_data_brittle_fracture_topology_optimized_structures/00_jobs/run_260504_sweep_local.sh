#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"

input_root="${1:-$project_dir/resources/260504_dcb_beta_phi_a_rho_var_min_max}"
split="${2:-spectral}"
epsilon="${3:-0.015}"

processor_number="${PROCESSOR_NUMBER:-1}"
python_bin="${PYTHON_BIN:-python3}"
mpiexec_bin="${MPIEXEC_BIN:-mpirun}"
run_mesh_conversion="${RUN_MESH_CONVERSION:-1}"

phasefield_script="${PHASEFIELD_SCRIPT:-$project_dir/000_template/01_phasefield_dcb_260504_folder.py}"
mesh_script="${MESH_SCRIPT:-$project_dir/000_template/04_mesh2dlfxmesh.py}"

echo "========================================="
echo "Local sweep started at $(date)"
echo "Input root: $input_root"
echo "Split: $split"
echo "Epsilon: $epsilon"
echo "MPI tasks per dataset: $processor_number"
echo "========================================="

if [ ! -d "$input_root" ]; then
    echo "Missing input root: $input_root"
    exit 1
fi

if [ ! -f "$phasefield_script" ]; then
    echo "Missing phase-field script: $phasefield_script"
    exit 1
fi

if [ "$run_mesh_conversion" != "0" ] && [ ! -f "$mesh_script" ]; then
    echo "Missing mesh conversion script: $mesh_script"
    exit 1
fi

mapfile -t mesh_folders < <(
    find "$input_root" -type f -name mesh.xdmf \
        -exec dirname {} \; | sort
)

if [ "${#mesh_folders[@]}" -eq 0 ]; then
    echo "No mesh.xdmf leaf folders found below $input_root"
    exit 1
fi

echo "Found ${#mesh_folders[@]} mesh leaf folders below $input_root"

mesh_folder_count=0
for host_folder in "${mesh_folders[@]}"; do
    mesh_folder_count=$((mesh_folder_count + 1))
    mapping="$host_folder/active_cells_mapping"

    if [ ! -f "$mapping" ]; then
        printf "# cell_data_X active_cells_to_be_meshed\n1 1\n" > "$mapping"
    fi

    for required_file in cell_data.csv connectivity.csv node_coords.csv points_data.csv mesh.xdmf mesh.h5; do
        if [ ! -f "$host_folder/$required_file" ]; then
            echo "Missing required mesh input: $host_folder/$required_file"
            exit 1
        fi
    done

    if [ "$run_mesh_conversion" != "0" ]; then
        echo "Preparing Dolfinx mesh for: $host_folder"
        "$python_bin" "$mesh_script" "$host_folder" 1
    fi

    if [ ! -f "$host_folder/dlfx_mesh_1.xdmf" ]; then
        echo "Missing Dolfinx mesh: $host_folder/dlfx_mesh_1.xdmf"
        echo "Set RUN_MESH_CONVERSION=1 or run the mesh conversion first."
        exit 1
    fi

    echo "========================================="
    echo "Running phase-field simulation for: $host_folder"
    echo "Started at $(date)"
    echo "========================================="

    if [ "$processor_number" -eq 1 ]; then
        "$python_bin" "$phasefield_script" "$host_folder" "$mesh_folder_count" auto "$split" --epsilon "$epsilon" </dev/null
    else
        "$mpiexec_bin" -n "$processor_number" \
            "$python_bin" "$phasefield_script" "$host_folder" "$mesh_folder_count" auto "$split" --epsilon "$epsilon" </dev/null
    fi
done

echo "Local sweep finished at $(date)"
