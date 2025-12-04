#!/bin/bash

# Resolve script directory (location of this .sh file)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# -------------------------
# Default parameter handling
# -------------------------
NHOLES=${1:-3}
WSTEG=${2:-0.1}
DHOLE=${3:-0.25}
E0=${4:-0.02}
E1=${5:-0.7}
MESH_FILE=${6:-"mesh_fracture_adaptive.xdmf"}
LAM_MICRO_PARAM=${7:-1.0}
MUE_MICRO_PARAM=${8:-1.0}
GC_MICRO_PARAM=${9:-1.0}

# EPS = 5 * E0 (unless user provides pos. 10)
EPS_PARAM_CALC=$(awk "BEGIN {print 5 * $E0}")
EPS_PARAM=${10:-$EPS_PARAM_CALC}

ELEMENT_ORDER=${11:-1}

# Derived parameter
LCRACK=$(awk "BEGIN {print $WSTEG + $DHOLE}")
echo "Calculated LCRACK: $LCRACK"

# -------------------------
# Create simulation folder
# -------------------------
RESULTS_DIR="$SCRIPT_DIR/results"
SIM_DIR="$RESULTS_DIR/simulation_$WSTEG"

echo "Creating simulation directory: $SIM_DIR"
mkdir -p "$SIM_DIR"

echo "Copying template into simulation folder..."
cp -r "$SCRIPT_DIR/000_template/"* "$SIM_DIR"/

# Move into simulation directory
cd "$SIM_DIR" || exit 1

# -------------------------
# Execute scripts
# -------------------------

echo "Meshing effective stress RVE"
python3 mesh_effective_stiffness.py --dhole "$DHOLE" --wsteg "$WSTEG" --e0 "$E0"
if [ $? -ne 0 ]; then
  echo "Error: Meshing effective stiffness problem failed."
  exit 1
fi

echo "Running effective stiffness computation..."
python3 run_effective_stiffness.py --lam_micro_param "$LAM_MICRO_PARAM" --mue_micro_param "$MUE_MICRO_PARAM"
if [ $? -ne 0 ]; then
  echo "Error: Computing effective stiffness failed."
  exit 1
fi

echo "Meshing fracture RVE with adaptive meshing..."
python3 mesh_fracture_adaptive.py --nholes "$NHOLES" --dhole "$DHOLE" --wsteg "$WSTEG" --e0 "$E0" --e1 "$E1"
if [ $? -ne 0 ]; then
  echo "Error: Meshing fracture problem failed."
  exit 1
fi

echo "Checking mesh data..."
python3 get_mesh_info.py --mesh_file "$MESH_FILE"
if [ $? -ne 0 ]; then
  echo "Error: Checking mesh data failed."
  exit 1
fi

echo "Running fracture simulation via mpirun..."
mpirun -np 8 python3 run_simulation.py \
    --mesh_file "$MESH_FILE" \
    --in_crack_length "$LCRACK" \
    --lam_micro_param "$LAM_MICRO_PARAM" \
    --mue_micro_param "$MUE_MICRO_PARAM" \
    --gc_micro_param "$GC_MICRO_PARAM" \
    --eps_param "$EPS_PARAM" \
    --element_order "$ELEMENT_ORDER"

if [ $? -ne 0 ]; then
  echo "Error: Simulation failed."
  exit 1
fi

echo "All scripts completed successfully."

