#!/usr/bin/env bash
# Local smoke test of one case (inside the dolfinx container / an environment with alex on PYTHONPATH).
# Example (short run):  WSTEG=1.0 MATERIAL=vonmises STOP_FRACTION=0.3 PROCESSOR_NUMBER=4 ./run_one_local.sh
# Example (cubic):      WSTEG=1.0 EMBEDDING=cubic ./run_one_local.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/000_template"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_local}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIEXEC_BIN="${MPIEXEC_BIN:-mpirun}"
PROCESSOR_NUMBER="${PROCESSOR_NUMBER:-6}"

WSTEG="${WSTEG:-1.0}"
MATERIAL="${MATERIAL:-vonmises}"
GC="${GC:-1.0}"
SIG_Y="${SIG_Y:-1.0}"
HARD="${HARD:-0.2222222}"
EMBEDDING="${EMBEDDING:-isotropic}"
NROWS="${NROWS:-3}"
CRACK_FACTOR="${CRACK_FACTOR:-1.0}"
GEOMETRY="${GEOMETRY:-porous}"
BETA="${BETA:-0.2}"
ETA="${ETA:-0.00001}"
SIG_Y_EMBEDDING="${SIG_Y_EMBEDDING:-1000000.0}"
BC_MODULI="${BC_MODULI:-effective}"
STOP_FRACTION="${STOP_FRACTION:-0.9}"
POSTPROCESSING_INTERVAL="${POSTPROCESSING_INTERVAL:-400}"
NHOLES="${NHOLES:-6}"; DHOLE="${DHOLE:-1.0}"; E0="${E0:-0.02}"; E1="${E1:-0.6}"; EPS_PARAM="${EPS_PARAM:-0.1}"

LCRACK="$(awk "BEGIN {print ${CRACK_FACTOR} * (${WSTEG} + ${DHOLE})}")"
RUN_NAME="${RUN_NAME:-local_$(date +%Y%m%d_%H%M%S)_${MATERIAL}_ws${WSTEG}_${EMBEDDING}}"
RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"
mkdir -p "${RESULTS_DIR}"
cp -a "${TEMPLATE_DIR}" "${RUN_DIR}"
cd "${RUN_DIR}"
echo "run dir: ${RUN_DIR}"

"${PYTHON_BIN}" mesh_effective_stiffness.py --dhole "$DHOLE" --wsteg "$WSTEG" --e0 "$E0" --geometry "$GEOMETRY"
"${PYTHON_BIN}" run_effective_stiffness.py --lam_micro_param 1.0 --mue_micro_param 1.0
"${PYTHON_BIN}" mesh_fracture_adaptive.py --nholes "$NHOLES" --dhole "$DHOLE" --wsteg "$WSTEG" --e0 "$E0" --e1 "$E1" --nrows "$NROWS" --geometry "$GEOMETRY"
"${PYTHON_BIN}" get_mesh_info.py --mesh_file mesh_fracture_adaptive.xdmf
"${MPIEXEC_BIN}" -np "${PROCESSOR_NUMBER}" "${PYTHON_BIN}" run_simulation.py \
    --mesh_file mesh_fracture_adaptive.xdmf --in_crack_length "$LCRACK" \
    --lam_micro_param 1.0 --mue_micro_param 1.0 --gc_micro_param "$GC" --eps_param "$EPS_PARAM" --element_order 1 \
    --material "$MATERIAL" --sig_y "$SIG_Y" --hard "$HARD" --sig_y_embedding "$SIG_Y_EMBEDDING" \
    --beta "$BETA" --eta "$ETA" --embedding "$EMBEDDING" --bc_moduli "$BC_MODULI" \
    --postprocessing_interval "$POSTPROCESSING_INTERVAL" --stop_fraction "$STOP_FRACTION"
echo "done: ${RUN_DIR}"
