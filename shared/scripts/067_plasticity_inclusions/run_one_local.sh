#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/000_template"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIEXEC_BIN="${MPIEXEC_BIN:-mpirun}"
PROCESSOR_NUMBER="${PROCESSOR_NUMBER:-6}"

WSTEG="${WSTEG:-0.25}"
STIFFNESS_SCALE="${STIFFNESS_SCALE:-0.5}"
GC_SCALE="${GC_SCALE:-1.5}"

NHOLES="${NHOLES:-6}"
DHOLE="${DHOLE:-1.0}"
E0="${E0:-0.02}"
E1="${E1:-0.6}"
MESH_FILE="${MESH_FILE:-mesh_fracture_adaptive.xdmf}"

LAM_MATRIX_PARAM="${LAM_MATRIX_PARAM:-1.0}"
MUE_MATRIX_PARAM="${MUE_MATRIX_PARAM:-1.0}"
GC_MATRIX_PARAM="${GC_MATRIX_PARAM:-1.0}"
SIG_Y_MATRIX_PARAM="${SIG_Y_MATRIX_PARAM:-1.0}"
HARD_MATRIX_PARAM="${HARD_MATRIX_PARAM:-0.2222222}"
SIG_Y_INCLUSION_SCALE="${SIG_Y_INCLUSION_SCALE:-1.0}"
EPS_PARAM="${EPS_PARAM:-0.1}"
ELEMENT_ORDER="${ELEMENT_ORDER:-1}"
POSTPROCESSING_INTERVAL="${POSTPROCESSING_INTERVAL:-10}"
WRITE_MATERIAL_FIELDS_FIRST_STEP="${WRITE_MATERIAL_FIELDS_FIRST_STEP:-1}"

multiply() {
    awk "BEGIN {print $1 * $2}"
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing required file: $1"
        exit 1
    fi
}

require_file "${TEMPLATE_DIR}/mesh_effective_stiffness.py"
require_file "${TEMPLATE_DIR}/run_effective_stiffness.py"
require_file "${TEMPLATE_DIR}/mesh_fracture_adaptive.py"
require_file "${TEMPLATE_DIR}/get_mesh_info.py"
require_file "${TEMPLATE_DIR}/run_simulation.py"

LAM_INCLUSION_PARAM="$(multiply "${LAM_MATRIX_PARAM}" "${STIFFNESS_SCALE}")"
MUE_INCLUSION_PARAM="$(multiply "${MUE_MATRIX_PARAM}" "${STIFFNESS_SCALE}")"
GC_INCLUSION_PARAM="$(multiply "${GC_MATRIX_PARAM}" "${GC_SCALE}")"
SIG_Y_INCLUSION_PARAM="$(multiply "${SIG_Y_MATRIX_PARAM}" "${SIG_Y_INCLUSION_SCALE}")"
HARD_INCLUSION_PARAM="${HARD_MATRIX_PARAM}"
LCRACK="$(awk "BEGIN {print ${WSTEG} + ${DHOLE}}")"

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-simulation_${timestamp}_WSTEG${WSTEG}_KINC${STIFFNESS_SCALE}_GCINC${GC_SCALE}_local}"
RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"

if [ -e "${RUN_DIR}" ]; then
    echo "Run directory already exists: ${RUN_DIR}"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
cp -a "${TEMPLATE_DIR}" "${RUN_DIR}"

echo "========================================="
echo "067 local plasticity-inclusion run"
echo "Run directory: ${RUN_DIR}"
echo "MPI ranks for fracture simulation: ${PROCESSOR_NUMBER}"
echo "WSTEG=${WSTEG}, KINC=${STIFFNESS_SCALE}, GCINC=${GC_SCALE}"
echo "lam/mue/gc/sig_y matrix=${LAM_MATRIX_PARAM}/${MUE_MATRIX_PARAM}/${GC_MATRIX_PARAM}/${SIG_Y_MATRIX_PARAM}"
echo "lam/mue/gc/sig_y inclusion=${LAM_INCLUSION_PARAM}/${MUE_INCLUSION_PARAM}/${GC_INCLUSION_PARAM}/${SIG_Y_INCLUSION_PARAM}"
echo "NHOLES=${NHOLES}, DHOLE=${DHOLE}, E0=${E0}, E1=${E1}"
echo "Started at $(date)"
echo "========================================="

cd "${RUN_DIR}"

echo "[1/5] Creating effective-stiffness mesh"
"${PYTHON_BIN}" mesh_effective_stiffness.py \
    --dhole "${DHOLE}" \
    --wsteg "${WSTEG}" \
    --e0 "${E0}"

echo "[2/5] Running effective-stiffness solve"
"${PYTHON_BIN}" run_effective_stiffness.py \
    --lam_matrix_param "${LAM_MATRIX_PARAM}" \
    --mue_matrix_param "${MUE_MATRIX_PARAM}" \
    --lam_inclusion_param "${LAM_INCLUSION_PARAM}" \
    --mue_inclusion_param "${MUE_INCLUSION_PARAM}"

echo "[3/5] Creating adaptive fracture mesh"
"${PYTHON_BIN}" mesh_fracture_adaptive.py \
    --nholes "${NHOLES}" \
    --dhole "${DHOLE}" \
    --wsteg "${WSTEG}" \
    --e0 "${E0}" \
    --e1 "${E1}"

echo "[4/5] Reporting fracture mesh info"
"${PYTHON_BIN}" get_mesh_info.py \
    --mesh_file "${MESH_FILE}"

echo "[5/5] Running fracture simulation on ${PROCESSOR_NUMBER} MPI ranks"
"${MPIEXEC_BIN}" -np "${PROCESSOR_NUMBER}" "${PYTHON_BIN}" run_simulation.py \
    --mesh_file "${MESH_FILE}" \
    --in_crack_length "${LCRACK}" \
    --lam_matrix_param "${LAM_MATRIX_PARAM}" \
    --mue_matrix_param "${MUE_MATRIX_PARAM}" \
    --gc_matrix_param "${GC_MATRIX_PARAM}" \
    --sig_y_matrix_param "${SIG_Y_MATRIX_PARAM}" \
    --hard_matrix_param "${HARD_MATRIX_PARAM}" \
    --lam_inclusion_param "${LAM_INCLUSION_PARAM}" \
    --mue_inclusion_param "${MUE_INCLUSION_PARAM}" \
    --gc_inclusion_param "${GC_INCLUSION_PARAM}" \
    --sig_y_inclusion_param "${SIG_Y_INCLUSION_PARAM}" \
    --hard_inclusion_param "${HARD_INCLUSION_PARAM}" \
    --eps_param "${EPS_PARAM}" \
    --element_order "${ELEMENT_ORDER}" \
    --postprocessing_interval "${POSTPROCESSING_INTERVAL}" \
    $([ "${WRITE_MATERIAL_FIELDS_FIRST_STEP}" = "1" ] && printf '%s' "--write_material_fields_first_step")

echo "========================================="
echo "Finished at $(date)"
echo "Outputs are in: ${RUN_DIR}"
echo "========================================="
