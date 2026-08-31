#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/000_template"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

DEFAULT_MESH_XDMF="${SCRIPT_DIR}/input_mesh/dlfx_mesh.xdmf"
MESH_XDMF="${1:-${MESH_XDMF:-${DEFAULT_MESH_XDMF}}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIEXEC_BIN="${MPIEXEC_BIN:-mpirun}"
PROCESSOR_NUMBER="${PROCESSOR_NUMBER:-4}"
MATERIAL="${MATERIAL:-custom}"
LAM="${LAM:-}"
MU="${MU:-}"
YOUNGS_MODULUS="${YOUNGS_MODULUS:-}"
POISSON_RATIO="${POISSON_RATIO:-}"
GC="${GC:-1.0}"
EPSILON_FACTOR="${EPSILON_FACTOR:-20.0}"

if [ ! -f "${MESH_XDMF}" ]; then
    echo "Mesh XDMF not found: ${MESH_XDMF}" >&2
    echo "Usage: $0 [/path/to/mesh.xdmf]" >&2
    exit 1
fi

if [ "${MESH_XDMF##*.}" != "xdmf" ]; then
    echo "Mesh input must be an .xdmf file: ${MESH_XDMF}" >&2
    exit 1
fi

if ! [[ "${PROCESSOR_NUMBER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROCESSOR_NUMBER must be a positive integer." >&2
    exit 1
fi

MATERIAL_ARGS=(--material "${MATERIAL}")
if [ -n "${LAM}" ] || [ -n "${MU}" ]; then
    if [ -z "${LAM}" ] || [ -z "${MU}" ]; then
        echo "LAM and MU must be provided together." >&2
        exit 1
    fi
    MATERIAL_ARGS+=(--lam "${LAM}" --mu "${MU}")
fi
if [ -n "${YOUNGS_MODULUS}" ] || [ -n "${POISSON_RATIO}" ]; then
    if [ -z "${YOUNGS_MODULUS}" ] || [ -z "${POISSON_RATIO}" ]; then
        echo "YOUNGS_MODULUS and POISSON_RATIO must be provided together." >&2
        exit 1
    fi
    MATERIAL_ARGS+=(
        --youngs-modulus "${YOUNGS_MODULUS}"
        --poisson-ratio "${POISSON_RATIO}"
    )
fi

MESH_DIR="$(cd "$(dirname "${MESH_XDMF}")" && pwd)"
MESH_FILENAME="$(basename "${MESH_XDMF}")"
MESH_BASE="${MESH_FILENAME%.xdmf}"
MESH_H5="${MESH_DIR}/${MESH_BASE}.h5"

if [ "${MESH_FILENAME}" = "mesh.xdmf" ] && [ -f "${MESH_DIR}/dlfx_mesh.xdmf" ]; then
    echo "Note: dlfx_mesh.xdmf also exists beside this file." >&2
    echo "For the 011 workflow, pass dlfx_mesh.xdmf to use the converted DOLFINx mesh." >&2
fi

if [ ! -f "${MESH_H5}" ]; then
    echo "Matching HDF5 mesh file not found: ${MESH_H5}" >&2
    echo "The launcher expects mesh.xdmf + mesh.h5 with the same basename." >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-simulation_${timestamp}_${MESH_BASE}_linear_np${PROCESSOR_NUMBER}}"
RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"

if [ -e "${RUN_DIR}" ]; then
    echo "Run directory already exists: ${RUN_DIR}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
cp -a "${TEMPLATE_DIR}" "${RUN_DIR}"
cp "${MESH_XDMF}" "${RUN_DIR}/${MESH_FILENAME}"
cp "${MESH_H5}" "${RUN_DIR}/${MESH_BASE}.h5"

SHARED_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${SHARED_DIR}/utils${PYTHONPATH:+:${PYTHONPATH}}"

MPI_ARGS=(-np "${PROCESSOR_NUMBER}")
if "${MPIEXEC_BIN}" --version 2>&1 | grep -q "Open MPI"; then
    MPI_ARGS=(--oversubscribe -np "${PROCESSOR_NUMBER}")
    if [ "$(id -u)" -eq 0 ]; then
        MPI_ARGS=(--allow-run-as-root "${MPI_ARGS[@]}")
    fi
fi

echo "========================================="
echo "068 local CT-scan fracture run"
echo "Container run directory: ${RUN_DIR}"
echo "Mesh: ${MESH_XDMF}"
echo "MPI processes: ${PROCESSOR_NUMBER}"
echo "Finite-element order: 1 (linear)"
echo "Material input: ${MATERIAL}"
echo "lambda=${LAM:-auto}, mu=${MU:-auto}, E=${YOUNGS_MODULUS:-auto}, nu=${POISSON_RATIO:-auto}"
echo "Gc=${GC}, epsilon factor=${EPSILON_FACTOR}"
echo "Started at $(date)"
echo "========================================="

cd "${RUN_DIR}"
"${MPIEXEC_BIN}" "${MPI_ARGS[@]}" "${PYTHON_BIN}" run_simulation.py \
    --mesh-base "${MESH_BASE}" \
    "${MATERIAL_ARGS[@]}" \
    --gc "${GC}" \
    --epsilon-factor "${EPSILON_FACTOR}"

echo "========================================="
echo "Finished at $(date)"
echo "Outputs: ${RUN_DIR}"
echo "========================================="
