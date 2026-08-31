#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOLFINX_ALEX_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MESH_XDMF="${1:-}"

if [ -z "${MESH_XDMF}" ]; then
    echo "Usage: $0 /absolute/host/path/to/mesh.xdmf" >&2
    exit 2
fi

if [ ! -f "${MESH_XDMF}" ]; then
    echo "Mesh XDMF not found: ${MESH_XDMF}" >&2
    exit 1
fi

MESH_DIR="$(cd "$(dirname "${MESH_XDMF}")" && pwd)"
MESH_FILENAME="$(basename "${MESH_XDMF}")"
MESH_BASE="${MESH_FILENAME%.xdmf}"

if [ "${MESH_FILENAME}" = "mesh.xdmf" ] && [ -f "${MESH_DIR}/dlfx_mesh.xdmf" ]; then
    echo "Note: dlfx_mesh.xdmf also exists beside this file." >&2
    echo "For the 011 workflow, pass dlfx_mesh.xdmf to use the converted DOLFINx mesh." >&2
fi

if [ ! -f "${MESH_DIR}/${MESH_BASE}.h5" ]; then
    echo "Matching HDF5 mesh file not found: ${MESH_DIR}/${MESH_BASE}.h5" >&2
    exit 1
fi

PROCESSOR_NUMBER="${PROCESSOR_NUMBER:-4}"
MATERIAL="${MATERIAL:-custom}"
LAM="${LAM:-}"
MU="${MU:-}"
YOUNGS_MODULUS="${YOUNGS_MODULUS:-}"
POISSON_RATIO="${POISSON_RATIO:-}"
GC="${GC:-1.0}"
EPSILON_FACTOR="${EPSILON_FACTOR:-20.0}"

docker compose -f "${DOLFINX_ALEX_ROOT}/docker-compose.yml" run --rm --no-deps \
    -v "${MESH_DIR}:/input:ro" \
    -e PROCESSOR_NUMBER="${PROCESSOR_NUMBER}" \
    -e MATERIAL="${MATERIAL}" \
    -e LAM="${LAM}" \
    -e MU="${MU}" \
    -e YOUNGS_MODULUS="${YOUNGS_MODULUS}" \
    -e POISSON_RATIO="${POISSON_RATIO}" \
    -e GC="${GC}" \
    -e EPSILON_FACTOR="${EPSILON_FACTOR}" \
    alex-dolfinx \
    /home/scripts/068-ct-scan-fracture-local/run_local.sh "/input/${MESH_FILENAME}"
