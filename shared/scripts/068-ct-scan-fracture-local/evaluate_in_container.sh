#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOLFINX_ALEX_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

docker compose -f "${DOLFINX_ALEX_ROOT}/docker-compose.yml" run --rm --no-deps \
    alex-dolfinx \
    python3 /home/scripts/068-ct-scan-fracture-local/evaluation.py "$@"
