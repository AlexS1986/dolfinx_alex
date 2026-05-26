#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 RESULT_ROOT [additional 09_evaluation arguments]"
    echo "Run this locally after copying completed cluster results back to this workspace."
    exit 1
fi

RESULT_ROOT=$1
shift
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${SCRIPT_DIR}/plots/260526_beta005/evaluation}"

python3 "${SCRIPT_DIR}/09_evaluation_260504_parameter_space.py" \
    "$RESULT_ROOT" \
    --output-folder "$OUTPUT_FOLDER" \
    --fixed-beta 0.05 \
    --a-values 6 \
    "$@"
