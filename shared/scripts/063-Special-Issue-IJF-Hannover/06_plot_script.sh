#!/bin/bash

PYTHON_SCRIPT="02_evaluation.py"

# === Settings ===
MIN_INDEX=5

# === Input folders ===
INPUT1="/home/scripts/063-Special-Issue-IJF-Hannover/resources/volumetric_2"
INPUT2="/home/scripts/063-Special-Issue-IJF-Hannover/resources/spectral_2"

# === Output folders (plots inside each input folder) ===
OUTPUT1="${INPUT1}/plots"
OUTPUT2="${INPUT2}/plots"

# Create plot directories if they do not exist
mkdir -p "$OUTPUT1"
mkdir -p "$OUTPUT2"

echo "========================================"
echo "Running volumetric dataset..."
echo "Input:  $INPUT1"
echo "Output: $OUTPUT1"
echo "Min index: $MIN_INDEX"
echo "========================================"

python3 "$PYTHON_SCRIPT" \
    --base_folder "$INPUT1" \
    --output_folder "$OUTPUT1" \
    --ext "_volumetric" \
    --min_index "$MIN_INDEX"

echo "========================================"
echo "Running spectral dataset..."
echo "Input:  $INPUT2"
echo "Output: $OUTPUT2"
echo "Min index: $MIN_INDEX"
echo "========================================"

python3 "$PYTHON_SCRIPT" \
    --base_folder "$INPUT2" \
    --output_folder "$OUTPUT2" \
    --ext "_spectral" \
    --min_index "$MIN_INDEX"

echo "========================================"
echo "Done."
echo "Plots written to:"
echo "  $OUTPUT1"
echo "  $OUTPUT2"
echo "========================================"
