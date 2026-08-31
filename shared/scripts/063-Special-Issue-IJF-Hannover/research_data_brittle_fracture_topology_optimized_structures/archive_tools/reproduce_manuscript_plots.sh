#!/bin/bash
set -euo pipefail

ARCHIVE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULT_ROOT="${RESULT_ROOT:-${ARCHIVE_ROOT}/results/new_W_whole_boundary}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARCHIVE_ROOT}/reproduced_plots/new_W_whole_boundary}"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$RESULT_ROOT" ]; then
    echo "Missing primary result root: $RESULT_ROOT" >&2
    exit 1
fi

mkdir -p \
    "$OUTPUT_ROOT/beta001/evaluation" \
    "$OUTPUT_ROOT/beta001/manuscript_overviews" \
    "$OUTPUT_ROOT/beta005/evaluation" \
    "$OUTPUT_ROOT/beta005/manuscript_overviews" \
    "$OUTPUT_ROOT/beta_comparison/manuscript_overviews"

echo "Generating beta_phi=0.01 evaluation plots"
"$PYTHON" "$ARCHIVE_ROOT/09_evaluation_260504_parameter_space.py" \
    "$RESULT_ROOT" \
    --output-folder "$OUTPUT_ROOT/beta001/evaluation" \
    --fixed-beta 0.01 \
    --splits spectral \
    --a-values 6 \
    --omit-a-in-titles

echo "Generating beta_phi=0.01 field overviews"
"$PYTHON" "$ARCHIVE_ROOT/08_plot_phasefield_overview.py" \
    "$RESULT_ROOT" \
    --output "$OUTPUT_ROOT/beta001/manuscript_overviews/phasefield_s_overview.pdf" \
    --fields E gc sigma_c sig_vol sig_dev s \
    --fixed-beta 0.01 \
    --splits spectral \
    --a-values 6

echo "Generating the beta_phi=0.01 volume-constraint plot"
"$PYTHON" "$ARCHIVE_ROOT/10_plot_rho_omega_constraint.py" \
    "$RESULT_ROOT" \
    --output "$OUTPUT_ROOT/beta001/rho_omega_constraint.pdf" \
    --fixed-beta 0.01 \
    --epsilon 0.015 \
    --a-value 6 \
    --split spectral

echo "Generating beta_phi=0.05 evaluation plots with shared reference cases"
"$PYTHON" "$ARCHIVE_ROOT/09_evaluation_260504_parameter_space.py" \
    "$RESULT_ROOT" \
    --output-folder "$OUTPUT_ROOT/beta005/evaluation" \
    --fixed-beta 0.05 \
    --shared-constant-beta 0.01 \
    --splits spectral \
    --a-values 6 \
    --omit-a-in-titles

echo "Generating beta_phi=0.05 field overviews"
"$PYTHON" "$ARCHIVE_ROOT/08_plot_phasefield_overview.py" \
    "$RESULT_ROOT" \
    --output "$OUTPUT_ROOT/beta005/manuscript_overviews/phasefield_s_overview.pdf" \
    --fields E gc sigma_c s \
    --fixed-beta 0.05 \
    --shared-constant-beta 0.01 \
    --splits spectral \
    --a-values 6

echo "Generating beta-comparison field overviews"
"$PYTHON" "$ARCHIVE_ROOT/08_plot_phasefield_overview.py" \
    "$RESULT_ROOT" \
    --output "$OUTPUT_ROOT/beta_comparison/manuscript_overviews/phasefield_s_overview.pdf" \
    --fields E gc sigma_c s \
    --include-beta-variants \
    --cases vary \
    --rows-by-beta \
    --show-beta-title \
    --splits spectral \
    --a-values 6

echo "Reproduced plots are in: $OUTPUT_ROOT"

