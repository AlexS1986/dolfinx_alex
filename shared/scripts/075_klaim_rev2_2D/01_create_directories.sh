#!/bin/bash
# Create one simulation folder per case of cases_2D.list under $HPC_SCRATCH/<project>/ and copy the template.
# Usage: ./01_create_directories.sh [cases_file] [--match TEXT] [--dry-run]
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
TEMPLATE_FOLDER="$SCRIPT_DIR/000_template"
CASES_FILE="$SCRIPT_DIR/cases_2D.list"
MATCH_TEXT=""
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --match) MATCH_TEXT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) echo "Usage: $0 [cases_file] [--match TEXT] [--dry-run]"; exit 0 ;;
        *) CASES_FILE="$1"; shift ;;
    esac
done

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi
if [ ! -f "$CASES_FILE" ]; then
    echo "Error: cases file not found: $CASES_FILE"
    exit 1
fi

BASE_WORKING_DIR="${HPC_SCRATCH}/${PROJECT_NAME}"
mkdir -p "$BASE_WORKING_DIR"
created=0
skipped=0

while read -r name wsteg material gc sig_y hard embedding nrows crack_factor geometry newton_min_iters rest; do
    [ "${newton_min_iters:--}" = "-" ] && newton_min_iters=""
    [ -z "$name" ] && continue
    case "$name" in \#*) continue ;; esac
    if [ -n "$MATCH_TEXT" ] && [[ "$name" != *"$MATCH_TEXT"* ]]; then
        continue
    fi
    target="${BASE_WORKING_DIR}/${name}"
    if [ -d "$target" ]; then
        echo "exists, skipping: $target"
        skipped=$((skipped + 1))
        continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would create: $target"
        continue
    fi
    mkdir -p "$target"
    cp -a "${TEMPLATE_FOLDER}/." "$target/"
    rm -rf "$target/__pycache__"
    cat > "$target/case.env" <<EOF
CASE_NAME=$name
WSTEG=$wsteg
MATERIAL=$material
GC_MICRO_PARAM=$gc
SIG_Y=$sig_y
HARD=$hard
EMBEDDING=$embedding
NROWS=$nrows
CRACK_FACTOR=$crack_factor
GEOMETRY=$geometry
NEWTON_MIN_ITERS=$newton_min_iters
EOF
    echo "created: $target"
    created=$((created + 1))
done < "$CASES_FILE"

echo "Created $created folders, skipped $skipped existing folders in $BASE_WORKING_DIR"
