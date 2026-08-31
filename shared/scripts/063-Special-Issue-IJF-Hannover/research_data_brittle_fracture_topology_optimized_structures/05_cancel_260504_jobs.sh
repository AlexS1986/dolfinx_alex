#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
working_dir=$(basename "$SCRIPT_DIR")
DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run]"
    echo "  Cancels queued/running SLURM jobs belonging to ${working_dir} 260504 sweep."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "${USER:-}" ]; then
    echo "Error: USER is not defined."
    exit 1
fi

job_ids=$(
    squeue --noheader --user "$USER" --format "%A|%j|%Z" \
        | awk -F'|' -v project="$working_dir" '
            index($2, "063_") || index($2, "dcb260504") || index($3, "/" project "/") || index($3, project) { print $1 }
        ' \
        | sort -u
)

if [ -z "$job_ids" ]; then
    echo "No queued or running 260504 sweep jobs found for ${working_dir}."
    exit 0
fi

echo "Jobs to cancel:"
echo "$job_ids"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run: no jobs cancelled."
    exit 0
fi

scancel $job_ids
echo "Cancelled $(echo "$job_ids" | wc -w) jobs."
