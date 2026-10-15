#!/bin/bash
# Cancel queued/running SLURM jobs of this project. Usage: ./04_cancel_jobs.sh [--dry-run] [--match TEXT]
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
DRY_RUN=0
MATCH_TEXT=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --match) MATCH_TEXT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
job_rows=$(squeue --noheader --user "$USER" --format "%A|%j|%T|%Z" \
    | awk -F'|' -v project="$PROJECT_NAME" -v match_text="$MATCH_TEXT" '
        { pm = index($4, "/" project "/") || index($2, "r2_"); em = (match_text == "") || index($2, match_text) || index($4, match_text);
          if (pm && em) print $0 }' | sort -u)
if [ -z "$job_rows" ]; then
    echo "No queued or running jobs found for $PROJECT_NAME."
    exit 0
fi
printf "%s\n" "$job_rows" | awk -F'|' '{printf "  %s  %-28s %-10s %s\n", $1, $2, $3, $4}'
job_ids=$(printf "%s\n" "$job_rows" | awk -F'|' '{print $1}')
if [ "$DRY_RUN" -eq 1 ]; then echo "Dry run: nothing cancelled."; exit 0; fi
scancel $job_ids
echo "Cancelled $(echo "$job_ids" | wc -w) jobs."
