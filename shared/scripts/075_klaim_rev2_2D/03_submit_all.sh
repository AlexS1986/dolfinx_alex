#!/bin/bash
# Submit job_script.sh of every case folder. Usage: ./03_submit_all.sh [--dry-run] [--max N] [--match TEXT]
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi
BASE_DIR="${HPC_SCRATCH}/${PROJECT_NAME}"
ACCOUNT="${ACCOUNT:-p0023647}"     # passed as sbatch -A, overrides the #SBATCH -A line of the job script (no implicit account, 2026-10-27)
if [ "$ACCOUNT" = "l0003507" ]; then echo "ERROR: account l0003507 is never used (rule 2026-10-27, 046_plasticity/CLAUDE.md, Konto). Use p0023647." >&2; exit 1; fi
DRY_RUN=0
MAX_SUBMISSIONS=""
MATCH_TEXT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --max) MAX_SUBMISSIONS="$2"; shift 2 ;;
        --match) MATCH_TEXT="$2"; shift 2 ;;
        -h|--help) echo "Usage: $0 [--dry-run] [--max N] [--match TEXT]"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

submitted=0
skipped_limit=0
queued_names=$(squeue --noheader --user "$USER" --format "%120j" 2>/dev/null | sed "s/ *$//" || true)
for folder_path in "${BASE_DIR}"/*/; do
    [ -d "$folder_path" ] || continue
    folder_name=$(basename "$folder_path")
    if [ -n "$MATCH_TEXT" ] && [[ "$folder_name" != *"$MATCH_TEXT"* ]]; then
        continue
    fi
    job_script="${folder_path}job_script.sh"
    if [ ! -f "$job_script" ]; then
        echo "missing job script: $job_script"
        continue
    fi
    if [ -n "$MAX_SUBMISSIONS" ] && [ "$submitted" -ge "$MAX_SUBMISSIONS" ]; then
        break
    fi
    if ls "${folder_path}"/*.out.* >/dev/null 2>&1; then
        echo "already has SLURM output, skipping (delete *.out.* to resubmit): $folder_name"
        continue
    fi
    if echo "$queued_names" | grep -qx "r2_${folder_name}"; then
        echo "already queued (pending or running), skipping: $folder_name"
        continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would submit: sbatch -A $ACCOUNT $job_script"
    else
        echo "submitting: $job_script"
        out=$(sbatch -A "$ACCOUNT" "$job_script" 2>&1) && echo "$out" || {
            echo "$out"
            if echo "$out" | grep -q "AssocMaxSubmitJobLimit\|QOSMaxSubmitJobPerUserLimit\|job submit limit"; then
                echo "submit limit of the account reached -> stop here; run this script (or 08_fill_queue.sh) again later"
                skipped_limit=1
                break
            fi
            exit 1
        }
    fi
    submitted=$((submitted + 1))
done
echo "$submitted jobs $([ "$DRY_RUN" -eq 1 ] && echo 'would be ')submitted."
[ "$skipped_limit" -eq 1 ] && exit 3
exit 0
