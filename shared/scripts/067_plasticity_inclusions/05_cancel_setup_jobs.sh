#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${SCRIPT_DIR}")"
DRY_RUN=0
MATCH_TEXT="${MATCH_TEXT:-}"

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--match TEXT]

Cancel queued/running SLURM jobs for ${PROJECT_NAME}.

Options:
  --dry-run       Print matching jobs without cancelling them.
  --match TEXT    Further restrict matches to jobs whose name or workdir contains TEXT.

Examples:
  $0 --dry-run
  $0
  $0 --match WSTEG0.25
  $0 --match KINC1.0_GCINC1.0
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --match)
            if [ "$#" -lt 2 ]; then
                echo "Error: --match requires a value."
                usage
                exit 1
            fi
            MATCH_TEXT="$2"
            shift 2
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

if ! command -v squeue >/dev/null 2>&1; then
    echo "Error: squeue was not found. This script is intended for the SLURM cluster login node."
    exit 1
fi

job_rows=$(
    squeue --noheader --user "${USER}" --format "%A|%j|%T|%Z" \
        | awk -F'|' -v project="${PROJECT_NAME}" -v match_text="${MATCH_TEXT}" '
            {
                project_match = index($2, project) || index($4, "/" project "/") || index($4, project)
                extra_match = (match_text == "") || index($2, match_text) || index($4, match_text)
                if (project_match && extra_match) {
                    print $0
                }
            }
        ' \
        | sort -u
)

if [ -z "${job_rows}" ]; then
    echo "No queued or running jobs found for ${PROJECT_NAME}."
    if [ -n "${MATCH_TEXT}" ]; then
        echo "Additional match filter was: ${MATCH_TEXT}"
    fi
    exit 0
fi

echo "Matching jobs:"
printf "%s\n" "${job_rows}" | awk -F'|' '{printf "  %s  %-24s %-12s %s\n", $1, $2, $3, $4}'

job_ids=$(printf "%s\n" "${job_rows}" | awk -F'|' '{print $1}' | sort -u)

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "Dry run: no jobs cancelled."
    exit 0
fi

if ! command -v scancel >/dev/null 2>&1; then
    echo "Error: scancel was not found. Cannot cancel matching jobs."
    exit 1
fi

scancel ${job_ids}
echo "Cancelled $(printf "%s\n" "${job_ids}" | wc -w) jobs for ${PROJECT_NAME}."
