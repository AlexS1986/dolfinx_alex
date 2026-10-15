#!/bin/bash
# Progress overview of all case folders of this project on the cluster. Usage: ./07_status.sh [--match TEXT]
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
BASE_DIR="${HPC_SCRATCH}/${PROJECT_NAME}"
MATCH_TEXT="${2:-}"; [ "${1:-}" = "--match" ] || MATCH_TEXT=""
printf "%-34s %-10s %7s %9s %9s  %s\n" case slurm steps t x_ct "last line / exit"
for d in "$BASE_DIR"/*/; do
    name=$(basename "$d")
    [ -n "$MATCH_TEXT" ] && [[ "$name" != *"$MATCH_TEXT"* ]] && continue
    state=$(squeue --noheader --user "$USER" --format "%j %T" 2>/dev/null | awk -v n="r2_$name" '$1==n {print $2}' | head -1)
    [ -z "$state" ] && state="-"
    g="$d/run_simulation_graphs.txt"
    if [ -f "$g" ]; then
        steps=$(grep -vc "^#" "$g"); last=$(grep -v "^#" "$g" | tail -1); t=$(echo "$last" | awk '{print $1}'); x=$(echo "$last" | awk '{print $4}')
    else steps=0; t="-"; x="-"; fi
    exitline=$(ls "$d"/*.out.* 2>/dev/null | tail -1 | xargs -r grep -h "finished with exit code\|Simulation stopped early\|Error\|error" 2>/dev/null | tail -1 | cut -c1-60)
    printf "%-34s %-10s %7s %9s %9s  %s\n" "$name" "$state" "$steps" "$t" "$x" "$exitline"
done
