#!/bin/bash
# Progress overview of all case folders of this project on the cluster.
# Usage: ./07_status.sh [--match TEXT]
# Columns: SLURM state, accepted steps, pseudo-time t, crack tip x_ct, progress = share of the crack path from the initial
# crack tip to the stop position (x_min + STOP_FRACTION*(x_max - x_min)), elapsed and remaining wall time (h) of a running
# job, a linear ETA for the rest (rough: crack jumps make the progress uneven) and the exit line of finished jobs.
# "!" after the ETA marks a running job whose linear ETA exceeds its remaining time limit.
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
BASE_DIR="${HPC_SCRATCH}/${PROJECT_NAME}"
MATCH_TEXT="${2:-}"; [ "${1:-}" = "--match" ] || MATCH_TEXT=""

queue=$(squeue --noheader --user "$USER" --format "%j|%T|%M|%L" 2>/dev/null)
to_hours() {   # SLURM time [D-]HH:MM:SS / MM:SS -> hours
    awk -v s="$1" 'BEGIN { if (s == "" || s ~ /INVALID|UNLIMITED/) { print ""; exit }
        d = 0; if (index(s, "-") > 0) { split(s, a, "-"); d = a[1]; s = a[2] }
        n = split(s, b, ":"); if (n == 3) { h = b[1]; m = b[2]; x = b[3] } else if (n == 2) { h = 0; m = b[1]; x = b[2] } else { h = 0; m = 0; x = b[1] }
        printf "%.1f", d * 24 + h + m / 60 + x / 3600 }'
}

printf "%-28s %-8s %7s %9s %9s %5s %6s %6s %7s  %s\n" case slurm steps t x_ct prog elap_h left_h eta_h "exit"
for d in "$BASE_DIR"/*/; do
    name=$(basename "$d")
    [ -n "$MATCH_TEXT" ] && [[ "$name" != *"$MATCH_TEXT"* ]] && continue
    q=$(echo "$queue" | awk -F'|' -v n="r2_$name" '$1 == n' | head -1)
    state=$(echo "$q" | cut -d'|' -f2); [ -z "$state" ] && state="-"
    elap=$(to_hours "$(echo "$q" | cut -d'|' -f3)"); left=$(to_hours "$(echo "$q" | cut -d'|' -f4)")

    g="$d/run_simulation_graphs.txt"
    steps=0; t="-"; x="-"; x0=""
    if [ -f "$g" ]; then
        xcol=4; c="$d/run_simulation_graphs_columns.txt"   # 2D: column 4 = x_ct; 3D has J_z before it -> index from the columns file
        [ -f "$c" ] && xcol=$(( $(tr ' ' '\n' < "$c" | grep ":x_ct$" | cut -d: -f1) + 1 ))
        steps=$(grep -vc "^#" "$g")
        if [ "$steps" -gt 0 ]; then
            last=$(grep -v "^#" "$g" | tail -1); t=$(echo "$last" | awk '{print $1}'); x=$(echo "$last" | awk -v k="$xcol" '{print $k}')
            x0=$(grep -v "^#" "$g" | head -1 | awk -v k="$xcol" '{print $k}')
        fi
    fi

    out=$(ls -t "$d"/*.out.* 2>/dev/null | head -1)
    xr=""; [ -n "$out" ] && xr=$(grep -h -m1 "x_min, x_max" "$out" 2>/dev/null | sed 's/.*://')
    sf=$(grep -h "^STOP_FRACTION=" "$d/job_script.sh" 2>/dev/null | cut -d= -f2); [ -z "$sf" ] && sf=0.9
    prog=$(awk -v x="$x" -v x0="$x0" -v r="$xr" -v f="$sf" 'BEGIN {
        if (x == "-" || x0 == "" || r == "") { print "-"; exit }
        split(r, a, ","); xmin = a[1] + 0; xmax = a[2] + 0; tgt = xmin + f * (xmax - xmin)
        if (tgt <= x0 + 1e-12) { print "-"; exit }
        p = (x - x0) / (tgt - x0) * 100; if (p < 0) p = 0; if (p > 100) p = 100; printf "%.0f%%", p }')

    eta="-"
    if [ "$state" = "RUNNING" ] && [ -n "$elap" ] && [[ "$prog" == *% ]]; then
        eta=$(awk -v e="$elap" -v p="${prog%\%}" -v l="$left" 'BEGIN { if (p + 0 <= 0) { print "-"; exit }
            r = e * (100 - p) / p; flag = (l != "" && r > l + 0) ? "!" : ""; printf "%.1f%s", r, flag }')
    fi
    exitline=$([ -n "$out" ] && grep -h "finished with exit code\|Simulation stopped early\|Error\|error" "$out" 2>/dev/null | tail -1 | sed 's/^\[[^]]*\] //' | cut -c1-40)
    printf "%-28s %-8s %7s %9s %9s %5s %6s %6s %7s  %s\n" "$name" "$state" "$steps" "$t" "$x" "$prog" "${elap:--}" "${left:--}" "$eta" "$exitline"
done
