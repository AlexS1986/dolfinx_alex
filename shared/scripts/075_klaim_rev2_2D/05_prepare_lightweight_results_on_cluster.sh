#!/usr/bin/env bash
# Run on the cluster login node after the jobs finished: collects the light-weight results
# (tables, parameters, logs, event lists, stiffness matrices; plus the event-snapshot field files
# and the regular field files of the w_s = 1.0 reference cases) into one tar.gz for download.
# Usage: ./05_prepare_lightweight_results_on_cluster.sh [--with-fields]   (--with-fields adds ALL h5/xdmf files)
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
SOURCE_DIR="${SOURCE_DIR:-${HPC_SCRATCH}/${PROJECT_NAME}}"
TARGET_DIR="${TARGET_DIR:-${HPC_SCRATCH}/${PROJECT_NAME}_lightweight}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${TARGET_DIR}.tar.gz}"
WITH_FIELDS=0
[ "${1:-}" = "--with-fields" ] && WITH_FIELDS=1

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
cd "$SOURCE_DIR"

find . -type d -exec mkdir -p "$TARGET_DIR/{}" \;
find . -type f \( -name parameters.txt -o -name 'run_simulation_graphs*.txt' -o -name 'run_simulation_log.txt' \
    -o -name 'run_simulation_events.txt' -o -name 'effective_stiffness_matrix.txt' -o -name case.env -o -name job_script.sh \
    -o -name '*.out.*' -o -name '*.err.*' -o -name graphs.png -o -name 'run_simulation_events.xdmf' -o -name 'run_simulation_events.h5' \) \
    -exec cp --parents {} "$TARGET_DIR" \;

if [ "$WITH_FIELDS" -eq 1 ]; then
    find . -type f \( -name 'run_simulation.xdmf' -o -name 'run_simulation.h5' \) -exec cp --parents {} "$TARGET_DIR" \;
else
    # regular field output only for the reference cases (name contains ws1.0)
    for d in ./*ws1.0*/; do
        [ -d "$d" ] || continue
        for f in run_simulation.xdmf run_simulation.h5; do
            [ -f "$d/$f" ] && cp --parents "$d/$f" "$TARGET_DIR" || true
        done
    done
fi

tar -czf "$ARCHIVE_PATH" -C "$(dirname "$TARGET_DIR")" "$(basename "$TARGET_DIR")"
echo "Archive for download: $ARCHIVE_PATH"
du -sh "$ARCHIVE_PATH"
