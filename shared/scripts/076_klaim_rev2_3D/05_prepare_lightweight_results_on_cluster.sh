#!/usr/bin/env bash
# Run on the cluster login node after the jobs finished: collects the light-weight results
# (tables, parameters, logs, event lists, stiffness matrices; plus the event-snapshot field files
# and the regular field files of the w_s = 1.0 reference cases) into one tar.gz for download.
# Usage: ./05_prepare_lightweight_results_on_cluster.sh [--tables-only | --with-fields]
#   --tables-only : only tables, parameters, event lists, stiffness (tens of MB; enough for all J / energy evaluations)
#   --with-fields : additionally ALL h5/xdmf files (tens of GB)
# SLURM .out/.err files are not copied in full (the dolfinx/Newton logs reach GB); only their last 300 lines (*.tail.txt).
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
SOURCE_DIR="${SOURCE_DIR:-${HPC_SCRATCH}/${PROJECT_NAME}}"
TARGET_DIR="${TARGET_DIR:-${HPC_SCRATCH}/${PROJECT_NAME}_lightweight}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${TARGET_DIR}.tar.gz}"
WITH_FIELDS=0; TABLES_ONLY=0
[ "${1:-}" = "--with-fields" ] && WITH_FIELDS=1
[ "${1:-}" = "--tables-only" ] && TABLES_ONLY=1

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
cd "$SOURCE_DIR"

find . -type d -not -path '*/.cache*' -exec mkdir -p "$TARGET_DIR/{}" \;
find . -type f \( -name parameters.txt -o -name 'run_simulation_graphs*.txt' -o -name 'run_simulation_log.txt' \
    -o -name 'run_simulation_events.txt' -o -name 'effective_stiffness_matrix.txt' -o -name case.env -o -name job_script.sh \
    -o -name graphs.png \) -not -path '*/.cache/*' -exec cp --parents {} "$TARGET_DIR" \;
# only the tails of the SLURM logs
find . -type f \( -name '*.out.*' -o -name '*.err.*' \) -not -path '*/.cache/*' | while read -r f; do
    tail -n 300 "$f" | grep -v "INFO|" > "$TARGET_DIR/$f.tail.txt" || true
done
[ "$TABLES_ONLY" -eq 1 ] && { tar -czf "$ARCHIVE_PATH" -C "$(dirname "$TARGET_DIR")" "$(basename "$TARGET_DIR")"; echo "Archive for download: $ARCHIVE_PATH"; du -sh "$ARCHIVE_PATH"; exit 0; }
find . -type f \( -name 'run_simulation_events.xdmf' -o -name 'run_simulation_events.h5' \) -exec cp --parents {} "$TARGET_DIR" \;

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
