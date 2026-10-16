#!/bin/bash
# Short scaling + smoke test on the cluster: runs the reference case (default S1_vM_ws1.0, truncated at
# STOP_FRACTION of the crack path) with several core counts and short time limits, so that
#   (a) the new scripts are verified end to end, and
#   (b) the cost per time step vs. core count is measured  ->  python3 scaling_report.py
# Usage (login node, inside this folder, after git pull):
#   ./00_scaling_test.sh                  # 16 32 48 cores, STOP_FRACTION=0.20, 6 h limit
#   CORES="32 64" STOP_FRACTION=0.15 ./00_scaling_test.sh
#   CASE=S1_EqSig_ws0.375 ./00_scaling_test.sh
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
CASES_FILE="$SCRIPT_DIR/cases_$( [ "$PROJECT_NAME" = "076_klaim_rev2_3D" ] && echo 3D || echo 2D ).list"
CASE="${CASE:-$( [ "$PROJECT_NAME" = "076_klaim_rev2_3D" ] && echo S1_3D_vM_ws1.0 || echo S1_vM_ws1.0 )}"
CORES="${CORES:-16 32 48}"
STOP_FRACTION="${STOP_FRACTION:-0.20}"
TIME_LIMIT="${TIME_LIMIT:-360}"          # minutes
SUBMIT="${SUBMIT:-1}"
CONSTRAINT="${CONSTRAINT:-}"             # e.g. CONSTRAINT=i02 to test the OpenBLAS fix on the Sapphire Rapids nodes (passed through to 02_create_jobs.sh)

if [ -z "${HPC_SCRATCH:-}" ]; then echo "HPC_SCRATCH not defined"; exit 1; fi
BASE_DIR="${HPC_SCRATCH}/${PROJECT_NAME}"
mkdir -p "$BASE_DIR"

# temporary cases file with one line per core count (same physics, different folder names)
tmp_cases=$(mktemp)
line=$(grep -E "^${CASE}[[:space:]]" "$CASES_FILE" || true)
if [ -z "$line" ]; then echo "case $CASE not found in $CASES_FILE"; exit 1; fi
for n in $CORES; do
    echo "$line" | awk -v n="$n" '{ $1 = "scaling_" $1 "_n" n; print }' >> "$tmp_cases"
done
cat "$tmp_cases"

"$SCRIPT_DIR/01_create_directories.sh" "$tmp_cases" --match scaling_
"$SCRIPT_DIR/02_create_jobs.sh" --match scaling_

for n in $CORES; do
    folder="$BASE_DIR/scaling_${CASE}_n${n}"
    js="$folder/job_script.sh"
    [ -n "$CONSTRAINT" ] && sed -i -e "s#^\#SBATCH -C .*#\#SBATCH -C ${CONSTRAINT}#" "$js"
    sed -i -e "s|^#SBATCH -n .*|#SBATCH -n ${n}|" \
           -e "s|^#SBATCH -t .*|#SBATCH -t ${TIME_LIMIT}|" \
           -e "s|^STOP_FRACTION=.*|STOP_FRACTION=${STOP_FRACTION}|" \
           -e "s|srun -n [0-9]* apptainer exec --bind \$bindpath \$container python3 \$working_directory/run_simulation.py|srun -n ${n} apptainer exec --bind \$bindpath \$container python3 \$working_directory/run_simulation.py|" \
           "$js"
    grep -n "SBATCH -n\|SBATCH -t\|SBATCH -C\|^STOP_FRACTION\|run_simulation.py" "$js" | head -6
    if [ "$SUBMIT" -eq 1 ]; then sbatch "$js"; else echo "(not submitted: SUBMIT=0)"; fi
done
rm -f "$tmp_cases"
echo "When the jobs are done:  python3 $SCRIPT_DIR/scaling_report.py"
