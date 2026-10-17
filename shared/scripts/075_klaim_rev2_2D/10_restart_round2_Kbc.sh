#!/bin/bash
# Round 2 of the 2D porous runs with K_bc chosen per case so that J_bc = gc (K_BC_RULE=equal_gc in 02_create_jobs.sh).
# Round 1 (K_bc^2 = 2.5 for all cases, 2026-10-16/17) overdrove the small-w_s runs: J_bc/J_eff up to 4.4 and the crack
# 3-4 w_c ahead of the K-field centre, which broke the Eq J_c / Eq sigma* scaling for w_s <= 0.4.
# What this script does (run on the cluster login node inside this folder, after git pull):
#   1. cancels the queued/running porous 2D jobs of this project (S1_, S3_, S4_, S5_; the vM/EqJc strips and all 3D jobs stay),
#      [Eq sigma*] additionally gets J_c = 0.5523 (equal 1D strength at beta = 0.2, see cases_2D.list) -> its strip is rerun too,
#   2. moves the round-1 case folders to $HPC_SCRATCH/<project>_round1_K2.5/ (same file system, instant),
#   3. recreates the case folders from the template, writes the job scripts and submits them (03_submit_all.sh).
# Usage: ./10_restart_round2_Kbc.sh [--dry-run]
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
BASE_DIR="${HPC_SCRATCH:?HPC_SCRATCH is not defined}/${PROJECT_NAME}"
PARK_DIR="${BASE_DIR}_round1_K2.5"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
PATTERN='^S[1345]_|^S2_strip_EqSig$'    # porous 2D cases + the Eq sigma* strip (J_c 0.5679 -> 0.5523); the other strips stay

# 1. cancel
# match by job name (%Z is the sbatch cwd on Lichtenberg, not the scratch folder): porous 2D = r2_S1/S3/S4/S5 without _3D_, plus the EqSig strip
rows=$(squeue --noheader --user "${USER:-$(id -un)}" --format "%A|%j|%T|%Z" | awk -F'|' '($2 ~ /^r2_S[1345]_/ && $2 !~ /_3D_/) || $2 == "r2_S2_strip_EqSig"')
if [ -n "$rows" ]; then
    echo "cancelling:"; printf "%s\n" "$rows" | awk -F'|' '{printf "  %s  %-28s %s\n", $1, $2, $3}'
    ids=$(printf "%s\n" "$rows" | cut -d'|' -f1)
    if [ "$DRY" -eq 0 ]; then
        scancel $ids
        # wait until the jobs have left the queue: a COMPLETING job still writes (by path) into its folder and would
        # re-create it after the mv below, and 03_submit_all.sh skips names that are still in squeue (seen 2026-10-17)
        for i in $(seq 1 60); do
            left=$(squeue --noheader --jobs "$(echo $ids | tr ' ' ',')" --format "%A" 2>/dev/null | wc -l)
            [ "$left" -eq 0 ] && break
            echo "  waiting for $left job(s) to finish cancelling ..."; sleep 10
        done
    fi
else
    echo "no porous 2D jobs in the queue"
fi

# 2. park round 1
mkdir -p "$PARK_DIR"
n=0
for d in "$BASE_DIR"/*/; do
    name=$(basename "$d"); [[ "$name" =~ $PATTERN ]] || continue
    if [ -e "$PARK_DIR/$name" ]; then echo "already parked, leaving in place: $name"; continue; fi
    echo "park: $name -> $PARK_DIR/"
    [ "$DRY" -eq 0 ] && mv "$d" "$PARK_DIR/"
    n=$((n + 1))
done
echo "parked $n folders"
[ "$DRY" -eq 1 ] && { echo "dry run: stopping before 01/02/03"; exit 0; }

# 3. recreate and submit
"$SCRIPT_DIR/01_create_directories.sh"
"$SCRIPT_DIR/02_create_jobs.sh"
grep -h "^K_BC_RULE=" "$BASE_DIR"/S*/job_script.sh | sort | uniq -c
"$SCRIPT_DIR/03_submit_all.sh" || true
echo "== done; check with ./07_status.sh and, if some cases were skipped as 'already queued', run ./03_submit_all.sh again"
