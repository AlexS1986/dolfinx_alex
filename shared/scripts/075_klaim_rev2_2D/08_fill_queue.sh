#!/bin/bash
# Top up the SLURM queue with the revision-2 jobs (2D and 3D) in priority order until the account's submit limit
# is reached (sbatch: AssocMaxSubmitJobLimit). Already queued, running or finished cases are skipped by 03_submit_all.sh,
# so this script can be run every morning until it reports "everything submitted".
# Usage: ./08_fill_queue.sh [--dry-run]
SCRIPTS=$(dirname "$(realpath "$0")")/..
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1
ORDER=(
    "075_klaim_rev2_2D S1_vM"      # the paper sweep with plasticity (longest runs first)
    "075_klaim_rev2_2D S1_EqSig"
    "075_klaim_rev2_2D S1_EqJc"
    "076_klaim_rev2_3D S1_3D"
    "075_klaim_rev2_2D S3_"        # cubic embedding
    "076_klaim_rev2_3D S3_3D"
    "075_klaim_rev2_2D S4_"        # sensitivity
    "075_klaim_rev2_2D S5_"        # representativity probes
)
for spec in "${ORDER[@]}"; do
    set -- $spec
    "$SCRIPTS/$1/03_submit_all.sh" --match "$2" ${DRY:+--dry-run} 2>&1 | grep -v "already has SLURM output\|already queued"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 3 ]; then echo "== submit limit reached; run ./08_fill_queue.sh again later"; exit 0; fi
    if [ "$rc" -ne 0 ]; then echo "== 03_submit_all.sh failed (exit $rc)"; exit "$rc"; fi
done
echo "== everything submitted"
