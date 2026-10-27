#!/bin/bash
# Resubmit cases that were stopped by the SLURM time limit (or otherwise ended without finishing) so that they continue
# from their restart data (run_simulation.py --restart auto, see RESTART_NACH_TIMEOUT.md). Identical in 075 and 076.
#
# A case folder is resubmitted when
#   * its job (name from "#SBATCH -J" of job_script.sh) is not in squeue, and
#   * the newest *.out.* does not contain "finished with exit code 0", and
#   * the last sacct state of the job name is TIMEOUT, or the newest *.err.* says "DUE TO TIME LIMIT", or a checkpoint
#     (run_simulation_checkpoint.json) exists that is not marked finished.
# Folders that never ran (no *.out.*) are left to 03_submit_all.sh; crashed runs (exit code != 0) are only listed.
#
# Folders created before the restart code (2026-10-19) are upgraded first: run_simulation.py and klaim_restart.py are
# copied from 000_template (old script kept as run_simulation.py.pre_restart) and job_script.sh is rewritten by
# 02_create_jobs.sh --exact <folder>. Such runs continue from their last field output (run_simulation.xdmf, every
# POSTPROCESSING_INTERVAL steps) because they have no checkpoint.
#
# Account / nodes as in 02_create_jobs.sh: -A $ACCOUNT (default p0023647) and --exclude=$EXCLUDE are passed to sbatch
# (override the values in old job scripts), no -C; OPENBLAS_CORETYPE=SkylakeX is set inside job_script.sh.
#
# Usage: ./11_resubmit_timeouts.sh [--dry-run] [--match TEXT] [--max N]
#        ACCOUNT=special00008 ./11_resubmit_timeouts.sh ...      (other account)
#        SACCT_DAYS=30 ...                                         (sacct look-back, default 21 days)
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
TEMPLATE_DIR="$SCRIPT_DIR/000_template"
BASE_DIR="${HPC_SCRATCH:?HPC_SCRATCH is not defined}/${PROJECT_NAME}"
ACCOUNT="${ACCOUNT:-p0023647}"
if [ "$ACCOUNT" = "l0003507" ]; then echo "ERROR: account l0003507 is never used (rule 2026-10-27, 046_plasticity/CLAUDE.md, Konto). Use p0023647." >&2; exit 1; fi
EXCLUDE="${EXCLUDE:-mpsd0138,mpsd0319,mpsd0401,mpsd0415,mpsd0557}"
SACCT_DAYS="${SACCT_DAYS:-21}"
DRY_RUN=0
MATCH_TEXT=""
MAX_SUBMISSIONS=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --match) MATCH_TEXT="$2"; shift 2 ;;
        --max) MAX_SUBMISSIONS="$2"; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-SkylakeX}"   # (also set in job_script.sh)

queued_names=$(squeue --noheader --user "$USER" --format "%200j" 2>/dev/null | sed "s/ *$//" || true)
# state of the newest job (highest job ID) per job name (-X = allocations only)
sacct_states=$(sacct --noheader --parsable2 -X --user "$USER" --starttime "$(date -d "-${SACCT_DAYS} days" +%F)" \
               --format JobID,JobName,State 2>/dev/null \
               | awk -F'|' '{ split($3, s, " "); id = $1 + 0; if (!($2 in max) || id >= max[$2]) { max[$2] = id; st[$2] = $1 "|" s[1] } }
                            END { for (n in st) print n "|" st[n] }' || true)

printf "%-30s %-10s %-12s %-10s %s\n" case sacct checkpoint action detail
submitted=0
for folder_path in "$BASE_DIR"/*/; do
    [ -d "$folder_path" ] || continue
    name=$(basename "$folder_path")
    [ -n "$MATCH_TEXT" ] && [[ "$name" != *"$MATCH_TEXT"* ]] && continue
    job_script="${folder_path}job_script.sh"
    [ -f "$job_script" ] || continue
    job_name=$(awk '/^#SBATCH -J /{print $3; exit}' "$job_script")
    [ -n "$job_name" ] || job_name="r2_${name}"

    out=$(ls -t "$folder_path"*.out.* 2>/dev/null | head -1 || true)
    err=$(ls -t "$folder_path"*.err.* 2>/dev/null | head -1 || true)
    state=$(echo "$sacct_states" | awk -F'|' -v n="$job_name" '$1 == n {print $3}' | head -1)
    ckpt="-"
    if [ -f "${folder_path}run_simulation_checkpoint.json" ]; then
        if grep -q '"finished": true' "${folder_path}run_simulation_checkpoint.json"; then ckpt="finished"; else ckpt="yes"; fi
    elif [ -f "${folder_path}run_simulation.xdmf" ]; then
        ckpt="fields"   # legacy run: field output only
    fi

    if echo "$queued_names" | grep -qx "$job_name"; then
        printf "%-30s %-10s %-12s %-10s %s\n" "$name" "${state:--}" "$ckpt" "skip" "in squeue"; continue
    fi
    if [ -z "$out" ]; then
        printf "%-30s %-10s %-12s %-10s %s\n" "$name" "${state:--}" "$ckpt" "skip" "never ran (03_submit_all.sh)"; continue
    fi
    if grep -q "finished with exit code 0" "$out" || [ "$ckpt" = "finished" ]; then
        printf "%-30s %-10s %-12s %-10s %s\n" "$name" "${state:--}" "$ckpt" "skip" "finished"; continue
    fi
    timeout=0
    [ "$state" = "TIMEOUT" ] && timeout=1
    [ -n "$err" ] && grep -q "DUE TO TIME LIMIT" "$err" && timeout=1
    if [ "$timeout" -eq 0 ] && [ "$ckpt" != "yes" ]; then
        reason=$(grep -h "finished with exit code" "$out" | tail -1 | sed 's/^\[[^]]*\] //')
        printf "%-30s %-10s %-12s %-10s %s\n" "$name" "${state:--}" "$ckpt" "skip" "no timeout (${reason:-state ${state:-unknown}}) - check $(basename "$out")"; continue
    fi
    if [ -n "$MAX_SUBMISSIONS" ] && [ "$submitted" -ge "$MAX_SUBMISSIONS" ]; then
        printf "%-30s %-10s %-12s %-10s %s\n" "$name" "${state:--}" "$ckpt" "later" "--max reached"; continue
    fi

    # --- upgrade folders created before the restart code
    upgrade=""
    grep -q "klaim_restart" "${folder_path}run_simulation.py" 2>/dev/null || upgrade="py"
    grep -q -- "--restart" "$job_script" || upgrade="${upgrade:+$upgrade+}job"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf "%-30s %-10s %-12s %-10s %s\n" "$name" "${state:--}" "$ckpt" "would sub" "${upgrade:+upgrade $upgrade; }sbatch -A $ACCOUNT --exclude=$EXCLUDE $job_script"
        submitted=$((submitted + 1)); continue
    fi
    if [[ "$upgrade" == *py* ]]; then
        cp -p "${folder_path}run_simulation.py" "${folder_path}run_simulation.py.pre_restart"
        cp "$TEMPLATE_DIR/run_simulation.py" "$TEMPLATE_DIR/klaim_restart.py" "$folder_path"
    fi
    [ -f "${folder_path}klaim_restart.py" ] || cp "$TEMPLATE_DIR/klaim_restart.py" "$folder_path"
    if [[ "$upgrade" == *job* ]]; then
        "$SCRIPT_DIR/02_create_jobs.sh" --exact "$name" >/dev/null
        grep -q -- "--restart" "$job_script" || { echo "ERROR: job_script.sh of $name could not be upgraded"; exit 1; }
    fi
    sb_out=$(sbatch -A "$ACCOUNT" --exclude="$EXCLUDE" "$job_script" 2>&1) || {
        echo "$sb_out"
        if echo "$sb_out" | grep -q "AssocMaxSubmitJobLimit\|QOSMaxSubmitJobPerUserLimit\|job submit limit"; then
            echo "submit limit of the account reached -> stop here; run this script again later"; exit 3
        fi
        exit 1
    }
    printf "%-30s %-10s %-12s %-10s %s\n" "$name" "${state:--}" "$ckpt" "submitted" "${upgrade:+upgraded $upgrade; }$sb_out"
    submitted=$((submitted + 1))
done
echo "$submitted job(s) $([ "$DRY_RUN" -eq 1 ] && echo 'would be ')resubmitted."
