#!/bin/bash
# OpenBLAS positive control on an i02 node (Sapphire Rapids) with the container the jobs use:
# once without and once with the OPENBLAS_CORETYPE fix. Results land in $HPC_SCRATCH/<project>/blas_check/.
# Expected: without fix "core: Cooperlake", dgemm deviation ~1e1; with fix "core: SkylakeX", both ~1e-13.
# Only after the "with fix" run prints RESULT: OK may CONSTRAINT in 02_create_jobs.sh include i02.
# Usage: ACCOUNT=p0023647 ./09_blas_check_i02.sh            then, when the two jobs are done:  ./09_blas_check_i02.sh --show
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
OUT="$HPC_SCRATCH/$PROJECT_NAME/blas_check"
container="$HOME/dolfinx_alex/alex-dolfinx.sif"
ACCOUNT="${ACCOUNT:-p0023647}"
if [ "$ACCOUNT" = "l0003507" ]; then echo "ERROR: account l0003507 is never used (rule 2026-10-27, 046_plasticity/CLAUDE.md, Konto). Use p0023647." >&2; exit 1; fi
mkdir -p "$OUT" && cp "$SCRIPT_DIR/blas_check.py" "$OUT/"
if [ "${1:-}" = "--show" ]; then
    for f in "$OUT"/*.out; do echo "== $f"; grep -v "^\[I\]\|^  \[I\]" "$f" | grep "host\|core:\|dgemm\|dgesv\|RESULT\|OPENBLAS"; done
    exit 0
fi
for mode in nofix fix; do
    if [ "$mode" = "fix" ]; then env="OPENBLAS_CORETYPE=SkylakeX,APPTAINERENV_OPENBLAS_CORETYPE=SkylakeX"; else env="NONE"; fi
    sbatch -A "$ACCOUNT" -p deflt -C i02 --exclude=mpsd0138,mpsd0319,mpsd0401,mpsd0415 -n 1 -N 1 --mem-per-cpu=8000 -t 10 -J "blas_$mode" \
        -o "$OUT/blas_$mode.out" -e "$OUT/blas_$mode.err" --export="ALL,$env" \
        --wrap "cd $OUT && apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$OUT:/work $container python3 $OUT/blas_check.py"
done
echo "two jobs submitted; when finished:  $0 --show"
