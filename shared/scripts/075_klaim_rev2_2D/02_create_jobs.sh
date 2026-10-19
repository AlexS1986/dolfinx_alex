#!/bin/bash
# Write job_script.sh into every case folder (reads case.env written by 01_create_directories.sh).
# Global (study-wide) parameters are set here. Usage: ./02_create_jobs.sh [--match TEXT | --exact FOLDER]
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
JOB_TEMPLATE_PATH="$SCRIPT_DIR/00_jobs/job_template.sh"
MATCH_TEXT=""
EXACT_NAME=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --match) MATCH_TEXT="$2"; shift 2 ;;
        --exact) EXACT_NAME="$2"; shift 2 ;;   # only this folder (used by 11_resubmit_timeouts.sh)
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${HPC_SCRATCH:-}" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi
BASE_DIR="${HPC_SCRATCH}/${PROJECT_NAME}"

# ------------------------------------------------------------------ global parameters (paper values)
NHOLES=6
DHOLE=1.0
E0=0.02                 # element edge length along the crack path  (Delta h = 0.02 L)
E1=0.6                  # coarse element size
MESH_FILE="mesh_fracture_adaptive.xdmf"
LAM_MICRO_PARAM=1.0
MUE_MICRO_PARAM=1.0
EPS_PARAM=0.1           # epsilon = 0.1 L
ELEMENT_ORDER=1
BETA=0.2                # cubic degradation function slope parameter (manuscript: 0.2)
ETA=0.00001             # residual stiffness (manuscript: 1e-5)
MOB=1000.0              # mobility
SIG_Y_EMBEDDING=1000000.0   # linear elastic embedding (manuscript); set to 1.0 to mimic the old global yield stress
BC_MODULI="effective"   # surfing K-field with (E_eff, nu_eff) of the embedding (= all previous runs, manuscript corrected accordingly); "fixed" = E_BC/NU_BC below
K_BC_RULE="${K_BC_RULE:-equal_gc}"   # value | equal_gc
# equal_gc: K_bc^2 = 2 mu_bc gc/(1 - nu_bc) so that J_bc = gc for every case (2026-10-17: with the fixed K_bc^2 = 2.5 the
#   small-w_s runs were overdriven, J_bc/J_eff up to 4.4, crack 3-4 w_c ahead of the K-field; the elastic sets become self-similar)
E_BC=2.5
NU_BC=0.25
STOP_FRACTION=0.9       # stop when x_ct >= STOP_FRACTION * domain length (0.9 = paper); small values for tests
POSTPROCESSING_INTERVAL=400
EVENT_SNAPSHOTS=1

# ------------------------------------------------------------------ resources
MEMORY_VALUE=4000       # MB per core
PROCESSOR_NUMBER=16       # scaling test 2026-10-15: 1.57/1.35/1.46 s per step on 16/32/48 cores -> 16 (6 jobs per i01 node)
TIME=7200               # minutes (5 days -> partition long; longest 2D run w_s = 3.0 estimated ~50 h on 16 cores)
TIME_SHORT=1440         # minutes (24 h -> default partition deflt, more capacity) for cases with w_s <= WSTEG_SHORT_MAX (estimated <= 10 h)
WSTEG_SHORT_MAX=1.0     # (elastic cases only, see below)
CONSTRAINT="${CONSTRAINT:-}"        # node type(s), e.g. "i01" or "i01|i02"; empty = no constraint (handoff 2026-10-16: leave out -C i01; i02 needs the OpenBLAS fix, which the template sets)
EXCLUDE="${EXCLUDE:-mpsd0138,mpsd0319,mpsd0401,mpsd0415,mpsd0557}"   # faulty nodes (handoff 2026-10-16; mpsd0557: SIGBUS 16.10.)
ACCOUNT="${ACCOUNT:-p0023647}"     # SLURM account (handoff: p0023647; p0025962 = follow-on phase once as12vapa is a member; special00008 only sparingly; never l0003507). p0023647 had MaxSubmit=0 on 2026-10-16.

count=0
for folder_path in "${BASE_DIR}"/*/; do
    [ -d "$folder_path" ] || continue
    folder_name=$(basename "$folder_path")
    [ -f "$folder_path/case.env" ] || { echo "no case.env in $folder_name, skipping"; continue; }
    if [ -n "$MATCH_TEXT" ] && [[ "$folder_name" != *"$MATCH_TEXT"* ]]; then
        continue
    fi
    [ -n "$EXACT_NAME" ] && [ "$folder_name" != "$EXACT_NAME" ] && continue
    # shellcheck disable=SC1090
    unset NEWTON_MIN_ITERS GEOMETRY
    source "$folder_path/case.env"
    job_name="r2_${CASE_NAME}"
    time_limit=$TIME; awk -v w="$WSTEG" -v m="$WSTEG_SHORT_MAX" 'BEGIN{exit !(w+0 <= m+0)}' && time_limit=$TIME_SHORT
    [ "${GEOMETRY:-porous}" = "strip" ] && time_limit=$TIME     # the vM strip needs > 24 h (tiny steps)
    [ "$MATERIAL" = "vonmises" ] && time_limit=$TIME            # round 1: four vM runs with w_s <= 1 hit the 24 h limit in the last pore
    k_rule=$K_BC_RULE
    sig_y_emb=$SIG_Y_EMBEDDING
    case "${GEOMETRY:-porous}" in
        strip) k_rule=value ;;                                   # homogeneous strips (Sec. 3.5) keep K_bc^2 = 2.5 Jc0 mu0 (J_bc = 0.94 Jc0); the crack regulates itself there
        solid) k_rule=value; sig_y_emb=$SIG_Y ;;                 # solid = strip mesh + the same yield stress everywhere (one homogeneous material, 2026-10-17)
    esac
    sed -e "s|{PROJECT_NAME}|${PROJECT_NAME}|g" \
        -e "s|{FOLDER_NAME}|${folder_name}|g" \
        -e "s|{JOB_NAME}|${job_name}|g" \
        -e "s|{WSTEG}|${WSTEG}|g" \
        -e "s|{MATERIAL}|${MATERIAL}|g" \
        -e "s|{GC_MICRO_PARAM}|${GC_MICRO_PARAM}|g" \
        -e "s|{SIG_Y}|${SIG_Y}|g" \
        -e "s|{HARD}|${HARD}|g" \
        -e "s|{EMBEDDING}|${EMBEDDING}|g" \
        -e "s|{NROWS}|${NROWS}|g" \
        -e "s|{CRACK_FACTOR}|${CRACK_FACTOR}|g" \
        -e "s|{GEOMETRY}|${GEOMETRY}|g" \
        -e "s|{NEWTON_MIN_ITERS}|${NEWTON_MIN_ITERS:-}|g" \
        -e "s|{NHOLES}|${NHOLES}|g" \
        -e "s|{DHOLE}|${DHOLE}|g" \
        -e "s|{E0}|${E0}|g" \
        -e "s|{E1}|${E1}|g" \
        -e "s|{MESH_FILE}|${MESH_FILE}|g" \
        -e "s|{LAM_MICRO_PARAM}|${LAM_MICRO_PARAM}|g" \
        -e "s|{MUE_MICRO_PARAM}|${MUE_MICRO_PARAM}|g" \
        -e "s|{EPS_PARAM}|${EPS_PARAM}|g" \
        -e "s|{ELEMENT_ORDER}|${ELEMENT_ORDER}|g" \
        -e "s|{BETA}|${BETA}|g" \
        -e "s|{ETA}|${ETA}|g" \
        -e "s|{MOB}|${MOB}|g" \
        -e "s|{SIG_Y_EMBEDDING}|${sig_y_emb}|g" \
        -e "s|{BC_MODULI}|${BC_MODULI}|g" \
        -e "s|{K_BC_RULE}|${k_rule}|g" \
        -e "s|{E_BC}|${E_BC}|g" \
        -e "s|{NU_BC}|${NU_BC}|g" \
        -e "s|{STOP_FRACTION}|${STOP_FRACTION}|g" \
        -e "s|{POSTPROCESSING_INTERVAL}|${POSTPROCESSING_INTERVAL}|g" \
        -e "s|{EVENT_SNAPSHOTS}|${EVENT_SNAPSHOTS}|g" \
        -e "s|{MEMORY_VALUE}|${MEMORY_VALUE}|g" \
        -e "s|{PROCESSOR_NUMBER}|${PROCESSOR_NUMBER}|g" \
        -e "s|{TIME}|${time_limit}|g" \
        -e "s#{CONSTRAINT}#${CONSTRAINT}#g" \
        -e "s#{ACCOUNT}#${ACCOUNT}#g" \
        -e "s#{EXCLUDE}#${EXCLUDE}#g" \
        "${JOB_TEMPLATE_PATH}" > "${folder_path}/job_script.sh"
    [ -z "$CONSTRAINT" ] && sed -i "/^#SBATCH -C $/d" "${folder_path}/job_script.sh"
    [ -z "$EXCLUDE" ] && sed -i "/^#SBATCH --exclude=$/d" "${folder_path}/job_script.sh"
    if grep -q "{[A-Z_]*}" "${folder_path}/job_script.sh"; then
        echo "WARNING: unreplaced placeholder in ${folder_path}/job_script.sh:"
        grep -o "{[A-Z_]*}" "${folder_path}/job_script.sh" | sort -u
    fi
    count=$((count + 1))
    echo "wrote ${folder_path}job_script.sh"
done
echo "Wrote $count job scripts. SLURM account: ${ACCOUNT}"
[ "$ACCOUNT" != "p0023647" ] && echo "WARNING: account is ${ACCOUNT}, not p0023647 (shell variable ACCOUNT set?). Unset it unless intended: unset ACCOUNT"
