#!/bin/bash
# Write job_script.sh into every case folder (reads case.env written by 01_create_directories.sh).
# Global (study-wide) parameters are set here. Usage: ./02_create_jobs.sh [--match TEXT]
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
JOB_TEMPLATE_PATH="$SCRIPT_DIR/00_jobs/job_template.sh"
MATCH_TEXT=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --match) MATCH_TEXT="$2"; shift 2 ;;
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
NL_EFFECTIVE=5          # mesh density parameter of the unit cell (ronny.mesh), as in 061
NL_FRACTURE=5           # mesh density parameter of the fracture mesh (ronny.mesh), as in 061
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
E_BC=2.5
NU_BC=0.25
STOP_FRACTION=0.9       # stop when x_ct >= STOP_FRACTION * domain length (0.9 = paper); small values for tests
POSTPROCESSING_INTERVAL=200
EVENT_SNAPSHOTS=1

# ------------------------------------------------------------------ resources
MEMORY_VALUE=5000       # MB per core
PROCESSOR_NUMBER=64
TIME=10080              # minutes (7 days)
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
    # shellcheck disable=SC1090
    source "$folder_path/case.env"
    job_name="r2_${CASE_NAME}"  # 3D
    sed -e "s|{PROJECT_NAME}|${PROJECT_NAME}|g" \
        -e "s|{FOLDER_NAME}|${folder_name}|g" \
        -e "s|{JOB_NAME}|${job_name}|g" \
        -e "s|{WSTEG}|${WSTEG}|g" \
        -e "s|{MATERIAL}|${MATERIAL}|g" \
        -e "s|{GC_MICRO_PARAM}|${GC_MICRO_PARAM}|g" \
        -e "s|{SIG_Y}|${SIG_Y}|g" \
        -e "s|{HARD}|${HARD}|g" \
        -e "s|{EMBEDDING}|${EMBEDDING}|g" \
        -e "s|{CRACK_FACTOR}|${CRACK_FACTOR}|g" \
        -e "s|{NHOLES}|${NHOLES}|g" \
        -e "s|{DHOLE}|${DHOLE}|g" \
        -e "s|{NL_EFFECTIVE}|${NL_EFFECTIVE}|g" \
        -e "s|{NL_FRACTURE}|${NL_FRACTURE}|g" \
        -e "s|{MESH_FILE}|${MESH_FILE}|g" \
        -e "s|{LAM_MICRO_PARAM}|${LAM_MICRO_PARAM}|g" \
        -e "s|{MUE_MICRO_PARAM}|${MUE_MICRO_PARAM}|g" \
        -e "s|{EPS_PARAM}|${EPS_PARAM}|g" \
        -e "s|{ELEMENT_ORDER}|${ELEMENT_ORDER}|g" \
        -e "s|{BETA}|${BETA}|g" \
        -e "s|{ETA}|${ETA}|g" \
        -e "s|{MOB}|${MOB}|g" \
        -e "s|{SIG_Y_EMBEDDING}|${SIG_Y_EMBEDDING}|g" \
        -e "s|{BC_MODULI}|${BC_MODULI}|g" \
        -e "s|{E_BC}|${E_BC}|g" \
        -e "s|{NU_BC}|${NU_BC}|g" \
        -e "s|{STOP_FRACTION}|${STOP_FRACTION}|g" \
        -e "s|{POSTPROCESSING_INTERVAL}|${POSTPROCESSING_INTERVAL}|g" \
        -e "s|{EVENT_SNAPSHOTS}|${EVENT_SNAPSHOTS}|g" \
        -e "s|{MEMORY_VALUE}|${MEMORY_VALUE}|g" \
        -e "s|{PROCESSOR_NUMBER}|${PROCESSOR_NUMBER}|g" \
        -e "s|{TIME}|${TIME}|g" \
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
echo "Wrote $count job scripts."
