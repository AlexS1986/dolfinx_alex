#!/usr/bin/env bash
# Campaign 260930_M1e5: rerun of all a = 6 cases used in the manuscript with mobility M = 1e5 mm^3/(Nmm s)
# and the residual-based reaction forces (columns 14-17 of result_graphs_*.txt).
# 36 runs: beta_phi = 0.01: 5 structures x 4 beta_s; beta_phi = 0.001 and 0.05: E_var, rho = 0.3/0.6, 4 beta_s each.
# Runs locally in Docker (image dolfinx_alex-alex-dolfinx, DOLFINx 0.7.3), PARALLEL runs at a time with
# 6 MPI tasks each. Restartable: finished runs (file DONE in the simulation folder) are skipped.
#
#   HPC_SCRATCH   campaign root (default: external disk, see below)
#   PARALLEL      simultaneous runs (default 2)
#   WALL_LIMIT    seconds per run before the container is stopped (default 10800)
#   MOBILITY      default 100000
#   Solver settings (2026-09-30 diagnosis, MOBILITY_STUDY_FINDINGS.md section 11): NEWTON_ATOL 1e-8 (the dolfinx
#   default 1e-10 is below the round-off floor of the residual and made every load drop end in a dt -> 1e-14 walk),
#   NEWTON_MAX_IT 20, PHASEFIELD_DT_MIN 1e-10 s (converged steps down to 1e-9 s occur in load drops), common end point PHASEFIELD_T_END 0.03 s (u_y = 0.03 mm).
#
# Usage:  bash 05_run_260930_mobility_1e5_campaign_local.sh            # all runs
#         bash 05_run_260930_mobility_1e5_campaign_local.sh --list     # print the run list only
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHARED=$(cd "$SCRIPT_DIR/../../.." && pwd)
PROJ=/home/scripts/063-Special-Issue-IJF-Hannover/$(basename "$SCRIPT_DIR")
CAMPAIGN_ROOT="${HPC_SCRATCH:-/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/campaign_260930_M1e5}"
PARALLEL="${PARALLEL:-2}"
WALL_LIMIT="${WALL_LIMIT:-21600}"
MOBILITY="${MOBILITY:-100000}"
NEWTON_ATOL="${NEWTON_ATOL:-1e-8}"
NEWTON_MAX_IT="${NEWTON_MAX_IT:-20}"
PHASEFIELD_DT_MIN="${PHASEFIELD_DT_MIN:-1e-10}"
PHASEFIELD_T_END="${PHASEFIELD_T_END:-0.03}"
INPUT_ROOT_NAME=260504_dcb_beta_phi_a_rho_var_min_max
R1="$SCRIPT_DIR/resources/260504_dcb_beta_phi_a_rho_var_min_max"
R5="$SCRIPT_DIR/resources/260526_a_6_rho_0_3-0_6_beta_phi_0_05_var_min_max/cluster_input_var"

# group/leaf relative to the resource root; "5:" marks the beta_phi = 0.05 resource root
MAIN_LEAVES=(
  "1:beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_3_var"
  "1:beta_phi_0_001-0_01_a_3-6_rho_0_3_min/beta_0_01_a_6_rho_0_3_min"
  "1:beta_ph_0_001-0_01_a_3-6_rho_0_3-0_6_max/beta_0_01_a_6_rho_0_3_max"
  "1:beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_6_var"
  "1:beta_ph_0_001-0_01_a_3-6_rho_0_3-0_6_max/beta_0_01_a_6_rho_0_6_max"
)
BETA_LEAVES=(
  "1:beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_001_a_6_rho_0_3_var"
  "1:beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_001_a_6_rho_0_6_var"
  "5:beta_phi_0_05_a_6_rho_0_3-0_6_var/beta_0_05_a_6_rho_0_3_var"
  "5:beta_phi_0_05_a_6_rho_0_3-0_6_var/beta_0_05_a_6_rho_0_6_var"
)

# run list, most important first: main results at beta_s = 0.03, other beta_s, then the beta_phi study
JOBS=()
for leaf in "${MAIN_LEAVES[@]}"; do JOBS+=("0.03 $leaf"); done
for eps in 0.06 0.015 0.045; do for leaf in "${MAIN_LEAVES[@]}"; do JOBS+=("$eps $leaf"); done; done
for eps in 0.03 0.06 0.015 0.045; do for leaf in "${BETA_LEAVES[@]}"; do JOBS+=("$eps $leaf"); done; done

label() { sed 's/\./_/g' <<< "$1"; }

run_case() {
  local eps=$1 spec=$2
  local root=$R1; [ "${spec%%:*}" = "5" ] && root=$R5
  local leaf=${spec#*:}
  local leaf_name; leaf_name=$(basename "$leaf")
  local sim="$CAMPAIGN_ROOT/simulation_CAMPAIGN260930_M1e5_SPLITspectral_EPS$(label "$eps")_M$(label "$MOBILITY")_${leaf_name}"
  local name="mobcamp_EPS$(label "$eps")_${leaf_name}"
  if [ -f "$sim/DONE" ]; then echo "SKIP $name (already done)"; return 0; fi
  rm -rf "$sim"; mkdir -p "$sim/resources/$INPUT_ROOT_NAME/$leaf"
  cp -p "$SCRIPT_DIR/000_template/01_phasefield_dcb_260504_folder.py" "$SCRIPT_DIR/000_template/04_mesh2dlfxmesh.py" "$sim/"
  for f in active_cells_mapping cell_data.csv connectivity.csv mesh.h5 mesh.xdmf node_coords.csv points_data.csv; do
    cp -p "$root/$leaf/$f" "$sim/resources/$INPUT_ROOT_NAME/$leaf/$f" || { echo "FAILED $name: missing input $f"; return 1; }
  done
  printf 'split=spectral\nepsilon=%s\nmobility=%s\nleaf=%s\nsource_root=%s\nNEWTON_ATOL=%s\nNEWTON_MAX_IT=%s\nPHASEFIELD_DT_MIN=%s\nPHASEFIELD_T_END=%s\nstarted=%s\n' \
    "$eps" "$MOBILITY" "$leaf" "$root" "$NEWTON_ATOL" "$NEWTON_MAX_IT" "$PHASEFIELD_DT_MIN" "$PHASEFIELD_T_END" "$(date)" > "$sim/run_parameters.txt"
  local start; start=$(date +%s)
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --rm --name "$name" --cpus 6 -v "$SHARED":/home -v "$sim":/work -w /work \
    -e PYTHONPATH=$PROJ/code/vendor:/home/utils:/usr/local/dolfinx-real/lib/python3.10/dist-packages:/usr/local/lib \
    -e PROCESSOR_NUMBER=6 -e PHASEFIELD_SCRIPT=/work/01_phasefield_dcb_260504_folder.py \
    -e NEWTON_ATOL="$NEWTON_ATOL" -e NEWTON_MAX_IT="$NEWTON_MAX_IT" -e PHASEFIELD_DT_MIN="$PHASEFIELD_DT_MIN" -e PHASEFIELD_T_END="$PHASEFIELD_T_END" \
    -e MESH_SCRIPT=/work/04_mesh2dlfxmesh.py dolfinx_alex-alex-dolfinx \
    bash -c "bash $PROJ/00_jobs/run_260929_mobility_local.sh /work/resources/$INPUT_ROOT_NAME spectral $eps $MOBILITY > /work/run_local.log 2>&1" >/dev/null \
    || { echo "FAILED $name: docker run"; return 1; }
  local status="finished"
  while docker ps --format '{{.Names}}' | grep -q "^$name$"; do
    sleep 30
    if [ $(( $(date +%s) - start )) -gt "$WALL_LIMIT" ]; then docker stop -t 10 "$name" >/dev/null 2>&1; status="stopped at wall limit"; fi
  done
  local graph; graph=$(find "$sim" -name 'result_graphs_*.txt' | head -1)
  local steps=0; [ -n "$graph" ] && steps=$(grep -vc '^#' "$graph")
  if [ "$steps" -lt 5 ]; then echo "FAILED $name: only $steps steps, see $sim/run_local.log"; return 1; fi
  echo "$status after $(( $(date +%s) - start )) s, $steps steps" > "$sim/DONE"
  echo "DONE $name: $status after $(( $(date +%s) - start )) s, $steps steps"
}

if [ "${1:-}" = "--list" ]; then printf '%s\n' "${JOBS[@]}"; echo "${#JOBS[@]} runs"; exit 0; fi
if [ "${1:-}" = "--one" ]; then run_case "$2" "$3"; exit $?; fi

mkdir -p "$CAMPAIGN_ROOT"
echo "Campaign root: $CAMPAIGN_ROOT, ${#JOBS[@]} runs, $PARALLEL at a time, M = $MOBILITY, started $(date)"
printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -L 1 bash "${BASH_SOURCE[0]}" --one
echo "ALLDONE $(date)"
