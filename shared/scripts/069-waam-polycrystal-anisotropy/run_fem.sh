#!/bin/bash
# Run all WAAM elastic FE computations inside the dolfinx v0.7.3 container:
#   * KUBC homogenization  -> Chom_<MAT>.json          (full 6x6 stiffness)
#   * uniaxial tension      -> Emodul_<MAT>_<orient>.json (directional E-modulus)
#
# The meshes must already be staged in ./inputs/ (run prepare_inputs.py on the
# host first). Reads config.json for the single-crystal constants.
#
# From dolfinx_alex/ (host):
#   docker compose up -d --build
#   docker compose exec alex-dolfinx bash -c \
#     "cd /home/scripts/069-waam-polycrystal-anisotropy && bash run_fem.sh"
# or inside the container, in this folder:  bash run_fem.sh
#
# Env:
#   MATERIALS="316L 17-4PH"   steels (default both)
#   N=300                     RVE grain count (mesh name waam_<MAT>_n<N>.xdmf)
#   ORIENTS="V H 45deg"       tensile-bar orientations (default all three)
#   HOMOG=1 UNIAXIAL=1        toggle each computation set (default both on)
#   NP=1                      MPI ranks (NP>1 -> mpirun -np NP)
set -e
cd "$(dirname "$0")"

MATERIALS="${MATERIALS:-316L 17-4PH}"
N="${N:-300}"
ORIENTS="${ORIENTS:-V H 45deg}"
HOMOG="${HOMOG:-1}"
UNIAXIAL="${UNIAXIAL:-1}"
NP="${NP:-1}"

if [ "$NP" -gt 1 ] 2>/dev/null; then
    RUN="mpirun -np $NP python3"
else
    RUN="python3"
fi

run_one() {                     # $1 script  $2 mesh  $3 ori  $4 tag
    local script="$1" mesh="$2" ori="$3" tag="$4"
    if [ ! -f "$mesh" ] || [ ! -f "$ori" ]; then
        echo ">> SKIP $tag (missing $(basename "$mesh") or $(basename "$ori"))"
        return
    fi
    echo ""
    echo ">>>>>> $script  [$tag] <<<<<<"
    $RUN "$script" --mesh "$mesh" --ori "$ori" --tag "$tag"
}

# ---- 1) homogenization: full effective stiffness per steel -----------------
if [ "$HOMOG" = "1" ]; then
    echo "########## KUBC homogenization ##########"
    for M in $MATERIALS; do
        run_one homogenize_rve.py \
            "inputs/waam_${M}_n${N}.xdmf" "inputs/grain_ori_${M}.txt" "$M"
    done
fi

# ---- 2) uniaxial tensile test: directional E-modulus -----------------------
if [ "$UNIAXIAL" = "1" ]; then
    echo ""
    echo "########## uniaxial tension (directional E) ##########"
    for M in $MATERIALS; do
        for O in $ORIENTS; do
            run_one uniaxial_tension.py \
                "inputs/spec_${M}_${O}.xdmf" "inputs/grain_ori_${M}_${O}.txt" "${M}_${O}"
        done
    done
fi

echo ""
echo "==================== FEM done ===================="
ls -1 Chom_*.json Emodul_*.json 2>/dev/null | sed 's/^/  result: /' || true
echo "(Chom_<MAT>.json = 6x6 stiffness + directional E; Emodul_<MAT>_<orient>.json = uniaxial E)"
