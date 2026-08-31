#!/usr/bin/env bash
# 071 - LOCAL SMOKE TEST, runs entirely inside the dolfinx container and needs
# no EBSD data. Minutes, not hours. Use it after every change to the model.
#
#   docker compose up -d                       # in .../dolfinx_alex
#   docker exec -it alex-dolfinx bash -lc \
#     "cd /home/scripts/071-waam-n1-fracture-2d && bash run_local_test.sh"
#
# Stages
#   1  numpy material selftest                (crystal math, ROI rotation)
#   2  dolfinx selftest of the new PF class   (incl. isotropic limit vs alex)
#   3  homogeneous verification run           (J must approach K^2/E')
#   4  synthetic microstructure run           (full pipeline, both directions)
#   5  evaluation                             (Gc_eff, figure)
#
# Knobs:  NP=<ranks>  STEPS=<max time steps per run>  STAGES="1 2 3"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NP="${NP:-4}"
# This is a SMOKE TEST: it checks that the machinery runs, not that Gc_eff is
# right. With 250 steps the crack leaves the transient and travels a few tens
# of um - evaluate_gc_eff.py will report a Gc_eff but flag it as covering only
# a fraction of the patch. A representative number needs the crack to cross the
# WHOLE patch: STEPS=0 (unlimited), which takes far longer.
STEPS="${STEPS:-250}"
STAGES="${STAGES:-1 2 3 4 5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIEXEC_BIN="${MPIEXEC_BIN:-mpirun}"
OUT="${OUT:-${SCRIPT_DIR}/test_out}"

EPSILON="${EPSILON:-8.0}"
GC="${GC:-10.0}"
LX="${LX:-400}"
LY="${LY:-200}"

mkdir -p "${OUT}"
has() { for s in ${STAGES}; do [ "$s" = "$1" ] && return 0; done; return 1; }
hr() { printf '\n=== %s ===\n' "$1"; }

if has 1; then
    hr "1/5  numpy material selftest"
    "${PYTHON_BIN}" selftest_material.py
fi

if has 2; then
    hr "2/5  dolfinx selftest of StaticPhaseFieldProblem2D_anisotropic"
    "${PYTHON_BIN}" selftest_phasefield.py
fi

if has 3; then
    hr "3/5  homogeneous verification run (J should approach K^2/E')"
    "${PYTHON_BIN}" mesh_fracture_micro.py --Lx "${LX}" --Ly "${LY}" \
        --epsilon "${EPSILON}" --name mesh_test --outdir "${OUT}"
    "${MPIEXEC_BIN}" -np "${NP}" "${PYTHON_BIN}" run_fracture_simulation.py \
        --mesh_file "${OUT}/mesh_test.xdmf" --tag homog \
        --epsilon "${EPSILON}" --Gc "${GC}" --K_scale 1.0 \
        --max_steps "${STEPS}" --postprocessing_interval 10 --outdir "${OUT}"
fi

if has 4; then
    hr "4/5  synthetic microstructure, both crack directions"
    # The crack always runs along +x, so the crack PATH is the patch's x extent
    # AFTER any rotation. To compare the two directions on the same geometry,
    # the transverse case therefore needs a patch that is TALL before rotation
    # - exactly how the real ROIs are cut (micro_long is wide, micro_trans is a
    # tall strip inside the transition zone). Rotating the wide patch instead
    # would leave the crack only LY um of path in a very tall, narrow domain.
    "${PYTHON_BIN}" make_test_micro.py synth --Lx "${LX}" --Ly "${LY}" \
        --out "${OUT}/micro_test.npz"
    "${PYTHON_BIN}" make_test_micro.py synth --Lx "${LY}" --Ly "${LX}" \
        --out "${OUT}/micro_test_tall.npz"

    # longitudinal: patch as generated
    "${PYTHON_BIN}" mesh_fracture_micro.py --micro "${OUT}/micro_test.npz" \
        --epsilon "${EPSILON}" --name mesh_test_long --outdir "${OUT}"
    "${MPIEXEC_BIN}" -np "${NP}" "${PYTHON_BIN}" run_fracture_simulation.py \
        --mesh_file "${OUT}/mesh_test_long.xdmf" --micro "${OUT}/micro_test.npz" \
        --tag test_long --epsilon "${EPSILON}" --Gc "${GC}" --K_scale 1.1 \
        --max_steps "${STEPS}" --postprocessing_interval 10 --outdir "${OUT}"

    # transverse: the tall patch, rotated by 90 deg (mesh must be rotated too)
    # -> same domain shape and same crack path length as the longitudinal case
    "${PYTHON_BIN}" mesh_fracture_micro.py --micro "${OUT}/micro_test_tall.npz" \
        --rotated --epsilon "${EPSILON}" --name mesh_test_trans --outdir "${OUT}"
    "${MPIEXEC_BIN}" -np "${NP}" "${PYTHON_BIN}" run_fracture_simulation.py \
        --mesh_file "${OUT}/mesh_test_trans.xdmf" --micro "${OUT}/micro_test_tall.npz" \
        --rotate_ccw90 --tag test_trans --epsilon "${EPSILON}" --Gc "${GC}" \
        --K_scale 1.1 --max_steps "${STEPS}" --postprocessing_interval 10 \
        --outdir "${OUT}"
fi

if has 5; then
    hr "5/5  evaluation"
    "${PYTHON_BIN}" evaluate_gc_eff.py --glob 'run_fracture_simulation_*_graphs.txt' \
        --outdir "${OUT}" || true
fi

hr "done"
echo "Ausgaben in ${OUT}"
