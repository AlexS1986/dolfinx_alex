#!/usr/bin/env bash
# 077: all 1D-bar runs of revision 2, serial each, in parallel, in the alex-dolfinx container (run from anywhere on the Mac).
set -u
SHARED="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMG="${IMG:-dolfinx_alex-alex-dolfinx}"
COMMON="--beta 0.2 --mu 1.0 --lam 0.0 --eta 1e-5 --eps 0.1"
run() { # script tag args...
  local script=$1 tag=$2; shift 2
  docker run --rm -v "$SHARED":/home -w /home/scripts/077_klaim_rev2_1D_bar "$IMG" \
    python3 "$script" $COMMON --tag "$tag" "$@" > "$SHARED/scripts/077_klaim_rev2_1D_bar/stdout$tag.txt" 2>&1
  echo "finished $tag (exit $?)"
}
run run_simulation_plasticity.py     _vM_gc1.0          --gc 1.0 --mob 1000 --sig_y 1.0 --hard 0.2222222 --tend 4.0 &
run run_simulation_linear_elastic.py _el_gc1.0          --gc 1.0    --mob 1000 --tend 2.0 &
run run_simulation_linear_elastic.py _el_gc0.5523       --gc 0.5523 --mob 1000 --tend 2.0 &
run run_simulation_linear_elastic.py _el_gc0.5679       --gc 0.5679 --mob 1000 --tend 2.0 &
run run_simulation_plasticity.py     _vM_gc1.0_rate0.5  --gc 1.0 --mob 1000  --sig_y 1.0 --hard 0.2222222 --tend 8.0 --rate 0.5 &
run run_simulation_plasticity.py     _vM_gc1.0_M10000   --gc 1.0 --mob 10000 --sig_y 1.0 --hard 0.2222222 --tend 4.0 &
wait
