#!/usr/bin/env bash
# 2D-Plane-Stress-Rechnung der WAAM-N=1-Uebergangszone -- dolfinx v0.7.3.
# IM DOLFINX-CONTAINER ausfuehren. Alle Ergebnisse (E_<tag>.json,
# fields_<tag>.npz, ps_<tag>.xdmf) stammen ausschliesslich aus diesem Lauf.
#
#   bash run_fem.sh              # rechnet alle drei Standardfaelle
#   CASES=roi bash run_fem.sh    # nur den Fall s(x)=1
#   NP=4 bash run_fem.sh         # mit 4 MPI-Ranks
#
# Eigener Fall ohne Skript:
#   python3 solve_plane_stress.py --micro micro_roi.npz --tag meinfall \
#           --sfun "1 + 0.3*np.exp(-((x-1050.)/400.)**2)"
set -euo pipefail
cd "$(dirname "$0")"

MICRO="${MICRO:-micro_roi.npz}"
NP="${NP:-1}"
CASES="${CASES:-roi roi_s133 roi_gauss}"

sfun_for() {
  case "$1" in
    roi)       echo "1.0" ;;
    roi_s133)  echo "1.33" ;;
    roi_gauss) echo "1 + 0.50*np.exp(-((x-1050.)/350.)**2)" ;;
    *)         echo "${SFUN:-1.0}" ;;
  esac
}

if [ ! -f "$MICRO" ]; then
  echo "$MICRO fehlt - preprocess_ebsd_to_grid.py zuerst auf dem Host laufen lassen."
  exit 1
fi

for tag in $CASES; do
  sfun="$(sfun_for "$tag")"
  echo "=== $tag :  s(x) = $sfun ==="
  if [ "$NP" -gt 1 ]; then
    mpirun -np "$NP" python3 solve_plane_stress.py --micro "$MICRO" \
           --tag "$tag" --sfun "$sfun"
  else
    python3 solve_plane_stress.py --micro "$MICRO" --tag "$tag" --sfun "$sfun"
  fi
done

echo
echo "Fertig. Ergebnisse: E_<tag>.json, fields_<tag>.npz, ps_<tag>.xdmf"
echo "Abbildung erzeugen:  python3 make_figures.py"
