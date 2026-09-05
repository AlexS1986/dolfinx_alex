#!/usr/bin/env bash
# 2D-Plane-Stress-Rechnung mit eigen-skalierter Uebergangszonen-Steifigkeit
# (Projekt 072) -- dolfinx v0.7.3. IM DOLFINX-CONTAINER ausfuehren.
# Alle berichteten Ergebnisse stammen ausschliesslich aus diesen Laeufen.
#
#   bash run_fem.sh                  # rechnet alle Standardfaelle
#   CASES=base bash run_fem.sh      # nur ein Fall
#   NP=4 bash run_fem.sh            # mit 4 MPI-Ranks
#
# Eigener Fall ohne Skript (Faktoren = numpy-Ausdruecke in x [um]):
#   python3 solve_plane_stress_eigen.py --micro "$MICRO" --tag meinfall \
#           --aC44 "1 + 0.4*np.exp(-((x-1050.)/350.)**2)"
set -euo pipefail
cd "$(dirname "$0")"

# Mikrostruktur aus 070 (EBSD-exakt, unveraendert wiederverwendet)
MICRO="${MICRO:-../070-waam-n1-transition-2d/micro_roi.npz}"
NP="${NP:-1}"
CASES="${CASES:-base s133 k160 cp160 c44_160}"

# tag -> aK aCp aC44 (nur region 1 = Uebergangszone)
factors_for() {
  case "$1" in
    base)    echo "1.0 1.0 1.0" ;;                 # reine Mikrostruktur = 070 roi
    s133)    echo "1.33 1.33 1.33" ;;              # 070 roi_s133 (Aequivalenzcheck)
    k160)    echo "1.60 1.0 1.0" ;;                # nur Kompressionsmodul K
    cp160)   echo "1.0 1.60 1.0" ;;                # nur tetragonaler Schub C'
    c44_160) echo "1.0 1.0 1.60" ;;                # nur trigonaler Schub C44
    *)       echo "${AK:-1.0} ${ACP:-1.0} ${AC44:-1.0}" ;;
  esac
}

if [ ! -f "$MICRO" ]; then
  echo "$MICRO fehlt - 070/preprocess_ebsd_to_grid.py zuerst auf dem Host laufen lassen."
  exit 1
fi

for tag in $CASES; do
  read -r aK aCp aC44 <<< "$(factors_for "$tag")"
  echo "=== $tag :  aK=$aK  aCp=$aCp  aC44=$aC44 ==="
  if [ "$NP" -gt 1 ]; then
    mpirun -np "$NP" python3 solve_plane_stress_eigen.py --micro "$MICRO" \
           --tag "$tag" --aK "$aK" --aCp "$aCp" --aC44 "$aC44"
  else
    python3 solve_plane_stress_eigen.py --micro "$MICRO" \
           --tag "$tag" --aK "$aK" --aCp "$aCp" --aC44 "$aC44"
  fi
done

echo
echo "Fertig. Ergebnisse: E_<tag>.json, fields_<tag>.npz, ps_<tag>.xdmf"
echo "Fit gegen den DIC-Zielwert:  python3 fit_eigen.py --model B|C --engine dolfinx"
