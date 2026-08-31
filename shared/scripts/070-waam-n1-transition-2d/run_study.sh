#!/usr/bin/env bash
# Hoehenstudie: FE-Bestaetigung der vorbereiteten Auswertefenster.
# IM DOLFINX-CONTAINER ausfuehren (v0.7.3).
#
# Voraussetzung (auf dem Host, nur numpy/scipy/PIL):
#   python3 study_rois.py --txt .../WAAM_N=1_A12D_Uebergangsbereich.txt \
#                         --bmp .../WAAM_N=1_A12D_Uebergangsbereich.bmp
# schreibt micro_band1..4.npz, micro_full.npz und study_cases.json.
#
#   bash run_study.sh                 # alle Faelle, s(x) = 1
#   CASES="full" NP=8 bash run_study.sh
#
# `full` hat 905 025 Zellen (1,8 Mio. Freiheitsgrade) -- mit mehreren MPI-Ranks
# rechnen. Der Loeser ist CG+GAMG, der Speicherbedarf skaliert linear.
#
# Ergebnisse: E_<tag>.json, fields_<tag>.npz, ps_<tag>.xdmf je Fall.
# Danach:  python3 report/make_study_figs.py --src ... --fe
# ersetzt die Schaetzerkurven durch die FE-Profile.
set -euo pipefail
cd "$(dirname "$0")"

NP="${NP:-1}"
CASES="${CASES:-band1 band2 band3 band4 full}"
SFUN="${SFUN:-1.0}"

for tag in $CASES; do
  micro="micro_${tag}.npz"
  if [ ! -f "$micro" ]; then
    echo "$micro fehlt - study_rois.py zuerst auf dem Host laufen lassen."
    exit 1
  fi
  echo "=== $tag :  s(x) = $SFUN  ($micro) ==="
  if [ "$NP" -gt 1 ]; then
    mpirun -np "$NP" python3 solve_plane_stress.py --micro "$micro" \
           --tag "$tag" --sfun "$SFUN"
  else
    python3 solve_plane_stress.py --micro "$micro" --tag "$tag" --sfun "$SFUN"
  fi
done

echo
echo "Fertig. Vergleich mit den Schranken:  python3 study_bounds.py"
