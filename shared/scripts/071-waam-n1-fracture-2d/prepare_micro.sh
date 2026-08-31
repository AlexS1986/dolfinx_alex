#!/usr/bin/env bash
# 071 - build the two microstructure patches for the fracture study.
#
# Runs on the HOST (numpy/pandas/scipy/PIL, no dolfinx), because the EBSD raw
# data lives outside the container mount. It reuses the preprocessing of
# project 070 unchanged - 071 adds no second EBSD reader.
#
#   micro_long.npz   ROI of 070: 17-4PH | transition | 316L along x.
#                    The crack later runs along +x = the build-direction axis
#                    and crosses the transition zone.
#   micro_trans.npz  a tall strip INSIDE the transition zone. It is used with
#                    --rotate_ccw90, so the crack again runs along +x in the
#                    FE frame but now transverse to the build direction.
#
# Usage:   bash prepare_micro.sh            (defaults below)
#          EBSD_DIR=... P070=... bash prepare_micro.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- where things are on the host ------------------------------------------
NEPER_ROOT="${NEPER_ROOT:-/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/Meshing/Neper}"
EBSD_DIR="${EBSD_DIR:-${NEPER_ROOT}/data/04_anisotropy_waam/data_c04/uebergangsbereich}"
P070="${P070:-${SCRIPT_DIR}/../070-waam-n1-transition-2d}"

TXT="${TXT:-${EBSD_DIR}/WAAM_N=1_A12D_Uebergangsbereich.txt}"
BMP="${BMP:-${EBSD_DIR}/WAAM_N=1_A12D_Uebergangsbereich.bmp}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- ROIs (map frame, um; origin top-left, y DOWN) -------------------------
# identical to config.json -> "roi"; keep the two in sync.
LONG_ROI="${LONG_ROI:-30 1339 1900 2225}"
LONG_ZONES="${LONG_ZONES:-667 1495}"
# Querstreifen in der MISCHZONE, nicht im 070-Zonenrechteck: x = 667..1495 ist
# flaechenmaessig zu 88.6 % Austenit (ueber die volle Kartenhoehe gemessen),
# x = 450..1000 hat FCC-Anteil 0.67. Die Zonengrenzen sind gleich dem Streifen,
# damit der ganze Streifen region 1 ist. Abweichung von 070 - im Bericht nennen.
TRANS_ROI="${TRANS_ROI:-450 700 1000 2600}"
TRANS_ZONES="${TRANS_ZONES:-450 1000}"
STEP="${STEP:-3.371}"

for f in "${TXT}" "${BMP}"; do
    [ -f "${f}" ] || { echo "missing EBSD input: ${f}"; exit 1; }
done
[ -f "${P070}/preprocess_ebsd_to_grid.py" ] || {
    echo "missing 070 preprocessing: ${P070}/preprocess_ebsd_to_grid.py"; exit 1; }

cd "${P070}"
echo "[1/2] micro_long  (Riss in Aufbaurichtung, quer durch die Uebergangszone)"
"${PYTHON_BIN}" preprocess_ebsd_to_grid.py --txt "${TXT}" --bmp "${BMP}" \
    --roi ${LONG_ROI} --step "${STEP}" --zones ${LONG_ZONES} \
    --tag long --outdir "${SCRIPT_DIR}"

echo "[2/2] micro_trans (Riss quer zur Aufbaurichtung, in der Mischzone)"
"${PYTHON_BIN}" preprocess_ebsd_to_grid.py --txt "${TXT}" --bmp "${BMP}" \
    --roi ${TRANS_ROI} --step "${STEP}" --zones ${TRANS_ZONES} \
    --tag trans --outdir "${SCRIPT_DIR}"

echo
echo "fertig:"
ls -lh "${SCRIPT_DIR}"/micro_long.npz "${SCRIPT_DIR}"/micro_trans.npz
echo "micro_trans.npz IMMER mit --rotate_ccw90 verwenden."
