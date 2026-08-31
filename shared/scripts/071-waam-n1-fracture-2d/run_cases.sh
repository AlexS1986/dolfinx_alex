#!/usr/bin/env bash
# 071 - the production runs: 2 crack directions x 3 stiffness variants of the
# transition zone (the same three s(x) variants as project 070).
#
# Requires micro_long.npz and micro_trans.npz (see prepare_micro.sh, host side).
# Run inside the container:
#
#   docker exec -it alex-dolfinx bash -lc \
#     "cd /home/scripts/071-waam-n1-fracture-2d && NP=10 bash run_cases.sh"
#
# Subsets:  CASES="long_s1 trans_s1" bash run_cases.sh
# Knobs:    NP, EPSILON, GC, K_SCALE, STEPS (0 = unlimited), OUT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NP="${NP:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIEXEC_BIN="${MPIEXEC_BIN:-mpirun}"
OUT="${OUT:-${SCRIPT_DIR}/results}"

EPSILON="${EPSILON:-12.0}"
GC="${GC:-10.0}"
K_SCALE="${K_SCALE:-1.05}"
STEPS="${STEPS:-0}"
PPI="${PPI:-50}"

# --- Netzaufloesung -----------------------------------------------------------
PRESET="${PRESET:-resolved}"
if [ "${PRESET}" = "coarse" ]; then
    # schnell, aber 59 % der 17-4PH-Flaeche liegt in Koernern < 3h: die feinen
    # Koerner werden von --stiffness_average homogenisiert, nicht aufgeloest.
    # Nur zum Ausprobieren, nicht fuer berichtete Zahlen.
    E_FINE="${E_FINE:-4.0}";  E_PATCH="${E_PATCH:-12.0}"; E_FAR="${E_FAR:-60.0}"
else
    # aufgeloest: Median-Korn der 17-4PH-Zone hat 5.4 um Aequivalentdurchmesser,
    # bei h = 2 um sind nur noch 8.9 % der Zone-0-Flaeche unteraufgeloest.
    # ~592k DOF (long) / ~448k DOF (trans).
    E_FINE="${E_FINE:-2.0}";  E_PATCH="${E_PATCH:-4.0}";  E_FAR="${E_FAR:-40.0}"
fi
CORRIDOR="${CORRIDOR:-60.0}"; MX_FRAC="${MX_FRAC:-0.15}"; MY_FRAC="${MY_FRAC:-0.5}"

# the three transition-zone stiffness variants of 070
S_1="1.0"
S_133="1.33"
S_GAUSS="1 + 0.50*np.exp(-((x-1050.)/350.)**2)"

# Sechs Faelle: 2 Rissrichtungen x 3 Steifigkeitsvarianten der Uebergangszone.
# Die Kernfrage (Richtungsabhaengigkeit von Gc_eff bei konstantem Gc) beantworten
# schon long_s1 + trans_s1; die s-Varianten sagen zusaetzlich, wie stark die in
# 070 angenommene Steifigkeitsueberhoehung auf Gc_eff durchschlaegt.
# Interpretationshinweis: im LAENGSfall variiert s(x) ENTLANG des Risspfads
# (der Riss durchquert die Uebergangszone), im QUERfall QUER dazu - dort ist der
# ganze Streifen region 1, s=1.33 wirkt also uniform, das Gauss-Profil als
# Steifigkeitsgradient senkrecht zum Risspfad.
# Empfehlung: erst long_s1 rechnen, Laufzeit und J-Plateau pruefen, dann den Rest.
CASES="${CASES:-long_s1 long_s133 long_sgauss trans_s1 trans_s133 trans_sgauss}"

mkdir -p "${OUT}"

sfun_of() {
    case "$1" in
        *_s1)     printf '%s' "${S_1}" ;;
        *_s133)   printf '%s' "${S_133}" ;;
        *_sgauss) printf '%s' "${S_GAUSS}" ;;
        *) echo "unknown case $1" >&2; exit 1 ;;
    esac
}

micro_of() {
    case "$1" in
        long_*)  printf '%s' "micro_long.npz" ;;
        trans_*) printf '%s' "micro_trans.npz" ;;
        *) echo "unknown case $1" >&2; exit 1 ;;
    esac
}

for case_name in ${CASES}; do
    MICRO="$(micro_of "${case_name}")"
    SFUN="$(sfun_of "${case_name}")"
    ROT=""
    MESHFLAG=""
    if [[ "${case_name}" == trans_* ]]; then
        ROT="--rotate_ccw90"
        MESHFLAG="--rotated"
    fi
    [ -f "${MICRO}" ] || { echo "missing ${MICRO} - run prepare_micro.sh on the host first"; exit 1; }

    MESH_NAME="mesh_${MICRO%.npz}_${PRESET}"
    if [ ! -f "${OUT}/${MESH_NAME}.xdmf" ]; then
        echo "--- Netz ${MESH_NAME} (PRESET=${PRESET})"
        "${PYTHON_BIN}" mesh_fracture_micro.py --micro "${MICRO}" ${MESHFLAG} \
            --epsilon "${EPSILON}" \
            --e-fine "${E_FINE}" --e-patch "${E_PATCH}" --e-far "${E_FAR}" \
            --corridor "${CORRIDOR}" \
            --margin-x-frac "${MX_FRAC}" --margin-y-frac "${MY_FRAC}" \
            --name "${MESH_NAME}" --outdir "${OUT}"
    fi

    echo "=== ${case_name}   micro=${MICRO}  s(x)=${SFUN}  ${ROT}"
    "${MPIEXEC_BIN}" -np "${NP}" "${PYTHON_BIN}" run_fracture_simulation.py \
        --mesh_file "${OUT}/${MESH_NAME}.xdmf" \
        --micro "${MICRO}" ${ROT} \
        --sfun "${SFUN}" \
        --tag "${case_name}" \
        --epsilon "${EPSILON}" --Gc "${GC}" --K_scale "${K_SCALE}" \
        --max_steps "${STEPS}" --postprocessing_interval "${PPI}" \
        --outdir "${OUT}"
done

echo
echo "=== Auswertung ==="
"${PYTHON_BIN}" evaluate_gc_eff.py ${CASES} --outdir "${OUT}"
