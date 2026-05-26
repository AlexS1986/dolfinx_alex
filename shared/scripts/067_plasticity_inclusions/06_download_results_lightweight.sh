#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-as1vapa@lcluster14.hrz.tu-darmstadt.de}"
REMOTE_BASE="${REMOTE_BASE:-/work/scratch/as12vapa/067_plasticity_inclusions}"
LOCAL_RESULTS="${LOCAL_RESULTS:-/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/067_plasticity_inclusions/results}"
CONTROL_PATH="${CONTROL_PATH:-${TMPDIR:-/tmp}/067_plasticity_inclusions_ssh_%r@%h:%p}"

mkdir -p "$LOCAL_RESULTS"

cleanup() {
    ssh -S "$CONTROL_PATH" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Opening SSH master connection to $REMOTE_HOST"
echo "You may be asked for your password and 2FA now."
ssh -M -S "$CONTROL_PATH" -fN "$REMOTE_HOST"

SSH_BASE=(
    ssh
    -S "$CONTROL_PATH"
    -o ControlMaster=no
    "$REMOTE_HOST"
)

echo "Downloading folder structure plus parameters.txt and run_simulation_graphs.txt"
"${SSH_BASE[@]}" \
    "cd '$REMOTE_BASE' && find . -type f \( -name parameters.txt -o -name run_simulation_graphs.txt \) -print | sort | tar -cf - --files-from=-" \
    | tar -xf - -C "$LOCAL_RESULTS"

first_wsteg_dir=""
while IFS= read -r parameters_file; do
    if grep -Eiq 'wsteg[^0-9+-]*1([.]0+)?([^0-9.]|$)' "$parameters_file"; then
        first_wsteg_dir="$(dirname "$parameters_file")"
        break
    fi
done < <(find "$LOCAL_RESULTS" -type f -name parameters.txt | sort)

if [[ -z "$first_wsteg_dir" ]]; then
    echo "No parameters.txt with wsteg = 1.0 was found."
    exit 1
fi

relative_case_dir="${first_wsteg_dir#"$LOCAL_RESULTS"/}"
echo "First wsteg = 1.0 folder:"
echo "  $relative_case_dir"
echo "Downloading run_simulation.xdmf and run_simulation.h5 for that folder"

"${SSH_BASE[@]}" \
    "cd '$REMOTE_BASE' && printf '%s\n' './$relative_case_dir/run_simulation.xdmf' './$relative_case_dir/run_simulation.h5' | tar -cf - --files-from=-" \
    | tar -xf - -C "$LOCAL_RESULTS"

echo "Done."
echo "Downloaded data are in:"
echo "  $LOCAL_RESULTS"
