#!/usr/bin/env bash
# Run on the Mac: downloads the archive created by 05_prepare_lightweight_results_on_cluster.sh and
# unpacks it into ./results/. Usage: ./06_download_results_lightweight.sh
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
REMOTE_HOST="${REMOTE_HOST:-as1vapa@lcluster14.hrz.tu-darmstadt.de}"
REMOTE_ARCHIVE="${REMOTE_ARCHIVE:-/work/scratch/as12vapa/${PROJECT_NAME}_lightweight.tar.gz}"
LOCAL_RESULTS="${LOCAL_RESULTS:-${SCRIPT_DIR}/results}"
mkdir -p "$LOCAL_RESULTS"
echo "Downloading $REMOTE_HOST:$REMOTE_ARCHIVE (password + 2FA may be requested)"
scp "$REMOTE_HOST:$REMOTE_ARCHIVE" "$LOCAL_RESULTS/"
tar -xzf "$LOCAL_RESULTS/$(basename "$REMOTE_ARCHIVE")" -C "$LOCAL_RESULTS" --strip-components=1
echo "Results in $LOCAL_RESULTS"
