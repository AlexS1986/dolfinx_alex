#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-/work/scratch/as12vapa/067_plasticity_inclusions}"
TARGET_DIR="${TARGET_DIR:-/work/scratch/as12vapa/067_plasticity_inclusions_lightweight}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${TARGET_DIR}.tar.gz}"

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

cd "$SOURCE_DIR"

echo "Copying folder structure, parameters.txt, and run_simulation_graphs.txt"
find . -type d -exec mkdir -p "$TARGET_DIR/{}" \;
find . -type f \( -name parameters.txt -o -name run_simulation_graphs.txt \) \
    -exec cp --parents {} "$TARGET_DIR" \;

first_wsteg_dir=""
while IFS= read -r parameters_file; do
    if grep -Eiq 'wsteg[^0-9+-]*1([.]0+)?([^0-9.]|$)' "$parameters_file"; then
        first_wsteg_dir="$(dirname "$parameters_file")"
        break
    fi
done < <(find . -type f -name parameters.txt | sort)

if [[ -z "$first_wsteg_dir" ]]; then
    echo "No parameters.txt with wsteg = 1.0 was found." >&2
    exit 1
fi

echo "First wsteg = 1.0 folder:"
echo "  $first_wsteg_dir"

mkdir -p "$TARGET_DIR/$first_wsteg_dir"
for file_name in run_simulation.xdmf run_simulation.h5; do
    if [[ -f "$first_wsteg_dir/$file_name" ]]; then
        cp "$first_wsteg_dir/$file_name" "$TARGET_DIR/$first_wsteg_dir/"
    else
        echo "Warning: missing $first_wsteg_dir/$file_name" >&2
    fi
done

echo "Creating archive:"
echo "  $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$(dirname "$TARGET_DIR")" "$(basename "$TARGET_DIR")"

echo "Done."
echo "Lightweight folder:"
echo "  $TARGET_DIR"
echo "Archive for download:"
echo "  $ARCHIVE_PATH"
