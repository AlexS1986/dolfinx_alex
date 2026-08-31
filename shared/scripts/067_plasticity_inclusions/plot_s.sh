#!/bin/bash
#SBATCH -J screenshots_s_field
#SBATCH -A p0023647
#SBATCH -t 720
#SBATCH --mem-per-cpu=4000
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e /work/scratch/as12vapa/067_plasticity_inclusions_03_06_2026/%x.err.%j
#SBATCH -o /work/scratch/as12vapa/067_plasticity_inclusions_03_06_2026/%x.out.%j
#SBATCH --mail-type=END,FAIL
#SBATCH -C i01

set -euo pipefail

working_directory="${HPC_SCRATCH}/067_plasticity_inclusions_03_06_2026"
container="$HOME/dolfinx_alex/alex-dolfinx.sif"
bindpath="$HOME/dolfinx_alex/shared:/home,$working_directory:/work"

echo "========================================="
echo "Job started at $(date)"
echo "Working directory: $working_directory"
echo "Container: $container"
echo "========================================="

if [ ! -d "$working_directory" ]; then
    echo "Missing working directory: $working_directory"
    exit 1
fi

cd "$working_directory"

cat > "$working_directory/create_s_screenshots.py" <<'PY'
import faulthandler
from pathlib import Path
import re
import sys

faulthandler.enable()

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import numpy as np

def safe_time_string(value):
    text = f"{float(value):.12g}"
    text = text.replace("-", "m").replace(".", "p")
    text = re.sub(r"[^0-9a-zA-Z_p]+", "_", text)
    return f"t_{text}"

def time_value(key):
    return float(key.replace("_", ".", 1))

def read_parameters(path):
    parameters = {}
    if not path.exists():
        return parameters
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value
    return parameters

def add_inclusions(ax, parameters):
    wsteg = float(parameters.get("wsteg", 1.0))
    diameter = float(parameters.get("dinclusion", parameters.get("dhole", 1.0)))
    nholes = int(float(parameters.get("nholes", 6)))
    cell_width = wsteg + diameter
    x0 = 2.0
    y0 = -1.5 * cell_width

    for row in range(3):
        for column in range(nholes):
            center_x = x0 + (column + 0.5) * cell_width
            center_y = y0 + (row + 0.5) * cell_width
            ax.add_patch(
                Circle(
                    (center_x, center_y),
                    diameter / 2.0,
                    facecolor="#5B8DB8",
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=1,
                )
            )

root = Path("/work")
h5_files = sorted(root.rglob("run_simulation.h5"))

if not h5_files:
    print("No run_simulation.h5 files found below /work", flush=True)
    sys.exit(1)

print(f"Found {len(h5_files)} run_simulation.h5 files", flush=True)

written = 0
skipped = 0
errors = 0

for h5_path in h5_files:
    print(f"\nProcessing: {h5_path}", flush=True)

    try:
        with h5py.File(h5_path, "r") as h5:
            required = ("/Mesh/Grid/geometry", "/Mesh/Grid/topology")
            missing = [name for name in required if name not in h5]
            if missing:
                raise KeyError(f"missing datasets: {missing}")
            if "/Function/s" not in h5:
                print("  WARNING: field 's' not found, skipping", flush=True)
                skipped += 1
                continue

            points = np.asarray(h5["/Mesh/Grid/geometry"][:])
            triangles = np.asarray(h5["/Mesh/Grid/topology"][:], dtype=np.int64)
            keys = sorted(h5["/Function/s"].keys(), key=time_value)
            if not keys:
                print("  WARNING: field 's' has no time steps, skipping", flush=True)
                skipped += 1
                continue

            final_key = keys[-1]
            final_time = time_value(final_key)
            values = np.asarray(h5[f"/Function/s/{final_key}"][:]).reshape(-1)

        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError(f"unexpected geometry shape {points.shape}")
        if triangles.ndim != 2 or triangles.shape[1] < 3:
            raise ValueError(f"unexpected topology shape {triangles.shape}")

        triangles = triangles[:, :3]
        triangulation = mtri.Triangulation(
            points[:, 0],
            points[:, 1],
            triangles,
        )

        if len(values) == len(points):
            shading = "gouraud"
        elif len(values) == len(triangles):
            shading = "flat"
        else:
            raise ValueError(
                f"field 's' has {len(values)} values for "
                f"{len(points)} points and {len(triangles)} cells"
            )

        output_name = f"s_field_screenshot_{safe_time_string(final_time)}.png"
        output_path = h5_path.with_name(output_name)

        fig, ax = plt.subplots(figsize=(12.0, 8.0))
        parameters = read_parameters(h5_path.with_name("parameters.txt"))
        #add_inclusions(ax, parameters)
        image = ax.tripcolor(
            triangulation,
            values,
            shading=shading,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            alpha=0.68,
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
            zorder=2,
        )
        ax.triplot(
            triangulation,
            color="#707070",
            linewidth=0.12,
            alpha=0.45,
            zorder=3,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.01)
        colorbar.set_label(r"$s$", fontsize=20)
        colorbar.ax.tick_params(labelsize=16)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(
            f"  Final time: {final_time}; points: {len(points)}; "
            f"cells: {len(triangles)}; values: {len(values)}",
            flush=True,
        )
        print(f"  Wrote: {output_path}", flush=True)
        written += 1

    except Exception as e:
        plt.close("all")
        print(f"  ERROR while processing {h5_path}: {e}", file=sys.stderr, flush=True)
        errors += 1

print(
    f"\nDone. Wrote {written}, skipped {skipped}, errors {errors}.",
    flush=True,
)
if errors or written == 0:
    sys.exit(1)
PY

srun -n 1 apptainer exec \
    --bind "$bindpath" \
    "$container" \
    python3 -u /work/create_s_screenshots.py \
    </dev/null

echo "========================================="
echo "Job finished at $(date)"
echo "========================================="
