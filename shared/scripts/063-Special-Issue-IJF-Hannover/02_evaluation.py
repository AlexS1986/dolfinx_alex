import os
import glob
import re
import argparse
import json
import numpy as np
import alex.evaluation as ev  # plotting utilities


def extract_number(filename):
    match = re.search(r"result_graphs_(\d+)_", filename)
    if not match:
        match = re.search(r"result_graphs_(\d+)_fromfile", filename)
    return int(match.group(1)) if match else float('inf')


def collect_data(files, stride=1, min_index=0):
    data_to_plot = []
    legend_entries = []
    file_indices = []
    max_values = {"Ry": [], "Work": [], "Fracture": [], "Elastic": []}

    for f in sorted(files, key=extract_number)[::stride]:
        idx = extract_number(f)
        if idx < min_index:
            continue

        with open(f, "r") as infile:
            lines = [l for l in infile if not l.startswith("#")]
        if not lines:
            continue

        raw_data = np.loadtxt(lines)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        u_y_abs = np.abs(raw_data[:, 1])
        R_y_abs = np.abs(raw_data[:, 2])
        Work = np.abs(raw_data[:, 4])
        Fracture = np.abs(raw_data[:, 5])
        Elastic = np.abs(raw_data[:, 6])

        # --- REQUIRED INDEXING (KEPT EXACTLY AS REQUESTED) ---
        arr = np.full((raw_data.shape[0], 6), np.nan)
        arr[:, 1] = u_y_abs
        arr[:, 2] = R_y_abs
        arr[:, 3] = Work
        arr[:, 4] = Fracture
        arr[:, 5] = Elastic

        data_to_plot.append(arr.T)
        legend_entries.append(f"{idx}")
        file_indices.append(idx)

        max_values["Ry"].append(np.max(R_y_abs))
        max_values["Work"].append(np.max(Work))
        max_values["Fracture"].append(np.max(Fracture))
        max_values["Elastic"].append(np.max(Elastic))

    return data_to_plot, legend_entries, file_indices, max_values


def collect_volumes(base_folders, types, min_index=0):
    vol_data = {}
    for key, folder in base_folders.items():
        vol_data[key] = {"indices": [], "vols": []}
        for f in glob.glob(os.path.join(folder, "vol_*_" + types[key] + ".json")):
            idx_match = re.search(r"vol_(\d+)_", f)
            if not idx_match:
                continue
            idx = int(idx_match.group(1))
            if idx < min_index:
                continue
            with open(f, "r") as infile:
                js = json.load(infile)
                vol = js.get("vol", None)
            if vol is not None:
                vol_data[key]["indices"].append(idx)
                vol_data[key]["vols"].append(vol)
    return vol_data


def filter_indices_and_values(indices, values, min_index):
    new_idx, new_vals = [], []
    for i, v in zip(indices, values):
        if i >= min_index:
            new_idx.append(i)
            new_vals.append(v)
    return new_idx, new_vals


def main():
    parser = argparse.ArgumentParser(
        description="Plot energy-related quantities, maxima, and volumes vs index."
    )

    parser.add_argument(
        "--base_folder",
        default="/home/scripts/063-Special-Issue-IJF-Hannover/resources/",
        help="Base folder containing min/max/vary subfolders."
    )

    parser.add_argument(
        "--ext",
        default="volumetric",
        help="Optional filename extension for output plots."
    )

    parser.add_argument(
        "--min_index",
        type=int,
        default=5,
        help="Exclude all data below this index."
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        help="Folder where plots will be written."
    )

    args = parser.parse_args()

    base_folder = args.base_folder
    script_path = os.path.dirname(os.path.abspath(__file__))

    if args.output_folder is not None:
        output_folder = os.path.abspath(args.output_folder)
    else:
        output_folder = script_path

    os.makedirs(output_folder, exist_ok=True)

    folders = {
        "vary": os.path.join(base_folder, "dcb_var_bcpos_E_var", "export"),
        "min": os.path.join(base_folder, "dcb_var_bcpos_E_min", "export"),
        "max": os.path.join(base_folder, "dcb_var_bcpos_E_max", "export"),
        "avg": os.path.join(base_folder, "dcb_var_bcpos_E_var", "export"),
    }

    patterns = {
        "vary": "result_graphs_*_vary.txt",
        "min": "result_graphs_*_min.txt",
        "max": "result_graphs_*_max.txt",
        "avg": "result_graphs_*_fromfile.txt"
    }

    types = {
        "vary": "vary",
        "min": "min",
        "max": "max",
        "avg": "fromfile"
    }

    all_indices = {}
    all_max = {"Ry": {}, "Work": {}, "Fracture": {}, "Elastic": {}}

    xlabel = "$u_y$ / mm"
    stride_for_curves = 2
    stride_for_max = 1

    # ===============================
    # Curve plots (vs u_y)
    # ===============================
    for key in ["vary", "min", "max", "avg"]:
        folder = folders[key]
        pattern = patterns[key]

        if not os.path.isdir(folder):
            continue

        files = glob.glob(os.path.join(folder, pattern))
        if not files:
            continue

        data_to_plot, legend_entries, file_indices, _ = collect_data(
            files, stride=stride_for_curves, min_index=args.min_index
        )

        if not data_to_plot:
            continue

        if key == "avg":
            legend_entries = [f"$a={i}$" for i in file_indices]
        else:
            legend_entries = [f"$a={i}$" for i in file_indices]

        _, _, file_indices_all, max_values_all = collect_data(
            files, stride=stride_for_max, min_index=args.min_index
        )

        all_indices[key] = file_indices_all
        for qty in ["Ry", "Work", "Fracture", "Elastic"]:
            all_max[qty][key] = max_values_all[qty]

        # --- Ry vs uy ---
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=2,
            output_filename=os.path.join(output_folder, f"Ry_vs_uy_{key}{args.ext}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel,
            ylabel="$R_y$ / (N/mm)",
            usetex=True, use_colors=True,
            legend_outside=True,
            figsize=(15, 7),
            vary_linestyles=True,
            mark_peak=True,
            annotate_peak=True,
            x_range=[0,0.05]
        )

        # --- Work ---
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=3,
            output_filename=os.path.join(output_folder, f"Work_vs_uy_{key}{args.ext}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel,
            ylabel="Work $G_c$ / mm",
            usetex=True, use_colors=True,
            legend_outside=True,
            figsize=(15, 7),
            vary_linestyles=True,
            mark_peak=True,
            annotate_peak=True,
            x_range=[0,0.05]
        )

        # --- Fracture ---
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=4,
            output_filename=os.path.join(output_folder, f"FractureEnergy_vs_uy_{key}{args.ext}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel,
            ylabel="Fracture Energy $G_c$ / mm",
            usetex=True, use_colors=True,
            legend_outside=True,
            figsize=(15, 7),
            vary_linestyles=True,
            mark_peak=True,
            annotate_peak=True,
            x_range=[0,0.05]
        )

        # --- Elastic ---
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=5,
            output_filename=os.path.join(output_folder, f"ElasticEnergy_vs_uy_{key}{args.ext}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel,
            ylabel="Elastic Energy / mm",
            usetex=True, use_colors=True,
            legend_outside=True,
            figsize=(15, 7),
            vary_linestyles=True,
            mark_peak=True,
            annotate_peak=True,
            x_range=[0,0.05]
        )

    # ===============================
    # MAX PLOTS
    # ===============================

    def make_max_plot(quantity, title, ylabel, filename):
        x_vals, y_vals, labels = [], [], []
        for key in all_indices:
            indices, vals = filter_indices_and_values(
                all_indices[key], all_max[quantity][key], args.min_index
            )
            if indices:
                x_vals.append(indices)
                y_vals.append(vals)
                labels.append(f"Max {quantity} ({key})")

        if x_vals:
            ev.plot_multiple_lines(
                x_values=x_vals,
                y_values=y_vals,
                title=title,
                x_label="$a$",
                y_label=ylabel,
                legend_labels=labels,
                output_file=os.path.join(output_folder, filename),
                figsize=(12, 8),
                usetex=True,
                show_markers=True,
                use_colors=True,
                bold_text=True
            )

    make_max_plot("Ry", "",
                  "$R_y$ / (N/mm)",
                  f"max_Ry_vs_index{args.ext}.png")

    make_max_plot("Work", "",
                  "Work $G_c$ / mm",
                  f"max_Work_vs_index{args.ext}.png")

    make_max_plot("Fracture", "",
                  "Fracture Energy $G_c$ / mm",
                  f"max_FractureEnergy_vs_index{args.ext}.png")

    make_max_plot("Elastic", "",
                  "Elastic Energy / mm",
                  f"max_ElasticEnergy_vs_index{args.ext}.png")

    # ===============================
    # Volume plot
    # ===============================
    vol_data = collect_volumes(folders, types, min_index=args.min_index)

    x_vals, y_vals, labels = [], [], []
    for key, dct in vol_data.items():
        indices, vals = filter_indices_and_values(
            dct["indices"], dct["vols"], args.min_index
        )
        if indices:
            x_vals.append(indices)
            y_vals.append(vals)
            labels.append(f"Volume ({key})")

    if x_vals:
        ev.plot_multiple_lines(
            x_values=x_vals,
            y_values=y_vals,
            title="Volumes vs $a$",
            x_label="$a$",
            y_label="Volume",
            legend_labels=labels,
            output_file=os.path.join(output_folder, f"volumes_vs_index{args.ext}.png"),
            figsize=(12, 8),
            usetex=True,
            show_markers=True,
            use_colors=True,
            bold_text=True,
            markers_only=True
        )

    print(f"\nAll plots written to: {output_folder}")


if __name__ == "__main__":
    main()
