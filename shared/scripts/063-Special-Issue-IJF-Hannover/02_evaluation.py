import os
import glob
import re
import argparse
import json
import numpy as np
import alex.evaluation as ev  # plotting utilities


SIMULATION_FOLDER_RE = re.compile(
    r"^simulation_.*_SPLIT(?P<split>spectral|volumetric)_EPS(?P<epsilon>[0-9_]+)$"
)

RESULT_FILE_RE = re.compile(
    r"^result_graphs_beta_(?P<beta>0_\d+)_a_(?P<a>\d+)_rho_(?P<rho>0_\d+)"
    r"_(?P<structure>min|max|var)_(?P<case>min|max|vary)_(?P<split>spectral|volumetric)"
    r"(?:_eps(?P<epsilon>[0-9_]+))?\.txt$"
)

VOLUME_FILE_RE = re.compile(
    r"^vol_beta_(?P<beta>0_\d+)_a_(?P<a>\d+)_rho_(?P<rho>0_\d+)"
    r"_(?P<structure>min|max|var)_(?P<case>min|max|vary)_(?P<split>spectral|volumetric)"
    r"(?:_eps(?P<epsilon>[0-9_]+))?\.json$"
)


def parse_float(token):
    return float(token.replace("_", "."))


def parse_simulation_folder(path):
    match = SIMULATION_FOLDER_RE.match(os.path.basename(os.path.normpath(path)))
    if not match:
        return None
    return {
        "split": match.group("split"),
        "epsilon": parse_float(match.group("epsilon")),
    }


def parse_result_filename(filename):
    name = os.path.basename(filename)
    match = RESULT_FILE_RE.match(name)
    if match:
        return {
            "a": int(match.group("a")),
            "beta": parse_float(match.group("beta")),
            "rho": parse_float(match.group("rho")),
            "case": match.group("case"),
            "split": match.group("split"),
            "epsilon": parse_float(match.group("epsilon")) if match.group("epsilon") else None,
        }

    old_match = re.search(r"result_graphs_(\d+)_(?P<case>min|max|vary|fromfile)", name)
    if old_match:
        return {
            "a": int(old_match.group(1)),
            "beta": None,
            "rho": None,
            "case": old_match.group("case"),
            "split": None,
            "epsilon": None,
        }
    return None


def parse_volume_filename(filename):
    name = os.path.basename(filename)
    match = VOLUME_FILE_RE.match(name)
    if match:
        return {
            "a": int(match.group("a")),
            "beta": parse_float(match.group("beta")),
            "rho": parse_float(match.group("rho")),
            "case": match.group("case"),
            "split": match.group("split"),
            "epsilon": parse_float(match.group("epsilon")) if match.group("epsilon") else None,
        }

    old_match = re.search(r"vol_(\d+)_(?P<case>min|max|vary|fromfile)\.json", name)
    if old_match:
        return {
            "a": int(old_match.group(1)),
            "beta": None,
            "rho": None,
            "case": old_match.group("case"),
            "split": None,
            "epsilon": None,
        }
    return None


def result_sort_key(filename):
    parsed = parse_result_filename(filename)
    if parsed is None:
        return (float("inf"), float("inf"), float("inf"), filename)
    return (
        parsed["a"],
        -1.0 if parsed["beta"] is None else parsed["beta"],
        -1.0 if parsed["rho"] is None else parsed["rho"],
        filename,
    )


def extract_number(filename):
    parsed = parse_result_filename(filename)
    return parsed["a"] if parsed else float('inf')


def label_for_file(filename):
    parsed = parse_result_filename(filename)
    if parsed is None:
        return os.path.basename(filename)
    if parsed["beta"] is None or parsed["rho"] is None:
        return f"$a={parsed['a']}$"
    return (
        f"$a={parsed['a']}$, "
        f"$\\beta={parsed['beta']:g}$, "
        f"$\\rho={parsed['rho']:g}$"
    )


def eps_suffix(epsilon):
    if epsilon is None:
        return ""
    return f"_eps{epsilon:g}".replace(".", "_")


def discover_contexts(base_folder):
    base_folder = os.path.abspath(base_folder)
    direct_meta = parse_simulation_folder(base_folder)

    if direct_meta is not None:
        return [make_context(base_folder, direct_meta)]

    simulation_folders = [
        os.path.join(base_folder, entry)
        for entry in sorted(os.listdir(base_folder)) if os.path.isdir(os.path.join(base_folder, entry))
        and parse_simulation_folder(os.path.join(base_folder, entry)) is not None
    ] if os.path.isdir(base_folder) else []

    if simulation_folders:
        return [
            make_context(folder, parse_simulation_folder(folder))
            for folder in simulation_folders
        ]

    return [{
        "name": "legacy",
        "root": base_folder,
        "data_root": base_folder,
        "split": None,
        "epsilon": None,
        "suffix": "",
    }]


def make_context(folder, meta):
    resource_root = os.path.join(folder, "resources", "260504_dcb_beta_phi_a_rho_var_min_max")
    data_root = resource_root if os.path.isdir(resource_root) else folder
    split = meta["split"]
    epsilon = meta["epsilon"]
    return {
        "name": os.path.basename(os.path.normpath(folder)),
        "root": folder,
        "data_root": data_root,
        "split": split,
        "epsilon": epsilon,
        "suffix": f"_{split}{eps_suffix(epsilon)}",
    }


def collect_files_by_case(context, case):
    files = glob.glob(os.path.join(context["data_root"], "**", "result_graphs_*.txt"), recursive=True)
    selected = []
    for filename in files:
        parsed = parse_result_filename(filename)
        if parsed is None or parsed["case"] != case:
            continue
        if context["split"] is not None and parsed["split"] not in (None, context["split"]):
            continue
        if context["epsilon"] is not None and parsed["epsilon"] is not None:
            if not np.isclose(parsed["epsilon"], context["epsilon"]):
                continue
        selected.append(filename)
    return selected


def collect_data(files, stride=1, min_index=0):
    data_to_plot = []
    legend_entries = []
    file_indices = []
    max_values = {"Ry": [], "Work": [], "Fracture": [], "Elastic": []}

    for f in sorted(files, key=result_sort_key)[::stride]:
        idx = extract_number(f)
        if idx < min_index:
            continue

        with open(f, "r", encoding="utf-8") as infile:
            lines = [l for l in infile if l.strip() and not l.startswith("#")]
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
        legend_entries.append(label_for_file(f))
        file_indices.append(idx)

        max_values["Ry"].append(np.max(R_y_abs))
        max_values["Work"].append(np.max(Work))
        max_values["Fracture"].append(np.max(Fracture))
        max_values["Elastic"].append(np.max(Elastic))

    return data_to_plot, legend_entries, file_indices, max_values


def collect_volumes(context, cases, min_index=0):
    vol_data = {}
    for key in cases:
        vol_data[key] = {"indices": [], "vols": []}
        volume_files = glob.glob(os.path.join(context["data_root"], "**", "vol_*.json"), recursive=True)
        for f in sorted(volume_files):
            parsed = parse_volume_filename(f)
            if parsed is None or parsed["case"] != key:
                continue
            if context["split"] is not None and parsed["split"] not in (None, context["split"]):
                continue
            if context["epsilon"] is not None and parsed["epsilon"] is not None:
                if not np.isclose(parsed["epsilon"], context["epsilon"]):
                    continue
            idx = parsed["a"]
            if idx < min_index:
                continue
            with open(f, "r", encoding="utf-8") as infile:
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
        default=None,
        help="Results root, one simulation_* folder, or a legacy resources folder."
    )

    parser.add_argument(
        "--ext",
        default="",
        help="Optional extra suffix for output plot filenames."
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
        help="Folder where plots will be written. Defaults to ./plots next to this script."
    )

    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(__file__))
    base_folder = args.base_folder
    if base_folder is None:
        base_folder = os.path.join(script_path, "results")

    if args.output_folder is not None:
        output_folder = os.path.abspath(args.output_folder)
    else:
        output_folder = os.path.join(script_path, "plots")

    os.makedirs(output_folder, exist_ok=True)

    xlabel = "$u_y$ / mm"
    stride_for_curves = 2
    stride_for_max = 1
    cases = ["vary", "min", "max"]
    contexts = discover_contexts(base_folder)
    plotted_anything = False

    # ===============================
    # Curve plots (vs u_y)
    # ===============================
    for context in contexts:
        all_indices = {}
        all_max = {"Ry": {}, "Work": {}, "Fracture": {}, "Elastic": {}}
        suffix = context["suffix"] + args.ext

        print(f"Scanning {context['name']} below {context['data_root']}")

        for key in cases:
            files = collect_files_by_case(context, key)
            if not files:
                continue

            data_to_plot, legend_entries, file_indices, _ = collect_data(
                files, stride=stride_for_curves, min_index=args.min_index
            )

            if not data_to_plot:
                continue

            _, _, file_indices_all, max_values_all = collect_data(
                files, stride=stride_for_max, min_index=args.min_index
            )

            all_indices[key] = file_indices_all
            for qty in ["Ry", "Work", "Fracture", "Elastic"]:
                all_max[qty][key] = max_values_all[qty]

            plotted_anything = True

            # --- Ry vs uy ---
            ev.plot_multiple_columns(
                data_objects=data_to_plot,
                col_x=1, col_y=2,
                output_filename=os.path.join(output_folder, f"Ry_vs_uy_{key}{suffix}.png"),
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
                output_filename=os.path.join(output_folder, f"Work_vs_uy_{key}{suffix}.png"),
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
                output_filename=os.path.join(output_folder, f"FractureEnergy_vs_uy_{key}{suffix}.png"),
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
                output_filename=os.path.join(output_folder, f"ElasticEnergy_vs_uy_{key}{suffix}.png"),
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
                      f"max_Ry_vs_index{suffix}.png")

        make_max_plot("Work", "",
                      "Work $G_c$ / mm",
                      f"max_Work_vs_index{suffix}.png")

        make_max_plot("Fracture", "",
                      "Fracture Energy $G_c$ / mm",
                      f"max_FractureEnergy_vs_index{suffix}.png")

        make_max_plot("Elastic", "",
                      "Elastic Energy / mm",
                      f"max_ElasticEnergy_vs_index{suffix}.png")

        # ===============================
        # Volume plot
        # ===============================
        vol_data = collect_volumes(context, cases, min_index=args.min_index)

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
                output_file=os.path.join(output_folder, f"volumes_vs_index{suffix}.png"),
                figsize=(12, 8),
                usetex=True,
                show_markers=True,
                use_colors=True,
                bold_text=True,
                markers_only=True
            )

    if not plotted_anything:
        raise SystemExit(f"No matching result files found below {base_folder}")

    print(f"\nAll plots written to: {output_folder}")


if __name__ == "__main__":
    main()
