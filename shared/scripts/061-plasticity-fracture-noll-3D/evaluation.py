import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{type1cm}"
})

import os
import numpy as np
import math

import alex.postprocessing as pp
import alex.linearelastic as le
import alex.evaluation as ev
import evaluation_utils as ev_ut

# =====================================================
# BASIC PARAMETERS
# =====================================================
element_size = 0.01
epsilon_param = 0.1
gc_num_factor = 1.12
starting_hole_index = 2

# =====================================================
# LABELS
# =====================================================
steg_width_label = "$w_s$"
xlabel_time = "$t / [ L / {\\dot{x}}_{\\mathrm{bc}} ]$"
xlabel_xct = "$x_{ct} / L$"
ylabel_xct = "crack tip position $/ L$"
ylabel_Jx = "$J_x / J_c^{\\mathrm{num}}$"
ylabel_Jx_max = "$J_x^{\\mathrm{max}} / J_c^{\\mathrm{num}}$"

vm2d_label = r"\textbf{von Mises 2D}"
vm3d_label = r"\textbf{von Mises 3D}"

script_path = os.path.dirname(__file__)

# =====================================================
# DATA DIRECTORIES
# =====================================================
vm2d_results_dir = os.path.join(
    script_path,
    "..",
    "046_plasticity",
    "results",
    "gc_1.0_elastic",
)

vm3d_results_dir = os.path.join(
    script_path,
    ".",
    "results",
)

# =====================================================
# HELPERS
# =====================================================
def make_column_positive(data, col):
    min_val = data[col].min()
    if min_val < 0:
        data[col] = data[col] - min_val

def make_columns_positive(data, cols):
    for col in cols:
        make_column_positive(data, col)

# =====================================================
# LOAD + NORMALIZE SINGLE DATASETS (w_s = 1.0)
# =====================================================
def load_single_vm_dataset(results_dir, is_3d=False):
    sim_folder = ev_ut.find_simulation_by_wsteg(
        results_dir,
        wsteg_value_in=1.0,
    )
    if sim_folder is None:
        raise FileNotFoundError("No dataset found for w_s = 1.0")

    data = pd.read_csv(
        os.path.join(sim_folder, "run_simulation_graphs.txt"),
        delim_whitespace=True,
        header=None,
        skiprows=1,
    )

    ev_ut.normalize_Jx_to_Gc_num(gc_num_factor, data)

    if is_3d:
        # DROP column 3
        data.drop(columns=[3], inplace=True)

        # REINDEX
        data.reset_index(drop=True, inplace=True)
        data.columns = range(data.shape[1])

        # Offset columns 3 and 4
        make_columns_positive(data, [3, 4])
    else:
        # Offset columns 3 and 4
        make_columns_positive(data, [3, 4])

    return data, sim_folder


data_vm2d, vm2d_sim_folder = load_single_vm_dataset(vm2d_results_dir, is_3d=False)
data_vm3d, vm3d_sim_folder = load_single_vm_dataset(vm3d_results_dir, is_3d=True)

# =====================================================
# TIME EVOLUTION PLOT (von Mises 3D)
# =====================================================
output_time_plot = os.path.join(
    script_path,
    "PAPER_VM3D_xct_vs_time.png",
)

ev.plot_columns_multiple_y(
    data=data_vm3d,
    col_x=0,
    col_y_list=[3, 4],
    output_filename=output_time_plot,
    legend_labels=["$x_{ct}$", "$x_{bc}$"],
    usetex=True,
    title=" ",
    plot_dots=False,
    xlabel=xlabel_time,
    ylabel=ylabel_xct,
    x_range=[-0.1, 17],
)

# =====================================================
# READ HOLE GEOMETRY
# =====================================================
parameter_file = os.path.join(vm2d_sim_folder, "parameters.txt")
parameters = pp.read_parameters_file(parameter_file)

nholes = int(parameters.get("nholes", parameters.get("Nholes")))
dhole = parameters["dhole"]
wsteg = parameters["wsteg"]

start_pos, end_pos = ev_ut.ramberg_osgood_positions(
    nholes, dhole, wsteg
)
hole_positions = sorted(start_pos + end_pos)

# =====================================================
# Jx vs x_ct (2D vs 3D)
# =====================================================
output_Jx_vs_xct = os.path.join(
    script_path,
    "PAPER_VM_Jx_vs_xct_2D_vs_3D.png",
)

ev.plot_multiple_columns(
    data_objects=[data_vm2d, data_vm3d],
    col_x=3,
    col_y=1,
    output_filename=output_Jx_vs_xct,
    vlines=[hole_positions, hole_positions],
    legend_labels=[vm2d_label, vm3d_label],
    usetex=True,
    xlabel=xlabel_xct,
    ylabel=ylabel_Jx,
    y_range=[0.0, 2.5],
    x_range=[0, 15],
    markers_only=True,
    marker_size=3,
    use_colors=True,
    legend_outside=False,
)

# =====================================================
# MULTI-SIMULATION Jx vs x_ct (ALL w_s)
# =====================================================
def prepare_vm_multidata(results_dir, is_3d=False):
    sim_results = ev.read_all_simulation_data(results_dir)

    processed = []
    for data, param in sim_results:
        ev_ut.normalize_Jx_to_Gc_num(gc_num_factor, data)

        if is_3d:
            data.drop(columns=[3], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data.columns = range(data.shape[1])
            make_columns_positive(data, [3, 4])
        else:
            make_columns_positive(data, [3, 4])

        processed.append((data, param))

    return ev_ut.data_for_plot_wsteg_in_legend(
        ev_ut.normalize_Jx_to_Gc_num,
        gc_num_factor,
        processed,
        starting_hole_index,
        steg_width_label,
        ev_ut.filter_data_by_column_bounds,
        ev_ut.get_x_range_between_ramberg_osgoods,
        hole_positions,
    )


vm2d_multidata, vm2d_legends = prepare_vm_multidata(vm2d_results_dir, is_3d=False)
vm3d_multidata, vm3d_legends = prepare_vm_multidata(vm3d_results_dir, is_3d=True)

# --- 2D ---
ev.plot_multiple_columns(
    data_objects=vm2d_multidata,
    col_x=3,
    col_y=1,
    output_filename=os.path.join(
        script_path,
        "PAPER_VM2D_Jx_vs_xct_all_wsteg.png",
    ),
    legend_labels=vm2d_legends,
    xlabel=xlabel_xct,
    ylabel=ylabel_Jx,
    usetex=True,
    markers_only=True,
    use_colors=True,
    y_range=[0, 2],
)

# --- 3D ---
ev.plot_multiple_columns(
    data_objects=vm3d_multidata,
    col_x=3,
    col_y=1,
    output_filename=os.path.join(
        script_path,
        "PAPER_VM3D_Jx_vs_xct_all_wsteg.png",
    ),
    legend_labels=vm3d_legends,
    xlabel=xlabel_xct,
    ylabel=ylabel_Jx,
    usetex=True,
    markers_only=True,
    use_colors=True,
    y_range=[0, 2],
)

# =====================================================
# Jx_max vs w_s (2D vs 3D)
# =====================================================
def compute_Jxmax_vs_wsteg(results_dir, is_3d=False):
    sim_results = ev.read_all_simulation_data(results_dir)

    wsteg_vals = []
    Jx_max_vals = []

    for data, param in sim_results:
        ev_ut.normalize_Jx_to_Gc_num(gc_num_factor, data)

        if is_3d:
            data.drop(columns=[3], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data.columns = range(data.shape[1])
            make_columns_positive(data, [3, 4])
        else:
            make_columns_positive(data, [3, 4])

        nh = int(param.get("nholes", param.get("Nholes")))
        dh = param["dhole"]
        ws = param["wsteg"]

        _, x_high = ev_ut.get_x_range_between_ramberg_osgoods(
            nh, dh, ws, starting_hole_index, starting_hole_index + 1
        )

        window = ev_ut.filter_data_by_column_bounds(
            data,
            column_index=3,
            low_bound=x_high - 1.01 * ws,
            upper_bound=x_high + 0.01 * ws,
        )

        Jx_max_vals.append(np.max(window[1]))
        wsteg_vals.append(ws)

    order = np.argsort(wsteg_vals)
    return (
        np.array(wsteg_vals)[order].tolist(),
        np.array(Jx_max_vals)[order].tolist(),
    )


w_vm2d, Jxmax_vm2d = compute_Jxmax_vs_wsteg(vm2d_results_dir, is_3d=False)
w_vm3d, Jxmax_vm3d = compute_Jxmax_vs_wsteg(vm3d_results_dir, is_3d=True)

# =====================================================
# FINAL PLOT
# =====================================================
output_Jxmax_vs_wsteg = os.path.join(
    script_path,
    "PAPER_VM_Jxmax_vs_wsteg_2D_vs_3D.png",
)

ev.plot_multiple_lines(
    x_values=[w_vm2d, w_vm3d],
    y_values=[Jxmax_vm2d, Jxmax_vm3d],
    x_label="$w_s / L$",
    y_label=ylabel_Jx_max,
    legend_labels=[vm2d_label, vm3d_label],
    output_file=output_Jxmax_vs_wsteg,
    usetex=True,
    markers_only=False,
    use_colors=True,
    marker_size=6,
    y_range=[0, 2.5],
    x_range=[0.375, 3.0],
    bold_text=True,
)

print("✅ PAPER plots generated successfully (2D + 3D).")
