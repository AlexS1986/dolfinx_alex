import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{type1cm}"
})

from matplotlib.ticker import MaxNLocator, LogLocator
import matplotlib.colors as mcolors
import os
import numpy as np
import math
import alex.postprocessing as pp
import alex.homogenization as hom
import alex.linearelastic as le
import alex.evaluation as ev
from scipy.signal import savgol_filter
import evaluation_utils as ev_ut

# -------------------------------
# BASIC PARAMETERS
# -------------------------------
element_size = 0.01
epsilon_param = 0.1
gc_num_quotient = 1.12
starting_hole_to_evaluate = 2

steg_width_label = "$w_s$"
J_x_label = "$J_{x} / J_c^{\\mathrm{num}}$"
J_x_max_label = "$J_{x}^{\\mathrm{max}} / J_c^{\\mathrm{num}}$"
crack_tip_position_label = "$x_{\\mathrm{ct}}$"
t_label = "$t / [ L / {\\dot{x}}_{\\mathrm{bc}} ]$"

script_path = os.path.dirname(__file__)

# -------------------------------
# DIRECTORIES
# -------------------------------
data_directory_sig = os.path.join(
    script_path, "..", "045_standard_holes_PAPER", "results",
    "interpolation_as_in_plasticity_gc_0.5679",
)

data_directory_Jc = os.path.join(
    script_path, "..", "045_standard_holes_PAPER", "results", "gc_1.0",
)

data_directory_vonmises = os.path.join(
    script_path, "..", "046_plasticity", "results", "gc_1.0_elastic",
)

# -------------------------------
# LOAD + NORMALIZE DATA
# -------------------------------
def load_and_normalize(directory):
    folder = ev_ut.find_simulation_by_wsteg(directory, wsteg_value_in=1.0)
    if folder is None:
        raise FileNotFoundError(f"No dataset found in {directory}")
    data = pd.read_csv(
        os.path.join(folder, "run_simulation_graphs.txt"),
        delim_whitespace=True,
        header=None,
        skiprows=1,
    )
    ev_ut.normalize_Jx_to_Gc_num(gc_num_quotient, data)
    return data, folder


data_sig, simulation_data_folder_sig = load_and_normalize(data_directory_sig)
data_Jc, simulation_data_folder_Jc = load_and_normalize(data_directory_Jc)
data_vonmises, simulation_data_folder_vonmises = load_and_normalize(data_directory_vonmises)

# -------------------------------
# von Mises crack-tip evolution
# -------------------------------
output_file = os.path.join(
    script_path, "PAPER_00_xct_pf_vs_xct_KI_vonMises.png",
)
ev.plot_columns_multiple_y(
    data=data_vonmises,
    col_x=0,
    col_y_list=[3, 4],
    output_filename=output_file,
    legend_labels=[crack_tip_position_label, "$x_{\\mathrm{bc}}$"],
    usetex=True,
    title=" ",
    plot_dots=False,
    xlabel=t_label,
    ylabel="crack tip position $/ L$",
    x_range=[-0.1, 17],
)

# -------------------------------
# READ HOLE POSITIONS
# -------------------------------
parameter_path = os.path.join(simulation_data_folder_Jc, "parameters.txt")
parameters = pp.read_parameters_file(parameter_path)

start_positions, end_positions = ev_ut.ramberg_osgood_positions(
    int(parameters["nholes"]),
    parameters["dhole"],
    parameters["wsteg"],
)
hole_positions = sorted(start_positions + end_positions)

# -------------------------------
# LABELS
# -------------------------------
sig_label = r"\textbf{Eq}$\mathbf{\sigma^*}$"
elastic_label_Jc = r"\textbf{Eq}$\mathbf{J_c}$"
vonmises_label = r"\textbf{vM}"

xlabel = "$x_{ct} / L$"
ylabel = "$J_{x} / J_c^{\\mathrm{num}}$"

# =====================================================
# MAIN COMBINED PLOT (ORDER: vM → Jc → Sigma*)
# =====================================================
output_file = os.path.join(
    script_path, "PAPER_clean_Jx_vs_xct_SIG_Jc_vonMises.png",
)
ev.plot_multiple_columns(
    data_objects=[data_vonmises, data_Jc, data_sig],
    col_x=3,
    col_y=1,
    output_filename=output_file,
    vlines=[hole_positions, hole_positions],
    legend_labels=[vonmises_label, elastic_label_Jc, sig_label],
    usetex=True,
    xlabel=xlabel,
    ylabel=ylabel,
    y_range=[0.0, 2.5],
    x_range=[0, 15],
    markers_only=True,
    marker_size=3,
    use_colors=True,
    legend_outside=True,
)

# =====================================================
# PAPER_01 SINGLE-DATASET PLOTS
# =====================================================
ev.plot_columns(
    data_sig, 3, 1,
    os.path.join(script_path, "PAPER_01_Jx_vs_xct_SIG.png"),
    vlines=hole_positions, xlabel=xlabel, ylabel=J_x_label, usetex=True,
)

ev.plot_columns(
    data_vonmises, 3, 1,
    os.path.join(script_path, "PAPER_01_Jx_vs_xct_vonMises.png"),
    vlines=hole_positions, xlabel=xlabel, ylabel=J_x_label, usetex=True,
)

ev.plot_columns(
    data_Jc, 3, 1,
    os.path.join(script_path, "PAPER_01_Jx_vs_xct_Jc.png"),
    vlines=hole_positions, xlabel=xlabel, ylabel=J_x_label, usetex=True,
)

# =====================================================
# PAPER_06b-TYPE PLOT (Jx_max vs w_steg)
# ORDER: vM → Jc → Sigma*
# =====================================================

def data_vs_wsteg_small(
    normalize_Jx_to_Gc_num,
    gc_num_quotient,
    data_directory,
    starting_hole,
    filter_data_by_column_bounds,
    get_x_range_between_ramberg_osgoods,
):
    simulation_results = ev.read_all_simulation_data(data_directory)

    Jx_max_values = []
    wsteg_values = []

    for sim in simulation_results:
        data = sim[0]
        normalize_Jx_to_Gc_num(gc_num_quotient, data)

        param = sim[1]
        nhole_loc = int(param["nholes"])
        dhole_loc = param["dhole"]
        wsteg_loc = param["wsteg"]
        wsteg_values.append(wsteg_loc)

        x_low, x_high = get_x_range_between_ramberg_osgoods(
            nhole_loc, dhole_loc, wsteg_loc, starting_hole, starting_hole + 1
        )

        data_in_range = filter_data_by_column_bounds(
            data, 3,
            low_bound=x_high - 1.01 * wsteg_loc,
            upper_bound=x_high + 0.01 * wsteg_loc,
        )

        Jx_max_values.append(np.max(data_in_range[1]))

    sorted_idx = np.argsort(wsteg_values)
    return (
        list(np.array(Jx_max_values)[sorted_idx]),
        list(np.array(wsteg_values)[sorted_idx]),
    )


Jx_max_master = []
w_steg_master = []

for directory in [data_directory_vonmises, data_directory_Jc, data_directory_sig]:
    Jx_sorted, wsteg_sorted = data_vs_wsteg_small(
        ev_ut.normalize_Jx_to_Gc_num,
        gc_num_quotient,
        directory,
        starting_hole_to_evaluate,
        ev_ut.filter_data_by_column_bounds,
        ev_ut.get_x_range_between_ramberg_osgoods,
    )
    Jx_max_master.append(Jx_sorted)
    w_steg_master.append(wsteg_sorted)

output_file = os.path.join(
    script_path, "PAPER_06b_Jx_vs_wsteg_three_datasets.png",
)
ev.plot_multiple_lines(
    x_values=w_steg_master,
    y_values=Jx_max_master,
    x_label="$w_s / L$",
    y_label=J_x_max_label,
    legend_labels=[vonmises_label, elastic_label_Jc, sig_label],
    output_file=output_file,
    usetex=True,
    markers_only=False,
    use_colors=True,
    marker_size=6,
    y_range=[0, 2.5],
    x_range=[0.375, 3.0],
    bold_text=True,
)

print("All plots generated successfully with ordering: vM → Eq Jc → Eq σ*.")
