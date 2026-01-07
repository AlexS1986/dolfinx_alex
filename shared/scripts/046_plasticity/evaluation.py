import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{type1cm}"
})

from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator

import alex.postprocessing as pp
import alex.homogenization as hom
import alex.linearelastic as le
import math
import alex.evaluation as ev
from scipy.signal import savgol_filter

import evaluation_utils as ev_ut


# -------------------------------
#       BASIC PARAMETERS
# -------------------------------
element_size = 0.01
epsilon_param = 0.1
gc_num_quotient = 1.12 # from analysis without crack


starting_hole_to_evaluate = 2
steg_width_label = "$w_s$"
J_x_label = "$J_{x} / J_c^{\mathrm{num}}$"

script_path = os.path.dirname(__file__)


# -------------------------------
#    DIRECTORIES (cleaned)
# -------------------------------

# 1. Ramberg–Osgood
data_directory_ramberg_osgoods = os.path.join(script_path,'..','044_ramberg_osgood_holes_PAPER','results','K_constant')

# 2. Linear elastic Jc = 1.0
data_directory_Jc = os.path.join(script_path,'..','045_standard_holes_PAPER','results','gc_1.0')

# 3. von Mises plasticity (user fills this)
data_directory_vonmises = os.path.join(script_path,'.','results','January2026')


# -------------------------------
#      LOAD + NORMALIZE RO
# -------------------------------
simulation_data_folder_ro = ev_ut.find_simulation_by_wsteg(
    data_directory_ramberg_osgoods, wsteg_value_in=1.0
)

data_ro = pd.read_csv(
    os.path.join(simulation_data_folder_ro, 'run_simulation_graphs.txt'),
    delim_whitespace=True, header=None, skiprows=1
)

# smooth RO
smoothed_col1 = savgol_filter(data_ro[1], window_length=200, polyorder=5)
data_ro[1] = smoothed_col1
data_ro = data_ro.iloc[::5].reset_index(drop=True)

ev_ut.normalize_Jx_to_Gc_num(gc_num_quotient, data_ro)


# -------------------------------
#      LOAD + NORMALIZE Jc
# -------------------------------
simulation_data_folder_Jc = ev_ut.find_simulation_by_wsteg(
    data_directory_Jc, wsteg_value_in=1.0
)

data_Jc = pd.read_csv(
    os.path.join(simulation_data_folder_Jc, 'run_simulation_graphs.txt'),
    delim_whitespace=True, header=None, skiprows=1
)

ev_ut.normalize_Jx_to_Gc_num(gc_num_quotient, data_Jc)


# -------------------------------
#   LOAD + NORMALIZE von Mises
# -------------------------------
simulation_data_folder_vonmises = ev_ut.find_simulation_by_wsteg(
    data_directory_vonmises, wsteg_value_in=1.0
)

if simulation_data_folder_vonmises is None:
    raise FileNotFoundError("No matching von Mises dataset found for wsteg=1.0.")

data_vonmises = pd.read_csv(
    os.path.join(simulation_data_folder_vonmises,'run_simulation_graphs.txt'),
    delim_whitespace=True, header=None, skiprows=1
)

ev_ut.normalize_Jx_to_Gc_num(gc_num_quotient, data_vonmises)


# -------------------------------
#   READ PARAMETERS FOR HOLES
# -------------------------------
parameter_path = os.path.join(simulation_data_folder_Jc, "parameters.txt")
parameters = pp.read_parameters_file(parameter_path)
nhole = int(parameters["nholes"])
dhole = parameters["dhole"]
wsteg = parameters["wsteg"]

start_positions, end_positions = ev_ut.ramberg_osgood_positions(nhole, dhole, wsteg)
hole_positions = start_positions + end_positions
hole_positions.sort()


# -------------------------------
#        LABELS
# -------------------------------
ramberg_osgood_label = r"Ramberg--Osgood"
elastic_label_Jc = r"Jc = 1.0"
vonmises_label = r"von Mises"

xlabel = "$x_{ct} / L$"
ylabel = "$J_{x} / J_c^{\\mathrm{num}}$"


# -------------------------------
#    MAIN COMBINED PLOT (clean)
# -------------------------------

output_file = os.path.join(
    script_path,
    'PAPER_clean_Jx_vs_xct_RO_Jc_vonMises.png'
)

ev.plot_multiple_columns(
    [data_ro, data_Jc, data_vonmises],
    3, 1,
    output_file,
    vlines=[hole_positions, hole_positions],
    legend_labels=[ramberg_osgood_label, elastic_label_Jc, vonmises_label],
    usetex=True,
    xlabel=xlabel,
    ylabel=ylabel,
    y_range=[0.0, 2.5],
    x_range=[0, 15],
    markers_only=True,
    marker_size=3,
    use_colors=True,
    legend_outside=True
)


output_file = os.path.join(script_path, 'PAPER_04_Jx_vs_xct_all_ramberg_osgoods.png')  
simulation_results_ramberg_osgoods = ev.read_all_simulation_data(data_directory_ramberg_osgoods)
data_to_plot_sorted_ramberg_osgoods, legend_entries_sorted = ev_ut.data_for_plot_wsteg_in_legend(ev_ut.normalize_Jx_to_Gc_num, gc_num_quotient, simulation_results_ramberg_osgoods, starting_hole_to_evaluate, steg_width_label, ev_ut.filter_data_by_column_bounds, ev_ut.get_x_range_between_ramberg_osgoods, hole_positions)
ev.plot_multiple_columns(data_objects=data_to_plot_sorted_ramberg_osgoods,
                      col_x=3,
                      col_y=1,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$x_{ct} / L$",ylabel=J_x_label,
                      usetex=True,markers_only=True,use_colors=True,
                      y_range=[0,2])

###############################################
# 2. VON MISES PLASTICITY MULTI-SIMULATION PLOT
###############################################
output_file_vm = os.path.join(script_path, 'PAPER_04_Jx_vs_xct_all_vonMises.png')

simulation_results_vonmises = ev.read_all_simulation_data(
    data_directory_vonmises
)

data_to_plot_sorted_vonmises, legend_entries_sorted_vonmises = \
    ev_ut.data_for_plot_wsteg_in_legend(
        ev_ut.normalize_Jx_to_Gc_num,
        gc_num_quotient,
        simulation_results_vonmises,
        starting_hole_to_evaluate,
        steg_width_label,
        ev_ut.filter_data_by_column_bounds,
        ev_ut.get_x_range_between_ramberg_osgoods,   # same function works
        hole_positions
    )

ev.plot_multiple_columns(
    data_objects=data_to_plot_sorted_vonmises,
    col_x=3,
    col_y=1,
    output_filename=output_file_vm,
    legend_labels=legend_entries_sorted_vonmises,
    xlabel="$x_{ct} / L$",
    ylabel=J_x_label,
    usetex=True,
    markers_only=True,
    use_colors=True,
    y_range=[0, 2]
)