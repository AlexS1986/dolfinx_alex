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
gc_num_quotient = 1.12  # from analysis without crack
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
# 1. SIGMA* (replaces Ramberg–Osgood)
data_directory_sig = os.path.join(
    script_path,
    "..",
    "045_standard_holes_PAPER",
    "results",
    "interpolation_as_in_plasticity_gc_0.5679",
)

# 2. Linear elastic Jc = 1.0
data_directory_Jc = os.path.join(
    script_path,
    "..",
    "045_standard_holes_PAPER",
    "results",
    "gc_1.0",
)

# 3. von Mises plasticity
data_directory_vonmises = os.path.join(
    script_path,
    "..",
    "046_plasticity",
    "results",
    "gc_1.0_elastic",
)

# -------------------------------
# LOAD + NORMALIZE SIGMA*
# -------------------------------
simulation_data_folder_sig = ev_ut.find_simulation_by_wsteg(
    data_directory_sig,
    wsteg_value_in=1.0,
)
if simulation_data_folder_sig is None:
    raise FileNotFoundError("No matching sigma* dataset found for wsteg=1.0.")

data_sig = pd.read_csv(
    os.path.join(simulation_data_folder_sig, "run_simulation_graphs.txt"),
    delim_whitespace=True,
    header=None,
    skiprows=1,
)
ev_ut.normalize_Jx_to_Gc_num(gc_num_quotient, data_sig)

# -------------------------------
# LOAD + NORMALIZE Jc
# -------------------------------
simulation_data_folder_Jc = ev_ut.find_simulation_by_wsteg(
    data_directory_Jc,
    wsteg_value_in=1.0,
)
if simulation_data_folder_Jc is None:
    raise FileNotFoundError("No matching Jc dataset found for wsteg=1.0.")

data_Jc = pd.read_csv(
    os.path.join(simulation_data_folder_Jc, "run_simulation_graphs.txt"),
    delim_whitespace=True,
    header=None,
    skiprows=1,
)
ev_ut.normalize_Jx_to_Gc_num(gc_num_quotient, data_Jc)

# -------------------------------
# LOAD + NORMALIZE von Mises
# -------------------------------
simulation_data_folder_vonmises = ev_ut.find_simulation_by_wsteg(
    data_directory_vonmises,
    wsteg_value_in=1.0,
)
if simulation_data_folder_vonmises is None:
    raise FileNotFoundError("No matching von Mises dataset found for wsteg=1.0.")

data_vonmises = pd.read_csv(
    os.path.join(simulation_data_folder_vonmises, "run_simulation_graphs.txt"),
    delim_whitespace=True,
    header=None,
    skiprows=1,
)
ev_ut.normalize_Jx_to_Gc_num(gc_num_quotient, data_vonmises)


output_file = os.path.join(
    script_path,
    "PAPER_00_xct_pf_vs_xct_KI_vonMises.png",
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
    ylabel="crack tip position" + " $/ L$",
    x_range=[-0.1, 17],
    # vlines=[ramberg_osgood_positions_out, ramberg_osgood_positions_out],
)


# -------------------------------
# READ PARAMETERS FOR HOLES
# -------------------------------
parameter_path = os.path.join(simulation_data_folder_Jc, "parameters.txt")
parameters = pp.read_parameters_file(parameter_path)
nhole = int(parameters["nholes"])
dhole = parameters["dhole"]
wsteg = parameters["wsteg"]

start_positions, end_positions = ev_ut.ramberg_osgood_positions(
    nhole, dhole, wsteg
)
hole_positions = start_positions + end_positions
hole_positions.sort()

# -------------------------------
# LABELS
# -------------------------------
sig_label = r"\textbf{Eq}$\mathbf{\sigma^*}$"
elastic_label_Jc = r"\textbf{Eq}$\mathbf{J_c}$"
vonmises_label = r"\textbf{vM}"
xlabel = "$x_{ct} / L$"
ylabel = "$J_{x} / J_c^{\\mathrm{num}}$"

# -------------------------------
# MAIN COMBINED PLOT
# -------------------------------
output_file = os.path.join(
    script_path,
    "PAPER_clean_Jx_vs_xct_SIG_Jc_vonMises.png",
)
ev.plot_multiple_columns(
    data_objects=[data_sig, data_Jc, data_vonmises],
    col_x=3,
    col_y=1,
    output_filename=output_file,
    vlines=[hole_positions, hole_positions],
    legend_labels=[sig_label, elastic_label_Jc, vonmises_label],
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
# MULTI-SIMULATION PLOTS
# =====================================================

# --- SIGMA* ---
output_file_sig_all = os.path.join(
    script_path,
    "PAPER_04_Jx_vs_xct_all_SIG.png",
)
simulation_results_sig = ev.read_all_simulation_data(
    data_directory_sig
)
data_to_plot_sorted_sig, legend_entries_sorted_sig = ev_ut.data_for_plot_wsteg_in_legend(
    ev_ut.normalize_Jx_to_Gc_num,
    gc_num_quotient,
    simulation_results_sig,
    starting_hole_to_evaluate,
    steg_width_label,
    ev_ut.filter_data_by_column_bounds,
    ev_ut.get_x_range_between_ramberg_osgoods,
    hole_positions,
)
ev.plot_multiple_columns(
    data_objects=data_to_plot_sorted_sig,
    col_x=3,
    col_y=1,
    output_filename=output_file_sig_all,
    legend_labels=legend_entries_sorted_sig,
    xlabel=xlabel,
    ylabel=J_x_label,
    usetex=True,
    markers_only=True,
    use_colors=True,
    y_range=[0, 2],
)

# --- von Mises ---
output_file_vm_all = os.path.join(
    script_path,
    "PAPER_04_Jx_vs_xct_all_vonMises.png",
)
simulation_results_vonmises = ev.read_all_simulation_data(
    data_directory_vonmises
)
data_to_plot_sorted_vonmises, legend_entries_sorted_vonmises = ev_ut.data_for_plot_wsteg_in_legend(
    ev_ut.normalize_Jx_to_Gc_num,
    gc_num_quotient,
    simulation_results_vonmises,
    starting_hole_to_evaluate,
    steg_width_label,
    ev_ut.filter_data_by_column_bounds,
    ev_ut.get_x_range_between_ramberg_osgoods,
    hole_positions,
)
ev.plot_multiple_columns(
    data_objects=data_to_plot_sorted_vonmises,
    col_x=3,
    col_y=1,
    output_filename=output_file_vm_all,
    legend_labels=legend_entries_sorted_vonmises,
    xlabel=xlabel,
    ylabel=J_x_label,
    usetex=True,
    markers_only=True,
    use_colors=True,
    y_range=[0, 2],
)

# =====================================================
# PAPER_01 SINGLE-DATASET PLOTS
# =====================================================

# --- PAPER_01: SIGMA* ---
ev.plot_columns(
    data_sig,
    col_x=3,
    col_y=1,
    output_filename=os.path.join(
        script_path,
        "PAPER_01_Jx_vs_xct_SIG.png",
    ),
    vlines=hole_positions,
    xlabel=xlabel,
    ylabel=J_x_label,
    usetex=True,
    title=" ",
    plot_dots=False,
)

# --- PAPER_01: von Mises ---
ev.plot_columns(
    data_vonmises,
    col_x=3,
    col_y=1,
    output_filename=os.path.join(
        script_path,
        "PAPER_01_Jx_vs_xct_vonMises.png",
    ),
    vlines=hole_positions,
    xlabel=xlabel,
    ylabel=J_x_label,
    usetex=True,
    title=" ",
    plot_dots=False,
)

# --- PAPER_01: Linear elastic Jc ---
ev.plot_columns(
    data_Jc,
    col_x=3,
    col_y=1,
    output_filename=os.path.join(
        script_path,
        "PAPER_01_Jx_vs_xct_Jc.png",
    ),
    vlines=hole_positions,
    xlabel=xlabel,
    ylabel=J_x_label,
    usetex=True,
    title=" ",
    plot_dots=False,
)

# =====================================================
# PAPER_06b-TYPE PLOT (Jx_max vs w_steg) FOR THREE DATASETS
# =====================================================


def data_vs_wsteg_small(
    normalize_Jx_to_Gc_num,
    gc_num_quotient,
    data_directory,
    starting_hole,
    filter_data_by_column_bounds,
    get_x_range_between_ramberg_osgoods,
):
    """
    Reduced version of data_vs_wsteg from the large evaluation.py,
    using helpers from evaluation_utils.
    """
    simulation_results = ev.read_all_simulation_data(data_directory)

    KIc_effs = []
    vol_ratios = []
    wsteg_values = []
    Jx_max_values = []
    JxXEstar_max_values = []

    for sim in simulation_results:
        data = sim[0]
        normalize_Jx_to_Gc_num(gc_num_quotient, data)

        param = sim[1]
        nhole_loc = int(param["nholes"])
        dhole_loc = param["dhole"]
        wsteg_loc = param["wsteg"]
        wsteg_values.append(wsteg_loc)

        w_cell = dhole_loc + wsteg_loc
        vol_cell = w_cell ** 2
        vol_ratio_material = (vol_cell - math.pi * (dhole_loc / 2.0) ** 2) / vol_cell
        vol_ratios.append(vol_ratio_material)

        x_low, x_high = get_x_range_between_ramberg_osgoods(
            nhole_loc, dhole_loc, wsteg_loc, starting_hole, starting_hole + 1
        )
        low_boun = x_high - (1.01 * wsteg_loc)
        upper_boun = x_high + (0.01 * wsteg_loc)

        data_in_x_range = filter_data_by_column_bounds(
            data, 3, low_bound=low_boun, upper_bound=upper_boun
        )
        Jx_max = np.max(data_in_x_range[1])
        Jx_max_values.append(Jx_max)

        lam_eff = param["lam_effective"]
        mue_eff = param["mue_effective"]
        E_eff = le.get_emod(lam_eff, mue_eff)
        nu_eff = le.get_nu(lam_eff, mue_eff)
        E_star = E_eff / (1.0 - nu_eff ** 2)
        KIc_eff = np.sqrt(Jx_max * E_star)
        KIc_effs.append(KIc_eff)
        JxXEstar_max_values.append(E_eff * Jx_max)

    sorted_indices = sorted(range(len(wsteg_values)), key=lambda i: wsteg_values[i])

    Jx_max_values_sorted = [Jx_max_values[i] for i in sorted_indices]
    JxXEstar_max_values_sorted = [JxXEstar_max_values[i] for i in sorted_indices]
    KIc_effs_sorted = [KIc_effs[i] for i in sorted_indices]
    wsteg_values_sorted = [wsteg_values[i] for i in sorted_indices]
    vol_ratios_sorted = [vol_ratios[i] for i in sorted_indices]

    return (
        Jx_max_values_sorted,
        JxXEstar_max_values_sorted,
        KIc_effs_sorted,
        wsteg_values_sorted,
        vol_ratios_sorted,
    )


KIc_master = []
w_steg_master = []
Jx_max_master = []
JxXEstar_max_master = []

# 1) use SIGMA* directory as RO-like dataset
(
    Jx_max_values_sorted,
    JxXEstar_max_values_sorted,
    KIc_effs_sorted,
    wsteg_values_sorted,
    vol_ratios_sorted,
) = data_vs_wsteg_small(
    ev_ut.normalize_Jx_to_Gc_num,
    gc_num_quotient,
    data_directory_sig,
    starting_hole_to_evaluate,
    ev_ut.filter_data_by_column_bounds,
    ev_ut.get_x_range_between_ramberg_osgoods,
)
KIc_master.append(KIc_effs_sorted.copy())
w_steg_master.append(wsteg_values_sorted.copy())
Jx_max_master.append(Jx_max_values_sorted.copy())
JxXEstar_max_master.append(JxXEstar_max_values_sorted.copy())

# 2) Jc directory
(
    Jx_max_values_sorted,
    JxXEstar_max_values_sorted,
    KIc_effs_sorted,
    wsteg_values_sorted,
    vol_ratios_sorted,
) = data_vs_wsteg_small(
    ev_ut.normalize_Jx_to_Gc_num,
    gc_num_quotient,
    data_directory_Jc,
    starting_hole_to_evaluate,
    ev_ut.filter_data_by_column_bounds,
    ev_ut.get_x_range_between_ramberg_osgoods,
)
KIc_master.append(KIc_effs_sorted.copy())
w_steg_master.append(wsteg_values_sorted.copy())
Jx_max_master.append(Jx_max_values_sorted.copy())
JxXEstar_max_master.append(JxXEstar_max_values_sorted.copy())

# 3) von Mises directory (Pi-like)
(
    Jx_max_values_sorted,
    JxXEstar_max_values_sorted,
    KIc_effs_sorted,
    wsteg_values_sorted,
    vol_ratios_sorted,
) = data_vs_wsteg_small(
    ev_ut.normalize_Jx_to_Gc_num,
    gc_num_quotient,
    data_directory_vonmises,
    starting_hole_to_evaluate,
    ev_ut.filter_data_by_column_bounds,
    ev_ut.get_x_range_between_ramberg_osgoods,
)
KIc_master.append(KIc_effs_sorted.copy())
w_steg_master.append(wsteg_values_sorted.copy())
Jx_max_master.append(Jx_max_values_sorted.copy())
JxXEstar_max_master.append(JxXEstar_max_values_sorted.copy())


output_file = os.path.join(
    script_path,
    "PAPER_06b_Jx_vs_wsteg_three_datasets.png",
)
ev.plot_multiple_lines(
    x_values=w_steg_master,
    y_values=Jx_max_master,
    x_label="$w_s / L$",
    y_label=J_x_max_label,
    legend_labels=[sig_label, elastic_label_Jc,vonmises_label],
    output_file=output_file,
    usetex=True,
    markers_only=False,
    use_colors=True,
    marker_size=6,
    y_range=[0, 2.5],
    x_range=[0.375, 3.0],
    bold_text=True,
)

print("Ramberg–Osgood removed. Sigma* dataset fully integrated. All plots generated.")
