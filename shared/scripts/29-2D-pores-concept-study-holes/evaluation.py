import pandas as pd
import matplotlib.pyplot as plt
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


def find_simulation_by_wsteg(path, wsteg_value_in):
    # Iterate over all directories starting with "simulation_"
    for directory in os.listdir(path):
        dir_path = os.path.join(path, directory)
        
        # Check if it is a directory and starts with "simulation_"
        if os.path.isdir(dir_path) and directory.startswith("simulation_"):
            parameters_file = os.path.join(dir_path, "parameters.txt")
            
            # Check if parameters.txt exists in the directory
            if os.path.isfile(parameters_file):
                # Open and read the file line by line
                with open(parameters_file, "r") as file:
                    for line in file:
                        # Look for the line that starts with "wsteg="
                        if line.startswith("wsteg="):
                            # Extract the value and convert to float
                            try:
                                wsteg_value = float(line.split("=")[1].strip())
                                # Compare with the given value
                                if wsteg_value == wsteg_value_in:
                                    return dir_path
                            except ValueError:
                                continue  # Skip line if conversion fails

    # Return None if no directory with the matching wsteg value is found
    return None


# Define the path to the file based on the script directory
script_path = os.path.dirname(__file__)
data_directory = os.path.join(script_path,'lam_mue_1.0_coarse')


simulation_data_folder = find_simulation_by_wsteg(data_directory,wsteg_value_in=1.75)

#simulation_data_folder= os.path.join(script_path,"simulation_20241204_010751")

data_path = os.path.join(simulation_data_folder, 'run_simulation_graphs.txt')
parameter_path = os.path.join(simulation_data_folder,"parameters.txt")

# Load the data from the text file, skipping the first row
data = pd.read_csv(data_path, delim_whitespace=True, header=None, skiprows=1)

# Display the first few rows of the data to understand its structure
# print(data.head()


def read_all_simulation_data(base_path):
    # List to store tuples of (data, parameters)
    simulation_results = []

    # Iterate through each item in the base directory
    for folder_name in os.listdir(base_path):
        # Check if the folder name starts with "simulation_"
        if folder_name.startswith("simulation_"):
            # Define paths for the data and parameter files
            data_path = os.path.join(base_path, folder_name, "run_simulation_graphs.txt")
            parameter_path = os.path.join(base_path, folder_name, "parameters.txt")
            
            # Try to read the data and parameter files
            try:
                # Read the simulation data
                data = pd.read_csv(data_path, delim_whitespace=True, header=None, skiprows=1)
                
                # Read the parameters
                parameters = pp.read_parameters_file(parameter_path)
                
                # Store the data and parameters together as a tuple
                simulation_results.append((data, parameters))
                
            except Exception as e:
                print(f"Error reading files in {folder_name}: {e}")
    
    return simulation_results


def filter_data_by_column_bounds(data, column_index, low_bound, upper_bound):
    """
    Filters the DataFrame to include only rows where the values in the specified column
    are between low_bound and upper_bound.

    Parameters:
    - data (pd.DataFrame): The input data.
    - column_index (int): The index of the column to apply the filter on.
    - low_bound (float): The lower bound of the filter range.
    - upper_bound (float): The upper bound of the filter range.

    Returns:
    - pd.DataFrame: A new DataFrame with only the rows where the specified column's values
                    are within the given bounds.
    """
    # Apply the filter condition
    filtered_data = data[(data.iloc[:, column_index] >= low_bound) & (data.iloc[:, column_index] <= upper_bound)]
    
    return filtered_data


def normalize_columns(data, columns_to_normalize,x_upper=99999,x_lower=-99999):
    """
    Normalizes specified columns of a DataFrame to values between 0 and 1.
    NaN values are replaced with -1 before normalization.

    Parameters:
    - data: pd.DataFrame, the DataFrame containing the data to normalize.
    - columns_to_normalize: list of str, column names to be normalized.

    Returns:
    - pd.DataFrame, a DataFrame with the specified columns normalized.
    """
    # Make a copy of the DataFrame to avoid modifying the original data
    normalized_data = data.copy()
    
    # Normalize each specified column
    for column in columns_to_normalize:
        # Replace NaN values with -1
        normalized_data[column].fillna(-1, inplace=True)

        min_val = normalized_data[column].min()
        max_val = normalized_data[column].max()

        # Check for the case where all values are the same
        if min_val == max_val:
            
            if np.isclose(max_val, x_upper,rtol=0.001):
                normalized_data[column] = 1.0
            elif np.isclose(max_val,x_lower,rtol=0.001):
                normalized_data[column] = 0.0
            else:
                normalized_data[column] = -1.0
                
            # If all values are the same, set the normalized values to 0
            
        else:
            # Apply normalization
            normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)

    return normalized_data

def shift_columns(data, columns_to_adjust, x_upper=99999, x_lower=-99999):
    """
    Adjusts specified columns of a DataFrame by subtracting the minimum value in each column 
    to ensure values start from 0. NaN values are replaced with -1 before adjustment.

    Parameters:
    - data: pd.DataFrame, the DataFrame containing the data to adjust.
    - columns_to_adjust: list of str, column names to be adjusted.

    Returns:
    - pd.DataFrame, a DataFrame with the specified columns adjusted.
    """
    # Make a copy of the DataFrame to avoid modifying the original data
    adjusted_data = data.copy()
    
    # Adjust each specified column
    for column in columns_to_adjust:
        # Replace NaN values with -1
        adjusted_data[column].fillna(-1, inplace=True)

        min_val = adjusted_data[column].min()
        max_val = adjusted_data[column].max()

        # Check for the case where all values are the same
        if min_val == max_val:
            if np.isclose(max_val, x_upper, rtol=0.001):
                adjusted_data[column] = x_upper
            elif np.isclose(max_val, x_lower, rtol=0.001):
                adjusted_data[column] = x_lower
            else:
                adjusted_data[column] = -1.0
        else:
            # Subtract the minimum value to adjust
            adjusted_data[column] = adjusted_data[column] - min_val

    return adjusted_data

def normalize_column_to_scale(data, column_to_normalize, x_upper, x_lower):
    """
    Normalizes a specified column of a DataFrame to values between 0 and 1,
    based on a given x_upper and x_lower which define the new scale.

    Parameters:
    - data: pd.DataFrame, the DataFrame containing the data to normalize.
    - column_to_normalize: str, the column name to be normalized.
    - x_upper: float, the value that corresponds to 1 in the normalized range.
    - x_lower: float, the value that corresponds to 0 in the normalized range.

    Returns:
    - pd.DataFrame, a DataFrame with the specified column normalized.
    """
    # Make a copy of the DataFrame to avoid modifying the original data
    normalized_data = data.copy()

    # Replace NaN values with -1 before normalization
    normalized_data[column_to_normalize].fillna(-1, inplace=True)

    # Normalize the column based on the given x_upper and x_lower
    min_val = x_lower
    max_val = x_upper

    # Check for the case where all values are the same as x_upper or x_lower
    column_values = normalized_data[column_to_normalize]
    
    if column_values.min() == column_values.max():
        # If all values are the same, check if they match the bounds
        if np.isclose(column_values.min(), x_upper, rtol=0.001):
            normalized_data[column_to_normalize] = 1.0
        elif np.isclose(column_values.min(), x_lower, rtol=0.001):
            normalized_data[column_to_normalize] = 0.0
        else:
            normalized_data[column_to_normalize] = -1.0
    else:
        # Apply normalization to scale values between 0 and 1 based on x_upper and x_lower
        normalized_data[column_to_normalize] = (
            (column_values - x_lower) / (x_upper - x_lower)
        ).clip(0, 1)  # Ensure the values stay between 0 and 1

    return normalized_data




def plot_columns(data, col_x, col_y, output_filename, vlines=None, hlines=None, 
                 xlabel=None, ylabel=None, title=None, 
                 xlabel_fontsize=16, ylabel_fontsize=16, title_fontsize=18, 
                 tick_fontsize=14, figsize=(10, 6), usetex=False):
    
    # Enable LaTeX rendering if requested
    if usetex:
        plt.rcParams['text.usetex'] = True

    # Set figure dimensions
    plt.figure(figsize=figsize)
    
    # Plot the data
    plt.plot(data[col_x], data[col_y], marker='o', linestyle='-')
    
    # Set custom labels and title, with specific font sizes
    plt.xlabel(xlabel if xlabel else f'Column {col_x}', fontsize=xlabel_fontsize)
    plt.ylabel(ylabel if ylabel else f'Column {col_y}', fontsize=ylabel_fontsize)
    plt.title(title if title else f' ', fontsize=title_fontsize)
    
    # Add dashed vertical lines if provided
    if vlines is not None:
        for vline in vlines:
            plt.axvline(x=vline, color='gray', linestyle='--', linewidth=1)
    
    # Add dashed horizontal lines if provided
    if hlines is not None:
        for hline in hlines:
            plt.axhline(y=hline, color='gray', linestyle='--', linewidth=1)
    
    # Set the maximum number of ticks on each axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit x-axis to 10 ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit y-axis to 10 ticks
    
    # Set fontsize for axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Save the plot as a PNG file
    plt.savefig(output_filename)
    plt.close()  # Close the figure to prevent display in some environments
    print(f"Plot saved as {output_filename}")

def plot_multiple_columns(data_objects, col_x, col_y, output_filename, 
                          vlines=None, hlines=None, xlabel=None, ylabel=None, 
                          title=None, legend_labels=None, 
                          xlabel_fontsize=16, ylabel_fontsize=16, title_fontsize=18, 
                          tick_fontsize=14, legend_fontsize=14, figsize=(10, 6), 
                          usetex=False, log_y=False):
    """
    Plots multiple datasets with the same x and y columns, allowing individual vertical and horizontal lines for each,
    with an optional logarithmic y-axis.
    
    Parameters:
        data_objects (list): List of data objects (DataFrames or dict-like) to be plotted.
        col_x (str): Column name for the x-axis.
        col_y (str): Column name for the y-axis.
        output_filename (str): File name to save the plot.
        vlines (list of lists): List of vertical line positions for each data object.
        hlines (list of lists): List of horizontal line positions for each data object.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        legend_labels (list): List of labels for the legend, corresponding to each data object.
        xlabel_fontsize (int): Font size for the x-axis label.
        ylabel_fontsize (int): Font size for the y-axis label.
        title_fontsize (int): Font size for the plot title.
        tick_fontsize (int): Font size for the axis tick labels.
        legend_fontsize (int): Font size for the legend labels.
        figsize (tuple): Figure dimensions as (width, height) in inches.
        usetex (bool): Whether to use LaTeX for rendering text in labels.
        log_y (bool): Whether to display the y-axis in logarithmic scale.
    """
    
    # Enable LaTeX rendering if requested
    if usetex:
        plt.rcParams['text.usetex'] = True

    # Set figure dimensions
    plt.figure(figsize=figsize)
    
    # Expand the color palette to support more datasets
    colors = list(mcolors.CSS4_COLORS.values())  # Use CSS4 color names for a broader palette
    
    for i, data in enumerate(data_objects):
        color = colors[i % len(colors)]
        
        # Plot the data
        plt.plot(data[col_x], data[col_y], marker='o', linestyle='-', color=color, 
                 label=legend_labels[i] if legend_labels else f'Data {i+1}')
        
        # Add dashed vertical lines specific to this data object
        if vlines and i < len(vlines):
            for vline in vlines[i]:
                plt.axvline(x=vline, color=color, linestyle='--', linewidth=1)
        
        # Add dashed horizontal lines specific to this data object
        if hlines and i < len(hlines):
            for hline in hlines[i]:
                plt.axhline(y=hline, color=color, linestyle='--', linewidth=1)
    
    # Set the y-axis to logarithmic scale if requested, and add minor ticks for readability
    ax = plt.gca()
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))  # Adds minor ticks for each order of magnitude
        ax.yaxis.set_minor_formatter(plt.NullFormatter())  # Hide minor tick labels to avoid clutter
    
    # Set custom labels and title, with specific font sizes
    plt.xlabel(xlabel if xlabel else f'Column {col_x}', fontsize=xlabel_fontsize)
    plt.ylabel(ylabel if ylabel else f'Column {col_y}', fontsize=ylabel_fontsize)
    plt.title(title if title else f' ', fontsize=title_fontsize)
    
    # Add legend with custom font size
    plt.legend(fontsize=legend_fontsize)
    
    # Set the maximum number of ticks on each axis
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit x-axis to 10 ticks
    if not log_y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # For linear y-axis, limit to 10 ticks
    
    # Set fontsize for axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Save the plot as a PNG file
    plt.savefig(output_filename)
    plt.close()  # Close the figure to prevent display in some environments
    print(f"Plot saved as {output_filename}")

def hole_positions(Nholes, dhole, wsteg):
    hole_start_positions = []
    hole_end_positions = []

    for n in range(Nholes):
        # Calculate the center of the hole
        x_center = (dhole + wsteg) * 1.5 + n * (dhole + wsteg)
        
        # Calculate the start and end positions of the hole
        x_start = x_center - dhole / 2
        x_end = x_center + dhole / 2
        
        # Append the results to the lists
        hole_start_positions.append(x_start)
        hole_end_positions.append(x_end)

    return hole_start_positions, hole_end_positions

def get_x_range_between_holes(Nholes, dhole, wsteg, start_hole, end_hole):
    # Get all hole start and end positions
    hole_starts, hole_ends = hole_positions(Nholes, dhole, wsteg)
    
    # Get the start position of the start_hole and the end position of the end_hole
    x_start = hole_starts[start_hole]
    x_end = hole_starts[end_hole]
    
    return x_start, x_end

parameters = pp.read_parameters_file(parameter_path)
Nholes = int(parameters["nholes"])
dhole = parameters["dhole"]
wsteg = parameters["wsteg"]
             
start_positions, end_positions = hole_positions(Nholes, 
                                                dhole,
                                                wsteg)


hole_positions_out = start_positions + end_positions
hole_positions_out.sort()


# Specify the output file path
output_file = os.path.join(script_path, 'all_Jx_vs_xct_pf.png')
plot_columns(data, 3, 1, output_file,vlines=hole_positions_out,xlabel="$x_{ct} / L$",ylabel="$J_{x} / G_c$", usetex=False, title=" ")

output_file = os.path.join(script_path, 'all_Jx_vs_A_pf.png')
plot_columns(data, 9, 1, output_file,vlines=None,xlabel="$x_{ct} / L$",ylabel="$J_{x} / G_c$", usetex=False, title=" ")


output_file = os.path.join(script_path, 'all_A_vs_t_pf.png')
plot_columns(data, 0, 9, output_file,vlines=None,xlabel="$t / T$",ylabel="$A[-]$", usetex=False, title=f"wsteg: {wsteg}")


output_file = os.path.join(script_path, 'range_Jx_vs_xct_pf.png')
x_low,x_high = get_x_range_between_holes(Nholes,dhole,wsteg,1,3)
low_boun = x_low-wsteg/8
upper_boun = x_high-wsteg/8
data_in_x_range = filter_data_by_column_bounds(data,3,low_bound=low_boun, upper_bound=upper_boun)
hole_postions_in_range = [hp for hp in hole_positions_out if low_boun <= hp <= upper_boun]
plot_columns(data_in_x_range, 3, 1, output_file,vlines=hole_postions_in_range,xlabel="xct_pf",ylabel="Jx",title="")

output_file = os.path.join(script_path, 'range_Jx_vs_A.png')
plot_columns(data_in_x_range, 9, 1, output_file,vlines=None,xlabel="A_pf",ylabel="Jx",title="")

output_file = os.path.join(script_path, 'range_A_vs_t.png')
data_shifted = shift_columns(data_in_x_range,[1,9])
plot_columns(data_shifted, 0, 9, output_file,vlines=None,xlabel="t",ylabel="A_pf",title=f"wsteg: {wsteg}")


# plot all curves
simulation_results = read_all_simulation_data(data_directory)
# output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
data_to_plot = []
legend_entries = []
wsteg_values = []

for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    
    Nholes = int(param["nholes"])
    dhole = param["dhole"]
    wsteg = param["wsteg"]
    wsteg_values.append(wsteg)

    
    x_low,x_high = get_x_range_between_holes(Nholes,dhole,wsteg,1,3)
    low_boun = x_low-wsteg/8
    upper_boun = x_high-wsteg/8
    data_in_x_range = filter_data_by_column_bounds(data,3,low_bound=low_boun, upper_bound=upper_boun)
    hole_postions_in_range = [hp for hp in hole_positions_out if low_boun <= hp <= upper_boun]
    
    data_to_plot.append(data_in_x_range)
    legend_entry = f"$w_s$: {wsteg}"
    legend_entries.append(legend_entry)
    
sorted_indices = sorted(range(len(wsteg_values)), key=lambda i: wsteg_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]



output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')  
plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=3,
                      col_y=1,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$x_{ct} / L$",ylabel="$J_{x} / G_c$")

output_file = os.path.join(script_path, 'Jx_vs_t_all.png')  
plot_multiple_columns(data_objects=data_to_plot_sorted,col_x=0,col_y=1,output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$t / T$",ylabel="$J_{x} / G_c$")

output_file = os.path.join(script_path, 'dt_vs_xct_all.png')  
plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=3,
                      col_y=10,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$x_{ct} / L$",ylabel="$dt / T$",
                      log_y=True)


# only crack growth - normalized
data_to_plot = []
legend_entries = []
wsteg_values = []
for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    
    Nholes = int(param["nholes"])
    dhole = param["dhole"]
    wsteg = param["wsteg"]
    wsteg_values.append(wsteg)
    
    x_low,x_high = get_x_range_between_holes(Nholes,dhole,wsteg,1,2)
    low_boun = x_high-(1.01*wsteg) #x_high-wsteg-0.01
    upper_boun = x_high + (0.01*wsteg)    #x_high+0.01
    data_in_x_range = filter_data_by_column_bounds(data,3,low_bound=low_boun, upper_bound=upper_boun)
    data_in_x_range_norm = normalize_columns(data_in_x_range, [0,1], x_upper=x_high, x_lower=x_high-wsteg)
    data_in_x_range_norm = normalize_column_to_scale(data_in_x_range_norm, 3, x_upper=x_high, x_lower=x_high-wsteg) # takes starting values ne x_high, x_low into account?
    # data_in_x_range_norm = normalize_column_to_scale(data_in_x_range_norm, 9, x_upper=x_high, x_lower=x_high-wsteg)
    
    hole_postions_in_range = [hp for hp in hole_positions_out if low_boun <= hp <= upper_boun]
    
    
    data_to_plot.append(data_in_x_range_norm)
    legend_entry = f"$w_s$: {wsteg}"
    legend_entries.append(legend_entry)

sorted_indices = sorted(range(len(wsteg_values)), key=lambda i: wsteg_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]

output_file = os.path.join(script_path, 'Jx_vs_xct_in_between_normalized.png')  
plot_multiple_columns(data_objects=data_to_plot_sorted,col_x=3,col_y=1,output_filename=output_file,legend_labels=legend_entries_sorted,xlabel="xct_pfm [wsteg]", ylabel="Jx_norm", title="Crack growth in steg")


output_file = os.path.join(script_path, 'Jx_vs_t_in_between_normalized.png')  
plot_multiple_columns(data_objects=data_to_plot_sorted,col_x=0,col_y=1,output_filename=output_file,legend_labels=legend_entries_sorted,xlabel="t", ylabel="Jx_norm", title="Crack growth in steg")

output_file = os.path.join(script_path, 'xct_vs_t_in_between_normalized.png')
data_without_xct_max = [filter_data_by_column_bounds(data,3,low_bound=0.0, upper_bound=0.99) for data in data_to_plot_sorted]   
plot_multiple_columns(data_objects=data_without_xct_max,col_x=0,col_y=3,output_filename=output_file,legend_labels=legend_entries_sorted,xlabel="t", ylabel="xct_pfm [wsteg]", title="Crack growth in steg")


output_file = os.path.join(script_path, 'xct_vs_t_in_between_normalized_single.png')
plot_columns(data_without_xct_max[6], 0, 3, output_file,vlines=None,xlabel="t", ylabel="xct_pfm [wsteg]", usetex=False, title=" ")

output_file = os.path.join(script_path, 'A_vs_t_in_between_normalized_single.png')
plot_columns(data_without_xct_max[6], 0, 9, output_file,vlines=None,xlabel="t", ylabel="A [-]", usetex=False, title=" ")

output_file = os.path.join(script_path, 'dt_vs_xct_in_between.png')  
plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=3,
                      col_y=10,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$x_{ct} / L$",ylabel="$dt / T$",
                      log_y=True)

KIc_master  = []
w_steg_master = []
Jx_max_master = []

simulation_results = read_all_simulation_data(data_directory)
# computing KIc 
KIc_effs = []
vol_ratios = []
wsteg_values = []
Jx_max_values = []
for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    
    Nholes = int(param["nholes"])
    dhole = param["dhole"]
    wsteg = param["wsteg"]
    
    
    wsteg_values.append(wsteg)
    w_cell = dhole+wsteg
    vol_cell = w_cell ** 2
    vol_ratio_material = (vol_cell - math.pi * (dhole/2)**2)/vol_cell
    vol_ratios.append(vol_ratio_material)
    
    x_low,x_high = get_x_range_between_holes(Nholes,dhole,wsteg,1,2)
    low_boun = x_high-(1.01*wsteg) #x_high-wsteg-0.01
    upper_boun = x_high + (0.01*wsteg)    #x_high+0.01
    data_in_x_range = filter_data_by_column_bounds(data,3,low_bound=low_boun, upper_bound=upper_boun)
    
    Jx_max = np.max(data_in_x_range[1])
    Jx_max_values.append(Jx_max)
    
    lam_eff = param["lam_effective"]
    mue_eff = param["mue_effective"]
    E_eff = le.get_emod(lam_eff,mue_eff)
    nu_eff = le.get_nu(lam_eff,mue_eff)
    
    E_star = E_eff/ (1-nu_eff**2)
    
    KIc_eff = np.sqrt(Jx_max*E_star) 
    
    KIc_effs.append(KIc_eff)

    
sorted_indices = sorted(range(len(wsteg_values)), key=lambda i: wsteg_values[i])
Jx_max_values_sorted = [Jx_max_values[i] for i in sorted_indices]
KIc_effs_sorted = [KIc_effs[i] for i in sorted_indices]
wsteg_values_sorted = [wsteg_values[i] for i in sorted_indices]
vol_ratios_sorted = [vol_ratios[i] for i in sorted_indices]


KIc_master.append(KIc_effs_sorted.copy())

w_steg_master.append(wsteg_values_sorted.copy())
Jx_max_master.append(Jx_max_values_sorted.copy())



simulation_results = read_all_simulation_data(data_directory)
# computing KIc 
KIc_effs = []
vol_ratios = []
wsteg_values = []
Jx_max_values = []
for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    
    Nholes = int(param["nholes"])
    dhole = param["dhole"]
    wsteg = param["wsteg"]
    
    
    wsteg_values.append(wsteg)
    w_cell = dhole+wsteg
    vol_cell = w_cell ** 2
    vol_ratio_material = (vol_cell - math.pi * (dhole/2)**2)/vol_cell
    vol_ratios.append(vol_ratio_material)
    
    x_low,x_high = get_x_range_between_holes(Nholes,dhole,wsteg,1,2)
    low_boun = x_high-(1.01*wsteg) #x_high-wsteg-0.01
    upper_boun = x_high + (0.01*wsteg)    #x_high+0.01
    data_in_x_range = filter_data_by_column_bounds(data,3,low_bound=low_boun, upper_bound=upper_boun)
    
    Jx_max = np.max(data_in_x_range[1])
    Jx_max_values.append(Jx_max)
    
    lam_eff = param["lam_effective"]
    mue_eff = param["mue_effective"]
    E_eff = le.get_emod(lam_eff,mue_eff)
    nu_eff = le.get_nu(lam_eff,mue_eff)
    
    E_star = E_eff/ (1-nu_eff**2)
    
    KIc_eff = np.sqrt(Jx_max*E_star) 
    
    KIc_effs.append(KIc_eff)

    
sorted_indices = sorted(range(len(wsteg_values)), key=lambda i: wsteg_values[i])
Jx_max_values_sorted = [Jx_max_values[i] for i in sorted_indices]
KIc_effs_sorted = [KIc_effs[i] for i in sorted_indices]
wsteg_values_sorted = [wsteg_values[i] for i in sorted_indices]
vol_ratios_sorted = [vol_ratios[i] for i in sorted_indices]


KIc_master.append(KIc_effs_sorted.copy())

w_steg_master.append(wsteg_values_sorted.copy())
Jx_max_master.append(Jx_max_values_sorted.copy())

def plot_multiple_lines(x_values, y_values, title='', x_label='', y_label='', legend_labels=None, output_file='plot.png'):
    """
    Plots multiple lines on the same graph and saves the output to a file.
    
    Parameters:
    - x_values: 2D list or numpy array containing x values for each line (shape: [n_lines, n_points]).
    - y_values: 2D list or numpy array containing y values for each line (shape: [n_lines, n_points]).
    - title: Title of the plot (default: '').
    - x_label: Label for the x-axis (default: '').
    - y_label: Label for the y-axis (default: '').
    - legend_labels: List of labels for each line in the legend (default: None).
    - output_file: File path (with extension) to save the plot (default: 'plot.png').
    """
    # Check if the dimensions of x_values and y_values match
    if len(x_values) != len(y_values):
        raise ValueError("The number of x and y value sets must match.")
    
    # Check if legend_labels are provided, otherwise default to numbered labels
    if legend_labels is None:
        legend_labels = [f"Line {i+1}" for i in range(len(x_values))]
    
    # Create a new figure
    plt.figure()
    
    # Plot each line
    for i in range(len(x_values)):
        plt.plot(x_values[i], y_values[i], label=legend_labels[i])
    
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Add legend
    plt.legend()
    
    # Save the plot to the specified file
    plt.savefig(output_file)
    
    # Close the plot to free up memory
    plt.close()
    
output_file = os.path.join(script_path,"KIc_vs_wsteg_varying_stiffness.png")
plot_multiple_lines(w_steg_master,KIc_master,x_label="$w_s$",y_label="KIc",legend_labels=["lam,mue=1.0", "lam,mue=0.5"],output_file=output_file)
output_file = os.path.join(script_path,"Jx_vs_wsteg_varying_stiffness.png")
plot_multiple_lines(w_steg_master,Jx_max_master,x_label="$w_s$",y_label="Jx_max",legend_labels=["lam,mue=1.0", "lam,mue=0.5"],output_file=output_file)



 
# Define the function to generate the plot and save it as a PNG file
def plot_KIc_vs_wsteg(KIc_effs_sorted, wsteg_values_sorted, output_path):
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(wsteg_values_sorted, KIc_effs_sorted, marker='o', linestyle='-', color='b')
    plt.xlabel('wsteg')
    plt.ylabel('KIc')
    plt.title('KIc vs. wsteg')
    plt.grid(True)
    
    # Save the plot to the specified path without displaying it interactively
    plt.savefig(output_path, format='png')
    plt.close()  # Close the plot to prevent it from displaying interactively


plot_KIc_vs_wsteg(KIc_effs_sorted,wsteg_values_sorted,os.path.join(script_path,"KIc_vs_wsteg.png"))

plot_KIc_vs_wsteg(Jx_max_values_sorted,wsteg_values_sorted,os.path.join(script_path,"Jxmax_vs_wsteg.png"))


def plot_KIc_div_volratios_vs_wsteg(KIc_effs_sorted, wsteg_values_sorted, vol_ratios_sorted, output_path, 
                                    xlabel_fontsize=18, ylabel_fontsize=18, title_fontsize=18, 
                                    tick_fontsize=16, figsize=(8, 8), usetex=False):
    """
    Plots KIc divided by volume ratios against the square root of wsteg values.
    
    Parameters:
        KIc_effs_sorted (list): Sorted KIc effective values.
        wsteg_values_sorted (list): Sorted wsteg values.
        vol_ratios_sorted (list): Sorted volume ratios.
        output_path (str): Path to save the output plot.
        xlabel_fontsize (int): Font size for the x-axis label.
        ylabel_fontsize (int): Font size for the y-axis label.
        title_fontsize (int): Font size for the title.
        tick_fontsize (int): Font size for the axis tick labels.
        figsize (tuple): Figure dimensions as (width, height) in inches.
        usetex (bool): Whether to use LaTeX for rendering text in labels.
    """
    
    # Enable LaTeX rendering if requested
    if usetex:
        plt.rcParams['text.usetex'] = True

    # Calculate KIc divided by volume ratios
    KIc_div_volratios = [kic / vol for kic, vol in zip(KIc_effs_sorted, vol_ratios_sorted)]
    wsteg_values_sorted = [math.sqrt(wsteg) for wsteg in wsteg_values_sorted]
    #wsteg_values_sorted = [wsteg for wsteg in wsteg_values_sorted]
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(wsteg_values_sorted, KIc_div_volratios, marker='o', linestyle='-', color='b')
    
    # Set axis labels and title with customizable font sizes
    plt.xlabel(r'$\sqrt{{w_s} / {L}}$', fontsize=xlabel_fontsize)
    #plt.xlabel(r'${w_s}$ / ${L}$', fontsize=xlabel_fontsize)
    plt.ylabel(r'$K_{Ic}$ / ${\Phi \sqrt{G_c\mu}}$', fontsize=ylabel_fontsize)
    plt.title(' ', fontsize=title_fontsize)
    
    # Add grid
    plt.grid(True)
    
    # Set tick label font size
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Save the plot to the specified path
    plt.savefig(output_path, format='png')
    plt.close()  # Close the plot to prevent interactive display
    print(f"Plot saved as {output_path}")

plot_KIc_div_volratios_vs_wsteg(KIc_effs_sorted,wsteg_values_sorted,vol_ratios_sorted,
                                os.path.join(script_path,"KIc_norm_vs_wsteg.png"))





##### Estimating G_c_effective

def Gc_eff(E_eff_s, L, sig_ff):
    return (L * sig_ff ** 2) / E_eff_s

def sig_ff_for_isolated_hole(dhole,epsilon,sig_loc_bar):
    a = dhole/2.0
    def antiderivative(r):
        # a = dhole / 2.0
        return 2.0 * r - a ** 2 / r - a ** 4 / (r ** 3)
    length_over_which_is_averaged = 1.0*epsilon
    out = (sig_loc_bar) * (length_over_which_is_averaged * 2.0) / (antiderivative(a+length_over_which_is_averaged) - antiderivative(a))
    return out

def sig_ff_medium_steg(dhole,wsteg,sig_loc_bar,epsilon):
    r = dhole / 2
    w = wsteg + dhole 
    sig_ff = sig_loc_bar / (3.0 + (4.0 *r) / (w - 2.0 * r)) # estimation
    # correct to averaged value, same as isolated holem how?
    return sig_ff

def sig_ff_thin_steg(dhole,wsteg,sig_loc_bar):
    return sig_loc_bar * wsteg / dhole

def sig_c_2D(la,mu,Gc,epsilon,i): # CK Diss Page 122
    def kappa(i):
        if i == 1:
            return (2.0 * la + 2.0 * mu) / (la + 2.0 * mu)
        elif i == 2:
            return (la + 2.0 * mu) / (2.0 * mu)
        else:
            raise NotImplementedError()
    return 3.0 * math.sqrt((3.0*kappa(i)*mu*Gc)/epsilon) / 16.0 

def e_star(la_eff,mu_eff):
    E_eff = le.get_emod(la_eff,mu_eff)
    nu_eff = le.get_nu(la_eff,mu_eff) 
    E_star = E_eff/ (1-nu_eff**2)
    return E_star

def Gc_eff_estimate(Gc_local,la_local,mu_local,la_eff,mu_eff,epsilon,dhole,wsteg,L):
    sig_c_2D_val = sig_c_2D(la_local,mu_local,Gc_local,epsilon,1)
    if wsteg / dhole <= 0.01:
        sig_ff = sig_ff_thin_steg(dhole,wsteg,sig_c_2D_val)
    elif wsteg/dhole < 5.0:
        sig_ff = sig_ff_medium_steg(dhole,wsteg,sig_c_2D_val,epsilon)
    else:
        sig_ff = sig_ff_for_isolated_hole(dhole,epsilon,sig_c_2D_val)
    E_star = e_star(la_eff,mu_eff)
    
    return Gc_eff(E_star,L,sig_ff)
       

simulation_results = read_all_simulation_data(data_directory)
wsteg_values = []
Gc_eff_est = []

L = 60.0
for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    
    lam_eff = param["lam_effective"]
    mue_eff = param["mue_effective"]
    Gc_local = param["Gc_simulation"]
    la_local = param["lam_micro_simulation"]
    mu_local = param["mue_micro_simulation"]
    epsilon = param["eps_simulation"]
    wsteg = param["wsteg"]
    dhole = param["dhole"]
    
    
    
    gc_estimate = Gc_eff_estimate(Gc_local=Gc_local,
                                  la_local=la_local,
                                  mu_local=mu_local,
                                  la_eff = lam_eff,
                                  mu_eff = mue_eff,
                                  epsilon=epsilon,
                                  dhole=dhole,
                                  wsteg=wsteg,
                                  L=L)
    
    wsteg_values.append(wsteg)
    Gc_eff_est.append(gc_estimate)
    
sorted_indices = sorted(range(len(wsteg_values)), key=lambda i: wsteg_values[i])
Gc_eff_est_sorted = [Gc_eff_est[i] for i in sorted_indices]
wsteg_values_sorted = [wsteg_values[i] for i in sorted_indices]


plot_KIc_vs_wsteg(Gc_eff_est_sorted,wsteg_values_sorted,os.path.join(script_path,"Gc_est_vs_wsteg.png"))
    
    
    
    
    
    