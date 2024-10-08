import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

reference_L_global = 1.2720814740168862

def plot_results_2yaxis(results_dict, keys, column_indices, save_path, scaling_factors=None, y_labels=None, show_legend=True, label_fontsize=16, tick_fontsize=12):
    """
    Plots up to two result columns vs time for a set of keys and saves the plot to disk.
    The first column is plotted on the left y-axis and the second on the right y-axis.

    :param results_dict: Dictionary containing the results data.
                         The keys are folder names and values are numpy arrays with the data.
    :param keys: List of keys to plot.
    :param column_indices: List or tuple of up to two column indices to plot against time.
    :param save_path: File path to save the plot.
    :param scaling_factors: List of scaling factors for each key. If None, no scaling is applied.
    :param y_labels: List or tuple of y-axis labels. The first label is for the left y-axis, 
                     the second (if any) is for the right y-axis. If None, defaults are used.
    :param show_legend: Boolean to determine whether to show the legend. Default is True.
    :param label_fontsize: Font size for the axis labels. Default is 12.
    :param tick_fontsize: Font size for the axis tick numbers. Default is 10.
    """
    if scaling_factors is None:
        scaling_factors = [1.0] * len(keys)
    
    if len(keys) != len(scaling_factors):
        raise ValueError("The length of keys and scaling_factors must be the same.")
    
    if len(column_indices) > 2:
        raise ValueError("column_indices must contain one or two indices.")

    if y_labels is None:
        y_labels = [f'Column {col_index}' for col_index in column_indices]
    elif len(y_labels) != len(column_indices):
        raise ValueError("The length of y_labels must match the length of column_indices.")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Set the figure size for better readability

    ax2 = None
    if len(column_indices) > 1:
        ax2 = ax1.twinx()  # Create a secondary y-axis

    for key, scaling_factor in zip(keys, scaling_factors):
        if key in results_dict:
            data = results_dict[key]
            time = data[:, 0]

            # Plot the first column on the left y-axis
            result_column1 = data[:, column_indices[0]] * scaling_factor
            ax1.plot(time, result_column1, label=f"{key} (scaled by {scaling_factor})")

            if len(column_indices) > 1:
                # Plot the second column on the right y-axis
                result_column2 = data[:, column_indices[1]] * scaling_factor
                ax2.plot(time, result_column2, linestyle='--', label=f"{key} (scaled by {scaling_factor}, secondary)")
                
        else:
            print(f"Key '{key}' not found in the results dictionary.")

    ax1.set_xlabel('Time', fontsize=label_fontsize)
    ax1.set_ylabel(y_labels[0], fontsize=label_fontsize)  # Left y-axis label
    ax1.tick_params(axis='both', labelsize=tick_fontsize)  # Set font size for ticks on left y-axis

    if len(column_indices) > 1 and ax2 is not None:
        ax2.set_ylabel(y_labels[1], fontsize=label_fontsize)  # Right y-axis label
        ax2.tick_params(axis='both', labelsize=tick_fontsize)  # Set font size for ticks on right y-axis

    # Combine legends from both axes if show_legend is True
    if show_legend:
        if ax2 is not None:
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot to disk
    plt.savefig(save_path)
    plt.close()  # Close the plot to free up memory





def plot_results(results_dict, keys, column_index, save_path, scaling_factors=None):
    """
    Plots a given result column vs time for a set of keys and saves the plot to disk.

    :param results_dict: Dictionary containing the results data.
                         The keys are folder names and values are numpy arrays with the data.
    :param keys: List of keys to plot.
    :param column_index: Index of the result column to plot against time.
    :param save_path: File path to save the plot.
    :param scaling_factors: List of scaling factors for each key. If None, no scaling is applied.
    """
    if scaling_factors is None:
        scaling_factors = [1.0] * len(keys)
    
    if len(keys) != len(scaling_factors):
        raise ValueError("The length of keys and scaling_factors must be the same.")
    
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size for better readability

    for key, scaling_factor in zip(keys, scaling_factors):
        if key in results_dict:
            data = results_dict[key]
            time = data[:, 0]
            result_column = data[:, column_index] * scaling_factor  # Apply scaling
            
            plt.plot(time, result_column, label=f"{key} (scaled by {scaling_factor})")
        else:
            print(f"Key '{key}' not found in the results dictionary.")

    plt.xlabel('Time')
    plt.ylabel(f'Column {column_index}')
    # plt.title(f'Results for Column {column_index} vs Time')
    plt.legend()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot to disk
    plt.savefig(save_path)
    plt.close()  # Close the plot to free up memory


# Extract script path and name
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# Directory containing the simulation folders
directory_path = script_path

# Regular expression pattern to extract the required values from the outer folder names
outer_pattern = re.compile(
    r"simulation_\d+_\d{6}_(?P<mesh_name>[a-zA-Z0-9_]+)_lam(?P<lam_value>\d+\.\d+)_mue(?P<mue_value>\d+\.\d+)_Gc(?P<Gc_value>\d+\.\d+)_eps(?P<eps_value>\d+\.\d+)_order(?P<order_value>\d+)"
)

# Dictionary to store the data
results_dict = {}
first_level_keys = set()


# Iterate over all the directories
for folder_name in os.listdir(directory_path):
    outer_match = outer_pattern.match(folder_name)
    if outer_match:
        # Extract the values from the outer directory name
        mesh_name = outer_match.group("mesh_name")
        lam_value = float(outer_match.group("lam_value"))
        mue_value = float(outer_match.group("mue_value"))
        Gc_value = float(outer_match.group("Gc_value"))
        eps_value = float(outer_match.group("eps_value"))
        order_value = int(outer_match.group("order_value"))
        
        # Get the path of the outer folder
        outer_folder_path = os.path.join(directory_path, folder_name)
        
        # Check for the data file at the first level
        data_file_path = os.path.join(outer_folder_path, "pfmfrac_function_graphs.txt")
        
        first_level_found = False
        if not os.path.isfile(data_file_path):
            # List the contents of the outer folder to find the inner folder
            inner_folder_name = None
            for item in os.listdir(outer_folder_path):
                if item.startswith("simulation_") and os.path.isdir(os.path.join(outer_folder_path, item)):
                    inner_folder_name = item
                    break
            
            if inner_folder_name:
                # Construct the inner folder path
                inner_folder_path = os.path.join(outer_folder_path, inner_folder_name)
                
                # Path to the data file
                data_file_path = os.path.join(inner_folder_path, "pfmfrac_function_graphs.txt")
        else:
            first_level_found = True
        
        if os.path.isfile(data_file_path):
            # Read the data from the file
            data = []
            with open(data_file_path, 'r') as file:
                for line in file:
                    # Skip lines that start with #
                    if line.startswith('#'):
                        continue
                    # Split the line into columns and convert to float
                    data.append(list(map(float, line.split())))
            
            # Store the data in the dictionary
            key = f"{mesh_name}_lam{lam_value}_mue{mue_value}_Gc{Gc_value}_eps{eps_value}_order{order_value}"
            results_dict[key] = np.array(data)
            if first_level_found:
                first_level_keys.add(key)
            
           

def extract_parameters(key):
    # Regular expression pattern to match the key format
    pattern = re.compile(
        r"(?P<mesh_name>[a-zA-Z0-9_]+)_lam(?P<lam_value>\d+\.\d+)_mue(?P<mue_value>\d+\.\d+)_Gc(?P<Gc_value>\d+\.\d+)_eps(?P<eps_value>\d+\.\d+)_order(?P<order_value>\d+)"
    )
    
    match = pattern.match(key)
    if match:
        # Extract the values and convert them to appropriate types
        mesh_name = match.group("mesh_name")
        lam_value = float(match.group("lam_value"))
        mue_value = float(match.group("mue_value"))
        Gc_value = float(match.group("Gc_value"))
        eps_value = float(match.group("eps_value"))
        order_value = int(match.group("order_value"))
        return (mesh_name, lam_value, mue_value, Gc_value, eps_value, order_value)
    else:
        raise ValueError("Key format is incorrect")

def remove_parameter(param_string, param_names):
    """
    Removes the specified parameters and their values from the parameter string.

    :param param_string: The parameter string in the format 
                         "mesh_name_lam{lam_value}_mue{mue_value}_Gc{Gc_value}_eps{eps_value}_order{order_value}"
    :param param_names: List of lists/tuples of parameter names to remove (e.g., [['mesh_name'], ['lam', 'mue'], ['Gc']], etc.)
    :return: The parameter string with the specified parameters removed.
    """
    # Split the parameter string into parts
    parts = param_string.split('_')
    
    # Initialize indices to remove
    indices_to_remove = []
    
    # Handle removal of 'mesh_name' separately since it's at the beginning
    if ['mesh_name'] in param_names:
        indices_to_remove.append(0)  # Add index 0 for 'mesh_name'
    
    # Identify the indices of other parameters to remove
    for param_set in param_names:
        if param_set == ['mesh_name']:
            continue  # Skip since we've already handled 'mesh_name'
        
        found = False
        for i, part in enumerate(parts):
            if any(part.startswith(param_name) for param_name in param_set):
                found = True
                indices_to_remove.append(i)
                if len(part) > len(max(param_set, key=len)):  # Check length of longest parameter name in set
                    # Also remove the value if it exists after the parameter name
                    indices_to_remove.append(i + 1)
                break
        if not found:
            raise ValueError(f"No parameters '{param_set}' found in the parameter string")
    
    # Remove identified parameters and their values
    new_parts = [part for i, part in enumerate(parts) if i not in indices_to_remove]
    
    # Reconstruct the parameter string without the specified parameters
    new_param_string = '_'.join(new_parts)
    
    return new_param_string

# Example usage
# param_string = "coarse_pores_lam10.0_mue10.0_Gc0.5_eps50.0_order2"
# param_names = [['mesh_name'], ['lam', 'mue'], ['Gc']]
# new_param_string = remove_parameter(param_string, param_names)
# print(new_param_string)  # Output: "lam10.0_mue10.0_eps50.0_order2"


# 0.

keys = [ "medium_pores_lam1.0_mue1.0_Gc1.0_eps50.0_order2" ]
# parameter_names = ['Gc']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"TEST" + ".png")
plot_results_2yaxis(results_dict, keys, [1,4], save_path,y_labels=["Jx","crack length"],show_legend=False)

# 1. Gc
keys = ["coarse_pores_lam10.0_mue10.0_Gc0.5_eps50.0_order2", 
        "coarse_pores_lam10.0_mue10.0_Gc1.0_eps50.0_order2",
        "coarse_pores_lam10.0_mue10.0_Gc1.5_eps50.0_order2"]
parameter_names = ['Gc']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  remove_parameter(keys[0],
                        param_names=[parameter_names]) + ".png")
plot_results(results_dict, keys, column_index, save_path)

# 2. Stiffness
keys = ["coarse_pores_lam10.0_mue10.0_Gc1.0_eps50.0_order2", 
        "coarse_pores_lam10.0_mue10.0_Gc1.0_eps50.0_order2",
        "coarse_pores_lam15.0_mue10.0_Gc1.0_eps50.0_order2"]
parameter_names = ['lam','mue']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  remove_parameter(keys[0],
                        param_names=[parameter_names]) + ".png")
plot_results(results_dict, keys, column_index, save_path)


# 3. Mesh
keys = ["coarse_pores_lam1.0_mue1.0_Gc0.5_eps25.0_order2", 
        "medium_pores_lam1.0_mue1.0_Gc0.5_eps25.0_order2",
        "fine_pores_lam1.0_mue1.0_Gc0.5_eps25.0_order1"]
parameter_names = ['mesh_name']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  remove_parameter(keys[0],
                        param_names=[parameter_names]) + ".png")
plot_results(results_dict, keys, column_index, save_path)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  remove_parameter(keys[0],
                        param_names=[parameter_names]) + "_crack_tip_position.png")
plot_results(results_dict, keys, column_index=4, save_path=save_path)

# 4. Epsilon
# TODO: try to show that maximum energy release rate is the same if results are normalized by effective stress in phase field model

keys = ["coarse_pores_lam10.0_mue10.0_Gc1.0_eps25.0_order2", 
        "coarse_pores_lam10.0_mue10.0_Gc1.0_eps50.0_order2",
        "coarse_pores_lam10.0_mue10.0_Gc1.0_eps100.0_order2"]
parameter_names = ['eps']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  remove_parameter(keys[0],
                        param_names=[parameter_names]) + ".png")
plot_results(results_dict, keys, column_index, save_path)


# 4a. Epsilon
# With normalization w.r.t effective stress, SCALE BY NUMERIC Gc
keys = ["coarse_pores_lam10.0_mue10.0_Gc1.0_eps25.0_order2", 
        "coarse_pores_lam10.0_mue10.0_Gc1.0_eps50.0_order2",
        "coarse_pores_lam10.0_mue10.0_Gc1.0_eps100.0_order2"]

i = 0
def get_sig_c(extract_parameters,keys):
    def sig_c_quadr_deg(Gc, mu, epsilon):
        return 9.0/16.0 * math.sqrt(Gc*2.0*mu/(6.0*epsilon))
    
    def get_epsilon(eps_factor):
        refernce_L = reference_L_global 
        return refernce_L/eps_factor
    
    sig_c = np.zeros_like(keys,dtype=float)
    
    for i in range(0,len(keys)):
        key = keys[i]
        mue = extract_parameters(key)[2]
        gc = extract_parameters(key)[3]
        eps_factor = extract_parameters(key)[4]
        sig_c[i] =  sig_c_quadr_deg(mu=mue,Gc=gc,epsilon=get_epsilon(eps_factor))
    return sig_c

sig_c = get_sig_c(extract_parameters,keys)  
scaling_factors = 1.0 / (sig_c *  reference_L_global)

parameter_names = ['eps']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  remove_parameter(keys[0],
                        param_names=[parameter_names]) + "_TEST.png")
plot_results(results_dict, keys, column_index, save_path, scaling_factors=scaling_factors)




def create_max_dict(results_dict, column_index=1):
    """
    Creates a new dictionary that stores the maximum value of a specified column for each key in the results dictionary.

    :param results_dict: Dictionary containing the results data.
                         The keys are folder names and values are numpy arrays with the data.
    :param column_index: Index of the column for which to find the maximum value. Default is 1.
    :return: Dictionary with the same keys as results_dict and maximum values of the specified column as values.
    """
    max_dict = {}
    
    for key, data in results_dict.items():
        max_value = np.max(data[:, column_index])
        max_dict[key] = max_value
    
    return max_dict

def poisson_ratio(lam, mue):
    nu = lam / (2 * (lam + mue))
    return nu

# Example function modified for your specific plotting needs
def plot_max_Jx_vs_sig_c(results_dict, keys, plot_title, save_path, special_keys=None):
    # Create the max dictionary
    max_Jx_dict = create_max_dict(results_dict, column_index=1)
    
    # Compute sig_c for all keys in max_Jx_dict
    sig_c_values = get_sig_c(extract_parameters, keys)
    
    # Create a colormap for the mesh types
    mesh_colors = {
        "fine_pores": mcolors.to_rgba('darkred'),
        "medium_pores": mcolors.to_rgba('red'),
        "coarse_pores": mcolors.to_rgba('lightcoral')
    }
    
    # Marker types based on (lam, mue) values
    marker_types = {
        (1.0, 1.0): 'o',       # Circle
        (1.5, 1.0): 'p',      # Cross
        (1.0, 1.5): 'v',      # Dot
        (10.0, 10.0): '^',     # Triangle
        (15.0, 10.0): 's'      # Square
    }
    
    # Plot max_Jx_dict values vs sig_c_values with colors and markers based on mesh type and (lam, mue)
    plt.figure(figsize=(10, 20))
    
    unique_labels = set()
    
    for key, sig_c in zip(keys, sig_c_values):
        params = extract_parameters(key)
        if params:
            mesh_type = params[0]
            lam_mue = (params[1], params[2])
            
            color = mesh_colors.get(mesh_type, 'black')
            marker = marker_types.get(lam_mue, 'x')  # Default marker is 'x' if (lam, mue) is not in the marker_types dictionary
            label = mesh_type if mesh_type not in unique_labels else ""
            
            # Check if the current key is in the special_keys set
            if special_keys and key in special_keys:
                edge_color = 'black'
                linewidth = 2.0
            else:
                edge_color = 'none'
                linewidth = 0
            
            plt.scatter(sig_c, max_Jx_dict[key], color=color, marker=marker, s=100, label=label, edgecolor=edge_color, linewidth=linewidth)
            unique_labels.add(mesh_type)
            
            # Display Gc and inverse of eps values as tuples to the right of the markers
            plt.text(sig_c, max_Jx_dict[key], f"({params[3]}, {1.0/params[4]})", fontsize=9, ha='left', va='bottom')
    
    # Create custom legend handles
    handles = []
    
    # Add mesh type legend handles
    for mesh_type, color in mesh_colors.items():
        handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=12, label=mesh_type))
    
    # Add marker type legend handles with Poisson ratio included
    for (lam, mue), marker in marker_types.items():
        nu = poisson_ratio(lam, mue)
        handles.append(mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=12, label=f'lam={lam}, mue={mue} (nu={nu:.2f})'))
    
    plt.xlabel('Sig_c')
    plt.ylabel('Max Jx')
    plt.title(plot_title)
    plt.legend(handles=handles, title="Legend", loc='best')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()



max_Jx_dict = create_max_dict(results_dict, column_index=1)
# Compute sig_c for all keys in max_Jx_dict
keys = list(max_Jx_dict.keys())


def filter_keys(results_dict, target_Gc=None, target_eps=None, target_lam=None, target_mue=None, target_mesh_types=None):
    filtered_keys = []
    for key in results_dict.keys():
        params = extract_parameters(key)
        if params:
            mesh_type = params[0]
            Gc_value = params[3]
            eps_value = params[4]
            lam_value = params[1]
            mue_value = params[2]
            
            # Check if the key meets all specified criteria
            if (target_Gc is None or np.isclose(Gc_value, target_Gc).any()) and \
               (target_eps is None or np.isclose(eps_value, target_eps).any()) and \
               (target_lam is None or np.isclose(lam_value, target_lam).any()) and \
               (target_mue is None or np.isclose(mue_value, target_mue).any()) and \
               (target_mesh_types is None or mesh_type in target_mesh_types):
                filtered_keys.append(key)
    return filtered_keys

def compute_gc_num(mesh_name, gc, eps_factor, h_all):
    """
    Computes the gc_num value.

    :param mesh_name: Name of the mesh type.
    :param gc: Gc value.
    :param eps: Epsilon value.
    :param h_all: Dictionary mapping mesh types to their corresponding h values.
    :return: Computed gc_num value.
    """
    h_value = h_all[mesh_name]
    eps = reference_L_global / eps_factor
    return gc * (1.0 + h_value / eps / 4.0)

mesh_colors = {
        "fine_pores": mcolors.to_rgba('darkred'),
        "medium_pores": mcolors.to_rgba('red'),
        "coarse_pores": mcolors.to_rgba('lightcoral')
    }

def plot_gc_num_vs_gc(results_dict, keys, h_all, save_path):
    """
    Plots gc_num vs Gc for a set of keys and saves the plot to disk.

    :param results_dict: Dictionary containing the results data.
                         The keys are folder names and values are numpy arrays with the data.
    :param keys: List of keys to plot.
    :param h_all: Dictionary mapping mesh types to their corresponding h values.
    :param save_path: File path to save the plot.
    """
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size for better readability

    # mesh_colors = {
    #     "fine_pores": mcolors.to_rgba('darkred'),
    #     "medium_pores": mcolors.to_rgba('red'),
    #     "coarse_pores": mcolors.to_rgba('lightcoral')
    # }
    
    marker_types = {
        (1.0, 1.0): 'o',       # Circle
        (1.5, 1.0): 'p',      # Cross
        (1.0, 1.5): 'v',      # Dot
        (10.0, 10.0): '^',     # Triangle
        (15.0, 10.0): 's'      # Square
    }
    
    unique_labels = set()
    
    for key in keys:
        params = extract_parameters(key)
        if params:
            mesh_type = params[0]
            lam_value = params[1]
            mue_value = params[2]
            gc_value = params[3]
            eps_value = params[4]
            
            gc_num_value = compute_gc_num(mesh_name=mesh_type, gc=gc_value, eps_factor=eps_value, h_all=h_all)
            
            color = mesh_colors.get(mesh_type, 'black')
            marker = marker_types.get((lam_value, mue_value), 'x')  # Default marker is 'x' if (lam, mue) is not in the marker_types dictionary
            label = mesh_type if mesh_type not in unique_labels else ""
            
            plt.scatter(gc_value, gc_num_value, color=color, marker=marker, s=100, label=label)
            unique_labels.add(mesh_type)
            
            # Display gc_num values as text next to the markers
            plt.text(gc_value, gc_num_value, f"{1.0/eps_value:.2f}", fontsize=9, ha='right', va='bottom')
    
    # Create custom legend handles
    handles = []
    
    # Add mesh type legend handles
    for mesh_type, color in mesh_colors.items():
        handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=12, label=mesh_type))
    
    # Add marker type legend handles with Poisson ratio included
    for (lam, mue), marker in marker_types.items():
        handles.append(mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=12, label=f'lam={lam}, mue={mue}'))
    
    plt.xlabel('Gc')
    plt.ylabel('gc_num')
    plt.title('gc_num vs Gc')
    plt.legend(handles=handles, title="Legend", loc='best')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()

# MESH SIZES

h_coarse_mean =  0.024636717648428213 #0.040704583134024946
h_all = {
    "coarse_pores": h_coarse_mean,
    "medium_pores": h_coarse_mean/2.0,
    "fine_pores": h_coarse_mean/4.0,
}

# PORE SIZES
pore_size_coarse =  0.183
pore_size_all = {
    "coarse_pores": pore_size_coarse,
    "medium_pores": pore_size_coarse/2.0,
    "fine_pores": pore_size_coarse/4.0,
}

# def gc_num(mesh_name,gc,eps):
#     return gc * (1.0 + h_all(mesh_name)/eps)

mesh_all = ["fine_pores","medium_pores", "coarse_pores"]
gc_all = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
eps_all = np.array([25.0, 33.0, 40.0, 50.0, 100.0])
lam_all = np.array([1.0, 1.5, 10.0, 15.0])  
mue_all = np.array([1.0, 1.5, 10.0]) 

keys_to_plot = keys # ['fine_pores_lam1.0_mue1.0_Gc1.0_eps1.0_order1', 'medium_pores_lam10.0_mue10.0_Gc2.0_eps0.5_order2']
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "max_Jx_vs_sig_c.png")
plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, special_keys=first_level_keys)

# 1. fixed Gc
target_Gc_values = np.array([1.0])  
target_eps_values = np.array([25.0, 50.0, 100.0])  
target_lam_values = np.array([1.0, 10.0, 15.0])  
target_mue_values = np.array([1.0, 10.0])  
target_mesh_types = ["medium_pores", "coarse_pores"]  

filtered_keys = filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "max_Jx_vs_sig_c_medium_coarse.png")
plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, special_keys=first_level_keys)


# 2. varying stiffness
result_test = max_Jx_dict["coarse_pores_lam1.5_mue1.0_Gc1.0_eps50.0_order2"]

target_Gc_values = np.array([1.0])  
target_eps_values = np.array([50.0])  
target_lam_values = lam_all  
target_mue_values = mue_all
target_mesh_types = mesh_all 

filtered_keys = filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "max_Jx_vs_sig_c_varying_stiffness.png")
plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, special_keys=first_level_keys)

# 3. varying Gc
target_Gc_values = gc_all 
target_eps_values = np.array([50.0])  
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["medium_pores", "coarse_pores" ,"fine_pores"]  

filtered_keys = filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "max_Jx_vs_sig_c_varying_Gc.png")
plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, special_keys=first_level_keys)

# 4. varying eps
target_Gc_values = np.array([1.0])  
target_eps_values = eps_all  
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["medium_pores", "coarse_pores" ,"fine_pores"]  

filtered_keys = filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "max_Jx_vs_sig_c_varying_eps.png")
plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, special_keys=first_level_keys)


# 5. varying eps fixed sig_c
target_Gc_values = gc_all 
target_eps_values = eps_all 
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["medium_pores"]  

filtered_keys = filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "max_Jx_vs_sig_c_varying_eps_fixed_sigc.png")
plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, special_keys=first_level_keys)



# 6. plot all gc_num
target_Gc_values = gc_all 
target_eps_values = eps_all 
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["coarse_pores"]
filtered_keys = filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)

save_path = os.path.join(directory_path, "gc_num_vs_gc.png")
plot_gc_num_vs_gc(results_dict, filtered_keys, h_all, save_path)

# 7. plot ratio eps / h 

def plot_eps_h_ratio(keys, h_all, reference_L_global, mesh_colors, output_file, lower_limit_eps_h=0.0, lower_limit_pore_size_eps=0.0):
    ratios = []
    epss = []
    colors = []
    outlines = []
    
    for key in keys:
        params = extract_parameters(key)
        eps_factor = float(params[4])
        mesh_type = params[0]
        if eps_factor is not None and mesh_type in h_all:
            h = h_all[mesh_type]
            eps = reference_L_global / eps_factor
            ratio = eps / h
            ratios.append(ratio)
            epss.append(eps)
            colors.append(mesh_colors[mesh_type])
            
            # Determine if the point meets the criteria
            pore_size = pore_size_all[mesh_type]
            ratio_pore_size_eps = pore_size / eps
            if ratio >= lower_limit_eps_h and ratio_pore_size_eps >= lower_limit_pore_size_eps:
                outlines.append('green')
            else:
                outlines.append('none')
    
    plt.figure(figsize=(10, 6))
    
    for eps, ratio, color, outline in zip(epss, ratios, colors, outlines):
        plt.scatter(eps, ratio, c=[color], edgecolors=outline, marker='o', linewidths=1.5)
    
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\epsilon/h$')
    plt.title(r'Ratio of $\epsilon/h$ vs $\epsilon$')
    
    # Create custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mesh_colors[mesh], markersize=10, label=mesh) for mesh in mesh_colors]
    plt.legend(handles=handles, title="Mesh Types")
    
    plt.grid(True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()
    
    
target_Gc_values =  np.array([1.0])  
target_eps_values = eps_all 
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["coarse_pores", "medium_pores", "fine_pores"]
filtered_keys = filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)

save_path = os.path.join(directory_path, "eps_h_ratio.png")
plot_eps_h_ratio(filtered_keys, h_all, reference_L_global, mesh_colors, output_file=save_path, lower_limit_eps_h=2.0, lower_limit_pore_size_eps=2.0)
# plot_eps_h_ratio(filtered_keys,output_file=save_path)


# 8. plot ratio pore size vs eps and pore size vs h
def plot_ratio_pore_size_eps(results_dict, save_path):
    plt.figure(figsize=(10, 6))
    
    for key in results_dict.keys():
        params = extract_parameters(key)
        mesh_name = params[0]
        eps_factor = float(params[4])
        eps_value = reference_L_global / eps_factor
        pore_size = pore_size_all[mesh_name]
        ratio = pore_size / eps_value
        plt.scatter(eps_value, ratio, color=mesh_colors[mesh_name], label=mesh_name)
        
    plt.xlabel('eps')
    plt.ylabel('pore_size/eps')
    plt.title('Ratio of Pore Size to Eps for All Mesh Types')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_ratio_pore_size_h(results_dict, save_path):
    plt.figure(figsize=(10, 6))
    
    for key in results_dict.keys():
        params = extract_parameters(key)
        mesh_name = params[0]
        h_value = h_all[mesh_name]
        pore_size = pore_size_all[mesh_name]
        ratio = pore_size / h_value
        plt.scatter(h_value, ratio, color=mesh_colors[mesh_name], label=mesh_name)
        
    plt.xlabel('h')
    plt.ylabel('pore_size/h')
    plt.title('Ratio of Pore Size to h for All Mesh Types')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
# Assuming results_dict is already defined and populated
save_path_ratio_eps = os.path.join(directory_path, "ratio_pore_size_eps.png")
plot_ratio_pore_size_eps(results_dict, save_path_ratio_eps)

save_path_ratio_h = os.path.join(directory_path, "ratio_pore_size_h.png")
plot_ratio_pore_size_h(results_dict, save_path_ratio_h)



