import os
import pandas as pd
import alex.evaluation as ev
import numpy as np 


Jc = 1.0
sigma_y0 = 1.0
H = 0.2222222
E = 2.0
eps_reg = 0.1  # epsilon in the formula

term1 = (sigma_y0**2 / H) + (5.0 / 18.0) * (Jc / eps_reg)
term2 = np.sqrt(
    4.0 * (sigma_y0**2 / H)**2
    + (8.0 / 9.0) * (sigma_y0**2 * Jc) / (H * eps_reg)
    + (25.0 / 324.0) * (Jc / eps_reg)**2
)

epsilon_c = np.sqrt(
    (1.0 / 3.0) * (E + H) / (E * H) * (term1 + term2)
) 

print(f"Fracture strain epsilon_c = {epsilon_c:.6f}")

epsilon_s = (
    np.sqrt(
        (E + H) / (E * H)
        * ((sigma_y0**2) / H + (1.0 / 6.0) * Jc / eps_reg)
    )
    - sigma_y0
) 

print(f"Secondary strain epsilon_s = {epsilon_s:.6f}")

epsilon_c_elastic = (5.0/3.0) * np.sqrt( (Jc)  / (E * 15.0*eps_reg))
print(f"epsilon c elastic = {epsilon_c_elastic:.6f}")


# --------------------------------------------------
# Get script path
# --------------------------------------------------
script_path = os.path.dirname(__file__)

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
# Eq Jc (gc = 1.0)
path_gc1 = os.path.join(
    script_path, 'run_simulation_linear_elastic_graphs_gc1.0.txt'
)
data_linear_elastic_gc1 = pd.read_csv(
    path_gc1, delim_whitespace=True, header=None, skiprows=1
)

# Eq sigma* (scaled Jc = 0.5679)
path_gc_scaled = os.path.join(
    script_path, 'run_simulation_linear_elastic_graphs0.5679.txt'
)
data_linear_elastic_gc_scaled = pd.read_csv(
    path_gc_scaled, delim_whitespace=True, header=None, skiprows=1
)

# Reduced stiffness (mu = 0.25 mu0)
path_gc_mu025 = os.path.join(
    script_path, 'run_simulation_linear_elastic_graphs_mu0.25.txt'
)
data_linear_elastic_gc_mu025 = pd.read_csv(
    path_gc_mu025, delim_whitespace=True, header=None, skiprows=1
)

# Plasticity
path_plasticity = os.path.join(
    script_path, 'run_simulation_plasticity_graphs.txt'
)
data_plasticity = pd.read_csv(
    path_plasticity, delim_whitespace=True, header=None, skiprows=1
)

# # --------------------------------------------------
# # Plasticity post-peak energy drop (von Mises)
# # --------------------------------------------------
# E_total_vm = data_plasticity[2].values

# idx_peak = E_total_vm.argmax()
# E_peak = E_total_vm[idx_peak]
# E_min_after_peak = E_total_vm[idx_peak + 1:].min()
# max_energy_drop_vm = E_peak - E_min_after_peak

# print(f"Peak E_total (von Mises): {E_peak:.6e}")
# print(f"Min E_total after peak (von Mises): {E_min_after_peak:.6e}")
# print(f"Maximal post-peak E_total drop (von Mises): {max_energy_drop_vm:.6e}")

# --------------------------------------------------
# Compute maxima (stress)
# --------------------------------------------------
max_gc1 = data_linear_elastic_gc1[1].max()
max_gc_scaled = data_linear_elastic_gc_scaled[1].max()
max_gc_mu025 = data_linear_elastic_gc_mu025[1].max()
max_plasticity = data_plasticity[1].max()

# --------------------------------------------------
# Compute maxima (energy)
# --------------------------------------------------
max_pi_gc1 = data_linear_elastic_gc1[2].max()
max_pi_gc_scaled = data_linear_elastic_gc_scaled[2].max()
max_pi_gc_mu025 = data_linear_elastic_gc_mu025[2].max()
max_pi_plasticity = data_plasticity[2].max()

# --------------------------------------------------
# Labels (stress)
# --------------------------------------------------
label_gc1 = (
    r"\textbf{Eq$\mathbf{J_c}$} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_gc1:.2f}" + r"$\sigma_y$)"
)

label_gc_scaled = (
    r"\textbf{Eq$\mathbf{\sigma^*}$} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_gc_scaled:.2f}" + r"$\sigma_y$)"
)

label_gc_mu025 = (
    r"\textbf{Reduced stiffness} "
    r"($\mu = 0.25\mu_0$, "
    r"max $\sigma^*$ $\approx$ " + f"{max_gc_mu025:.2f}" + r"$\sigma_y$)"
)

label_plasticity = (
    r"\textbf{Plasticity} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_plasticity:.2f}" + r"$\sigma_y$)"
)

# --------------------------------------------------
# Labels (energy)
# --------------------------------------------------
label_pi_gc1 = (
    r"\textbf{Eq$\mathbf{J_c}$} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_gc1:.2f}" + r"$J_c^0L$)"
)

label_pi_gc_scaled = (
    r"\textbf{Eq$\mathbf{\sigma^*}$} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_gc_scaled:.2f}" + r"$J_c^0L$)"
)

label_pi_gc_mu025 = (
    r"\textbf{Reduced stiffness} "
    r"($\mu = 0.25\mu_0$, "
    r"max $\Pi^*$ $\approx$ " + f"{max_pi_gc_mu025:.2f}" + r"$J_c^0L$)"
)

label_pi_plasticity = (
    r"\textbf{Plasticity} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_plasticity:.2f}" + r"$J_c^0L$)"
)

# --------------------------------------------------
# Output files
# --------------------------------------------------
output_file_stress = os.path.join(
    script_path, 'PAPER_reaction_forces_1D_EQ_ONLY.png'
)
output_file_energy = os.path.join(
    script_path, 'PAPER_energy_vs_displacement_1D_EQ_ONLY.png'
)

vlines = [
    [epsilon_c,epsilon_s,epsilon_c_elastic ]
]

# --------------------------------------------------
# Plot stress
# --------------------------------------------------
ev.plot_multiple_columns(
    [
        data_linear_elastic_gc1,
        data_linear_elastic_gc_scaled,
        #data_linear_elastic_gc_mu025,
        data_plasticity
    ],
    0, 1,
    output_file_stress,
    legend_labels=[
        label_gc1,
        label_gc_scaled,
        #label_gc_mu025,
        label_plasticity
    ],
    usetex=True,
    xlabel=r"$\varepsilon / \sqrt{J_c^{0} / (L\mu_0)}$",
    ylabel=r"$\sigma / \sigma_y$",
    y_range=[0.0, 2.0],
    x_range=[0.0, 1.5],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20,
    figsize=(10, 8),
    vlines=vlines
)

# --------------------------------------------------
# Plot energy
# --------------------------------------------------
ev.plot_multiple_columns(
    [
        data_linear_elastic_gc1,
        data_linear_elastic_gc_scaled,
        #data_linear_elastic_gc_mu025,
        data_plasticity
    ],
    0, 2,
    output_file_energy,
    legend_labels=[
        label_pi_gc1,
        label_pi_gc_scaled,
       # label_pi_gc_mu025,
        label_pi_plasticity
    ],
    usetex=True,
    xlabel=r"$u_{0} / \sqrt{J_c^{0}L / \mu_0}$",
    ylabel=r"$\Pi / J_c^0L$",
    y_range=None,
    x_range=[0.0, 1.5],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20,
    figsize=(10, 8)
)

