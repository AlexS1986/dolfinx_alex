import os
import pandas as pd
import alex.evaluation as ev
import numpy as np

# --------------------------------------------------
# Material parameters
# --------------------------------------------------
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

epsilon_c_elastic = (5.0 / 3.0) * np.sqrt(Jc / (E * 15.0 * eps_reg))
print(f"epsilon c elastic = {epsilon_c_elastic:.6f}")

# --------------------------------------------------
# Get script path
# --------------------------------------------------
script_path = os.path.dirname(__file__)

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
# Eq Jc
path_gc1 = os.path.join(
    script_path, 'run_simulation_linear_elastic_graphs_gc1.0.txt'
)
data_linear_elastic_gc1 = pd.read_csv(
    path_gc1, delim_whitespace=True, header=None, skiprows=1
)

# Eq sigma*
path_gc_scaled = os.path.join(
    script_path, 'run_simulation_linear_elastic_graphs0.5679.txt'
)
data_linear_elastic_gc_scaled = pd.read_csv(
    path_gc_scaled, delim_whitespace=True, header=None, skiprows=1
)

# Plasticity (von Mises)
path_plasticity = os.path.join(
    script_path, 'run_simulation_plasticity_graphs.txt'
)
data_plasticity = pd.read_csv(
    path_plasticity, delim_whitespace=True, header=None, skiprows=1
)

# --------------------------------------------------
# Compute maxima (stress)
# --------------------------------------------------
max_plasticity = data_plasticity[1].max()
max_gc1 = data_linear_elastic_gc1[1].max()
max_gc_scaled = data_linear_elastic_gc_scaled[1].max()

# --------------------------------------------------
# Compute maxima (energy)
# --------------------------------------------------
max_pi_plasticity = data_plasticity[2].max()
max_pi_gc1 = data_linear_elastic_gc1[2].max()
max_pi_gc_scaled = data_linear_elastic_gc_scaled[2].max()

# --------------------------------------------------
# Labels (stress)
# --------------------------------------------------
label_plasticity = (
    r"\textbf{vM} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_plasticity:.2f}" + r"$\sigma_y$)"
)

label_gc1 = (
    r"\textbf{Eq$\mathbf{J_c}$} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_gc1:.2f}" + r"$\sigma_y$)"
)

label_gc_scaled = (
    r"\textbf{Eq$\mathbf{\sigma^*}$} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_gc_scaled:.2f}" + r"$\sigma_y$)"
)

# --------------------------------------------------
# Labels (energy)
# --------------------------------------------------
label_pi_plasticity = (
    r"\textbf{vM} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_plasticity:.2f}" + r"$J_c^0L$)"
)

label_pi_gc1 = (
    r"\textbf{Eq$\mathbf{J_c}$} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_gc1:.2f}" + r"$J_c^0L$)"
)

label_pi_gc_scaled = (
    r"\textbf{Eq$\mathbf{\sigma^*}$} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_gc_scaled:.2f}" + r"$J_c^0L$)"
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
    # [epsilon_c, epsilon_s, epsilon_c_elastic]
]

# --------------------------------------------------
# Plot stress (ORDER: vM → EqJc → EqSigma)
# --------------------------------------------------
ev.plot_multiple_columns(
    [
        data_plasticity,
        data_linear_elastic_gc1,
        data_linear_elastic_gc_scaled
    ],
    col_x=0,
    col_y=1,
    output_filename=output_file_stress,
    legend_labels=[
        label_plasticity,
        label_gc1,
        label_gc_scaled
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
    vlines=[[1.18, 1.08, 0.82]],          # 3 vertical lines for the first dataset
    vline_colors=[['blue', 'red', 'green']],
    mark_peak=True# Corresponding colors
)


# --------------------------------------------------
# Plot energy (ORDER: vM → EqJc → EqSigma)
# --------------------------------------------------
ev.plot_multiple_columns(
    [
        data_plasticity,
        data_linear_elastic_gc1,
        data_linear_elastic_gc_scaled
    ],
    0, 2,
    output_file_energy,
    legend_labels=[
        label_pi_plasticity,
        label_pi_gc1,
        label_pi_gc_scaled
    ],
    usetex=True,
    xlabel=r"$\varepsilon / \sqrt{J_c^{0} / (L\mu_0)}$",
    ylabel=r"$\Pi / J_c^0L$",
    y_range=None,
    x_range=[0.0, 1.5],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20,
    figsize=(10, 8)
)

