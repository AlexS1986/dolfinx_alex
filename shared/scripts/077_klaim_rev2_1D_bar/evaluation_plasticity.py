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
script_path = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Load datasets (077 runs: beta = 0.2, mu = 1, lam = 0, eta = 1e-5, eps = 0.1, M = 1000)
# 1st column = end displacement u_0 = strain (bar length 2L, u = -/+u_0), 2nd = R_x (bar height L = 1), 3rd = W
# --------------------------------------------------
GC_EQSIG = "0.5523"  # [Eq sigma*], equal 1D fracture stress to vM at beta = 0.2 (075/bar1d_homogeneous_check.py)


def load(name):
    return pd.read_csv(os.path.join(script_path, name), sep=r"\s+", header=None, skiprows=1)


# Eq Jc
data_linear_elastic_gc1 = load('run_simulation_linear_elastic_el_gc1.0_graphs.txt')
# Eq sigma*
data_linear_elastic_gc_scaled = load('run_simulation_linear_elastic_el_gc' + GC_EQSIG + '_graphs.txt')
# Plasticity (von Mises)
data_plasticity = load('run_simulation_plasticity_vM_gc1.0_graphs.txt')

# --------------------------------------------------
# Table of all runs: sigma* = max R_x, eps* = u_0 at the peak, Pi* = max W (and W at the peak)
# --------------------------------------------------
runs = [
    ("[vM]", "1.0", 'run_simulation_plasticity_vM_gc1.0_graphs.txt'),
    ("[Eq Jc]", "1.0", 'run_simulation_linear_elastic_el_gc1.0_graphs.txt'),
    ("[Eq sigma*]", "0.5523", 'run_simulation_linear_elastic_el_gc0.5523_graphs.txt'),
    ("[Eq sigma*] old", "0.5679", 'run_simulation_linear_elastic_el_gc0.5679_graphs.txt'),
    ("[vM] rate 0.5", "1.0", 'run_simulation_plasticity_vM_gc1.0_rate0.5_graphs.txt'),
    ("[vM] M = 1e4", "1.0", 'run_simulation_plasticity_vM_gc1.0_M10000_graphs.txt'),
]
# eps_f: strain at which R_x has collapsed below 10 % of sigma* (complete fracture) -- this is what the v2.0 manuscript
# reported as eps* (1.18 / 1.08 / 0.82)
print(f"{'set':16s} {'gc':>7s} {'sigma*':>8s} {'eps*':>7s} {'eps_f':>7s} {'Pi*=maxW':>9s} {'W(eps*)':>8s}")
peaks = {}
for label, gc_str, fname in runs:
    if not os.path.exists(os.path.join(script_path, fname)):
        print(f"{label:16s} missing {fname}")
        continue
    d = load(fname)
    i = d[1].idxmax()
    eps_f = d[0][i + int(np.argmax(d[1][i:].values < 0.1 * d[1][i]))]
    peaks[label] = (d[1][i], d[0][i], d[2].max(), d[2][i], eps_f)
    print(f"{label:16s} {gc_str:>7s} {d[1][i]:8.4f} {d[0][i]:7.4f} {eps_f:7.4f} {d[2].max():9.4f} {d[2][i]:8.4f}")
for label in ("[vM] rate 0.5", "[vM] M = 1e4"):
    if label in peaks and "[vM]" in peaks:
        ref = peaks["[vM]"]
        print(f"{label}: d sigma* = {100 * (peaks[label][0] / ref[0] - 1):+.2f} %, d eps* = {100 * (peaks[label][1] / ref[1] - 1):+.2f} %")

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
    r"\textbf{[vM]} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_plasticity:.2f}" + r"$\sigma_y$)"
)

label_gc1 = (
    r"\textbf{[Eq$\mathbf{J_c}$]} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_gc1:.2f}" + r"$\sigma_y$)"
)

label_gc_scaled = (
    r"\textbf{[Eq$\mathbf{\sigma^*}$]} "
    r"(max $\sigma^*$ $\approx$ " + f"{max_gc_scaled:.2f}" + r"$\sigma_y$)"
)

# --------------------------------------------------
# Labels (energy)
# --------------------------------------------------
label_pi_plasticity = (
    r"\textbf{[vM]} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_plasticity:.2f}" + r"$J_c^0L$)"
)

label_pi_gc1 = (
    r"\textbf{[Eq$\mathbf{J_c}$]} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_gc1:.2f}" + r"$J_c^0L$)"
)

label_pi_gc_scaled = (
    r"\textbf{[Eq$\mathbf{\sigma^*}$]} "
    r"(max $\Pi^*$ $\approx$ " + f"{max_pi_gc_scaled:.2f}" + r"$J_c^0L$)"
)

# --------------------------------------------------
# Output files
# --------------------------------------------------
output_file_stress = os.path.join(
    script_path, 'PAPER_reaction_forces_1D.png'
)
output_file_energy = os.path.join(
    script_path, 'PAPER_energy_vs_displacement_1D.png'
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
    vlines=[[peaks[k][4] for k in ('[vM]', '[Eq Jc]', '[Eq sigma*]')]],  # eps_f (stress collapse) as in v2.0; peaks marked by mark_peak
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
