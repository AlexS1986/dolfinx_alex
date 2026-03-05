import numpy as np
import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# --------------------------------------------------
# Digitized circular pore data (EDIT if needed)
# --------------------------------------------------
x_data = np.array([0.5, 0.8, 1.0, 1.2, 2.0, 3.0, 4.0])  # w_s / L
y_data = np.array([0.48, 0.95, 1.05, 1.12, 1.22, 1.25, 1.26])  # Jmax / Geff

# --------------------------------------------------
# Model: exponential saturation
# --------------------------------------------------
def saturation_model(x, A, B, C):
    return A - B * np.exp(-C * x)

# --------------------------------------------------
# Fit
# --------------------------------------------------
initial_guess = [1.3, 1.0, 1.0]
params, covariance = curve_fit(saturation_model, x_data, y_data, p0=initial_guess)

A, B, C = params

print("Fitted parameters:")
print(f"A = {A:.6f}")
print(f"B = {B:.6f}")
print(f"C = {C:.6f}")

# --------------------------------------------------
# Smooth curve in w_s/L
# --------------------------------------------------
x_fit = np.linspace(min(x_data), max(x_data), 500)
y_fit = saturation_model(x_fit, A, B, C)

# --------------------------------------------------
# Porosity function
# phi = (pi/4) * L^2 / (L + w_s)^2
# Using x = w_s/L  →  phi = (pi/4) / (1 + x)^2
# --------------------------------------------------
phi_fit = (np.pi / 4.0) / (1.0 + x_fit)**2

# Jmax as function of porosity (parametric form)
J_phi = y_fit

# --------------------------------------------------
# Plotting
# --------------------------------------------------
fig = plt.figure(figsize=(10, 4))

# ---- Plot 1: Jmax vs w_s/L ----
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(x_data, y_data, label='Data')
ax1.plot(x_fit, y_fit, label='Fit')
ax1.set_xlabel(r'$w_s/L$')
ax1.set_ylabel(r'$J_{max}/G_{eff}$')
ax1.legend()
ax1.grid(True)

# Show parameters inside plot
param_text = (
    f"A = {A:.4f}\n"
    f"B = {B:.4f}\n"
    f"C = {C:.4f}"
)
ax1.text(0.05, 0.05, param_text,
         transform=ax1.transAxes,
         fontsize=10,
         verticalalignment='bottom',
         bbox=dict(boxstyle="round"))

# ---- Plot 2: Jmax vs porosity ----
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(phi_fit, J_phi)
ax2.set_xlabel(r'Porosity $\phi$')
ax2.set_ylabel(r'$J_{max}/G_{eff}$')
ax2.grid(True)

plt.tight_layout()

# --------------------------------------------------
# Save to same folder as script
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "circular_fit_with_porosity.png")

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlot saved to: {output_path}")