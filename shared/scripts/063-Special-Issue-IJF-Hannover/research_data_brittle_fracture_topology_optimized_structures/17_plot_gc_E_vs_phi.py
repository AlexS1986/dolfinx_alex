"""Fig. 2 of the A03 revision: normalized fracture toughness G_c/G_{c,1} and
Young's modulus E/E_1 as functions of the material design variable phi.

G_c relation and parameters as used in 01_phasefield_dcb_260504_folder.py
(compute_jmax_grid_from_porosity, A=1.243657, B=3.150239, C=2.850765,
clipped to [0.1, 1]); argument (1-phi_min)(1-phi) with phi_min = 0.5.
E(phi) = (E1-E0) phi^q + E0 with E0/E1 = 0.6, q = 2 (Eq. E_phi).
Output: Images_A02/Gc_E_vs_phi.png (run from the compendium root).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B, C = 1.243657, 3.150239, 2.850765
PHI_MIN, E0_E1, Q = 0.5, 0.6, 2

# Computer Modern via matplotlib's bundled cmr10 + mathtext "cm"
# (no LaTeX/cm-super needed; same look as the usetex plots of the paper)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

phi = np.linspace(0.0, 1.0, 4001)
arg = np.clip((1.0 - PHI_MIN) * (1.0 - phi), 1e-12, None)
gc = np.clip(A - B * np.exp(-C * (np.sqrt(np.pi / (4.0 * arg)) - 1.0)), 0.1, 1.0)
e = E0_E1 + (1.0 - E0_E1) * phi**Q

c_gc, c_e = "#2a6fdb", "#d9822b"
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(phi, gc, color=c_gc, lw=2.4)
ax.plot(phi, e, color=c_e, lw=2.4)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.08)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$G_c/G_{c,\!1},\;\;E/E_1$")
ax.grid(True, color="0.88", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.175, 0.34, r"$G_c/G_{c,\!1}$", color=c_gc, fontsize=18, ha="right", va="center")
ax.text(0.80, 0.79, r"$E/E_1$", color=c_e, fontsize=18, ha="left", va="top")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gc_E_vs_phi.png") \
    if not os.path.isdir("68c3b8d0b7dca7b64b8b7a93") else "68c3b8d0b7dca7b64b8b7a93/Images_A02/Gc_E_vs_phi.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
