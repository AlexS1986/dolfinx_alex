"""Plateau of J_x (per unit thickness) in the 3D solid reference runs, mean per 0.1 L bin of the crack position.
Usage (Mac): python3 solid_reference_plateau.py   (reads ./results, uses ../075_klaim_rev2_2D/evaluation_rev2.py)"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "075_klaim_rev2_2D"))
import evaluation_rev2 as ev
R = os.path.join(HERE, "results")
for n in ["S2_3D_solid_vM", "S2_3D_solid_vM_ws0.25", "S2F_3D_solid_vM_ws1.0", "S2F_3D_solid_vM_ws2.0", "S9F_3D_solid_vM_ws1.0_h07"]:
    f = os.path.join(R, n)
    if not os.path.isdir(f): print(n, "missing"); continue
    c = ev.Case(f)
    x = c.c("x_ct"); J = c.c("Jx"); off = c.x_offset
    L = 2 * off
    print(f"{n:28s} wsteg {c.wsteg} h_band {c.params.get('h_band','-')} thick {c.params.get('thickness_z')} domain +-{off:.2f}  x_ct {x[0]:.2f}..{x[-1]:.2f}  steps {len(x)}")
    # J vs crack position in bins of 0.1 L
    edges = np.linspace(-off, off, 11)
    row = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        row.append(f"{np.mean(J[m]):.3f}" if m.any() else "  -  ")
    print("   J_x mean per 0.1L bin:", " ".join(row))
    for lo, hi in [(0.3, 0.7), (0.4, 0.6)]:
        m = (x > -off + lo * L) & (x < -off + hi * L)
        print(f"   plateau {lo}-{hi} L: {np.mean(J[m]):.4f} (n={m.sum()})" if m.any() else f"   plateau {lo}-{hi}: not reached")
