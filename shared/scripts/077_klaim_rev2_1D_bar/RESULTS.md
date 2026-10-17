# 077 – 1D bar (manuscript Sec. 3.6, Figs. 5–6), revision 2 — results 2026-10-17

Scripts copied from `050-…_PAPER` (left unchanged) and given CLI arguments (`--gc --beta --mu --lam --eta --eps --mob
--sig_y --hard --tend --rate --tag`). The physics is unchanged: `StaticPhaseFieldProblem_plasticity_noll` / `StaticPhaseFieldProblem2D`
with `pf.cubic_degradation(beta)`, `psisurf_from_function`, u_y = 0 on all nodes, u_x = ∓rate·t at x = ∓1, bar [-1,1]×[0,1],
100×1 triangles (Δh = 0.02).
Changes to the elastic script compared with 050: `degrad_cubic` → `cubic_degradation(beta=0.2)`, mu 0.25 → 1, N 200 → 100, and the weak
zone |x| < 0.03 → |x| < 0.15 (the same as the plasticity script).
Run them with `./run_all_local.sh` (docker image `dolfinx_alex-alex-dolfinx`, serial, 6 runs in parallel, ≈ 10 min). Then run
`evaluation_plasticity.py` on the Mac (needs pandas, matplotlib, and the Mac TeX Live because the container's TeX has no `type1cm`):
`PYTHONPATH=<shared>/utils python3 evaluation_plasticity.py`.

Common parameters: β = 0.2, µ = 1, λ = 0, η = 1e-5, ε = 0.1, M = 1000, σ_y = 1, H = 0.2222222.
The first graph column is the end displacement u_0. With a bar of length 2L and u = ∓u_0, this equals the strain ε (L = 1, so ε is
already in units of √(J_c^0/(µ0 L))). σ = R_x (height 1).

## Results

| set | J_c/J_c^0 | σ*/σ_y (max R_x) | ε* at peak | ε_f (R_x < 0.1 σ*) | Π* = max W | W(ε* at peak) |
|---|---|---|---|---|---|---|
| [vM] | 1.0 | 1.2803 | 1.0016 | 1.1731 | 2.1188 | 1.6729 |
| [Eq J_c] | 1.0 | 1.7227 | 0.9666 | 1.0703 | 2.1931 | 1.8339 |
| [Eq σ*] | 0.5523 | 1.2812 | 0.7216 | 0.8006 | 1.2293 | 1.0230 |
| [Eq σ*] old (reference) | 0.5679 | 1.2991 | 0.7316 | 0.8117 | 1.2634 | 1.0514 |
| [vM], rate halved (Tend 8, rate 0.5) | 1.0 | 1.2801 (−0.02 %) | 1.0008 (−0.08 %) | 1.1711 | 2.1116 | 1.6677 |
| [vM], M = 10000 | 1.0 | 1.2800 (−0.02 %) | 1.0066 (+0.50 %) | 1.1682 | 2.1076 | 1.6856 |

* Homogeneous estimate (`075/bar1d_homogeneous_check.py`, β = 0.2): vM 1.2806 / ε* 1.006, Eq J_c 1.7230 / 0.968, Eq σ* (0.5523)
  1.2805 / 0.720. The FE peaks agree within 0.06 % (σ*) and 0.7 % (ε*). The 0.99·J_c weak zone makes FE Eq σ* 0.05 % stronger than vM,
  which is negligible.
* The vM peak does not depend on the rate: σ* changes by ≤ 0.02 % and ε* by ≤ 0.5 % (< 1 %).
* The 077 vM run reproduces the 050 run of 2026-05 exactly (σ* 1.2803, ε* 1.0016, max W 2.1188). Eq J_c is within 0.06 % of the
  old 050 gc1.0 run (1.7237, N = 200).

## Discrepancies with the v2.0 manuscript, explained

1. **ε\* in the manuscript is the strain where the stress collapses, not the strain at the peak.** The x axis is strain (= u_0).
   The vertical lines in the v2.0 figure (1.18 / 1.08 / 0.82) sit at the stress drop. The 050 data give 1.17 / 1.08 / 0.82 there,
   and the peaks are at 1.00 / 0.97 / 0.73. The text calls ε* the "fracture strain" and evaluates Π at ε* ≈ max W. That is only
   consistent with the collapse strain: Π at the peak would be 1.67 / 1.83 / 1.02.
2. **σ\*_vM = 1.31 matches β = 0.1** (homogeneous: 1.32), not β = 0.2 (1.28). The vM peak in the v2.0 figure was evidently produced
   with β = 0.1 or an older code version. At β = 0.2, equal strength needs J_c = 0.5523; 0.5679 gives 1.30, which is +1.5 % above vM.
3. **Π\*:** 2.16 / 2.22 / 1.29 → 2.12 / 2.19 / 1.23. vM changes with β (see 2). The Eq J_c difference (2.22 → 2.19) comes from N = 200 → 100
   and the weak-zone width in the old elastic run. Eq σ* also changes with J_c (0.5679 → 0.5523; Π ∝ J_c for elastic).
4. **Weak zone:** the manuscript says "from x = −3Δh to x = 3Δh" (= ±0.06L at Δh = 0.02L). The vM script (050 and 077) uses |x| < 0.15L
   (= ±7.5Δh). The old elastic script used ±0.03L with Δh = 0.01. The weak zone only selects where the crack forms and has no measurable
   effect on σ*. Correct the text to ±7.5Δh (= ±0.15L), or rerun with ±3Δh (one parameter in both scripts).

## Numbers to change in manuscript Sec. 3.6 (`revision_2/manuscript_revised_template.tex`, ≈ l. 759–769)

| quantity | v2.0 text | new (077, β = 0.2) |
|---|---|---|
| σ* [vM] | 1.31σ_y | **1.28σ_y** |
| σ* [Eq J_c] | 1.72σ_y | 1.72σ_y (unchanged) |
| J_c of [Eq σ*] | 0.5679J_c^0 | **0.5523J_c^0** (also Sec. 3.x parameter-set definition, 2D/3D text, CLAUDE.md §6) |
| σ* [Eq σ*] | 1.30σ_y ("almost equivalent") | **1.28σ_y** (equal to vM within 0.1 %) |
| ε* [vM] (collapse) | 1.18 | **1.17** (peak: 1.00) |
| ε* [Eq J_c] (collapse) | 1.08 | **1.07** (peak: 0.97) |
| ε* [Eq σ*] (collapse) | 0.82 | **0.80** (peak: 0.72) |
| Π* [Eq J_c] | 2.22J_c^0L | **2.19J_c^0L** |
| Π* [vM] | 2.16J_c^0L | **2.12J_c^0L** |
| Π* [Eq σ*] | 1.29J_c^0L | **1.23J_c^0L** |
| weak zone | −3Δh … 3Δh | −7.5Δh … 7.5Δh (= ±0.15L), or rerun |

Recommendation: keep ε* as the collapse ("fracture") strain, because it is consistent with Π(ε*) and Fig. 5. Define it explicitly in one
sentence (e.g. "the strain at which the stress drops to zero").
The qualitative statements stay correct: Eq J_c is the strongest, vM fails at the largest strain, and Eq σ* dissipates the least.
Only "almost equivalent fracture stress" can now become "the same fracture stress".

New figures (not yet copied to `KLAIM_2025/pics`): `PAPER_reaction_forces_1D.{png,pgf,pdf}` and `PAPER_energy_vs_displacement_1D.{png,pgf,pdf}`.
The vertical lines mark ε_f and the dots mark the peaks.
