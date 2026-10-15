# Manuscript corrections resulting from the code audit (2026-10-15)

Source: `046_plasticity/KLAIM_2025/schlueter_medda_mueller_klaim_2025.tex` (v2.0 backup:
`schlueter_medda_mueller_klaim_2025_v2.0_backup_20261015.tex`). Round-1 `\added/\deleted` markup is still in the file;
the new edits use `\replaced{old}{new}` where possible, equations were changed in place.

## Applied in the tex (2026-10-15)

| # | location | change | reason |
|---|---|---|---|
| 1 | eq. (18), ψ_fr | `(1−s)²/(2ε) + 2ε|∇s|²` → `(1−s)²/(4ε) + ε|∇s|²` | this is what the code (`alex.phasefield.surf`, `psisurf_from_function`) integrates; eqs. (19) and (26) of the manuscript already correspond to this form; the old eq. (18) was inconsistent with them by a factor 2 (R1: "mathematical consistency") |
| 2 | eq. (2), κ | `(3−ν)/(1+ν)` (plane stress) → `3 − 4ν` (plane strain) | code uses `3 − 4ν`; all 2D runs are plane strain |
| 3 | Sec. 3.3 | dolfinx 0.7.4 → 0.7.3 | verified runtime in the research-data archive |
| 4 | Sec. 4.2 and 5.1, surfing BC | "E_bc = 2.5µ0, ν_bc = 0.25" → effective elastic constants of the surrounding material (E_0^eff, ν_0^eff) as the auxiliary K-field parameters (`\replaced`) | the code (old and reruns, `--bc_moduli effective`) evaluates the K-field with µ_eff/ν_eff of the embedding; Sec. 3.5 (homogeneous strip, E_0 = 2.5µ0, ν_0 = 0.25) is correct as written. Final decision 2026-10-15. |

## To be updated after the reruns (numbers that change because β = 0.2 is now used consistently)

* Sec. 3.5: J_c^num ≈ 1.50 / 1.12 / 0.64 J_c^0 (from study S2 of `075_klaim_rev2_2D`, `evaluation_output/Jc_num_strip.csv`).
* Sec. 4.2: saturation values 1.80 / 1.9 / 1.12 J_c^num and the w_s ≤ 0.6 L statement (from `master_table.csv`).
  Note: re-evaluating the v2.0 data with the paper's own window definition gives 1.86/1.12 = 1.66 J_c^num for [vM] at
  w_s = 3 (not 1.80) — check how 1.80 was obtained before quoting any saturation value again.
* Sec. 5: 3D saturation 1.15 vs 1.25 J_c^{num,pl}; η now 10⁻⁵ as stated (old runs used 10⁻³).
* Sec. 3.6 (1-D bar, 050): already computed with β = 0.2 → numbers stay (σ*, ε*, Π*).
* All PAPER_* figures of Secs. 3.5, 4, 5 → regenerate from the new data.

## Known remaining inconsistencies (decide, not yet changed)

* Sec. 3.5 / Fig. 2: "strip of size 16L × 2L" — the computational domain of 062 is 16L × 20L (same mesh generator as
  the porous case, h0 = 20); only the picture is cropped. State the domain size correctly or crop statement.
* Sec. 3.1, β = 0.2: now true for the reruns. If any v2.0 figure is kept (e.g. 1-D bar, which used 0.2) it is consistent.
* Sec. 2, "u_z = 0 on z-planes" (3D): the code prescribes the full K-field displacement (u_x, u_y, 0) on all outer faces
  incl. the z-faces; the text could say "the surfing displacement with u_z = 0 is prescribed on the entire outer boundary".
* Sec. 4.1: "λ_eff = (C12+C21)/2, µ_eff = C33" is the isotropic approximation of a cubic (square-symmetric) stiffness;
  with the new `effective_stiffness_table.csv` the Zener ratio can be reported and the approximation justified (R1-M4).
* Fig. 3 (vM strip) in v2.0 used a synthetic continuation of the final descent (documented in the research-data README).
  The S2 rerun replaces this curve with untouched solver output.
