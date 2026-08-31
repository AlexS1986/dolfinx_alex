# AGENTS.md — 070 WAAM N=1 transition 2D plane stress

Read `README.md` here first; canonical project docs live in
`.../Meshing/Neper/data/04_anisotropy_waam/neper_pipeline/{AGENTS.md,documentation.txt}`.

Material assignment lives in ONE place: `materials_2d.py` + `config.json`
(region -> material -> per-crystal-system cubic constants; every grain then
gets its own tensor via its own Bunge rotation). Both solvers import it, so
they can never drift apart on the material law. `phase` in the XDMF is only
the crystal system (2 values) - the grain structure is `grain_id`, the
per-grain stiffness is `E_x_local_GPa`. Do not "fix" phase to show grains.

**Kurzfassung:** `report/report_kurz.md` -> `report/WAAM_N1_transition_kurzfassung.pdf`
(6 Seiten, `pandoc report_kurz.md -o WAAM_N1_transition_kurzfassung.pdf
--pdf-engine=xelatex`). Eigenstaendig lesbar, nutzt dieselben Abbildungen wie
der Langbericht. Sie enthaelt Zahlen redundant - bei jeder Ergebnisaenderung
BEIDE Dokumente nachziehen (Zahlen der Kurzfassung stammen aus Kapitel 2/3 des
Langberichts).

**Report:** `report/report.md` -> `report/WAAM_N1_transition_report.pdf`
(pandoc + xelatex via `report/build_report.sh`). Same rules as the 069 report:
source and PDF always change together; every `{#fig:label}` / `\label{tbl:...}`
must be referenced with `\ref{...}` in the running text (check with
`grep -oE '\{#fig:[a-zA-Z0-9_]+' report.md` vs `grep -oE 'ref\{fig:[a-zA-Z0-9_]+'`).
Report figures come from `report/make_report_figs.py` (inputs + material only);
the FE overview figure comes from `make_figures.py` (dolfinx output).
Always pass BOTH bmp arguments to `make_report_figs.py` (`--bmp` annotated,
`--src` raw scan) - without `--src`, panel (a) of `fig_mesh_zoom.png` silently
comes out blank. Likewise the FE-dependent figures silently drop their FE
points if `E_roi.json` / `fields_roi.npz` are missing: check `fig_bounds.png`
and `fig_mesh_zoom.png` after every regeneration.

**Deposited copy:** a snapshot of the report + the load-bearing scripts lives
in `Publications/02_WAAM_N1_Mikrostruktur/waam_n1_transition_2d/`
(same convention as `waam_elastic_anisotropy/` next to it). It is a copy, not
a second working tree: change things HERE, then re-deposit. Keep the two in
sync whenever `report.md`/the PDF change.

**Result policy (hard):** every reported number, table and figure comes from
`solve_plane_stress.py` (dolfinx). `reference_solver_numpy.py` is a
verification tool only; it writes to `verification/` and `make_figures.py`
must never read that folder or fall back to it. Do not reintroduce a fallback.
Analytic quantities computed straight from `materials_2d.py` (Voigt/Reuss
bounds, per-grain E_x, the arithmetic strip estimator) are NOT solver output
and may be reported — but label them as bounds/estimator, never as FE.

**Height study (report chapter 3):** `study_rois.py` cuts six evaluation
windows (4 stacked bands + the reference band + the full map height) out of
ONE full-map pixel->grain assignment, cached in `_fullmap_assign.npz`;
`study_bounds.py` computes bounds/estimator per window without any solver
(`study_stats.json`); `run_study.sh` runs the same cases in dolfinx;
`report/make_study_figs.py` draws the chapter's three figures and swaps the
estimator for the FE curve as soon as `E_<tag>.json` exists. Zone boundaries
for the new windows come from the microstructure (BCC half-value + onset of
coarse grains), calibrated against the marked reference band — do not
hard-code the reference boundaries for other windows.

Rules (inherited from 069 + specifics):
- dolfinx **v0.7.3 API** in `solve_plane_stress.py` (ufl.VectorElement/
  TensorElement + dlfx.fem.FunctionSpace). Script is intentionally
  **standalone** (no `alex` imports) so it can be cross-checked outside the
  container against `reference_solver_numpy.py`.
- Voigt 3D [xx,yy,zz,yz,xz,xy], 2D [xx,yy,xy], ENGINEERING shear everywhere.
- Orientation frames: npz Euler angles are TSL MAP frame (y down); FE is y-up
  -> always go through `plane_stress_crystal.FLIP_X180`. Don't "fix" this by
  negating angles.
- Any change to the crystal math must keep `python3 selftest.py` ALL PASS;
  extend the tests when adding functionality.
- Cross-check policy: after changing solver or preprocessing, run the numpy
  reference and the dolfinx solver on the same npz and compare E_*.json
  (agreement to ~1e-6 rel. for identical grids; both are Q1 on the same mesh).
- Single-crystal constants in `config.json` are literature placeholders
  shared with 069 - flag this in any reported number. The transition zone has
  its own `"trans"` entry (default = parent-phase values); s(x) is a separate,
  spatially varying scalar knob and must not be confused with the constants.
- `build_cell_tensors` runs redundantly on every MPI rank (whole grid). Fine
  at this problem size; if the grid grows, restrict it to the rank's own
  (cj, ci) before optimising anything else.
- Per-pixel grain reconstruction quality metric: mean |dRGB| vs BMP
  (currently ~17). Report it whenever preprocessing parameters change.
