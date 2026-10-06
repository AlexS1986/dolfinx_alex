# Revision proposal, phase 1 (proposal only, no edits) — A03, 2026-10-02

Scope: manuscript `68c3b8d0b7dca7b64b8b7a93/main_revised_template.tex` (local HEAD `d00482b`) and
`response_to_reviewers_template.tex`, after campaign `260930_M1e5`. Sources for every number:
`plots/energy_consistent/energy_consistent_summary.md` (tables "M = 100000", first peak, total of both
strips) and `MOBILITY_STUDY_FINDINGS.md` (§ numbers below refer to that file). Nothing has been edited.

Preliminaries
- `git pull` in the Overleaf folder fails from the Cowork shell (proxy 403, as documented in CLAUDE.md).
  **Alex: please pull before phase 2.** Everything below is based on the local state (`d00482b`,
  uncommitted build files only).
- Two CLAUDE.md versions exist: the compendium root (`../CLAUDE.md`, header "Status as of 2026-09-01",
  contains the 2026-09-30/10-02 campaign notes, H-S "decision pending") and the project copy in the
  Claude project (header 2026-09-29, contains the 2026-09-02 cost-reinterpretation decision, no campaign
  notes). The prompt says the H-S decision is still pending, so this proposal treats the physical
  interpretation of φ as open and marks every place where it enters (tag **[DEPENDS ON φ DECISION]**).
  Wording below uses neutral placeholders: "graded structures" for \Evar, "reference with the
  softer/stiffer material grade" for \Emin/\Emax.

---------------------------------------------------------------------------------------------------

## A. Story check

### A.1 What the submitted text claims and what the new data say

| claim in the submitted/current text | where | status at M = 10⁵, first peak, total reaction |
|---|---|---|
| \Evar "comparable to or even improved over the fixed-porosity references for selected measures" | abstract | **not supported as stated**: at ρ = 0.3 \Evar is 26–32 % below \Emax (work at first peak −50 to −57 %); at ρ = 0.6 it is 19–21 % above \Emax (+42 to +52 %); vs \Emin at ρ = 0.3 it is +82 to +92 % (+219 to +260 %). The sign depends on the budget. |
| "max R_y of \Evar is about 36–64 % higher than \Emin" | Sec. 3.2 (peak metrics), conclusion to-do | **withdrawn, replaced by +82 to +92 %** (§12). |
| "For ρ = 0.3, \Evar is again comparable to \Emax" | Sec. 3.2, response curves | **withdrawn**: −26 to −32 %. |
| "For ρ = 0.6, \Evar comparable to, but slightly stronger than, \Emax"; "\Evar exceeds \Emax in work at peak" | Sec. 3.2 | **replaced**: +19 to +21 % in load, +42 to +52 % in work (§12) — now a clear, not a slight, margin. |
| "\Emin shows only a weak dependence on β_s" | Sec. 3.2, panel (b) discussion | **withdrawn**: all first peaks fall by ≈27 % from β_s = 0.015 to 0.06 mm, \Emin included (§10, §12). The flat \Emin trend was a rate artefact of M = 100. |
| "β_φ affects the peak reaction force only mildly"; "no strong interaction with β_s" | abstract, Sec. 3.3, conclusion | **holds at ρ = 0.3 only** (three β_φ within 2 %, β_φ = 0.001 up to 8 % lower at β_s = 0.06). **Qualified at ρ = 0.6**: β_φ = 0.05 is 7–14 % weaker than β_φ = 0.001 in the first peak, its margin over \Emax drops to +7 to +18 %; in the maximum load the difference disappears (§12, §13). |
| "variation of β_φ notably shifts the failure displacement, especially for ρ = 0.3" | Sec. 3.3 | **withdrawn for ρ = 0.3**: first-peak displacements 0.00947 / 0.00925 / 0.00928 mm (β_φ = 0.001/0.01/0.05, β_s = 0.03). At ρ = 0.6 the shift is now the β_φ = 0.05 case (0.01346 vs 0.01566/0.01530 mm). |
| "\Evar reaches maximum reaction forces similar to \Emax while having a much lower effective σ_c" (ρ = 0.3) | Sec. 3.2, Fig. max_Ry_vs_sig_c | **withdrawn at ρ = 0.3** (ordering now \Emax > \Evar > \Emin, same as the σ_c ordering). The statement "\Evar performs better than \Emax at comparable σ_c" **holds at ρ = 0.6**. Figure must be regenerated before the text is rewritten. |
| "In every simulation this peak is attained before the eventual failure of the Newton iteration"; "post-peak branch not captured reliably" (added 2026-09-29) | Sec. 3.2, conclusion | **superseded**: all 36 runs reach the common end point u_y = 0.03 mm (one wall-limit stop at 0.0284 mm), no solver stops (§12). |
| "hinting towards a more efficient use of the material even with regard to failure" | Sec. 3.2 (sample edit already deletes it) | **withdrawn** (reviewer 1 objected anyway; the new numbers remove the basis at ρ = 0.3). |
| stiffness gain 2.5 % / 8 % | Sec. 3.1 | unchanged (TO result). |
| crack patterns: "\Evar more widespread damage", failure at fixed boundaries for \Emin/\Emax, 45° members for \Evar | Sec. 3.2, 3.3 | **unknown until the s-fields of the M = 10⁵ runs are plotted** (the published overviews are M = 100 final states at an earlier termination point; new final states are at u_y = 0.03 mm). Text to be re-checked after the figures exist (work plan step 3). |

### A.2 Is the new story consistent and defensible?

Yes, and it is a better paper than the submitted one, for three reasons.

1. The result is now physically readable. The optimizer maximizes stiffness per budget. At the tight budget
   (ρ = 0.3) it spreads the cheaper, softer grade (the one with the lowest fracture resistance in the G_c
   mapping) over large parts of the load path, because that grade gives more stiffness per cost unit; the
   stiff-grade reference \Emax is a thinner truss made entirely of the toughest material. The graded
   structure is 2.5 % stiffer but its first crack event occurs at u_y = 0.00925 mm instead of 0.01456 mm
   (β_s = 0.03), i.e. at 30 % lower load. At the generous budget (ρ = 0.6) the optimizer can place the stiff,
   tough grade where tensile stresses act and use the soft grade as filler; the result is both 8 % stiffer and
   19–21 % stronger than \Emax. The sign of the fracture effect of stiffness-driven grading depends on the
   budget and cannot be read off the compliance. **[DEPENDS ON φ DECISION]** for the words "grade/cost", not
   for the mechanism.
2. This strengthens, not weakens, the paper's identity (sequential design-and-assessment): the point of a
   post-optimization fracture assessment is exactly that it can reverse the ranking suggested by stiffness.
   The last conclusion paragraph ("stiffness-based TO should be followed by an explicit failure assessment")
   now has a quantitative example: a 2.5 % stiffness gain bought with a 30 % loss in first-peak load.
3. The reviewers asked for accurately reported results and tempered conclusions (editor), for the mobility
   contribution (R1) and for quantitative highlights (R2). The revised numbers answer all three directly.

What the story no longer supports and must not be claimed anywhere: that graded material is "comparable to
the solid/stiff reference" in general; that grading "uses material more efficiently with regard to failure";
that \Emin is insensitive to β_s; that β_φ is unimportant for the peak at ρ = 0.6.

What stays true and can be emphasized: \Evar vs \Emin is a robust, large advantage at every β_s, β_φ and M
(the ranking never changed); the ρ = 0.6 advantage over \Emax is robust in all three measures (first peak
+19–21 %, maximum load +28–33 %, work to 0.03 mm +30–43 %, §13); every first peak scales with β_s as expected
from σ_c ∝ β_s^(−1/2) (≈ −27 % over the range, consistent with (0.015/0.06)^(1/2) = 0.5 only in trend, not in
magnitude — say "decreases monotonically", not "scales with").

### A.3 Honest abstract-level statement (one paragraph, neutral wording)

> Within the investigated configuration, the fracture assessment shows that the stiffness gain obtained by
> grading the material does not translate into fracture strength in a uniform way. At the lower budget
> (ρ = 0.3) the graded structures reach a first-peak load 82 to 92 % above that of the reference built from the
> softer material grade, but 26 to 32 % below that of the reference built from the stiffer grade, because the
> stiffness-optimal use of the softer grade places material of low fracture resistance in the load path. At
> the higher budget (ρ = 0.6) the graded structures exceed the stiffer-grade reference by 19 to 21 % in
> first-peak load and by 42 to 52 % in the work up to it. The graded structures develop more distributed
> damage [to be confirmed from the new s-fields] and keep carrying load after the first crack event. The
> length scale of the material gradient has little influence on the first-peak load at ρ = 0.3 and reduces it
> by up to 14 % at ρ = 0.6 for the coarsest gradient; the phase-field length scale lowers all first peaks
> monotonically, by about 27 % over the investigated range, without changing the ranking of the structures.
> These results indicate that a stiffness-driven distribution of material grades must be followed by an
> explicit fracture assessment, since the sign of its effect on structural strength depends on the budget.

**→ Question 1 (see section E): do you agree with this story before any wording is drafted?**

---------------------------------------------------------------------------------------------------

## B. Change proposal for the manuscript (section by section)

Legend: **[T]** text only, **[F]** needs a new or regenerated figure, **[Tab]** new table, **[C]** compute
first (number not yet in a summary file), **[H]** coordinate with Hannover.

### B.1 Section 2.2 "A phase field model of brittle fracture" (model and implementation)

B.1.1 Mobility paragraph (the \replaced{} block of 2026-09-29, lines ≈ 985–1005) **[T]**
- old: "The mobility is chosen as M = 100 mm³/(Nmm s). The energy dissipated by this regularization, D_s …, is
  monitored in all simulations and its magnitude relative to the fracture energy and to the external work is
  reported in Section 3."
- new: "The mobility is chosen as M = 10⁵ mm³/(Nmm s). This value follows from a convergence study in M
  (Section 3.2 / Fig. [mobility convergence]): the first-peak loads of the investigated structures decrease
  monotonically with M and follow R(M) ≈ R_∞ + c M^(−2/3) for large M; at M = 10⁵ they lie within about 1 % of
  the extrapolated rate-independent limit, and the energy dissipated by the rate term up to the first peak is
  below 0.05 % of the external work. Since inertia is neglected, the response as a function of the imposed
  displacement depends on M and v₀ only through the ratio M/v₀, so that M = 10⁵ at v₀ = 1 mm/s is equivalent to
  M = 100 at a loading rate of 10⁻³ mm/s. The dissipation D_s (Eq. phase_field_dissipation) is monitored in
  all simulations and reported in Section 3.2."
  Sources: §1, §8.2, §10 (M^(−2/3) fit on ρ = 0.3; E_max converged at 10⁴ to 0.1 %; E_min β_s = 0.06 is the
  worst case: 10⁴ → 10⁵ −3.8 %, 10⁵ ≈ 1 % above the limit; D_s/W at peak 5·10⁻⁶–5·10⁻⁴).
  Caveat for the wording: the M^(−2/3) law is verified for ρ = 0.3 and for E_min at β_s = 0.06; ρ = 0.6 \Evar
  "does not follow this law" in §8.2 but its 10⁴ → 10⁵ step is −1.1 %. Write "for the cases investigated" and
  show the data in the figure instead of claiming the law universally.

B.1.2 Time stepping and solver paragraph (lines ≈ 1041–1060) **[T]**
- old: "… starting with a maximum time increment of Δt = 0.001 s: if the Newton iteration does not converge …
  Δt is reduced by a factor of two … All simulations are continued until Newton convergence can no longer be
  obtained under further time-step reduction, i.e., when Δt < 10⁻¹⁴ s."
- new: keep Δt_max = 0.001 s and the halving rule; add "The Newton iteration is terminated when the Euclidean
  norm of the residual falls below 10⁻⁸ or after 20 iterations, in which case the increment is repeated with
  the halved time step; the smallest admissible time step is 10⁻¹⁰ s. All simulations are run to a common end
  displacement u_y = 0.03 mm, which is about twice the largest first-peak displacement of all cases, so that the
  response after the first crack event is available for every structure." (§11, §12; the first peak is
  unchanged by the tolerance to six digits, §11 variant D.) State units of the residual norm as in the code
  (check `alex/solution.py` before writing). Do not mention the previous tolerance here (that belongs to the
  response letter, see C.1).

B.1.3 Irreversibility threshold: already written (s_irr = 10⁻³) **[T, done]** — no change, but the response
bullet should note that it applies unchanged to the recomputed runs.

B.1.4 Limitations paragraph, item (iii) **[T]**
- old: "The rate regularization introduces the artificial dissipation D_s, whose magnitude is monitored but
  which is not a physical material property."
- new: add "At the chosen mobility its contribution up to the first peak is below 0.05 % of the external
  work; after the first crack event D_s is the energy released in unstable crack growth under displacement
  control and does not vanish with increasing M (Section 3.2)." (§3, §10)

B.1.5 Internal-length sentence ("This internal length scale needs to be small compared to the geometrical
features …", line ≈ 836) — to be tempered per R1 (already in the to-do); not affected by the new numbers.

B.1.6 G_c mapping (Sec. 2.2.1) **[DEPENDS ON φ DECISION, H]** — unchanged by the new data; its justification
paragraph is not drafted here.

### B.2 Section 3.2 "Failure behavior of topology optimized structures"

B.2.1 Loading strips (lines ≈ 1568–1582) **[T]**
- old: "The displacement is applied on two strips of width w = 0.075 mm …" + Eqs. left_strip/right_strip.
- new: keep the strip definitions; add "On the mesh with element size h = 0.01 mm the prescribed displacement
  acts on the seven nodes of each strip, i.e. on six element edges with a loaded length of 0.06 mm." (§8.4)
  Question 5: alternatively redefine w = 0.06 mm (1.97 ≤ x ≤ 2.03) and drop the 0.075 mm; the loaded node set
  is identical either way.

B.2.2 Work definition and its numerical evaluation (lines ≈ 1584–1630) **[T]**
- keep Eq. (total_boundary_work) as the definition (decision 2026-09-30).
- old (sentence + Eq. total_boundary_work_trapezoidal): "Numerically, the work increment between two stored
  states is computed directly from the stress expression by trapezoidal integration, ΔW_n = ½ ∫ [σ_n n_f +
  σ_{n−1} n_f]·(u_n − u_{n−1}) dA".
- new: "Numerically, the boundary tractions are represented by the energy-consistent nodal reactions of the
  discrete system, i.e. by the entries of the assembled residual vector at the nodes with prescribed
  displacement [cite: e.g. Hughes 2000 or Zienkiewicz–Taylor; check bib]. With the resultant R_y of these
  reactions on the two loaded strips, the work increment between two stored states follows by trapezoidal
  integration, ΔW_n = ½ (R_{y,n} + R_{y,n−1}) (u_{y,n} − u_{y,n−1}), W_n = Σ ΔW_i. The fixed boundary portions
  carry reactions but contribute no work because their displacement increment vanishes."
  Replace the sentence "Thus, the evaluation is not restricted to the two surfaces …" accordingly (it then
  reduces to the last sentence above). Add the verification: "The balance W = Π_el + Π_frac + D_s closes to
  0.2 % at the first peak and to within [x] % at the end of the simulations" **[C]** (0.2 % at peak from §12
  report; the end-state ratio for all 36 runs is not yet tabulated — §8.4/§11 give 0.96–0.998 for single runs).
- Eq. (left_strip_reaction_force), lines ≈ 1876–1890 **[T]**: redefine R_y as the magnitude of the sum of the
  vertical nodal reactions on ∂Ω_L ∪ ∂Ω_R ("total reaction force of both loaded strips"); remove "at one loaded
  strip" from the caption of the response figure. Add one sentence: "The two strips do not fail
  simultaneously; the total reaction is therefore the quantity compared below." (§8.4)

B.2.3 Strength measure: first peak (lines ≈ 1892–1920 incl. the \added post-peak paragraph) **[T]**
- old: "…the reaction force reaches a peak at the onset of the first failure event and then drops in all
  cases. In every simulation, this peak reaction force is attained before the eventual failure of the Newton
  iteration described above. Therefore, termination of the simulation does not prevent evaluation of the peak,
  and max R_y is used in the following as the measure characterizing the strength of the structure." plus the
  added paragraph "The post-peak branch … is not captured reliably for all cases …".
- new: "… the reaction force reaches a first peak at the first crack event and drops. Since both ends of the
  structure are clamped, every structure keeps carrying load after this event, the remaining members are
  reloaded, and further crack events follow at larger displacement (Fig. [response, panel a], shown up to the
  common end displacement u_y = 0.03 mm). Complete separation is not reached within this range for any
  structure. Structural strength is therefore characterized by the first peak of the total reaction force,
  R_y^(1), defined as the load at the first drop exceeding 1 % of the current value, and by the external work
  W^(1) accumulated up to it. The maximum load within 0 ≤ u_y ≤ 0.03 mm and the total work up to this end
  displacement are reported as secondary measures; they depend on the end displacement and on the clamped
  supports and are interpreted accordingly (Table [secondary])." Drop the sentence "In every simulation this
  peak … Newton iteration". Keep (shortened) the limitation list of the added paragraph, but as "the post-peak
  response is affected by (i)–(iii) of Section 2.2 and is therefore used only for the secondary measures".
  Note on \Emin and ρ = 0.6 \Evar: "For \Emin (ρ = 0.3) and \Evar (ρ = 0.6) the reaction force exceeds the first
  peak again before u_y = 0.03 mm (factors 1.41–1.50 and 1.06–1.13)" (§12 table, bracket values).

B.2.4 Quantitative statements on the response curves (lines ≈ 1911–1920) **[T]**
- old: "For ρ = 0.6, the \Evar structures are comparable to, but slightly stronger than, the \Emax structures.
  For ρ = 0.3, \Evar is again comparable to \Emax and shows a clearly larger maximum reaction force than \Emin."
- new (β_s = 0.03, β_φ = 0.01, first peak, N/mm): "For ρ = 0.3, the first peak of \Evar (97.0) lies 84 % above
  \Emin (52.6) and 31 % below \Emax (139.8); it is reached at u_y = 0.00925 mm, well before the first peak of
  \Emax at 0.01456 mm, although \Evar is the stiffer structure. For ρ = 0.6, \Evar (259.2) exceeds \Emax (213.8)
  by 21 %." Then the mechanism sentence from A.2 (**[DEPENDS ON φ DECISION]** wording). "Recall that in all
  cases the same ⟨mass/budget⟩ is employed." (comma → period, R2).

B.2.5 Work paragraph (lines ≈ 1922–1935) **[T]**: "the work evaluated at peak load shows the same trends as the
maximum reaction force" → "…as the first-peak load, with larger relative differences because the response is
nearly linear up to the first peak: +230 %, −56 %, +50 % for the three comparisons at β_s = 0.03" (summary,
M = 100000, "W at peak consistent": +229.9 / −55.6 / +50.1 %). Crosses in the figure mark the first peak.

B.2.6 Energy paragraph and Table tab:dissipation_ratios (lines ≈ 1937–1965) **[T, Tab, C]**
- Fill the table with M = 10⁵ values for the five structures at β_s = 0.03: D_s/W and D_s/(Π_frac + D_s) at the
  first peak and at u_y = 0.03 mm **[C]** (not in any summary yet; the §10 statement "D_s/W at peak 5·10⁻⁶ to
  5·10⁻⁴" is from the convergence runs). Interpretation sentence: "Up to the first peak the rate
  regularization contributes less than 0.05 % of the external work. After the first crack event D_s rises to
  the order of Π_frac; this is the energy released by unstable crack growth under displacement control, which
  the quasi-static problem cannot store and which does not vanish for larger M (Section [mobility convergence])."
  (§3, §10). Remove the \mycomment{AS}{TODO …}.
- Panel (d) sentence: "apart from small numerical deviations" → "to within 0.2 % at the first peak" (§12).

B.2.7 Peak-metrics figure discussion (lines ≈ 1985–2030) **[T, F]**
- "maximum reaction force generally decreases with increasing β_s" → "first-peak load decreases monotonically
  with β_s for all structures, by 27–28 % from β_s = 0.015 to 0.06 mm" (§12). Keep the σ_c ∝ β_s^(−1/2)
  motivation as a trend argument only.
- old: "The \Evar structures remain comparable to the corresponding homogeneous-porosity reference structures
  …; over the considered range of β_s, max R_y is about 36–64 % higher [than \Emin]."
- new: "Over the considered range of β_s, the first-peak load of \Evar is 82–92 % above \Emin and 26–32 % below
  \Emax for ρ = 0.3, and 19–21 % above \Emax for ρ = 0.6. The ranking does not change with β_s." (§12)
- old panel (b): "Compared with the force-based measure, the \Evar case performs even better … for ρ = 0.6,
  \Evar exceeds \Emax over the considered values of β_s. It is also notable that \Emin shows only a weak
  dependence on β_s."
- new: "The work up to the first peak amplifies the differences: +219 to +260 %, −50 to −57 % and +42 to +52 %
  for the three comparisons." Delete the \Emin sentence.
- Panels (c), (d) (Π_el, Π_frac at the peak): keep only if regenerated **[F]**; (d) "mainly independent of β_s"
  must be re-checked from the new data **[C]**. Proposal: reduce the figure to panels (a), (b) and move Π_el,
  Π_frac at the first peak into the new results table (B.2.10). Question 6.

B.2.8 σ_c indicator figure and paragraph (lines ≈ 2032–2070) **[T, F, C]**
- Figure must be regenerated with first-peak totals (not available in `plots/energy_consistent/`).
- Text: the claim "\Evar reaches maximum reaction forces similar to \Emax while having a much lower effective
  σ_c" is withdrawn for ρ = 0.3. Expected replacement (to be verified against the new figure): "For ρ = 0.3 the
  ranking of the three structures follows the ranking of the effective indicator, but the differences in
  first-peak load (+84 %, −31 %) are much larger than those in σ_c; for ρ = 0.6, \Evar exceeds \Emax at a lower
  effective σ_c. The indicator therefore does not determine the first-peak load alone; the spatial arrangement
  of stiffness and fracture resistance within the topology is relevant." Final wording only after the figure.

B.2.9 Crack-pattern paragraphs (lines ≈ 1805–1842) **[T, F]** — rewrite only after the M = 10⁵ s-fields are
plotted (final state at u_y = 0.03 mm and, proposed in addition, the state at the first peak). The sample
edit in the Revision Working Note (deleting "more efficient use of the material") is adopted. Cannot be
drafted now.

B.2.10 New table: first-peak measures **[Tab]** — five structures × four β_s: R_y^(1) [N/mm], u_y^(1) [mm],
W^(1) [Nmm/mm] (+ Π_el, Π_frac at the first peak if panels (c), (d) are dropped). All values are in
`energy_consistent_summary.md` (M = 100000, β_φ = 0.01) except Π_el/Π_frac **[C]**. Makes every percentage in
the text traceable; recommended.

B.2.11 New table: secondary measures **[Tab]** — maximum load and total work up to u_y = 0.03 mm, with the
ratio to the first peak, for β_s = 0.03 (or all β_s), and the three comparisons: max load +21 to +37 %, −26 to
−32 %, +28 to +33 %; W to 0.03 mm +23 to +36 %, −1 to +4 %, +30 to +43 % (§13). Caption must name the end
displacement, the clamped supports and that \Emin at β_s = 0.015 ends at 0.0284 mm. Text: "the maximum load
equals the first peak for \Emax and for \Evar at ρ = 0.3; for \Emin and for \Evar at ρ = 0.6 it is the reloading
maximum, still rising at the end displacement for several cases" (§13). Do not interpret "dissipated energy"
as toughness (§13 note).

B.2.12 New paragraph + figure: mobility convergence **[T, F]** — place at the end of Sec. 3.2 (or as a short
Sec. 3.4 "Influence of the mobility"): first-peak load vs M for the five structures at β_s = 0.03 (M = 10²,
10³, 10⁴, 10⁵ where available, §1/§10) and \Emin at β_s = 0.06; normalized by the M = 10⁵ value, because the
M ≤ 10⁴ runs have traction-based forces only (the traction/residual ratio is M-independent, §10, so the
normalized curve is exact); dashed M^(−2/3) extrapolation for ρ = 0.3. Text: "M = 100 overestimates the first
peak by about 7 % (\Emax) up to a factor of about two (\Emin, β_s = 0.06: 61.65 vs 31.44 N/mm, traction
measure); the effect is largest for structures with low fracture
resistance, in which the relaxation time β_s/(M G_c) of the rate term is comparable to the loading time."
(§1, §10). Keep it factual, no reference to the submitted version in the manuscript.

### B.3 Section 3.3 "Influence of the … length scale" (β_φ) **[T, F]**

- Peak pairs paragraph (lines ≈ 2224–2247): replace the six (u_y, R_y) pairs by the first-peak totals at
  M = 10⁵, β_s = 0.03: ρ = 0.3: (0.00947, 94.7), (0.00925, 97.0), (0.00928, 96.3); ρ = 0.6: (0.01566, 266.4),
  (0.01530, 259.2), (0.01346, 233.4) mm / N/mm (summary, M = 100000).
- old: "variation of β_φ notably shifts the failure displacement, especially for ρ = 0.3, but this is only
  mildly reflected in the peak reaction-force measure".
- new: "For ρ = 0.3 the three gradient scales give the same first-peak displacement and load within about
  2 % (2.4 % at β_s = 0.03); the
  finer topology obtained with β_φ = 0.001 does not fail earlier [re-check the crack patterns, B.2.9]. For
  ρ = 0.6 the coarsest gradient β_φ = 0.05 reaches its first peak at a 12 % smaller displacement and a 7–14 %
  lower load than β_φ = 0.001 for β_s ≤ 0.045 mm, so that its margin over \Emax reduces to +7 to +18 %; in the
  maximum load up to u_y = 0.03 mm this difference disappears (+28 to +40 % for all β_φ)." (§12, §13)
- Interaction sentence (lines ≈ 2249–2255): "no strong interaction between β_φ and β_s" → "For ρ = 0.3, the
  curves for the three β_φ follow the same trend in β_s, with β_φ = 0.001 falling up to 8 % below the others at
  β_s = 0.06 mm; for ρ = 0.6 the ordering β_φ = 0.001 > 0.01 > 0.05 persists over the whole β_s range, with the
  spread narrowing at β_s = 0.06 mm (+26 / +19 / +18 %). Within the investigated configuration, the gradient
  scale and the fracture length scale therefore act largely independently on the first-peak load." Add the
  R1 caveat (φ fields clustered at the bounds limit this conclusion) — already in the to-do.
- Figures: replace both β-comparison plots by M = 10⁵ versions (B.5).

### B.4 Abstract, introduction, conclusion **[T]** (general sections; **[DEPENDS ON φ DECISION]** for wording)

- Abstract: replace the three sentences from "Within the investigated configuration …" to "… structural
  strength measure" by A.3 (condensed to 4–5 sentences).
- Introduction: no quantitative claims found (lines 297–530 checked); only the φ wording.
- Conclusion, paragraph 2: "Structural strength was characterized by the largest vertical reaction force …
  This peak was obtained before the nonlinear solution procedure eventually failed …" → "… by the first peak of
  the total reaction force, i.e. the load at the first crack event, and by the work up to it; all simulations
  were continued to a common end displacement, and the maximum load and total work within this range serve as
  secondary measures."
- Conclusion, paragraph 3: "Their peak load and external work … are comparable to, and in several comparisons
  greater than, those of the homogeneous-porosity references, while at the same time reaching higher
  structural stiffness" → the two-budget statement of A.3 with the numbers (+82–92 / −26–32 / +19–21 %; 2.5 /
  8 % stiffness). Keep "cannot be explained by a volume-averaged one-dimensional crack-nucleation indicator
  alone" only in the form of B.2.8.
- Conclusion, paragraph 4 (β_φ): "influence on the magnitude of the peak load is comparatively mild" →
  "negligible at the lower budget and up to 14 % at the higher budget for the coarsest gradient"; the
  "no pronounced interaction" sentence → "largely independent" as in B.3.
- Conclusion, paragraph 5: keep; add the sentence "In the present example a stiffness gain of 2.5 % was
  accompanied by a 26–32 % lower first-peak load relative to the stiffer-grade reference, whereas a gain of
  8 % came with a 19–21 % higher one." This is the R2 "quantitative highlights" item.
- Title: unchanged by this (open, Hannover wording).

### B.5 Figures: replacement map (`plots/energy_consistent/` → `Images_A02/`)

| manuscript figure (label) | current file in Images_A02 | replacement | status |
|---|---|---|---|
| fig:spectral_a6_eps003_response_energy_grid (4 panels) | Response_energy_grid_vs_uy_spectral_a_6_eps0_03.png | new 4-panel grid from campaign histories: (a) total R_y to 0.03 mm, (b) W, (c) Π_el, Π_frac, D_s, (d) W vs Π_tot; crosses at the first peak | **[F] script**: panel (a) exists as `Ry_total_vs_uy_beta0_01_eps0_03_M100000.png` but needs legend cleanup ("energy-consistent (to peak)/traction integral" entries, "max R_y" → "first peak"); (b)–(d) need `14_energy_consistent_metrics.py` extension |
| fig:spectral_a6_peak_metrics_vs_epsilon (4 panels) | Peak_metrics_grid_vs_epsilon_spectral_a_6.png | (a) `max_Ry_total_vs_beta_s_beta0_01_M100000.pdf`, (b) `W_at_peak_vs_beta_s_beta0_01_M100000.pdf` exist; (c), (d) Π_el/Π_frac at first peak need generation, or drop (Question 6) | **[F]** |
| fig:spectral_a6_max_Ry_vs_sig_c | max_Ry_vs_sig_c_spectral.png | regenerate with first-peak totals | **[F] script** |
| fig:spectral_a6_s_overview_eps0015 … eps006 (4 figs) | phasefield_s_overview-1…4.png | regenerate from campaign `results_*.xdmf` (external disk, 7.2 GB): final state u_y = 0.03 mm; proposed additional row/figure: state at the first peak | **[F] script** (`08_plot_phasefield_overview.py` on the campaign root; needs the Docker/venv stack, not the Cowork shell) |
| fig:spectral_a6_sig_vol_overview, sig_dev_overview | phasefield_sig_vol/dev_overview.png | unchanged in content (t = 0.003 s is elastic and M-independent, §1); to be replaced by principal tensile stress + direction (R1) from the new xdmf for consistency | **[F]** (R1 item, not caused by the new numbers) |
| fig:spectral_a6_E/gc/sigma_c_overview, fig:rho_omega_constraint, fig:beta005_E/gc/sigma_c | — | unchanged (inputs) | — |
| fig:beta005_s_overview_eps003 | phasefield_s_overview_beta005.png | regenerate from campaign (β_φ = 0.001/0.05 runs) | **[F] script** |
| fig:beta_comparison_Ry_vs_uy | Beta_comparison_Ry_vs_uy_spectral_a_6_eps0_03.png | new: \Evar curves for the three β_φ, both ρ, M = 10⁵, to 0.03 mm, first peaks marked | **[F] script** (the existing `Ry_total_vs_uy_beta0_0xx_M100000` plots are per β_φ with all structures) |
| fig:beta_comparison_max_Ry_vs_epsilon | Beta_comparison_max_Ry_vs_epsilon_spectral_a_6.png | new: \Evar first peak vs β_s for the three β_φ, with \Emax reference | **[F] script** |
| NEW fig: mobility convergence | — | first-peak load/first peak at 10⁵ vs M (log), five structures β_s = 0.03 + \Emin β_s = 0.06, M^(−2/3) fit | **[F] script** (`12_mobility_study_eval.py` + `mobility_convergence_260930/` histories) |
| NEW fig: crack evolution snapshots (R2) | — | \Evar ρ = 0.3 and 0.6, β_s = 0.03, β_φ = 0.01: s at first peak, after the first drop, at u_y = 0.03 mm (3 × 2 panels) | **[F] script** (campaign xdmf) |
| Table tab:dissipation_ratios | — | D_s ratios at M = 10⁵ (B.2.6) | **[Tab, C]** |
| NEW Table first-peak measures (B.2.10), NEW Table secondary measures (B.2.11) | — | from `energy_consistent_summary.md` + `energy_consistent_runs.csv` | **[Tab]** |

Captions to change in any case **[T]**: response figure (a) "reaction force R_y at one loaded strip" → "total
reaction force of both loaded strips"; peak-metrics (b) "total-boundary external work, defined in Eq.
(total_boundary_work)" → "external work up to the first peak, Eq. (total_boundary_work) evaluated from the
nodal reactions (Section 3.2)" — this also resolves the column-4/column-13 inconsistency (§9); all "maximum
reaction force" → "first-peak load R_y^(1)"; all "porosity" in captions **[DEPENDS ON φ DECISION]**.
`manuscript_picture_list.txt` gets a new block "Campaign 260930_M1e5 sources".

---------------------------------------------------------------------------------------------------

## C. Change proposal for the response to reviewers (arguments, not final text)

### C.0 Opening statement (before the point-by-point part; also answers the editor's "results accurately
reported, overstated conclusions rewritten")

Argument: all fracture results were recomputed and the quantitative statements changed. Three
methodological refinements, presented as what they are:
(i) The reviewer's question on the mobility term led to a convergence study in M. It showed that the first-peak
loads of the structures with low fracture resistance were still rate-affected at the submitted M = 100 (the
relaxation time of the rate term was comparable to the loading time there). The study was therefore repeated
at M = 10⁵, within about 1 % of the rate-independent limit, with the M^(−2/3) convergence documented in a new
figure. (Direct, positive answer to R1.)
(ii) In the revised evaluation the reaction forces and the external work are obtained from the
energy-consistent nodal reactions of the discrete system instead of a surface integral of the interpolated
stress over the loaded strips. This closes the energy balance W = Π_el + Π_frac + D_s to 0.2 % at the peak and
affects the absolute values and the structure-to-structure ratios. The reaction is now the total of both
loaded strips.
(iii) Time stepping and solver tolerances were adapted so that every simulation resolves the unstable crack
growth up to a common end displacement u_y = 0.03 mm, which makes the full post-peak response and the secondary
measures available.
Then the list of withdrawn/replaced statements (A.1, the first eight rows) in two or three sentences, with
"indicates" and "within the investigated configuration". No "error/bug/mistake/wrong/incorrect", no apology,
no claim that the submitted results were qualitatively right where they were not (ρ = 0.3 \Evar vs \Emax, the
\Emin trend, β_φ at ρ = 0.6), no claim that M = 100 was negligible.

### C.1 Reviewer 1, mobility / artificial damping (template lines ≈ 311–350; current draft to be replaced)

Argument: (a) the term is a rate regularization with the quasi-static limit M → ∞ (keep from the current
draft); (b) the convergence study: first-peak load vs M, M^(−2/3), M = 10⁵ within ≈1 %, equivalence M/v₀;
(c) the portion of dissipation: up to the first peak D_s/W < 0.05 % and D_s/(Π_frac + D_s) [value from
B.2.6], i.e. the regularization is energetically negligible before the first crack event; after it, D_s is of
the order of Π_frac — this part is the energy released by unstable crack growth under displacement control
(snap-through), it does not vanish with M and is a property of the quasi-static problem, not of the
regularization (§3, §10); (d) physical motivation: none claimed, the term is numerical (as in the manuscript);
(e) staggered scheme: not used; the monolithic scheme with rate regularization is the implementation's
strategy; the regularization is what makes the monolithic Newton iteration converge through the crack
events, and the convergence study shows that it does not alter the results at the chosen M. Related changes:
B.1.1, B.1.4, B.2.6, B.2.12, new figure, all recomputed results.

### C.2 Reviewer 1, Fig. 18a / vanishing reaction forces / failure in bending-dominated regions (lines ≈ 353–380)

Argument changes: the simulations now run to a common end displacement; the reaction forces do not vanish
because both ends are clamped and the remaining members are reloaded (the structure is statically
indeterminate), with further partial crack events (new full curves, Fig. response (a)); complete separation is
not reached within 0.03 mm. For this reason the primary measure is the first peak (first crack event), with
the maximum load and total work up to 0.03 mm as explicitly labelled secondary measures. The diffuse-zone
observation in bending-dominated members is acknowledged as before (spectral split, β_s/member width); the
rate-regularization part of that explanation is now reduced by the larger M. Related changes: B.2.3, B.2.11,
limitations paragraph, new s-field figures (after B.2.9).

### C.3 Reviewer 1, "more efficient use of the material" (lines ≈ 382–403, currently TODO)

Argument: the statement is withdrawn. The recomputed results show that the graded structures are 26–32 %
below the stiffer-grade reference in first-peak load at ρ = 0.3 and 19–21 % above it at ρ = 0.6; the damage
patterns are described without an efficiency interpretation, and the width of the damaged zones is discussed
with the limitations named by the reviewer (spectral split, regularization length; the mobility contribution
is now small before the peak). Related changes: B.2.9 text (after the new figures), abstract, conclusion.

### C.4 Reviewer 1, principal stresses (editor block, lines ≈ 102–136; and TO block "Stress Measures")

Not changed by the new numbers; the new plots can be produced from the recomputed runs at t = 0.003 s (elastic
state, identical for all M). Argument as planned (replace or supplement σ_vol/σ_dev by σ_1 and its direction;
relate to the driving force ψ_e⁺). Related change: B.5 row sig_vol/sig_dev.

### C.5 Reviewer 1, mesh details (TO block, lines ≈ 527–540)

Add to the planned mesh table (per Ω_f: number of triangles, min/mean/max edge length, h/β_s, TO mesh
h = 0.01 mm) the loaded-strip discretization: seven nodes / six edges per strip, 0.06 mm (B.2.1), and the
statement that the reaction forces are evaluated from the nodal reactions, which makes them independent of the
stress interpolation on the strip facets. (Do not discuss the old traction integral's mesh dependence.)

### C.6 Reviewer 1, fat cracks / non-local damage (lines ≈ 469–497)

Unchanged argument; add that the rate contribution to the crack-bridging effect is now bounded by the
convergence study (first peaks within ≈1 % of the rate-independent limit), so the remaining width effects are
those of β_s and the spectral split.

### C.7 Reviewer 2, crack-evolution snapshots (lines ≈ 642–655)

Argument: new figure with the phase field at the first peak, after the first load drop and at u_y = 0.03 mm for
\Evar at ρ = 0.3 and 0.6 (β_s = 0.03); describe where the first event occurs (to be read from the figure — the
reviewer assumes the fixed boundaries) and how the pattern develops; cross-reference the full response curves.
Related changes: B.5 new figure, B.2.9 text.

### C.8 Reviewer 2, quantitative highlights in the conclusion (lines ≈ 671–684)

Argument: done, with the recomputed numbers (A.3 / B.4); state explicitly that the numbers differ from the
submitted version and refer to C.0 for the reasons.

### C.9 Reviewer 2, "why not fracture in the optimization" and E_min(0.3) ≈ E_max(0.6) similarity

Unchanged by the new results (text-only items as planned). The post-peak reloading observation (clamped
supports) can be mentioned in the first as an example of path dependence.

### C.10 Items untouched by the new work

R1: distinction from Cheng/Jansen, G_c mapping validation (**[DEPENDS ON φ DECISION]**), functional/Larsen,
spectral split, κ_s, irreversibility (done), Fig. 4 schematic (**[H]**), literature comparison (**[H]**),
intermediate densities (**[H]**), φ clusters (partly: the β_φ conclusion is now qualified, C.1-adjacent);
R2 minor items; title.

---------------------------------------------------------------------------------------------------

## D. Work plan after the green light (ordered; effort in working days for Alex + Claude Code)

0. Alex: `git pull` Overleaf; confirm the φ-wording status with Hannover (their target ~8 Oct); send the
   co-authors (Müller first) the one-paragraph story (A.3) and the change table (A.1) — they should agree to
   the changed message before the text is written. **[H]**, 0.5 d calendar, blocks nothing below except B.4.
1. Evaluation scripts (`14_energy_consistent_metrics.py`, new `15_manuscript_figures_M1e5.py` or extension):
   4-panel response/energy grid, peak-metrics (a)–(d) or (a)–(b), σ_c figure, β_φ comparison plots, first-peak
   table (CSV/LaTeX), secondary-measures table, D_s ratio table, energy-balance-at-end column. Files:
   `plots/energy_consistent/`, `results/campaign_260930_M1e5/`. 1 d. Needs the Python stack (venv/Docker; not
   the Cowork shell).
2. Mobility-convergence figure: extend `12_mobility_study_eval.py` with `mobility_convergence_260930/` and the
   campaign M = 10⁵ values; normalized plot + M^(−2/3) fit. 0.5 d.
3. Field plots from the campaign xdmf (external disk): `08_plot_phasefield_overview.py` with the campaign
   root → s-overviews (final + first peak) for β_φ = 0.01 (4 β_s) and the β_φ comparison; crack-evolution
   snapshots; principal stress σ_1 + direction at t = 0.003 s (R1 item). 1 d (I/O-bound, 7.2 GB).
   → then re-check the crack-pattern text (B.2.9) and the "first onset" statement for C.7.
4. Copy curated figures to `Images_A02/` (new names, keep the old files until the final build), update
   `manuscript_picture_list.txt`. 0.25 d.
5. Manuscript text, with `changes` markup, in `main_revised_template.tex`: B.1 (Sec. 2.2), B.2 (Sec. 3.2 incl.
   tables), B.3 (Sec. 3.3), B.4 (abstract, conclusion; φ wording per Hannover). Order: 2.2 → 3.2 → 3.3 →
   conclusion → abstract. 1.5 d. Rule: every number copied from the summary files, no semicolon constructions,
   "indicates"/"within the investigated configuration".
6. Response letter: C.0 opening, C.1–C.9 blocks, "Related changes" bullets for every edit of step 5. 1 d.
7. Verification pass by a separate agent: every percentage in the .tex against `energy_consistent_summary.md`
   / FINDINGS; figure–caption–text consistency (first peak vs maximum); build `main_revised_template.tex` and
   the response (latexmk, colored markup check PDFs in `review_feedback/`). 0.5 d.
8. Compendium: `CAMPAIGNS.csv`, `README.md`, `DATA_DICTIONARY.md` (columns 14–17), `MANIFEST.csv`,
   `SHA256SUMS`; mark `results/new_W_whole_boundary/` as superseded for the fracture metrics; `CLAUDE.md`
   §4 "Key facts" (M, tolerances, first-peak definition, new headline numbers). 0.5 d. Can wait until after
   submission except CLAUDE.md.
Total ≈ 6 working days of Claude Code/Alex time + Hannover's TO part in parallel; 13 calendar days remain.
Critical path: step 3 (figures from xdmf) → B.2.9/C.7 text; step 0 (co-author agreement) → abstract/conclusion.
Coordination with Hannover **[H]**: Sec. 2.1/3.1, Fig. 4, mesh data, literature, intermediate densities, φ
clusters, the φ/ρ wording; and they must see the new numbers before they write the TO-side motivation.

---------------------------------------------------------------------------------------------------

## E. Questions (story first)

1. **Story.** Do you agree with A.2/A.3: "stiffness-driven grading can reduce (ρ = 0.3: −26 to −32 % vs
   \Emax) or increase (ρ = 0.6: +19 to +21 %) the fracture strength, depending on the budget; the sequential
   assessment is needed because the sign is not predictable from the compliance"? Only then I draft wording.
2. **φ interpretation.** Is the 2026-09-02 cost reinterpretation (project CLAUDE.md) in force, so that the
   neutral placeholders become "material grade/cost", or is it still open? Either way, no "physical
   motivation" paragraph is drafted; but the abstract and conclusion wording in B.4 cannot be finalized
   without it.
3. **Mobility section placement.** Short Sec. 3.4 "Influence of the mobility" with the new figure, or a
   paragraph at the end of Sec. 3.2 (my preference: end of 3.2, keeps the section structure)?
4. **Secondary measures.** Table only (B.2.11) or table + a figure of max load/W to 0.03 mm vs β_s?
   My preference: table only, to keep the first peak as the visible strength measure.
5. **Strip width.** Keep w = 0.075 mm with the added sentence (seven nodes, six edges, 0.06 mm), or redefine
   the strips as 1.97 ≤ x ≤ 2.03 mm (w = 0.06 mm)? The constrained node set is identical.
6. **Peak-metrics figure.** Keep four panels (needs Π_el/Π_frac at the first peak, small script work) or
   reduce to (a) first-peak load and (b) work, with Π_el/Π_frac in the new table?
7. **s-field figures.** Final state at u_y = 0.03 mm only (as before), or an additional row with the state at
   the first peak (my recommendation: both; the first-peak state is what the primary measure refers to)?
8. **Residual-forces reference.** Which textbook citation do you want for "energy-consistent nodal reactions"
   (Hughes 2000? Zienkiewicz & Taylor? Bathe?) — none of them is in `a02.bib`/`a03.bib` as far as I checked.
9. **Co-authors.** May the A.1 table and A.3 paragraph go to Müller and Hannover as is (English), before
   phase 2 starts?

---------------------------------------------------------------------------------------------------

## F. Decisions (Alex, 2026-10-02 09:21)

1. Story A.2/A.3 agreed.
2. φ is a material grade / material cost (2026-09-02 decision in force). Placeholders become "material
   grade/cost" wording; no physical-motivation paragraph is drafted by Claude.
3. Mobility convergence as a paragraph + figure at the end of Sec. 3.2.
4. Secondary measures: table only.
5. Strip width: open (Q5 not answered).
6. Peak-metrics figure: keep four panels as in the original (regenerate Π_el, Π_frac at the first peak).
7. s-field figures: show the state at the first peak (defined point) instead of the final state, to keep the
   manuscript length; final states only in the crack-evolution snapshot figure (R2).
8. Reference for energy-consistent nodal reactions: Hughes, The Finite Element Method, Dover 2000
   (recommended; add to a03.bib).
9. A.1/A.3 are not sent to the co-authors as is.
Green light for phase 2: not yet given explicitly.
