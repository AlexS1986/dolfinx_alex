# KLAIM 2025 — Revision 2 plan: reviewer requests → actions → simulations

Status 2026-10-15. Deadline **2026-11-01** (extended). Decision: major revision (R1), minor (R3), R2 satisfied.
Working rule (R. Müller, 2026-09-13): do all the small items, attack the big ones strategically, escalate to the
editor if R1 still blocks after this round.

## 1. Reviewer requests mapped to the kind of work they need

| # | Request | Kind of work | Simulation needed? |
|---|---|---|---|
| R1-M1 | Scholarship / literature breadth, low reference count | Text + bibliography (add Braides 1996, Cagnetti 2019, Hill 1963, Hossain-line follow-ups, PF-plasticity papers, representativity/RVE-size literature) | no |
| R1-M2 | Homogenization concept: "no homogenization of fracture toughness"; "representative" must be earned (Hill); representativity of the Hossain experiment unproven; material property must not depend on BCs; validity of PF-plasticity as macro model | Text: rewrite intro + Sec. 2 framing (operational "effective crack resistance of a numerical experiment", explicitly *not* a homogenized property; use Hill's RVE definition; replace "representative region" by "resolved porous region" / "specimen"). Cite Braides/Cagnetti (R3-C2) as the cases where homogenization *is* established and say why J2-plasticity + unstable renucleation is outside that class | no (optional probe, see T2-R4) |
| R1-M3 | No mechanism study (energy partition, sensitivity, reduced model, scaling, asymptotics) | (a) energy partition (R3-C1), (b) scaling argument for the elastic model (free), (c) small sensitivity study σ_y/H, (d) a reduced 1-D/ligament argument for the two regimes | (a) rough: no; rigorous: **yes (T1-R1)**; (c) **yes (T2-R3)** |
| R1-M4 | Embedding matrix isotropic although microstructure is cubic; consistency of embedding; corner/notch effects | (a) compute full effective C for all w_s (2D, 3D) and report Zener ratio A = 2C44/(C11−C12); (b) argue via J-path-independence in the homogeneous elastic embedding + Hossain's observation; (c) one or two reruns with cubic embedding to show insensitivity; (d) note that the embedding/porous interface is at least 1 cell from the crack path and that corners are at distance ≥ 2w_c | (a) cheap, local/1-core; (c) **yes (T1-R2)** |
| R1 minor 0–26 | Semicolons, intro structure, "eq. (1)", wording, η well-posedness, M logic, isotropy before eq. (21), discretization order, Δh ≤ ε/2 in words, "predicted" nucleation, evidence for 1-D estimate (24), figure quality (25), axis-label capitalization | Text; for minor 24 cite Kuhn 2013 / Schlüter 2018 and quote the 2D/3D numbers you already have (σ* in porous runs vs. 1-D σ*); regenerate low-quality figures (Fig. 9 → vector pgf) | no |
| R3-C1 | Mechanisms: why vM < Eq σ* for small w_s (nucleation regime) and > for large w_s (propagation regime); stress state near void wall at renucleation; **energy partition based on Eq. (14)** | Energy partition per renucleation event (jump in E_el, J_cΔA, plastic and viscous dissipation) vs. w_s; stress/α fields in the ligament just before renucleation for w_s = 0.375 and 1.0 (vM vs Eq σ*) | partition from existing tables: **no** (rough); field snapshots at the right instant: **yes (T1-R1)** |
| R3-C2 | Homogenization of surface energy exists (Braides 1996, Cagnetti 2019, Michel & Suquet 2022) | Text + refs | no |
| R3-C3 | Eq J_c defined with evaluated J_c^eff equal to vM's | Text: scaling argument — for the rate-independent linear-elastic PF model J_x,c^eff ∝ J_c, confirmed by Eq J_c vs Eq σ* (both ≈ 1.9–2.0 × own J_c^num). An elastic set with J_c^num = 1.50 J_c^0 would give J_x,c^eff ≈ 2.85 J_c^0 > vM's 2.0 J_c^0 → statement "vM lies between" becomes "vM is *below* an elastic material of equal crack resistance but far above one of equal strength" | no |
| Code/manuscript consistency (found during survey, see CLAUDE.md §6) | β (0.1 in code vs 0.2 in text), η_3D (1e-3 vs 1e-5), ψ_fr factor 2 (eq. 18), surfing-BC E_bc, dolfinx 0.7.3 | Text fixes after checking the container's `alex` version; **do not rerun sweeps for this** | no |

## 2. Simulation tiers

### Tier 0 — no cluster, start now (post-processing with existing data)
* **T0-A Energy partition (rough)** from the 14-column vM tables (10 w_s) and 12-column elastic tables (7 w_s each) in
  `KLAIM_2025/submission_ready_20260525/KLAIM_2025_research_data/scripts/...`. Quantities per renucleation event:
  ΔE_el release, J_cΔA created, ΔE_pl, residual (= viscous snap-through dissipation + degraded plastic work).
  First look (w_s = 1.0): residual 21 % of ΔW (vM) vs 72–84 % (elastic) → the brittle ligaments release far more energy
  per event than the fracture energy consumes; plasticity "pre-spends" part of it. Script/plot:
  `046_plasticity/revision2_scratch/energy_partition_feasibility_ws1.0.png`. Good enough for a qualitative
  figure; not good enough for a quantitative statement without T1-R1.
* **T0-B Scaling argument** (R3-C3, R1-M3): no computation; verify with Eq J_c vs Eq σ* data normalized by own J_c^num.
* **T0-C Effective stiffness anisotropy** (R1-M4): (i) `grep -A3 "^\[\[" $HPC_SCRATCH/046_plasticity/*/wsteg_job.out.*`
  and the same for 045 and 061 on the cluster — the 3×3 / 6×6 matrices are in the SLURM logs if scratch was not purged;
  (ii) otherwise rerun `mesh_effective_stiffness.py` + `run_effective_stiffness.py` for all w_s locally in the container
  (seconds in 2D, minutes in 3D) with a 3-line patch that writes the full matrix to `parameters.txt`.
  Deliverable: table of C11, C12, C33 (2D) / C11, C12, C44 (3D) and Zener ratio vs w_s.
* **T0-D 1-D reduced/scaling argument for the two regimes** (R3-C1, R1-M3): nucleation regime (w_s ≲ ε-scale, ligament
  ≈ a 1-D bar) → fracture governed by σ* and the dissipated Π* of the 1-D test, where vM dissipates less per *strain* than
  Eq σ*…; propagation regime → plastic zone size r_p ~ (K/σ_y)² vs w_s sets the interaction length. Write-up only.

### Tier 1 — recommended cluster reruns (submit this week)
* **T1-R1 Instrumented mechanism runs (2D).** Same physics, meshes and parameters as the paper runs; only the
  post-processing is extended. New folder `068_mechanism_runs` (copy `046/000_template`, `045/000_template` and the
  067 cluster scripts). Cases: w_s ∈ {0.375, 1.0, 3.0} × {vM, Eq J_c, Eq σ*} = 9 jobs (minimum: vM + Eq σ* at
  0.375 and 1.0 = 4 jobs). Additional outputs per step: fracture energy ∫J_c γ(s) over the porous region only,
  E_el split porous/embedding, cumulative **undegraded** plastic dissipation ∫∫σ_Y(α) α̇, cumulative viscous
  dissipation ∫∫ ṡ²/M, max σ_vM / σ_h / σ_yy in the next intact ligament; field snapshots (s, α, σ_yy, σ_h, σ_vM)
  triggered when J_x reaches a new running maximum and right after the next renucleation (not every 400 steps).
  Also write the full effective C matrix. Regression check: J_x(x_ct) must reproduce the existing curves.
  Serves R3-C1, R1-M3(a) and gives the stress-state figures.
* **T1-R2 Cubic embedding (2D vM, w_s = 1.0 and 0.375).** Replace the isotropic effective stiffness in the embedding
  cells by the full plane-strain cubic C from the unit cell (code change in `run_simulation.py`: cell-wise stiffness
  tensor instead of (λ, µ) functions; the `alex` elastic problem needs an orthotropic branch). Report ΔJ_x,c^eff
  in %. If the Zener ratio from T0-C turns out ≈ 1 (±5 %) the text argument may suffice and R2 can be dropped.
  2 jobs (optionally + 3D w_s = 1.0 if the 3D anisotropy is larger, 1 job on 64 cores).

### Tier 2 — only if queue time allows (results needed by ~Oct 25)
* **T2-R3 Sensitivity σ_y, H** (R1-M3, and the conclusion already promises it): vM at w_s = 1.0 with
  σ_y ∈ {0.75, 1.5} and H ∈ {0.11, 0.44} µ0 (4 jobs); add w_s = 0.375 for the nucleation regime if cheap (4 more).
  Deliverable: J_x,c^eff vs σ_y and vs H at fixed w_s; one paragraph + one figure.
* **T2-R4 Representativity probe** (R1-M2/round-1 point 7): vM, w_s = 1.0 with (a) initial crack shifted by w_c/2,
  (b) 5 instead of 3 pore rows. 2 jobs. Honest alternative: keep it as stated limitation (already in the text).
  Recommended only if T1 is running smoothly by Oct 20.

### Not recommended
* Re-running the whole sweeps for the β/η/ψ_fr discrepancies — fix the text to what was computed.
* A new elastic set with J_c^num matched to vM (R3 says not needed; the scaling argument answers it exactly).
* 3D reruns for energies (R3 asked for 2D porous discussion only).

## 2b. Status 2026-10-15 (evening)
* Decision: rerun **everything** (2D sweeps, strip calibration, 3D sweep) with corrected settings (β = 0.2, η = 1e-5,
  elastic embedding) plus all proposed studies. Folders `075_klaim_rev2_2D` (52 jobs) and `076_klaim_rev2_3D` (10 jobs)
  are set up with case lists, job generators, submission/cancel/download scripts and `evaluation_rev2.py`
  (tested on the v2.0 tables; reproduces J_c^num = 1.49 for the vM strip and the J_eff curves).
* Manuscript: eq. (18), eq. (2) κ, dolfinx version and the surfing-BC moduli statements corrected
  (`manuscript_corrections_from_code_audit.md`); v2.0 source backed up.
* Open: smoke test on the cluster (no dolfinx reachable from here), then submit S2 + S1 w_s = 1.0/0.375 first.

## 3. Timeline (deadline Nov 1)
* Oct 15–17: T0-A/B/C analyses; prepare `068_…` folder, patch post-processing, local smoke test (`run_one_local.sh`,
  `mpirun -np 8`, w_s = 1.0 for a few hundred steps); push to cluster `$HOME`, create jobs on `$HPC_SCRATCH`, submit T1.
* Oct 17–24: text work (intro, homogenization framing, all minors, eq. 18 fix, Braides/Cagnetti/Hill refs,
  figure quality), response letter skeleton.
* Oct 24–27: evaluate T1 (and T2 if available), new figures: energy partition vs w_s, stress state in the ligament,
  Zener table, cubic-embedding comparison.
* Oct 28–31: final manuscript, marked-up version, point-by-point response, cover letter to editor; co-author check.

## 4. Open questions for the author
1. Typical wall-clock time of one 2D vM run (32 cores) and one elastic run on Lichtenberg, and current queue situation
   — decides whether 9 or 4 T1-R1 jobs are realistic.
2. Is `$HPC_SCRATCH/046_plasticity` / `045_standard_holes_PAPER` / `061-…` still on the cluster (SLURM `.out` logs with
   the stiffness matrices, `.h5` field files)? If yes, T0-C needs no computation and some stress-field pictures can be
   extracted without T1-R1.
3. Can the external drive `MacbookExtension` be connected (full `046/048/067 results` incl. h5 files)?
4. Which `alex` git revision is in `alex-dolfinx.sif` on the cluster — needed to settle β = 0.1 vs 0.2.
5. Round-1 markup: accept all `\added/\deleted` of round 1 before starting round-2 edits (recommended), yes/no?
