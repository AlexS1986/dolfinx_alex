# Independent check, revised manuscript and response letter (2026-10-05)

This check was done by a separate agent that did not see how the revision was written. All first-peak and secondary numbers were recalculated from the 36 `result_graphs_*_M100000.txt` files.

Line numbers refer to the following files:
- MS = `main_revised_template.tex`
- RL = `response_to_reviewers_template.tex`

## High

- [ ] **H1: the 2.5 % stiffness gain at ρ = 0.3 is compared with the wrong reference.**
  - At ρ = 0.3, E_min is stiffer than E_max: its compliance is 4.0665e-7, against 4.7695e-7 for E_max.
  - E_var (3.9618e-7) is therefore 2.6 % stiffer than E_min but **17 % stiffer than E_max**.
  - The text says "2.5 % stiffer than the reference made of the stiffest grade" in three places: MS l.2825 (conclusion), MS l.2904 (closing sentence) and RL l.1056.
  - Fix: name the reference correctly. "17 % stiffer than E_max but 26–32 % lower first peak" is also the stronger message.
  - Sec. 3.1, l.1547 (Hannover) should name its reference too.

## Medium

- [ ] **M1:** the headline ranges (82–92 / 26–32 / 19–21 %) hold only for β_φ = 0.01.
  - Where: abstract and RL item 2.
  - Fix: add "for the reference gradient scale".
- [ ] **M2:** the influence of β_φ at ρ = 0.3 is described in four different ways.
  - Where: abstract l.261, Sec. 3.3, conclusion, RL l.197 and RL l.871.
  - The data show ≤ 2.4 % for β_s ≤ 0.03 mm and up to 8 % for larger β_s. Use one formulation everywhere.
- [ ] **M3: the G_c bounds are extra assumptions.**
  - The bounds 0.1 and 1.0 are not taken from S&M 2025.
  - The fit covers φ ≥ 0.30 only, so φ < 0.30 is an extrapolation.
  - The first cracks at ρ = 0.3 form exactly in this region.
  - Fix: add one sentence to the G_c paragraph (MS l.1201).
- [ ] **M4:** the σ_c paragraph says "equal to or higher at the same σ_c" (MS l.2370).
  - At ρ = 0.3 the σ_c ranges of E_var and E_max do not overlap.
  - Fix: drop "equal".
- [ ] **M5: the energy balance deviates up to 7 % after the load drops.**
  - Case: E_max, ρ = 0.3, β_s = 0.03, which is the case shown in Fig. 18(d).
  - The text only states the deviations at the first peak and at the end (MS l.1741), and says panel (d) "verifies" the balance (l.2184).
- [ ] **M6:** "act largely independently, with the exception of…" (MS l.2743) covers two of the three cases.
  - The conclusion also says the maximum load is "not affected" (l.2871), but it varies by 24–40 %.
- [ ] **M7:** "confirmed by the maximum load" (MS l.2834) holds for the load ranking only, not for the work ranking at ρ = 0.3.
- [ ] **M8:** the mechanism is stated as a fact in the abstract (l.257) and the conclusion (l.2828).
  - Fix: "which is attributed to…", and soften "cannot be inferred".
- [ ] **M9: synonyms are inconsistent.**
  - Grades: the abstract uses "softer/stiffer grade", the rest of the text "cheaper/most expensive/stiffest".
  - References: called "single-grade reference", "reference made of…" or "homogeneous material".
  - Fix: choose one pair of terms and use it throughout.
- [ ] **M10: porosity wording remains in Sec. 2.1 and 3.1** (Hannover).
  - 12 hits in Sec. 2.1 and 24 in Sec. 3.1, including the captions of Figs. 5–7.
  - Until this is changed, RL item 4 and the "captions" entry are only partly fulfilled.
- [ ] **M11: the letter promises things the manuscript does not contain.**
  - Non-local damage "named as future work" (RL l.753): not in the manuscript.
  - Spectral split "important direction for future work" (RL l.422): not in the outlook.
  - Bridging "bounded by the convergence study" (RL l.757): the study covers only the first peak.
- [ ] **M12:** RL l.194 says the withdrawn statement concerned the peak load of E_min.
  - In the submitted version it was about the work up to the peak. Correct the letter.
- [ ] **M13:** strong form, Eq. (27), MS l.1041: with a varying G_c the term must read 2β_s ∇·(G_c∇s), not G_c·2β_s Δs. The ∇G_c·∇s term is missing.
- [ ] **M14:** the abstract has about 360 words. Springer abstracts are typically 150–250 words.
- [ ] **M15:** "first crack in an interior member at ρ = 0.6" (conclusion l.2852).
  - For β_φ = 0.001 there is also a simultaneous crack at the clamped edge (Sec. 3.3, Fig. 25).
- [ ] **M16: not verifiable from the data supplied to the agent.**
  - Mesh statistics.
  - s_irr = 1e-3.
  - The Newton relative criterion.
  - "0.3 mm away".
  - A/B/C in the submitted Fig. 2.
  - "First peaks not affected by the solver change": confirmed for one case only.

## Low

- [ ] **L1:** MS l.929 reads "Here, where Π_frac…". Delete "where".
- [ ] **L2: semicolons in new text.**
  - MS l.1775.
  - Caption of Fig. 21 (l.2501).
  - Caption of Fig. 18 (l.2037–2046).
  - RL l.239.
- [ ] **L3:** MS l.2395, sentence on max load = first peak is ambiguous.
  - Suggested: "For E_max at both budgets (except ρ = 0.6, β_s = 0.06 mm) and for E_var at ρ = 0.3 …".
- [ ] **L4:** MS l.2326 says Π_frac "does not decrease with β_s". It is not monotonic.
  - Suggested: "is 25–38 % larger at 0.06 than at 0.015 mm".
- [ ] **L5:** MS l.522–527 is a run-on sentence (gap paragraph). Split it.
- [ ] **L6:** the roadmap sentence (MS l.577) does not mention β_φ or the new Sec. 3.2.1/3.2.2.
- [ ] **L7: crack-event displacements.**
  - 0.0188 → 0.0186 mm (MS l.2013).
  - "0.0191 for both edges" → 0.0189 and 0.0191 mm (MS l.2017).
- [ ] **L8:** the threshold reads "below 0.1 %" in Sec. 2.2 and "below 0.06 %" in Sec. 3.2. Harmonize.
  - Also change "approach … as M^{-2/3}" to "are consistent with".
- [ ] **L9:** the stiffness gain at ρ = 0.3 is 2.57 %, so write 2.6 %.
- [ ] **L10:** the E-set macros put a thin space before punctuation ("E_max ,") at l.1946, 2329 and 2733.
- [ ] **L11: leftover notes still in the manuscript.**
  - Revision Working Note (p. 2).
  - \mycomment notes at l.277, 298 and 1869.
- [ ] **L12: letter values that hold only for a subset of cases.**
  - "23 to 61 %" (RL l.509).
  - "for two structures" (RL l.565).
  - "27 to 32 %" (RL l.195) holds only for β_φ = 0.01.
- [ ] **L13:** "post-peak D_s does not vanish for larger M (Sec. 3.2.2)": the section shows no post-peak data.
  - The data do support the claim: from M = 1e4 to 1e5, D_s/(Π_frac+D_s) goes 0.388→0.374, 0.361→0.351 and 0.475→0.458.
  - Fix: add one sentence with these values.
- [ ] **L14:** Sec. 3.1, l.1548, "proving that…" is an overclaim (Hannover).
- [ ] **L15:** the first-peak definition says "1 % of the current load". The script uses 1 % below the running maximum.

## Verified OK

- **G_c relation:** A/B/C match the code. The thresholds φ ≤ 0.145 and φ ≥ 0.564 are correct, and so is the cheapest grade (E = 0.6 E_1, G_c = 0.1 G_c,1).
- **First peak, ρ = 0.3:**
  - vs E_min: +82.2…+92.0 % (load) and +218.8…+260.4 % (work).
  - vs E_max: −26.3…−31.6 % (load) and −49.9…−56.6 % (work).
- **First peak, ρ = 0.6:** +18.6…+21.3 % (load) and +42.1…+51.8 % (work).
- **β_s influence:** the first peaks drop by 27.1–32.0 % from β_s = 0.015 to 0.06 mm, and the ranking is unchanged.
- **Tables 3–6:** all values check.
- **Secondary measures, mobility study and β_φ study:** all values check.
- **Energy balance:** within 0.23 % at the first peak and 2.7 % at the end.
- **Strip reactions:** left and right differ by at most 0.91 % before the first crack.
- **Letter cross-references:** all match the compiled manuscript. No "??" appears.
