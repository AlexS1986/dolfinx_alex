#!/usr/bin/env python3
"""Round-2 edit script, part 14 (2026-10-28): regime boundary in the conclusion + consistency pass.

Only manuscript_revised_template.tex is touched. One-shot: fails loudly on the first anchor that is
gone or not unique. Backups: scratchpad backup_pre_part14/.

E1  Conclusion, mechanisms paragraph (inside the existing \\replaced block): the thin-ligament sentence
    now makes the same distinction as Sec. 4.4 - yielding across the width is observed for the recorded
    runs with w_s <= 0.6L, the nucleation-controlled regime in terms of J_eff extends to w_s = 0.5L and
    the [vM]/[Eq sigma*] curves cross between 0.5L and 0.6L (Jeff_summary_2D.csv: vM 0.926 < EqSig 0.949
    at 0.5L, vM 1.084 > EqSig 1.009 at 0.6L).
E2  Fig. 10 caption: the period of the original caption followed the closing brace of the \\added text,
    which already ends with a period -> ".." in the clean build. The original period is kept, the one
    inside the \\added text is dropped.
E3  Sec. 5.1, J_bc sentence (inside \\added): ratio J_bc/J_eff is 1.366/1.223 = 1.12 ... 0.972/1.402 = 0.69
    (master_table.csv, Table 4 constants) -> "0.7 to 1.1" instead of "0.7 to 1.2".
E4  Sec. 4.1 (original text): the three panels of Fig. 8 lie side by side (a, b, c), so "(top)" -> "a)";
    the 2D block is written 6 x 3 (x times y) as in "the 6 x 3 porous domain" two paragraphs later and
    as 6 x 3 x 2 in Sec. 5.1.
E5  Introduction (original text): the outline sentence promises the formulation for both the linear
    elastic and the von Mises model, which is Section 3 (3.1 ductile, 3.2 brittle), not Section 3.1.
"""
HERE = "/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/046_plasticity/KLAIM_2025/revision_2/"
MS = HERE + "manuscript_revised_template.tex"


class Doc:
    def __init__(self, path):
        self.path = path
        self.t = open(path, encoding="utf-8").read()

    def rep(self, old, new, n=1):
        c = self.t.count(old)
        if c != n:
            raise SystemExit("%s: anchor found %d times (expected %d):\n%s" % (self.path, c, n, old[:200]))
        self.t = self.t.replace(old, new)

    def save(self):
        open(self.path, "w", encoding="utf-8").write(self.t)


ms = Doc(MS)

# ---------------------------------------------------------------- E1 conclusion: regime boundary
ms.rep(
    "Thin ligaments of the plastic material, $w_s \\le 0.6L$, yield across their width and reach this stress only "
    "after a large plastic strain, which keeps the driving force at re-nucleation low and makes the result sensitive "
    "to the yield stress and the hardening modulus. In wide ligaments,",
    "Thin ligaments of the plastic material ($w_s \\le 0.6L$ in the recorded runs) yield across their width and reach "
    "this stress only after a large plastic strain, which keeps the driving force at re-nucleation low and makes the "
    "result sensitive to the yield stress and the hardening modulus; in terms of the effective crack resistance, this "
    "nucleation-controlled regime extends to $w_s = 0.5L$, where the plastic set lies below the linear elastic set of "
    "equal strength, and the two curves cross between $0.5L$ and $0.6L$. In wide ligaments,",
)

# ---------------------------------------------------------------- E2 Fig. 10 caption: double period
ms.rep(
    "The picture at the top shows the phase field $s$ of the middle pore row after complete separation (\\vMLabel).}.",
    "The picture at the top shows the phase field $s$ of the middle pore row after complete separation (\\vMLabel)}.",
)

# ---------------------------------------------------------------- E3 Sec. 5.1: J_bc / J_eff ratio
ms.rep(
    "which is close to the effective crack resistances found below (ratio $0.7$ to $1.2$).}",
    "which is close to the effective crack resistances found below (ratio $0.7$ to $1.1$).}",
)

# ---------------------------------------------------------------- E4 Sec. 4.1: panel reference and 6 x 3
ms.rep(
    "A $3 \\times 6$ array of these cells is assembled, as depicted by the red regions in "
    "Fig.~\\ref{fig:setup_crack_growth} (top), where $w_s$ denotes the distance between pores.",
    "A \\replaced{$6 \\times 3$}{$3 \\times 6$} array of these cells is assembled, as depicted by the red regions in "
    "Fig.~\\ref{fig:setup_crack_growth}\\replaced{a)}{ (top)}, where $w_s$ denotes the distance between pores.",
)

# ---------------------------------------------------------------- E5 introduction: outline pointer
ms.rep(
    "Section~\\ref{sec:phase_fiel_model_ductile} presents the phase-field\nfracture formulation for both "
    "linear-elastic and von Mises-type plasticity material models.",
    "Section~\\replaced{\\ref{sec:simulation_phase_field_models}}{\\ref{sec:phase_fiel_model_ductile}} presents the "
    "phase-field\nfracture formulation for both linear-elastic and von Mises-type plasticity material models.",
)

ms.save()
print("part 14 applied: manuscript E1-E5 (5 anchors)")
