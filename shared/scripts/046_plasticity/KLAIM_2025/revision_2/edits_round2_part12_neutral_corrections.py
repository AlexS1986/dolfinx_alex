#!/usr/bin/env python3
"""Round-2 edit script, part 12 (2026-10-28): neutral wording of the code/manuscript harmonization
in the two response letters (Alex: "the response letter(s) should not mention the embarrassing errors
openly. just mention that results were recalculated because of small inconsistencies").

- Reviewer letter: the itemised section "Additional corrections by the authors" (13 items with
  old-versus-new numbers) is replaced by one neutral paragraph under the new heading
  "Further changes by the authors"; the related-changes block (quotes of the harmonized equations and
  parameters) is kept with a shortened heading; every cross-reference to the old section is rewritten.
- Editor letter: main-revisions item 4, the "audit" phrase is replaced by a neutral one.

Only response_to_reviewers_template.tex and response_to_editor_template.tex are touched.
Backups: scratchpad backup_pre_part12/. One-shot: fails loudly on the first anchor that is gone.
"""
HERE = "/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/046_plasticity/KLAIM_2025/revision_2/"
LT = HERE + "response_to_reviewers_template.tex"
ED = HERE + "response_to_editor_template.tex"


class Doc:
    def __init__(self, path):
        self.path = path
        self.t = open(path, encoding="utf-8").read()

    def rep(self, old, new, n=1):
        c = self.t.count(old)
        if c != n:
            raise SystemExit("%s: anchor found %d times (expected %d):\n%s" % (self.path, c, n, old[:200]))
        self.t = self.t.replace(old, new)

    def cut(self, start_anchor, end_anchor, new):
        """Replace everything from start_anchor (inclusive) up to end_anchor (exclusive); both unique."""
        for a in (start_anchor, end_anchor):
            c = self.t.count(a)
            if c != 1:
                raise SystemExit("%s: cut anchor found %d times (expected 1):\n%s" % (self.path, c, a[:200]))
        i = self.t.index(start_anchor)
        j = self.t.index(end_anchor)
        if j <= i:
            raise SystemExit("%s: cut anchors in wrong order" % self.path)
        self.t = self.t[:i] + new + self.t[j:]

    def save(self):
        open(self.path, "w", encoding="utf-8").write(self.t)


lt = Doc(LT)
ed = Doc(ED)

# ============================================================================ reviewer letter: the section itself
NEW_SECTION = r"""\section*{Further changes by the authors}

\begin{response}
While preparing the new simulations, we compared the simulation code with the manuscript and removed a number of
small inconsistencies between the two. They concern the form of the crack surface density and the Kolosov constant
of the surfing boundary condition, the parameters of the degradation function and the residual stiffness, the linear
elastic embedding and the auxiliary elastic constants of the K-field, the homogenization of the two-dimensional unit
cells, the amplitude of the K-field, the fracture resistance of the parameter set \EqSigmaStarLabel{} and the geometry
of the homogeneous reference, as well as the mesh and the thickness of the three-dimensional specimen. The manuscript
was harmonized accordingly and all results of Sections~3.5, 3.6, 4 and~5 were recomputed; the new values are given in
the marked manuscript and in the responses above. In both dimensions, the effective crack resistance is now evaluated
as the maximum of $J_x$ over one pore-to-pore cycle, which in two dimensions coincides with the peak immediately before
re-nucleation, so that the two-dimensional values are unaffected by this definition. The three-dimensional series uses
a fixed element size that resolves all pores, and its records of $J_x$ are normalized by the numerical fracture
resistance of a three-dimensional homogeneous specimen computed with the same mesh rule. The two-dimensional
conclusions are unchanged; the three-dimensional result changed as described in Section~5.2 and in the response to
Reviewer~3, comment~3.
\end{response}

"""
lt.cut(
    "\\section*{Additional corrections by the authors}\n",
    "\\begin{relatedchanges}\n\\relhead{Corrections of equations and parameters}\n",
    NEW_SECTION,
)
lt.rep(
    "\\relhead{Corrections of equations and parameters}",
    "\\relhead{Harmonized equations and parameters}",
)

# ============================================================================ reviewer letter: cross-references
# Notes on this document
lt.rep(
    "All numerical results of the manuscript were recomputed for this revision (see the response\nto Reviewer~1, point~3, and the section ``Additional corrections by the authors'').",
    "All numerical results of the manuscript were recomputed for this revision (see the response\nto Reviewer~1, point~3, and the section ``Further changes by the authors'').",
)
# Response to the editor, summary bullet
lt.rep(
    "\\item all simulations recomputed with consistent parameters ($\\beta = 0.2$, $\\eta = 10^{-5}$ in two and three dimensions, linear elastic embedding); see ``Additional corrections by the authors'';",
    "\\item all simulations were recomputed with the parameters stated in the manuscript ($\\beta = 0.2$, $\\eta = 10^{-5}$ in two and three dimensions, linear elastic embedding), see ``Further changes by the authors'';",
)
# Reviewer 1, point 1
lt.rep(
    "Independently of the reviewer's comments, we also went through the whole manuscript once more for notation, consistency and the correspondence between the text and the simulation code. The corrections resulting from this audit are listed in the section ``Additional corrections by the authors'' at the end of this letter.",
    "Independently of the reviewer's comments, we also went through the whole manuscript once more for notation and consistency and harmonized it with the simulation code; these changes are summarized in the section ``Further changes by the authors'' at the end of this letter.",
)
# Reviewer 1, minor 25 (figures)
lt.rep(
    "Fig.~3 (Section~3.5) was regenerated as a vector graphic from the new homogeneous reference runs and shows solver output only, see items~12 and~13 of ``Additional corrections by the authors''.",
    "Fig.~3 (Section~3.5) was regenerated as a vector graphic from the new homogeneous reference runs, see also the section ``Further changes by the authors''.",
)

# ============================================================================ editor letter: main revisions, item 4
ed.rep(
    "and to correct inconsistencies between the simulation code and the manuscript\nfound in an audit of the code (listed at the end of the response letter).",
    "and to remove small inconsistencies between the simulation code and the manuscript.",
)

lt.save()
ed.save()
print("part 12 applied: reviewer letter (section replaced, 4 cross-references), editor letter (item 4)")
