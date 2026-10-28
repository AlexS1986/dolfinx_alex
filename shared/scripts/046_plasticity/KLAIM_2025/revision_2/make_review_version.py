#!/usr/bin/env python3
"""Internal-review version of the revised manuscript for the co-authors (2026-10-28).

From manuscript_revised_template.tex:
  - the "Revision Working Note" section (internal bookkeeping) is replaced by a short
    "Note to the co-authors" that explains the document set and the provisional (red) values
  - manuscript_review_marked.tex  : changes markup kept (new text underlined, removed text struck out)
  - manuscript_review_clean.tex   : all changes accepted (make_clean_manuscript.py)
Both are built with pdflatex (twice, inline bibliography, no bibtex) and copied, together with the
current response letter, into reading/ under *_COAUTHORS_<date>.pdf. Nothing in reading/ is overwritten.
The working templates are not modified.
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-10-28"   # suffix of the reading/ copies, e.g. 2026-10-28b
MS = os.path.join(HERE, "manuscript_revised_template.tex")
MARKED = os.path.join(HERE, "manuscript_review_marked.tex")
CLEAN = os.path.join(HERE, "manuscript_review_clean.tex")
LETTER_PDF = os.path.join(HERE, "response_to_reviewers_template.pdf")
READING = os.path.join(HERE, "reading")

NOTE = r"""\section*{Note to the co-authors}
This is the second revised version of the manuscript (decision of 10 September 2026, deadline 1 November 2026),
prepared for internal review before submission. The marked version shows every change against the submitted version
(new text underlined, removed text struck out); the clean version has all changes accepted. The response to the editor
and the reviewers is a separate document.

One set of results is still provisional and is printed in \textcolor{red}{red}: the three-dimensional effective crack
resistances normalized by the crack resistance of the homogeneous material (Section~\ref{sec:3D_fracture_behavior},
abstract, conclusion). The homogeneous reference computed so far is plastic through the whole thickness, whereas the
porous specimens carry a linear elastic layer on each $z$ face that lies on the crack plane
(Section~\ref{sec:3D_effective_medium}). Two reference simulations with the same elastic layers are running; the first
completes on 28 October, the second by 31 October. Their first values indicate that the normalized three-dimensional
results rise by about $5$ to $8\,\%$, so that the statement that the porous material does not exceed the homogeneous
value may turn into a small enhancement for the wider walls. All two-dimensional results, the three-dimensional values
in units of $J_c^0$, the three-dimensional mechanism and the comparison between the two models are final. This note and
the red marking will be removed before submission.

"""

t = open(MS, encoding="utf-8").read()
a = t.find(r"\section*{Revision Working Note}")
b = t.find(r"\section{Introduction}")
assert 0 < a < b, "Revision Working Note / Introduction anchors not found"
t = t[:a] + NOTE + t[b:]
assert t.count(r"\section*{Note to the co-authors}") == 1
open(MARKED, "w", encoding="utf-8").write(t)

r = subprocess.run([sys.executable, os.path.join(HERE, "make_clean_manuscript.py"), MARKED, CLEAN],
                   capture_output=True, text=True)
print(r.stdout.strip())
assert r.returncode == 0, r.stderr
c = open(CLEAN, encoding="utf-8").read()
assert r"\section*{Note to the co-authors}" in c, "note lost in the clean build"
assert "Revision Working Note" not in c and "Revision Working Note" not in t


def build(tex):
    base = os.path.splitext(os.path.basename(tex))[0]
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex)],
                       cwd=HERE, capture_output=True, text=True)
    log = open(os.path.join(HERE, base + ".log"), encoding="utf-8", errors="replace").read()
    errors = [l for l in log.splitlines() if l.startswith("!")]
    undefined = len(re.findall(r"Reference `[^']+' on page \d+ undefined", log)) + ("There were undefined" in log)
    pages = re.search(r"Output written on .*\((\d+) pages", log)
    print(f"{base}: {pages.group(1) if pages else '?'} pages, {len(errors)} errors, {undefined} undefined")
    assert not errors and not undefined, errors[:3]
    return os.path.join(HERE, base + ".pdf")


marked_pdf = build(MARKED)
clean_pdf = build(CLEAN)
# the letter is built under its own name from the current source (the working PDF may be stale or open in Acrobat)
LETTER_TEX = os.path.join(HERE, "response_review_letter.tex")
shutil.copy2(os.path.join(HERE, "response_to_reviewers_template.tex"), LETTER_TEX)
letter_pdf = build(LETTER_TEX)

for src, name in ((clean_pdf, "manuscript_clean_COAUTHORS"), (marked_pdf, "manuscript_marked_COAUTHORS"),
                  (letter_pdf, "response_letter_COAUTHORS")):
    dst = os.path.join(READING, f"{name}_{DATE}.pdf")
    assert not os.path.exists(dst), f"refusing to overwrite {dst}"
    shutil.copy2(src, dst)
    print("->", dst)
