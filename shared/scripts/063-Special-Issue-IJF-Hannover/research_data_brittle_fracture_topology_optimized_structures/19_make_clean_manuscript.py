#!/usr/bin/env python3
"""Create the clean revised manuscript (all changes accepted) from the
marked-up working copy.

Usage (in the Overleaf folder):
    python3 ../19_make_clean_manuscript.py main_revised_template.tex main_revised_clean.tex

- \\added{x} -> x, \\deleted{x} -> (removed), \\replaced{new}{old} -> new,
  \\changed{x} -> x (also inside equations); a preceding \\protect is dropped
- \\mycomment{who}{text} notes are removed
- the "Revision Working Note" section is removed
- the changes package and the citation boxing needed only for the markup
  are removed from the preamble
Re-run after every change of main_revised_template.tex; do not edit the
clean file by hand.
"""
import re
import sys

src, dst = sys.argv[1:3]
t = open(src, encoding="utf-8").read()


def close(s, i):
    """index of the brace closing the group that opens at s[i] == '{'"""
    d = 0
    k = i
    while k < len(s):
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "%":
            k = s.find("\n", k)
            if k < 0:
                return len(s) - 1
            continue
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return k
        k += 1
    raise ValueError("unbalanced braces at %d" % i)


def accept(s):
    out = []
    i = 0
    pat = re.compile(r"(\\protect\s*)?\\(added|deleted|replaced|changed|mycomment)\s*\{")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            return "".join(out)
        # leave commented-out lines untouched
        ls = s.rfind("\n", 0, m.start()) + 1
        if "%" in s[ls:m.start()].replace("\\%", ""):
            out.append(s[i:m.end()])
            i = m.end()
            continue
        out.append(s[i:m.start()])
        b = m.end() - 1
        e = close(s, b)
        kind = m.group(2)
        if kind in ("added", "changed"):
            out.append(accept(s[b + 1:e]))
            i = e + 1
        elif kind == "deleted":
            i = e + 1
        elif kind == "replaced":
            e2 = close(s, e + 1)
            out.append(accept(s[b + 1:e]))
            i = e2 + 1
        else:  # mycomment{who}{text}
            e2 = close(s, e + 1)
            i = e2 + 1


head, sep, body = t.partition("\\begin{document}")
body = accept(body)

# drop the Revision Working Note section (up to the next \section)
body = re.sub(r"\\section\*\{Revision Working Note\}.*?(?=\\section\{)", "", body, flags=re.S)

# preamble: remove the changes package and the citation boxing for the markup
head = re.sub(r"\\usepackage(\[[^\]]*\])?\{changes\}[^\n]*\n", "", head)
head = re.sub(r"%[^\n]*citations inside \\added[^\n]*\n(%[^\n]*\n)*\\AtBeginDocument\{\\let\\origcite\\cite[^\n]*\n", "", head)
# keep \changed usable if it is defined in the preamble
open(dst, "w", encoding="utf-8").write(head + sep + body)

left = re.findall(r"\\(added|deleted|replaced|mycomment)\{", body)
print("written %s (%d markup commands left, should be 0)" % (dst, len(left)))
