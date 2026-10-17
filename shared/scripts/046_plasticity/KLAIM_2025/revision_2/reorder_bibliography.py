#!/usr/bin/env python3
"""Reorder the inline thebibliography of the manuscript by order of first citation in the *clean* text
(\deleted{...} removed, \replaced{new}{old} -> new), so that the numbered references appear in order.
Usage: python3 reorder_bibliography.py manuscript_revised_template.tex   (file is rewritten in place)"""
import re
import sys

path = sys.argv[1]
s = open(path).read()


def strip_braced(text, cmd, keep):
    """remove \\cmd{A} (keep=None), or replace \\cmd{A}{B} by A (keep=0)"""
    out, i = [], 0
    while True:
        j = text.find("\\" + cmd + "{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = j + len(cmd) + 2
        args = []
        for _ in range(1 if keep is None else 2):
            depth, start = 1, k
            while depth:
                c = text[k]
                depth += 1 if c == "{" else -1 if c == "}" else 0
                k += 1
            args.append(text[start:k - 1])
            if k < len(text) and text[k] == "{":
                k += 1
        if keep is not None:
            out.append(args[keep])
        i = k


head, bib, tail = re.split(r"(\\begin\{thebibliography\}\{\d+\}.*?\\end\{thebibliography\})", s, flags=re.S)
body = head[head.index("\\begin{document}"):]
clean = strip_braced(body, "deleted", None)
clean = strip_braced(clean, "replaced", 0)
clean = strip_braced(clean, "mycomment", None)
clean = "\n".join(l for l in clean.split("\n") if not l.lstrip().startswith("%"))
order = []
for m in re.finditer(r"\\cite\{([^}]*)\}", clean):
    for key in m.group(1).split(","):
        key = key.strip()
        if key and key not in order:
            order.append(key)

items = re.findall(r"(\\bibitem\[.*?\]\{([^}]+)\}.*?\\endbibitem)", bib, flags=re.S)
bykey = {k: it for it, k in items}
missing = [k for k in order if k not in bykey]
if missing:
    print("cited but not in bibliography:", missing)
    sys.exit(1)
unused = [k for it, k in items if k not in order]
ordered = order + unused
preamble = bib[:bib.index("\\bibitem[")]
preamble = re.sub(r"\\begin\{thebibliography\}\{\d+\}", "\\\\begin{thebibliography}{%d}" % len(ordered), preamble)
newbib = preamble + "\n\n".join("%%%% %d\n%s" % (i + 1, bykey[k]) for i, k in enumerate(ordered)) + "\n\n\\end{thebibliography}"
open(path, "w").write(head + newbib + tail)
print("bibliography reordered: %d entries (%d cited in order of first citation, %d uncited at the end)" % (len(ordered), len(order), len(unused)))
if unused:
    print("uncited:", unused)
