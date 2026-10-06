#!/usr/bin/env python3
"""Insert verbatim quotes of the marked-up manuscript changes into the
response letter.

Usage (in the Overleaf folder, after compiling the manuscript so that
main_revised_template.aux is current):

    python3 ../18_build_response_quotes.py \
        main_revised_template.tex main_revised_template.aux \
        response_to_reviewers_template.tex

In the letter, a quote request is a comment line of the form

    %@quote <kind>:<anchor>

directly inside a relatedchanges item. Consecutive request lines form one
group, followed by the generated block between
    %@generated-begin
    %@generated-end
which this script (re)creates. Everything outside these blocks is left
unchanged, so the script can be re-run after every change of the
manuscript.

Kinds
  cmd:<text>     the \\added/\\deleted/\\replaced/\\changed command whose
                 source contains <text> (must be unique)
  cmd:<text>@<n> the n-th of several commands that contain <text>
  cmdall:<text>  all such commands that contain <text>
  env:<text>     the displayed math environment (equation/align) that
                 contains <text>, e.g. a \\label name
Quoted text: \\ref/\\eqref and \\cite are resolved to the numbers of the
compiled manuscript, \\label and \\mycomment are removed. Each change
becomes one bullet of the surrounding relatedchanges list, preceded by its location
(title, abstract, section, figure or table caption).
"""
import re
import sys

MS, AUX, LETTER = sys.argv[1:4]
ms = open(MS, encoding="utf-8").read()
aux = open(AUX, encoding="utf-8").read()

labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))
bibs = {k: v for k, v in re.findall(r"\\bibcite\{([^}]*)\}\{\{([^}]*)\}", aux)}

doc_start = ms.index("\\begin{document}")


def grab(i):
    """index of the brace that closes the group opening at ms[i] == '{'"""
    d = 0
    k = i
    while k < len(ms):
        c = ms[k]
        if c == "\\":
            k += 2
            continue
        if c == "%":
            k = ms.index("\n", k)
            continue
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return k
        k += 1
    raise ValueError("unbalanced braces at %d" % i)


# ---------------------------------------------------------------- commands
cmds = []  # (start, end, text)
pat = re.compile(r"\\(added|deleted|replaced|changed)\{")
i = doc_start
while True:
    m = pat.search(ms, i)
    if not m:
        break
    ls = ms.rfind("\n", 0, m.start()) + 1
    if "%" in ms[ls:m.start()].replace("\\%", ""):
        i = m.end()
        continue
    e = grab(m.end() - 1)
    if m.group(1) == "replaced":
        e = grab(e + 1)
    cmds.append((m.start(), e + 1, ms[m.start():e + 1]))
    i = e + 1

# ---------------------------------------------------------------- locations
heads = []  # (pos, kind, title)
for m in re.finditer(r"\\(section|subsection|subsubsection)(\*?)(\[[^\]]*\])?\{", ms):
    if m.start() < doc_start or m.group(2):
        continue
    heads.append((m.start(), m.group(1)))
sec_numbers = []
counters = [0, 0, 0]
for pos, kind in heads:
    lvl = {"section": 0, "subsection": 1, "subsubsection": 2}[kind]
    counters[lvl] += 1
    for j in range(lvl + 1, 3):
        counters[j] = 0
    sec_numbers.append((pos, ".".join(str(c) for c in counters[: lvl + 1])))

floats = []  # (start, end, label)
for m in re.finditer(r"\\begin\{(figure|table)\*?\}", ms):
    env = m.group(1)
    end = ms.index("\\end{%s" % env, m.end())
    lab = re.search(r"\\label\{([^}]*)\}", ms[m.start():end])
    floats.append((m.start(), end, env, lab.group(1) if lab else None))

title_pos = ms.index("\\title", doc_start)
abstract_pos = ms.index("\\abstract{", doc_start)
keywords_pos = ms.index("\\keywords{", doc_start)
maketitle_pos = ms.index("\\maketitle", doc_start)
concl_pos = ms.index("\\label{sec_conclusions}")
app_pos = ms.find("\\begin{appendices}")


def location(pos):
    if pos < abstract_pos:
        return "Title"
    if pos < keywords_pos:
        return "Abstract"
    if pos < maketitle_pos:
        return "Keywords"
    for s, e, env, lab in floats:
        if s <= pos <= e:
            n = labels.get(lab, "?")
            return ("Caption of Fig.~%s" if env == "figure" else "Caption of Table~%s") % n
    if app_pos >= 0 and pos > app_pos:
        n = len(re.findall(r"\\section\{", ms[app_pos:pos]))
        return "Appendix~%s" % chr(ord("A") + max(n, 1) - 1)
    sec = None
    for p, num in sec_numbers:
        if p <= pos:
            sec = num
    if sec is None:
        return "Introduction"
    if sec == "1":
        return "Introduction"
    if pos > concl_pos:
        return "Conclusion"
    return "Section~%s" % sec


# ---------------------------------------------------------------- cleaning
def resolve(text):
    text = re.sub(r"\\label\{[^}]*\}", "", text)

    def mycomment(t):
        out = ""
        k = 0
        while True:
            j = t.find("\\mycomment{", k)
            if j < 0:
                return out + t[k:]
            out += t[k:j]
            # two arguments
            a = j + len("\\mycomment")
            d = 0
            n = 0
            q = a
            while n < 2:
                if t[q] == "{":
                    d += 1
                elif t[q] == "}":
                    d -= 1
                    if d == 0:
                        n += 1
                q += 1
            k = q

    text = mycomment(text)
    text = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(%s)" % labels.get(m.group(1), "??"), text)
    text = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "??"), text)

    def cite(m):
        keys = [k.strip() for k in m.group(2).split(",")]
        return "[" + ", ".join(bibs.get(k, "?") for k in keys) + "]"

    text = re.sub(r"\\cite(p|t)?\{([^}]*)\}", lambda m: cite(m), text)
    text = text.replace("\\protect", "")
    return text


def math_env(anchor):
    hits = []
    for m in re.finditer(r"\\begin\{(equation|align)\*?\}", ms):
        end = ms.index("\\end{%s" % m.group(1), m.end())
        end = ms.index("}", end) + 1
        body = ms[m.start():end]
        if " ".join(anchor.split()) in " ".join(body.split()):
            hits.append((m.start(), body, m.group(1)))
    if len(hits) != 1:
        raise SystemExit("env anchor %r matches %d environments" % (anchor, len(hits)))
    pos, body, env = hits[0]
    body = re.sub(r"\\begin\{%s\*?\}" % env, "\\\\begin{%s*}" % env, body, count=1)
    body = re.sub(r"\\end\{%s\*?\}" % env, "\\\\end{%s*}" % env, body)
    num = re.findall(r"\\label\{([^}]*)\}", ms[pos:pos + len(body) + 50])
    nums = [labels.get(n, "?") for n in num if n in labels]
    where = location(pos)
    head = where + (", Eqs.~(%s)" % ")--(".join([nums[0], nums[-1]]) if len(nums) > 1
                    else (", Eq.~(%s)" % nums[0] if nums else ""))
    body = resolve(body)
    body = re.sub(r"\n[ \t]*(?=\n)", "", body)  # no empty lines in math
    return pos, head, body


ERRORS = []


def quotes_for(request):
    try:
        return _quotes_for(request)
    except SystemExit as e:
        ERRORS.append(str(e))
        return []


def _quotes_for(request):
    kind, _, anchor = request.partition(":")
    anchor = anchor.strip()
    if kind in ("cmd", "cmdall"):
        nth = None
        m = re.match(r"(.*)@(\d+)$", anchor, re.S)
        if m:
            anchor, nth = m.group(1), int(m.group(2))
        a = " ".join(anchor.split())
        hits = [c for c in cmds if a in " ".join(c[2].split())]
        if nth is not None:
            hits = hits[nth - 1:nth]
        if kind == "cmd" and len(hits) != 1:
            raise SystemExit("cmd anchor %r matches %d commands" % (anchor, len(hits)))
        if not hits:
            raise SystemExit("cmdall anchor %r matches nothing" % anchor)
        return [(c[0], location(c[0]), resolve(c[2]), False) for c in hits]
    if kind == "env":
        pos, head, body = math_env(anchor)
        return [(pos, head, body, True)]
    raise SystemExit("unknown request %r" % request)


# ---------------------------------------------------------------- letter
lines = open(LETTER, encoding="utf-8").read().split("\n")
out = []
used = set()
k = 0
while k < len(lines):
    line = lines[k]
    if line.lstrip().startswith("%@quote "):
        group = []
        while k < len(lines) and lines[k].lstrip().startswith("%@quote "):
            group.append(lines[k])
            k += 1
        # drop an existing generated block
        if k < len(lines) and lines[k].strip() == "%@generated-begin":
            while lines[k].strip() != "%@generated-end":
                k += 1
            k += 1
        out.extend(group)
        quotes = []
        for g in group:
            quotes.extend(quotes_for(g.strip()[len("%@quote "):]))
        quotes.sort(key=lambda q: q[0])
        out.append("%@generated-begin")
        for pos, where, body, is_math in quotes:
            used.add(pos)
            out.append("\\item \\msloc{%s}" % where)
            out.append(body)
        out.append("%@generated-end")
        continue
    out.append(line)
    k += 1
if ERRORS:
    print("\n".join(ERRORS))
    raise SystemExit("letter not written: %d anchor errors" % len(ERRORS))
open(LETTER, "w", encoding="utf-8").write("\n".join(out))

# report manuscript changes that are not quoted anywhere
missing = [(ms.count("\n", 0, c[0]) + 1, " ".join(c[2].split())[:90])
           for c in cmds if c[0] not in used]
print("quoted %d of %d markup commands" % (len(used & {c[0] for c in cmds}), len(cmds)))
for ln, txt in missing:
    print("  not quoted: l.%d %s" % (ln, txt))
