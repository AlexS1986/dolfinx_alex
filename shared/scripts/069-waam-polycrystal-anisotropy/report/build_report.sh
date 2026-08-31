#!/bin/bash
# Build the report PDF from report.md (Markdown source, LaTeX/xelatex output).
#
# Figures and tables are auto-numbered by LaTeX via {#fig:label} / \label{tbl:label}
# in the captions and referenced in the text with \ref{...} -> no manual renumbering.
# Requires: pandoc + a LaTeX toolchain with xelatex (TeXLive) and the DejaVu fonts.
#
# Usage:  bash build_report.sh
set -e
cd "$(dirname "$0")"
OUT="WAAM_anisotropy_report.pdf"
pandoc report.md -o "$OUT" --pdf-engine=xelatex
echo "wrote $OUT"

# Optional: a Word version for co-authors. Cross-references (\ref) do NOT resolve
# in docx, so numbers are baked in via a LaTeX intermediate is not trivial; if you
# need docx, ask and we add a numbered-fallback build. For review, share the PDF.
