#!/bin/bash
# Build the report PDF from report.md (Markdown source, LaTeX/xelatex output).
#
# Figures and tables are auto-numbered by LaTeX via {#fig:label} / \label{tbl:label}
# in the captions and referenced in the text with \ref{...} -> no manual renumbering.
# Requires: pandoc + a LaTeX toolchain with xelatex (TeXLive) and the DejaVu fonts.
#
# The figures are produced by
#   python3 make_report_figs.py --bmp <..._mit_AR_Bereich.bmp> --src <...Uebergangsbereich.bmp>
# (input data + material assignment only, no FE solve needed).
#
# Usage:  bash build_report.sh
set -e
cd "$(dirname "$0")"
OUT="WAAM_N1_transition_report.pdf"
pandoc report.md -o "$OUT" --pdf-engine=xelatex
echo "wrote $OUT"
