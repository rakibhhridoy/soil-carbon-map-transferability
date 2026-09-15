#!/usr/bin/env bash
# Build MDBC manuscript figures: D3 (server-side) -> SVG -> vector PDF.
# Requires: node (npm install already run), rsvg-convert.
# Output PDFs overwrite manuscript/figures/*.pdf (same names the LaTeX expects).
set -euo pipefail
cd "$(dirname "$0")"

node build_figures.mjs
node build_journal_figures.mjs
mkdir -p ../manuscript/figures
for f in svg/*.svg; do
  name=$(basename "$f" .svg)
  rsvg-convert -f pdf -o "../manuscript/figures/${name}.pdf" "$f"
  echo "  pdf -> manuscript/figures/${name}.pdf"
done
# showcase finish (talks, cover letter, press): same data, stronger vector depth; PDF + PNG
FIG_STYLE=showcase node build_journal_figures.mjs
mkdir -p ../manuscript/figures/showcase
for f in svg/showcase/*.svg; do
  name=$(basename "$f" .svg)
  rsvg-convert -f pdf -o "../manuscript/figures/showcase/${name}.pdf" "$f"
  rsvg-convert -f png -z 8 -b white -o "../manuscript/figures/showcase/${name}.png" "$f"
  echo "  showcase -> manuscript/figures/showcase/${name}.{pdf,png}"
done
echo "figures built."
