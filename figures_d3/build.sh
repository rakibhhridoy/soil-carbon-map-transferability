#!/usr/bin/env bash
# Build MDBC manuscript figures: D3 (server-side) -> SVG -> vector PDF.
# Requires: node (npm install already run), rsvg-convert.
# Output PDFs overwrite manuscript/figures/*.pdf (same names the LaTeX expects).
set -euo pipefail
cd "$(dirname "$0")"

node build_figures.mjs
mkdir -p ../manuscript/figures
for f in svg/*.svg; do
  name=$(basename "$f" .svg)
  rsvg-convert -f pdf -o "../manuscript/figures/${name}.pdf" "$f"
  echo "  pdf -> manuscript/figures/${name}.pdf"
done
echo "figures built."
