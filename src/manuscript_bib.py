#!/usr/bin/env python3
"""Generate a BibTeX file from the inline thebibliography of manuscript/Manuscript.tex.

The long-form manuscript carries its references as \\bibitem entries (Author. Title.
Journal, vol(num):pages, year.). The journal builds use BibTeX, so this converts every
entry into an @article/@book record with the same citation key, preserving LaTeX
accents. Run after editing the bibliography; verify the count matches.

Usage: python3 src/manuscript_bib.py OUT.bib [OUT2.bib ...]
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = {"vovk2005", "ipcc2014wetlands", "verra2023vm0033", "simard2019data"}


def parse():
    tex = (ROOT / "manuscript/Manuscript.tex").read_text()
    bib = tex[tex.index(r"\begin{thebibliography}"):tex.index(r"\end{thebibliography}")]
    items = re.findall(r"\\bibitem\{([^}]+)\}\s*\n(.*?)(?=\n\\bibitem|\Z)", bib, re.S)
    out = []
    for key, body in items:
        lines = [l.strip() for l in body.strip().split("\n") if l.strip() and not l.strip().startswith("%")]
        parts = [p.strip().rstrip(".") for p in " ".join(lines).split(r"\newblock") if p.strip()]
        authors = parts[0] if parts else ""
        authors = re.sub(r"(?<!\\)~", " ", authors)                   # ties, but keep \~n accents
        authors = re.sub(r",?\s+et al\.?", ", others", authors)
        authors = re.sub(r",?\s+and\s+", ", ", authors)
        authors = " and ".join(a.strip() for a in authors.split(", ") if a.strip())
        title = parts[1] if len(parts) > 1 else ""
        src = parts[2] if len(parts) > 2 else ""
        m = re.search(r"\\emph\{([^}]*)\},?\s*([^,]*?),?\s*(\d{4})\s*$", src)
        if m:
            journal, volpage, year = m.group(1), m.group(2).strip(), m.group(3)
        else:
            journal = re.sub(r"\\emph\{([^}]*)\}", lambda mm: mm.group(1), src)
            volpage = ""; ym = re.search(r"(\d{4})", src); year = ym.group(1) if ym else ""
        vol = num = pages = ""
        mv = re.match(r"([\w.]+)(?:\((\w+)\))?:(.*)", volpage)
        if mv:
            vol, num, pages = mv.group(1), mv.group(2) or "", mv.group(3)
        elif volpage:
            vol = volpage
        title_clean = re.sub(r"\\emph\{([^}]*)\}", lambda mm: mm.group(1), title)
        if key in BOOKS or title.startswith("\\emph"):
            fields = [f"  author = {{{authors}}}", f"  title = {{{title_clean}}}",
                      f"  publisher = {{{journal}}}", f"  year = {{{year}}}"]
            entry = "@book"
        else:
            fields = [f"  author = {{{authors}}}", f"  title = {{{title_clean}}}", f"  journal = {{{journal}}}"]
            if vol: fields.append(f"  volume = {{{vol}}}")
            if num: fields.append(f"  number = {{{num}}}")
            if pages: fields.append(f"  pages = {{{pages}}}")
            fields.append(f"  year = {{{year}}}")
            entry = "@article"
        out.append((key, f"{entry}{{{key},\n" + ",\n".join(fields) + "\n}\n"))
    return out


def main():
    out = parse()
    text = "\n".join(e for _, e in out)
    for dest in sys.argv[1:]:
        Path(dest).write_text(text)
    print(f"{len(out)} entries -> {sys.argv[1:]}")
    bad = [k for k, e in out if "year = {}" in e or "journal = {}" in e]
    if bad:
        print("check manually:", bad)


if __name__ == "__main__":
    main()
