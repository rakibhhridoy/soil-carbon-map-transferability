#!/usr/bin/env python3
"""Transform the clean single-column Manuscript.tex into a NatComms-style two-column
manuscript (manuscript_natcomm.tex), matching the Paper5 NatComm template:
 - extarticle twocolumn 9pt, sffamily section heads, "Article" banner title block
 - numeric \\cite with a manual thebibliography (verified entries below)
 - tables -> table* (full width), the delta map -> figure* (full width)
The body prose is reused verbatim (no retyping); only formatting/citations change.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "Manuscript.tex").read_text()

# ---- 1. extract abstract text (for the title block) ----
abs = re.search(r"\\begin\{abstract\}\s*\\noindent(.*?)\\end\{abstract\}", src, re.S).group(1).strip()

# ---- 2. extract body: from after the keywords block to before the bibliography ----
# body starts at the Introduction section, ends before \bibliographystyle
body = src[src.index("\\section{Introduction}"): src.index("\\bibliographystyle")]

# ---- 3. transforms on the body ----
# normalise all natbib cite variants to plain numeric \cite (extarticle has no natbib)
for cmd in ("citep", "citet", "citealp", "citealt", "citeyear", "citeauthor"):
    body = body.replace("\\" + cmd + "{", "\\cite{")
# section/subsection -> unnumbered sffamily; drop the Introduction header (flows from title)
body = body.replace("\\section{Introduction}\n", "")
body = re.sub(r"\\section\{([^}]*)\}", r"\\section*{\1}", body)
body = re.sub(r"\\subsection\{([^}]*)\}", r"\\subsection*{\1}", body)
# tables full-width
body = body.replace("\\begin{table}[t]", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}")
# the delta map -> full-width figure*
body = re.sub(
    r"\\begin\{figure\}\[t\]\s*\\centering\s*\\includegraphics\[width=\\linewidth\]\{fig_map\.pdf\}(.*?)\\end\{figure\}",
    lambda m: "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.86\\textwidth]{fig_map.pdf}"
              + m.group(1) + "\\end{figure*}",
    body, flags=re.S)
# single-column figures fill their column
body = body.replace("width=0.62\\linewidth", "width=\\linewidth")
# strip editorial macros if any survived
body = body.replace("% ---------------------------------------------------------------------\n", "")

# ---- 4. preamble + title block ----
PRE = r"""% ============================================================
% MDBC : NatComms-style two-column manuscript (formatting only).
% Generated from Manuscript.tex by build_natcomm.py; body content identical.
% Build: pdflatex -> pdflatex -> pdflatex  (refs are inline thebibliography)
% ============================================================
\documentclass[twocolumn,9pt]{extarticle}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage{xcolor}
\usepackage[top=1.7cm,bottom=1.9cm,left=1.6cm,right=1.6cm,columnsep=0.7cm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{caption}
\usepackage{siunitx}
\usepackage{textcomp}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{marvosym}
\usepackage{orcidlink}
\usepackage{hyperref}
\usepackage{url}
\usepackage[mathlines,switch]{lineno}
\definecolor{natblue}{HTML}{0B6FB8}
\definecolor{natgrey}{HTML}{5A5A5A}
\hypersetup{colorlinks=true, linkcolor=black, citecolor=natblue, urlcolor=natblue,
  pdfauthor={Md Rakib Hasan},
  pdftitle={Global maps overstate the reliability of blue-carbon credits}}
\sisetup{detect-all}
\captionsetup{labelfont=bf, labelsep=period, font=small, justification=justified,
  singlelinecheck=false}
\renewcommand{\figurename}{Fig.}
\titleformat{\section}{\normalfont\sffamily\bfseries\large}{}{0pt}{}
\titleformat{\subsection}{\normalfont\sffamily\bfseries\normalsize}{}{0pt}{}
\titlespacing*{\section}{0pt}{10pt}{3pt}
\titlespacing*{\subsection}{0pt}{7pt}{2pt}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}
\fancyfoot[C]{\small\sffamily\thepage}

\begin{document}
\twocolumn[{%
\vspace*{-6pt}
{\sffamily\bfseries\large Article}\\[5pt]
{\color{black}\rule{\textwidth}{1.0pt}}\\[9pt]
\noindent\resizebox{\textwidth}{!}{\sffamily\bfseries Global maps overstate the reliability of blue-carbon credits}\par
\vspace{14pt}
{\centering\normalsize
Md~Rakib~Hasan\,\orcidlink{0009-0002-4007-7590}\textsuperscript{1,2}\,\Letter\par}
\vspace{9pt}
{\small\noindent ABSTRACTTEXT\par}
\vspace{8pt}
{\color{black}\rule{\textwidth}{0.4pt}}\\[4pt]
{\footnotesize\color{natgrey}%
\textsuperscript{1}Fermium Systems, Dhaka 1207, Bangladesh.\;
\textsuperscript{2}Department of Soil, Water and Environment, University of Dhaka, Dhaka 1000, Bangladesh.\;
\Letter\,e-mail: \href{mailto:rakibhhridoy@fermium.systems}{rakibhhridoy@fermium.systems}}
\vspace{8pt}
{\color{black}\rule{\textwidth}{0.4pt}}
\vspace{6pt}
}]
\thispagestyle{fancy}
\linenumbers

"""
PRE = PRE.replace("ABSTRACTTEXT", abs)

# ---- 5. bibliography (verified entries) ----
BIB = (HERE / "natcomm_refs.tex").read_text()

out = PRE + body + "\n" + BIB + "\n\\end{document}\n"
(HERE / "manuscript_natcomm.tex").write_text(out)
print("wrote manuscript_natcomm.tex")
# report any cite key not present in the bib
keys = set(re.findall(r"\\cite\{([^}]+)\}", body))
keys = {k.strip() for grp in keys for k in grp.split(",")}
bibkeys = set(re.findall(r"\\bibitem\{([^}]+)\}", BIB))
missing = keys - bibkeys
print("cite keys in body:", len(keys), "| bib entries:", len(bibkeys))
print("MISSING from bib:", sorted(missing) if missing else "none")
print("unused bib entries:", sorted(bibkeys - keys) if (bibkeys - keys) else "none")
