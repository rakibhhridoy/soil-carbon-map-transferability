#!/usr/bin/env python3
"""Replace prose em-dashes (---) with context-appropriate punctuation across the
manuscript, supplementary, and cover letter. En-dashes (--) in ranges (0--100) and
compound names (Amazon--Amap\\'a) are left untouched, as are LaTeX comment headers.
Whitespace-flexible matching (\\s+) so source line-wrapping does not matter. Each
pattern must match exactly once."""
import re
from pathlib import Path

# (single-spaced source snippet containing ---, replacement snippet)
REP = {
 "Manuscript.tex": [
  ("blue carbon---terrestrial soil carbon transfers far better. In unsamp",
   "blue carbon; terrestrial soil carbon transfers far better. In unsamp"),
  ("the large majority of it---roughly 10~Pg---belowground",
   "the large majority of it, roughly 10~Pg, belowground"),
  ("underpin operational decisions---setting baselines for blue-carbon credits",
   "underpin operational decisions: setting baselines for blue-carbon credits"),
  ("the operationally decisive question---can soil-carbon knowledge",
   "the operationally decisive question, can soil-carbon knowledge"),
  ("coastline actually faces---has not been posed",
   "coastline actually faces, has not been posed"),
  ("\\emph{specific to blue carbon} --- terrestrial soil",
   "\\emph{specific to blue carbon}, terrestrial soil"),
  ("under the identical protocol --- and is driven by",
   "under the identical protocol, and is driven by"),
  ("confounds two distinct failures --- getting a delta's mean level wrong",
   "confounds two distinct failures, getting a delta's mean level wrong"),
  ("internal spatial pattern --- we separate them",
   "internal spatial pattern; we separate them"),
  ("the finite-sample coverage guarantee --- which assumes",
   "the finite-sample coverage guarantee, which assumes"),
  ("test data are exchangeable --- breaks under cross-delta shift",
   "test data are exchangeable, breaks under cross-delta shift"),
  ("gets delta means right --- exactly the behaviour",
   "gets delta means right, exactly the behaviour"),
  ("\\citep{hartdavis2021eot20}) --- the dominant physical control",
   "\\citep{hartdavis2021eot20}), the dominant physical control"),
  ("\\emph{concept shift} --- a change in the environment--carbon",
   "\\emph{concept shift}, a change in the environment--carbon"),
  ("relationship itself between deltas --- which we demonstrate directly",
   "relationship itself between deltas, which we demonstrate directly"),
  ("no globally-shared model --- however well-resolved its covariates --- can transfer",
   "no globally-shared model, however well-resolved its covariates, can transfer"),
  ("the operational GSOCmap product --- the FAO Global Soil Organic Carbon map",
   "the operational GSOCmap product (the FAO Global Soil Organic Carbon map"),
  ("SoilGrids predictions \\citep{hengl2017,poggio2021soilgrids} --- directly against",
   "SoilGrids predictions \\citep{hengl2017,poggio2021soilgrids}), directly against"),
  ("transfer gap of only $0.44$ $R^2$ units --- versus near-zero",
   "transfer gap of only $0.44$ $R^2$ units, versus near-zero"),
  ("span an order of magnitude---from \\SI{106}",
   "span an order of magnitude, from \\SI{106}"),
  ("(Musi--Banyuasin)---a heterogeneity that",
   "(Musi--Banyuasin), a heterogeneity that"),
  ("seven sizeable deltas---including the Niger",
   "seven sizeable deltas, including the Niger"),
  ("8\\% of global mangrove area---carry no usable in-situ cores",
   "8\\% of global mangrove area, carry no usable in-situ cores"),
  ("(Table~\\ref{tab:crediting})---error that the products'",
   "(Table~\\ref{tab:crediting}), error that the products'"),
  ("global maps \\emph{can} be trusted---inside the AOA---and turns",
   "global maps \\emph{can} be trusted, inside the AOA, and turns"),
  ("dominant shift axes---tidal range in macrotidal deltas",
   "dominant shift axes (tidal range in macrotidal deltas"),
  ("regime in the Sundarbans---are consistent with a physical",
   "regime in the Sundarbans) are consistent with a physical"),
  ("region-specific structure --- and our few-shot experiment",
   "region-specific structure, and our few-shot experiment"),
  ("beyond where they were trained---and a reusable, uncertainty-aware",
   "beyond where they were trained, and a reusable, uncertainty-aware"),
 ],
 "supplementary.tex": [
  ("five \\emph{usable} mangrove soil cores---cores that pass",
   "five \\emph{usable} mangrove soil cores, cores that pass"),
  ("ample usable cores---the Everglades (North America)",
   "ample usable cores: the Everglades (North America)"),
  ("tidal constituents---which are masked on land---were sampled",
   "tidal constituents, which are masked on land, were sampled"),
  ("corrupt or truncated on read---SoilGrids texture",
   "corrupt or truncated on read (SoilGrids texture"),
  ("fine terrain and hydrology tiles---and were excluded",
   "fine terrain and hydrology tiles) and were excluded"),
  ("flip sign between deltas --- for instance the SOC response",
   "flip sign between deltas: for instance, the SOC response"),
  ("strongly negative in the Mekong --- direct evidence that",
   "strongly negative in the Mekong, direct evidence that"),
 ],
 "cover_letter.tex": [
  ("driven by concept shift --- the environment--carbon",
   "driven by concept shift (the environment--carbon"),
  ("covariate slopes that reverse sign --- rather than by missing predictors",
   "covariate slopes that reverse sign) rather than by missing predictors"),
  ("global model can be trusted --- portable to any coastal system",
   "global model can be trusted; portable to any coastal system"),
 ],
}


def to_regex(snippet):
    # escape, then let any run of escaped whitespace match \s+
    pat = re.escape(snippet)
    pat = re.sub(r"(\\\s)+", r"\\s+", pat)
    return pat


def main():
    for fn, reps in REP.items():
        p = Path(fn)
        s = p.read_text()
        for old, new in reps:
            pat = to_regex(old)
            hits = re.findall(pat, s)
            if len(hits) != 1:
                raise SystemExit(f"[{fn}] expected 1 match, found {len(hits)}: {old!r}")
            s = re.sub(pat, new.replace("\\", "\\\\"), s)
        p.write_text(s)
        print(f"{fn}: applied {len(reps)} replacements")


if __name__ == "__main__":
    main()
