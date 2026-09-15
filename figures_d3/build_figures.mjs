// MDBC manuscript figures, server-side D3 -> SVG (journal-print, colorblind-safe).
// SVGs are converted to vector PDF by build.sh (rsvg-convert). One source of truth:
// data/processed/figure_data.json (written by src/manuscript_assets.py).
import { JSDOM } from "jsdom";
import * as d3 from "d3";
import { feature } from "topojson-client";
import { readFileSync, writeFileSync } from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);

const DATA = JSON.parse(readFileSync("../data/processed/figure_data.json", "utf8"));
const OUT = "svg";

// ---- journal-print style tokens ----
const FONT = "Times, 'Times New Roman', serif";
const INK = "#212121", GRID = "#E0E0E0", AXIS = "#424242";
// Okabe-Ito colorblind-safe palette
// Same palette as the journal figures: red = does not transfer, blue = transfers.
// (Key names are historical; the hues are the groundwater-paper palette.)
const OI = { green: "#1565C0", vermillion: "#C62828", blue: "#7B1FA2",
             orange: "#E65100", sky: "#90CAF9", purple: "#6A1B9A", grey: "#9E9E9E" };
const SEQ = d3.interpolateRgbBasis(["#FFF7BC", "#FEC44F", "#E65100", "#B71C1C"]);

function svgRoot(w, h) {
  const dom = new JSDOM("<!DOCTYPE html><body></body>");
  const body = d3.select(dom.window.document.body);
  const svg = body.append("svg")
    .attr("xmlns", "http://www.w3.org/2000/svg")
    .attr("width", w).attr("height", h)
    .attr("viewBox", `0 0 ${w} ${h}`)
    .attr("font-family", FONT);
  svg.append("rect").attr("width", w).attr("height", h).attr("fill", "white");
  return { dom, svg };
}
function save(dom, name) {
  writeFileSync(`${OUT}/${name}.svg`, dom.window.document.body.innerHTML);
  console.log("  wrote", `${OUT}/${name}.svg`);
}
// hatched fill: light colour tint background + diagonal colour strokes (print/grayscale
// robust). Returns url(#id) for use as a fill.
function hatch(defs, id, color, angle, { spacing = 6, width = 1.3 } = {}) {
  const tint = d3.interpolateRgb(color, "white")(0.74);
  const p = defs.append("pattern").attr("id", id).attr("patternUnits", "userSpaceOnUse")
    .attr("width", spacing).attr("height", spacing)
    .attr("patternTransform", `rotate(${angle})`);
  p.append("rect").attr("width", spacing).attr("height", spacing).attr("fill", tint);
  p.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", spacing)
    .attr("stroke", color).attr("stroke-width", width);
  return `url(#${id})`;
}

function axisStyle(g) {
  g.selectAll("path,line").attr("stroke", AXIS).attr("stroke-width", 0.7);
  g.selectAll("text").attr("fill", INK).attr("font-size", 11);
}
function title(svg, x, y, t) {
  svg.append("text").attr("x", x).attr("y", y).attr("font-size", 13)
     .attr("font-weight", "bold").attr("fill", INK).text(t);
}
function panelLabel(svg, x, y, t) {
  svg.append("text").attr("x", x).attr("y", y).attr("font-size", 13)
     .attr("font-weight", "bold").attr("fill", INK).text(t);
}

// ===================================================================== fig_map
function figMap() {
  const W = 920, H = 430, m = { t: 36, r: 16, b: 30, l: 16 };
  const { dom, svg } = svgRoot(W, H);
  title(svg, m.l, 22, "Core leave-one-delta-out benchmark");

  const topo = require("world-atlas/land-110m.json");
  const land = feature(topo, topo.objects.land);
  const proj = d3.geoNaturalEarth1().fitExtent(
    [[m.l, m.t], [W - m.r, H - m.b]], { type: "Sphere" });
  const path = d3.geoPath(proj);
  const g = svg.append("g");
  g.append("path").attr("d", path({ type: "Sphere" }))
    .attr("fill", "#F5F5F5").attr("stroke", GRID).attr("stroke-width", 0.6);
  g.append("path").attr("d", path(land)).attr("fill", "#ECECEC")
    .attr("stroke", "#D5D5D5").attr("stroke-width", 0.4);

  const deltas = DATA.deltas;
  const rext = d3.extent(deltas, d => d.area_km2);
  const r = d3.scaleSqrt().domain(rext).range([5, 20]);
  const cext = d3.extent(deltas, d => d.soc_med);
  const color = d3.scaleSequential(SEQ).domain(cext);

  const nodes = deltas.map(d => {
    const p = proj([d.lon, d.lat]); return { ...d, x: p[0], y: p[1] }; });
  d3.forceSimulation(nodes)
    .force("x", d3.forceX(d => proj([d.lon, d.lat])[0]).strength(0.6))
    .force("y", d3.forceY(d => proj([d.lon, d.lat])[1]).strength(0.6))
    .force("collide", d3.forceCollide(d => r(d.area_km2) + 1.5))
    .stop().tick(180);

  const gd = svg.append("g");
  nodes.forEach(d => {
    gd.append("circle").attr("cx", d.x).attr("cy", d.y).attr("r", r(d.area_km2))
      .attr("fill", color(d.soc_med)).attr("stroke", INK).attr("stroke-width", 0.8)
      .attr("opacity", 0.92);
    gd.append("text").attr("x", d.x).attr("y", d.y - r(d.area_km2) - 3)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", INK)
      .text(d.name);
  });

  // size legend: circles bottom-aligned on a common baseline so labels clear the
  // largest circle; horizontal spacing scaled to the largest radius.
  const sizes = [500, 2000, 6000];
  const maxR = r(d3.max(sizes));
  const lx = m.l + 8 + maxR, baseY = H - 30;       // baseline circles sit on
  svg.append("text").attr("x", m.l + 8).attr("y", baseY - 2 * maxR - 8).attr("font-size", 10)
     .attr("fill", AXIS).text("mangrove area (km²)");
  let cx = lx;
  sizes.forEach((v, i) => {
    const rr = r(v);
    if (i > 0) cx += r(sizes[i - 1]) + rr + 12;     // gap = both radii + padding
    svg.append("circle").attr("cx", cx).attr("cy", baseY - rr).attr("r", rr)
       .attr("fill", "none").attr("stroke", AXIS).attr("stroke-width", 0.8);
    svg.append("text").attr("x", cx).attr("y", baseY + 13).attr("text-anchor", "middle")
       .attr("font-size", 9).attr("fill", AXIS).text(v);
  });
  // color legend. Drawn as discrete solid-colour strips rather than an SVG gradient:
  // axial-shading PDFs from rsvg-convert do not survive embedding via \includegraphics
  // (the shading is dropped, leaving a flat fill), so plain rects are used instead.
  const gw = 150, gx = W - m.r - gw - 8, gy = H - 52;
  const nStrip = 60, sw = gw / nStrip;
  d3.range(nStrip).forEach(i => {
    svg.append("rect").attr("x", gx + i * sw).attr("y", gy)
       .attr("width", sw + 0.6).attr("height", 10)
       .attr("fill", SEQ((i + 0.5) / nStrip)).attr("stroke", "none");
  });
  svg.append("rect").attr("x", gx).attr("y", gy).attr("width", gw).attr("height", 10)
     .attr("fill", "none").attr("stroke", AXIS).attr("stroke-width", 0.5);
  svg.append("text").attr("x", gx).attr("y", gy - 5).attr("font-size", 10).attr("fill", AXIS)
     .text("median SOC₀₋₁₀₀ (Mg ha⁻¹)");
  [cext[0], cext[1]].forEach((v, i) =>
    svg.append("text").attr("x", gx + i * gw).attr("y", gy + 22)
       .attr("text-anchor", i ? "end" : "start").attr("font-size", 9).attr("fill", AXIS)
       .text(Math.round(v)));
  save(dom, "fig_map");
}

// =================================================================== fig_tiers
// Story: a model's apparent skill collapses as the validation tier becomes more
// honest about spatial / delta structure. Gradient boosting is the headline (bold
// bars + value labels); ridge is overlaid; at the leave-one-delta-out tier the eight
// per-delta R2 values are shown as points to expose the spread. An annotation marks
// the inflation between the conventionally-reported number and the honest one.
function figTiers() {
  const W = 520, H = 360, m = { t: 60, r: 16, b: 64, l: 52 };
  const { dom, svg } = svgRoot(W, H);
  const defs = svg.append("defs");
  title(svg, m.l - 36, 22, "Apparent skill collapses out-of-distribution");
  svg.append("text").attr("x", m.l - 36).attr("y", 38).attr("font-size", 10.5)
     .attr("fill", AXIS).text("gradient-boosting R² under progressively honest validation");

  const tiers = [["t1", "Random", "k-fold"], ["t1g", "Site-grouped", "k-fold"],
                 ["t2", "Spatial", "block"], ["t3", "Leave-one-", "delta-out"]];
  const FLOOR = -2.4;                       // y-axis floor; clip extreme per-delta points
  const fillH = hatch(defs, "h-tier", OI.vermillion, -45);
  const x0 = d3.scaleBand().domain(tiers.map(t => t[0])).range([m.l, W - m.r]).padding(0.34);
  const y = d3.scaleLinear().domain([FLOOR, 0.8]).range([H - m.b, m.t]);

  // shaded "conventionally reported" vs "honest" zones
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", x0("t1g") - m.l)
     .attr("height", H - m.b - m.t).attr("fill", "#F5F5F5");
  svg.append("text").attr("x", (m.l + x0("t1g")) / 2).attr("y", m.t - 6)
     .attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", OI.blue)
     .text("conventionally reported");

  svg.selectAll(".grid").data(y.ticks(6)).join("line").attr("class", "grid")
     .attr("x1", m.l).attr("x2", W - m.r).attr("y1", d => y(d)).attr("y2", d => y(d))
     .attr("stroke", GRID).attr("stroke-width", 0.5);
  svg.append("g").attr("transform", `translate(${m.l},0)`)
     .call(d3.axisLeft(y).ticks(6)).call(axisStyle);
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0))
     .attr("stroke", INK).attr("stroke-width", 1);
  svg.append("text").attr("x", 14).attr("y", (m.t + H - m.b) / 2)
     .attr("transform", `rotate(-90,14,${(m.t + H - m.b) / 2})`)
     .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", INK).text("R²");

  // gradient-boosting bars with value labels
  tiers.forEach(([t, l1, l2]) => {
    const v = DATA.tiers.histgb[t];
    const xx = x0(t), bw = x0.bandwidth();
    svg.append("rect").attr("x", xx).attr("y", v >= 0 ? y(v) : y(0)).attr("width", bw)
       .attr("height", Math.abs(y(Math.max(v, FLOOR)) - y(0)))
       .attr("fill", fillH).attr("stroke", OI.vermillion).attr("stroke-width", 0.9);
    svg.append("text").attr("x", xx + bw / 2).attr("y", v >= 0 ? y(v) - 5 : y(v) + 13)
       .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", "bold")
       .attr("fill", INK).text(d3.format("+.2f")(v));
    [l1, l2].forEach((ln, i) =>
      svg.append("text").attr("x", xx + bw / 2).attr("y", H - m.b + 16 + i * 12)
         .attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", INK).text(ln));
  });

  // ridge overlaid as a line across tiers (secondary model)
  const lr = d3.line().x(d => x0(d) + x0.bandwidth() / 2)
                .y(d => y(Math.max(DATA.tiers.ridge[d], FLOOR)));
  svg.append("path").attr("d", lr(tiers.map(t => t[0]))).attr("fill", "none")
     .attr("stroke", OI.blue).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
  tiers.forEach(([t]) => svg.append("circle").attr("cx", x0(t) + x0.bandwidth() / 2)
     .attr("cy", y(Math.max(DATA.tiers.ridge[t], FLOOR))).attr("r", 2.8).attr("fill", OI.blue));

  // per-delta LODO R2 points (jittered) at the LODO tier, clipped to floor
  const jit = d3.randomNormal.source(d3.randomLcg(11))(0, x0.bandwidth() / 8);
  let clipped = 0;
  (DATA.perdelta || []).forEach(d => {
    const cx = x0("t3") + x0.bandwidth() / 2 + jit();
    const off = d.r2 < FLOOR; if (off) clipped++;
    const cy = y(Math.max(d.r2, FLOOR));
    svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 2.6)
       .attr("fill", off ? "none" : INK).attr("stroke", INK).attr("stroke-width", 0.8)
       .attr("fill-opacity", 0.55);
  });
  svg.append("text").attr("x", x0("t3") + x0.bandwidth() / 2).attr("y", H - m.b - 4)
     .attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", "#424242")
     .text(`8 deltas${clipped ? ` (${clipped} below axis)` : ""}`);

  // inflation callout: a short arrow from the random-kfold bar top to the LODO bar
  defs.append("marker").attr("id", "arr").attr("viewBox", "0 0 10 10")
     .attr("refX", 8).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6)
     .attr("orient", "auto").append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", INK);
  const cxText = (x0("t2") + x0.bandwidth() / 2);
  svg.append("text").attr("x", cxText).attr("y", y(-1.0)).attr("text-anchor", "middle")
     .attr("font-size", 10).attr("font-style", "italic").attr("fill", INK)
     .text("conventional validation");
  svg.append("text").attr("x", cxText).attr("y", y(-1.0) + 13).attr("text-anchor", "middle")
     .attr("font-size", 10).attr("font-style", "italic").attr("fill", INK)
     .text("inflates skill 0.65 → −1.71");
  svg.append("path")
     .attr("d", `M${cxText + 70},${y(-1.05)} C${x0("t3") - 6},${y(-1.05)} ${x0("t3") - 6},${y(-1.5)} ${x0("t3") + 6},${y(-1.6)}`)
     .attr("fill", "none").attr("stroke", INK).attr("stroke-width", 1).attr("marker-end", "url(#arr)");

  // legend
  svg.append("rect").attr("x", W - m.r - 150).attr("y", m.t + 2).attr("width", 11).attr("height", 11)
     .attr("fill", fillH).attr("stroke", OI.vermillion).attr("stroke-width", 0.9);
  svg.append("text").attr("x", W - m.r - 135).attr("y", m.t + 11).attr("font-size", 10).attr("fill", INK)
     .text("gradient boosting");
  svg.append("line").attr("x1", W - m.r - 150).attr("x2", W - m.r - 139).attr("y1", m.t + 24).attr("y2", m.t + 24)
     .attr("stroke", OI.blue).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
  svg.append("text").attr("x", W - m.r - 135).attr("y", m.t + 27).attr("font-size", 10).attr("fill", INK).text("ridge");
  svg.append("circle").attr("cx", W - m.r - 145).attr("cy", m.t + 38).attr("r", 2.6)
     .attr("fill", INK).attr("fill-opacity", 0.55).attr("stroke", INK).attr("stroke-width", 0.8);
  svg.append("text").attr("x", W - m.r - 135).attr("y", m.t + 41).attr("font-size", 10).attr("fill", INK)
     .text("per-delta (LODO)");
  save(dom, "fig_tiers");
}

// ===================================================================== fig_aoa
function figAoa() {
  const W = 860, H = 330, m = { t: 36, r: 18, b: 46, l: 130 };
  const { dom, svg } = svgRoot(W, H);
  const defs = svg.append("defs");
  const fRmse = hatch(defs, "h-rmse", OI.vermillion, 45);
  const fAoa = hatch(defs, "h-aoa", OI.green, -45);
  const pd = DATA.perdelta.slice().sort((a, b) => a.rmse - b.rmse);
  const panelW = (W - m.l - m.r - 60) / 2;
  const y = d3.scaleBand().domain(pd.map(d => d.name)).range([m.t, H - m.b]).padding(0.28);

  // panel a: RMSE
  panelLabel(svg, m.l - 118, 22, "a");
  title(svg, m.l - 104, 22, "Per-delta error");
  const ax = m.l, x1 = d3.scaleLinear().domain([0, d3.max(pd, d => d.rmse)]).nice().range([ax, ax + panelW]);
  pd.forEach(d => svg.append("rect").attr("x", ax).attr("y", y(d.name))
     .attr("width", x1(d.rmse) - ax).attr("height", y.bandwidth())
     .attr("fill", fRmse).attr("stroke", OI.vermillion).attr("stroke-width", 0.7));
  svg.append("g").attr("transform", `translate(0,${H - m.b})`)
     .call(d3.axisBottom(x1).ticks(5)).call(axisStyle);
  svg.append("g").attr("transform", `translate(${ax},0)`).call(d3.axisLeft(y)).call(axisStyle)
     .select(".domain").remove();
  svg.append("text").attr("x", ax + panelW / 2).attr("y", H - 8).attr("text-anchor", "middle")
     .attr("font-size", 11).attr("fill", INK).text("LODO RMSE (Mg ha⁻¹)");

  // panel b: AOA inside
  const bx = ax + panelW + 60;
  panelLabel(svg, bx - 14, 22, "b");
  title(svg, bx, 22, "Inside AOA");
  const x2 = d3.scaleLinear().domain([0, 1]).range([bx, bx + panelW]);
  svg.append("rect").attr("x", bx).attr("y", m.t).attr("width", panelW).attr("height", H - m.b - m.t)
     .attr("fill", "#F5F5F5").attr("stroke", GRID).attr("stroke-width", 0.5);
  pd.forEach(d => {
    const w = Math.max(2, x2(d.aoa) - bx);
    svg.append("rect").attr("x", bx).attr("y", y(d.name)).attr("width", w).attr("height", y.bandwidth())
       .attr("fill", w > 3 ? fAoa : OI.green).attr("stroke", OI.green).attr("stroke-width", 0.7);
  });
  svg.append("g").attr("transform", `translate(0,${H - m.b})`)
     .call(d3.axisBottom(x2).ticks(3).tickFormat(d3.format(".0%"))).call(axisStyle);
  svg.append("text").attr("x", bx + panelW / 2).attr("y", H - 8).attr("text-anchor", "middle")
     .attr("font-size", 11).attr("fill", INK).text("fraction inside AOA");
  if (d3.max(DATA.perdelta, d => d.aoa) < 0.01)
    svg.append("text").attr("x", bx + panelW / 2).attr("y", (m.t + H - m.b) / 2)
       .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#9E9E9E")
       .text("all deltas ≈ 0%");
  save(dom, "fig_aoa");
}

// ================================================================= fig_fewshot
// Story: within-delta skill (correlation) is the star. From the pure leave-one-delta-out
// baseline (k=0), adding a handful of local cores recovers most of the lost skill. The
// shaded band is the skill recovered over that baseline; a local-only ridge shows the
// recovery is driven by the local data, not transfer. RMSE is a secondary right-axis line.
function figFewshot() {
  if (!DATA.fewshot) return;
  const W = 520, H = 360, m = { t: 56, r: 58, b: 52, l: 56 };
  const { dom, svg } = svgRoot(W, H);
  title(svg, m.l - 40, 22, "A few local cores recover most of the lost skill");
  svg.append("text").attr("x", m.l - 40).attr("y", 38).attr("font-size", 10.5)
     .attr("fill", AXIS).text("within-delta correlation for an otherwise-unsampled delta");
  const fs = DATA.fewshot;
  const base = fs[0].pearson;                      // k=0 pure-LODO baseline
  const x = d3.scaleLinear().domain(d3.extent(fs, d => d.k)).range([m.l, W - m.r]);
  const yP = d3.scaleLinear().domain([0, 0.75]).range([H - m.b, m.t]);
  const yR = d3.scaleLinear().domain([55, d3.max(fs, d => d.rmse) * 1.04]).nice().range([H - m.b, m.t]);

  svg.selectAll(".grid").data(yP.ticks(6)).join("line").attr("class", "grid")
     .attr("x1", m.l).attr("x2", W - m.r).attr("y1", d => yP(d)).attr("y2", d => yP(d))
     .attr("stroke", GRID).attr("stroke-width", 0.5);

  // recovered-skill band: between the k=0 baseline and the global+local curve
  const areaP = d3.area().x(d => x(d.k)).y0(yP(base)).y1(d => yP(d.pearson));
  svg.append("path").attr("d", areaP(fs)).attr("fill", OI.blue).attr("fill-opacity", 0.12);
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", yP(base)).attr("y2", yP(base))
     .attr("stroke", OI.blue).attr("stroke-width", 0.8).attr("stroke-dasharray", "2 3");

  // axes
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(yP).ticks(6)).call(axisStyle);
  svg.append("g").attr("transform", `translate(${W - m.r},0)`).call(d3.axisRight(yR).ticks(5)).call(axisStyle);
  svg.append("g").attr("transform", `translate(0,${H - m.b})`)
     .call(d3.axisBottom(x).tickValues(fs.map(d => d.k))).call(axisStyle);
  svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 10).attr("text-anchor", "middle")
     .attr("font-size", 11).attr("fill", INK).text("local calibration cores k");
  svg.append("text").attr("x", 15).attr("y", (m.t + H - m.b) / 2).attr("transform", `rotate(-90,15,${(m.t + H - m.b) / 2})`)
     .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", OI.blue).text("within-delta r");
  svg.append("text").attr("x", W - 13).attr("y", (m.t + H - m.b) / 2).attr("transform", `rotate(-90,${W - 13},${(m.t + H - m.b) / 2})`)
     .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", OI.vermillion).text("RMSE (Mg ha⁻¹)");

  const loc = fs.filter(d => d.pearson_local != null && !Number.isNaN(d.pearson_local));
  const lineP = d3.line().x(d => x(d.k)).y(d => yP(d.pearson));
  const lineLoc = d3.line().x(d => x(d.k)).y(d => yP(d.pearson_local));
  const lineR = d3.line().x(d => x(d.k)).y(d => yR(d.rmse));

  // RMSE secondary (faint)
  svg.append("path").attr("d", lineR(fs)).attr("fill", "none").attr("stroke", OI.vermillion)
     .attr("stroke-width", 1.4).attr("stroke-opacity", 0.55);
  fs.forEach(d => svg.append("circle").attr("cx", x(d.k)).attr("cy", yR(d.rmse)).attr("r", 2.4)
     .attr("fill", OI.vermillion).attr("fill-opacity", 0.55));

  // global+local (primary, solid blue)
  svg.append("path").attr("d", lineP(fs)).attr("fill", "none").attr("stroke", OI.blue).attr("stroke-width", 2.4);
  fs.forEach(d => svg.append("circle").attr("cx", x(d.k)).attr("cy", yP(d.pearson)).attr("r", 3.6)
     .attr("fill", OI.blue).attr("stroke", "white").attr("stroke-width", 0.8));
  // local-only ridge (orange dashed)
  if (loc.length) {
    svg.append("path").attr("d", lineLoc(loc)).attr("fill", "none").attr("stroke", OI.orange)
       .attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
    loc.forEach(d => svg.append("rect").attr("x", x(d.k) - 3).attr("y", yP(d.pearson_local) - 3)
       .attr("width", 6).attr("height", 6).attr("fill", OI.orange));
  }

  // annotations
  svg.append("circle").attr("cx", x(0)).attr("cy", yP(base)).attr("r", 3.6)
     .attr("fill", "none").attr("stroke", INK).attr("stroke-width", 1);
  svg.append("text").attr("x", x(0) + 6).attr("y", yP(base) + 14).attr("font-size", 9.5)
     .attr("fill", INK).text(`k=0: pure LODO (r=${base.toFixed(2)})`);
  const kr = fs.find(d => d.k === 25) || fs[fs.length - 1];
  svg.append("text").attr("x", x(kr.k) - 4).attr("y", yP(kr.pearson) - 8).attr("text-anchor", "end")
     .attr("font-size", 9.5).attr("font-style", "italic").attr("fill", OI.blue)
     .text(`r=${kr.pearson.toFixed(2)} at k=${kr.k}`);
  // lower-right wedge of the band: clear of the global+local and local-only lines
  svg.append("text").attr("x", x(19)).attr("y", yP(0.225)).attr("text-anchor", "middle")
     .attr("font-size", 9.5).attr("fill", OI.blue).attr("fill-opacity", 0.85)
     .text("recovered skill");

  // legend
  const lx = m.l + 8, ly0 = m.t + 2;
  const leg = [["global + local", OI.blue, "solid"], ["local only (ridge)", OI.orange, "dash"],
               ["RMSE (right axis)", OI.vermillion, "faint"]];
  leg.forEach(([t, c, s], i) => {
    const ly = ly0 + i * 14;
    svg.append("line").attr("x1", lx).attr("x2", lx + 16).attr("y1", ly).attr("y2", ly)
       .attr("stroke", c).attr("stroke-width", s === "solid" ? 2.4 : 1.8)
       .attr("stroke-dasharray", s === "dash" ? "5 3" : null)
       .attr("stroke-opacity", s === "faint" ? 0.55 : 1);
    svg.append("text").attr("x", lx + 22).attr("y", ly + 3.5).attr("font-size", 10).attr("fill", INK).text(t);
  });
  save(dom, "fig_fewshot");
}

// ================================================================== fig_region
// Per-region within-region correlation across the ~29 data-driven regions, by
// continent: shows the collapse holds at scale (most regions near or below zero).
function figRegion() {
  if (!DATA.region) return;
  const W = 620, H = 320, m = { t: 36, r: 16, b: 64, l: 50 };
  const { dom, svg } = svgRoot(W, H);
  title(svg, m.l - 34, 22, "Transfer collapse holds across 29 regions");
  const conts = [...new Set(DATA.region.map(d => d.continent))].sort();
  const cx = d3.scaleBand().domain(conts).range([m.l, W - m.r]).padding(0.4);
  const y = d3.scaleLinear().domain([-0.7, 0.7]).range([H - m.b, m.t]);
  const col = d3.scaleOrdinal().domain(conts)
    .range([OI.blue, OI.vermillion, OI.green, OI.orange, OI.sky, OI.purple, OI.grey]);
  // gridlines + zero line
  svg.selectAll(".g").data(y.ticks(7)).join("line").attr("class", "g")
     .attr("x1", m.l).attr("x2", W - m.r).attr("y1", d => y(d)).attr("y2", d => y(d))
     .attr("stroke", GRID).attr("stroke-width", 0.5);
  // global median within-region r + 95% bootstrap CI band (the tight overall bound)
  const rs = DATA.region_summary || {};
  const gmed = rs.lodo_median_pearson, gci = rs.lodo_median_pearson_ci;
  if (gci) {
    svg.append("rect").attr("x", m.l).attr("y", y(gci[1])).attr("width", W - m.r - m.l)
       .attr("height", y(gci[0]) - y(gci[1])).attr("fill", OI.vermillion).attr("fill-opacity", 0.10);
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(gmed)).attr("y2", y(gmed))
       .attr("stroke", OI.vermillion).attr("stroke-width", 1.4).attr("stroke-dasharray", "6 3");
    svg.append("text").attr("x", W - m.r - 2).attr("y", y(gci[1]) - 4).attr("text-anchor", "end")
       .attr("font-size", 9).attr("font-style", "italic").attr("fill", OI.vermillion)
       .text(`median r=${gmed.toFixed(2)} (95% CI ${gci[0].toFixed(2)} to ${gci[1].toFixed(2)})`);
  }
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0))
     .attr("stroke", INK).attr("stroke-width", 1);
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(7)).call(axisStyle);
  svg.append("text").attr("x", 13).attr("y", (m.t + H - m.b) / 2)
     .attr("transform", `rotate(-90,13,${(m.t + H - m.b) / 2})`).attr("text-anchor", "middle")
     .attr("font-size", 11).attr("fill", INK).text("within-region r (out-of-region)");
  const jitter = d3.randomNormal.source(d3.randomLcg(7))(0, cx.bandwidth() / 7);
  DATA.region.forEach(d => {
    svg.append("circle")
       .attr("cx", cx(d.continent) + cx.bandwidth() / 2 + jitter())
       .attr("cy", y(Math.max(-0.7, Math.min(0.7, d.pearson))))
       .attr("r", 3 + Math.sqrt(d.n) / 6).attr("fill", col(d.continent))
       .attr("fill-opacity", 0.78).attr("stroke", INK).attr("stroke-width", 0.4);
  });
  // median marker per continent
  conts.forEach(c => {
    const v = DATA.region.filter(d => d.continent === c).map(d => d.pearson).sort(d3.ascending);
    const med = d3.median(v);
    svg.append("line").attr("x1", cx(c) + 4).attr("x2", cx(c) + cx.bandwidth() - 4)
       .attr("y1", y(med)).attr("y2", y(med)).attr("stroke", INK).attr("stroke-width", 1.6);
  });
  conts.forEach(c => svg.append("text").attr("x", cx(c) + cx.bandwidth() / 2)
    .attr("y", H - m.b + 16).attr("text-anchor", "middle").attr("font-size", 10)
    .attr("fill", INK).attr("transform", `rotate(12,${cx(c) + cx.bandwidth() / 2},${H - m.b + 16})`)
    .text(c));
  svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle")
     .attr("font-size", 10).attr("fill", "#9E9E9E")
     .text("each point = one held-out region (size ∝ √n); black bar = continental median; "
         + "red band = 95% CI of overall median");
  save(dom, "fig_region");
}

// ================================================================== fig_biomes
// Cross-biome transfer under one protocol: per-region out-of-region r (points),
// median with 95% CI (bar + band), and the replicate-core ceiling (hollow marker).
function figBiomes() {
  if (!DATA.biomes) return;
  const B = DATA.biomes.filter(b => !b.underpowered);
  const W = 680, H = 360, m = { t: 40, r: 16, b: 78, l: 52 };
  const { dom, svg } = svgRoot(W, H);
  title(svg, m.l - 36, 22, "Global models transfer in mineral soils, not in carbon-dense wetland soils");
  const short = { terrestrial_conc_d30: "Terrestrial\nmineral (conc.)", terrestrial_stock_d30: "Terrestrial\nmineral (stock)",
                  mangrove: "Mangrove", marsh_d30: "Salt marsh", seagrass_d30: "Seagrass", permafrost_d100: "Permafrost" };
  const x = d3.scaleBand().domain(B.map(b => b.tag)).range([m.l, W - m.r]).padding(0.35);
  const y = d3.scaleLinear().domain([-0.85, 1.05]).range([H - m.b, m.t]);
  svg.selectAll(".g").data(y.ticks(7)).join("line").attr("class", "g")
     .attr("x1", m.l).attr("x2", W - m.r).attr("y1", d => y(d)).attr("y2", d => y(d))
     .attr("stroke", GRID).attr("stroke-width", 0.5);
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0))
     .attr("stroke", INK).attr("stroke-width", 1);
  // class shading: mineral vs organic
  const firstOrg = B.findIndex(b => b.cls === "organic");
  if (firstOrg > 0) {
    const xs = x(B[firstOrg].tag) - x.step() * x.paddingInner() / 2;
    svg.append("rect").attr("x", xs).attr("y", m.t).attr("width", W - m.r - xs).attr("height", H - m.b - m.t)
       .attr("fill", OI.vermillion).attr("fill-opacity", 0.05);
    svg.append("text").attr("x", (m.l + xs) / 2).attr("y", H - m.b - 6).attr("text-anchor", "middle")
       .attr("font-size", 10).attr("fill", OI.blue).text("mineral upland soils");
    svg.append("text").attr("x", (xs + W - m.r) / 2).attr("y", m.t + 12).attr("text-anchor", "middle")
       .attr("font-size", 10).attr("fill", OI.vermillion).text("carbon-dense wetland and organic soils");
  }
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(7)).call(axisStyle);
  svg.append("text").attr("x", 13).attr("y", (m.t + H - m.b) / 2)
     .attr("transform", `rotate(-90,13,${(m.t + H - m.b) / 2})`).attr("text-anchor", "middle")
     .attr("font-size", 11).attr("fill", INK).text("within-region r (out-of-region prediction)");
  const jitter = d3.randomNormal.source(d3.randomLcg(11))(0, x.bandwidth() / 8);
  B.forEach(b => {
    const cx = x(b.tag) + x.bandwidth() / 2, col = b.cls === "mineral" ? OI.blue : OI.vermillion;
    // CI band + median bar
    if (b.ci) svg.append("rect").attr("x", x(b.tag)).attr("y", y(b.ci[1])).attr("width", x.bandwidth())
       .attr("height", y(b.ci[0]) - y(b.ci[1])).attr("fill", col).attr("fill-opacity", 0.15);
    b.regions.forEach(d => svg.append("circle").attr("cx", cx + jitter())
       .attr("cy", y(Math.max(-0.85, Math.min(1.05, d.r)))).attr("r", 2.6 + Math.sqrt(d.n) / 8)
       .attr("fill", col).attr("fill-opacity", 0.55).attr("stroke", INK).attr("stroke-width", 0.35));
    svg.append("line").attr("x1", x(b.tag) + 3).attr("x2", x(b.tag) + x.bandwidth() - 3)
       .attr("y1", y(b.median_r)).attr("y2", y(b.median_r)).attr("stroke", INK).attr("stroke-width", 2);
    // ceiling: hollow diamond
    if (b.ceiling != null) {
      const cy = y(b.ceiling), s = 6;
      svg.append("path").attr("d", `M${cx},${cy - s}L${cx + s},${cy}L${cx},${cy + s}L${cx - s},${cy}Z`)
         .attr("fill", "white").attr("stroke", INK).attr("stroke-width", 1.2);
      svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", y(b.median_r)).attr("y2", cy - s)
         .attr("stroke", INK).attr("stroke-width", 0.7).attr("stroke-dasharray", "2 2");
    }
    // x label (two lines) + n
    const lab = (short[b.tag] || b.label).split("\n");
    lab.forEach((t, i) => svg.append("text").attr("x", cx).attr("y", H - m.b + 16 + i * 12)
       .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", INK).text(t));
    svg.append("text").attr("x", cx).attr("y", H - m.b + 16 + lab.length * 12)
       .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#9E9E9E")
       .text(`n=${b.n.toLocaleString()}, ${b.n_regions} regions`);
  });
  svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle")
     .attr("font-size", 10).attr("fill", "#9E9E9E")
     .text("points = held-out regions (size ∝ √n); bar = median, band = 95% CI; ◇ = replicate-core ceiling on attainable r");
  save(dom, "fig_biomes");
}

// ================================================================ fig_protocol
// Certification decision tree (four steps -> four verdicts), with the mangrove
// benchmark's own numbers annotated at each step.
function figProtocol() {
  const W = 700, H = 440, { dom, svg } = svgRoot(W, H);
  title(svg, 16, 22, "Certifying a spatial prediction before it is used at an unsampled location");
  const box = (x, y, w, h, lines, fill, stroke = INK, size = 12, bold = false) => {
    svg.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 6)
       .attr("fill", fill).attr("stroke", stroke).attr("stroke-width", 1);
    lines.forEach((t, i) => svg.append("text").attr("x", x + w / 2)
       .attr("y", y + h / 2 - (lines.length - 1) * 7.5 + i * 15).attr("text-anchor", "middle")
       .attr("font-size", size).attr("font-weight", bold ? "bold" : "normal").attr("fill", INK).text(t));
  };
  const arrow = (x1, y1, x2, y2, label) => {
    svg.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
       .attr("stroke", INK).attr("stroke-width", 1).attr("marker-end", "url(#arr)");
    if (label) svg.append("text").attr("x", (x1 + x2) / 2 + 5).attr("y", (y1 + y2) / 2 - 3)
       .attr("font-size", 11).attr("font-style", "italic").attr("fill", AXIS).text(label);
  };
  svg.append("defs").append("marker").attr("id", "arr").attr("viewBox", "0 0 10 10").attr("refX", 9)
     .attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
     .append("path").attr("d", "M0,0L10,5L0,10Z").attr("fill", INK);
  const q = "#F5F5F5", v = { usable: "#E3F2FD", level: "#FFF8E1", local: "#FFEBEE" };
  const L = 16, QW = 340, QH = 50, RX = 392, RW = 296;
  // steps
  box(L, 50, QW, QH, ["1  Inside the AOA paired with random-validation skill?",
                      "mangroves: threshold 0.001; 0% of any unsampled delta"], q);
  box(L, 132, QW, QH, ["2  Inside the AOA paired with the out-of-region error?",
                       "threshold 0.26; median 94% of a held-out delta"], q);
  box(L, 214, QW, QH, ["3  Out-of-region skill ≥ half the replicate ceiling?",
                       "r = −0.09 [−0.20, 0.03] against a ceiling of 0.74"], q);
  box(L, 296, QW, QH, ["4  How many local cores close the gap?",
                       "few-shot curve: 10 cores reach half the ceiling"], q);
  arrow(L + QW / 2, 100, L + QW / 2, 132, "no: reported R² does not apply here (expected)");
  arrow(L + QW / 2, 182, L + QW / 2, 214, "yes");
  arrow(L + QW / 2, 264, L + QW / 2, 296, "no");
  // verdicts
  box(RX, 130, RW, 50, ["LOCAL CORES REQUIRED", "extrapolation: no error estimate applies"], v.local, INK, 12, true);
  arrow(L + QW, 155, RX, 155, "no");
  box(RX, 212, RW, 50, ["USABLE", "out-of-region error applies; pattern recovered"], v.usable, INK, 12, true);
  arrow(L + QW, 237, RX, 237, "yes");
  box(RX, 294, RW, 50, ["LEVEL ONLY / LOCAL CORES FOR PATTERN", "k cores from the few-shot curve"], v.level, INK, 11, true);
  arrow(L + QW, 319, RX, 319, "");
  // footer
  svg.append("text").attr("x", 16).attr("y", 380).attr("font-size", 10.5).attr("fill", "#9E9E9E")
     .text("Every quantity comes from the model's own training table: two AOA thresholds (folds matched to the error each certifies),");
  svg.append("text").attr("x", 16).attr("y", 396).attr("font-size", 10.5).attr("fill", "#9E9E9E")
     .text("leave-one-region-out skill with a bootstrap CI, the replicate-core ceiling √ICC and the few-shot calibration curve (open code).");
  svg.append("text").attr("x", 16).attr("y", 416).attr("font-size", 10.5).attr("fill", "#9E9E9E")
     .text("Passing step 2 while failing step 3 is the mangrove, salt-marsh, seagrass and permafrost outcome; mineral upland soils pass step 3.");
  save(dom, "fig_protocol");
}

// ================================================================== composites
// Multi-panel figures for the journal builds: each panel is a previously saved SVG,
// nested with its own viewBox so typography is identical to the single figures.
function compose(name, panels, { gap = 18, pad = 10 } = {}) {
  // panels: [{file, x, y, w, h, label}] in outer units
  const W = Math.max(...panels.map(p => p.x + p.w)) + pad, H = Math.max(...panels.map(p => p.y + p.h)) + pad;
  const { dom, svg } = svgRoot(W, H);
  for (const p of panels) {
    const src = readFileSync(`${OUT}/${p.file}.svg`, "utf8");
    const inner = new JSDOM(src).window.document.querySelector("svg");
    const vb = inner.getAttribute("viewBox");
    const g = svg.append("svg").attr("x", p.x).attr("y", p.y).attr("width", p.w).attr("height", p.h)
                 .attr("viewBox", vb).attr("preserveAspectRatio", "xMinYMin meet");
    g.node().innerHTML = inner.innerHTML;
    svg.append("text").attr("x", p.lx ?? p.x + 4).attr("y", p.y + 16).attr("font-size", 16)
       .attr("font-weight", "bold").attr("fill", INK).text(p.label);
  }
  save(dom, name);
}
function figComposites() {
  // Fig 1: map (920x430) over tiers (520x360)
  compose("fig1_benchmark", [
    { file: "fig_map",   x: 10, y: 10,  w: 920, h: 430, label: "a" },
    { file: "fig_tiers", x: 10, y: 456, w: 920, h: 637, label: "b" }]);
  // Fig 2: AoA per delta (860x330) over 29 regions (620x320 -> scaled to 860 wide)
  compose("fig2_aoa_regions", [
    { file: "fig_aoa",    x: 10, y: 10,  w: 860, h: 330, label: "a" },
    { file: "fig_region", x: 10, y: 356, w: 860, h: 444, label: "b" }]);
  // Fig 4: few-shot (520x360 -> 700 wide) over protocol (700x420)
  compose("fig4_fix", [
    { file: "fig_fewshot",  x: 100, y: 10,  w: 520, h: 360, label: "a", lx: 14 },
    { file: "fig_protocol", x: 10,  y: 386, w: 700, h: 440, label: "b" }]);
}

import { mkdirSync } from "fs";
mkdirSync(OUT, { recursive: true });
figMap(); figTiers(); figAoa(); figFewshot(); figRegion(); figBiomes(); figProtocol(); // figComposites();  // journal figures now come from build_journal_figures.mjs
console.log("done.");
