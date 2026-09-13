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
const INK = "#1a1a1a", GRID = "#d9d9d9", AXIS = "#555";
// Okabe-Ito colorblind-safe palette
const OI = { blue: "#0072B2", vermillion: "#D55E00", green: "#009E73",
             orange: "#E69F00", sky: "#56B4E9", purple: "#CC79A7", grey: "#999999" };
const SEQ = d3.interpolateViridis;

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
    .attr("fill", "#f4f7fb").attr("stroke", GRID).attr("stroke-width", 0.6);
  g.append("path").attr("d", path(land)).attr("fill", "#e9ecef")
    .attr("stroke", "#cfd4da").attr("stroke-width", 0.4);

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
      .attr("fill", color(d.soc_med)).attr("stroke", "#111").attr("stroke-width", 0.8)
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
     .attr("height", H - m.b - m.t).attr("fill", "#f0f6fb");
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
       .attr("fill", off ? "none" : "#333").attr("stroke", "#333").attr("stroke-width", 0.8)
       .attr("fill-opacity", 0.55);
  });
  svg.append("text").attr("x", x0("t3") + x0.bandwidth() / 2).attr("y", H - m.b - 4)
     .attr("text-anchor", "middle").attr("font-size", 8.5).attr("fill", "#555")
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
     .attr("fill", "#333").attr("fill-opacity", 0.55).attr("stroke", "#333").attr("stroke-width", 0.8);
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
     .attr("fill", "#f4f7fb").attr("stroke", GRID).attr("stroke-width", 0.5);
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
       .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#777")
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
       .attr("fill-opacity", 0.78).attr("stroke", "#222").attr("stroke-width", 0.4);
  });
  // median marker per continent
  conts.forEach(c => {
    const v = DATA.region.filter(d => d.continent === c).map(d => d.pearson).sort(d3.ascending);
    const med = d3.median(v);
    svg.append("line").attr("x1", cx(c) + 4).attr("x2", cx(c) + cx.bandwidth() - 4)
       .attr("y1", y(med)).attr("y2", y(med)).attr("stroke", "#000").attr("stroke-width", 1.6);
  });
  conts.forEach(c => svg.append("text").attr("x", cx(c) + cx.bandwidth() / 2)
    .attr("y", H - m.b + 16).attr("text-anchor", "middle").attr("font-size", 10)
    .attr("fill", INK).attr("transform", `rotate(12,${cx(c) + cx.bandwidth() / 2},${H - m.b + 16})`)
    .text(c));
  svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle")
     .attr("font-size", 10).attr("fill", "#777")
     .text("each point = one held-out region (size ∝ √n); black bar = continental median; "
         + "red band = 95% CI of overall median");
  save(dom, "fig_region");
}

import { mkdirSync } from "fs";
mkdirSync(OUT, { recursive: true });
figMap(); figTiers(); figAoa(); figFewshot(); figRegion();
console.log("done.");
