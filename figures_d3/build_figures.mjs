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

  // size legend
  const lx = m.l + 8, ly = H - 70;
  svg.append("text").attr("x", lx).attr("y", ly - 8).attr("font-size", 10)
     .attr("fill", AXIS).text("mangrove area (km²)");
  [500, 2000, 6000].forEach((v, i) => {
    const cx = lx + 14 + i * 52;
    svg.append("circle").attr("cx", cx).attr("cy", ly + 14).attr("r", r(v))
       .attr("fill", "none").attr("stroke", AXIS).attr("stroke-width", 0.8);
    svg.append("text").attr("x", cx).attr("y", ly + 36).attr("text-anchor", "middle")
       .attr("font-size", 9).attr("fill", AXIS).text(v);
  });
  // color legend
  const gw = 150, gx = W - m.r - gw - 8, gy = H - 52;
  const defs = svg.append("defs");
  const grad = defs.append("linearGradient").attr("id", "soc").attr("x1", "0").attr("x2", "1");
  d3.range(0, 1.01, 0.1).forEach(t =>
    grad.append("stop").attr("offset", `${t * 100}%`).attr("stop-color", SEQ(t)));
  svg.append("rect").attr("x", gx).attr("y", gy).attr("width", gw).attr("height", 10)
     .attr("fill", "url(#soc)").attr("stroke", AXIS).attr("stroke-width", 0.5);
  svg.append("text").attr("x", gx).attr("y", gy - 5).attr("font-size", 10).attr("fill", AXIS)
     .text("median SOC₀₋₁₀₀ (Mg ha⁻¹)");
  [cext[0], cext[1]].forEach((v, i) =>
    svg.append("text").attr("x", gx + i * gw).attr("y", gy + 22)
       .attr("text-anchor", i ? "end" : "start").attr("font-size", 9).attr("fill", AXIS)
       .text(Math.round(v)));
  save(dom, "fig_map");
}

// =================================================================== fig_tiers
function figTiers() {
  const W = 440, H = 320, m = { t: 36, r: 14, b: 56, l: 50 };
  const { dom, svg } = svgRoot(W, H);
  const defs = svg.append("defs");
  title(svg, m.l - 34, 22, "Skill collapses out-of-distribution");
  const tiers = [["t1", "Random\nk-fold"], ["t2", "Spatial\nblock"], ["t3", "LODO\n(median)"]];
  const models = [["ridge", OI.blue], ["histgb", OI.vermillion]];
  const fill = { ridge: hatch(defs, "h-ridge", OI.blue, 45),
                 histgb: hatch(defs, "h-histgb", OI.vermillion, -45) };
  const x0 = d3.scaleBand().domain(tiers.map(t => t[0])).range([m.l, W - m.r]).padding(0.3);
  const x1 = d3.scaleBand().domain(models.map(d => d[0])).range([0, x0.bandwidth()]).padding(0.12);
  const vals = models.flatMap(([mm]) => tiers.map(([t]) => DATA.tiers[mm][t]));
  const y = d3.scaleLinear().domain([Math.min(-2, d3.min(vals)) - 0.2, Math.max(0.8, d3.max(vals))])
              .nice().range([H - m.b, m.t]);

  svg.append("g").attr("transform", `translate(${m.l},0)`)
     .call(d3.axisLeft(y).ticks(6)).call(axisStyle);
  svg.selectAll(".grid").data(y.ticks(6)).join("line").attr("class", "grid")
     .attr("x1", m.l).attr("x2", W - m.r).attr("y1", d => y(d)).attr("y2", d => y(d))
     .attr("stroke", GRID).attr("stroke-width", 0.5);
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0))
     .attr("stroke", INK).attr("stroke-width", 1);
  svg.append("text").attr("x", 14).attr("y", (m.t + H - m.b) / 2)
     .attr("transform", `rotate(-90,14,${(m.t + H - m.b) / 2})`)
     .attr("text-anchor", "middle").attr("font-size", 12).attr("fill", INK).text("R²");

  tiers.forEach(([t, lab]) => {
    models.forEach(([mm, col]) => {
      const v = DATA.tiers[mm][t];
      const xx = x0(t) + x1(mm), yy = v >= 0 ? y(v) : y(0);
      svg.append("rect").attr("x", xx).attr("y", yy).attr("width", x1.bandwidth())
         .attr("height", Math.abs(y(v) - y(0))).attr("fill", fill[mm])
         .attr("stroke", col).attr("stroke-width", 0.8);
    });
    lab.split("\n").forEach((ln, i) =>
      svg.append("text").attr("x", x0(t) + x0.bandwidth() / 2).attr("y", H - m.b + 16 + i * 12)
         .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", INK).text(ln));
  });
  // legend
  models.forEach(([mm, col], i) => {
    const lx = W - m.r - 96, ly = m.t + 4 + i * 16;
    svg.append("rect").attr("x", lx).attr("y", ly - 9).attr("width", 11).attr("height", 11)
       .attr("fill", fill[mm]).attr("stroke", col).attr("stroke-width", 0.8);
    svg.append("text").attr("x", lx + 16).attr("y", ly).attr("font-size", 11).attr("fill", INK)
       .text(mm === "histgb" ? "gradient boosting" : "ridge");
  });
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
  svg.append("text").attr("x", bx + panelW / 2).attr("y", (m.t + H - m.b) / 2)
     .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#777")
     .text("all deltas ≈ 0%");
  save(dom, "fig_aoa");
}

// ================================================================= fig_fewshot
function figFewshot() {
  if (!DATA.fewshot) return;
  const W = 460, H = 330, m = { t: 36, r: 54, b: 48, l: 56 };
  const { dom, svg } = svgRoot(W, H);
  title(svg, m.l - 40, 22, "Few-shot calibration of an unsampled delta");
  const fs = DATA.fewshot;
  const x = d3.scaleLinear().domain(d3.extent(fs, d => d.k)).range([m.l, W - m.r]);
  const yL = d3.scaleLinear().domain([0, d3.max(fs, d => d.rmse) * 1.1]).nice().range([H - m.b, m.t]);
  const yR = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);

  svg.selectAll(".grid").data(yL.ticks(5)).join("line").attr("class", "grid")
     .attr("x1", m.l).attr("x2", W - m.r).attr("y1", d => yL(d)).attr("y2", d => yL(d))
     .attr("stroke", GRID).attr("stroke-width", 0.5);
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(yL).ticks(5)).call(axisStyle);
  svg.append("g").attr("transform", `translate(${W - m.r},0)`).call(d3.axisRight(yR).ticks(5)).call(axisStyle);
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(fs.length)).call(axisStyle);
  svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 8).attr("text-anchor", "middle")
     .attr("font-size", 11).attr("fill", INK).text("local calibration cores k");
  svg.append("text").attr("x", 14).attr("y", (m.t + H - m.b) / 2).attr("transform", `rotate(-90,14,${(m.t + H - m.b) / 2})`)
     .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", OI.vermillion).text("RMSE (Mg ha⁻¹)");
  svg.append("text").attr("x", W - 12).attr("y", (m.t + H - m.b) / 2).attr("transform", `rotate(-90,${W - 12},${(m.t + H - m.b) / 2})`)
     .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", OI.blue).text("within-delta r");

  const lineR = d3.line().x(d => x(d.k)).y(d => yL(d.rmse));
  const lineP = d3.line().x(d => x(d.k)).y(d => yR(d.pearson));
  // local-only correlation (where defined): the baseline that stays near zero
  const loc = fs.filter(d => d.pearson_local != null && !Number.isNaN(d.pearson_local));
  const lineLoc = d3.line().x(d => x(d.k)).y(d => yR(d.pearson_local));
  svg.append("path").attr("d", lineR(fs)).attr("fill", "none").attr("stroke", OI.vermillion).attr("stroke-width", 2);
  svg.append("path").attr("d", lineP(fs)).attr("fill", "none").attr("stroke", OI.blue)
     .attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
  if (loc.length) svg.append("path").attr("d", lineLoc(loc)).attr("fill", "none")
     .attr("stroke", OI.grey).attr("stroke-width", 1.6).attr("stroke-dasharray", "2 2");
  fs.forEach(d => {
    svg.append("circle").attr("cx", x(d.k)).attr("cy", yL(d.rmse)).attr("r", 3.2).attr("fill", OI.vermillion);
    svg.append("rect").attr("x", x(d.k) - 3).attr("y", yR(d.pearson) - 3).attr("width", 6).attr("height", 6).attr("fill", OI.blue);
  });
  loc.forEach(d => svg.append("circle").attr("cx", x(d.k)).attr("cy", yR(d.pearson_local))
     .attr("r", 2.6).attr("fill", OI.grey));
  // legend
  [["global + local", OI.blue], ["local only", OI.grey]].forEach(([t, c], i) =>
    svg.append("text").attr("x", m.l + 8).attr("y", m.t + 4 + i * 13).attr("font-size", 9)
       .attr("fill", c).text(t + " (r)"));
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
     .text("each point = one held-out region (size ∝ √n); bar = continental median");
  save(dom, "fig_region");
}

import { mkdirSync } from "fs";
mkdirSync(OUT, { recursive: true });
figMap(); figTiers(); figAoa(); figFewshot(); figRegion();
console.log("done.");
