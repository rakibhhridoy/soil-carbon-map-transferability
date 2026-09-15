// Journal figures (Nature / Science builds), server-side D3 -> SVG. House style:
// Helvetica, 6-7 pt type, bold lower-case panel letters, no in-panel titles, solid fills,
// one Okabe-Ito palette, thin axes, no gridlines. Drawn at the referee-build text width
// (160 mm = 454 pt, 1 unit = 1 pt) so type sizes are real points in the PDF.
// Data: data/processed/figure_data.json (src/manuscript_assets.py) and
// figure_data_extra.json (src/figure_data_extra.py). No number is computed here except
// bootstrap intervals for the per-delta correlations in Fig. 1c (Pearson r on log stock,
// the reported metric, from the per-core layer of the same soc_lodo.py run).
import { JSDOM } from "jsdom";
import * as d3 from "d3";
import { feature } from "topojson-client";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);

const DATA = JSON.parse(readFileSync("../data/processed/figure_data.json", "utf8"));
const X = JSON.parse(readFileSync("../data/processed/figure_data_extra.json", "utf8"));
const RL = JSON.parse(readFileSync("../data/processed/region_lodo_results.json", "utf8"));
const VP = JSON.parse(readFileSync("../data/processed/variance_partition.json", "utf8"));
const T1 = JSON.parse(readFileSync("../data/processed/tier1_inventory.json", "utf8"));
const FS = JSON.parse(readFileSync("../data/processed/fewshot_calibration.json", "utf8"));
const PM = JSON.parse(readFileSync("../data/processed/published_map_test.json", "utf8"));
// Two finishes from one code path: FIG_STYLE=subtle (submission, default) or showcase
// (talks, cover letter, press). Every effect is pure vector: shadows are stacked offset
// translucent copies, fades are stacked translucent bands (rsvg-convert rasterises SVG
// filters and its gradient shadings do not survive \includegraphics).
const STYLE = process.env.FIG_STYLE === "showcase" ? "showcase" : "subtle";
const FX = STYLE === "showcase"
  ? { layers: 8, step: 0.62, alpha: 0.06, halo: 2.0, hi: 0.6, ocean: "#EAF0F2", land: "#ECECEC", landEdge: 0.7, extrude: 7, card: 1.7 }
  : { layers: 3, step: 0.34, alpha: 0.055, halo: 1.6, hi: 0.4, ocean: "#F5F8F9", land: "#ECECEC", landEdge: 0.5, extrude: 0, card: 1.5 };
const OUT = STYLE === "showcase" ? "svg/showcase" : "svg";

const FONT = "Helvetica, Arial, sans-serif";
// One palette for every figure: red against charcoal, with amber, brown and the greys for
// secondary series. No blue and no teal anywhere, and no red/green pairing.
const PAL = { red: "#C62828", charcoal: "#37474F", amber: "#F9A825", orange: "#E65100",
              brown: "#5D4037", greygreen: "#607D63",
              ink: "#212121", grey: "#424242", mid: "#9E9E9E", light: "#E0E0E0" };
// Red carries what does not transfer, charcoal what transfers or is attainable.
const FAIL = PAL.red, HOLD = PAL.charcoal;
const INK = PAL.ink, AXIS = PAL.ink, MUTE = PAL.grey, LIGHT = PAL.light;
const FS_T = 7, FS_S = 6, FS_L = 8;     // text, small, panel letter
const TW = 454;                          // text width in pt

function svgRoot(w, h) {
  const dom = new JSDOM("<!DOCTYPE html><body></body>");
  const svg = d3.select(dom.window.document.body).append("svg")
    .attr("xmlns", "http://www.w3.org/2000/svg").attr("width", w).attr("height", h)
    .attr("viewBox", `0 0 ${w} ${h}`).attr("font-family", FONT).attr("font-size", FS_T);
  svg.append("rect").attr("width", w).attr("height", h).attr("fill", "white");
  return { dom, svg };
}
function save(dom, name) {
  mkdirSync(OUT, { recursive: true });
  writeFileSync(`${OUT}/${name}.svg`, dom.window.document.body.innerHTML);
  console.log("  wrote", `${OUT}/${name}.svg`);
}
function letter(svg, x, y, t) {
  svg.append("text").attr("x", x).attr("y", y).attr("font-size", FS_L).attr("font-weight", "bold")
     .attr("fill", INK).text(t);
}
function axis(g, ax, { size = FS_T, color = AXIS } = {}) {
  g.call(ax);
  g.selectAll("path,line").attr("stroke", color).attr("stroke-width", 0.5);
  g.selectAll("text").attr("fill", INK).attr("font-size", size).attr("font-family", FONT);
  g.select(".domain").attr("stroke", color);
  return g;
}
function txt(g, x, y, t, { size = FS_T, anchor = "start", color = INK, weight = "normal", rotate = 0, style = "normal", halo = false } = {}) {
  const mk = () => {
    const e = g.append("text").attr("x", x).attr("y", y).attr("font-size", size).attr("text-anchor", anchor)
       .attr("fill", color).attr("font-weight", weight).attr("font-style", style).text(t);
    if (rotate) e.attr("transform", `rotate(${rotate},${x},${y})`);
    return e;
  };
  // halo: a white stroked copy underneath keeps labels legible where they sit on data
  if (halo) mk().attr("fill", "white").attr("stroke", "white").attr("stroke-width", FX.halo).attr("stroke-linejoin", "round");
  return mk();
}
function ylabel(g, x, y, t) { txt(g, x, y, t, { anchor: "middle", rotate: -90 }); }
function hline(g, x1, x2, y, color = AXIS, dash = null, w = 0.5) {
  const l = g.append("line").attr("x1", x1).attr("x2", x2).attr("y1", y).attr("y2", y)
     .attr("stroke", color).attr("stroke-width", w);
  if (dash) l.attr("stroke-dasharray", dash);
  return l;
}
// soft drop shadow: `make(parent)` appends the silhouette; copies are offset down-right,
// widened by a growing stroke and drawn at low opacity, largest first
function dropShadow(g, make, k = 1) {
  const L = Math.max(1, Math.round(FX.layers * Math.min(1, 0.35 + k * 0.65)));
  for (let i = L; i >= 1; i--) {
    const d = i * FX.step * k;
    const grp = g.append("g").attr("transform", `translate(${(d * 0.45).toFixed(2)},${d.toFixed(2)})`).attr("opacity", FX.alpha);
    const e = make(grp), open = e.attr("fill") === "none";
    const sw = open ? (parseFloat(e.attr("data-w") || 1) + d * 0.9) : d * 0.9;
    e.attr("fill", open ? "none" : INK).attr("stroke", INK).attr("stroke-width", sw.toFixed(2))
      .attr("stroke-linejoin", "round").attr("stroke-linecap", "round").attr("opacity", null).attr("fill-opacity", null);
  }
}
// thin light line along the top edge of a filled shape
function topHighlight(g, x, y, w, alpha = FX.hi) {
  if (w <= 1.2) return;
  g.append("line").attr("x1", x + 0.5).attr("x2", x + w - 0.5).attr("y1", y + 0.45).attr("y2", y + 0.45)
    .attr("stroke", "white").attr("stroke-width", 0.7).attr("opacity", alpha);
}
// interval band that deepens toward its centre value (nested translucent rects)
function nestedBand(g, x, w, yTop, yBot, yMid, color, total, n = 18) {
  g.append("rect").attr("x", x).attr("y", yTop).attr("width", w).attr("height", yBot - yTop).attr("fill", color).attr("opacity", total * 0.45);
  for (let i = 1; i <= n; i++) {
    const t = i / (n + 1), y1 = yTop + (yMid - yTop) * t, y2 = yBot + (yMid - yBot) * t;
    g.append("rect").attr("x", x).attr("y", y1).attr("width", w).attr("height", Math.max(0.2, y2 - y1)).attr("fill", color).attr("opacity", total * 0.55 / n);
  }
}
// vertical fade inside an arbitrary area (clip path + stacked bands, strongest at the top)
let CLIP = 0;
function areaFade(svg, g, d, yTop, yBot, x0, x1, color, alpha, n = 18) {
  const id = `clip${STYLE}${++CLIP}`;
  svg.append("defs").append("clipPath").attr("id", id).append("path").attr("d", d);
  const cg = g.append("g").attr("clip-path", `url(#${id})`);
  for (let i = 0; i < n; i++)
    cg.append("rect").attr("x", x0).attr("y", yTop).attr("width", x1 - x0).attr("height", (yBot - yTop) * (i + 1) / n)
      .attr("fill", color).attr("opacity", alpha / n);
}
const fmt2 = d3.format("+.2f"), fmt1 = d3.format(".1f");
// signed two-decimal value with a true minus and no "+" (protocol text); -0.00 prints as 0.00
const fmtS = v => { const t = d3.format(".2f")(Math.abs(v) < 0.005 ? 0 : v); return t.replace("-", "−"); };
const rng = d3.randomLcg(7);
function pearson(a, b) {
  const n = a.length, ma = d3.mean(a), mb = d3.mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const da = a[i] - ma, db = b[i] - mb; sab += da * db; saa += da * da; sbb += db * db; }
  return sab / Math.sqrt(saa * sbb);
}
function bootCI(a, b, B = 2000) {
  const n = a.length, rs = [];
  for (let k = 0; k < B; k++) {
    const ia = [], ib = [];
    for (let i = 0; i < n; i++) { const j = Math.floor(rng() * n); ia.push(a[j]); ib.push(b[j]); }
    const r = pearson(ia, ib); if (Number.isFinite(r)) rs.push(r);
  }
  rs.sort(d3.ascending);
  return [d3.quantileSorted(rs, 0.025), d3.quantileSorted(rs, 0.975)];
}
const DELTA_NAME = Object.fromEntries(DATA.deltas.map(d => [d.id, d.name]));
const CONT_ORDER = ["N.America", "S.America", "Africa", "Asia", "Oceania", "other"];
const CONT_LABEL = { "N.America": "N. America", "S.America": "S. America", Africa: "Africa", Asia: "Asia", Oceania: "Oceania", other: "Other" };

// ===================================================================== Fig. 1
// a  cores, regions and deltas on a world map; b validation tiers; c per-delta r with CI
function fig1() {
  const W = TW, H = 372, { dom, svg } = svgRoot(W, H);
  // ---- a: map
  const mh = 214, m = { t: 12, l: 0, r: 0, b: 4 };
  letter(svg, 2, 9, "a");
  const topo = require("world-atlas/land-110m.json");
  const land = feature(topo, topo.objects.land);
  const proj = d3.geoNaturalEarth1().fitExtent([[m.l, m.t], [W - m.r, mh - m.b]], { type: "Sphere" });
  const path = d3.geoPath(proj);
  const g = svg.append("g");
  const sphereD = path({ type: "Sphere" }), landD = path(land);
  if (STYLE === "showcase") dropShadow(g, s => s.append("path").attr("d", sphereD), 1.6);
  g.append("path").attr("d", sphereD).attr("fill", FX.ocean).attr("stroke", LIGHT).attr("stroke-width", 0.4);
  const sid = `sphere${STYLE}`;
  svg.append("defs").append("clipPath").attr("id", sid).append("path").attr("d", sphereD);
  const gland = g.append("g").attr("clip-path", `url(#${sid})`);
  dropShadow(gland, s => s.append("path").attr("d", landD), 1.5);
  for (let i = FX.extrude; i >= 1; i--)      // showcase: a thin slab under the land
    gland.append("path").attr("d", landD).attr("transform", `translate(${(i * 0.18).toFixed(2)},${(i * 0.32).toFixed(2)})`).attr("fill", "#CFCFCF");
  gland.append("path").attr("d", landD).attr("fill", FX.land).attr("stroke", "#fff").attr("stroke-width", FX.landEdge).attr("stroke-linejoin", "round");
  // mangrove extent (GMW cells)
  // one path for all cells keeps the PDF small
  let dm = "";
  X.gmw.forEach(([lon, lat]) => { const p = proj([lon, lat]); if (p) dm += `M${(p[0] - 0.3).toFixed(2)},${(p[1] - 0.3).toFixed(2)}h0.6v0.6h-0.6Z`; });
  g.append("path").attr("d", dm).attr("fill", PAL.greygreen).attr("opacity", 0.62);
  // region hulls
  const byRegion = d3.group(X.cores.filter(c => c.region), c => c.region);
  const deltaRegionColor = PAL.mid;
  const gh = g.append("g");
  for (const [rid, cs] of byRegion) {
    const pts = cs.map(c => proj([c.lon, c.lat])).filter(Boolean);
    if (pts.length < 3) continue;
    const hull = d3.polygonHull(pts);
    if (!hull) continue;
    // pad the hull by a small offset so single-site clusters remain visible
    const cx = d3.mean(hull, p => p[0]), cy = d3.mean(hull, p => p[1]);
    const padded = hull.map(([x, y]) => { const dx = x - cx, dy = y - cy, L = Math.hypot(dx, dy) || 1; return [x + dx / L * 2.2, y + dy / L * 2.2]; });
    const hullD = "M" + padded.map(p => p.join(",")).join("L") + "Z";
    if (STYLE === "showcase") dropShadow(gh, s => s.append("path").attr("d", hullD), 0.5);
    gh.append("path").attr("d", hullD)
      .attr("fill", deltaRegionColor).attr("fill-opacity", 0.25).attr("stroke", deltaRegionColor).attr("stroke-width", 0.7);
  }
  // cores
  let dc = "";
  X.cores.forEach(c => { const p = proj([c.lon, c.lat]); if (p) dc += `M${(p[0] - 0.5).toFixed(2)},${p[1].toFixed(2)}a0.5,0.5 0 1,0 1,0a0.5,0.5 0 1,0 -1,0Z`; });
  g.append("path").attr("d", dc).attr("fill", INK).attr("opacity", 0.75);
  // delta labels, with leaders placed by hand-tuned offsets (dx, dy in pt)
  const off = { everglades: [-34, -18], amazon_amapa: [-16, 22], saloum_gambia: [-14, -22, "middle"], rufiji: [28, -6],
                zambezi: [20, 16], sundarbans: [-18, -20], mekong: [34, -12], musi_banyuasin: [24, 28] };
  const gl = g.append("g");
  DATA.deltas.forEach(d => {
    const p = proj([d.lon, d.lat]); const [dx, dy, anc] = off[d.id] || [12, -10];
    const anchor = anc || (dx > 0 ? "start" : "end"), mid = anchor === "middle";
    dropShadow(gl, s => s.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 3.7), 0.7);
    gl.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 3.9).attr("fill", "white").attr("opacity", 0.75);
    gl.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 3.2).attr("fill", "none").attr("stroke", FAIL).attr("stroke-width", 0.9);
    gl.append("line").attr("x1", p[0] + Math.sign(dx) * 3.4).attr("y1", p[1] + (mid ? -2.4 : 0)).attr("x2", p[0] + dx).attr("y2", p[1] + dy + (mid ? 2.5 : -2))
      .attr("stroke", FAIL).attr("stroke-width", 0.5);
    txt(gl, p[0] + dx + (mid ? 0 : dx > 0 ? 1.5 : -1.5), p[1] + dy, `${d.name} (${d.n})`, { size: FS_S, anchor, halo: true });
  });
  // legend
  const lg = svg.append("g").attr("transform", `translate(6,${mh - 30})`);
  lg.append("rect").attr("x", 0).attr("y", -1).attr("width", 6).attr("height", 3).attr("fill", PAL.greygreen).attr("opacity", 0.85);
  txt(lg, 9, 2, "mangrove extent (GMW v3)", { size: FS_S });
  lg.append("circle").attr("cx", 3).attr("cy", 9).attr("r", 1).attr("fill", INK);
  txt(lg, 9, 11, "soil core (n = 2,489)", { size: FS_S });
  lg.append("rect").attr("x", 0).attr("y", 15).attr("width", 6).attr("height", 5).attr("fill", PAL.mid).attr("fill-opacity", 0.12).attr("stroke", PAL.mid).attr("stroke-width", 0.6);
  txt(lg, 9, 20, "250 km region held out (29)", { size: FS_S });
  lg.append("circle").attr("cx", 3).attr("cy", 27).attr("r", 2.6).attr("fill", "none").attr("stroke", FAIL).attr("stroke-width", 0.9);
  txt(lg, 9, 29, "core delta held out (8; cores in brackets)", { size: FS_S });

  // ---- b: validation tiers
  const by = mh + 14, bh = H - by - 4;
  const pb = { l: 34, r: 8, t: 8, b: 26 }, bw = 226;
  letter(svg, 2, by - 3, "b");
  const gb = svg.append("g").attr("transform", `translate(0,${by})`);
  const tiers = [["t1", "Random\nk-fold"], ["t1g", "Site-grouped\nk-fold"], ["t2", "Spatial\nblock"], ["t3", "Leave-one-\ndelta-out"]];
  const xb = d3.scaleBand().domain(tiers.map(t => t[0])).range([pb.l, bw - pb.r]).paddingInner(0.35).paddingOuter(0.2);
  const yb = d3.scaleLinear().domain([-2.2, 0.8]).range([bh - pb.b, pb.t]);
  axis(gb.append("g").attr("transform", `translate(${pb.l},0)`), d3.axisLeft(yb).ticks(6).tickSize(2.5).tickPadding(2).tickFormat(d3.format("+.1f")));
  ylabel(gb, 12, (yb.range()[0] + yb.range()[1]) / 2, "R²");
  hline(gb, pb.l, bw - pb.r, yb(0), AXIS, null, 0.5);
  const hg = DATA.tiers.histgb, rd = DATA.tiers.ridge;
  const perDelta = DATA.perdelta.map(d => d.r2);
  tiers.forEach(([k, lab]) => {
    const v = hg[k], x0 = xb(k), w = xb.bandwidth();
    const ry = Math.min(yb(0), yb(v)), rh = Math.abs(yb(v) - yb(0));
    dropShadow(gb, s => s.append("rect").attr("x", x0).attr("y", ry).attr("width", w).attr("height", rh), 0.9);
    gb.append("rect").attr("x", x0).attr("y", ry).attr("width", w).attr("height", rh)
      .attr("fill", (k === "t1" || k === "t1g") ? HOLD : FAIL);
    topHighlight(gb, x0, ry, w);
    txt(gb, x0 + w / 2, v >= 0 ? yb(v) - 3 : yb(0) - 3, fmt2(v), { anchor: "middle", size: FS_S, halo: true });
    const dia = d3.symbol(d3.symbolDiamond, 14)(), dt = `translate(${x0 + w / 2},${yb(rd[k])})`;
    dropShadow(gb, s => s.append("path").attr("d", dia).attr("transform", dt), 0.5);
    gb.append("path").attr("d", dia).attr("transform", dt)
      .attr("fill", "white").attr("stroke", INK).attr("stroke-width", 0.7);
    lab.split("\n").forEach((s, i) => txt(gb, x0 + w / 2, bh - pb.b + 9 + i * 7.5, s, { anchor: "middle", size: FS_S }));
  });
  // per-delta LODO points (jittered), values below the axis marked
  const x3 = xb("t3") + xb.bandwidth() / 2;
  perDelta.forEach((v, i) => {
    const jx = x3 + (i - 3.5) * 2.2;
    if (v < yb.domain()[0]) gb.append("path").attr("d", d3.symbol(d3.symbolTriangle, 10)()).attr("transform", `translate(${jx},${yb.range()[0] - 2}) rotate(180)`).attr("fill", INK);
    else {
      dropShadow(gb, s => s.append("circle").attr("cx", jx).attr("cy", yb(v)).attr("r", 1.5), 0.45);
      gb.append("circle").attr("cx", jx).attr("cy", yb(v)).attr("r", 1.5).attr("fill", INK).attr("stroke", "white").attr("stroke-width", 0.4);
    }
  });
  // legend
  const lb = gb.append("g").attr("transform", `translate(${pb.l + 8},${yb(-0.75)})`);
  lb.append("rect").attr("x", 0).attr("y", 0).attr("width", 7).attr("height", 5).attr("fill", PAL.mid);
  txt(lb, 10, 4.5, "gradient boosting", { size: FS_S });
  lb.append("path").attr("d", d3.symbol(d3.symbolDiamond, 14)()).attr("transform", "translate(3.5,11)").attr("fill", "white").attr("stroke", INK).attr("stroke-width", 0.7);
  txt(lb, 10, 13.5, "ridge", { size: FS_S });
  lb.append("circle").attr("cx", 3.5).attr("cy", 20).attr("r", 1.5).attr("fill", INK);
  txt(lb, 10, 22.5, "held-out delta (8)", { size: FS_S });

  // ---- c: per-delta within-delta r with bootstrap CI, vs the published map
  const cx0 = bw + 10, cw = W - cx0;
  letter(svg, cx0, by - 3, "c");
  const gc2 = svg.append("g").attr("transform", `translate(${cx0},${by})`);
  const pc = { l: 62, r: 8, t: 8, b: 26 };
  const deltas = DATA.perdelta.map(d => d.delta);
  const rows = deltas.map(id => {
    // reported within-delta r is on log stock (soc_lodo.t3_lodo); same here
    const L = X.lodo[id], lo = L.obs.map(Math.log1p), lp = L.pred.map(Math.log1p);
    const r = pearson(lo, lp); const ci = bootCI(lo, lp);
    const pm = PM.per_delta.find(p => p.delta_id === id);
    return { id, name: DELTA_NAME[id], r, ci, n: L.n, map: pm ? pm.pearson : null };
  }).sort((a, b) => d3.descending(a.r, b.r));
  const yc = d3.scaleBand().domain(rows.map(r => r.id)).range([pc.t, bh - pc.b]).padding(0.3);
  const xc = d3.scaleLinear().domain([-0.8, 1]).range([pc.l, cw - pc.r]).clamp(true);
  axis(gc2.append("g").attr("transform", `translate(0,${bh - pc.b})`), d3.axisBottom(xc).ticks(5).tickSize(2.5).tickPadding(2).tickFormat(d3.format("+.1f")));
  txt(gc2, (xc.range()[0] + xc.range()[1]) / 2, bh - pc.b + 17, "within-delta correlation r (held out)", { anchor: "middle" });
  gc2.append("line").attr("x1", xc(0)).attr("x2", xc(0)).attr("y1", pc.t).attr("y2", bh - pc.b).attr("stroke", AXIS).attr("stroke-width", 0.5);
  // replicate ceiling (benchmark median) as a reference line
  const ceil = DATA.biomes.find(b => b.tag === "mangrove").ceiling;
  gc2.append("line").attr("x1", xc(ceil)).attr("x2", xc(ceil)).attr("y1", pc.t).attr("y2", bh - pc.b).attr("stroke", HOLD).attr("stroke-width", 0.5).attr("stroke-dasharray", "2,1.5");
  txt(gc2, xc(ceil) - 2, yc.range()[1] - 2, "replicate ceiling", { anchor: "end", size: FS_S, color: HOLD });
  rows.forEach(r => {
    const y = yc(r.id) + yc.bandwidth() / 2;
    txt(gc2, pc.l - 4, y + 2.2, `${r.name} (${r.n})`, { anchor: "end", size: FS_S });
    gc2.append("line").attr("x1", xc(r.ci[0])).attr("x2", xc(r.ci[1])).attr("y1", y).attr("y2", y).attr("stroke", FAIL).attr("stroke-width", 0.8);
    if (r.map !== null) {
      const sq = d3.symbol(d3.symbolSquare, 10)(), st = `translate(${xc(r.map)},${y})`;
      dropShadow(gc2, s => s.append("path").attr("d", sq).attr("transform", st), 0.5);
      gc2.append("path").attr("d", sq).attr("transform", st).attr("fill", "white").attr("stroke", PAL.brown).attr("stroke-width", 0.8);
    }
    dropShadow(gc2, s => s.append("circle").attr("cx", xc(r.r)).attr("cy", y).attr("r", 2.1), 0.55);
    gc2.append("circle").attr("cx", xc(r.r)).attr("cy", y).attr("r", 2.1).attr("fill", FAIL).attr("stroke", "white").attr("stroke-width", 0.4);
  });
  const lc = gc2.append("g").attr("transform", `translate(${xc(-0.78)},${pc.t - 3})`);
  lc.append("circle").attr("cx", 3).attr("cy", 0).attr("r", 2).attr("fill", FAIL);
  txt(lc, 8, 2.2, "this study (95% CI)", { size: FS_S });
  lc.append("path").attr("d", d3.symbol(d3.symbolSquare, 10)()).attr("transform", "translate(70,0)").attr("fill", "white").attr("stroke", PAL.brown).attr("stroke-width", 0.8);
  txt(lc, 75, 2.2, "published 30 m map", { size: FS_S });
  save(dom, "fig1_benchmark");
}

// ===================================================================== Fig. 2
// a  dissimilarity index of held-out cores against both thresholds; b 29-region r with ceilings
function fig2() {
  const W = TW, H = 190, { dom, svg } = svgRoot(W, H);
  // ---- a
  const aw = 218, pa = { l: 62, r: 6, t: 24, b: 24 };
  letter(svg, 2, 9, "a");
  const ga = svg.append("g");
  const ids = DATA.perdelta.map(d => d.delta);
  const thrG = d3.median(ids.map(i => X.lodo[i].thr_grouped)), thrR = d3.median(ids.map(i => X.lodo[i].thr_random));
  const xa = d3.scaleLinear().domain([0, 1.0]).range([pa.l, aw - pa.r]).clamp(true);
  const ya = d3.scaleBand().domain(ids).range([pa.t, H - pa.b]).padding(0.25);
  axis(ga.append("g").attr("transform", `translate(0,${H - pa.b})`), d3.axisBottom(xa).ticks(5).tickSize(2.5).tickPadding(2));
  txt(ga, (xa.range()[0] + xa.range()[1]) / 2, H - pa.b + 17, "dissimilarity index of held-out cores", { anchor: "middle" });
  // random-CV AoA threshold sits at the origin; the out-of-region threshold is per delta
  ga.append("line").attr("x1", xa(thrR)).attr("x2", xa(thrR)).attr("y1", pa.t - 2).attr("y2", H - pa.b).attr("stroke", FAIL).attr("stroke-width", 0.9);
  txt(ga, xa(thrR) + 2, pa.t - 15, `random-CV AoA: DI ≤ ${d3.format(".3f")(thrR)} (0% inside)`, { anchor: "start", size: FS_S, color: FAIL });
  ids.forEach(id => {
    const L = X.lodo[id], y = ya(id) + ya.bandwidth() / 2, d = L.di_grouped.filter(v => v > 0).sort(d3.ascending);
    const q = [0.05, 0.25, 0.5, 0.75, 0.95].map(p => d3.quantileSorted(d, p));
    txt(ga, pa.l - 4, y + 2.2, DELTA_NAME[id], { anchor: "end", size: FS_S });
    ga.append("rect").attr("x", xa(0)).attr("y", y - ya.bandwidth() / 2).attr("width", xa(L.thr_grouped) - xa(0)).attr("height", ya.bandwidth()).attr("fill", HOLD).attr("opacity", 0.14);
    ga.append("line").attr("x1", xa(L.thr_grouped)).attr("x2", xa(L.thr_grouped)).attr("y1", y - ya.bandwidth() / 2).attr("y2", y + ya.bandwidth() / 2).attr("stroke", HOLD).attr("stroke-width", 0.9);
    ga.append("line").attr("x1", xa(q[0])).attr("x2", xa(q[4])).attr("y1", y).attr("y2", y).attr("stroke", INK).attr("stroke-width", 0.5);
    const bx0 = xa(q[1]), by0 = y - ya.bandwidth() / 2 + 1.5, bw0 = xa(q[3]) - xa(q[1]), bh0 = ya.bandwidth() - 3;
    dropShadow(ga, s => s.append("rect").attr("x", bx0).attr("y", by0).attr("width", bw0).attr("height", bh0), 0.6);
    ga.append("rect").attr("x", bx0).attr("y", by0).attr("width", bw0).attr("height", bh0).attr("fill", PAL.grey);
    topHighlight(ga, bx0, by0, bw0);
    ga.append("line").attr("x1", xa(q[2])).attr("x2", xa(q[2])).attr("y1", y - ya.bandwidth() / 2 + 1.5).attr("y2", y + ya.bandwidth() / 2 - 1.5).attr("stroke", "white").attr("stroke-width", 0.8);
    const pd = DATA.perdelta.find(p => p.delta === id);
    txt(ga, aw - pa.r, y + 2.2, `${Math.round(pd.aoa * 100)}%`, { anchor: "end", size: FS_S, color: HOLD });
  });
  const l2 = ga.append("g").attr("transform", `translate(${xa(thrR) + 2},${pa.t - 5})`);
  l2.append("rect").attr("x", 0).attr("y", -4).attr("width", 8).attr("height", 5).attr("fill", HOLD).attr("opacity", 0.2);
  l2.append("line").attr("x1", 8).attr("x2", 8).attr("y1", -5).attr("y2", 2).attr("stroke", HOLD).attr("stroke-width", 0.9);
  txt(l2, 11, 0.5, "out-of-region AoA (threshold per delta)", { size: 5.5, color: HOLD });
  txt(ga, aw - pa.r, pa.t - 15, "inside", { anchor: "end", size: FS_S, color: HOLD });

  // ---- b: 29 regions
  const bx = aw + 14, bw = W - bx, pb = { l: 34, r: 4, t: 14, b: 24 };
  letter(svg, bx - 8, 9, "b");
  const gb = svg.append("g").attr("transform", `translate(${bx},0)`);
  const regs = RL.per_region.map(r => ({ ...r, ceiling: X.ceiling[r.region] ? X.ceiling[r.region].r_max : null }));
  const conts = CONT_ORDER.filter(c => regs.some(r => r.continent === c));
  const xb = d3.scaleBand().domain(conts).range([pb.l, bw - pb.r]).padding(0.15);
  const yb = d3.scaleLinear().domain([-0.6, 1]).range([H - pb.b, pb.t]);
  axis(gb.append("g").attr("transform", `translate(${pb.l},0)`), d3.axisLeft(yb).ticks(6).tickSize(2.5).tickPadding(2).tickFormat(d3.format("+.1f")));
  ylabel(gb, 10, (yb.range()[0] + yb.range()[1]) / 2, "within-region r (held out)");
  hline(gb, pb.l, bw - pb.r, yb(0), AXIS, null, 0.5);
  const s = RL.summary;
  nestedBand(gb, pb.l, bw - pb.r - pb.l, yb(s.lodo_median_pearson_ci[1]), yb(s.lodo_median_pearson_ci[0]), yb(s.lodo_median_pearson), FAIL, 0.2);
  hline(gb, pb.l, bw - pb.r, yb(s.lodo_median_pearson), FAIL, "2,1.5", 0.8);
  txt(gb, xb("S.America") + 2, yb(s.lodo_median_pearson) + 8, `median ${fmt2(s.lodo_median_pearson)} (95% CI)`, { anchor: "start", size: FS_S, color: FAIL, halo: true });
  const rs = d3.scaleSqrt().domain([10, d3.max(regs, r => r.n)]).range([1.3, 4]);
  conts.forEach(c => {
    const cr = regs.filter(r => r.continent === c), x0 = xb(c), w = xb.bandwidth();
    cr.forEach((r, i) => {
      const x = x0 + w * (0.2 + 0.6 * (cr.length === 1 ? 0.5 : i / (cr.length - 1)));
      if (r.ceiling !== null) {
        gb.append("line").attr("x1", x).attr("x2", x).attr("y1", yb(r.pearson)).attr("y2", yb(r.ceiling)).attr("stroke", LIGHT).attr("stroke-width", 0.5);
        const dia = d3.symbol(d3.symbolDiamond, 9)(), dt = `translate(${x},${yb(r.ceiling)})`;
        dropShadow(gb, s => s.append("path").attr("d", dia).attr("transform", dt), 0.45);
        gb.append("path").attr("d", dia).attr("transform", dt).attr("fill", "white").attr("stroke", HOLD).attr("stroke-width", 0.6);
      }
      dropShadow(gb, s => s.append("circle").attr("cx", x).attr("cy", yb(r.pearson)).attr("r", rs(r.n)), 0.55);
      gb.append("circle").attr("cx", x).attr("cy", yb(r.pearson)).attr("r", rs(r.n)).attr("fill", FAIL).attr("stroke", "white").attr("stroke-width", 0.4);
    });
    const med = d3.median(cr, r => r.pearson);
    hline(gb, x0 + 2, x0 + w - 2, yb(med), "white", null, 1.8);
    hline(gb, x0 + 2, x0 + w - 2, yb(med), INK, null, 1);
    txt(gb, x0 + w / 2, H - pb.b + 9, CONT_LABEL[c], { anchor: "middle", size: FS_S });
    txt(gb, x0 + w / 2, H - pb.b + 16, `${cr.length}`, { anchor: "middle", size: FS_S, color: MUTE });
  });
  save(dom, "fig2_aoa_regions");
}

// ===================================================================== Fig. 3
// a  cross-biome r vs ceiling; b variance partition; c predictability of level and pattern
function fig3() {
  const W = TW, H = 300, { dom, svg } = svgRoot(W, H);
  const biomes = DATA.biomes.filter(b => !b.underpowered).map(b => ({ ...b, key: b.tag.replace(/_d\d+$/, "") }));
  const short = { terrestrial_conc: "Mineral\n(conc.)", terrestrial_stock: "Mineral\n(stock)", mangrove: "Mangrove", marsh: "Salt marsh", seagrass: "Seagrass", permafrost: "Permafrost" };
  // ---- a
  const ah = 160, pa = { l: 34, r: 6, t: 12, b: 30 };
  letter(svg, 2, 9, "a");
  const ga = svg.append("g");
  const xa = d3.scaleBand().domain(biomes.map(b => b.key)).range([pa.l, W - pa.r]).padding(0.25);
  const ya = d3.scaleLinear().domain([-0.8, 1]).range([ah - pa.b, pa.t]);
  const first = biomes.findIndex(b => b.cls !== "mineral");
  if (first > 0) {
    const x0 = xa(biomes[first].key) - xa.step() * xa.padding() / 2;
    ga.append("rect").attr("x", x0).attr("y", pa.t - 4).attr("width", W - pa.r - x0).attr("height", ah - pa.b - pa.t + 4).attr("fill", FAIL).attr("opacity", 0.06);
    txt(ga, W - pa.r - 2, pa.t + 3, "carbon-dense soils", { anchor: "end", size: FS_S, color: FAIL });
    txt(ga, pa.l + 4, pa.t + 3, "mineral soils", { anchor: "start", size: FS_S, color: HOLD });
  }
  axis(ga.append("g").attr("transform", `translate(${pa.l},0)`), d3.axisLeft(ya).ticks(7).tickSize(2.5).tickPadding(2).tickFormat(d3.format("+.1f")));
  ylabel(ga, 10, (ya.range()[0] + ya.range()[1]) / 2, "within-region r (held out)");
  hline(ga, pa.l, W - pa.r, ya(0), AXIS, null, 0.5);
  biomes.forEach(b => {
    const x0 = xa(b.key), w = xa.bandwidth(), xc = x0 + w / 2, col = b.cls === "mineral" ? HOLD : FAIL;
    const rsz = d3.scaleSqrt().domain([10, d3.max(b.regions, r => r.n)]).range([1.2, 4]);
    // CI band, median bar
    nestedBand(ga, x0 + 4, w - 8, ya(b.ci[1]), ya(b.ci[0]), ya(b.median_r), col, 0.3);
    // region points (jitter deterministic)
    b.regions.forEach((r, i) => {
      const jx = xc + ((i * 7919) % 100 / 100 - 0.5) * (w - 16);
      const y = Math.max(ya.domain()[0], r.r);
      dropShadow(ga, s => s.append("circle").attr("cx", jx).attr("cy", ya(y)).attr("r", rsz(r.n)), 0.4);
      ga.append("circle").attr("cx", jx).attr("cy", ya(y)).attr("r", rsz(r.n)).attr("fill", col).attr("opacity", 0.75).attr("stroke", "white").attr("stroke-width", 0.3);
    });
    hline(ga, x0 + 4, x0 + w - 4, ya(b.median_r), "white", null, 2.2);
    hline(ga, x0 + 4, x0 + w - 4, ya(b.median_r), INK, null, 1.2);
    // ceiling
    ga.append("line").attr("x1", xc).attr("x2", xc).attr("y1", ya(b.median_r)).attr("y2", ya(b.ceiling)).attr("stroke", MUTE).attr("stroke-width", 0.5).attr("stroke-dasharray", "1.5,1.5");
    const dia = d3.symbol(d3.symbolDiamond, 16)(), dt = `translate(${xc},${ya(b.ceiling)})`;
    dropShadow(ga, s => s.append("path").attr("d", dia).attr("transform", dt), 0.6);
    ga.append("path").attr("d", dia).attr("transform", dt).attr("fill", "white").attr("stroke", HOLD).attr("stroke-width", 0.7);
    txt(ga, xc + 5, ya(b.ceiling) + 2.2, d3.format(".2f")(b.ceiling), { size: FS_S, color: MUTE, halo: true });
    txt(ga, x0 + w - 4, ya(Math.max(b.ci[1], b.median_r)) - 2.5, fmt2(b.median_r), { anchor: "end", size: FS_S, weight: "bold", halo: true });
    short[b.key].split("\n").forEach((s, i) => txt(ga, xc, ah - pa.b + 9 + i * 7.5, s, { anchor: "middle", size: FS_S }));
    txt(ga, xc, ah - pa.b + 9 + short[b.key].split("\n").length * 7.5, `${d3.format(",")(b.n)} cores, ${b.n_regions} regions`, { anchor: "middle", size: 5.5, color: MUTE });
  });
  const la = ga.append("g").attr("transform", `translate(${pa.l + 6},${ah - pa.b - 14})`);
  la.append("path").attr("d", d3.symbol(d3.symbolDiamond, 16)()).attr("transform", "translate(3,2)").attr("fill", "white").attr("stroke", INK).attr("stroke-width", 0.7);
  txt(la, 9, 4, "replicate ceiling", { size: FS_S });
  la.append("line").attr("x1", 0).attr("x2", 6).attr("y1", 9).attr("y2", 9).attr("stroke", INK).attr("stroke-width", 1.2);
  txt(la, 9, 11, "median, 95% CI", { size: FS_S });

  // ---- b: variance partition
  const by = ah + 14, bh = H - by, bw = 218, pb = { l: 62, r: 6, t: 8, b: 24 };
  letter(svg, 2, by - 3, "b");
  const gb = svg.append("g").attr("transform", `translate(0,${by})`);
  const vp = VP.rows.filter(r => biomes.some(b => b.key === r.biome)).sort((a, b) => biomes.findIndex(x => x.key === a.biome) - biomes.findIndex(x => x.key === b.biome));
  const yb = d3.scaleBand().domain(vp.map(r => r.biome)).range([pb.t, bh - pb.b]).padding(0.3);
  const xb = d3.scaleLinear().domain([0, 1]).range([pb.l, bw - pb.r]);
  axis(gb.append("g").attr("transform", `translate(0,${bh - pb.b})`), d3.axisBottom(xb).ticks(5, "%").tickSize(2.5).tickPadding(2));
  txt(gb, (xb.range()[0] + xb.range()[1]) / 2, bh - pb.b + 17, "share of variance of log stock", { anchor: "middle" });
  const comps = [["between_regions", FAIL, "between regions"], ["between_sites_within", PAL.amber, "between sites within region"], ["replicate", PAL.mid, "replicate cores"]];
  vp.forEach(r => {
    let x = 0; const y = yb(r.biome);
    const tot = r.between_regions + r.between_sites_within + r.replicate;
    dropShadow(gb, s => s.append("rect").attr("x", xb(0)).attr("y", y).attr("width", xb(tot) - xb(0)).attr("height", yb.bandwidth()), 0.8);
    txt(gb, pb.l - 4, y + yb.bandwidth() / 2 + 2.2, short[r.biome].replace("\n", " "), { anchor: "end", size: FS_S });
    comps.forEach(([k, col]) => {
      gb.append("rect").attr("x", xb(x)).attr("y", y).attr("width", xb(x + r[k]) - xb(x)).attr("height", yb.bandwidth()).attr("fill", col);
      if (k === "between_regions") txt(gb, xb(x + r[k] / 2), y + yb.bandwidth() / 2 + 2, `${Math.round(r[k] * 100)}%`, { anchor: "middle", size: FS_S, color: "white" });
      x += r[k];
    });
    topHighlight(gb, xb(0), y, xb(tot) - xb(0));
  });
  const lb = gb.append("g").attr("transform", `translate(${pb.l},${pb.t - 6})`);
  comps.forEach(([k, col, lab], i) => {
    const xo = [0, 60, 150][i];
    lb.append("rect").attr("x", xo).attr("y", -4).attr("width", 6).attr("height", 5).attr("fill", col);
    txt(lb, xo + 8, 0.5, lab, { size: 5.5 });
  });

  // ---- c: level vs pattern predictability
  const cx0 = bw + 14, cw = W - cx0, pc = { l: 62, r: 10, t: 8, b: 24 };
  letter(svg, cx0 - 8, by - 3, "c");
  const gc = svg.append("g").attr("transform", `translate(${cx0},${by})`);
  const xc = d3.scaleLinear().domain([-0.4, 0.8]).range([pc.l, cw - pc.r]);
  axis(gc.append("g").attr("transform", `translate(0,${bh - pc.b})`), d3.axisBottom(xc).ticks(6).tickSize(2.5).tickPadding(2).tickFormat(d3.format("+.1f")));
  txt(gc, (xc.range()[0] + xc.range()[1]) / 2, bh - pc.b + 17, "R² (held out)", { anchor: "middle" });
  gc.append("line").attr("x1", xc(0)).attr("x2", xc(0)).attr("y1", pc.t).attr("y2", bh - pc.b).attr("stroke", AXIS).attr("stroke-width", 0.5);
  vp.forEach(r => {
    const y = yb(r.biome) + yb.bandwidth() / 2;
    txt(gc, pc.l - 4, y + 2.2, short[r.biome].replace("\n", " "), { anchor: "end", size: FS_S });
    const a = Math.max(xc.domain()[0], r.r2_between_regions), b = Math.max(xc.domain()[0], r.r2_within_region_median);
    gc.append("line").attr("x1", xc(a)).attr("x2", xc(b)).attr("y1", y).attr("y2", y).attr("stroke", LIGHT).attr("stroke-width", 0.6);
    const sq = d3.symbol(d3.symbolSquare, 14)(), st = `translate(${xc(b)},${y})`;
    dropShadow(gc, s => s.append("path").attr("d", sq).attr("transform", st), 0.55);
    gc.append("path").attr("d", sq).attr("transform", st).attr("fill", PAL.charcoal).attr("stroke", "white").attr("stroke-width", 0.35);
    dropShadow(gc, s => s.append("circle").attr("cx", xc(a)).attr("cy", y).attr("r", 2.4), 0.55);
    gc.append("circle").attr("cx", xc(a)).attr("cy", y).attr("r", 2.4).attr("fill", FAIL).attr("stroke", "white").attr("stroke-width", 0.35);
    if (r.r2_between_regions < xc.domain()[0]) txt(gc, xc(a) + 4, y + 2, `${d3.format("+.2f")(r.r2_between_regions)}`, { size: 5.5, color: MUTE });
  });
  const lc = gc.append("g").attr("transform", `translate(${pc.l},${pc.t - 6})`);
  lc.append("circle").attr("cx", 3).attr("cy", -1.5).attr("r", 2.4).attr("fill", FAIL);
  txt(lc, 8, 0.5, "regional mean (held out)", { size: 5.5 });
  lc.append("path").attr("d", d3.symbol(d3.symbolSquare, 14)()).attr("transform", "translate(84,-1.5)").attr("fill", PAL.charcoal);
  txt(lc, 89, 0.5, "within-region pattern (local fit)", { size: 5.5 });
  save(dom, "fig_biomes");
}

// ===================================================================== Fig. 4
// a  few-shot calibration; b Tier-1 default vs national cores; c certification protocol
function fig4() {
  const W = TW, H = 310, { dom, svg } = svgRoot(W, H);
  const lw = 224;
  // ---- a
  const ah = 120, pa = { l: 34, r: 8, t: 10, b: 24 };
  letter(svg, 2, 9, "a");
  const ga = svg.append("g");
  const ks = FS.ks, xa = d3.scaleLinear().domain([0, 25]).range([pa.l, lw - pa.r]);
  const ya = d3.scaleLinear().domain([-0.4, 0.8]).range([ah - pa.b, pa.t]);
  axis(ga.append("g").attr("transform", `translate(0,${ah - pa.b})`), d3.axisBottom(xa).tickValues(ks).tickSize(2.5).tickPadding(2));
  axis(ga.append("g").attr("transform", `translate(${pa.l},0)`), d3.axisLeft(ya).ticks(5).tickSize(2.5).tickPadding(2).tickFormat(d3.format("+.1f")));
  txt(ga, (xa.range()[0] + xa.range()[1]) / 2, ah - pa.b + 17, "local cores added to training, k", { anchor: "middle" });
  ylabel(ga, 10, (ya.range()[0] + ya.range()[1]) / 2, "within-delta r");
  hline(ga, pa.l, lw - pa.r, ya(0), AXIS, null, 0.5);
  const ceil = DATA.biomes.find(b => b.tag === "mangrove").ceiling;
  hline(ga, pa.l, lw - pa.r, ya(ceil / 2), MUTE, "2,1.5", 0.5);
  txt(ga, pa.l + 3, ya(ceil / 2) - 2.5, "half the replicate ceiling", { anchor: "start", size: FS_S, color: MUTE, halo: true });
  // per-delta traces
  Object.entries(FS.per_delta).forEach(([id, s]) => {
    const pts = ks.filter(k => s[k] && Number.isFinite(s[k].pearson)).map(k => [xa(k), ya(s[k].pearson)]);
    ga.append("path").attr("d", d3.line()(pts)).attr("fill", "none").attr("stroke", HOLD).attr("stroke-width", 0.5).attr("opacity", 0.35);
  });
  const med = ks.map(k => [k, FS.summary[k].median_pearson]), loc = ks.filter(k => FS.summary[k].median_pearson_localonly !== null).map(k => [k, FS.summary[k].median_pearson_localonly]);
  // fade under the median curve, down to r = 0
  const areaD = d3.area().x(d => xa(d[0])).y0(ya(0)).y1(d => ya(d[1]))(med);
  areaFade(svg, ga, areaD, ya(d3.max(med, d => d[1])), ya(0), pa.l, lw - pa.r, HOLD, STYLE === "showcase" ? 0.5 : 0.32);
  const medD = d3.line().x(d => xa(d[0])).y(d => ya(d[1]))(med), locD = d3.line().x(d => xa(d[0])).y(d => ya(d[1]))(loc);
  dropShadow(ga, s => s.append("path").attr("d", medD).attr("fill", "none").attr("data-w", 1.4), 0.7);
  ga.append("path").attr("d", medD).attr("fill", "none").attr("stroke", HOLD).attr("stroke-width", 1.4).attr("stroke-linecap", "round");
  med.forEach(d => {
    dropShadow(ga, s => s.append("circle").attr("cx", xa(d[0])).attr("cy", ya(d[1])).attr("r", 2.1), 0.5);
    ga.append("circle").attr("cx", xa(d[0])).attr("cy", ya(d[1])).attr("r", 2.1).attr("fill", HOLD).attr("stroke", "white").attr("stroke-width", 0.5);
  });
  ga.append("path").attr("d", locD).attr("fill", "none").attr("stroke", PAL.amber).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,2");
  loc.forEach(d => {
    const sq = d3.symbol(d3.symbolSquare, 12)(), st = `translate(${xa(d[0])},${ya(d[1])})`;
    dropShadow(ga, s => s.append("path").attr("d", sq).attr("transform", st), 0.5);
    ga.append("path").attr("d", sq).attr("transform", st).attr("fill", PAL.amber).attr("stroke", "white").attr("stroke-width", 0.4);
  });
  const la = ga.append("g").attr("transform", `translate(${xa(9)},${ya(0) + 6})`);
  la.append("line").attr("x1", 0).attr("x2", 8).attr("y1", 2).attr("y2", 2).attr("stroke", HOLD).attr("stroke-width", 1.4);
  txt(la, 11, 4, "global model + k local cores (median)", { size: FS_S });
  la.append("line").attr("x1", 0).attr("x2", 8).attr("y1", 10).attr("y2", 10).attr("stroke", PAL.amber).attr("stroke-width", 1.2).attr("stroke-dasharray", "3,2");
  txt(la, 11, 12, "ridge on the k local cores alone", { size: FS_S });
  la.append("line").attr("x1", 0).attr("x2", 8).attr("y1", 18).attr("y2", 18).attr("stroke", HOLD).attr("stroke-width", 0.5).attr("opacity", 0.5);
  txt(la, 11, 20, "single delta", { size: FS_S });

  // ---- b: Tier-1
  const by = ah + 12, bh = H - by, pb = { l: 62, r: 8, t: 4, b: 24 };
  letter(svg, 2, by - 3, "b");
  const gb = svg.append("g").attr("transform", `translate(0,${by})`);
  const pc = T1.per_country.slice().sort((a, b) => d3.descending(a.ratio_obs_to_tier1, b.ratio_obs_to_tier1));
  const xb = d3.scaleLog().domain([0.1, 2]).range([pb.l, lw - pb.r]);
  const yb = d3.scaleBand().domain(pc.map(c => c.iso3)).range([pb.t, bh - pb.b]).padding(0.2);
  axis(gb.append("g").attr("transform", `translate(0,${bh - pb.b})`), d3.axisBottom(xb).tickValues([0.1, 0.2, 0.5, 1, 2]).tickFormat(d3.format(".1~f")).tickSize(2.5).tickPadding(2));
  txt(gb, (xb.range()[0] + xb.range()[1]) / 2, bh - pb.b + 17, "national median core stock / IPCC Tier 1 default (386 t C ha⁻¹)", { anchor: "middle", size: FS_S });
  gb.append("line").attr("x1", xb(1)).attr("x2", xb(1)).attr("y1", pb.t).attr("y2", bh - pb.b).attr("stroke", AXIS).attr("stroke-width", 0.6);
  const rs = d3.scaleSqrt().domain([0, d3.max(pc, c => c.mangrove_km2)]).range([0.8, 4.5]);
  pc.forEach(c => {
    const y = yb(c.iso3) + yb.bandwidth() / 2, t1 = T1.summary.tier1_Mgha;
    txt(gb, pb.l - 3, y + 2, c.country.replace("United Republic of Tanzania", "Tanzania").replace("Federated States of Micronesia", "Micronesia").replace("United States of America", "USA"), { anchor: "end", size: 5.5 });
    gb.append("line").attr("x1", xb(c.stock_ci_Mgha[0] / t1)).attr("x2", xb(c.stock_ci_Mgha[1] / t1)).attr("y1", y).attr("y2", y).attr("stroke", INK).attr("stroke-width", 0.5);
    dropShadow(gb, s => s.append("circle").attr("cx", xb(c.ratio_obs_to_tier1)).attr("cy", y).attr("r", rs(c.mangrove_km2)), 0.55);
    gb.append("circle").attr("cx", xb(c.ratio_obs_to_tier1)).attr("cy", y).attr("r", rs(c.mangrove_km2))
      .attr("fill", c.ratio_obs_to_tier1 < 1 ? FAIL : HOLD).attr("stroke", "white").attr("stroke-width", 0.4);
  });
  const lb = gb.append("g").attr("transform", `translate(${xb(0.105)},${pb.t + 4})`);
  [1000, 10000, 30000].forEach((v, i) => { lb.append("circle").attr("cx", 4).attr("cy", i * 9).attr("r", rs(v)).attr("fill", "none").attr("stroke", INK).attr("stroke-width", 0.5);
    txt(lb, 11, i * 9 + 2, `${d3.format(",")(v)} km²`, { size: 5.5, color: MUTE }); });

  // ---- c: protocol
  const cx0 = lw + 16, cw = W - cx0;
  letter(svg, cx0 - 8, 9, "c");
  const gc = svg.append("g").attr("transform", `translate(${cx0},0)`);
  const steps = [
    ["Inside the AoA paired with random-validation skill?", `benchmark: threshold ${d3.format(".3f")(d3.median(Object.values(X.lodo), v => v.thr_random))}; 0% of any unsampled delta`, "no", "reported R² does not apply", PAL.light, INK],
    ["Inside the AoA paired with the out-of-region error?", `benchmark: threshold ${d3.format(".2f")(d3.median(Object.values(X.lodo), v => v.thr_grouped))}; median ${d3.format(".1f")(100 * d3.median(DATA.perdelta, p => p.aoa))}% of a held-out delta`, "yes", "no: LOCAL CORES REQUIRED", FAIL, "white"],
    ["Out-of-region skill at least half the replicate ceiling?", `benchmark: r = ${fmtS(RL.summary.lodo_median_pearson)} [${fmtS(RL.summary.lodo_median_pearson_ci[0])}, ${fmtS(RL.summary.lodo_median_pearson_ci[1])}] against a ceiling of ${d3.format(".2f")(DATA.biomes.find(b => b.tag === "mangrove").ceiling)}`, "no", "yes: USABLE, out-of-region error applies", HOLD, "white"],
    ["How many local cores close the gap?", "benchmark: 10 cores reach half the ceiling", "", "LEVEL ONLY until k local cores are added", PAL.amber, INK],
  ];
  const sw = cw - 4, sh = 30, gap = 30, x0 = 2, y0 = 16;
  steps.forEach((s, i) => {
    const y = y0 + i * (sh + gap);
    dropShadow(gc, s => s.append("rect").attr("x", x0).attr("y", y).attr("width", sw).attr("height", sh).attr("rx", 2.5), FX.card);
    gc.append("rect").attr("x", x0).attr("y", y).attr("width", sw).attr("height", sh).attr("rx", 2.5).attr("fill", "#fcfcfc").attr("stroke", "#bdbdbd").attr("stroke-width", 0.5);
    gc.append("line").attr("x1", x0 + 2.5).attr("x2", x0 + sw - 2.5).attr("y1", y + sh - 0.35).attr("y2", y + sh - 0.35).attr("stroke", "#9e9e9e").attr("stroke-width", 0.5).attr("opacity", 0.6);
    // embossed step disc: shadow, disc, soft upper-left highlight
    dropShadow(gc, s => s.append("circle").attr("cx", x0 + 9).attr("cy", y + sh / 2).attr("r", 5.5), 0.6);
    gc.append("circle").attr("cx", x0 + 9).attr("cy", y + sh / 2).attr("r", 5.5).attr("fill", INK);
    gc.append("circle").attr("cx", x0 + 7.9).attr("cy", y + sh / 2 - 1.6).attr("r", 3.2).attr("fill", "white").attr("opacity", 0.16);
    txt(gc, x0 + 9, y + sh / 2 + 2.3, `${i + 1}`, { anchor: "middle", size: FS_S, color: "white", weight: "bold" });
    txt(gc, x0 + 19, y + 11, s[0], { size: FS_S, weight: "bold" });
    txt(gc, x0 + 19, y + 21, s[1], { size: 5.5, color: MUTE });
    // verdict chip on the exit of the step
    const chipW = s[3].length * 3.5 + 10, chipX = x0 + sw - chipW, chipY = y + sh + 7;
    dropShadow(gc, s2 => s2.append("rect").attr("x", chipX).attr("y", chipY).attr("width", chipW).attr("height", 11).attr("rx", 5.5), 0.9);
    gc.append("rect").attr("x", chipX).attr("y", chipY).attr("width", chipW).attr("height", 11).attr("rx", 5.5).attr("fill", s[4]);
    topHighlight(gc, chipX + 4, chipY + 0.4, chipW - 8, 0.45);
    txt(gc, chipX + chipW / 2, chipY + 7.8, s[3], { anchor: "middle", size: 5.5, color: s[5], weight: "bold" });
    gc.append("line").attr("x1", x0 + sw - 12).attr("x2", x0 + sw - 12).attr("y1", y + sh).attr("y2", chipY).attr("stroke", s[4] === PAL.light ? MUTE : s[4]).attr("stroke-width", 0.6);
    if (i < steps.length - 1) {
      gc.append("line").attr("x1", x0 + 9).attr("x2", x0 + 9).attr("y1", y + sh).attr("y2", y + sh + gap - 1).attr("stroke", INK).attr("stroke-width", 0.6);
      gc.append("path").attr("d", `M${x0 + 6.5},${y + sh + gap - 4}L${x0 + 9},${y + sh + gap - 0.5}L${x0 + 11.5},${y + sh + gap - 4}Z`).attr("fill", INK);
      txt(gc, x0 + 13, y + sh + gap / 2 + 2, s[2], { size: 5.5, color: MUTE, style: "italic" });
    }
  });
  txt(gc, x0, y0 + steps.length * (sh + gap) - 6, "arrows follow the mangrove benchmark;", { size: 5.5, color: MUTE });
  txt(gc, x0, y0 + steps.length * (sh + gap) + 1, "every quantity comes from the model's own training table", { size: 5.5, color: MUTE });
  save(dom, "fig4_fix");
}

fig1(); fig2(); fig3(); fig4();
console.log("done.");
