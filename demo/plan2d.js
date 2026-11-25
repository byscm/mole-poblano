// Top-down floor plan as SVG, drawn in the room's own axes (u along the main wall direction, v across), metres.
// createPlan(el, scene, objects, opts) -> {render, select, update}
//   opts: {editable, onSelect(id), onMove(id), onCommit(id, before), isOff(obj)}
const NS = "http://www.w3.org/2000/svg";

export function roomFrame(scene) {
  const m = (scene.main_dir_deg * Math.PI) / 180, U = [Math.cos(m), Math.sin(m)], V = [-U[1], U[0]];
  return {
    mdeg: scene.main_dir_deg,
    toUV: ([x, y]) => [x * U[0] + y * U[1], x * V[0] + y * V[1]],
    fromUV: ([u, v]) => [U[0] * u + V[0] * v, U[1] * u + V[1] * v],
  };
}

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

export function createPlan(host, scene, objects, opts = {}) {
  const F = roomFrame(scene);
  const out = scene.outline.map(F.toUV);
  const us = out.map((p) => p[0]), vs = out.map((p) => p[1]);
  const u0 = Math.min(...us), u1 = Math.max(...us), v0 = Math.min(...vs), v1 = Math.max(...vs);
  const pad = 0.75;
  const svg = el("svg", { class: "plan" + (opts.editable ? " edit" : ""), viewBox: `${-u1 - pad} ${v0 - pad} ${u1 - u0 + 2 * pad} ${v1 - v0 + 2 * pad}`, preserveAspectRatio: "xMidYMid meet" });
  host.innerHTML = ""; host.appendChild(svg);
  // turned 180 degrees so the plan reads like the 3D view (looking from the window wall); text is turned back upright
  const world = el("g", { transform: "rotate(180)" }, svg);

  const grid = el("g", { class: "grid" }, world);
  for (let u = Math.ceil(u0 * 2) / 2; u <= u1; u += 0.5) el("line", { x1: u, x2: u, y1: -v1, y2: -v0 }, grid);
  for (let v = Math.ceil(v0 * 2) / 2; v <= v1; v += 0.5) el("line", { x1: u0, x2: u1, y1: -v, y2: -v }, grid);
  el("polygon", { class: "floor", points: out.map(([u, v]) => `${u},${-v}`).join(" ") }, world);

  // overall dimensions
  const dim = el("g", {}, world);
  el("line", { class: "dimline", x1: u0, x2: u1, y1: -v1 - 0.35, y2: -v1 - 0.35 }, dim);
  el("text", { class: "dim", x: (u0 + u1) / 2, y: -v1 - 0.5, transform: `rotate(180 ${(u0 + u1) / 2} ${-v1 - 0.5})` }, dim).textContent = `${(u1 - u0).toFixed(2)} m`;
  el("line", { class: "dimline", x1: u1 + 0.35, x2: u1 + 0.35, y1: -v1, y2: -v0 }, dim);
  const t = el("text", { class: "dim", x: u1 + 0.5, y: -(v0 + v1) / 2, transform: `rotate(90 ${u1 + 0.5} ${-(v0 + v1) / 2})` }, dim);
  t.textContent = `${(v1 - v0).toFixed(2)} m`;

  // hand-built parts (headboard, TV panel, door): context only
  const parts = el("g", { class: "part" }, world);
  for (const p of scene.parts || []) {
    if (p.kind !== "box") continue;
    const [u, v] = F.toUV(p.center);
    const g = el("g", { transform: `translate(${u} ${-v}) rotate(${-(p.yaw - F.mdeg)})` }, parts);
    el("rect", { x: -p.size[0] / 2, y: -Math.max(p.size[1], 0.05) / 2, width: p.size[0], height: Math.max(p.size[1], 0.05) }, g);
  }

  const layer = el("g", {}, world), labels = el("g", {}, world);
  const nodes = {};
  let selected = null;

  for (const o of objects) {
    const flat = o.size[1] < 0.05 || o.size[2] < 0.02;
    const g = el("g", { class: "obj" + (flat ? " flat" : "") }, layer);
    const w = o.size[0], d = Math.max(o.size[1], 0.07);
    el("rect", { x: -w / 2, y: -d / 2, width: w, height: d, rx: 0.03 }, g);
    if (!flat) el("line", { class: "front", x1: -w / 2 + 0.05, x2: w / 2 - 0.05, y1: d / 2, y2: d / 2 }, g);
    const lab = el("g", { class: "obj" }, labels);
    const tx = el("text", {}, lab); tx.textContent = !flat && (o.size[0] >= 0.5 || o.size[1] >= 0.5) ? o.label : "";
    nodes[o.id] = { g, lab, o };
    g.addEventListener("pointerdown", (e) => down(e, o.id));
  }

  function update(id) {
    const { g, lab, o } = nodes[id]; const [u, v] = F.toUV(o.xy);
    g.setAttribute("transform", `translate(${u} ${-v}) rotate(${-(o.yaw_deg - F.mdeg)})`);
    lab.setAttribute("transform", `translate(${u} ${-v}) rotate(180)`);
    const off = opts.isOff ? opts.isOff(o) : false;
    g.classList.toggle("off", off); lab.classList.toggle("off", off);
  }
  function render() { for (const id in nodes) update(id); }
  function select(id) {
    selected = id;
    for (const k in nodes) nodes[k].g.classList.toggle("sel", k === id);
    if (id && nodes[id]) layer.appendChild(nodes[id].g);  // on top
  }

  function svgPoint(e) {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(world.getScreenCTM().inverse()); return [p.x, -p.y];  // -> (u, v)
  }
  let drag = null;
  function down(e, id) {
    e.stopPropagation(); opts.onSelect && opts.onSelect(id);
    if (!opts.editable) return;
    const o = nodes[id].o, [u, v] = F.toUV(o.xy), p = svgPoint(e);
    drag = { id, off: [u - p[0], v - p[1]], before: { xy: [...o.xy], yaw_deg: o.yaw_deg }, moved: false };
    svg.setPointerCapture(e.pointerId);
  }
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const p = svgPoint(e), o = nodes[drag.id].o;
    const nxy = F.fromUV([p[0] + drag.off[0], p[1] + drag.off[1]]);
    if (opts.canPlace && !opts.canPlace(nxy)) return;  // stays inside the room
    o.xy = nxy; drag.moved = true;
    update(drag.id); opts.onMove && opts.onMove(drag.id);
  });
  const up = () => { if (drag && drag.moved && opts.onCommit) opts.onCommit(drag.id, drag.before); drag = null; };
  svg.addEventListener("pointerup", up); svg.addEventListener("pointercancel", up);
  svg.addEventListener("pointerdown", () => opts.onSelect && opts.onSelect(null));

  render();
  return { render, select, update, dims: [u1 - u0, v1 - v0] };
}
