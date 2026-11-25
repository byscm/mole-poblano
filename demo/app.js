// Wizard: 1 upload video -> 2 map + object gallery (user approves) -> 3 model generation -> 4 3D + 2D editor.
import * as api from "./api.js";
import { createPlan, roomFrame } from "./plan2d.js";

const $ = (s) => document.querySelector(s);
const state = { files: [], scene: null, approved: {}, objects: [], original: {}, history: [], step: 1, maxStep: 1 };

// ---------- navigation ----------
function goStep(n) {
  state.step = n; state.maxStep = Math.max(state.maxStep, n);
  for (let i = 1; i <= 4; i++) $("#step" + i).classList.toggle("hidden", i !== n);
  document.querySelectorAll("#steps li").forEach((li) => {
    const s = +li.dataset.step; li.classList.toggle("on", s === n); li.classList.toggle("done", s < n);
  });
  window.scrollTo(0, 0);
  if (n === 2) showStep2(); if (n === 3) showStep3(); if (n === 4) showStep4();
}
document.querySelectorAll("#steps li").forEach((li) => li.addEventListener("click", () => {
  const s = +li.dataset.step; if (s < state.step && s >= 2) goStep(s);
}));

// ---------- step 1: upload + processing ----------
const drop = $("#drop"), fileInput = $("#fileInput");
function setFiles(list) {
  state.files = [...list].filter((f) => f.type.startsWith("video/") || /\.(mp4|mov|m4v)$/i.test(f.name));
  $("#fileList").innerHTML = state.files.map((f, i) => `<div class="file-item"><video src="${URL.createObjectURL(f)}" muted></video>
    <div><b>${f.name}</b><div class="muted small">${(f.size / 1e6).toFixed(1)} MB · take ${String.fromCharCode(65 + i)}</div></div></div>`).join("");
  $("#toProcess").disabled = !state.files.length;
}
fileInput.addEventListener("change", () => setFiles(fileInput.files));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); setFiles(e.dataTransfer.files); });
$("#toProcess").addEventListener("click", startProcessing);
$("#useDemo").addEventListener("click", () => { state.files = []; startProcessing(); });

async function startProcessing() {
  $("#uploadPane").classList.add("hidden"); $("#processPane").classList.remove("hidden");
  $("#processSub").textContent = state.files.length ? `${state.files.length} video(s) · this takes a few minutes.` : "Demo video: hotel room, 2 takes.";
  $("#stages").innerHTML = api.STAGES.map((s) => `<li><span class="dot"></span>${s.label}</li>`).join("");
  const lis = [...$("#stages").children], frames = $("#frames"); frames.innerHTML = "";
  const scene = await api.processVideo(state.files, $("#measure").value, ({ stage, frac, frame }) => {
    lis.forEach((li, i) => { li.classList.toggle("done", i < stage || (i === stage && frac >= 1)); li.classList.toggle("run", i === stage && frac < 1); });
    lis.forEach((li, i) => { li.querySelector(".dot").textContent = li.classList.contains("done") ? "✓" : ""; });
    $("#bar").style.width = `${((stage + frac) / api.STAGES.length) * 100}%`;
    if (frame) { const im = new Image(); im.src = frame.img; if (!frame.used) im.className = "drop-frame"; im.title = frame.used ? frame.name : frame.name + " (dropped: blurry or moving)"; frames.appendChild(im); }
    if (stage === 3 && frac === 0) $("#framesCaption").textContent = `${scene0Frames()} frames picked · grey ones were dropped (blurry or no walls)`;
  });
  state.scene = scene;
  state.approved = Object.fromEntries(scene.kinds.map((k) => [k.kind, true]));
  state.kinds = Object.fromEntries(scene.kinds.map((k) => [k.kind, k]));
  setTimeout(() => goStep(2), 500);
}
const scene0Frames = () => $("#frames").children.length;

// ---------- step 2: map + gallery ----------
let plan2 = null;
function polyArea(pts) { let a = 0; for (let i = 0; i < pts.length; i++) { const p = pts[i], q = pts[(i + 1) % pts.length]; a += p[0] * q[1] - q[0] * p[1]; } return Math.abs(a / 2); }

function showStep2() {
  const sc = state.scene, objs = sc.objects.map((o) => ({ ...o, xy: [...o.xy] }));
  const area = polyArea(sc.outline);
  $("#stats2").innerHTML = [
    [`${area.toFixed(1)} m²`, "floor area"], [`${sc.ceiling_m.toFixed(2)} m`, "ceiling height"],
    [sc.stats.objects, "objects"], [`${sc.stats.frames_used}/${sc.stats.frames_total}`, "frames used"],
  ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  $("#kindCount").textContent = sc.kinds.length;
  renderGallery();
  plan2 = null; setMapView("plan", objs);
}

function setMapView(v, objs) {
  document.querySelectorAll("#mapTabs button").forEach((b) => b.classList.toggle("on", b.dataset.v === v));
  const host = $("#mapView"), sc = state.scene;
  if (v === "plan") {
    plan2 = createPlan(host, sc, objs || state.scene.objects, {
      onSelect: (id) => { if (!id) return; const o = sc.objects.find((x) => x.id === id); highlightCard(o.kind, true); plan2.select(id); },
      isOff: (o) => !state.approved[o.kind],
    });
    $("#mapNote").textContent = "Real scale, in metres. Tap an object to find it in the list.";
  } else if (v === "raw") {
    host.innerHTML = `<img src="${sc.plans.plano_v0}" alt="raw plan from the video">`;
    $("#mapNote").textContent = "What was measured straight from the video (3D points seen from above), before cleaning it up.";
  } else {
    host.innerHTML = `<div class="frames">${sc.frames.map((f) => `<img src="${f.img}" class="${f.used ? "" : "drop-frame"}" title="${f.name}">`).join("")}</div>`;
    $("#mapNote").textContent = `${sc.stats.frames_used} frames used to measure the room; grey ones were dropped.`;
  }
}
document.querySelectorAll("#mapTabs button").forEach((b) => b.addEventListener("click", () => setMapView(b.dataset.v)));

function renderGallery() {
  const g = $("#gallery");
  g.innerHTML = state.scene.kinds.map((k) => {
    const [w, d, h] = k.size_m, on = state.approved[k.kind];
    const how = k.render === "model" ? "3D model" : k.render === "wall_plane" ? "image on the wall" : "image on the floor";
    return `<div class="card ${on ? "" : "rejected"}" id="card-${k.kind}">
      <div class="card-top"><h3>${k.label}${k.count > 1 ? ` ×${k.count}` : ""}<span class="tag ${k.sure === "photo" ? "" : "inf"}">${k.sure === "photo" ? "seen in the video" : "inferred"}</span></h3>
        <div class="meta">${w.toFixed(2)} × ${d.toFixed(2)} × ${h.toFixed(2)} m<br>${how}</div></div>
      <div class="flow">
        <div class="crops">${k.crops.map((c) => `<figure><img src="${c.img}" loading="lazy"><figcaption>fragment</figcaption></figure>`).join("")}</div>
        <div class="arrow">→</div>
        <img class="clean" src="${k.cutout || k.clean_view}" loading="lazy" alt="${k.label}">
      </div>
      <div class="card-sub">${k.crops.length} video fragment(s) → clean image of the whole object</div>
      <div class="card-actions">
        <button class="ok ${on ? "on" : ""}" data-k="${k.kind}" data-a="ok">✓ Approve</button>
        <button class="no ${on ? "" : "on"}" data-k="${k.kind}" data-a="no">✕ Remove</button>
        <button data-k="${k.kind}" data-a="regen">↻ Regenerate image</button>
      </div></div>`;
  }).join("");
  g.querySelectorAll("button[data-a]").forEach((b) => b.addEventListener("click", async () => {
    const k = b.dataset.k;
    if (b.dataset.a === "regen") {
      b.disabled = true; b.textContent = "Generating…";
      await api.regenerateView(state.kinds[k]); b.disabled = false; b.textContent = "↻ Regenerate image"; return;
    }
    state.approved[k] = b.dataset.a === "ok"; updateApprovals();
  }));
  g.querySelectorAll(".card").forEach((c) => c.addEventListener("mouseenter", () => {
    const k = c.id.slice(5), o = state.scene.objects.find((x) => x.kind === k); if (plan2 && o) plan2.select(o.id);
  }));
  updateApprovals();
}
function updateApprovals() {
  for (const k of state.scene.kinds) {
    const c = $("#card-" + k.kind), on = state.approved[k.kind];
    c.classList.toggle("rejected", !on); c.querySelector(".ok").classList.toggle("on", on); c.querySelector(".no").classList.toggle("on", !on);
  }
  const n = Object.values(state.approved).filter(Boolean).length;
  $("#confirm2").textContent = `Confirm ${n} object(s) and generate 3D models →`; $("#confirm2").disabled = !n;
  if (plan2) plan2.render();
}
function highlightCard(kind, scroll) {
  document.querySelectorAll(".card").forEach((c) => c.classList.toggle("hl", c.id === "card-" + kind));
  if (scroll) $("#card-" + kind)?.scrollIntoView({ behavior: "smooth", block: "center" });
}
$("#approveAll").addEventListener("click", () => { for (const k in state.approved) state.approved[k] = true; updateApprovals(); });
$("#back2").addEventListener("click", () => { $("#uploadPane").classList.remove("hidden"); $("#processPane").classList.add("hidden"); goStep(1); });
$("#confirm2").addEventListener("click", () => goStep(3));

// ---------- step 3: model generation ----------
let generated = false;
async function showStep3() {
  if (generated) return;
  const ks = state.scene.kinds.filter((k) => state.approved[k.kind]);
  $("#genGrid").innerHTML = ks.map((k) => `<div class="gen" id="gen-${k.kind}"><div class="img"><img src="${k.cutout || k.clean_view}"></div>
    <div class="name">${k.label}</div><div class="mini"><div></div></div><div class="st">Waiting</div></div>`).join("");
  $("#toEditor").disabled = true; let done = 0;
  await api.generateModels(ks, ({ kind, frac, done: fin }) => {
    const c = $("#gen-" + kind); c.classList.add("run"); c.querySelector(".mini div").style.width = frac * 100 + "%";
    const k = state.kinds[kind];
    c.querySelector(".st").textContent = fin ? (k.render === "model" ? "✓ Model ready" : "✓ Ready (flat image)") : k.render === "model" ? "Generating shape and texture…" : "Cutting out…";
    if (fin) { c.classList.remove("run"); c.classList.add("done"); done++; $("#bar3").style.width = (done / ks.length) * 100 + "%"; }
  });
  generated = true; $("#toEditor").disabled = false;
}
$("#toEditor").addEventListener("click", () => goStep(4));

// ---------- step 4: editor ----------
let editor = null, plan4 = null, selected = null;
async function showStep4() {
  if (editor) return;
  const sc = state.scene, F = roomFrame(sc);
  state.objects = sc.objects.filter((o) => state.approved[o.kind]).map((o) => ({ ...o, xy: [...o.xy] }));
  state.original = Object.fromEntries(state.objects.map((o) => [o.id, { xy: [...o.xy], yaw_deg: o.yaw_deg }]));
  const { createEditor } = await import("./editor3d.js");
  const cbs = {
    onSelect: (id) => select(id),
    onMove: (id) => { editor.sync(id); plan4.update(id); showSel(); },
    onCommit: (id, before) => state.history.push({ id, ...before }),
    canPlace: (xy) => insidePoly(sc.outline, xy),
  };
  editor = createEditor($("#viewport"), sc, state.objects, state.kinds, cbs);
  plan4 = createPlan($("#plan4"), sc, state.objects, { editable: true, ...cbs });
  $("#roomDims").textContent = `${plan4.dims[0].toFixed(2)} × ${plan4.dims[1].toFixed(2)} m`;
  editor.ready.then(() => $("#vpLoading").classList.add("hidden")).catch((e) => { $("#vpLoading").textContent = "Error loading models: " + e.message; });
}

function insidePoly(poly, [x, y]) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}
function select(id) { selected = id; editor.select(id); plan4.select(id); showSel(); }
function obj(id) { return state.objects.find((o) => o.id === id); }
function showSel() {
  const p = $("#selPanel"), o = obj(selected);
  if (!o) { p.innerHTML = `<div class="muted">Tap a piece of furniture in 3D or on the plan to edit it.</div>`; return; }
  const k = state.kinds[o.kind], F = roomFrame(state.scene), [u, v] = F.toUV(o.xy);
  const minU = Math.min(...state.scene.outline.map((q) => F.toUV(q)[0])), minV = Math.min(...state.scene.outline.map((q) => F.toUV(q)[1]));
  const rel = ((((o.yaw_deg - F.mdeg) % 360) + 360) % 360);
  if (p.dataset.id !== o.id) {
    p.dataset.id = o.id;
    p.innerHTML = `<div class="sel-head"><img src="${k.cutout || k.clean_view}"><div><h3>${o.label}</h3><div class="muted small">${o.id}</div></div></div>
      <div class="kv" id="selKv"></div>
      <label class="small muted">Rotation</label>
      <div class="rot-row"><button id="rL">↺ 15°</button><input type="range" id="rS" min="0" max="359" step="1"><button id="rR">↻ 15°</button></div>
      <div class="btn-row"><button id="rReset">Back to original spot</button><button id="rDesel">Done</button></div>`;
    $("#rL").onclick = () => rotate(-15); $("#rR").onclick = () => rotate(15);
    $("#rS").oninput = (e) => { const o2 = obj(selected); if (!$("#rS").dataset.start) { $("#rS").dataset.start = 1; state.history.push({ id: o2.id, xy: [...o2.xy], yaw_deg: o2.yaw_deg }); } o2.yaw_deg = F.mdeg + +e.target.value; refresh(o2.id, true); };
    $("#rS").onchange = () => { delete $("#rS").dataset.start; };
    $("#rReset").onclick = () => { const o2 = obj(selected); state.history.push({ id: o2.id, xy: [...o2.xy], yaw_deg: o2.yaw_deg }); Object.assign(o2, { xy: [...state.original[o2.id].xy], yaw_deg: state.original[o2.id].yaw_deg }); refresh(o2.id); };
    $("#rDesel").onclick = () => select(null);
  }
  $("#selKv").innerHTML = `<span>Size</span><span>${o.size[0].toFixed(2)} × ${o.size[1].toFixed(2)} × ${o.size[2].toFixed(2)} m</span>
    <span>Position</span><span>${(u - minU).toFixed(2)} m, ${(v - minV).toFixed(2)} m</span><span>Rotation</span><span>${Math.round(rel)}°</span>`;
  if (document.activeElement !== $("#rS")) $("#rS").value = Math.round(rel);
}
function refresh(id, keepSlider) { editor.sync(id); plan4.update(id); showSel(); }
function rotate(deg) {
  const o = obj(selected); if (!o) return;
  state.history.push({ id: o.id, xy: [...o.xy], yaw_deg: o.yaw_deg }); o.yaw_deg = (o.yaw_deg + deg + 360) % 360; refresh(o.id);
}
function undo() {
  const h = state.history.pop(); if (!h) return;
  const o = obj(h.id); o.xy = h.xy; o.yaw_deg = h.yaw_deg; refresh(o.id); select(o.id);
}
document.addEventListener("keydown", (e) => {
  if (state.step !== 4 || e.target.tagName === "INPUT") return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  else if (e.key === "r" || e.key === "R") rotate(e.shiftKey ? -15 : 15);
  else if (e.key === "Escape") select(null);
});
$("#tUndo").addEventListener("click", undo);
$("#tWalls").addEventListener("click", (e) => { const on = !e.target.classList.contains("on"); e.target.classList.toggle("on", on); editor.setWalls(on); });
$("#tTop").addEventListener("click", () => { editor.setView("top"); $("#tTop").classList.add("on"); $("#tOrbit").classList.remove("on"); });
$("#tOrbit").addEventListener("click", () => { editor.setView("orbit"); $("#tOrbit").classList.add("on"); $("#tTop").classList.remove("on"); });
$("#tExport").addEventListener("click", () => api.saveLayout({
  scene: state.scene.name, units: "m", frame: "world, z up (same as inventory_map.json)",
  objects: state.objects.map((o) => ({ id: o.id, kind: o.kind, xy: o.xy.map((x) => +x.toFixed(3)), yaw_deg: +o.yaw_deg.toFixed(1), size: o.size, base_z: o.base_z })),
}));

goStep(1);

// ?auto=2|3|4 runs the demo by itself up to that step (testing / presenting); add &fast for no pauses
const AUTO = +new URLSearchParams(location.search).get("auto");
if (AUTO >= 2) (async () => {
  const until = (f) => new Promise((r) => { const t = setInterval(() => { if (f()) { clearInterval(t); r(); } }, 100); });
  $("#useDemo").click(); await until(() => state.step === 2);
  if (AUTO >= 3) { goStep(3); await until(() => generated); }
  if (AUTO >= 4) goStep(4);
})();
