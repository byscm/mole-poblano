// 3D room editor: floor and walls from the scene outline, each object as its generated model (or an image plane for
// flat things) scaled to its real size, placed exactly like pipeline/build_house.py. Z is up, metres.
// createEditor(el, scene, objects, kinds, cb) -> {select, sync, setWalls, setView, ready}
//   cb: {onSelect(id), onMove(id), onCommit(id, before)}
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const hex = (h) => new THREE.Color("#" + h);
const HUNG = ["wall_sconce", "bedside_table", "shelf_unit", "stone_wall", "niche_painting", "small_painting", "curtains"];

export function createEditor(host, scene, objects, kinds, cb = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
  host.appendChild(renderer.domElement);

  const s3 = new THREE.Scene(); s3.background = new THREE.Color(0xe9e5de);
  const cen = scene.outline.reduce((a, p) => [a[0] + p[0] / scene.outline.length, a[1] + p[1] / scene.outline.length], [0, 0]);
  const cam = new THREE.PerspectiveCamera(42, 1, 0.05, 200);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.enableDamping = true; controls.maxPolarAngle = Math.PI / 2 - 0.04; controls.minDistance = 1.5; controls.maxDistance = 20;
  const m = (scene.main_dir_deg * Math.PI) / 180;
  function orbitView() {
    cam.position.set(cen[0] - 6.0 * Math.sin(m) + 2.5 * Math.cos(m), cen[1] + 6.0 * Math.cos(m) + 2.5 * Math.sin(m), 6.2);
    controls.target.set(cen[0], cen[1], 0.4);
  }
  // straight down, turned like the 2D plan (stone wall side up): a tiny offset toward +V sets which way is "up"
  function topView() { cam.position.set(cen[0] - 0.02 * Math.sin(m), cen[1] + 0.02 * Math.cos(m), 10); controls.target.set(cen[0], cen[1], 0); }
  orbitView();

  s3.add(new THREE.HemisphereLight(0xfffaf2, 0x8c8275, 1.7));
  const sun = new THREE.DirectionalLight(0xfff1de, 2.2);
  sun.position.set(cen[0] + 3, cen[1] - 4, 9); sun.target.position.set(cen[0], cen[1], 0); s3.add(sun, sun.target);
  sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 1, far: 25 }); sun.shadow.bias = -0.0005;

  // floor
  const shape = new THREE.Shape(scene.outline.map(([x, y]) => new THREE.Vector2(x, y)));
  const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: hex(scene.colors.floor), roughness: 0.6 }));
  floor.receiveShadow = true; s3.add(floor);

  // walls (dollhouse: the ones between the camera and the room hide themselves)
  const H = scene.ceiling_m, wallMat = new THREE.MeshStandardMaterial({ color: hex(scene.colors.wall), roughness: 0.9 });
  const walls = scene.walls.map(([p0, p1]) => {
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1], L = Math.hypot(dx, dy), a = Math.atan2(dy, dx);
    const w = new THREE.Mesh(new THREE.BoxGeometry(L + 0.1, 0.1, H), wallMat);
    const mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
    w.position.set(mid[0], mid[1], H / 2); w.rotation.z = a; w.castShadow = w.receiveShadow = true;
    let n = [-Math.sin(a), Math.cos(a)];
    if ((mid[0] - cen[0]) * n[0] + (mid[1] - cen[1]) * n[1] < 0) n = [-n[0], -n[1]];
    s3.add(w);
    return { mesh: w, mid, n, p0, dir: [dx / L, dy / L], L };
  });
  function nearestWall(xy) {
    let best = null;
    for (const w of walls) {
      const t = Math.max(0, Math.min(w.L, (xy[0] - w.p0[0]) * w.dir[0] + (xy[1] - w.p0[1]) * w.dir[1]));
      const d = Math.hypot(xy[0] - (w.p0[0] + w.dir[0] * t), xy[1] - (w.p0[1] + w.dir[1] * t));
      if (!best || d < best[0]) best = [d, w];
    }
    return best && best[0] < 0.45 ? best[1] : null;
  }

  // fixed parts: headboard, TV panel, TV, door (ceiling spots are skipped: the ceiling is open)
  const partObjs = [];
  for (const p of scene.parts || []) {
    if (p.kind !== "box") continue;
    const piv = new THREE.Group(); piv.position.set(...p.center); piv.rotation.z = (p.yaw * Math.PI) / 180;
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...p.size), new THREE.MeshStandardMaterial({ color: hex(p.mat), roughness: 0.55 }));
    mesh.rotation.y = ((p.pitch || 0) * Math.PI) / 180; mesh.castShadow = mesh.receiveShadow = true;
    piv.add(mesh); s3.add(piv); partObjs.push({ obj: piv, xy: [p.center[0], p.center[1]] });
  }

  // objects
  const loader = new GLTFLoader(), texLoader = new THREE.TextureLoader(), gltfCache = {};
  const loadGLB = (url) => (gltfCache[url] ||= loader.loadAsync(url));
  const pivots = {}, pickables = [];

  function texturedPlane(url, w, h) {
    const tex = texLoader.load(url); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
    const mat = new THREE.MeshStandardMaterial({ map: tex, transparent: true, alphaTest: 0.35, side: THREE.DoubleSide, roughness: 0.85 });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat); mesh.receiveShadow = true; return mesh;
  }

  async function build(o) {
    const k = kinds[o.kind] || {}; const [w, d, h] = o.size; const z = o.base_z || 0;
    const piv = new THREE.Group(); piv.userData.id = o.id; s3.add(piv); pivots[o.id] = piv;
    if (k.render === "wall_plane" && k.cutout) {
      const pl = texturedPlane(k.cutout, w, h); pl.rotation.x = Math.PI / 2; pl.position.z = z + h / 2; pl.castShadow = true; piv.add(pl);
    } else if (k.render === "floor_plane" && k.cutout) {
      const pl = texturedPlane(k.cutout, w, d); pl.position.z = z + Math.max(h, 0.004); piv.add(pl);
    } else if (k.model) {
      const gl = await loadGLB(k.model);
      const wrap = new THREE.Group(); const g = gl.scene.clone(true); g.rotation.x = Math.PI / 2; wrap.add(g);  // glTF y-up -> z-up
      wrap.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(wrap, true), mn = box.min, mx = box.max;
      const ex = mx.x - mn.x, ey = mx.y - mn.y, ez = mx.z - mn.z;
      // same as build_house.py: long furniture turned so its long side runs along the box width
      const align = (w >= d) !== (ex >= ey) && Math.abs(w - d) > 0.15 * Math.max(w, d) ? Math.PI / 2 : 0;
      const A = new THREE.Matrix4().makeRotationZ(align).multiply(new THREE.Matrix4().makeTranslation(-(mn.x + mx.x) / 2, -(mn.y + mx.y) / 2, -mn.z));
      const ext = align ? [ey, ex, ez] : [ex, ey, ez];
      const inner = new THREE.Group(); inner.matrixAutoUpdate = false;
      inner.matrix.copy(new THREE.Matrix4().makeScale(w / ext[0], d / ext[1], h / ext[2]).multiply(A));
      inner.add(wrap);
      const holder = new THREE.Group(); holder.position.z = z; piv.add(holder); holder.add(inner);
      wrap.traverse((c) => { if (c.isMesh) { c.castShadow = c.receiveShadow = true; } });
    } else {
      const b = new THREE.Mesh(new THREE.BoxGeometry(w, d, h), new THREE.MeshStandardMaterial({ color: 0xc8b9a6 }));
      b.position.z = z + h / 2; b.castShadow = true; piv.add(b);
    }
    piv.traverse((c) => { if (c.isMesh) { c.userData.id = o.id; pickables.push(c); } });
    sync(o.id);
  }

  const byId = Object.fromEntries(objects.map((o) => [o.id, o]));
  function sync(id) {
    const o = byId[id], piv = pivots[id]; if (!piv) return;
    piv.position.set(o.xy[0], o.xy[1], 0); piv.rotation.z = (o.yaw_deg * Math.PI) / 180;
    piv.visible = !o.off;
    if (id === selected) helper.setFromObject(piv);
  }

  // selection + dragging on the floor
  let selected = null;
  const helper = new THREE.BoxHelper(new THREE.Object3D(), 0xf29a1a); helper.visible = false; s3.add(helper);
  function select(id) {
    selected = id; helper.visible = !!(id && pivots[id]);
    if (helper.visible) helper.setFromObject(pivots[id]);
  }
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), ground = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), hitP = new THREE.Vector3();
  function setRay(e) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ndc, cam);
  }
  function pick(e) {
    setRay(e);
    const hits = ray.intersectObjects(pickables.filter((p) => isShown(p)), false);
    return hits.length ? hits[0].object.userData.id : null;
  }
  function isShown(o) { while (o) { if (!o.visible) return false; o = o.parent; } return true; }
  let drag = null, downAt = null;
  renderer.domElement.addEventListener("pointerdown", (e) => {
    downAt = [e.clientX, e.clientY];
    const id = pick(e);
    if (!id) return;
    controls.enabled = false; cb.onSelect && cb.onSelect(id);
    setRay(e); ray.ray.intersectPlane(ground, hitP);
    const o = byId[id];
    drag = { id, off: [o.xy[0] - hitP.x, o.xy[1] - hitP.y], before: { xy: [...o.xy], yaw_deg: o.yaw_deg }, moved: false };
    renderer.domElement.setPointerCapture(e.pointerId);
  });
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (!drag) { renderer.domElement.style.cursor = pick(e) ? "grab" : "default"; return; }
    setRay(e); if (!ray.ray.intersectPlane(ground, hitP)) return;
    const o = byId[drag.id], nxy = [hitP.x + drag.off[0], hitP.y + drag.off[1]];
    if (cb.canPlace && !cb.canPlace(nxy)) return;  // stays inside the room
    o.xy = nxy; drag.moved = true;
    renderer.domElement.style.cursor = "grabbing"; sync(drag.id); cb.onMove && cb.onMove(drag.id);
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (drag) { if (drag.moved && cb.onCommit) cb.onCommit(drag.id, drag.before); drag = null; controls.enabled = true; return; }
    if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 4) cb.onSelect && cb.onSelect(null);
  });

  // walls mode: "auto" = dollhouse cut-away, "off" = no walls
  let wallMode = "auto";
  function updateWalls() {
    const shown = new Map();
    for (const w of walls) {
      const dx = cam.position.x - w.mid[0], dy = cam.position.y - w.mid[1], L = Math.hypot(dx, dy) || 1;
      const vis = wallMode === "auto" && (w.n[0] * dx + w.n[1] * dy) / L < 0.25;
      w.mesh.visible = vis; shown.set(w, vis);
    }
    for (const p of partObjs) { const w = nearestWall(p.xy); p.obj.visible = !w || shown.get(w); }
    for (const o of objects) {  // pieces hung on a hidden wall go with it
      if (!pivots[o.id] || o.off) continue;
      if (HUNG.includes(o.kind)) { const w = nearestWall(o.xy); pivots[o.id].visible = !w || shown.get(w); }
    }
  }

  function resize() {
    const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host); resize();
  renderer.setAnimationLoop(() => { controls.update(); updateWalls(); if (selected && pivots[selected]) helper.setFromObject(pivots[selected]); renderer.render(s3, cam); });

  const ready = Promise.all(objects.map(build));
  return {
    ready, select, sync,
    setWalls: (on) => { wallMode = on ? "auto" : "off"; },
    setView: (v) => (v === "top" ? topView() : orbitView()),
  };
}
