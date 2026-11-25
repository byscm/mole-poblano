// Visor de cuartos 3D: volar / caminar, joystick táctil, vistas fijas.
// three.js r185 guardado en ./lib (funciona sin internet).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { t, loc, setLang, onLangChange, applyStatic } from './i18n.js';

// ---------- Ajustes ----------
const ALTURA_OJOS = 1.6;      // m sobre el piso en modo caminar
const RADIO = 0.22;           // m, "cuerpo" para chocar con paredes
let CELDA = 0.05;             // m, tamaño de celda del mapa de colisiones (0.25 en exteriores grandes)
const BANDA_MIN = 0.3;        // m sobre el piso: lo que esté entre BANDA_MIN y BANDA_MAX bloquea el paso
const BANDA_MAX = 1.75;
const VEL_CAMINAR = 1.4, VEL_VOLAR = 2.2, FACTOR_RAPIDO = 2.2;
const SENS_MOUSE = 0.0022, SENS_TACTIL = 0.0045;

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const lienzo = $('escena'), selCuarto = $('selCuarto'), divVistas = $('vistas');
// versión del paquete (visor.js?v=...): se agrega a los .json para que el CDN no dé copias viejas
const VER = new URL(import.meta.url).search;
const carga = $('carga'), cargaBarra = $('cargaBarra'), cargaTexto = $('cargaTexto'), cargaTitulo = $('cargaTitulo');
const esTactil = matchMedia('(pointer: coarse)').matches || ('ontouchstart' in window && navigator.maxTouchPoints > 0);
if (esTactil) document.body.classList.add('tactil');
applyStatic();

// ---------- Render ----------
const renderer = new THREE.WebGLRenderer({ canvas: lienzo, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, esTactil ? 1.5 : 2));
renderer.setSize(innerWidth, innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
// AgX, igual que los renders de Blender/Cycles.
renderer.toneMapping = THREE.AgXToneMapping;
renderer.toneMappingExposure = 1.0;
let exposicionBase = 1.0;   // 2^(exposición de la escena de Blender), por cuarto
let luzHorneada = true;     // usar la copia horneada si el cuarto la tiene
let aoActivo = !esTactil;   // oclusión ambiental (se apaga sola con luz horneada: ya la trae)

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1d21);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.9;

const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.05, 300);
camera.rotation.order = 'YXZ';

// Postproceso: escena → GTAO (sombras suaves en rincones y de contacto) → salida con AgX.
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const gtao = new GTAOPass(scene, camera, innerWidth, innerHeight);
gtao.updateGtaoMaterial({ radius: 0.45, distanceExponent: 1.5, thickness: 1.0, scale: 1.0, samples: 12 });
gtao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 16 });
gtao.blendIntensity = 1.0;
composer.addPass(gtao);
composer.addPass(new OutputPass());

// Luces fijas: cielo/piso suave + una "ventana" direccional. Se agregan luces puntuales por cuarto.
const hemi = new THREE.HemisphereLight(0xfff6ea, 0x6b5d50, 1.1);
scene.add(hemi);
const sol = new THREE.DirectionalLight(0xfff1dd, 0.9);
sol.position.set(3, 8, 4);
scene.add(sol);
let lucesCuarto = new THREE.Group();
scene.add(lucesCuarto);

const loader = new GLTFLoader();
loader.setMeshoptDecoder(MeshoptDecoder);

// ---------- Estado ----------
let lista = [], actual = null, modelo = null, mapa = null, vistas = [];
let modo = 'volar';
let yaw = 0, pitch = 0;
const teclas = new Set();
const joy = { x: 0, y: 0 };
let subirTactil = 0;
let transicion = null;
let pisoY = 0;

// ---------- Utilidades UI ----------
let avisoTimer = 0;
function aviso(txt, ms = 2200) {
  const a = $('aviso'); a.textContent = txt; a.style.display = 'block';
  clearTimeout(avisoTimer); avisoTimer = setTimeout(() => (a.style.display = 'none'), ms);
}
function progreso(frac, txt) {
  cargaBarra.style.width = `${Math.round(frac * 100)}%`;
  if (txt !== undefined) cargaTexto.textContent = txt;
}
const pausa = () => new Promise((r) => setTimeout(r, 0));

// ---------- Mapa de colisiones (rejilla 2D sobre el piso) ----------
// Se "pintan" en una rejilla los triángulos del modelo: los que están cerca del piso marcan piso,
// los que están entre BANDA_MIN y BANDA_MAX (paredes, muebles) marcan obstáculo.
async function construirMapa(raiz) {
  raiz.updateMatrixWorld(true);
  const caja = new THREE.Box3().setFromObject(raiz);
  const mallas = [];
  raiz.traverse((o) => { if (o.isMesh && o.geometry?.attributes?.position) mallas.push(o); });

  // Piso: altura más común de las superficies horizontales bajas (histograma a 2 cm).
  const hist = new Map();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), n = new THREE.Vector3();
  const recorrerTri = (malla, fn) => {
    const g = malla.geometry, pos = g.attributes.position, idx = g.index;
    const total = idx ? idx.count : pos.count;
    // Las mallas instanciadas (p. ej. paredes repetidas) se recorren una vez por instancia.
    const matrices = [];
    if (malla.isInstancedMesh) {
      for (let k = 0; k < malla.count; k++) {
        const mi = new THREE.Matrix4(); malla.getMatrixAt(k, mi);
        matrices.push(mi.premultiply(malla.matrixWorld));
      }
    } else matrices.push(malla.matrixWorld);
    for (const m of matrices) {
      for (let i = 0; i < total; i += 3) {
        const i0 = idx ? idx.getX(i) : i, i1 = idx ? idx.getX(i + 1) : i + 1, i2 = idx ? idx.getX(i + 2) : i + 2;
        a.fromBufferAttribute(pos, i0).applyMatrix4(m);
        b.fromBufferAttribute(pos, i1).applyMatrix4(m);
        c.fromBufferAttribute(pos, i2).applyMatrix4(m);
        fn(a, b, c);
      }
    }
  };
  const limiteBajo = caja.min.y + 0.6;
  for (const malla of mallas) {
    const bb = new THREE.Box3().setFromObject(malla);
    if (bb.min.y > limiteBajo) continue;
    recorrerTri(malla, (a, b, c) => {
      const y = (a.y + b.y + c.y) / 3;
      if (y > limiteBajo) return;
      e1.subVectors(b, a); e2.subVectors(c, a); n.crossVectors(e1, e2);
      const area = n.length() / 2; if (area < 1e-6) return;
      if (Math.abs(n.y) / (2 * area) < 0.9) return; // casi horizontal
      const k = Math.round(y / 0.02);
      hist.set(k, (hist.get(k) || 0) + area);
    });
  }
  let mejor = null, mejorArea = 0;
  for (const [k, v] of hist) if (v > mejorArea) { mejorArea = v; mejor = k; }
  const piso = mejor !== null ? mejor * 0.02 : caja.min.y;

  const x0 = caja.min.x - 0.5, z0 = caja.min.z - 0.5;
  const ancho = Math.ceil((caja.max.x - caja.min.x + 1) / CELDA), alto = Math.ceil((caja.max.z - caja.min.z + 1) / CELDA);
  const pisoM = new Uint8Array(ancho * alto), obst = new Uint8Array(ancho * alto);
  const yMin = piso + BANDA_MIN, yMax = piso + BANDA_MAX, yPiso = piso + 0.15;
  const marcar = (arr, x, z) => {
    const i = Math.floor((x - x0) / CELDA), j = Math.floor((z - z0) / CELDA);
    if (i >= 0 && j >= 0 && i < ancho && j < alto) arr[j * ancho + i] = 1;
  };
  const p = new THREE.Vector3();
  let hechas = 0;
  for (const malla of mallas) {
    const bb = new THREE.Box3().setFromObject(malla);
    hechas++;
    if (bb.min.y > yMax || bb.max.y < piso - 0.2) continue;   // techo, lámparas altas…
    recorrerTri(malla, (a, b, c) => {
      const lmin = Math.min(a.y, b.y, c.y), lmax = Math.max(a.y, b.y, c.y);
      if (lmin > yMax || lmax < piso - 0.2) return;
      const L = Math.max(a.distanceTo(b), b.distanceTo(c), c.distanceTo(a));
      const pasos = Math.max(1, Math.ceil(L / (CELDA * 0.7)));
      for (let s = 0; s <= pasos; s++) {
        for (let t = 0; t <= pasos - s; t++) {
          const u = s / pasos, v = t / pasos;
          p.set(a.x + (b.x - a.x) * u + (c.x - a.x) * v, a.y + (b.y - a.y) * u + (c.y - a.y) * v, a.z + (b.z - a.z) * u + (c.z - a.z) * v);
          if (p.y >= yMin && p.y <= yMax) marcar(obst, p.x, p.z);
          else if (p.y < yPiso && p.y > piso - 0.2) marcar(pisoM, p.x, p.z);
        }
      }
    });
    if (hechas % 20 === 0) { progreso(0.9 + 0.1 * hechas / mallas.length, t('load.walls')); await pausa(); }
  }
  // Rellenar huecos chicos del piso (cierre morfológico de 3 celdas).
  // dilatar (buscar=1: alguna vecina en 1) o erosionar (buscar=0: alguna vecina en 0 → queda en 0)
  const morfo = (src, r, buscar) => {
    const out = new Uint8Array(src.length);
    for (let j = 0; j < alto; j++) for (let i = 0; i < ancho; i++) {
      let encontrado = false;
      for (let dj = -r; dj <= r && !encontrado; dj++) for (let di = -r; di <= r; di++) {
        const ii = i + di, jj = j + dj;
        const v = ii >= 0 && jj >= 0 && ii < ancho && jj < alto ? src[jj * ancho + ii] : 0;
        if (v === buscar) { encontrado = true; break; }
      }
      out[j * ancho + i] = buscar ? (encontrado ? 1 : 0) : (encontrado ? 0 : 1);
    }
    return out;
  };
  const pisoCerrado = morfo(morfo(pisoM, 3, 1), 3, 0);
  // Libre = hay piso y no hay obstáculo.
  const libre = new Uint8Array(ancho * alto);
  for (let k = 0; k < libre.length; k++) libre[k] = pisoCerrado[k] && !obst[k] ? 1 : 0;
  // Distancia (en celdas) a lo no libre, para elegir vistas lejos de las paredes.
  const dist = new Float32Array(ancho * alto);
  for (let k = 0; k < dist.length; k++) dist[k] = libre[k] ? 1e9 : 0;
  for (let j = 0; j < alto; j++) for (let i = 0; i < ancho; i++) {
    const k = j * ancho + i; if (!dist[k]) continue;
    if (i > 0) dist[k] = Math.min(dist[k], dist[k - 1] + 1);
    if (j > 0) dist[k] = Math.min(dist[k], dist[k - ancho] + 1);
    if (i > 0 && j > 0) dist[k] = Math.min(dist[k], dist[k - ancho - 1] + 1.414);
    if (i < ancho - 1 && j > 0) dist[k] = Math.min(dist[k], dist[k - ancho + 1] + 1.414);
  }
  for (let j = alto - 1; j >= 0; j--) for (let i = ancho - 1; i >= 0; i--) {
    const k = j * ancho + i; if (!dist[k]) continue;
    if (i < ancho - 1) dist[k] = Math.min(dist[k], dist[k + 1] + 1);
    if (j < alto - 1) dist[k] = Math.min(dist[k], dist[k + ancho] + 1);
    if (i < ancho - 1 && j < alto - 1) dist[k] = Math.min(dist[k], dist[k + ancho + 1] + 1.414);
    if (i > 0 && j < alto - 1) dist[k] = Math.min(dist[k], dist[k + ancho - 1] + 1.414);
  }
  return { x0, z0, ancho, alto, libre, dist, piso, caja };
}

function libreEn(x, z) {
  if (!mapa) return true;
  const i = Math.floor((x - mapa.x0) / CELDA), j = Math.floor((z - mapa.z0) / CELDA);
  if (i < 0 || j < 0 || i >= mapa.ancho || j >= mapa.alto) return false;
  return mapa.libre[j * mapa.ancho + i] === 1;
}
function cabe(x, z) {
  if (!libreEn(x, z)) return false;
  for (let k = 0; k < 8; k++) {
    const ang = (k / 8) * Math.PI * 2;
    if (!libreEn(x + Math.cos(ang) * RADIO, z + Math.sin(ang) * RADIO)) return false;
  }
  return true;
}
// Lugar libre más cercano a (x,z).
function lugarLibre(x, z) {
  if (cabe(x, z)) return [x, z];
  for (let r = CELDA; r < 4; r += CELDA) {
    for (let k = 0; k < 24; k++) {
      const ang = (k / 24) * Math.PI * 2, xx = x + Math.cos(ang) * r, zz = z + Math.sin(ang) * r;
      if (cabe(xx, zz)) return [xx, zz];
    }
  }
  return [x, z];
}

// Vistas automáticas: puntos separados entre sí y lejos de paredes, mirando hacia lo más abierto.
function vistasAutomaticas(cant = 4) {
  const { ancho, alto, dist, x0, z0, piso } = mapa;
  const cand = [];
  for (let j = 0; j < alto; j += 2) for (let i = 0; i < ancho; i += 2) {
    const d = dist[j * ancho + i];
    if (d * CELDA >= 0.45 && d < 1e8) cand.push([i, j, d]);
  }
  if (!cand.length) return [];
  cand.sort((p, q) => q[2] - p[2]);
  const elegidos = [cand[0]];
  while (elegidos.length < cant) {
    let mejor = null, mejorVal = -1;
    for (const c of cand) {
      let dmin = Infinity;
      for (const e of elegidos) dmin = Math.min(dmin, Math.hypot(c[0] - e[0], c[1] - e[1]));
      const val = dmin * Math.min(1, (c[2] * CELDA) / 0.8);
      if (val > mejorVal) { mejorVal = val; mejor = c; }
    }
    if (!mejor || mejorVal * CELDA < 1.0) break;
    elegidos.push(mejor);
  }
  return elegidos.map(([i, j], n) => {
    const x = x0 + (i + 0.5) * CELDA, z = z0 + (j + 0.5) * CELDA;
    // Mirar hacia donde se ve más superficie del cuarto (cono de ~70°), no hacia un pasillo.
    const N = 48, largos = [];
    for (let k = 0; k < N; k++) {
      const ang = (k / N) * Math.PI * 2, dx = Math.cos(ang), dz = Math.sin(ang);
      let r = 0; while (r < 15 && libreEn(x + dx * r, z + dz * r)) r += CELDA;
      largos.push(r);
    }
    let mejorAng = 0, mejorArea = -1;
    const medio = Math.round(N * 35 / 360);
    const centro = mapa.caja.getCenter(new THREE.Vector3());
    const angCentro = Math.atan2(centro.z - z, centro.x - x);
    for (let k = 0; k < N; k++) {
      let area = 0;
      for (let d = -medio; d <= medio; d++) { const L = largos[(k + d + N) % N]; area += Math.min(L, 8) ** 2; }
      area *= 1 + 0.6 * Math.cos((k / N) * Math.PI * 2 - angCentro); // preferir mirar hacia el centro
      if (area > mejorArea) { mejorArea = area; mejorAng = (k / N) * Math.PI * 2; }
    }
    const lx = x + Math.cos(mejorAng) * 3, lz = z + Math.sin(mejorAng) * 3;
    return { pos: [x, piso + ALTURA_OJOS, z], mirar: [lx, piso + 1.2, lz] };
  });
}

// ---------- Luces por cuarto ----------
function ponerLuces() {
  exposicionBase = usandoHorneado && actual ? Math.pow(2, actual.exposicion ?? 0) : 1.0;
  aplicarExposicion();
  scene.remove(lucesCuarto);
  lucesCuarto.traverse((o) => o.dispose?.());
  lucesCuarto = new THREE.Group();
  const caja = mapa.caja, techo = caja.max.y - 0.35;
  if (actual?.exterior) {
    // Exterior: cielo y sol bajo de amanecer, sin focos de techo.
    scene.background = new THREE.Color(0x9fb8d6);
    hemi.color.set(0xdfe9ff); hemi.groundColor.set(0x4a5a2a); hemi.intensity = 1.0;
    sol.color.set(0xffd9a8); sol.intensity = 2.2;
    sol.position.set(80, 25, 45); sol.target.position.set(0, 0, 0); scene.add(sol.target);
    return;
  }
  scene.background = new THREE.Color(0x1b1d21);
  if (usandoHorneado) {
    // La luz ya viene en el modelo (Cycles). Solo un poco de luz para vidrios y lámparas sin hornear.
    hemi.color.set(0xffffff); hemi.groundColor.set(0x6b5d50); hemi.intensity = 0.6;
    sol.intensity = 0; scene.environmentIntensity = 0.35;
    return;
  }
  scene.environmentIntensity = 0.9;
  hemi.color.set(0xfff6ea); hemi.groundColor.set(0x6b5d50); hemi.intensity = 1.1;
  sol.color.set(0xfff1dd); sol.intensity = 0.9;
  // Luces puntuales suaves repartidas en una rejilla de ~3 m bajo el techo.
  const nx = Math.max(1, Math.round((caja.max.x - caja.min.x) / 3)), nz = Math.max(1, Math.round((caja.max.z - caja.min.z) / 3));
  for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) {
    const x = caja.min.x + ((i + 0.5) / nx) * (caja.max.x - caja.min.x);
    const z = caja.min.z + ((k + 0.5) / nz) * (caja.max.z - caja.min.z);
    const l = new THREE.PointLight(0xffe7c8, 4.5, 6.5, 1.6);
    l.position.set(x, techo, z);
    lucesCuarto.add(l);
  }
  scene.add(lucesCuarto);
  sol.position.set(caja.max.x + 4, caja.max.y + 6, caja.max.z + 4);
  sol.target.position.copy(caja.getCenter(new THREE.Vector3()));
  scene.add(sol.target);
}

// ---------- Luz horneada ----------
let usandoHorneado = false;
// Los materiales marcados "horneado" (extras del glTF) ya traen la luz de Cycles en su textura o en
// sus colores de vértice, guardada dividida entre `escala`. Se muestran sin luz en vivo.
function aplicarHorneado(raiz, escala) {
  raiz.traverse((o) => {
    if (!o.isMesh) return;
    const conv = (m) => {
      if (m?.userData?.luz_vertices && o.geometry.attributes.color) {
        // Textura original (con transparencia) × luz horneada en los vértices.
        const b = new THREE.MeshBasicMaterial({ map: m.map, vertexColors: true, side: THREE.DoubleSide, name: m.name,
          transparent: m.transparent, alphaTest: m.alphaTest || (m.transparent ? 0 : 0.5), opacity: m.opacity });
        if (!m.map) b.color.copy(m.color);
        b.color.multiplyScalar(escala);
        b.userData.horneado = 1; m.dispose();
        return b;
      }
      if (!m?.userData?.horneado) return m;
      const b = new THREE.MeshBasicMaterial({ map: m.map, vertexColors: m.vertexColors, side: m.side, name: m.name });
      b.color.setScalar(escala);
      if (m.userData.recorte) { b.alphaTest = 0.5; b.side = THREE.DoubleSide; } // cortinas, biombo…: recorte nítido
      if (b.map) b.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
      b.userData.horneado = 1;
      m.dispose();
      return b;
    };
    o.material = Array.isArray(o.material) ? o.material.map(conv) : conv(o.material);
  });
}
function aplicarExposicion() {
  renderer.toneMappingExposure = exposicionBase * parseFloat($('luz').value);
}
function actualizarBotonesLuz() {
  const b = $('btnHorneado');
  b.style.display = actual?.horneado ? '' : 'none';
  b.classList.toggle('activo', usandoHorneado);
  $('btnAO').classList.toggle('activo', aoEnUso());
}
// Con luz horneada la oclusión ya viene incluida; el botón la permite encima si se quiere.
let aoForzado = null;
const aoEnUso = () => (aoForzado ?? (aoActivo && !usandoHorneado));

// ---------- Render de esta vista ----------
// Manda la cámara actual al servidor local (servidor_visor.py), que renderiza con Blender/Cycles
// el .blend original del cuarto. La imagen se va limpiando: 8, 32 y 128 muestras (con denoiser).
let renderActivo = false, renderId = null, renderTimer = 0;
const renderImg = $('renderImg');
function textoRender(t, frac) {
  $('renderTexto').textContent = t;
  if (frac !== undefined) $('renderBarra').style.width = `${Math.round(frac * 100)}%`;
}
async function empezarRender() {
  if (renderActivo || vistaActual !== 'escena' || !actual) return;
  if (esTactil && !confirm(t('render.confirm'))) return;
  if (document.pointerLockElement) document.exitPointerLock();
  teclas.clear(); joy.x = joy.y = 0; subirTactil = 0; transicion = null;
  renderActivo = true; document.body.classList.add('renderizando');
  renderImg.classList.remove('activa'); renderImg.removeAttribute('src');
  $('btnDescargar').disabled = true;
  textoRender(t('render.opening'), 0.02);
  const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
  const p = camera.position, m = p.clone().addScaledVector(dir, 2);
  const esc = Math.min(1, 1600 / innerWidth);
  const cuerpo = { cuarto: actual.id, pos: p.toArray(), mirar: m.toArray(), fov: camera.fov,
    ancho: Math.round(innerWidth * esc), alto: Math.round(innerHeight * esc) };
  let r, d;
  try {
    r = await fetch('/api/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cuerpo) });
    d = await r.json().catch(() => ({}));
  } catch { r = null; }
  if (!renderActivo) return;
  if (!r || !r.ok || !d.id) {
    const msg = d?.error || t('render.needServer');
    textoRender(msg, 0); return;
  }
  renderId = d.id;
  const total = d.pasos?.length || 3;
  const revisar = async () => {
    if (!renderActivo || renderId !== d.id) return;
    let e = {};
    try { e = await (await fetch(`/api/render/${d.id}/estado`, { cache: 'no-store' })).json(); } catch { /* reintentar */ }
    if (!renderActivo || renderId !== d.id) return;
    if (e.url) {
      const url = `${e.url}?p=${e.paso}`;
      if (renderImg.dataset.url !== url) {
        renderImg.dataset.url = url;
        const img = new Image();
        img.onload = () => { if (renderId === d.id) { renderImg.src = url; renderImg.classList.add('activa'); } };
        img.src = url;
        const a = $('btnDescargar'); a.disabled = false; a.dataset.url = e.url;
      }
    }
    const listos = e.paso || 0;
    if (e.fase === 'listo') { textoRender(t('render.done', { m: e.muestras, s: e.segundos }), 1); return; }
    if (e.fase === 'error') { textoRender(t('render.failed'), listos / total); return; }
    if (e.fase === 'abriendo' || e.fase === 'preparando' || !e.fase) textoRender(t('render.loading'), 0.05);
    else textoRender(t('render.step', { m: e.muestras, i: listos + 1, n: total }), Math.max(0.08, listos / total));
    renderTimer = setTimeout(revisar, 700);
  };
  revisar();
}
function salirRender() {
  if (!renderActivo) return;
  renderActivo = false; renderId = null; clearTimeout(renderTimer);
  fetch('/api/render/cancelar', { method: 'POST', body: '{}' }).catch(() => {});
  document.body.classList.remove('renderizando'); renderImg.classList.remove('activa');
  lienzo.focus();
}
$('btnRender').onclick = () => empezarRender();
// El render usa Blender en esta PC: fuera de ella (por ejemplo en el sitio web) el botón no aparece.
$('secRender').style.display = 'none';  // the Blender render server is not part of this release
$('btnSalirRender').onclick = () => salirRender();
$('btnDescargar').onclick = () => {
  const url = $('btnDescargar').dataset.url; if (!url) return;
  const a = document.createElement('a'); a.href = url; a.download = `render_${actual.id}.png`; document.body.appendChild(a); a.click(); a.remove();
};
addEventListener('keydown', (e) => { if (renderActivo && e.code === 'Escape') salirRender(); });

// ---------- Ficha del objeto: doble clic (o doble toque) dice qué es y cuánto mide ----------
// cuartos/fichas/<id>.json trae cajas en coordenadas de Blender (Z arriba); glTF (x,y,z) = Blender (x,-z,y).
let fichas = null, fichasDe = null, marcoSel = null, fichaActual = null;
const rayo = new THREE.Raycaster();
const fichaEl = $('ficha');
async function cargarFichas(id) {
  fichas = null; fichasDe = null;
  try { const r = await fetch(`cuartos/fichas/${id}.json${VER}`); if (r.ok) { const d = await r.json(); if (actual?.id === id) { fichas = d.objetos; for (const o of fichas) o.caja0 = o.caja; } } } catch { /* sin fichas */ }
  if (actual?.id === id) fichasDe = id;
}
function cerrarFicha() {
  fichaEl.classList.remove('abierta'); document.body.classList.remove('con-ficha'); fichaActual = null; if (typeof gizmo !== 'undefined') gizmo.detach();
  if (marcoSel) { scene.remove(marcoSel); marcoSel.geometry.dispose(); marcoSel = null; }
  if (vistaActual === 'mapa') pedirMapa();
}
function buscarObjeto(nx, ny) {
  if (!modelo || !fichas) return null;
  rayo.setFromCamera(new THREE.Vector2(nx, ny), camera);
  const visible = (x) => { for (; x; x = x.parent) if (!x.visible) return false; return true; };
  const hit = rayo.intersectObject(modelo, true).find((h) => visible(h.object));
  if (!hit) return null;
  const p = hit.point, b = [p.x, -p.z, p.y], tol = 0.04;
  let mejor = null, puntos = Infinity;
  for (const o of fichas) {
    const { c, h, yaw: g } = o.caja, dx = b[0] - c[0], dy = b[1] - c[1], dz = b[2] - c[2];
    const lx = Math.cos(g) * dx + Math.sin(g) * dy, ly = -Math.sin(g) * dx + Math.cos(g) * dy;
    if (Math.abs(lx) > h[0] + tol || Math.abs(ly) > h[1] + tol || Math.abs(dz) > h[2] + tol) continue;
    // gana la caja más chica; paredes, piso y techo solo si no hay otra cosa
    const v = (h[0] + 0.02) * (h[1] + 0.02) * (h[2] + 0.02) + (o.tipo === 'estructura' ? 1e6 : 0);
    if (v < puntos) { puntos = v; mejor = o; }
  }
  return mejor;
}
const m2 = (v) => (v >= 1 ? `${v.toFixed(2)} m` : `${Math.round(v * 100)} cm`);
const escHtml = (t) => String(t).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
function mostrarFicha(o) {
  cerrarFicha();
  const { c, h, yaw: g } = o.caja;
  const caja = new THREE.BoxGeometry(2 * h[0] + 0.02, 2 * h[2] + 0.02, 2 * h[1] + 0.02);
  marcoSel = new THREE.LineSegments(new THREE.EdgesGeometry(caja), new THREE.LineBasicMaterial({ color: 0xf2b134, depthTest: false, transparent: true }));
  caja.dispose();
  marcoSel.position.set(c[0], c[2], -c[1]); marcoSel.rotation.y = g; marcoSel.renderOrder = 999;
  scene.add(marcoSel); fichaActual = o; seguirMarco(o);
  const md = o.medidas, plano = md.fondo < 0.005 || md.alto < 0.005;
  const filas = [[t('card.length'), m2(md.ancho)]];
  if (md.fondo >= 0.005) filas.push([t('card.depth'), m2(md.fondo)]);
  if (md.alto >= 0.005) filas.push([t('card.height'), m2(md.alto)]);
  if (md.sobrePiso > 0.03) filas.push([t('card.aboveFloor'), m2(md.sobrePiso)]);
  fichaEl.innerHTML = `<div class="cab"><h3>${escHtml(loc(o, 'nombre'))}${o.num ? ` <span class="num">${t('card.numOf', { num: o.num, total: o.cantidad })}</span>` : ''}${o.cambio ? ` <span class="num">${escHtml(t('card.insteadOf', { name: loc(o.base, 'nombre') }))}</span>` : ''}</h3>
    <button class="cerrar" aria-label="${t('close')}">✕</button></div>
    ${vistaActual === 'mapa' ? `<button class="ver3d">${t('card.view3d')}</button>` : ''}
    ${o.img ? `<img src="${escHtml(o.img)}" alt="">` : ''}
    ${loc(o, 'descripcion') ? `<p>${escHtml(loc(o, 'descripcion'))}</p>` : ''}
    <table>${filas.map(([a, v]) => `<tr><td>${a}</td><td>${v}</td></tr>`).join('')}</table>
    ${o.cantidad > 1 && !o.cambio ? `<span class="etq">${t('card.count', { n: o.cantidad })}</span> ` : ''}${o.cambio ? `<span class="etq">${escHtml(t('card.modelFrom', { room: loc(o.cambio.cat, 'cuartoNombre') }))}</span> ` : ''}${o.origen ? `<span class="etq">${escHtml(loc(o, 'origen'))}</span>` : ''}
    <div class="nota">${t('card.note')}${plano ? t('card.flat') : ''}.</div>
    ${htmlEditar(o)}
    ${htmlCambio(o)}`;
  fichaEl.querySelector('.cerrar').addEventListener('click', cerrarFicha);
  fichaEl.querySelector('.ver3d')?.addEventListener('click', () => abrirDesdeMapa(o));
  conectarEditar(o);
  conectarCambio(o);
  fichaEl.classList.add('abierta'); document.body.classList.add('con-ficha');
}

// ---------- Cambiar un objeto por otro parecido (bancos por sillas, etc.) y moverlo ----------
// Solo con "Luz Cycles": esos modelos traen cada mueble por separado (obj_*), así se puede ocultar o mover uno.
// Cada objeto editable cuelga de un "soporte" puesto en su lugar del piso y girado como el original:
// dentro van el original y, si se cambió, el reemplazo. Mover o girar el soporte mueve lo que se vea.
let catalogo = null; const cacheModelos = new Map();
// catalogo.json: muebles de los cuartos; catalogo_galeria.json: muebles de la galería de objetos (Meshy)
Promise.all(['cuartos/fichas/catalogo.json', 'cuartos/fichas/catalogo_galeria.json'].map((u) => fetch(u + VER).then((r) => (r.ok ? r.json() : null)).catch(() => null)))
  .then((ds) => { catalogo = ds.flatMap((d) => d?.modelos || []); });
const editable = (o) => !!(o.colocacion && usandoHorneado && modelo?.getObjectByName(o.nodo));
function soporte(o) {
  if (o.soporte) return o.soporte;
  const nodo = modelo?.getObjectByName(o.nodo); if (!nodo) return null;
  // el pivote va al centro del objeto (ahí salen las flechas y los aros)
  const b = o.colocacion, c = o.caja, sp = new THREE.Group(); sp.name = 'soporte_' + o.nodo;
  sp.position.set(c.c[0], c.c[2], -c.c[1]); sp.rotation.y = b.giro; o.piv = c.c[2] - b.base;
  modelo.add(sp); sp.updateMatrixWorld(true); sp.attach(nodo);
  o.rel0 = { c: sp.worldToLocal(new THREE.Vector3(c.c[0], c.c[2], -c.c[1])), h: c.h.slice(), yaw: c.yaw - b.giro };
  o.soporte = sp; o.nodoObj = nodo; o.inicio = { p: sp.position.clone(), q: sp.quaternion.clone() };
  return sp;
}
function seguirMarco(o) {
  if (!o.soporte) return;
  const sp = o.soporte, r = o.cambio ? o.cambio.rel : o.rel0; sp.updateMatrixWorld(true);
  const w = sp.localToWorld(r.c.clone());
  o.caja = { c: [w.x, -w.z, w.y], h: r.h, yaw: sp.rotation.y + r.yaw };
  o.medidas = { ...o.medidas, sobrePiso: Math.max(0, w.y - r.h[2]) };
  if (marcoSel && fichaActual === o) {
    marcoSel.position.copy(w);
    marcoSel.quaternion.copy(sp.quaternion).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), r.yaw));
  }
  if (vistaActual === 'mapa') pedirMapa();
}
function candidatos(o) {
  if (!o.colocacion || !catalogo) return [];
  return catalogo.filter((m) => m.familia === o.familia && !(m.cuarto === actual.id && m.clase === o.clase));
}
function htmlCambio(o) {
  if (!o.colocacion) return '';
  if (!usandoHorneado) return `<div class="cambio"><div class="nota">${t('swap.needCycles')}</div></div>`;
  const lista = candidatos(o); if (!lista.length && !o.cambio) return '';
  const iguales = fichas.filter((f) => f.clase === o.clase && f.colocacion).length;
  const opciones = (o.cambio ? [{ ...o.base, original: true }] : []).concat(lista);
  return `<div class="cambio"><b>${t('swap.title')}</b>
    ${iguales > 1 ? `<label class="todos"><input type="checkbox" id="cambiarTodos"> ${t('swap.all', { n: iguales })}</label>` : ''}
    ${grupoOpciones(opciones.filter((m) => m.cuarto !== 'galeria'), lista, '')}${grupoOpciones(opciones.filter((m) => m.cuarto === 'galeria'), lista, t('swap.gallery'))}
    <div class="nota">${t('swap.note')}</div></div>`;
}
function grupoOpciones(opciones, lista, titulo) {
  if (!opciones.length) return '';
  return `${titulo ? `<div class="subt">${titulo} (${opciones.length})</div>` : ''}<div class="opciones">${opciones.map((m) => `<button data-i="${m.original ? -1 : lista.indexOf(m)}" title="${escHtml(loc(m, 'nombre'))}">
      ${m.img ? `<img src="${escHtml(m.img)}" alt="" loading="lazy">` : '<span class="sinimg">?</span>'}<span>${escHtml(m.original ? t('swap.original') : loc(m, 'nombre'))}</span>
      ${m.original || m.cuarto === 'galeria' ? '' : `<small>${escHtml(loc(m, 'cuartoNombre') || '')}</small>`}</button>`).join('')}</div>`;
}
function conectarCambio(o) {
  const lista = candidatos(o), todos = () => fichaEl.querySelector('#cambiarTodos')?.checked;
  const objetivo = () => (todos() ? fichas.filter((f) => f.clase === o.clase && f.colocacion) : [o]);
  fichaEl.querySelectorAll('.opciones button').forEach((b) => b.addEventListener('click', async () => {
    const i = parseInt(b.dataset.i, 10);
    if (i < 0) { objetivo().forEach(restaurar); mostrarFicha(o); return; }
    b.disabled = true; b.classList.add('cargando');
    try { for (const f of objetivo()) await cambiarObjeto(f, lista[i]); } catch (err) { console.warn(err); aviso(t('swap.loadFailed')); }
    mostrarFicha(o);
  }));
}
async function modeloCatalogo(m) {
  if (!cacheModelos.has(m.archivo)) cacheModelos.set(m.archivo, loader.loadAsync(m.archivo));
  return (await cacheModelos.get(m.archivo)).scene;
}
async function cambiarObjeto(o, m) {
  const sp = soporte(o);
  if (!sp) { aviso(t('swap.cantSeparate')); return; }
  const fuente = await modeloCatalogo(m);
  if (o.cambio) { o.cambio.grupo.removeFromParent(); }
  const obj = fuente.clone(true);
  const bb = new THREE.Box3().setFromObject(obj), tam = bb.getSize(new THREE.Vector3()), cen = bb.getCenter(new THREE.Vector3());
  const k = (m.alto || tam.y) / tam.y;
  obj.position.set(-cen.x, -bb.min.y, -cen.z);
  const grupo = new THREE.Group(); grupo.scale.setScalar(k); grupo.position.y = -o.piv; grupo.add(obj); sp.add(grupo);
  // con la luz horneada el ambiente está bajo: se compensa para que el modelo no se vea oscuro
  obj.traverse((x) => { if (x.isMesh) for (const mt of [].concat(x.material)) if ('envMapIntensity' in mt) mt.envMapIntensity = 2.2; });
  o.nodoObj.visible = false;
  if (!o.base) o.base = { nombre: o.nombre, nombre_en: o.nombre_en, descripcion: o.descripcion, descripcion_en: o.descripcion_en, img: o.img, medidas: o.medidas, origen: o.origen, origen_en: o.origen_en };
  const w = tam.x * k, d = tam.z * k, h = tam.y * k;
  o.cambio = { cat: m, grupo, rel: { c: new THREE.Vector3(0, h / 2 - o.piv, 0), h: [w / 2, d / 2, h / 2], yaw: 0 } };
  Object.assign(o, { nombre: m.nombre, nombre_en: m.nombre_en, descripcion: m.descripcion, descripcion_en: m.descripcion_en, img: m.img, origen: undefined, origen_en: undefined,
    medidas: { ancho: Math.max(w, d), fondo: Math.min(w, d), alto: h, sobrePiso: 0 } });
  seguirMarco(o);
}
function restaurar(o) {
  if (!o.cambio) return;
  o.cambio.grupo.removeFromParent(); o.nodoObj.visible = true;
  Object.assign(o, o.base); delete o.cambio; seguirMarco(o);
}

// ---------- Mover y girar: flechas/aros sobre el objeto, barritas, pegado al piso, deshacer ----------
const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setSize(0.85); gizmo.setSpace('local');
scene.add(gizmo.getHelper());
let pegado = true, deslizando = false;
const historia = [], rehacerPila = [];
const estadoDe = (o) => ({ o, p: o.soporte.position.clone(), q: o.soporte.quaternion.clone() });
function guardarHistoria(o) { if (!o?.soporte) return; historia.push(estadoDe(o)); if (historia.length > 200) historia.shift(); rehacerPila.length = 0; actualizarBotonesHistoria(); }
function aplicarEstado(e) { e.o.soporte.position.copy(e.p); e.o.soporte.quaternion.copy(e.q); seguirMarco(e.o); if (fichaActual === e.o) refrescarEditar(e.o); }
function deshacer() { const e = historia.pop(); if (!e) { aviso(t('undo.none')); return; } rehacerPila.push(estadoDe(e.o)); aplicarEstado(e); actualizarBotonesHistoria(); }
function rehacer() { const e = rehacerPila.pop(); if (!e) { aviso(t('redo.none')); return; } historia.push(estadoDe(e.o)); aplicarEstado(e); actualizarBotonesHistoria(); }
function actualizarBotonesHistoria() {
  const d = fichaEl.querySelector('[data-acc="deshacer"]'), r = fichaEl.querySelector('[data-acc="rehacer"]');
  if (d) d.disabled = !historia.length; if (r) r.disabled = !rehacerPila.length;
}
function ajustarPiso(o) { // pegado al piso: a la altura original y sin inclinar
  const sp = o.soporte; sp.position.y = o.inicio.p.y;
  const e = new THREE.Euler().setFromQuaternion(sp.quaternion, 'YXZ'); sp.quaternion.setFromEuler(new THREE.Euler(0, e.y, 0, 'YXZ'));
}
function ponerEjes() {
  const rot = gizmo.mode === 'rotate';
  gizmo.showX = rot ? !pegado : true; gizmo.showZ = rot ? !pegado : true; gizmo.showY = rot ? true : !pegado;
}
gizmo.addEventListener('mouseDown', () => guardarHistoria(fichaActual));
gizmo.addEventListener('objectChange', () => {
  const o = fichaActual; if (!o?.soporte) return;
  if (pegado) ajustarPiso(o);
  seguirMarco(o); refrescarEditar(o);
});
function htmlEditar(o) {
  if (!editable(o)) return '';
  return `<div class="editar"><b>${t('edit.title')}</b>
    <div class="fila-editar"><div class="segmentos"><button data-modo="translate" title="${t('edit.move.title')}">${t('edit.move')}</button><button data-modo="rotate" title="${t('edit.rotate.title')}">${t('edit.rotate')}</button></div>
      <label class="todos"><input type="checkbox" id="pegado" ${pegado ? 'checked' : ''}> ${t('edit.snap')}</label></div>
    <label class="barrita">${t('edit.turn')} <input type="range" id="rGiro" min="-180" max="180" step="1"><span id="vGiro"></span></label>
    <label class="barrita">${t('edit.height')} <input type="range" id="rAltura" min="-0.5" max="2.5" step="0.01"><span id="vAltura"></span></label>
    <div class="acciones"><button data-acc="deshacer" title="Ctrl+Z">${t('edit.undo')}</button><button data-acc="rehacer" title="Ctrl+Y">${t('edit.redo')}</button><button data-acc="inicio" title="${t('edit.reset.title')}">${t('edit.reset')}</button></div>
    <div class="nota">${vistaActual === 'mapa' ? t('edit.noteMap') : t('edit.noteScene')} ${t('edit.keys')}</div></div>`;
}
function refrescarEditar(o) {
  const g = fichaEl.querySelector('#rGiro'); if (!g || !o.soporte) return;
  const e = new THREE.Euler().setFromQuaternion(o.soporte.quaternion, 'YXZ');
  let deg = Math.round(THREE.MathUtils.radToDeg(e.y - o.colocacion.giro)); deg = ((deg + 540) % 360) - 180;
  const alt = o.soporte.position.y - o.inicio.p.y;
  if (!deslizando) { g.value = deg; fichaEl.querySelector('#rAltura').value = alt.toFixed(2); }
  fichaEl.querySelector('#vGiro').textContent = `${deg}°`;
  fichaEl.querySelector('#vAltura').textContent = `${alt >= 0 ? '+' : ''}${Math.round(alt * 100)} cm`;
  fichaEl.querySelector('#rAltura').disabled = pegado;
}
function conectarEditar(o) {
  if (!fichaEl.querySelector('.editar')) return;
  soporte(o);
  const modos = fichaEl.querySelectorAll('[data-modo]');
  const marcar = () => modos.forEach((b) => b.classList.toggle('activo', gizmo.object === o.soporte && gizmo.mode === b.dataset.modo));
  modos.forEach((b) => b.addEventListener('click', () => {
    if (gizmo.object === o.soporte && gizmo.mode === b.dataset.modo) gizmo.detach();
    else { if (document.pointerLockElement) document.exitPointerLock(); gizmo.attach(o.soporte); gizmo.setMode(b.dataset.modo); ponerEjes(); }
    marcar();
  }));
  fichaEl.querySelector('#pegado').addEventListener('change', (e) => {
    pegado = e.target.checked; ponerEjes();
    if (pegado) { guardarHistoria(o); ajustarPiso(o); seguirMarco(o); }
    refrescarEditar(o);
  });
  const g = fichaEl.querySelector('#rGiro'), al = fichaEl.querySelector('#rAltura');
  const empezar = () => { if (!deslizando) { guardarHistoria(o); deslizando = true; } };
  const terminar = () => { deslizando = false; refrescarEditar(o); };
  g.addEventListener('input', () => {
    empezar();
    const e = new THREE.Euler().setFromQuaternion(o.soporte.quaternion, 'YXZ');
    e.y = o.colocacion.giro + THREE.MathUtils.degToRad(parseFloat(g.value));
    o.soporte.quaternion.setFromEuler(e); seguirMarco(o); refrescarEditar(o);
  });
  al.addEventListener('input', () => { empezar(); o.soporte.position.y = o.inicio.p.y + parseFloat(al.value); seguirMarco(o); refrescarEditar(o); });
  for (const r of [g, al]) r.addEventListener('change', terminar);
  fichaEl.querySelector('[data-acc="deshacer"]').addEventListener('click', deshacer);
  fichaEl.querySelector('[data-acc="rehacer"]').addEventListener('click', rehacer);
  fichaEl.querySelector('[data-acc="inicio"]').addEventListener('click', () => {
    guardarHistoria(o); o.soporte.position.copy(o.inicio.p); o.soporte.quaternion.copy(o.inicio.q); seguirMarco(o); refrescarEditar(o);
  });
  marcar(); refrescarEditar(o); actualizarBotonesHistoria();
}
addEventListener('keydown', (e) => {
  if (!(e.ctrlKey || e.metaKey) || e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  if (e.code === 'KeyZ') { e.preventDefault(); if (e.shiftKey) rehacer(); else deshacer(); }
  else if (e.code === 'KeyY') { e.preventDefault(); rehacer(); }
});
function elegirEn(nx, ny) {
  if (vistaActual !== 'escena' || renderActivo || !modelo) return;
  if (!fichas) { aviso(t('pick.noCards')); return; }
  const o = buscarObjeto(nx, ny);
  if (o) mostrarFicha(o); else { cerrarFicha(); aviso(t('pick.none')); }
}
let bloqueoDesde = 0;
document.addEventListener('pointerlockchange', () => { if (document.pointerLockElement === lienzo) bloqueoDesde = performance.now(); });
lienzo.addEventListener('dblclick', (e) => {
  if (esTactil) return;
  // con el mouse capturado se apunta con la mira (centro); si la captura acaba de empezar, donde se hizo clic
  const centro = document.pointerLockElement === lienzo && performance.now() - bloqueoDesde > 600;
  if (document.pointerLockElement) document.exitPointerLock();
  if (centro) elegirEn(0, 0);
  else elegirEn((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
});
// doble toque: dos toques cortos y quietos en el mismo lugar
const toques = new Map(); let ultimoToque = null;
lienzo.addEventListener('touchstart', (e) => { for (const t of e.changedTouches) toques.set(t.identifier, { x: t.clientX, y: t.clientY, t: performance.now(), mov: 0 }); }, { passive: true });
lienzo.addEventListener('touchmove', (e) => { for (const t of e.changedTouches) { const q = toques.get(t.identifier); if (q) q.mov = Math.max(q.mov, Math.hypot(t.clientX - q.x, t.clientY - q.y)); } }, { passive: true });
lienzo.addEventListener('touchend', (e) => {
  for (const t of e.changedTouches) {
    const q = toques.get(t.identifier); toques.delete(t.identifier);
    if (!q || q.mov > 12 || performance.now() - q.t > 300) { ultimoToque = null; continue; }
    const ahora = performance.now();
    if (ultimoToque && ahora - ultimoToque.t < 400 && Math.hypot(q.x - ultimoToque.x, q.y - ultimoToque.y) < 40) {
      ultimoToque = null; elegirEn((q.x / innerWidth) * 2 - 1, -(q.y / innerHeight) * 2 + 1);
    } else ultimoToque = { x: q.x, y: q.y, t: ahora };
  }
});
addEventListener('keydown', (e) => { if (e.code === 'Escape' && fichaEl.classList.contains('abierta')) cerrarFicha(); });

// ---------- Cargar cuarto ----------
function liberar(obj) {
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) for (const m of [].concat(o.material)) {
      for (const k in m) if (m[k]?.isTexture) m[k].dispose();
      m.dispose();
    }
  });
}

async function cargarCuarto(c) {
  actual = c;
  cerrarFicha(); cargarFichas(c.id); cacheModelos.clear(); historia.length = 0; rehacerPila.length = 0;
  selCuarto.value = c.id;
  escenaDe = c.id;
  CELDA = c.exterior ? 0.25 : 0.05;
  escalaVel = c.exterior ? 6 : 1;
  if (vistaActual === 'escena') { camera.near = 0.05; camera.far = c.exterior ? 1500 : 300; camera.updateProjectionMatrix(); }
  carga.classList.remove('oculta');
  cargaTitulo.textContent = loc(c, 'nombre');
  progreso(0, t('load.downloading'));
  if (modelo) { scene.remove(modelo); liberar(modelo); modelo = null; }
  let gltf;
  usandoHorneado = !!(luzHorneada && c.horneado);
  const archivo = usandoHorneado ? c.horneado : c.archivo;
  actualizarBotonesLuz();
  try {
    gltf = await loader.loadAsync(archivo, (e) => {
      if (e.total) progreso(0.85 * (e.loaded / e.total), t('load.downloadingMB', { a: (e.loaded / 1048576).toFixed(1), b: (e.total / 1048576).toFixed(1) }));
    });
  } catch (err) {
    progreso(0, t('load.failed', { file: archivo }));
    console.warn('Error al cargar', archivo, err);
    return;
  }
  modelo = gltf.scene;
  // Exposición de la escena de Blender (AgX) para que se vea como sus renders.
  exposicionBase = usandoHorneado ? Math.pow(2, c.exposicion ?? 0) : 1.0;
  aplicarExposicion();
  if (usandoHorneado) aplicarHorneado(modelo, c.escalaLuz ?? 2);
  modelo.traverse((o) => {
    if (o.isMesh) {
      for (const m of [].concat(o.material)) {
        if (m.map) m.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
        if ('envMapIntensity' in m) m.envMapIntensity = 1;
      }
    }
  });
  scene.add(modelo);
  modelo.visible = vistaActual === 'escena';
  progreso(0.9, t('load.walls'));
  await pausa();
  mapa = await construirMapa(modelo);
  pisoY = mapa.piso;
  ponerVistas((c.vistas && c.vistas.length) ? c.vistas : vistasAutomaticas(4));
  if (vistaActual === 'escena') { ponerLuces(); irAVista(0, true); }
  else { // se cambió de pestaña mientras cargaba: dejar la escena lista para cuando se vuelva
    const p = vistas[0]; if (p) { const a = anguloDesde(p.pos, p.mirar); camEscena = { pos: new THREE.Vector3(...p.pos), yaw: a.yaw, pitch: a.pitch }; }
  }
  progreso(1, '');
  // compilar shaders antes de quitar la carga para que no "trabe" el primer cuadro
  await renderer.compileAsync?.(scene, camera);
  carga.classList.add('oculta');
  guardarUrl();
}

// ---------- Proyecto y pestañas: Todo junto / Modelos / Mapa ----------
let vistaActual = 'escena', escenaDe = null, escalaVel = 1, camEscena = null;
let sueltos = [], iSuelto = -1, sueltoObj = null, cargaSuelto = 0;
const orbita = { yaw: 0.6, pitch: 0.35, dist: 3, centro: new THREE.Vector3() };
let iMapa = 0;

function guardarUrl() {
  const url = new URL(location.href);
  url.searchParams.set('cuarto', actual.id); url.searchParams.set('vista', vistaActual);
  if (vistaActual === 'modelos' && iSuelto >= 0) url.searchParams.set('modelo', String(iSuelto + 1)); else url.searchParams.delete('modelo');
  history.replaceState(null, '', url);
}

function vacio(txt) { const v = $('vacio'); v.textContent = txt || ''; v.style.display = txt ? 'block' : 'none'; }

function elegirProyecto(c, vista = vistaActual) {
  actual = c; selCuarto.value = c.id;
  if (sueltoObj) { scene.remove(sueltoObj); liberar(sueltoObj); sueltoObj = null; }
  sueltos = c.modelos || []; iSuelto = -1; iMapa = 0;
  dibujarTiras(); dibujarMapas();
  if (modelo && escenaDe !== c.id) { scene.remove(modelo); liberar(modelo); modelo = null; mapa = null; escenaDe = null; }
  ponerVista(vista);
}

function ponerVista(v) {
  vistaActual = v;
  const abierta = fichaActual; if (v === 'modelos') cerrarFicha();
  for (const b of document.querySelectorAll('#pestanas button')) b.classList.toggle('activo', b.dataset.vista === v);
  document.body.classList.remove('v-escena', 'v-modelos', 'v-mapa'); document.body.classList.add('v-' + v);
  if (document.pointerLockElement) document.exitPointerLock();
  if (v !== 'escena' && escenaDe === actual.id && !camEscena) camEscena = { pos: camera.position.clone(), yaw, pitch };
  vacio('');
  if (modelo) modelo.visible = v === 'escena';
  if (sueltoObj) sueltoObj.visible = v === 'modelos';
  if (v === 'escena') {
    camera.near = 0.05; camera.far = actual.exterior ? 1500 : 300; camera.updateProjectionMatrix();
    if (escenaDe !== actual.id) { camEscena = null; cargarCuarto(actual); return; }
    ponerLuces();
    if (camEscena) { camera.position.copy(camEscena.pos); yaw = camEscena.yaw; pitch = camEscena.pitch; camEscena = null; }
    if (abierta && fichaActual === abierta) mostrarFicha(abierta);
  } else if (v === 'modelos') {
    luzEstudio();
    if (!sueltos.length) vacio(t('empty.models'));
    else if (iSuelto < 0) verSuelto(0);
  } else if (v === 'mapa') {
    if (!(actual.mapas || []).length) vacio(t('empty.map'));
    else verMapa(iMapa);
    if (abierta && fichaActual === abierta) mostrarFicha(abierta);
  }
  guardarUrl();
}

function luzEstudio() {
  exposicionBase = 1.0; aplicarExposicion(); scene.environmentIntensity = 0.9;
  scene.background = new THREE.Color(0x2a2d33);
  hemi.color.set(0xffffff); hemi.groundColor.set(0x55504a); hemi.intensity = 1.3;
  sol.color.set(0xfff4e6); sol.intensity = 1.6; scene.add(sol.target);
}

function dibujarTiras() {
  const t = $('tiras'); t.innerHTML = '';
  sueltos.forEach((m, i) => {
    const b = document.createElement('button'); b.title = loc(m, 'nombre');
    b.innerHTML = (m.miniatura ? `<img loading="lazy" alt="">` : '<div class="sinmini">◆</div>') + '<span></span>';
    if (m.miniatura) b.querySelector('img').src = m.miniatura;
    b.querySelector('span').textContent = loc(m, 'nombre');
    b.onclick = () => verSuelto(i);
    t.appendChild(b);
  });
}

async function verSuelto(i) {
  if (!sueltos.length) return;
  i = (i + sueltos.length) % sueltos.length; iSuelto = i;
  const m = sueltos[i], yo = ++cargaSuelto;
  $('modeloNombre').textContent = loc(m, 'nombre'); $('modeloCuenta').textContent = `${i + 1} / ${sueltos.length}`;
  const botones = $('tiras').children;
  for (let k = 0; k < botones.length; k++) botones[k].classList.toggle('activo', k === i);
  botones[i]?.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
  guardarUrl();
  let gltf;
  try { gltf = await loader.loadAsync(m.archivo); } catch (err) { aviso(t('load.fileFailed', { file: m.archivo })); console.warn(err); return; }
  if (yo !== cargaSuelto) { liberar(gltf.scene); return; } // ya se pidió otro
  if (sueltoObj) { scene.remove(sueltoObj); liberar(sueltoObj); }
  sueltoObj = gltf.scene; sueltoObj.visible = vistaActual === 'modelos';
  scene.add(sueltoObj);
  const caja = new THREE.Box3().setFromObject(sueltoObj), esfera = caja.getBoundingSphere(new THREE.Sphere());
  orbita.centro.copy(esfera.center); orbita.radio = esfera.radius || 1;
  orbita.dist = orbita.radio / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.15;
  orbita.yaw = 0.6; orbita.pitch = 0.3;
  camera.near = orbita.radio / 100; camera.far = orbita.radio * 100; camera.updateProjectionMatrix();
}

// ---------- Mapa: imágenes originales + dos mapas que el visor dibuja en el momento ----------
// "Planta 3D (en vivo)": el mismo modelo 3D visto desde arriba, sin techo (cámara ortográfica bajo el techo).
// "Plano de objetos (en vivo)": dibujado desde las fichas, con lo que se haya movido o cambiado.
// Los dos comparten coordenadas: x/y de Blender, 100 unidades por metro.
const esPiso = (o) => /^(floor|suelo|pasillo_suelo)$/.test(o.clase);
const esTecho = (o) => /^(ceiling|techo)$/.test(o.clase);
const esVentana = (o) => /^(window|persiana|vidrio)/.test(o.clase);
const esPuerta = (o) => /^(door|puerta)/.test(o.clase);
function listaMapas() {
  const orig = (actual.mapas || []).map((m) => ({ ...m, nombre: loc(m, 'nombre') }));
  if (!(fichas && fichasDe === actual.id)) return orig;
  for (const m of orig) m.nombre += t('map.origPhoto');
  return [{ nombre: t('map.liveFloor'), vivo: 'planta' }, { nombre: t('map.livePlan'), vivo: 'plano' }, ...orig];
}
function dibujarMapas() {
  const d = $('mapaBotones'); d.innerHTML = '';
  const ms = listaMapas();
  if (ms.length < 2) { if (ms[0]) d.textContent = ms[0].nombre; return; }
  ms.forEach((m, i) => {
    const b = document.createElement('button'); b.textContent = m.nombre; b.onclick = () => verMapa(i);
    b.classList.toggle('activo', i === iMapa); d.appendChild(b);
  });
}
async function verMapa(i) {
  if (fichasDe !== actual.id) { await cargarFichas(actual.id); if (vistaActual !== 'mapa') return; }
  const ms = listaMapas(); if (!ms.length) return;
  iMapa = Math.max(0, Math.min(i, ms.length - 1)); const m = ms[iMapa];
  dibujarMapas(); $('mapaMarco').classList.remove('zoom');
  const img = $('mapaImg'), sv = $('mapaSvg');
  if (!m.vivo) {
    tipoMapa = null; sv.innerHTML = ''; sv.style.display = 'none'; img.style.display = ''; img.src = m.archivo;
    $('mapaNota').textContent = t('map.tapZoom');
    return;
  }
  img.style.display = 'none'; img.removeAttribute('src'); sv.style.display = '';
  let tipo = m.vivo, foto = null; const L = limitesPlano();
  if (tipo === 'planta') { foto = renderPlanta(L); if (!foto) tipo = 'plano'; }
  tipoMapa = tipo; fotoPlanta = foto; limMapa = L;
  redibujarMapa();
  $('mapaNota').innerHTML = (m.vivo === 'planta' && !foto ? t('map.openSceneFirst') : '')
    + t('map.legend');
  ajustarSvg(); requestAnimationFrame(ajustarSvg);
}
function esquinas(caja, extra = 0) {
  const { c, h, yaw: g } = caja, co = Math.cos(g), si = Math.sin(g), a = h[0] + extra, b = h[1] + extra;
  return [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([p, q]) => [c[0] + co * p * a - si * q * b, c[1] + si * p * a + co * q * b]);
}
function mueblesMapa() {
  return fichas.map((o, i) => ({ o, i }))
    .filter(({ o }) => o.tipo === 'mueble' && !esTecho(o) && o.medidas.sobrePiso < 1.6 && 4 * o.caja.h[0] * o.caja.h[1] > 0.004)
    .sort((a, b) => (a.o.caja.c[2] + a.o.caja.h[2]) - (b.o.caja.c[2] + b.o.caja.h[2]));
}
function limitesPlano() {
  for (const o of fichas) if (o.soporte) seguirMarco(o); // cajas al día con lo que se haya movido
  let base = fichas.filter((o) => o.tipo === 'estructura' && !esTecho(o));
  if (!base.length) base = fichas;
  const L = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  for (const o of base.concat(mueblesMapa().map((m) => m.o))) for (const [x, y] of esquinas(o.caja)) {
    L.x0 = Math.min(L.x0, x); L.x1 = Math.max(L.x1, x); L.y0 = Math.min(L.y0, y); L.y1 = Math.max(L.y1, y);
  }
  const m = 0.3; L.x0 -= m; L.y0 -= m; L.x1 += m; L.y1 += m;
  return L;
}
const movido = (o) => !!o.cambio || !!(o.soporte && (o.soporte.position.distanceTo(o.inicio.p) > 0.01 || o.soporte.quaternion.angleTo(o.inicio.q) > 0.01));
function frente(o) { // hacia dónde mira (x/y de Blender); los modelos miran hacia -Y de Blender
  if (o.soporte) { const v = new THREE.Vector3(0, 0, 1).applyQuaternion(o.soporte.getWorldQuaternion(new THREE.Quaternion())); const n = Math.hypot(v.x, v.z) || 1; return [v.x / n, -v.z / n]; }
  const g = o.colocacion.giro; return [Math.sin(g), -Math.cos(g)];
}
function dondeEstoy() { // cámara del recorrido: posición y hacia dónde mira (x/y de Blender)
  if (escenaDe !== actual.id) return null;
  const p = camEscena ? camEscena.pos : camera.position, y = camEscena ? camEscena.yaw : yaw;
  return { x: p.x, y: -p.z, dx: -Math.sin(y), dy: Math.cos(y) };
}
// Foto de arriba: cámara ortográfica justo bajo el techo mirando al piso (lo de arriba queda fuera).
function renderPlanta(L) {
  if (!modelo || escenaDe !== actual.id) return null;
  const techo = fichas.find(esTecho);
  let alto = techo ? techo.caja.c[2] - techo.caja.h[2] - 0.04 : 0;
  if (!techo) for (const o of fichas) alto = Math.max(alto, o.caja.c[2] + o.caja.h[2] + 0.5);
  const ancho = L.x1 - L.x0, fondo = L.y1 - L.y0, lado = esTactil ? 1200 : 1800, k = Math.min(lado / ancho, lado / fondo);
  const w = Math.round(ancho * k), h = Math.round(fondo * k);
  const cam = new THREE.OrthographicCamera(-ancho / 2, ancho / 2, fondo / 2, -fondo / 2, 0.01, alto + 10);
  cam.position.set((L.x0 + L.x1) / 2, alto, -(L.y0 + L.y1) / 2); cam.up.set(0, 0, -1);
  cam.lookAt(cam.position.x, -1, cam.position.z); cam.updateMatrixWorld(true);
  const ayudante = gizmo.getHelper();
  const antes = [[modelo, modelo.visible], [ayudante, ayudante.visible]].concat(sueltoObj ? [[sueltoObj, sueltoObj.visible]] : [], marcoSel ? [[marcoSel, marcoSel.visible]] : []);
  modelo.visible = true; ayudante.visible = false; if (sueltoObj) sueltoObj.visible = false; if (marcoSel) marcoSel.visible = false;
  const fondoAntes = scene.background, pr = renderer.getPixelRatio();
  scene.background = new THREE.Color(0xffffff);
  let url = null;
  try {
    renderer.setPixelRatio(1); renderer.setSize(w, h, false);
    renderer.render(scene, cam);
    url = lienzo.toDataURL('image/jpeg', 0.9);
  } catch (err) { console.warn(err); }
  scene.background = fondoAntes;
  renderer.setPixelRatio(pr); renderer.setSize(innerWidth, innerHeight, false);
  for (const [x, v] of antes) x.visible = v;
  return url;
}
function svgMapa(tipo, foto, L = limitesPlano()) {
  const S = 100, W = Math.round((L.x1 - L.x0) * S), H = Math.round((L.y1 - L.y0) * S);
  const X = (x) => ((x - L.x0) * S).toFixed(1), Y = (y) => ((L.y1 - y) * S).toFixed(1);
  const pts = (cs) => cs.map(([x, y]) => `${X(x)},${Y(y)}`).join(' ');
  const t = [];
  if (foto) t.push(`<image href="${foto}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="none"/>`);
  else {
    t.push(`<rect width="${W}" height="${H}" fill="#fff"/>`);
    for (let x = Math.ceil(L.x0); x <= L.x1; x++) t.push(`<line x1="${X(x)}" y1="0" x2="${X(x)}" y2="${H}" class="rej"/>`);
    for (let y = Math.ceil(L.y0); y <= L.y1; y++) t.push(`<line x1="0" y1="${Y(y)}" x2="${W}" y2="${Y(y)}" class="rej"/>`);
    for (const o of fichas) if (esPiso(o)) t.push(`<polygon points="${pts(esquinas(o.caja))}" class="piso"/>`);
    for (const o of fichas) if (o.tipo === 'estructura' && !esPiso(o) && !esTecho(o)) t.push(`<polygon points="${pts(esquinas(o.caja, 0.01))}" class="pared"/>`);
  }
  const muebles = mueblesMapa();
  // fantasma: dónde estaba lo que se movió o se cambió
  for (const { o } of muebles) if (movido(o) && o.caja0) t.push(`<polygon points="${pts(esquinas(o.caja0))}" class="fantasma"/>`);
  // una etiqueta por nombre, en el más grande
  const etiquetaEn = new Map();
  for (const { o, i } of muebles) { const a = o.caja.h[0] * o.caja.h[1], e = etiquetaEn.get(loc(o, 'nombre')); if (!e || a > e.a) etiquetaEn.set(loc(o, 'nombre'), { i, a }); }
  const textos = [];
  for (const { o, i } of muebles) {
    const mov = movido(o), alto = o.medidas.sobrePiso > 1.2, { c, h } = o.caja;
    const clase = ['caja', mov ? 'mov' : '', alto ? 'alto' : '', esVentana(o) ? 'ventana' : '', esPuerta(o) ? 'puerta' : ''].filter(Boolean).join(' ');
    let g = `<g class="mueble${o === fichaActual ? ' sel' : ''}${editable(o) ? ' movible' : ''}" data-i="${i}"><title>${escHtml(loc(o, 'nombre'))}</title><polygon points="${pts(esquinas(o.caja))}" class="${clase}${foto ? ' sobre' : ''}"/>`;
    const lado = Math.min(2 * h[0], 2 * h[1]) * S * 0.85, conImg = !foto && o.img && !alto && o.medidas.alto > 0.05 && lado > 12 && !esVentana(o) && !esPuerta(o);
    if (conImg) { const s = Math.min(lado, 80); g += `<image href="${escHtml(o.img)}" x="${(+X(c[0]) - s / 2).toFixed(1)}" y="${(+Y(c[1]) - s / 2).toFixed(1)}" width="${s.toFixed(1)}" height="${s.toFixed(1)}" preserveAspectRatio="xMidYMid meet"/>`; }
    if (o.colocacion && (!foto || mov)) {
      const [fx, fy] = frente(o), co = Math.cos(o.caja.yaw), si = Math.sin(o.caja.yaw);
      const ext = Math.abs(fx * co + fy * si) * h[0] + Math.abs(-fx * si + fy * co) * h[1];
      const p = (d, lat) => [c[0] + fx * d - fy * lat, c[1] + fy * d + fx * lat];
      g += `<polygon points="${pts([p(ext + 0.09, 0), p(ext + 0.01, 0.06), p(ext + 0.01, -0.06)])}" class="frente${mov ? ' mov' : ''}"/>`;
    }
    g += '</g>'; t.push(g);
    if (!foto && etiquetaEn.get(loc(o, 'nombre'))?.i === i && !esVentana(o)) {
      const ty = conImg ? +Y(c[1]) + Math.min(lado, 80) / 2 + 11 : +Y(c[1]) + 4;
      textos.push(`<text x="${X(c[0])}" y="${ty.toFixed(1)}" class="etq">${escHtml(loc(o, 'nombre').toLowerCase())}</text>`);
    }
  }
  t.push(...textos);
  const yo = dondeEstoy();
  if (yo) {
    const a = Math.atan2(-yo.dy, yo.dx) * 180 / Math.PI; // en el svg la y va hacia abajo
    t.push(`<g transform="translate(${X(yo.x)},${Y(yo.y)}) rotate(${a.toFixed(1)})" class="yo"><path d="M0,0 L60,-32 A68,68 0 0 1 60,32 Z" class="cono"/><circle r="9"/></g>`);
  }
  t.push(`<g class="escala" transform="translate(20,${H - 18})"><line x1="0" y1="0" x2="${S}" y2="0"/><text x="${S / 2}" y="-6">1 m</text></g>`);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" data-w="${W}" data-h="${H}" class="${foto ? 'conFoto' : ''}">${t.join('')}</svg>`;
}
function ajustarSvg() {
  const s = $('mapaSvg').firstElementChild, m = $('mapaMarco'); if (!s) return;
  const W = +s.dataset.w, H = +s.dataset.h, f = Math.min((m.clientWidth - 2) / W, (m.clientHeight - 2) / H) * (m.classList.contains('zoom') ? 2.5 : 1);
  s.style.width = `${Math.floor(W * f)}px`; s.style.height = `${Math.floor(H * f)}px`;
}
function acercar(e, el, ajustar) { // acercar o alejar hacia donde se tocó
  const marco = $('mapaMarco'), r = el.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
  marco.classList.toggle('zoom'); if (ajustar) ajustar();
  if (marco.classList.contains('zoom')) { const r2 = el.getBoundingClientRect(); marco.scrollLeft = fx * r2.width - marco.clientWidth / 2; marco.scrollTop = fy * r2.height - marco.clientHeight / 2; }
}
$('mapaImg').onclick = (e) => acercar(e, e.target);
// Editar desde el mapa: tocar un mueble abre su ficha (ahí mismo); arrastrarlo lo mueve por el piso.
// Con mouse se puede arrastrar de una vez; en el celular primero se toca (para no estorbar al desplazar el mapa).
var tipoMapa = null, fotoPlanta = null, limMapa = null, arrastre = null, dibujandoMapa = false, pedidoMapa = 0, relojFoto = 0;
function redibujarMapa() {
  if (vistaActual !== 'mapa' || !tipoMapa || !fichas) return;
  dibujandoMapa = true;
  try {
    for (const o of fichas) if (o.soporte) seguirMarco(o);
    $('mapaSvg').innerHTML = svgMapa(tipoMapa, fotoPlanta, limMapa || limitesPlano()); ajustarSvg();
  } finally { dibujandoMapa = false; }
}
function pedirMapa() { // se llama con cada cambio; junta los de un mismo cuadro y la foto 3D la rehace al terminar
  if (dibujandoMapa || vistaActual !== 'mapa' || !tipoMapa) return;
  if (!pedidoMapa) pedidoMapa = requestAnimationFrame(() => { pedidoMapa = 0; redibujarMapa(); });
  if (tipoMapa === 'planta') {
    clearTimeout(relojFoto);
    relojFoto = setTimeout(() => {
      if (arrastre?.movio) { pedirMapa(); return; }
      if (vistaActual !== 'mapa' || tipoMapa !== 'planta') return;
      dibujandoMapa = true; let f = null; try { f = renderPlanta(limMapa); } finally { dibujandoMapa = false; }
      if (f) { fotoPlanta = f; redibujarMapa(); }
    }, 350);
  }
}
function puntoSvg(e) { const s = $('mapaSvg').firstElementChild, r = s.getBoundingClientRect(); return { x: (e.clientX - r.left) / r.width * +s.dataset.w, y: (e.clientY - r.top) / r.height * +s.dataset.h }; }
$('mapaSvg').addEventListener('pointerdown', (e) => {
  const sv = $('mapaSvg'); if (!sv.firstElementChild || e.button > 0) return;
  // si hay varios encimados, gana el que ya está seleccionado
  const bajo = document.elementsFromPoint(e.clientX, e.clientY).map((x) => x.closest?.('[data-i]')).filter(Boolean);
  const g = bajo.find((x) => fichas?.[+x.dataset.i] === fichaActual) || e.target.closest('[data-i]'), o = g && fichas ? fichas[+g.dataset.i] : null;
  arrastre = { x: e.clientX, y: e.clientY, o, p0: puntoSvg(e), movio: false, id: e.pointerId,
    puede: !!(o && editable(o) && (e.pointerType === 'mouse' || o === fichaActual)) };
  if (arrastre.puede) { sv.setPointerCapture(e.pointerId); e.preventDefault(); }
});
$('mapaSvg').addEventListener('pointermove', (e) => {
  const a = arrastre; if (!a || !a.puede || e.pointerId !== a.id) return;
  if (!a.movio && Math.hypot(e.clientX - a.x, e.clientY - a.y) < 5) return;
  if (!a.movio) { a.movio = true; const sp = soporte(a.o); guardarHistoria(a.o); a.inicio = sp.position.clone(); if (fichaActual !== a.o) mostrarFicha(a.o); }
  const p = puntoSvg(e);
  a.o.soporte.position.set(a.inicio.x + (p.x - a.p0.x) / 100, a.inicio.y, a.inicio.z + (p.y - a.p0.y) / 100);
  seguirMarco(a.o); if (fichaActual === a.o) refrescarEditar(a.o);
});
$('mapaSvg').addEventListener('pointerup', (e) => {
  const a = arrastre; if (!a || e.pointerId !== a.id) return; arrastre = null;
  if (a.movio) { pedirMapa(); return; }
  if (Math.hypot(e.clientX - a.x, e.clientY - a.y) > 8) return; // fue desplazar el mapa
  if (a.o) { mostrarFicha(a.o); return; }
  if (fichaActual) { cerrarFicha(); return; }
  acercar(e, $('mapaSvg').firstElementChild, ajustarSvg);
});
$('mapaSvg').addEventListener('pointercancel', () => { arrastre = null; pedirMapa(); });
// Tocar un mueble en el mapa: vuelve a Todo junto mirando hacia él y abre su ficha.
function abrirDesdeMapa(o) {
  if (!o) return;
  if (escenaDe !== actual.id) { aviso(t('map.openSceneForItem')); return; }
  if (camEscena) {
    const p = camEscena.pos, c = o.caja.c, dx = c[0] - p.x, dz = -c[1] - p.z, dy = c[2] - p.y;
    camEscena.yaw = Math.atan2(-dx, -dz); camEscena.pitch = Math.max(-1.2, Math.min(0.6, Math.atan2(dy, Math.hypot(dx, dz))));
  }
  ponerVista('escena'); mostrarFicha(o);
}
for (const b of document.querySelectorAll('#pestanas button')) b.onclick = () => { ponerVista(b.dataset.vista); b.blur(); };
$('btnAnt').onclick = () => verSuelto(iSuelto - 1);
$('btnSig').onclick = () => verSuelto(iSuelto + 1);
lienzo.addEventListener('wheel', (e) => {
  if (vistaActual !== 'modelos') return;
  e.preventDefault(); orbita.dist = Math.min(orbita.radio * 20, Math.max(orbita.radio * 0.6, orbita.dist * Math.exp(e.deltaY * 0.0012)));
}, { passive: false });
// pellizco con dos dedos para acercar en Modelos
let pellizco = null;
lienzo.addEventListener('touchmove', (e) => {
  if (vistaActual !== 'modelos' || e.touches.length !== 2) { pellizco = null; return; }
  const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
  if (pellizco) orbita.dist = Math.min(orbita.radio * 20, Math.max(orbita.radio * 0.6, orbita.dist * pellizco / d));
  pellizco = d;
}, { passive: true });
lienzo.addEventListener('touchend', () => { pellizco = null; });

// ---------- Menú "Vistas": las del proyecto + las que guarda el usuario (en este navegador) ----------
let vistasBase = [];
const menuVistas = $('menuVistas');
const claveVistas = () => `visor_vistas_${actual?.id}`;
function vistasGuardadas() { try { return JSON.parse(localStorage.getItem(claveVistas()) || '[]'); } catch { return []; } }
function ponerVistas(base) { vistasBase = base; vistas = [...base, ...vistasGuardadas()]; dibujarVistas(); }
function dibujarVistas() {
  divVistas.innerHTML = ''; $('misVistas').innerHTML = '';
  vistas.forEach((v, i) => {
    const propia = i >= vistasBase.length, fila = document.createElement('div'); fila.className = 'fila-vista';
    const b = document.createElement('button');
    b.innerHTML = `<span class="num">${i < 9 ? i + 1 : ''}</span>`; b.append(loc(v, 'nombre') || t('view.n', { n: i + 1 }));
    b.title = i < 9 ? t('views.key', { n: i + 1 }) : '';
    b.onclick = () => { cerrarMenuVistas(); irAVista(i); lienzo.focus(); };
    fila.appendChild(b);
    if (propia) {
      const x = document.createElement('button'); x.textContent = '✕'; x.title = t('views.remove');
      x.onclick = (e) => { e.stopPropagation(); const g = vistasGuardadas(); g.splice(i - vistasBase.length, 1); localStorage.setItem(claveVistas(), JSON.stringify(g)); ponerVistas(vistasBase); };
      fila.appendChild(x);
    }
    (propia ? $('misVistas') : divVistas).appendChild(fila);
  });
}
function cerrarMenuVistas() { menuVistas.classList.remove('abierto'); $('btnVistas').setAttribute('aria-expanded', 'false'); }
$('btnVistas').onclick = (e) => {
  e.stopPropagation();
  const abrir = !menuVistas.classList.contains('abierto');
  menuVistas.classList.toggle('abierto', abrir); $('btnVistas').setAttribute('aria-expanded', String(abrir));
};
menuVistas.addEventListener('click', (e) => e.stopPropagation());
addEventListener('click', cerrarMenuVistas);
lienzo.addEventListener('touchstart', cerrarMenuVistas, { passive: true });
$('btnGuardarVista').onclick = () => {
  const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
  const r = (v) => Math.round(v * 100) / 100, p = camera.position, g = vistasGuardadas();
  const nombre = (prompt(t('views.prompt'), t('views.mine', { n: g.length + 1 })) || '').trim();
  if (!nombre) return;
  g.push({ nombre, pos: [r(p.x), r(p.y), r(p.z)], mirar: [r(p.x + dir.x * 2), r(p.y + dir.y * 2), r(p.z + dir.z * 2)] });
  localStorage.setItem(claveVistas(), JSON.stringify(g)); ponerVistas(vistasBase);
  aviso(t('views.saved', { name: nombre, key: vistas.length <= 9 ? vistas.length : '—' }));
};

function anguloDesde(pos, mirar) {
  const dx = mirar[0] - pos[0], dy = mirar[1] - pos[1], dz = mirar[2] - pos[2];
  const L = Math.hypot(dx, dy, dz) || 1;
  return { yaw: Math.atan2(-dx, -dz), pitch: Math.asin(dy / L) };
}

function irAVista(i, inmediato = false) {
  const v = vistas[i]; if (!v) return;
  let [x, y, z] = v.pos;
  if (modo === 'caminar') { [x, z] = lugarLibre(x, z); y = pisoY + ALTURA_OJOS; }
  const ang = anguloDesde(v.pos, v.mirar);
  if (inmediato) {
    camera.position.set(x, y, z); yaw = ang.yaw; pitch = ang.pitch;
    transicion = null;
  } else {
    let dyaw = ang.yaw - yaw; dyaw = Math.atan2(Math.sin(dyaw), Math.cos(dyaw));
    transicion = { t: 0, desde: camera.position.clone(), hasta: new THREE.Vector3(x, y, z), yaw0: yaw, yaw1: yaw + dyaw, p0: pitch, p1: ang.pitch };
  }
}

// ---------- Modos ----------
function ponerModo(m) {
  modo = m;
  $('btnVolar').classList.toggle('activo', m === 'volar');
  $('btnCaminar').classList.toggle('activo', m === 'caminar');
  document.body.classList.toggle('volar', m === 'volar');
  if (m === 'caminar' && mapa) {
    const [x, z] = lugarLibre(camera.position.x, camera.position.z);
    transicion = { t: 0, desde: camera.position.clone(), hasta: new THREE.Vector3(x, pisoY + ALTURA_OJOS, z), yaw0: yaw, yaw1: yaw, p0: pitch, p1: pitch };
    aviso(t('mode.walk'));
  } else aviso(t('mode.fly'));
}
document.body.classList.add('volar');
$('btnVolar').onclick = () => { ponerModo('volar'); lienzo.focus(); };
$('btnCaminar').onclick = () => { ponerModo('caminar'); lienzo.focus(); };
$('luz').oninput = () => aplicarExposicion();
$('btnHorneado').onclick = () => { luzHorneada = !luzHorneada; escenaDe = null; cargarCuarto(actual); lienzo.focus(); };
$('btnAO').onclick = () => { aoForzado = !aoEnUso(); actualizarBotonesLuz(); aviso(aoEnUso() ? t('ao.on') : t('ao.off')); lienzo.focus(); };

// ---------- Teclado y mouse ----------
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.code;
  if (vistaActual === 'modelos') {
    if (k === 'ArrowLeft') verSuelto(iSuelto - 1);
    if (k === 'ArrowRight') verSuelto(iSuelto + 1);
    return;
  }
  if (vistaActual !== 'escena' || renderActivo) return;
  teclas.add(k);
  if (k === 'KeyF') ponerModo('volar');
  if (k === 'KeyC') ponerModo('caminar');
  if (/^Digit[1-9]$/.test(k)) irAVista(parseInt(k.slice(5)) - 1);
  if (k === 'KeyV') copiarVista();
  if (['Space', 'ArrowUp', 'ArrowDown'].includes(k)) e.preventDefault();
});
addEventListener('keyup', (e) => teclas.delete(e.code));
addEventListener('blur', () => teclas.clear());

function girar(dx, dy, sens) {
  if (renderActivo) return;
  if (vistaActual === 'modelos') {
    orbita.yaw -= dx * sens * 1.6; orbita.pitch = Math.max(-1.4, Math.min(1.4, orbita.pitch + dy * sens * 1.6));
    return;
  }
  yaw -= dx * sens; pitch -= dy * sens;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
  transicion = null;
}
// Pointer lock en escritorio; si no se puede, arrastrar con el botón también sirve.
let arrastrando = false, ultimo = null;
lienzo.addEventListener('mousedown', (e) => {
  if (esTactil) return;
  if (gizmo.axis !== null || gizmo.dragging) return;
  arrastrando = true; ultimo = [e.clientX, e.clientY];
  if (renderActivo) { arrastrando = false; return; }
  if (vistaActual === 'escena' && document.pointerLockElement !== lienzo && !gizmo.object) {
    try { const r = lienzo.requestPointerLock(); if (r?.catch) r.catch(() => {}); } catch { /* sin pointer lock: se usa arrastre */ }
  }
});
addEventListener('mouseup', () => { arrastrando = false; });
addEventListener('mousemove', (e) => {
  if (document.pointerLockElement === lienzo) girar(e.movementX, e.movementY, SENS_MOUSE);
  else if (arrastrando && ultimo) { girar(e.clientX - ultimo[0], e.clientY - ultimo[1], SENS_MOUSE * 1.3); ultimo = [e.clientX, e.clientY]; }
});
document.addEventListener('pointerlockchange', () => {
  $('mira').style.display = document.pointerLockElement === lienzo ? 'block' : 'none';
});

function copiarVista() {
  const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
  const r = (v) => Math.round(v * 100) / 100;
  const p = camera.position;
  const txt = JSON.stringify({ nombre: 'Nueva vista', pos: [r(p.x), r(p.y), r(p.z)], mirar: [r(p.x + dir.x * 2), r(p.y + dir.y * 2), r(p.z + dir.z * 2)] });
  navigator.clipboard?.writeText(txt).catch(() => {});
  console.info('Vista actual (pégala en "vistas" de cuartos.json):', txt);
  aviso(t('views.copied', { txt }), 5000);
}

// ---------- Táctil: joystick + arrastrar para mirar ----------
const base = $('joystick'), perilla = base.querySelector('.perilla');
let dedoJoy = null, dedoMira = null, ultMira = null;
function moverPerilla(t) {
  const r = base.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2, max = r.width / 2 - 14;
  let dx = t.clientX - cx, dy = t.clientY - cy; const L = Math.hypot(dx, dy);
  if (L > max) { dx *= max / L; dy *= max / L; }
  perilla.style.transform = `translate(${dx}px, ${dy}px)`;
  joy.x = dx / max; joy.y = dy / max;
}
base.addEventListener('touchstart', (e) => { e.preventDefault(); const t = e.changedTouches[0]; dedoJoy = t.identifier; moverPerilla(t); }, { passive: false });
lienzo.addEventListener('touchstart', (e) => {
  e.preventDefault();
  if (gizmo.axis !== null || gizmo.dragging) return;
  for (const t of e.changedTouches) if (dedoMira === null) { dedoMira = t.identifier; ultMira = [t.clientX, t.clientY]; }
}, { passive: false });
addEventListener('touchmove', (e) => {
  for (const t of e.changedTouches) {
    if (t.identifier === dedoJoy) moverPerilla(t);
    else if (t.identifier === dedoMira) { girar(t.clientX - ultMira[0], t.clientY - ultMira[1], SENS_TACTIL); ultMira = [t.clientX, t.clientY]; }
  }
  if (dedoJoy !== null || dedoMira !== null) e.preventDefault();
}, { passive: false });
const finToque = (e) => {
  for (const t of e.changedTouches) {
    if (t.identifier === dedoJoy) { dedoJoy = null; joy.x = joy.y = 0; perilla.style.transform = ''; }
    if (t.identifier === dedoMira) dedoMira = null;
  }
};
addEventListener('touchend', finToque); addEventListener('touchcancel', finToque);
for (const [id, v] of [['btnSubir', 1], ['btnBajar', -1]]) {
  const b = $(id);
  b.addEventListener('touchstart', (e) => { e.preventDefault(); subirTactil = v; }, { passive: false });
  b.addEventListener('touchend', () => { subirTactil = 0; });
  b.addEventListener('mousedown', () => { subirTactil = v; });
  b.addEventListener('mouseup', () => { subirTactil = 0; });
}

// ---------- Bucle ----------
let tAnterior = performance.now();
const adelante = new THREE.Vector3(), derecha = new THREE.Vector3(), mov = new THREE.Vector3();
function paso(dt) {
  if (vistaActual === 'modelos') {
    if (!arrastrando && dedoMira === null) orbita.yaw += dt * 0.25; // gira solo, despacio
    const cp = Math.cos(orbita.pitch);
    camera.position.set(orbita.centro.x + Math.sin(orbita.yaw) * cp * orbita.dist, orbita.centro.y + Math.sin(orbita.pitch) * orbita.dist,
      orbita.centro.z + Math.cos(orbita.yaw) * cp * orbita.dist);
    // mirar un poco por debajo del centro: así el modelo queda más arriba, libre de la tira de miniaturas
    camera.lookAt(orbita.centro.x, orbita.centro.y - orbita.radio * 0.22, orbita.centro.z);
    sol.position.copy(camera.position).add(new THREE.Vector3(orbita.radio * 2, orbita.radio * 3, 0)); sol.target.position.copy(orbita.centro);
    return;
  }
  if (vistaActual !== 'escena') return;
  if (transicion) {
    transicion.t = Math.min(1, transicion.t + dt / 0.9);
    const s = transicion.t * transicion.t * (3 - 2 * transicion.t);
    camera.position.lerpVectors(transicion.desde, transicion.hasta, s);
    yaw = transicion.yaw0 + (transicion.yaw1 - transicion.yaw0) * s;
    pitch = transicion.p0 + (transicion.p1 - transicion.p0) * s;
    if (transicion.t >= 1) transicion = null;
  }
  camera.rotation.set(pitch, yaw, 0);

  let f = 0, r = 0, u = 0;
  if (teclas.has('KeyW') || teclas.has('ArrowUp')) f += 1;
  if (teclas.has('KeyS') || teclas.has('ArrowDown')) f -= 1;
  if (teclas.has('KeyD') || teclas.has('ArrowRight')) r += 1;
  if (teclas.has('KeyA') || teclas.has('ArrowLeft')) r -= 1;
  if (teclas.has('KeyE') || teclas.has('Space')) u += 1;
  if (teclas.has('KeyQ')) u -= 1;
  f -= joy.y; r += joy.x; u += subirTactil;
  if (!f && !r && !u) return;
  transicion = null;
  const rapido = teclas.has('ShiftLeft') || teclas.has('ShiftRight') ? FACTOR_RAPIDO : 1;

  if (modo === 'volar') {
    camera.getWorldDirection(adelante);
    derecha.crossVectors(adelante, camera.up).normalize();
    mov.set(0, 0, 0).addScaledVector(adelante, f).addScaledVector(derecha, r);
    if (mov.lengthSq() > 1) mov.normalize();
    mov.y += u;
    camera.position.addScaledVector(mov, VEL_VOLAR * escalaVel * rapido * dt);
  } else {
    adelante.set(-Math.sin(yaw), 0, -Math.cos(yaw));
    derecha.set(Math.cos(yaw), 0, -Math.sin(yaw));
    mov.set(0, 0, 0).addScaledVector(adelante, f).addScaledVector(derecha, r);
    if (mov.lengthSq() > 1) mov.normalize();
    mov.multiplyScalar(VEL_CAMINAR * (escalaVel > 1 ? 1.6 : 1) * rapido * dt);
    // sub-pasos para no saltarse paredes delgadas
    const n = Math.max(1, Math.ceil(mov.length() / (CELDA * 0.5)));
    const sx = mov.x / n, sz = mov.z / n;
    for (let k = 0; k < n; k++) {
      const x = camera.position.x, z = camera.position.z;
      if (cabe(x + sx, z + sz)) { camera.position.x += sx; camera.position.z += sz; }
      else if (cabe(x + sx, z)) camera.position.x += sx;       // deslizar contra la pared
      else if (cabe(x, z + sz)) camera.position.z += sz;
    }
    camera.position.y = pisoY + ALTURA_OJOS;
  }
}
function animar() {
  const ahora = performance.now(); const dt = Math.min(0.05, (ahora - tAnterior) / 1000); tAnterior = ahora;
  if (!renderActivo) paso(dt);   // mientras se renderiza no se puede recorrer
  if (vistaActual === 'mapa') return; // el mapa es una imagen encima; no hace falta dibujar 3D
  if (vistaActual === 'escena' && aoEnUso()) composer.render(dt);
  else renderer.render(scene, camera);
}
renderer.setAnimationLoop(animar);
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight, false);
  composer.setSize(innerWidth, innerHeight);
  if (vistaActual === 'mapa') ajustarSvg();
});
composer.setPixelRatio(renderer.getPixelRatio());

// ---------- Language (EN | ES) ----------
// Switching re-renders every text in place; the 3D scene is not reloaded.
for (const b of document.querySelectorAll('#idioma [data-lang]')) b.onclick = () => { setLang(b.dataset.lang); b.blur(); };
onLangChange(() => {
  for (const o of selCuarto.options) { const c = lista.find((x) => x.id === o.value); if (c) o.textContent = loc(c, 'nombre'); }
  if (!actual) return;
  if (!carga.classList.contains('oculta')) cargaTitulo.textContent = loc(actual, 'nombre');
  dibujarVistas();
  dibujarTiras();
  if (iSuelto >= 0 && sueltos[iSuelto]) {
    $('modeloNombre').textContent = loc(sueltos[iSuelto], 'nombre');
    const botones = $('tiras').children;
    for (let k = 0; k < botones.length; k++) botones[k].classList.toggle('activo', k === iSuelto);
  }
  dibujarMapas();
  if (vistaActual === 'modelos' && !sueltos.length) vacio(t('empty.models'));
  if (vistaActual === 'mapa') { if ((actual.mapas || []).length) verMapa(iMapa); else vacio(t('empty.map')); }
  if (fichaActual && vistaActual !== 'modelos') mostrarFicha(fichaActual);
});

// ---------- Inicio ----------
// Para pruebas / depuración desde la consola.
window.visor = { camera, scene, get modelo() { return modelo; }, get modo() { return modo; }, ponerModo, irAVista, get mapa() { return mapa; }, get vistas() { return vistas; }, teclas, cabe, get fichas() { return fichas; }, get catalogo() { return catalogo; }, soporte, cambiarObjeto, ponerVista, cargar: (id) => cargarCuarto(lista.find((c) => c.id === id)) };

(async function inicio() {
  try {
    const r = await fetch('cuartos.json' + VER, { cache: 'no-cache' });
    lista = (await r.json()).cuartos;
  } catch (err) {
    cargaTitulo.textContent = t('init.noJson');
    progreso(0, t('init.serve'));
    return;
  }
  for (const c of lista) {
    const o = document.createElement('option'); o.value = c.id; o.textContent = loc(c, 'nombre'); selCuarto.appendChild(o);
  }
  selCuarto.onchange = () => { elegirProyecto(lista.find((c) => c.id === selCuarto.value)); selCuarto.blur(); };
  const q = new URLSearchParams(location.search);
  const inicial = lista.find((c) => c.id === q.get('cuarto')) || lista[0];
  const vista = ['escena', 'modelos', 'mapa'].includes(q.get('vista')) ? q.get('vista') : 'escena';
  if (vista !== 'escena') { carga.style.transition = 'none'; carga.classList.add('oculta'); requestAnimationFrame(() => (carga.style.transition = '')); }
  elegirProyecto(inicial, vista);
  const nm = parseInt(q.get('modelo') || '0'); if (vista === 'modelos' && nm > 1) verSuelto(nm - 1);
})();
