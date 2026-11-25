// Minimal i18n layer for the viewer: English + Spanish UI strings, language choice and data-field lookup.
// Language: ?lang=en|es in the URL, else the saved choice (localStorage), else the browser language
// (Spanish if it starts with "es", English otherwise).
// Static markup uses data-i18n (text), data-i18n-html (markup), data-i18n-title, data-i18n-aria and data-i18n-alt.
// Data files keep the Spanish fields and add "<field>_en" next to them; loc() picks the right one.

const STRINGS = {
  es: {
    'page.title': 'Mole Poblano · Visor 3D',
    'lang.label': 'Idioma',
    'project': 'Proyecto',
    'loading': 'Cargando…',
    'tab.scene': 'Todo junto', 'tab.scene.title': 'La escena completa, para recorrerla',
    'tab.models': 'Modelos', 'tab.models.title': 'Cada modelo 3D por separado',
    'tab.map': 'Mapa', 'tab.map.title': 'Planos y vista desde arriba',
    'sec.views': 'Vistas', 'views.button': 'Vistas ▾', 'views.button.title': 'Puntos de vista guardados (teclas 1–9)',
    'views.save': '＋ Guardar esta vista',
    'sec.move': 'Moverse',
    'fly': 'Volar', 'fly.title': 'Volar libremente (tecla F)',
    'walk': 'Caminar', 'walk.title': 'Caminar a 1.6 m sin atravesar paredes (tecla C)',
    'sec.light': 'Luz',
    'baked': 'Luz Cycles', 'baked.title': 'Usar la luz horneada en Blender/Cycles (se ve como los renders) o luz en vivo',
    'ao': 'Sombras', 'ao.title': 'Sombras suaves en rincones y debajo de los muebles (oclusión ambiental)',
    'light.intensity': 'Intensidad de la luz',
    'render': 'Render', 'render.title': 'Render con luz realista desde esta vista (solo en esta PC)',
    'help.scene': '<b>Clic</b> en la escena para mirar con el mouse (<kbd>Esc</kbd> para soltar).<br>\n  <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> moverse · <kbd>Q</kbd>/<kbd>E</kbd> bajar/subir (volar) · <kbd>Shift</kbd> rápido<br>\n  <kbd>F</kbd> volar · <kbd>C</kbd> caminar · <kbd>1</kbd>–<kbd>9</kbd> vistas · <kbd>V</kbd> copiar vista actual<br>\n  <b>Doble clic</b> en un objeto para ver qué es y sus medidas',
    'help.models': '<b>Arrastra</b> para girar el modelo · <b>rueda</b> para acercar<br>\n  <kbd>←</kbd> <kbd>→</kbd> modelo anterior / siguiente',
    'prev': 'Anterior', 'next': 'Siguiente', 'up': 'Subir', 'down': 'Bajar', 'close': 'Cerrar',
    'map.alt': 'Mapa del proyecto',
    'map.tapZoom': 'Toca la imagen para acercar o alejar.',
    'map.origPhoto': ' (foto original)',
    'map.liveFloor': 'Planta 3D (en vivo)', 'map.livePlan': 'Plano de objetos (en vivo)',
    'map.openSceneFirst': 'Para ver la planta 3D abre primero <b>Todo junto</b>. Mientras, este es el plano. ',
    'map.legend': '<span class="lm naranja"></span> movido o cambiado (punteado: donde estaba) · <span class="lm azul"></span> dónde estás · toca un mueble para ver su ficha y moverlo (arrástralo), o el fondo para acercar.',
    'map.openSceneForItem': 'Abre primero Todo junto para ver ese mueble.',
    'card.aria': 'Ficha del objeto',
    'card.length': 'Largo', 'card.depth': 'Ancho / fondo', 'card.height': 'Alto', 'card.aboveFloor': 'Sobre el piso',
    'card.numOf': '({num} de {total})', 'card.insteadOf': '· en lugar de {name}',
    'card.view3d': 'Ver en 3D ›', 'card.count': 'Hay {n} en este cuarto', 'card.modelFrom': 'Modelo de: {room}',
    'card.note': 'Medidas aproximadas del modelo 3D (la caja naranja)', 'card.flat': '; es una superficie plana',
    'swap.needCycles': 'Para cambiar o mover este objeto, activa <b>Luz Cycles</b>.',
    'swap.title': 'Cambiar por otro', 'swap.all': 'Cambiar los {n} iguales', 'swap.gallery': 'De la galería de objetos',
    'swap.note': 'Es una prueba visual: no cambia las paredes para caminar y el botón Render sigue usando el original. Al recargar la página vuelve todo como estaba.',
    'swap.original': 'Original', 'swap.loadFailed': 'No se pudo cargar ese modelo.',
    'swap.cantSeparate': 'Este objeto no se puede separar del resto.',
    'undo.none': 'No hay nada que deshacer.', 'redo.none': 'No hay nada que rehacer.',
    'edit.title': 'Mover y girar', 'edit.move': '✥ Mover', 'edit.move.title': 'Flechas para moverlo',
    'edit.rotate': '⟳ Girar', 'edit.rotate.title': 'Aros para girarlo', 'edit.snap': 'Pegado al piso',
    'edit.turn': 'Giro', 'edit.height': 'Altura',
    'edit.undo': '↶ Deshacer', 'edit.redo': '↷ Rehacer', 'edit.reset': '⟲ A su lugar', 'edit.reset.title': 'Regresarlo a donde estaba',
    'edit.noteMap': 'Arrastra el mueble en el plano para moverlo; gíralo con la barrita.',
    'edit.noteScene': 'Elige Mover o Girar y arrastra las flechas o los aros que salen sobre el objeto.',
    'edit.keys': 'Ctrl+Z deshace, Ctrl+Y rehace.',
    'pick.noCards': 'Este proyecto no tiene fichas de objetos.', 'pick.none': 'Ahí no hay un objeto identificado.',
    'load.walls': 'Preparando paredes para caminar…',
    'load.downloading': 'Descargando modelo…', 'load.downloadingMB': 'Descargando modelo… {a} / {b} MB',
    'load.failed': 'No se pudo cargar {file}. ¿Abriste la página con serve.bat?',
    'load.fileFailed': 'No se pudo cargar {file}',
    'empty.models': 'Este proyecto no tiene modelos sueltos.', 'empty.map': 'Este proyecto no tiene mapa.',
    'view.n': 'Vista {n}', 'views.key': 'Tecla {n}', 'views.remove': 'Quitar esta vista',
    'views.prompt': 'Nombre de la vista:', 'views.mine': 'Mi vista {n}', 'views.saved': 'Vista guardada: {name} (tecla {key})',
    'views.copied': 'Vista copiada: {txt}',
    'mode.walk': 'Modo caminar: a 1.6 m, sin atravesar paredes', 'mode.fly': 'Modo volar: Q/E para bajar y subir',
    'ao.on': 'Sombras suaves activadas', 'ao.off': 'Sombras suaves desactivadas',
    'render.panel': 'Render de esta vista', 'render.preparing': 'Preparando…', 'render.download': 'Descargar PNG', 'render.exit': 'Salir',
    'render.confirm': 'El render usa la PC con Blender y puede tardar. En el celular solo funciona si la página está abierta en la misma PC. ¿Continuar?',
    'render.opening': 'Abriendo Blender…',
    'render.needServer': 'El render necesita abrir el visor con servir.bat (servidor con Blender), en esta misma PC.',
    'render.done': 'Listo: {m} muestras por píxel · {s} s',
    'render.failed': 'Blender falló. Revisa render/resultados/<fecha>/blender.log',
    'render.loading': 'Abriendo Blender y cargando el cuarto…',
    'render.step': 'Renderizando {m} muestras por píxel (paso {i} de {n})…',
    'init.noJson': 'No se pudo leer cuartos.json',
    'init.serve': 'Abre la página desde un servidor local (serve.bat o python -m http.server), no con doble clic en el archivo.',
  },
  en: {
    'page.title': 'Mole Poblano · 3D viewer',
    'lang.label': 'Language',
    'project': 'Project',
    'loading': 'Loading…',
    'tab.scene': 'All together', 'tab.scene.title': 'The whole scene, to walk around in',
    'tab.models': 'Models', 'tab.models.title': 'Each 3D model on its own',
    'tab.map': 'Map', 'tab.map.title': 'Floor plans and top-down view',
    'sec.views': 'Views', 'views.button': 'Views ▾', 'views.button.title': 'Saved viewpoints (keys 1–9)',
    'views.save': '＋ Save this view',
    'sec.move': 'Move',
    'fly': 'Fly', 'fly.title': 'Fly freely (F key)',
    'walk': 'Walk', 'walk.title': 'Walk at 1.6 m eye height without going through walls (C key)',
    'sec.light': 'Light',
    'baked': 'Cycles light', 'baked.title': 'Use the lighting baked in Blender/Cycles (looks like the renders) or live lighting',
    'ao': 'Shadows', 'ao.title': 'Soft shadows in corners and under furniture (ambient occlusion)',
    'light.intensity': 'Light intensity',
    'render': 'Render', 'render.title': 'Realistic-light render from this view (only on this PC)',
    'help.scene': '<b>Click</b> the scene to look around with the mouse (<kbd>Esc</kbd> to release).<br>\n  <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move · <kbd>Q</kbd>/<kbd>E</kbd> down/up (fly) · <kbd>Shift</kbd> fast<br>\n  <kbd>F</kbd> fly · <kbd>C</kbd> walk · <kbd>1</kbd>–<kbd>9</kbd> views · <kbd>V</kbd> copy current view<br>\n  <b>Double-click</b> an object to see what it is and its size',
    'help.models': '<b>Drag</b> to rotate the model · <b>wheel</b> to zoom<br>\n  <kbd>←</kbd> <kbd>→</kbd> previous / next model',
    'prev': 'Previous', 'next': 'Next', 'up': 'Up', 'down': 'Down', 'close': 'Close',
    'map.alt': 'Project map',
    'map.tapZoom': 'Tap the image to zoom in or out.',
    'map.origPhoto': ' (original image)',
    'map.liveFloor': '3D floor plan (live)', 'map.livePlan': 'Object plan (live)',
    'map.openSceneFirst': 'To see the 3D floor plan, open <b>All together</b> first. Meanwhile, here is the plan. ',
    'map.legend': '<span class="lm naranja"></span> moved or swapped (dashed: where it was) · <span class="lm azul"></span> where you are · tap a piece of furniture to see its info and move it (drag it), or the background to zoom.',
    'map.openSceneForItem': 'Open All together first to see that piece.',
    'card.aria': 'Object info',
    'card.length': 'Length', 'card.depth': 'Width / depth', 'card.height': 'Height', 'card.aboveFloor': 'Above the floor',
    'card.numOf': '({num} of {total})', 'card.insteadOf': '· instead of {name}',
    'card.view3d': 'View in 3D ›', 'card.count': 'There are {n} in this room', 'card.modelFrom': 'Model from: {room}',
    'card.note': 'Approximate measurements of the 3D model (the orange box)', 'card.flat': '; it is a flat surface',
    'swap.needCycles': 'To swap or move this object, turn on <b>Cycles light</b>.',
    'swap.title': 'Swap for another', 'swap.all': 'Swap all {n} identical ones', 'swap.gallery': 'From the object gallery',
    'swap.note': 'This is only a visual preview: it does not change the walls used for walking, and the Render button still uses the original. Reloading the page restores everything.',
    'swap.original': 'Original', 'swap.loadFailed': 'Could not load that model.',
    'swap.cantSeparate': 'This object cannot be separated from the rest.',
    'undo.none': 'Nothing to undo.', 'redo.none': 'Nothing to redo.',
    'edit.title': 'Move and rotate', 'edit.move': '✥ Move', 'edit.move.title': 'Arrows to move it',
    'edit.rotate': '⟳ Rotate', 'edit.rotate.title': 'Rings to rotate it', 'edit.snap': 'Stick to floor',
    'edit.turn': 'Turn', 'edit.height': 'Height',
    'edit.undo': '↶ Undo', 'edit.redo': '↷ Redo', 'edit.reset': '⟲ Reset', 'edit.reset.title': 'Put it back where it was',
    'edit.noteMap': 'Drag the piece on the plan to move it; turn it with the slider.',
    'edit.noteScene': 'Choose Move or Rotate and drag the arrows or rings that appear on the object.',
    'edit.keys': 'Ctrl+Z undoes, Ctrl+Y redoes.',
    'pick.noCards': 'This project has no object info.', 'pick.none': 'No identified object there.',
    'load.walls': 'Preparing walls for walking…',
    'load.downloading': 'Downloading model…', 'load.downloadingMB': 'Downloading model… {a} / {b} MB',
    'load.failed': 'Could not load {file}. Did you open the page with serve.bat?',
    'load.fileFailed': 'Could not load {file}',
    'empty.models': 'This project has no separate models.', 'empty.map': 'This project has no map.',
    'view.n': 'View {n}', 'views.key': 'Key {n}', 'views.remove': 'Remove this view',
    'views.prompt': 'View name:', 'views.mine': 'My view {n}', 'views.saved': 'View saved: {name} (key {key})',
    'views.copied': 'View copied: {txt}',
    'mode.walk': 'Walk mode: eye height 1.6 m, no going through walls', 'mode.fly': 'Fly mode: Q/E to go down and up',
    'ao.on': 'Soft shadows on', 'ao.off': 'Soft shadows off',
    'render.panel': 'Render of this view', 'render.preparing': 'Preparing…', 'render.download': 'Download PNG', 'render.exit': 'Exit',
    'render.confirm': 'Rendering uses the PC running Blender and can take a while. On a phone it only works if the page is open on that same PC. Continue?',
    'render.opening': 'Opening Blender…',
    'render.needServer': 'Rendering needs the viewer to be opened with servir.bat (server with Blender) on this same PC.',
    'render.done': 'Done: {m} samples per pixel · {s} s',
    'render.failed': 'Blender failed. Check render/resultados/<date>/blender.log',
    'render.loading': 'Opening Blender and loading the room…',
    'render.step': 'Rendering {m} samples per pixel (step {i} of {n})…',
    'init.noJson': 'Could not read cuartos.json',
    'init.serve': 'Open the page from a local server (serve.bat or python -m http.server), not by double-clicking the file.',
  },
};
export const LANGS = ['en', 'es'];
const STORE_KEY = 'visor_lang';

function initialLang() {
  const q = new URLSearchParams(location.search).get('lang');
  if (LANGS.includes(q)) return q;
  try { const s = localStorage.getItem(STORE_KEY); if (LANGS.includes(s)) return s; } catch { /* storage blocked */ }
  return (navigator.language || '').toLowerCase().startsWith('es') ? 'es' : 'en';
}
export let lang = initialLang();

// UI string with {name} placeholders; falls back to Spanish, then to the key itself.
export function t(key, vars) {
  const s = STRINGS[lang][key] ?? STRINGS.es[key] ?? key;
  return vars ? s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m)) : s;
}
// Data field for the active language: o.<key>_en in English when present, else the Spanish o.<key>.
export const loc = (o, key) => (lang === 'en' && o?.[key + '_en']) || o?.[key];

export function applyStatic(root = document) {
  document.documentElement.lang = lang;
  document.title = t('page.title');
  for (const el of root.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll('[data-i18n-html]')) el.innerHTML = t(el.dataset.i18nHtml);
  for (const el of root.querySelectorAll('[data-i18n-title]')) el.title = t(el.dataset.i18nTitle);
  for (const el of root.querySelectorAll('[data-i18n-aria]')) el.setAttribute('aria-label', t(el.dataset.i18nAria));
  for (const el of root.querySelectorAll('[data-i18n-alt]')) el.alt = t(el.dataset.i18nAlt);
  for (const b of root.querySelectorAll('[data-lang]')) b.classList.toggle('activo', b.dataset.lang === lang);
}

const listeners = [];
export const onLangChange = (fn) => listeners.push(fn);
export function setLang(l) {
  if (!LANGS.includes(l) || l === lang) return;
  lang = l;
  try { localStorage.setItem(STORE_KEY, l); } catch { /* storage blocked */ }
  // keep an explicit ?lang= in the URL in sync so a reload keeps the choice
  const url = new URL(location.href);
  if (url.searchParams.has('lang')) { url.searchParams.set('lang', l); history.replaceState(null, '', url); }
  applyStatic();
  for (const fn of listeners) fn();
}
