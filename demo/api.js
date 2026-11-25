// The seam between the page and the pipeline. In DEMO mode every call replays an already processed scene (the hotel)
// with realistic pacing; a real backend only has to implement the same three functions (upload + poll a job, etc.).
export const DEMO = true;
const SPEED = new URLSearchParams(location.search).has("fast") ? 0.08 : 1;  // ?fast: quick run-through for testing
const wait = (ms) => new Promise((r) => setTimeout(r, ms * SPEED));

export const STAGES = [
  { key: "upload", label: "Uploading the video" },
  { key: "frames", label: "Picking the sharpest frames" },
  { key: "depth", label: "Measuring each frame (depth and walls)" },
  { key: "align", label: "Aligning the cameras with each other" },
  { key: "plan", label: "Drawing the floor plan to scale" },
  { key: "inventory", label: "Detecting furniture and objects" },
  { key: "views", label: "Building a clean image of each object" },
];

// processVideo(files, measure, onProgress) -> scene  (onProgress({stage, frac, frame?}))
// Real backend: POST /api/scenes (multipart video + measure) -> job id; poll GET /api/jobs/<id>; GET /api/scenes/<id>/scene.json
export async function processVideo(files, measure, onProgress) {
  const scene = await (await fetch("data/scene.json")).json();
  const times = { upload: files.length ? 1200 : 400, depth: 1800, align: 1300, plan: 900, inventory: 1400, views: 1600 };
  for (let i = 0; i < STAGES.length; i++) {
    const st = STAGES[i].key;
    onProgress({ stage: i, frac: 0 });
    if (st === "frames") {
      for (let k = 0; k < scene.frames.length; k++) {
        await wait(70); onProgress({ stage: i, frac: (k + 1) / scene.frames.length, frame: scene.frames[k] });
      }
    } else {
      const T = times[st], n = 10;
      for (let k = 1; k <= n; k++) { await wait(T / n); onProgress({ stage: i, frac: k / n }); }
    }
  }
  return scene;
}

// generateModels(kinds, onProgress) -> {kind: modelUrl}   (onProgress({kind, frac, done}))
// Real backend: POST /api/scenes/<id>/models {kinds:[...]} -> job; each finished GLB arrives as it is done.
export async function generateModels(kinds, onProgress) {
  const out = {}; const queue = [...kinds];
  async function worker() {  // two at a time: the two ComfyUI instances share the GPU
    while (queue.length) {
      const k = queue.shift(); const T = k.render === "model" ? 1600 + Math.random() * 1200 : 600;
      for (let s = 1; s <= 12; s++) { await wait(T / 12); onProgress({ kind: k.kind, frac: s / 12, done: s === 12 }); }
      out[k.kind] = k.model;
    }
  }
  await Promise.all([worker(), worker()]);
  return out;
}

// regenerateView(kind) -> new clean image url (demo: same image after a pause)
export async function regenerateView(kind) { await wait(1500); return kind.clean_view; }

// saveLayout(layout): real backend would PUT /api/scenes/<id>/layout; the demo downloads it as a file
export async function saveLayout(layout) {
  const blob = new Blob([JSON.stringify(layout, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "room_layout.json"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
