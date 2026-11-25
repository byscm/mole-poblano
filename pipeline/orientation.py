# Which way each object faces, with Orient Anything V2 (local, GPU). Works on the transparent crops.
# Runs with the ComfyUI venv python (it has torch):
#   ...\ComfyUI\ComfyUI\.venv\Scripts\python.exe orientation.py --images a.png b.png   -> prints raw predictions
#   ...\ComfyUI\ComfyUI\.venv\Scripts\python.exe orientation.py --eval                  -> score vs hand labels
#   ...\ComfyUI\ComfyUI\.venv\Scripts\python.exe orientation.py <scene_folder>          -> writes facing into map.json
# Model code and weights: ORIENT_ANYTHING_DIR (downloaded from the authors' Hugging Face repos).
import json, math, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("ORIENT_ANYTHING_DIR", "orient-anything-v2")
WEIGHTS = os.path.join(MODEL_DIR, "weights", "rotmod_realrotaug_best.pt")
SCENES = os.path.join(os.path.dirname(HERE), "escenas")
SIZE = 518

# Azimuth -> our "relative facing" (0 = front points at the camera, +90 = viewer's right, 180 = away).
# Checked on 30 hand-labeled benchmark crops (--eval): the best fit was -azimuth + 5 deg, i.e. azimuth 0 =
# facing the camera, counterclockwise sign flipped. We use the plain convention (offset 0).
AZ_SIGN, AZ_OFFSET = -1, 0
MIN_CONFIDENCE = 0.6  # below this the prediction is ignored and the geometric facing is kept

_model = None


def load():
    global _model
    if _model is None:
        import torch
        sys.path.insert(0, MODEL_DIR)
        from vision_tower import VGGT_OriAny_Ref
        m = VGGT_OriAny_Ref(out_dim=900, dtype=torch.bfloat16, nopretrain=True)
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        _model = m.eval().to(device="cuda", dtype=torch.bfloat16)
    return _model


def prepare(rgba):
    """Same as the authors' demo: object centered on a square, 85 % of the side, white background, 518 px."""
    import torch
    a = np.array(rgba.convert("RGBA"))
    ys, xs = np.nonzero(a[..., 3] > 0)
    fg = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = max(fg.shape[:2])
    n = int(s / 0.85)
    sq = np.zeros((n, n, 4), np.uint8)
    y0, x0 = (n - fg.shape[0]) // 2, (n - fg.shape[1]) // 2
    sq[y0:y0 + fg.shape[0], x0:x0 + fg.shape[1]] = fg
    im = Image.alpha_composite(Image.new("RGBA", (n, n), (255, 255, 255, 255)), Image.fromarray(sq)).convert("RGB")
    im = im.resize((SIZE, SIZE), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)


def val_fit_alpha(distribute):
    """How many 'fronts' the object has (0 = none/round, 1, 2 or 4), fitted on the azimuth distribution.
    Copied from the authors' inference.py (importing it pulls in rembg, which we do not need)."""
    from functools import partial
    from scipy.integrate import trapezoid
    from scipy.optimize import curve_fit
    vm = lambda alpha, x, mu, kappa: np.exp(kappa * np.cos(alpha * (x - mu))) / (2 * np.pi)
    res = []
    for y in distribute:
        x = np.linspace(0, 2 * np.pi, 360)
        y = y / (trapezoid(y, x) + 1e-8)
        params, r2s = [], []
        for alpha in (1.0, 2.0, 4.0):
            try:
                f = partial(vm, alpha)
                p, _ = curve_fit(f, x, y, p0=[x[np.argmax(y)], 1])
                r2 = 1 - np.sum((y - f(x, *p)) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-8)
                params.append(p); r2s.append(r2)
                if r2 > 0.8:
                    break
            except Exception:
                params.append((0.0, 0.0)); r2s.append(0.0)
        k = int(np.argmax(r2s))
        alpha, (mu, kappa), r2 = (1.0, 2.0, 4.0)[k], params[k], r2s[k]
        ok = r2 >= 0.45 and kappa >= {1.0: 0.6, 2.0: 0.4, 4.0: 0.25}[alpha]
        res.append(alpha if ok else 0.0)
    return res


def predict(images):
    """images: list of RGBA PIL images (transparent background). Returns list of dicts azimuth/polar/rotation/n_dirs."""
    import torch, torch.nn.functional as F
    m = load()
    out = []
    with torch.no_grad():
        for img in images:
            x = prepare(img)[None, None].to(device="cuda", dtype=torch.bfloat16)  # (B=1, S=1, 3, H, W)
            enc = m(x).view(1, -1).float()
            az = int(torch.argmax(enc[:, :360], -1))
            el = int(torch.argmax(enc[:, 360:540], -1)) - 90
            ro = int(torch.argmax(enc[:, 540:900], -1)) - 180
            dist = F.sigmoid(enc[:, :360]).cpu().numpy()
            n_dirs = int(val_fit_alpha(distribute=dist.copy())[0])
            conf = float(dist[0].max())
            out.append(dict(azimuth=az, polar=el, rotation=ro, n_dirs=n_dirs, confidence=round(conf, 3)))
    return out


def to_relative(az, sign=None, offset=None):
    sign = AZ_SIGN if sign is None else sign
    offset = AZ_OFFSET if offset is None else offset
    return ((sign * az + offset + 180) % 360) - 180


def angle_diff(a, b):
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def evaluate():
    sheets = os.path.join(SCENES, "_facing_sheets")
    items = json.load(open(os.path.join(sheets, "index.json")))
    labels = {int(k): v for k, v in json.load(open(os.path.join(sheets, "labels.json"))).items() if not k.startswith("_")}
    cache_p = os.path.join(sheets, "orient_anything.json")
    if os.path.exists(cache_p):
        preds = {int(k): v for k, v in json.load(open(cache_p)).items()}
    else:
        idx = sorted(labels)
        crops = [Image.open(os.path.join(os.path.dirname(os.path.dirname(items[i]["img"])), "crops", f"object_{items[i]['id']}.png"))
                 for i in idx]
        preds = dict(zip(idx, predict(crops)))
        json.dump(preds, open(cache_p, "w"), indent=1)
    best = None
    for sign in (1, -1):
        for off in range(0, 360, 5):
            e = np.array([angle_diff(to_relative(preds[i]["azimuth"], sign, off), v) for i, v in labels.items()])
            if best is None or np.median(e) < best[0]:
                best = (float(np.median(e)), sign, off, e)
    med, sign, off, e = best
    print(f"best mapping: relative = {sign} * azimuth + {off}")
    print(f"within30 {np.mean(e <= 30):4.0%}  within60 {np.mean(e <= 60):4.0%}  backwards(>135) {np.mean(e > 135):4.0%}  median {med:5.1f}")
    for i, v in labels.items():
        p = preds[i]
        print(f"  {i:2d} {items[i]['name'][:28]:28s} label {v:5.0f}  pred {to_relative(p['azimuth'], sign, off):5.0f}  "
              f"az {p['azimuth']:3d} polar {p['polar']:3d} dirs {p['n_dirs']} conf {p['confidence']}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--eval":
        evaluate()
    elif args and args[0] == "--images":
        for path, p in zip(args[1:], predict([Image.open(a) for a in args[1:]])):
            print(path, p)
    elif args:
        apply_to_scene(args[0])
    else:
        sys.exit(__doc__)


def apply_to_scene(folder):
    """Replace the geometric facing in map.json with Orient Anything's, when it has a clear answer."""
    path = os.path.join(folder, "map.json")
    scene = json.load(open(path, encoding="utf-8"))
    objs = scene["objects"]
    preds = predict([Image.open(os.path.join(folder, o["crop"])) for o in objs])
    for o, p in zip(objs, preds):
        o["orientation"] = p
        o.setdefault("yaw_geometry_deg", o["yaw_deg"])
        to_cam = -np.array(o["center_m"][:2], float)  # camera is at (0, 0) of the floor frame
        if p["confidence"] < MIN_CONFIDENCE:  # n_dirs alone is not used: on the benchmark n_dirs=0 cases were still right
            o["facing_source"] = "geometry"
            note = "kept geometry (low confidence)"
        else:
            rel = math.radians(to_relative(p["azimuth"]))
            v = to_cam / max(np.linalg.norm(to_cam), 1e-9)
            front = np.array([v[0] * math.cos(rel) - v[1] * math.sin(rel), v[0] * math.sin(rel) + v[1] * math.cos(rel)])
            o["yaw_deg"] = round((math.degrees(math.atan2(front[1], front[0])) + 90) % 360, 1)
            # width/depth were measured along the geometric axes; if the facing turned ~90 deg, they swap
            turn = abs((o["yaw_deg"] - o["yaw_geometry_deg"]) % 180)
            if 45 < turn < 135 and not o.get("dims_swapped"):
                o["width_m"], o["depth_m"] = o["depth_m"], o["width_m"]
                o["dims_swapped"] = True
            o["front_reliable"] = p["n_dirs"] == 1
            o["facing_source"] = "orient-anything"
            note = f"relative {to_relative(p['azimuth']):+.0f} deg"
        print(f"  object {o['id']}: {o['name'][:28]:28s} faces {o['yaw_deg']:5.1f} deg  ({note}, "
              f"conf {p['confidence']}, fronts {p['n_dirs']})", flush=True)
    json.dump(scene, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
