# Tileable surface textures (floor, walls...) made by Qwen-Image 2.1 from photo crops of the real surface, seen
# straight on with flat light, then evened out (no dark corners) and made seamless by cross-fading with a half-shifted copy.
# Usage: python -I texture_gen.py <jobs.json> [port]
#   job: {"name", "out": folder, "refs": [png...], "what": "warm brown wood-look floor planks ...",
#         "area_m": [w, h] (real size the image covers), "w": px, "h": px, "seeds": [1, 2]}
# Writes <out>/<name>_s<seed>_raw.png (as generated) and <out>/<name>_s<seed>.png (seamless)
import json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import upload_image
from clean_view import workflow, run


def flatten(im, sigma_frac=0.12):
    """Remove slow light changes (dark corners, gradients): divide by a strong blur, keep the mean colour."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(im, float)
    s = sigma_frac * max(a.shape[:2])
    low = np.stack([gaussian_filter(a[..., c], s, mode="wrap") for c in range(3)], -1)
    out = a / np.maximum(low, 1) * a.reshape(-1, 3).mean(0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def seamless(im, frac=0.25):
    """Cross-fade with the image shifted by half, so opposite edges continue into each other (ghosting only near the
    middle band of width frac)."""
    a = np.asarray(im, float)
    h, w = a.shape[:2]
    r = np.roll(a, (h // 2, w // 2), (0, 1))

    def ramp(n):
        x = np.abs(np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)  # 0 in the middle, 1 at the edges
        return np.clip((x - (1 - 2 * frac)) / (2 * frac), 0, 1)  # 1 near the edges -> use the shifted copy there
    m = np.maximum.outer(ramp(h), ramp(w))[..., None]
    return Image.fromarray((a * (1 - m) + r * m).astype(np.uint8))


def main():
    jobs = json.load(open(sys.argv[1], encoding="utf-8"))
    port = sys.argv[2] if len(sys.argv) > 2 else "8188"
    url = f"http://127.0.0.1:{port}"
    for job in jobs:
        os.makedirs(job["out"], exist_ok=True)
        names = [upload_image(p, port) for p in job["refs"]]
        aw, ah = job["area_m"]
        prompt = (f"The {len(names)} reference images are photos of the same real surface: {job['what']}. "
                  f"Create a flat seamless texture of exactly this surface material, seen perfectly straight on "
                  f"(orthographic, no perspective), covering an area of {aw} m by {ah} m at real scale. "
                  f"Fill the whole image edge to edge with the material only. Same colors, tone and details as in the photos. "
                  f"Even flat diffuse light, no shadows, no reflections, no highlights, no vignetting. {job.get('extra', '')}")
        neg = "furniture, chair legs, objects, people, shadow, perspective, vanishing point, reflection, text, watermark, border, frame, blurry"
        open(os.path.join(job["out"], f"{job['name']}_prompt.txt"), "w", encoding="utf-8").write(prompt + "\n\nNEGATIVE: " + neg)
        for seed in job.get("seeds", [1, 2]):
            raw = os.path.join(job["out"], f"{job['name']}_s{seed}_raw.png")
            if not os.path.exists(raw):
                t0 = time.time()
                open(raw, "wb").write(run(url, workflow(prompt, neg, names, seed, job.get("w", 1024), job.get("h", 1024))))
                print(f"{os.path.basename(raw)}: {time.time() - t0:.0f} s", flush=True)
            seamless(flatten(Image.open(raw).convert("RGB"))).save(os.path.join(job["out"], f"{job['name']}_s{seed}.png"))


if __name__ == "__main__":
    main()
