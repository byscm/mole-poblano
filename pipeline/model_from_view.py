# 3D model (TRELLIS 2, same workflow as generate_models.py) from a clean view on a solid background (clean_view.py).
# The background is keyed out by its colour (taken from the image border), then TRELLIS runs up to 3 seeds; each
# result is scored with the SAME silhouette check as generate_models.py against a real photo crop of the object.
# Usage: python -I model_from_view.py <clean_view.png> <scene> <object_id> <out.glb> [port]
#   scene and object_id may be '-' (no detected object to check against: one seed, no score)
import glob, json, os, shutil, sys, uuid
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import upload_image
from generate_models import generate_one, SEEDS, GOOD
from silhouette import fit_pose
from fit_poses import elevation

KEY = 60  # colour distance from the background that is still background


def key_out(path):
    rgb = np.array(Image.open(path).convert("RGB")).astype(float)
    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3), rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    bg = np.median(border, 0)
    d = np.linalg.norm(rgb - bg, axis=-1)
    g = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])  # green spill / soft green shadows count as background too
    fg = (d > KEY) & ~((g > 40) & (bg[1] > max(bg[0], bg[2])))
    m = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    # keep the biggest blob only
    from scipy import ndimage
    lab, n = ndimage.label(np.array(m) > 127)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = 1 + int(np.argmax(sizes))
        m = Image.fromarray(((lab == keep) | ndimage.binary_fill_holes(lab == keep) * 0).astype(np.uint8) * 255)
    return rgb.astype(np.uint8), np.array(m)


def main():
    src, scene, oid, out = sys.argv[1], sys.argv[2], int(sys.argv[3]) if sys.argv[3] != "-" else None, sys.argv[4]
    port = sys.argv[5] if len(sys.argv) > 5 else "8188"
    check = scene != "-"  # "-": no real crop to check against (object not detected): one seed, no silhouette score
    if check:
        m = json.load(open(os.path.join(scene, "map.json"), encoding="utf-8"))
        o = next(x for x in m["objects"] if x["id"] == oid)
    rgb, mask = key_out(src)
    base = os.path.splitext(out)[0]
    Image.fromarray(rgb).save(base + "_in.png"); Image.fromarray(mask).save(base + "_mask.png")
    cut = np.dstack([rgb, mask]); Image.fromarray(cut).save(base + "_cutout.png")
    n_img, n_msk = upload_image(base + "_in.png", port), upload_image(base + "_mask.png", port)
    tag = uuid.uuid4().hex[:6]
    attempts = []
    for seed in SEEDS:
        glb = generate_one(n_img, n_msk, f"generated/clean_view_obj{oid}_{tag}_s{seed}", seed, port)
        if not check:
            attempts.append((0.0, seed, glb, {}))
            print(f"seed {seed}: done (no check)", flush=True)
            break
        q = fit_pose(glb, os.path.join(scene, o["crop"]), elevation(m, o))
        score = q["iou"] - (0.5 if q["relative_thickness"] < 0.02 else 0)
        attempts.append((score, seed, glb, q))
        print(f"seed {seed}: silhouette {q['iou']:.2f}  thickness {q['relative_thickness']:.3f}", flush=True)
        if score >= GOOD:
            break
    score, seed, glb, q = max(attempts, key=lambda t: t[0])
    shutil.copy(glb, out)
    json.dump(dict(seed=seed, source=src, **q, attempts=[(t[1], t[3]) for t in attempts]),
              open(base + "_quality.json", "w"), indent=1, default=float)
    print(f"-> keeping seed {seed}" + (f" (silhouette {q['iou']:.2f})" if q else ""), flush=True)


if __name__ == "__main__":
    main()
