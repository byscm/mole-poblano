# Qualities of each object, to recognise the same object in another photo (not only by where it is):
#   colors       dominant colors of its pixels (k-means in Lab), named, with their share
#   fingerprint  DINOv3 visual vector of its crop (shape / texture / material look), in object_tags.npz
#   size         floor sides sorted (big, small) and height
#   anchor_m     median floor position of the foot of its own 3D points (lowest 30 cm: pot, legs, base)
# Run with the ComfyUI .venv python (not -I), it needs torch for DINOv3:
#   ...\ComfyUI\ComfyUI\.venv\Scripts\python.exe object_tags.py <scene_folder> [more scene folders]
# Writes "tags" into each object of map.json and <scene>\object_tags.npz (ids, fingerprints)
import json, os, sys
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from room_layout import to_floor_points
from fingerprints import Fingerprints, to_square

# named colors (sRGB): what a person would say
PALETTE = {
    "black": (25, 25, 28), "dark grey": (75, 75, 78), "grey": (135, 135, 135), "light grey": (190, 190, 190),
    "white": (240, 240, 238), "beige": (215, 195, 160), "light wood": (190, 145, 95), "wood brown": (130, 85, 50),
    "dark brown": (70, 45, 30), "terracotta": (185, 100, 70), "red": (200, 40, 40), "orange": (235, 135, 40),
    "yellow": (230, 205, 60), "green": (70, 150, 60), "dark green": (35, 80, 40), "light blue": (135, 180, 230),
    "blue": (45, 85, 200), "purple": (130, 70, 170), "pink": (230, 140, 175), "silver": (165, 170, 178),
}


def srgb_to_lab(rgb):
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


PAL_NAMES = list(PALETTE)
PAL_LAB = srgb_to_lab(np.array([PALETTE[k] for k in PAL_NAMES]))


def kmeans(X, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
    return C, np.bincount(lab, minlength=k) / len(X)


def colors(rgb_pixels, k=3):
    if len(rgb_pixels) < 30:
        return []
    X = rgb_pixels[np.random.default_rng(0).choice(len(rgb_pixels), min(6000, len(rgb_pixels)), replace=False)]
    C, share = kmeans(srgb_to_lab(X), min(k, len(X)))
    out = []
    for c, s in sorted(zip(C, share), key=lambda x: -x[1]):
        d = np.linalg.norm((PAL_LAB - c) * np.array([0.6, 1, 1]), axis=1)  # lightness counts less (shade, sun)
        out.append(dict(name=PAL_NAMES[int(np.argmin(d))], share=round(float(s), 2), lab=[round(float(x), 1) for x in c]))
    merged = {}
    for c in out:  # same name twice -> one entry
        if c["name"] in merged:
            merged[c["name"]]["share"] = round(merged[c["name"]]["share"] + c["share"], 2)
        else:
            merged[c["name"]] = c
    return list(merged.values())


def main():
    fp = Fingerprints()
    for folder in [os.path.abspath(f) for f in sys.argv[1:]]:
        path = os.path.join(folder, "map.json")
        m = json.load(open(path, encoding="utf-8"))
        W, H = m["W"], m["H"]
        rgb = np.array(Image.open(os.path.join(folder, "photo.png")).convert("RGB"))
        P = np.load(os.path.join(folder, "moge_points.npz"))["points"].reshape(H, W, 3).astype(np.float64)
        if np.nanmedian(P[..., 2]) < 0:
            P = P * np.array([1, -1, -1])
        Q = to_floor_points(m, P)
        masks = np.load(os.path.join(folder, "object_masks.npz"))["masks"]
        crops, ids = [], []
        for idx, o in enumerate(m["objects"]):
            if idx >= len(masks):
                continue
            mk = Image.fromarray(masks[idx].astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
            full = np.array(mk) > 127
            inner = np.array(mk.filter(ImageFilter.MinFilter(7))) > 127  # away from the edges (no background bleed)
            if inner.sum() < 50:
                inner = full
            q = Q[inner & np.isfinite(Q).all(-1)]
            if len(q) > 30:  # the FOOT of the object (pot, legs, base): the same spot from any side, unlike branches
                q = q[q[:, 2] < np.percentile(q[:, 2], 3) + 0.3]
            dims = sorted([o["width_m"], o["depth_m"]], reverse=True)
            o["tags"] = dict(colors=colors(rgb[inner]), floor_sides_m=[round(x, 2) for x in dims], height_m=o["height_m"],
                             anchor_m=[round(float(x), 3) for x in np.median(q[:, :2], 0)] if len(q) else None,
                             anchor_scale=m.get("scale_applied", {}).get("factor", 1.0), fingerprint_row=len(ids))
            crops.append(to_square(rgb, full)); ids.append(o["id"])
        emb = fp.vectors(crops)
        np.savez(os.path.join(folder, "object_tags.npz"), ids=np.array(ids), emb=emb)
        json.dump(m, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{os.path.basename(folder)}: {len(ids)} objects tagged")
        for o in m["objects"]:
            t = o.get("tags")
            if t:
                print(f"  #{o['id']:2d} {o['name'][:28]:28s} " + ", ".join(f"{c['name']} {c['share']*100:.0f}%" for c in t["colors"]))


if __name__ == "__main__":
    main()
