# Scores renders against the real photos, same viewpoint, same size: PSNR (pixel error, higher = closer) and SSIM
# (structure: edges and shapes, 0..1, higher = closer), plus a sheet photo | method A | method B ... with the scores.
# Usage: python -I score_views.py <out_folder> <photos_folder> "<label>=<renders_folder>[|<prefix>]" ... -- <photo,photo,...>
#   renders are found as <renders_folder>/<prefix><photo>.png; photos as <photos_folder>/<photo>.png
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

a = sys.argv[1:]; cut = a.index("--"); PHOTOS = a[cut + 1].split(","); OUT, SRC = os.path.abspath(a[0]), os.path.abspath(a[1])
METHODS = []
for s in a[2:cut]:
    label, rest = s.split("=", 1); folder, _, prefix = rest.partition("|")
    METHODS.append((label, os.path.abspath(folder), prefix))
os.makedirs(OUT, exist_ok=True)
W, H = 358, 640


def load(p):
    return np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS), np.float64) / 255


def psnr(x, y):
    return float(10 * np.log10(1 / max(np.mean((x - y) ** 2), 1e-10)))


def blur(img, s=1.5):
    r = int(3 * s); k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    img = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, img)
    return np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, img)


def ssim(x, y):
    x, y = x.mean(2), y.mean(2); c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = blur(x), blur(y); sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return float(np.mean(((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))))


rows, scores = [], {m[0]: {} for m in METHODS}
f = ImageFont.truetype("arialbd.ttf", 22); fs = ImageFont.truetype("arial.ttf", 20)
for ph in PHOTOS:
    ref = load(os.path.join(SRC, ph + ".png")); tiles = [("Foto real", ref, "")]
    for label, folder, prefix in METHODS:
        im = load(os.path.join(folder, prefix + ph + ".png")); p, s = psnr(ref, im), ssim(ref, im)
        scores[label][ph] = dict(psnr=round(p, 2), ssim=round(s, 3)); tiles.append((label, im, f"SSIM {s:.2f}  PSNR {p:.1f}"))
    row = Image.new("RGB", (len(tiles) * (W + 8), H + 64), "white"); d = ImageDraw.Draw(row)
    for i, (lb, im, sc) in enumerate(tiles):
        row.paste(Image.fromarray((im * 255).astype(np.uint8)), (i * (W + 8), 32)); d.text((i * (W + 8) + 4, 4), lb, fill="black", font=f)
        d.text((i * (W + 8) + 4, H + 36), sc, fill=(160, 20, 20), font=fs)
    rows.append(row)
sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white"); y = 0
for r in rows: sheet.paste(r, (0, y)); y += r.height
sheet.save(os.path.join(OUT, "comparacion.jpg"), quality=88)
summary = {lb: dict(ssim=round(float(np.mean([v["ssim"] for v in d.values()])), 3), psnr=round(float(np.mean([v["psnr"] for v in d.values()])), 2)) for lb, d in scores.items()}
json.dump(dict(per_photo=scores, mean=summary), open(os.path.join(OUT, "scores.json"), "w"), indent=1)
print(json.dumps(summary))
