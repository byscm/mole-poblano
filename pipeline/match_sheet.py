# Picture of the objects matched between two photos by their qualities: crop A | crop B, color chips, scores.
# Usage: python -I match_sheet.py <scene_A> <scene_B> <fused_folder> <out.png>   (reads fused_folder\map.json)
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from object_tags import PALETTE

fa, fb, ff, out = sys.argv[1:5]
mf = json.load(open(os.path.join(ff, "map.json"), encoding="utf-8"))
ma = {o["id"]: o for o in json.load(open(os.path.join(fa, "map.json"), encoding="utf-8"))["objects"]}
mb = {o["id"]: o for o in json.load(open(os.path.join(fb, "map.json"), encoding="utf-8"))["objects"]}
rows = [o for o in mf["objects"] if o.get("fused") and isinstance(o["fused"].get("matched_by"), dict)]
F = lambda n, b=False: ImageFont.truetype("arialbd.ttf" if b else "arial.ttf", n)
CS, RH, W = 170, 200, 1500
img = Image.new("RGB", (W, 90 + RH * len(rows)), (250, 250, 248)); d = ImageDraw.Draw(img)
d.text((24, 24), "Same object in two photos, matched by its qualities (colour, look, height, size) and where its foot is",
       font=F(24, True), fill=(30, 30, 30))


def crop(folder, o):
    p = os.path.join(folder, o.get("crop") or f"crops\\object_{o['id']}.png")
    im = Image.open(p).convert("RGB") if os.path.exists(p) else Image.new("RGB", (CS, CS), (200, 200, 200))
    im.thumbnail((CS, CS)); c = Image.new("RGB", (CS, CS), (235, 235, 232)); c.paste(im, ((CS - im.width) // 2, (CS - im.height) // 2))
    return c


def chips(x, y, o):
    for c in (o.get("tags") or {}).get("colors", []):
        w = int(160 * c["share"])
        d.rectangle([x, y, x + w, y + 22], fill=PALETTE[c["name"]], outline=(120, 120, 120))
        if w > 40:
            d.text((x + w / 2, y + 11), c["name"], font=F(12), anchor="mm",
                   fill=(255, 255, 255) if sum(PALETTE[c["name"]]) < 330 else (20, 20, 20))
        x += w


for r, o in enumerate(rows):
    y = 80 + r * RH
    f = o["fused"]; m = f["matched_by"]
    ia = o["id"]; ib = f["other_id"]
    A, B = ma[ia], mb[ib]
    img.paste(crop(fa, A), (24, y)); img.paste(crop(fb, B), (24 + CS + 16, y))
    x = 24 + 2 * CS + 50
    d.text((x, y + 2), f"photo 1 #{ia}  {A['name']}", font=F(17, True), fill=(30, 80, 200))
    chips(x, y + 28, A)
    d.text((x, y + 60), f"photo 2 #{ib}  {B['name']}", font=F(17, True), fill=(200, 60, 160))
    chips(x, y + 86, B)
    d.text((x, y + 122), f"height {A['height_m']:.2f} / {B['height_m']:.2f} m     feet {m['feet_apart_m']:.2f} m apart after aligning",
           font=F(15), fill=(60, 60, 60))
    x2 = 1080
    for k, (label, v) in enumerate((("look", m["look"]), ("colour", m["color"]), ("height", m["height"]), ("size", m["size"]),
                                    ("QUALITY", m["quality"]), ("place", m["place"]))):
        yy = y + 4 + k * 26
        d.text((x2, yy + 9), label, font=F(15, label == "QUALITY"), fill=(40, 40, 40), anchor="lm")
        d.rectangle([x2 + 90, yy, x2 + 90 + 260, yy + 18], fill=(230, 230, 226))
        d.rectangle([x2 + 90, yy, x2 + 90 + int(260 * v), yy + 18], fill=(60, 150, 90) if v >= 0.6 else (220, 160, 60) if v >= 0.3 else (210, 80, 70))
        d.text((x2 + 360, yy + 9), f"{v:.2f}", font=F(15), fill=(40, 40, 40), anchor="lm")
    d.line([(24, y + RH - 12), (W - 24, y + RH - 12)], fill=(225, 225, 222))
img.save(out)
print("saved", out, len(rows), "pairs")
