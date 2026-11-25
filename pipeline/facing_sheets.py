# Contact sheets of the benchmark crops (numbered) so a person can label which way each object faces.
# Usage: python -I facing_sheets.py
# Writes: escenas\_facing_sheets\sheet_N.png and index.json (number -> room, object id, name)
import json, os
from PIL import Image, ImageDraw, ImageFont

SCENES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "escenas")
OUT = os.path.join(SCENES, "_facing_sheets")
SYM = ("lampara", "mesa_redonda", "maceta", "planta", "taburete")
os.makedirs(OUT, exist_ok=True)
rows = [r for r in json.load(open(os.path.join(SCENES, "_facing_eval.json"))) if not r["cls"].startswith(SYM)]
items = []
for r in rows:
    m = json.load(open(os.path.join(SCENES, f"azar_{r['room']:03d}", "_facing_eval", "map.json"), encoding="utf-8"))
    o = next(o for o in m["objects"] if o["name"] == r["name"])
    items.append(dict(room=r["room"], id=o["id"], name=o["name"],
                      img=os.path.join(SCENES, f"azar_{r['room']:03d}", "_facing_eval", "_vlm_inputs", f"grey_{o['id']}.png")))
T, COLS, PER = 300, 4, 12
try:
    f = ImageFont.truetype("arialbd.ttf", 28)
except OSError:
    f = ImageFont.load_default()
for s in range(0, len(items), PER):
    part = items[s:s + PER]
    sheet = Image.new("RGB", (COLS * T, ((len(part) + COLS - 1) // COLS) * T), (255, 255, 255))
    for k, it in enumerate(part):
        im = Image.open(it["img"]).convert("RGB")
        im.thumbnail((T - 8, T - 8))
        x, y = (k % COLS) * T, (k // COLS) * T
        sheet.paste(im, (x + (T - im.width) // 2, y + (T - im.height) // 2))
        d = ImageDraw.Draw(sheet)
        d.rectangle([x + 2, y + 2, x + 52, y + 36], fill=(255, 255, 0))
        d.text((x + 6, y + 4), str(s + k), font=f, fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, f"sheet_{s // PER}.png"))
json.dump(items, open(os.path.join(OUT, "index.json"), "w"), indent=1)
print(len(items), "items")
