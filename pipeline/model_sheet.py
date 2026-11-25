# Review sheet: per object one row = the clean view it was made from + 4 renders of the model (render_views.py).
# Usage: python -I model_sheet.py <models_folder> <out.png> name=view.png ...
import os, sys
from PIL import Image, ImageDraw

folder, out = sys.argv[1], sys.argv[2]
rows = [a.split("=", 1) for a in sys.argv[3:]]
S = 240
sheet = Image.new("RGB", (S * 5 + 10, (S + 22) * len(rows)), "white"); d = ImageDraw.Draw(sheet)
for r, (name, view) in enumerate(rows):
    y = r * (S + 22)
    d.text((6, y + 4), name, fill="black")
    ims = [view] + [os.path.join(folder, "renders", f"{name}_y{a:03d}.png") for a in (0, 90, 180, 270)]
    for k, p in enumerate(ims):
        if os.path.exists(p):
            t = Image.open(p).convert("RGB"); t.thumbnail((S - 6, S - 6)); sheet.paste(t, (k * S + 6, y + 20))
        else:
            d.text((k * S + 40, y + 120), "missing", fill="red")
sheet.save(out)
print(out)
