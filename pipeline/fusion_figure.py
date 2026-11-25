# Side-by-side picture: one photo alone vs two photos fused, only the objects of a measured gap.
# Usage: python -I fusion_figure.py <scene_A> <fused_folder> <idA_1> <idA_2> <idF_1> <idF_2> <real_m> <out.png>
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_plan import footprint
import known_sizes as K

fa, ff = sys.argv[1], sys.argv[2]
ia1, ia2, if1, if2 = (int(x) for x in sys.argv[3:7])
real, out = float(sys.argv[7]), sys.argv[8]
ma = json.load(open(os.path.join(fa, "map.json"), encoding="utf-8"))
mf = json.load(open(os.path.join(ff, "map.json"), encoding="utf-8"))
F = lambda s: ImageFont.truetype("arialbd.ttf" if "b" in s else "arial.ttf", int(s.strip("b")))
W, H, S = 760, 900, 95
img = Image.new("RGB", (2 * W, H + 120), (250, 250, 248)); d = ImageDraw.Draw(img)
cols = [(70, 140, 220), (230, 120, 60)]


def panel(x0, m, ids, title, cams):
    objs = [next(o for o in m["objects"] if o["id"] == i) for i in ids]
    pts = np.vstack([np.array(footprint(o)[1]) for o in objs] + [np.array(c[0])[None] for c in cams])
    mid = (pts.min(0) + pts.max(0)) / 2
    P = lambda p: (x0 + W / 2 + (p[0] - mid[0]) * S, 90 + H / 2 - (p[1] - mid[1]) * S)
    for gx in np.arange(-10, 10, 0.5):  # 50 cm grid, inside the panel only
        x = P((mid[0] + gx, 0))[0]
        if x0 + 10 < x < x0 + W - 10:
            d.line([(x, 80), (x, H + 40)], fill=(232, 232, 228))
        y = P((0, mid[1] + gx))[1]
        if 80 < y < H + 40:
            d.line([(x0 + 10, y), (x0 + W - 10, y)], fill=(232, 232, 228))
    for o, c in zip(objs, cols):
        d.polygon([P(q) for q in footprint(o)[1]], fill=tuple(int(v * 0.35 + 255 * 0.65) for v in c), outline=c, width=3)
        cx, cy = P(footprint(o)[0])
        d.text((cx, cy), o["name"].split(" (")[0], font=F("18b"), fill=c, anchor="mm")
    A, B = (np.array(footprint(o)[1]) for o in objs)
    best = None
    for P1, P2 in ((A, B), (B, A)):
        for p in P1:
            for k in range(4):
                a, b = P2[k], P2[(k + 1) % 4]; v = b - a; t = np.clip((p - a) @ v / (v @ v), 0, 1); q = a + t * v
                dd = np.linalg.norm(p - q)
                if best is None or dd < best[0]:
                    best = (dd, p, q)
    gap = K.edge_gap(A, B)
    d.line([P(best[1]), P(best[2])], fill=(20, 20, 20), width=4)
    mx, my = np.mean([P(best[1]), P(best[2])], 0)
    d.rounded_rectangle([mx + 12, my - 18, mx + 150, my + 18], 6, fill="white", outline=(20, 20, 20), width=2)
    d.text((mx + 81, my), f"{gap:.2f} m", font=F("24b"), fill=(20, 20, 20), anchor="mm")
    for c, name, col in cams:
        x, y = P(c); d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col); d.text((x + 14, y), name, font=F("17"), fill=col, anchor="lm")
    d.text((x0 + W - 30, H + 40), "grid 50 cm", font=F("15"), fill=(150, 150, 150), anchor="rm")
    d.text((x0 + 30, 30), title, font=F("26b"), fill=(30, 30, 30))
    err = (gap - real) / real * 100
    d.text((x0 + 30, H + 70), f"real (tape): {real:.2f} m     map: {gap:.2f} m     error {err:+.0f} %", font=F("22"), fill=(30, 30, 30))


panel(0, ma, (ia1, ia2), "One photo (room_photo2)", [((0, 0), "camera", (30, 80, 200))])
fb = mf["fused_from"]
panel(W, mf, (if1, if2), "Two photos fused", [((0, 0), "camera 1", (30, 80, 200)), (fb["camera_B_m"], "camera 2", (200, 60, 160))])
d.line([(W, 20), (W, H + 110)], fill=(200, 200, 200), width=2)
img.save(out)
print("saved", out)
