# SAM 3 masks of every item class in every frame, for fit_to_photos.py.
# Usage: python -I photo_masks.py <inventory_map.json> <out_folder> <phrases.json> [ports=8188,8190] [scale=0.25]
#   phrases.json: {"bed": ["bed", 1], "bedside_table": ["nightstand", 2], ...}  key -> (SAM phrase, max instances)
# Writes <out_folder>/masks/<photo>.npz with one (n, h, w) bool array per key (n may be 0).
import json, os, sys, threading
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comfy_client as cc

MAP, OUT, PHR = sys.argv[1:4]
PORTS = (sys.argv[4] if len(sys.argv) > 4 else "8188,8190").split(",")
SCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
cams = json.load(open(MAP, encoding="utf-8"))["cameras"]
phrases = json.load(open(PHR, encoding="utf-8"))
os.makedirs(f"{OUT}/masks", exist_ok=True)


def work(port, todo):
    for c in todo:
        dst = f"{OUT}/masks/{c['photo']}.npz"
        if os.path.exists(dst): continue
        src = os.path.join(c["folder"], "photo.png"); im = Image.open(src); w, h = round(im.width * SCALE), round(im.height * SCALE)
        name = cc.upload_image(src, port); res = {}
        for key, (ph, n) in phrases.items():
            ms, err = cc.sam_masks(name, ph, f"fit/{c['photo']}", port, n)
            if err: print(c["photo"], key, "ERROR", err[:200], flush=True)
            res[key] = np.array([np.array(Image.fromarray(m).resize((w, h), Image.BILINEAR)) for m in ms], bool).reshape(-1, h, w)
        np.savez_compressed(dst, **res)
        print(port, c["photo"], {k: len(v) for k, v in res.items()}, flush=True)


ths = [threading.Thread(target=work, args=(p, cams[i::len(PORTS)])) for i, p in enumerate(PORTS)]
[t.start() for t in ths]; [t.join() for t in ths]
print("DONE")
