# For each object in the map: if the gallery has a VERY similar model (DINOv3 fingerprint), use that one
# (real, already verified size) instead of generating a new one with TRELLIS.
# Usage (.venv python): python gallery_match.py <map_folder> [threshold=0.85]
import json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprints import Fingerprints, to_square

GALLERY = os.environ.get("GALLERY_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gallery"))
SCENE = sys.argv[1]
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85

scene_map = json.load(open(os.path.join(SCENE, "map.json"), encoding="utf-8"))
fp = Fingerprints()
for o in scene_map["objects"]:
    rgba = np.array(Image.open(os.path.join(SCENE, o["crop"])).convert("RGBA"))
    top = fp.search(to_square(rgba[..., :3], rgba[..., 3] > 127), k=1)
    o.pop("model", None)
    if top:
        sim, stem = top[0][0], top[0][1]
        o["gallery_match"] = {"model": stem, "similarity": round(sim, 3)}
        if sim >= THRESHOLD and os.path.exists(os.path.join(GALLERY, stem + ".glb")):
            o["model"] = os.path.join(GALLERY, stem + ".glb")
            try:  # verified real height of the model (works as a scale reference)
                o["real_height_m"] = json.load(open(os.path.join(GALLERY, stem + ".json"), encoding="utf-8"))["dims_finales"][2]
            except Exception:
                pass
    print(f"object {o['id']} {o['name']:28s} gallery: {top[0][1] if top else '-':26s} {top[0][0] if top else 0:.3f}  "
          f"-> {'USES GALLERY' if o.get('model') else 'generated with TRELLIS'}", flush=True)
json.dump(scene_map, open(os.path.join(SCENE, "map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
