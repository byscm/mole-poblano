# How well generated models match REAL photo crops of the object, searching ALL turns (a model made from a front
# view must be allowed to turn round to match a photo taken from behind). Same silhouettes as silhouette.py.
# Usage: python -I compare_models.py <out.json> --model name=path.glb ... --crop label=scene:id ...
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silhouette import glb_vertices, fake_base, crop_silhouette, rotate, silhouette_view
from fit_poses import elevation


def best_iou(v, crop, elev):
    sr, _ = crop_silhouette(crop)
    best = (0, 0)
    for y in range(0, 360, 5):
        sm = silhouette_view(rotate(v, y, 0), elev)
        iou = float((sm & sr).sum() / max(1, (sm | sr).sum()))
        best = max(best, (iou, y))
    return best


def main():
    out = sys.argv[1]
    models, crops = {}, {}
    args = sys.argv[2:]
    for k, a in enumerate(args):
        if a == "--model":
            n, p = args[k + 1].split("=", 1); models[n] = p
        if a == "--crop":
            n, p = args[k + 1].split("=", 1); crops[n] = p
    res = {}
    for n, p in models.items():
        v = glb_vertices(p)
        if len(v) > 60000:
            v = v[np.random.default_rng(0).choice(len(v), 60000, replace=False)]
        cut = fake_base(v)
        if cut is not None:
            v = v[v[:, 2] >= cut]
        res[n] = {}
        for c, spec in crops.items():
            scene, oid = spec.rsplit(":", 1)
            m = json.load(open(os.path.join(scene, "map.json"), encoding="utf-8"))
            o = next(x for x in m["objects"] if x["id"] == int(oid))
            iou, yaw = best_iou(v, os.path.join(scene, o["crop"]), elevation(m, o))
            res[n][c] = dict(iou=round(iou, 3), yaw=yaw)
            print(f"{n:12s} vs {c:28s}: silhouette {iou:.2f} (turned {yaw} deg)", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
