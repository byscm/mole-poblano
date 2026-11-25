# Measure how well we know which way each object faces, on the synthetic benchmark rooms (azar_XXX),
# where the true facing is known. Compares: geometry only (current), two VLM questions, and hybrids.
# Usage: python -I eval_facing.py [port] [room ...]
# Writes per room: azar_XXX\_facing_eval\ (detect_objects output + vlm_facing.json cache)
import json, math, os, subprocess, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comfy_client import ask_vlm
from facing import QUESTIONS, parse_answer, relative_angle, angle_diff

SCENES = os.path.join(os.path.dirname(HERE), "escenas")
PY = sys.executable
SYMMETRIC = ("lampara", "mesa_redonda", "maceta", "planta", "taburete")


def main():
    args = sys.argv[1:]
    port = args[0] if args else "8188"
    rooms = [int(a) for a in args[1:]] or list(range(1, 13))
    rows = []
    for r in rooms:
        esc = os.path.join(SCENES, f"azar_{r:03d}")
        out = os.path.join(esc, "_facing_eval")
        truth = json.load(open(os.path.join(esc, "verdad.json"), encoding="utf-8"))
        if not os.path.exists(os.path.join(out, "map.json")):
            subprocess.run([PY, "-I", os.path.join(HERE, "detect_objects.py"), os.path.join(esc, "render.png"), out, port,
                            "--fov", str(truth["camara"]["fov_x_grados"])], check=True, stdout=subprocess.DEVNULL)
        m = json.load(open(os.path.join(out, "map.json"), encoding="utf-8"))
        masks = np.load(os.path.join(out, "object_masks.npz"))["masks"]
        inst = np.load(os.path.join(esc, "verdad_mapas.npz"))["instancias"]
        GH, GW = masks.shape[1:]
        inst = np.array(Image.fromarray(inst.astype(np.int32)).resize((GW, GH), Image.NEAREST))
        cache_path = os.path.join(out, "vlm_facing.json")
        cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
        cam = np.array(truth["camara"]["location"][:2])
        for o, mk in zip(m["objects"], masks):
            best, iou = None, 0.0
            for t in truth["objetos"]:
                ti = inst == t["instancia"]
                u = (ti | mk).sum()
                v = (ti & mk).sum() / u if u else 0
                if v > iou:
                    best, iou = t, v
            if best is None or iou < 0.3:
                continue
            g = math.radians(best["giro_mundo"])
            true_rel = relative_angle(cam - np.array(best["centro"][:2]), np.array([math.sin(g), -math.cos(g)]))
            y = math.radians(o["yaw_deg"])
            c = np.array(o["center_m"])
            geo_rel = relative_angle(-c, np.array([math.sin(y), -math.cos(y)]))
            grey = os.path.join(out, "_vlm_inputs", f"grey_{o['id']}.png")
            ans = {}
            for key, q in QUESTIONS.items():
                ck = f"{o['id']}:{key}"
                if ck not in cache:
                    cache[ck] = ask_vlm(grey, q.format(name=o["name"]), port, max_len=20)
                    json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=1)
                ans[key] = parse_answer(key, cache[ck])
            rows.append(dict(room=r, cls=best["clase"], name=o["name"], true=true_rel, geo=geo_rel,
                             reliable=o["front_reliable"], **ans, raw={k: cache[f"{o['id']}:{k}"] for k in QUESTIONS}))
        print(f"room {r}: {sum(1 for x in rows if x['room'] == r)} matched objects", flush=True)

    json.dump(rows, open(os.path.join(SCENES, "_facing_eval.json"), "w", encoding="utf-8"), indent=1, default=float)
    rows = [x for x in rows if not x["cls"].startswith(SYMMETRIC)]
    print(f"\n{len(rows)} objects with a real front (symmetric classes left out)")

    def report(label, preds):
        e = np.array([angle_diff(p, x["true"]) for p, x in zip(preds, rows) if p is not None])
        n_none = sum(p is None for p in preds)
        print(f"{label:34s} n={len(e):3d} (no answer {n_none:2d})  within 30: {np.mean(e <= 30):4.0%}  "
              f"within 60: {np.mean(e <= 60):4.0%}  backwards (>135): {np.mean(e > 135):4.0%}  median err {np.median(e):5.1f}")

    report("geometry (current)", [x["geo"] for x in rows])
    for key in QUESTIONS:
        report(f"vlm '{key}'", [x[key] for x in rows])
        # hybrid: elongated -> geometric axis, VLM only picks which of the 4 axis directions; else VLM
        hyb = []
        for x in rows:
            v = x[key]
            if v is None:
                hyb.append(x["geo"])
            elif x["reliable"]:
                cands = [x["geo"] + k * 90 for k in range(4)]
                hyb.append(min(cands, key=lambda c: angle_diff(c, v)))
            else:
                hyb.append(v)
        report(f"hybrid geometry + '{key}'", hyb)
    print("\nper class (vlm vs truth, first question):")
    k0 = list(QUESTIONS)[0]
    for cls in sorted({x["cls"] for x in rows}):
        sub = [x for x in rows if x["cls"] == cls]
        e = [angle_diff(x[k0], x["true"]) for x in sub if x[k0] is not None]
        print(f"  {cls:24s} n={len(sub)}  vlm median err {np.median(e) if e else float('nan'):5.1f}   "
              f"geo median err {np.median([angle_diff(x['geo'], x['true']) for x in sub]):5.1f}")


if __name__ == "__main__":
    main()
