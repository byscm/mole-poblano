# One map from MANY photos of the same space: every photo is aligned to one reference photo with fuse_photos.py
# (pair by pair), then all objects are put in the reference frame. The same object seen in several photos is kept
# once, from the photo where it was closest to the camera (best depth, biggest crop). Each object remembers its
# photo ("source_folder") so its crop can be found.
# Usage: python -I merge_photos.py <out_map.json> <alignment.json>
#   alignment.json from vggt_align.py (or align_photos.py): per photo {folder, turn_deg, t_m, depth_scale}; the first
#   photo is the reference. depth_scale: how much bigger that photo's own map came out than the others (its objects
#   are shrunk by it about its camera, so all photos share one scale).
# Writes <out_map.json> (objects with body_center_m, yaw, size, crop, source_folder, seen_in) and prints the merges.
import json, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_plan import footprint

SAME_M = 0.55       # body centres closer than this (after alignment) ...
SAME_HEIGHT = 0.35  # ... and heights within this share of each other -> the same object
FLOOR_WORDS = re.compile(r"chair|sofa|couch|armchair|recliner|stool|table|bench|cabinet|sideboard|credenza|island|console|staircase|stairs", re.I)


def objects_in_ref(folder, turn_deg, t, k=1.0):
    m = json.load(open(os.path.join(folder, "map.json"), encoding="utf-8"))
    th = math.radians(turn_deg); R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = []
    for o in m["objects"]:
        c, _, _, g = footprint(o)  # body centre in this photo's frame
        o = dict(o)
        c = c * k
        for key in ("width_m", "depth_m", "height_m", "base_z_m"):
            if key in o: o[key] = round(o[key] * k, 3)
        o["camera_distance_m"] = float(np.linalg.norm(c))
        o["body_center_m"] = [round(float(v), 3) for v in (R @ c + t)]
        yaw = (o.get("yaw_from_appearance", o["yaw_deg"]) + turn_deg) % 360
        o["yaw_deg"] = o["yaw_from_appearance"] = round(yaw, 1)
        o["center_m"] = o["body_center_m"]  # (kept for tools that read center_m; body_center_m wins in the preview)
        # furniture that stands on the floor but was only seen from its top part (a chair back behind the table, a
        # table top): the box goes down to the floor; the picture card stays where the photo saw it
        o["card_base_z_m"], o["card_height_m"] = o.get("base_z_m", 0.0), o["height_m"]
        if FLOOR_WORDS.search(o["name"]) and 0.08 < o.get("base_z_m", 0.0) < 1.3:
            o["height_m"] = round(o["base_z_m"] + o["height_m"], 3); o["base_z_m"] = 0.0; o["extended_to_floor"] = True
        o["source_folder"] = os.path.abspath(folder)
        o["photo"] = os.path.basename(os.path.abspath(folder))
        out.append(o)
    return m, out


def main():
    out_path = sys.argv[1]
    al = json.load(open(sys.argv[2], encoding="utf-8"))
    allobjs, cams, mref = [], [], None
    for name, v in al["photos"].items():
        k = 1.0 / v.get("depth_scale", 1.0)
        m, ob = objects_in_ref(v["folder"], v["turn_deg"], np.array(v["t_m"], float), k)
        mref = mref or m
        allobjs += ob
        cams.append(dict(photo=name, folder=v["folder"], xy_m=v["t_m"], turn_deg=v["turn_deg"], shrink=round(k, 4)))
    # closest-to-camera first: it wins its group
    allobjs.sort(key=lambda o: o["camera_distance_m"])
    kept = []
    for o in allobjs:
        p = np.array(o["body_center_m"]); h = o["height_m"]
        twin = None
        for k in kept:
            if (np.linalg.norm(np.array(k["body_center_m"]) - p) < SAME_M and abs(k["height_m"] - h) <= SAME_HEIGHT * max(h, k["height_m"])
                    and abs(k.get("base_z_m", 0) - o.get("base_z_m", 0)) < 0.3):
                twin = k; break
        if twin:
            twin["seen_in"].append(f"{o['photo']}#{o['id']}")
        else:
            o = dict(o); o["seen_in"] = [f"{o['photo']}#{o['id']}"]
            kept.append(o)
    for i, o in enumerate(kept, 1):
        o["id_in_photo"] = o["id"]; o["id"] = i
    m = {k: v for k, v in mref.items() if k not in ("objects", "distances")}
    m.update(objects=kept, distances=[], cameras=cams, merged_from=[c["photo"] for c in cams])
    json.dump(m, open(out_path, "w", encoding="utf-8"), indent=1)
    for o in kept:
        print(f"{o['id']:3d} {o['name'][:32]:32s} h {o['height_m']:.2f}  at {o['body_center_m']}  from {o['photo']}  seen {len(o['seen_in'])}x")
    print(f"{len(kept)} objects from {len(allobjs)} detections in {len(cams)} photos")


if __name__ == "__main__":
    main()
