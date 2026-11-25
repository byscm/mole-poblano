# Fixes the scale of the map (when the photo has no EXIF, MoGe's scale usually comes out 5-25 % small).
#
# 1) With a MEASUREMENT YOU KNOW (the most reliable; it takes priority over everything else):
#      --measure "1-2=2.5"        from object 1 to object 2 there are 2.5 m (the distance shown on the plan)
#      --measure "3=1.80"         object 3 is 1.80 m tall
#      --measure "3:width=1.20"   object 3 is 1.20 m wide (also :depth, :height)
#      --measure "camera=1.60"    the photo was taken 1.60 m above the floor
#    Several, comma separated: "1-2=2.5,3=1.8" (the median is used).
# 2) Without --measure, automatic with references of known size: adult people (~1.70 m) and
#    gallery models (verified real size). Only whole objects (not cut by the image border).
#
# Usage: python -I fix_scale.py <map_folder> [--measure "..."]
# Writes: the corrected map.json (the original stays in map_unscaled.json; a rerun starts from it)
import json, os, re, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

PERSON_HEIGHT = 1.70
PERSON = re.compile(r"\b(person|people|man|men|woman|women|worker|boy|girl|child|kid)\b", re.I)
AUTO_RANGE = (0.6, 1.6)  # an automatic scale outside this is almost surely a bad reference


def from_measurements(scene_map, text):
    by_id = {o["id"]: o for o in scene_map["objects"]}
    dist = {}
    for p in scene_map["distances"]:
        dist[(p["a"], p["b"])] = dist[(p["b"], p["a"])] = p["distance_m"]
    refs = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        left, val = part.split("=")
        val = float(val.replace("m", "").strip())
        left = left.strip().lower()
        if left == "camera":
            refs.append(("camera measurement", 0, val / scene_map["camera_height_m"]))
        elif "-" in left:
            a, b = (int(x) for x in left.split("-"))
            if (a, b) not in dist:
                sys.exit(f"objects {a} and {b} are not in the map")
            refs.append((f"measurement {a}-{b}", a, val / dist[(a, b)]))
        else:
            i, _, field = left.partition(":")
            field = {"": "height", "height": "height", "width": "width", "depth": "depth"}[field.strip()]
            o = by_id.get(int(i))
            if o is None:
                sys.exit(f"object {i} is not in the map")
            refs.append((f"measurement {field} {i}", int(i), val / o[field + "_m"]))
    return refs


def automatic(scene_map):
    W, H = scene_map["W"], scene_map["H"]
    refs = []
    for o in scene_map["objects"]:
        x0, y0, x1, y1 = o["image_box"]
        whole = y0 > 3 and y1 < H - 3 and x0 > 3 and x1 < W - 3
        if not whole or o["height_m"] <= 0.05:
            continue
        if PERSON.search(o["name"]) or PERSON.search(o.get("type", "")):
            refs.append(("person", o["id"], PERSON_HEIGHT / o["height_m"]))
        elif o.get("model") and o.get("real_height_m"):
            refs.append(("gallery", o["id"], o["real_height_m"] / o["height_m"]))
    return refs


def scale_room(scene_map, s):
    """Walls, ceiling, floor and openings scale with the objects; wall distances are measured again."""
    room = scene_map["room"]
    pt = lambda v: [round(c * s, 3) for c in v]
    for key in ("walls", "walls_photo_only"):
        for w in room.get(key, []):
            w["start_m"], w["end_m"] = pt(w["start_m"]), pt(w["end_m"])
            for k in ("length_m", "height_m"):
                if w.get(k) is not None:
                    w[k] = round(w[k] * s, 2)
            w["assumed_parts_m"] = [[pt(a), pt(b)] for a, b in w.get("assumed_parts_m", [])]
    if room.get("ceiling_height_m"):
        room["ceiling_height_m"] = round(room["ceiling_height_m"] * s, 2)
    fl = room.get("floor") or {}
    if "outline_m" in fl:
        fl["outline_m"] = [pt(v) for v in fl["outline_m"]]
    if "visible_area_m2" in fl:
        fl["visible_area_m2"] = round(fl["visible_area_m2"] * s * s, 1)
    for o in room.get("openings", []):
        o["center_m"] = pt(o["center_m"])
        for k in ("width_m", "height_m", "bottom_m"):
            o[k] = round(o[k] * s, 2)
    for x in room.get("size_m", []):
        x["distance_m"] = round(x["distance_m"] * s, 2)
    from room_layout import measure_wall_distances
    measure_wall_distances(scene_map, room["walls"])


def main(extra_refs=()):
    """extra_refs: (label, id, ratio) measured elsewhere (e.g. known_sizes.py: a table top's height from the points)."""
    args = sys.argv[1:]
    measure = None
    if "--measure" in args:
        k = args.index("--measure"); measure = args[k + 1]; del args[k:k + 2]
    folder = args[0]
    path = os.path.join(folder, "map.json")
    original = os.path.join(folder, "map_unscaled.json")
    scene_map = json.load(open(path, encoding="utf-8"))
    if "scale_applied" in scene_map:
        if not measure:
            print("already corrected:", scene_map["scale_applied"])
            return
        # correct again from the original map (scales are not compounded)
        base = json.load(open(original, encoding="utf-8"))
        for o, ob in zip(scene_map["objects"], base["objects"]):  # keep what was added later (model, yaw...)
            for k in ("center_m", "base_z_m", "height_m", "width_m", "depth_m", "camera_distance_m"):
                o[k] = ob[k]
        scene_map["distances"] = base["distances"]
        scene_map["camera_height_m"] = base["camera_height_m"]
        if "room" in base:
            scene_map["room"] = base["room"]
        scene_map.pop("scale_applied")

    refs = (from_measurements(scene_map, measure) if measure else automatic(scene_map)) + list(extra_refs)
    source = "your measurement" if measure else "references"
    if not refs:
        print("no references of known size: the scale is not corrected (you can give one with --measure)")
        return
    s = float(np.median([r[2] for r in refs]))
    if not measure and abs(s - 1) < 0.03:
        print(f"scale already right ({s:.3f})")
        return
    if not measure and not AUTO_RANGE[0] <= s <= AUTO_RANGE[1]:
        print(f"automatic scale x{s:.2f} out of range: not applied (check the references or use --measure)")
        return
    if not os.path.exists(original):
        shutil.copy(path, original)
    for o in scene_map["objects"]:
        o["center_m"] = [round(c * s, 3) for c in o["center_m"]]
        for k in ("base_z_m", "height_m", "width_m", "depth_m", "camera_distance_m"):
            o[k] = round(o[k] * s, 3)
    for p in scene_map["distances"]:
        p["distance_m"] = round(p["distance_m"] * s, 3)
    scene_map["camera_height_m"] = round(scene_map["camera_height_m"] * s, 3)
    if scene_map.get("room"):
        scale_room(scene_map, s)
    scene_map["scale_applied"] = dict(factor=round(s, 3), source=source,
                                      references=[dict(kind=t, id=i, ratio=round(r, 3)) for t, i, r in refs])
    json.dump(scene_map, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"scale corrected x{s:.3f} with {source}: " + ", ".join(f"{t} ({r:.2f})" for t, i, r in refs))


if __name__ == "__main__":
    main()
