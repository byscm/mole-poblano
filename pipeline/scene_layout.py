# Where each 3D model goes in the room: reads a scene map (map.json) and a list of placements, writes the placed
# items (body centre on the floor plan, turn, target size, how sure we are) for build_scene.py (Blender).
# Usage: python -I scene_layout.py <map.json> <placements.json> <out_layout.json>
#   placements: [{...}, ...] or {"items": [{...}], "extra_views": [...], "shell": {...}, "walls": false (objects only)} with
#               [{"model": path.glb, "label": str,
#                  "at": object id | {"on": label} | {"pixel": [u, v, z]} | [x, y],
#                        map object (its body centre), on top of another item, the point of the photo pixel (u, v)
#                        at height z (the photo's ray cut at z: for things the map lacks, e.g. lamps), or a point
#                  "offset_m": [dx, dy] (optional, along the map axes),
#                  "yaw_deg": number (optional; default the object's facing), "yaw_add_deg": number (optional),
#                  "size_m": [width, depth, height] (stretch to exactly this) | "height_m": h (scale evenly to this height),
#                  "base_z_m": z (optional; default 0, or the top of the item it stands on), "hang_from_m": ceiling z (optional),
#                  "sure": "measured" | "map" | "assumed", "note": str}]
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_plan import footprint


def from_pixel(m, u, v, z):
    """Floor-plan point where the camera ray through photo pixel (u, v) is at height z."""
    ex, ey, ez = (np.array(a, float) for a in m["cv_axes_in_floor"])
    d = ex * (u - m["W"] / 2) / m["fx_px"] + ey * (v - m["H"] / 2) / m["fx_px"] + ez
    t = (z - m["camera_height_m"]) / d[2]
    return (t * d)[:2]


def main():
    m = json.load(open(sys.argv[1], encoding="utf-8"))
    places = json.load(open(sys.argv[2], encoding="utf-8"))
    views, shell, no_room = [], None, False
    if isinstance(places, dict):  # {"items": [...], "extra_views": [{name, map_folder, turn_deg, t_m}]}: other photos joined to the room
        no_room = places.get("walls") is False
        places, views, shell = places["items"], places.get("extra_views", []), places.get("shell")
    objs = {o["id"]: o for o in m["objects"]}
    done = {}
    out = []
    for p in places:
        at = p["at"]
        yaw = p.get("yaw_deg")
        base = p.get("base_z_m", 0.0)
        if isinstance(at, int):
            c, _, _, g = footprint(objs[at])
            xy = c
            if yaw is None:
                yaw = math.degrees(g)
        elif isinstance(at, dict) and "pixel" in at:
            xy = from_pixel(m, *at["pixel"])
        elif isinstance(at, dict):
            under = done[at["on"]]
            xy = np.array(under["xy_m"])
            base = p.get("base_z_m", under["base_z_m"] + under["top_m"])
        else:
            xy = np.array(at, float)
        xy = xy + np.array(p.get("offset_m", [0, 0]), float)
        yaw = (yaw or 0.0) + p.get("yaw_add_deg", 0.0)
        size = p.get("size_m")
        h = size[2] if size else p["height_m"]
        if "hang_from_m" in p:
            base = p["hang_from_m"] - h
        item = dict(label=p["label"], model=p["model"], xy_m=[round(float(v), 3) for v in xy], yaw_deg=round(yaw % 360, 1),
                    base_z_m=round(base, 3), top_m=h, size_m=size, height_m=None if size else h,
                    sure=p.get("sure", "map"), note=p.get("note", ""), from_object=at if isinstance(at, int) else None)
        done[p["label"]] = item
        out.append(item)
    walls = [] if no_room else [dict(id=w["id"], start_m=w["start_m"], end_m=w["end_m"], height_m=w["height_m"]) for w in m["room"]["walls"]]
    cams = [dict(name="camera A", xy_m=[0, 0], z_m=m.get("camera_height_m", 1.5))]
    if "fused_from" in m:
        cams.append(dict(name="camera B", xy_m=m["fused_from"]["camera_B_m"], z_m=m.get("camera_height_m", 1.5)))
    json.dump(dict(items=out, walls=walls, ceiling_m=m["room"].get("ceiling_height_m", 3.0),
                   floor_outline_m=[] if no_room else m["room"]["floor"].get("outline_m", []), cameras=cams, extra_views=views, shell=shell),
              open(sys.argv[3], "w", encoding="utf-8"), indent=1)
    for it in out:
        print(f"{it['label']:18s} at {it['xy_m']}  turn {it['yaw_deg']:6.1f}  base {it['base_z_m']:.2f}  [{it['sure']}]")


if __name__ == "__main__":
    main()
