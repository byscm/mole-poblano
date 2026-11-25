# One room plan from two: our walls (measured from the photo) + SpatialLM's (run_spatiallm.py).
#  - our wall with a SpatialLM wall on the same line  -> "photo + SpatialLM" (confirmed), stretched to SpatialLM's
#    length where SpatialLM saw more (that stretch is marked as assumed)
#  - SpatialLM wall with no match                      -> "SpatialLM" (not seen in the photo: assumed), unless
#    the photo sees things behind it (then it cannot be a wall and is dropped)
#  - SpatialLM doors and windows are kept on whichever wall they ended up on
#  - objects stay ours; SpatialLM's boxes are only stored for reference
# Usage: python -I merge_layouts.py <scene_folder>   (needs map.json with "room" and spatiallm.json)
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from room_layout import measure_wall_distances, to_floor_points

MATCH_DEG = 10       # same direction
MATCH_OFFSET = 0.45  # m between the two lines
MAX_STRETCH = 4.0    # m: a confirmed wall is not stretched further than this at either end
CORNER_GAP = 0.8     # m: wall ends this close to a shared corner are joined
SEE_THROUGH = 150    # visible points behind a SpatialLM-only wall that make it impossible


def seg(w):
    a, b = np.array(w["start_m"][:2], float), np.array(w["end_m"][:2], float)
    L = float(np.linalg.norm(b - a))
    return a, b, (b - a) / max(L, 1e-9), L


def crosses(a, b, pts, beyond=0.25):
    """How many points are seen THROUGH segment a-b from the camera at (0,0) (and lie > beyond m behind it)."""
    v = b - a
    n = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-9)
    side_cam = (np.zeros(2) - a) @ n
    side_p = (pts - a) @ n
    behind = (np.sign(side_p) != np.sign(side_cam)) & (np.abs(side_p) > beyond)
    p = pts[behind]
    if not len(p):
        return 0
    # intersection of the ray 0 -> p with the line through a, b: t along the wall
    den = p[:, 0] * v[1] - p[:, 1] * v[0]
    ok = np.abs(den) > 1e-9
    t = -(p[ok, 0] * a[1] - p[ok, 1] * a[0]) / den[ok]
    return int(((t > 0.05) & (t < 0.95)).sum())


def main():
    folder = os.path.abspath(sys.argv[1])
    path = os.path.join(folder, "map.json")
    m = json.load(open(path, encoding="utf-8"))
    s = json.load(open(os.path.join(folder, "spatiallm.json")))
    room = m["room"]
    photo_walls = room.get("walls_photo_only", room["walls"])  # rerunning starts again from the photo walls
    walls = [dict(w, source="photo", confirmed_by=[]) for w in json.loads(json.dumps(photo_walls))]
    theirs = [w for w in s["walls"] if seg(w)[3] > 0.05]

    # visible points (to reject SpatialLM walls the photo sees through)
    W, H = m["W"], m["H"]
    P = np.load(os.path.join(folder, "moge_points.npz"))["points"].reshape(H, W, 3).astype(np.float64)
    if np.nanmedian(P[..., 2]) < 0:
        P = P * np.array([1, -1, -1])
    Q = to_floor_points(m, P).reshape(-1, 3)
    Q = Q[np.isfinite(Q).all(1) & (Q[:, 2] > -0.1) & (Q[:, 2] < 2.2) & (np.linalg.norm(Q[:, :2], axis=1) < 12)]
    vis = Q[np.random.default_rng(0).choice(len(Q), min(20000, len(Q)), replace=False), :2]
    centre = np.median(vis, 0)  # a point inside the room, to tell which side of a wall is the room

    s_to_ours = {}
    for t in theirs:
        ta, tb, tv, tL = seg(t)
        best = None
        for w in walls:
            a, b, v, L = seg(w)
            ang = math.degrees(math.acos(min(1.0, abs(v @ tv))))
            if ang > MATCH_DEG:
                continue
            n = np.array([-v[1], v[0]])
            off = abs((((ta + tb) / 2) - a) @ n)
            t0, t1 = sorted(((ta - a) @ v, (tb - a) @ v))
            overlap = min(L, t1) - max(0.0, t0)
            if off < MATCH_OFFSET and overlap > -0.5 and (best is None or off < best[0]):
                best = (off, w, t0, t1)
        if best:
            off, w, t0, t1 = best
            s_to_ours[t["id"]] = w["id"]
            w["confirmed_by"].append(dict(spatiallm=t["id"], offset_m=round(off, 2)))
            w["source"] = "photo + SpatialLM"
            a, b, v, L = seg(w)
            new_a = a + v * max(t0, -MAX_STRETCH) if t0 < -0.3 else a
            new_b = a + v * min(t1, L + MAX_STRETCH) if t1 > L + 0.3 else b
            for old, new in ((a, new_a), (b, new_b)):
                if np.linalg.norm(new - old) > 0.3:
                    w.setdefault("assumed_parts_m", []).append([[round(float(x), 3) for x in old], [round(float(x), 3) for x in new]])
            w["start_m"], w["end_m"] = [round(float(x), 3) for x in new_a], [round(float(x), 3) for x in new_b]
            w["length_m"] = round(float(np.linalg.norm(new_b - new_a)), 2)

    added, dropped = [], []
    for t in theirs:
        if t["id"] in s_to_ours:
            continue
        ta, tb, tv, tL = seg(t)
        through = crosses(ta, tb, vis)
        if through > SEE_THROUGH:
            dropped.append(dict(id=t["id"], length_m=round(tL, 2), points_seen_behind=through))
            continue
        n = np.array([-tv[1], tv[0]])
        if (centre - ta) @ n < 0:
            n = -n
        added.append(dict(id=t["id"], start_m=[round(float(x), 3) for x in ta], end_m=[round(float(x), 3) for x in tb],
                          length_m=round(tL, 2), height_m=room.get("ceiling_height_m") or round(t["height_m"], 2),
                          height_source="ceiling" if room.get("ceiling_height_m") else "SpatialLM",
                          room_side=[round(float(x), 3) for x in n], source="SpatialLM", assumed_parts_m=[], confirmed_by=[]))
        s_to_ours[t["id"]] = t["id"]
    walls += added

    # close corners between the merged walls (ends within CORNER_GAP of where two walls meet)
    for i, w1 in enumerate(walls):
        for w2 in walls[i + 1:]:
            a1, b1, v1, L1 = seg(w1); a2, b2, v2, L2 = seg(w2)
            den = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(den) < 0.5:  # roughly parallel
                continue
            u = ((a2[0] - a1[0]) * v2[1] - (a2[1] - a1[1]) * v2[0]) / den
            x = a1 + u * v1
            e1 = min(("start_m", "end_m"), key=lambda e: np.linalg.norm(np.array(w1[e]) - x))
            e2 = min(("start_m", "end_m"), key=lambda e: np.linalg.norm(np.array(w2[e]) - x))
            if np.linalg.norm(np.array(w1[e1]) - x) < CORNER_GAP and np.linalg.norm(np.array(w2[e2]) - x) < CORNER_GAP:
                for w, e in ((w1, e1), (w2, e2)):
                    w[e] = [round(float(c), 3) for c in x]
                    w["length_m"] = round(float(np.linalg.norm(np.subtract(w["end_m"], w["start_m"]))), 2)

    openings = []
    for kind, items in (("door", s["doors"]), ("window", s["windows"])):
        for o in items:
            if o["wall"] not in s_to_ours:
                continue  # its wall was dropped
            cz, h = o["center_m"][2], o["height_m"]
            top = room.get("ceiling_height_m")
            if top and cz + h / 2 > top:  # an opening cannot go above the ceiling we measured
                h = max(0.1, top - max(0.0, cz - h / 2)); cz = top - h / 2
            openings.append(dict(kind=kind, wall=s_to_ours[o["wall"]], center_m=[round(float(x), 3) for x in o["center_m"][:2]],
                                 width_m=round(o["width_m"], 2), height_m=round(h, 2), bottom_m=round(max(0.0, cz - h / 2), 2),
                                 source="SpatialLM"))

    # room size: distance between facing parallel walls that overlap
    sizes = []
    for i, w1 in enumerate(walls):
        for w2 in walls[i + 1:]:
            a1, b1, v1, L1 = seg(w1); a2, b2, v2, L2 = seg(w2)
            if abs(v1 @ v2) < math.cos(math.radians(MATCH_DEG)) or np.array(w1["room_side"]) @ np.array(w2["room_side"]) > -0.9:
                continue
            t0, t1 = sorted(((a2 - a1) @ v1, (b2 - a1) @ v1))
            if min(L1, t1) - max(0, t0) < 0.5 or min(L1, L2) < 1.0:
                continue
            gap = abs((a2 - a1) @ np.array(w1["room_side"]))
            sizes.append(dict(between=[w1["id"], w2["id"]], distance_m=round(float(gap), 2)))
    sizes.sort(key=lambda x: -x["distance_m"])

    room["walls_photo_only"] = photo_walls
    room["walls"] = walls
    room["openings"] = openings
    room["size_m"] = sizes[:3]
    room["merged_with"] = dict(model=s["source"], dropped_walls=dropped,
                               spatiallm_ceiling_m=round(theirs[0]["height_m"], 2) if theirs else None,
                               spatiallm_boxes=s["boxes"])
    measure_wall_distances(m, walls)
    json.dump(m, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for w in walls:
        conf = ", ".join(f"{c['spatiallm']} {c['offset_m']:.2f} m off" for c in w["confirmed_by"])
        print(f"  {w['id']:4s} {w['length_m']:5.2f} m  {w['source']}" + (f"  ({conf})" if conf else ""))
    for d in dropped:
        print(f"  dropped SpatialLM {d['id']} ({d['length_m']} m): the photo sees {d['points_seen_behind']} points behind it")
    for o in openings:
        print(f"  {o['kind']} on {o['wall']}: {o['width_m']:.2f} x {o['height_m']:.2f} m, {o['bottom_m']:.2f} m above the floor")
    for x in sizes[:3]:
        print(f"  room: {x['distance_m']:.2f} m between {x['between'][0]} and {x['between'][1]}")


if __name__ == "__main__":
    main()
