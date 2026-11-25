# Room shell from the same photo: walls (as lines on the floor plan, with length and height), the ceiling
# height, the visible floor area, and the distance from every object to its nearest walls.
# Uses SAM 3 masks for "wall", "ceiling" and "floor" plus the MoGe-2 points already in the scene folder.
# Usage: python -I room_layout.py <scene_folder> [port]
# Writes into map.json: "room" {walls, ceiling_height_m, floor} and per object "wall_distances".
# Also writes room_masks.png (photo with the wall / ceiling / floor pixels colored, for checking).
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import DEFAULT_PORT, upload_image, sam_masks
from draw_plan import footprint, font

WALL_TOL = 0.05      # m: a point belongs to a wall line if it is this close (plus 1 % of its distance)
MIN_WALL_PTS = 400   # full-resolution pixels needed to accept a wall
SNAP_DEG = 12        # walls within this angle of the main direction (or 90 deg from it) are snapped to it
CORNER_GAP = 0.8     # m: wall ends this close to where two walls would meet are joined into a corner
LONG_WALL = 2.0      # m: two walls this long that point at each other are extended to meet...
MAX_EXTEND = 4.0     # m: ...if the missing part is shorter than this (it is drawn dashed: not seen, assumed)
ALIGN_DIST = 0.6     # m: an object this close to a wall is turned to line up with it (if within ALIGN_DEG)
ALIGN_DEG = 15
MIN_DEPTH = 0.3     # m: depth is never shortened below this
SECOND_WALL = 1.5    # m: the second-nearest wall distance is kept only if closer than this


def to_floor_points(scene_map, P):
    """MoGe camera points -> floor frame (x right, y forward, z up), with the map's scale."""
    ax = np.array(scene_map["cv_axes_in_floor"], float)  # Mx.T
    Mx = ax.T
    factor = scene_map.get("scale_applied", {}).get("factor", 1.0)
    d = scene_map["camera_height_m"] / factor
    n = Mx[2]
    return ((P + d * n) @ Mx.T) * factor


def normals(Q):
    """Per-pixel surface normal of a grid of 3D points (H, W, 3)."""
    du = np.zeros_like(Q); dv = np.zeros_like(Q)
    du[:, 1:-1] = Q[:, 2:] - Q[:, :-2]
    dv[1:-1] = Q[2:] - Q[:-2]
    n = np.cross(du, dv)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def fit_lines(xy, rng, max_lines=3):
    """RANSAC: up to max_lines straight walls (2D lines on the floor) in a cloud of vertical-surface points."""
    out = []
    pts = xy.copy()
    for _ in range(max_lines):
        if len(pts) < MIN_WALL_PTS:
            break
        tol = WALL_TOL + 0.01 * np.linalg.norm(pts, axis=1)
        best = None
        for _ in range(400):
            a, b = pts[rng.choice(len(pts), 2, replace=False)]
            v = b - a
            if np.linalg.norm(v) < 0.2:
                continue
            v /= np.linalg.norm(v)
            nrm = np.array([-v[1], v[0]])
            inl = np.abs((pts - a) @ nrm) < tol
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < MIN_WALL_PTS:
            break
        q = pts[best]
        c = q.mean(0)
        v = np.linalg.svd(q - c, full_matrices=False)[2][0]
        out.append((c, v, q))
        pts = pts[~best]
    return out


def fit_objects_to_walls(scene_map, out_walls):
    # ---- put the objects inside the room: align to a nearby wall, and push out of the wall if they go through it
    for o in scene_map["objects"]:
        for w in out_walls:
            a, b = np.array(w["start_m"]), np.array(w["end_m"])
            L = np.linalg.norm(b - a); v = (b - a) / max(L, 1e-9); nrm = np.array(w["room_side"])
            _, corners, _, g = footprint(o)
            ts = [(p - a) @ v for p in corners]
            if max(ts) < -0.2 or min(ts) > L + 0.2:
                continue
            ds = [(p - a) @ nrm for p in corners]
            if min(ds) > ALIGN_DIST:
                continue
            side = np.array([math.cos(g), math.sin(g)])  # width axis of the object
            turn = math.degrees(math.atan2(side[0] * v[1] - side[1] * v[0], side @ v))
            turn = (turn + 45) % 90 - 45  # to the nearest wall-parallel or wall-perpendicular axis
            turned = 0.0
            if abs(turn) < ALIGN_DEG and abs(turn) > 0.5:
                turned = turn
                o.setdefault("yaw_before_room_deg", o["yaw_deg"])
                o["yaw_deg"] = round((o["yaw_deg"] + turn) % 360, 1)
                o.pop("yaw_from_appearance", None)
                _, corners, _, _ = footprint(o)
                ds = [(p - a) @ nrm for p in corners]
            # goes through the wall: the depth was only a guess (only the front is seen), so shorten it first;
            # the measured visible face is moved only if it is itself behind the wall
            shortened = pushed = 0.0
            for _ in range(6):
                if min(ds) >= -0.005 or o["depth_m"] <= MIN_DEPTH:
                    break
                o.setdefault("depth_before_room_m", o["depth_m"])
                new = max(MIN_DEPTH, o["depth_m"] + min(ds))
                shortened += o["depth_m"] - new
                o["depth_m"] = round(new, 3)
                _, corners, _, _ = footprint(o)
                ds = [(p - a) @ nrm for p in corners]
            if min(ds) < -0.005:
                o.setdefault("center_before_room_m", list(o["center_m"]))
                pushed = -min(ds)
                shift = pushed * nrm
                o["center_m"] = [round(o["center_m"][0] + float(shift[0]), 3), round(o["center_m"][1] + float(shift[1]), 3)] + list(o["center_m"][2:])
            if turned or shortened or pushed:
                o.setdefault("fitted_to_walls", []).append(w["id"])
                print(f"  object {o['id']} {o['name']}: fitted to {w['id']} (turned {turned:+.0f} deg, depth -{shortened:.2f} m, "
                      f"moved {pushed:.2f} m)", flush=True)
    if any("center_before_room_m" in o for o in scene_map["objects"]):
        import itertools
        objs = scene_map["objects"]
        scene_map["distances"] = [{"a": p["id"], "b": q["id"], "distance_m": round(math.dist(p["center_m"][:2], q["center_m"][:2]), 2)}
                                  for p, q in itertools.combinations(objs, 2)]


def measure_wall_distances(scene_map, out_walls):
    # ---- distance from each object (its footprint rectangle) to its nearest wall (and a second one if close)
    for o in scene_map["objects"]:
        _, corners, _, _ = footprint(o)
        found = []
        for w in out_walls:
            a, b = np.array(w["start_m"]), np.array(w["end_m"])
            v = (b - a) / max(np.linalg.norm(b - a), 1e-9); nrm = np.array(w["room_side"])
            best = None
            for p in corners:
                t = (p - a) @ v
                if not -0.3 < t < np.linalg.norm(b - a) + 0.3:  # only walls the object is actually in front of
                    continue
                dist = (p - a) @ nrm
                if best is None or dist < best[0]:
                    best = (dist, p, a + v * t)
            if best is not None and best[0] > -0.15:
                found.append(dict(wall=w["id"], distance_m=round(max(0.0, float(best[0])), 2),
                                  from_m=[round(float(x), 3) for x in best[1]], to_m=[round(float(x), 3) for x in best[2]]))
        found.sort(key=lambda f: f["distance_m"])
        o["wall_distances"] = found[:1] + [f for f in found[1:2] if f["distance_m"] < SECOND_WALL]
        if o["wall_distances"]:
            print(f"  object {o['id']} {o['name']}: " + ", ".join(f"{f['distance_m']:.2f} m to {f['wall']}" for f in o["wall_distances"]), flush=True)


def main():
    folder = os.path.abspath(sys.argv[1])
    port = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PORT
    path = os.path.join(folder, "map.json")
    scene_map = json.load(open(path, encoding="utf-8"))
    W, H = scene_map["W"], scene_map["H"]
    P = np.load(os.path.join(folder, "moge_points.npz"))["points"].reshape(H, W, 3).astype(np.float64)
    if np.nanmedian(P[..., 2]) < 0:
        P = P * np.array([1, -1, -1])
    Q = to_floor_points(scene_map, P)
    N = normals(Q)
    ok = np.isfinite(Q).all(-1) & np.isfinite(N).all(-1)
    cam_h = scene_map["camera_height_m"]

    om = np.load(os.path.join(folder, "object_masks.npz"))
    objects_px = np.zeros((H, W), bool)
    for m in om["masks"]:
        objects_px |= np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    floor_px = np.array(Image.fromarray(om["floor"].astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127

    name = upload_image(os.path.join(folder, "photo.png"), port)
    ask = lambda phrase, k: sam_masks(name, phrase, "room_" + os.path.basename(folder), port, max_det=k)[0]
    wall_masks = ask("wall", 6)
    ceil_masks = ask("ceiling", 3) + ask("roof", 2)
    for m in ask("floor", 4):
        floor_px |= m
    print(f"SAM: {len(wall_masks)} wall masks, {len(ceil_masks)} ceiling/roof masks", flush=True)

    # ---- ceiling: pixels above the camera with a horizontal surface
    ceil_px = np.zeros((H, W), bool)
    for m in ceil_masks:
        ceil_px |= m
    ceil_px &= ok & ~objects_px & (Q[..., 2] > cam_h) & (np.abs(N[..., 2]) > 0.6)
    ceiling = round(float(np.median(Q[ceil_px][:, 2])), 2) if ceil_px.sum() > 300 else None
    print(f"ceiling: {ceiling} m ({ceil_px.sum()} px)", flush=True)

    # ---- floor: visible area on a 10 cm grid
    flo = ok & floor_px & ~objects_px & (np.abs(Q[..., 2]) < 0.1)
    fxy = Q[flo][:, :2]
    cells = {tuple(c) for c in np.floor(fxy / 0.1).astype(int)}
    floor_area = round(len(cells) * 0.01, 1)
    print(f"visible floor: {floor_area} m2", flush=True)

    # ---- walls: vertical-surface pixels of the wall masks -> straight lines on the plan
    rng = np.random.default_rng(0)
    raw = []
    wall_px = np.zeros((H, W), bool)
    for m in wall_masks:
        sel = m & ok & ~objects_px & ~ceil_px & (np.abs(N[..., 2]) < 0.35) & (Q[..., 2] > -0.1)
        if ceiling:
            sel &= Q[..., 2] < ceiling + 0.2
        wall_px |= sel
        q3 = Q[sel]
        if len(q3) > 40000:
            q3 = q3[rng.choice(len(q3), 40000, replace=False)]
        for c, v, q in fit_lines(q3[:, :2], rng):
            z = q3[np.abs((q3[:, :2] - c) @ np.array([-v[1], v[0]])) < WALL_TOL + 0.01 * np.linalg.norm(c), 2]
            raw.append(dict(c=c, v=v, q=q, top=float(np.percentile(z, 98)) if len(z) else 0.0))
    if not raw:
        print("no walls found", flush=True)

    # snap to the main direction (most rooms are square-ish); then merge duplicates
    if raw:
        main_v = max(raw, key=lambda w: len(w["q"]))["v"]
        base = math.atan2(main_v[1], main_v[0])
        for w in raw:
            ang = math.atan2(w["v"][1], w["v"][0])
            k = round((ang - base) / (math.pi / 2))
            if abs(math.degrees(ang - base - k * math.pi / 2)) < SNAP_DEG:
                a = base + k * math.pi / 2
                w["v"] = np.array([math.cos(a), math.sin(a)])
                nrm = np.array([-w["v"][1], w["v"][0]])
                w["c"] = w["c"] + nrm * np.median((w["q"] - w["c"]) @ nrm)  # keep the snapped line through the points
    merged = []
    for w in sorted(raw, key=lambda w: -len(w["q"])):
        nrm = np.array([-w["v"][1], w["v"][0]])
        for m_ in merged:
            if abs(abs(w["v"] @ m_["v"]) - 1) < 0.01 and abs((w["c"] - m_["c"]) @ np.array([-m_["v"][1], m_["v"][0]])) < 0.25:
                m_["q"] = np.vstack([m_["q"], w["q"]]); m_["top"] = max(m_["top"], w["top"])
                break
        else:
            merged.append(w)

    walls = []
    for w in merged:
        t = (w["q"] - w["c"]) @ w["v"]
        t0, t1 = np.percentile(t, 1), np.percentile(t, 99)
        walls.append(dict(a=w["c"] + w["v"] * t0, b=w["c"] + w["v"] * t1, top=w["top"], n=len(w["q"])))

    # join ends that almost meet into corners
    def inter(w1, w2):
        d1, d2 = w1["b"] - w1["a"], w2["b"] - w2["a"]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-6 * np.linalg.norm(d1) * np.linalg.norm(d2) or abs(den) / (np.linalg.norm(d1) * np.linalg.norm(d2)) < 0.5:
            return None
        s = ((w2["a"][0] - w1["a"][0]) * d2[1] - (w2["a"][1] - w1["a"][1]) * d2[0]) / den
        return w1["a"] + s * d1
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            x = inter(walls[i], walls[j])
            if x is None:
                continue
            long_pair = min(np.linalg.norm(w["b"] - w["a"]) for w in (walls[i], walls[j])) > LONG_WALL
            for w in (walls[i], walls[j]):
                k = min(("a", "b"), key=lambda e: np.linalg.norm(w[e] - x))
                gap = np.linalg.norm(w[k] - x)
                other = "b" if k == "a" else "a"
                beyond = (x - w[other]) @ (w[k] - w[other]) > (w[k] - w[other]) @ (w[k] - w[other])  # x is past this end
                if gap < CORNER_GAP or (long_pair and beyond and gap < MAX_EXTEND):
                    if gap > 0.3:  # the part we did not see: drawn dashed
                        w.setdefault("assumed", []).append((w[k].copy(), x.copy()))
                    w[k] = x

    # each wall: which side is the room (the camera side)
    out_walls = []
    for k, w in enumerate(walls, 1):
        v = w["b"] - w["a"]; L = float(np.linalg.norm(v)); v = v / max(L, 1e-9)
        nrm = np.array([-v[1], v[0]])
        if (np.zeros(2) - w["a"]) @ nrm < 0:
            nrm = -nrm
        out_walls.append(dict(id=f"W{k}", start_m=[round(float(x), 3) for x in w["a"]], end_m=[round(float(x), 3) for x in w["b"]],
                              assumed_parts_m=[[[round(float(x), 3) for x in p0], [round(float(x), 3) for x in p1]] for p0, p1 in w.get("assumed", [])],
                              length_m=round(L, 2), height_m=ceiling if ceiling else round(w["top"], 2),
                              height_source="ceiling" if ceiling else "top of visible wall",
                              room_side=[round(float(x), 3) for x in nrm], pixels=w["n"]))
        print(f"  wall W{k}: {L:.2f} m long, {out_walls[-1]['height_m']} m high, {w['n']} points", flush=True)

    fit_objects_to_walls(scene_map, out_walls)
    measure_wall_distances(scene_map, out_walls)

    hull = []
    if len(fxy) > 10:
        from scipy.spatial import ConvexHull
        h = ConvexHull(fxy)
        hull = [[round(float(x), 3) for x in fxy[i]] for i in h.vertices]
    scene_map["room"] = dict(walls=out_walls, ceiling_height_m=ceiling,
                             floor=dict(visible_area_m2=floor_area, outline_m=hull),
                             note="walls are lines on the floor plan; only what the photo shows is measured")
    json.dump(scene_map, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # check image: walls blue (with ids), ceiling orange, floor green
    rgb = np.array(Image.open(os.path.join(folder, "photo.png")).convert("RGB")).astype(float)
    for mask, col in ((flo, (60, 200, 90)), (ceil_px, (255, 160, 40)), (wall_px, (60, 120, 255))):
        rgb[mask] = rgb[mask] * 0.45 + np.array(col) * 0.55
    im = Image.fromarray(rgb.astype(np.uint8)); dr = ImageDraw.Draw(im)
    for w, raw_w in zip(out_walls, walls):
        a, b = np.array(w["start_m"]), np.array(w["end_m"])
        dist = np.abs((Q[..., :2] - a) @ np.array(w["room_side"])) < 0.06
        ys, xs = np.nonzero(dist & wall_px)
        if len(xs):
            dr.text((float(np.median(xs)), float(np.median(ys))), w["id"], font=font(28), fill=(255, 255, 255), anchor="mm",
                    stroke_width=3, stroke_fill=(0, 0, 0))
    im.save(os.path.join(folder, "room_masks.png"))
    print("room done", flush=True)


if __name__ == "__main__":
    main()
