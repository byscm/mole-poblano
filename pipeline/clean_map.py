# Turns the raw per-frame detections of an aligned scene into one clean map: every real object once, and one closed room
# outline. Detections are compared by the 3D footprint of their pixels (not just their centres), only between compatible
# kinds, and a group may never grow past the plausible size of that kind, so neighbouring chairs do not chain together.
# The outline comes from the floor actually seen plus the furniture standing on it, squared to the room's two directions.
# Usage: python clean_map.py <alignment_good.json> <out_folder>   (python with numpy, scipy, opencv, pillow)
# Writes <out_folder>/clean_map.json and clean_plan.png
import json, math, os, re, sys
from collections import Counter, defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from known_sizes import load_points

CELL = 0.05          # footprint grid (m)
FLOOR_CELL = 0.04    # outline grid (m)
GROUPS = [  # first match wins
    ("clutter", r"kettle|handle|spatula|golf|baseball|\brice\b|paper|cushion|\bboard\b|wreath"),
    ("stool", r"stool"),
    ("chair", r"chair|armrest|seat\b"),
    ("plant", r"plant|bonsai|\btree\b|fern|lily|orchid|flower pot|potted|leafy"),
    ("picture", r"painting|picture|\bframed?\b|artwork|poster|mirror"),
    ("clock", r"clock"),
    ("appliance", r"refrigerator|fridge|\boven\b|stove|cooktop|microwave|dishwasher|range hood|heater"),
    ("island", r"island"),
    ("surface", r"counter|sink"),
    ("sofa", r"sofa|couch"),
    ("table", r"table|desk"),
    ("cabinet", r"cabinet|sideboard|drawer|shelf|shelves|mantel|fireplace|console|wardrobe|credenza|\bstand\b"),
    ("decor", r"vase|figurine|sculpture|ornament|statue|flower|lamp|candle"),
    ("clutter", r"cup|mug|bottle|plate|bowl|dish|food|bread|cupcake|fruit|banana|towel|\bbag\b|\bbox|crate|cardboard|book|phone|\bpen\b|ruler|cable|wire|usb|"
                r"magnet|kettle|cooker|\bpot\b|\bcan\b|toy|\blid\b|container|bucket|basket|dumbbell|debris|wallet|sunglasses|speaker|laptop|parts"),
]
COMPATIBLE = {"table": {"table", "surface"}, "island": {"island", "surface", "cabinet"}, "surface": {"surface", "table", "island", "cabinet"},
              "cabinet": {"cabinet", "surface", "island", "appliance"}, "appliance": {"appliance", "cabinet"}}
MAX_SIZE_M = dict(chair=0.8, stool=0.65, table=3.6, island=3.2, surface=3.8, cabinet=3.8, appliance=1.5, plant=1.7, picture=1.4, decor=0.6, clutter=0.5,
                  sofa=3.6, clock=0.7, other=1.5)
ON_FLOOR = {"chair", "stool", "table", "island", "cabinet", "appliance", "sofa", "surface"}


def group_of(name):
    n = name.lower()
    for g, rx in GROUPS:
        if re.search(rx, n): return g
    return "other"


def compatible(a, b):
    return a == b or b in COMPATIBLE.get(a, ()) or a in COMPATIBLE.get(b, ())


def to_ref(P, v):
    k = 1 / v.get("depth_scale", 1.0); th = math.radians(v["turn_deg"]); c, s = math.cos(th), math.sin(th)
    x, y = P[..., 0] * k, P[..., 1] * k
    return np.stack([c * x - s * y + v["t_m"][0], s * x + c * y + v["t_m"][1], P[..., 2] * k], -1)


def resized(mask, W, H):
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127


def main_direction(al):
    seg = []
    for v in al.values():
        m = json.load(open(os.path.join(v["folder"], "map.json"), encoding="utf-8")); th = math.radians(v["turn_deg"]); c, s = math.cos(th), math.sin(th)
        for w in m.get("room", {}).get("walls", []):
            d = np.array(w["end_m"]) - np.array(w["start_m"]); L = float(np.linalg.norm(d))
            if L > 0.4: seg.append((math.degrees(math.atan2(s * d[0] + c * d[1], c * d[0] - s * d[1])) % 90, L))
    a = np.array(seg); return float(np.argmax([np.sum(a[:, 1] * np.cos(np.radians(4 * (a[:, 0] - t)))) for t in np.arange(0, 90, 0.5)]) * 0.5)


def load_detections(al):
    det, floor = [], []
    for name, v in al.items():
        f = v["folder"]; m = json.load(open(os.path.join(f, "map.json"), encoding="utf-8")); z = np.load(os.path.join(f, "object_masks.npz"))
        P = to_ref(load_points(f, m), v); W, H = m["W"], m["H"]; ok = np.isfinite(P).all(-1) & (P[..., 2] > -0.1) & (P[..., 2] < 3.2)
        cam = np.array(v["t_m"]); near = np.linalg.norm(P[..., :2] - cam, axis=-1) < 7.5
        fl = z["floor"]; fl = fl.any(0) if fl.ndim == 3 else fl
        fl = ndi.binary_erosion(resized(fl, W, H), iterations=3) & ok & near & (np.abs(P[..., 2]) < 0.12)
        q = P[fl][:, :2]; up = P[ok & near & (P[..., 2] > 1.3) & (P[..., 2] < 2.3)][:, :2]  # above the furniture: what the camera sees is wall
        floor.append((name, q[:: max(1, len(q) // 9000)], up[:: max(1, len(up) // 1500)]))
        for i, o in enumerate(m["objects"]):
            mk = resized(z["masks"][i], W, H); er = ndi.binary_erosion(mk, iterations=2); mk = er if er.sum() > 80 else mk
            q = P[mk & ok & near]
            if len(q) < 25: continue
            q = q[:: max(1, len(q) // 1800)]; lo, hi = np.percentile(q, 3, axis=0), np.percentile(q, 97, axis=0)
            q = q[((q >= lo - 0.03) & (q <= hi + 0.03)).all(1)]
            if len(q) < 15: continue
            cells = set(map(tuple, np.floor(q[:, :2] / CELL).astype(int)))
            det.append(dict(frame=name, clip=name[0], o=o, group=group_of(o["name"]), pts=q, cells=cells, grown=grow(cells), z0=float(np.percentile(q[:, 2], 3)),
                            z1=float(np.percentile(q[:, 2], 97)), dist=float(np.median(np.linalg.norm(q[:, :2] - cam, axis=1))), pixels=int(mk.sum())))
    return det, floor


def grow(cells, r=2):
    out = set()
    for x, y in cells:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r: out.add((x + dx, y + dy))
    return out


def extent(pts):
    return float(np.max(np.percentile(pts[:, :2], 97, axis=0) - np.percentile(pts[:, :2], 3, axis=0)))


def cluster(det):
    n = len(det); parent = list(range(n)); members = {i: [i] for i in range(n)}
    find = lambda a: a if parent[a] == a else find(parent[a])
    pairs = []
    for i in range(n):
        A = det[i]
        for j in range(i + 1, n):
            B = det[j]
            if A["frame"] == B["frame"] or not compatible(A["group"], B["group"]): continue
            if min(A["z1"], B["z1"]) - max(A["z0"], B["z0"]) < 0.3 * min(A["z1"] - A["z0"] + 0.05, B["z1"] - B["z0"] + 0.05): continue
            ov = len(A["grown"] & B["grown"]) / max(1, min(len(A["grown"]), len(B["grown"])))
            if ov >= 0.4: pairs.append((ov + (A["group"] == B["group"]) * 0.15, i, j))
    for _, i, j in sorted(pairs, reverse=True):
        a, b = find(i), find(j)
        if a == b: continue
        fa = {det[k]["frame"] for k in members[a]}; fb = {det[k]["frame"] for k in members[b]}
        if len(fa & fb) > 0.34 * min(len(fa), len(fb)): continue  # two things seen side by side in the same frames
        cap = max(MAX_SIZE_M[det[k]["group"]] for k in (a, b))
        if extent(np.concatenate([det[k]["pts"] for k in members[a] + members[b]])) > cap * 1.12: continue
        parent[b] = a; members[a] += members.pop(b)
    return list(members.values())


def describe(idx, det, main, single_ok):
    M = [det[k] for k in idx]; frames = {d["frame"] for d in M}
    votes = Counter(); gv = Counter()
    for d in M:
        w = math.sqrt(d["pixels"]) / (0.5 + d["dist"]); votes[d["o"]["name"]] += w; gv[d["group"]] += w
    group = gv.most_common(1)[0][0]
    best = max(M, key=lambda d: d["pixels"] / (0.5 + d["dist"]) ** 2)
    pts = np.concatenate([d["pts"] for d in M]); a = math.radians(main)
    yaw0 = best["o"].get("yaw_from_appearance", best["o"].get("yaw_deg", 0)) + math.degrees(math.atan2(0, 1))
    th = math.radians(best["o"].get("yaw_deg", 0)); v = {}
    yaw = main + round(((best["o"].get("yaw_deg", 0) - main) % 360) / 90) * 90  # squared to the room
    g = math.radians(yaw); ax, ay = np.array([math.cos(g), math.sin(g)]), np.array([-math.sin(g), math.cos(g)])
    pa, pb = pts[:, :2] @ ax, pts[:, :2] @ ay
    a0, a1, b0, b1 = np.percentile(pa, 3), np.percentile(pa, 97), np.percentile(pb, 3), np.percentile(pb, 97)
    c = ax * (a0 + a1) / 2 + ay * (b0 + b1) / 2
    top = float(np.median([d["z1"] for d in M])); base = 0.0 if group in ON_FLOOR else float(np.median([d["z0"] for d in M]))
    o = {k: x for k, x in best["o"].items() if k in ("type", "phrases", "orientation", "crop", "image_box", "facing_source")}
    o.update(name=votes.most_common(1)[0][0], group=group, clutter=group == "clutter", body_center_m=[round(float(x), 3) for x in c], center_m=[round(float(x), 3) for x in c],
             yaw_deg=round(yaw % 360, 1), width_m=round(float(a1 - a0), 3), depth_m=round(float(b1 - b0), 3), height_m=round(max(top - base, 0.02), 3), base_z_m=round(base, 3),
             source_folder=None, photo=best["frame"], id_in_photo=best["o"]["id"], seen_in=sorted(f"{d['frame']}#{d['o']['id']}" for d in M), n_frames=len(frames),
             n_clips=len({d["clip"] for d in M}), other_names=sorted(set(votes) - {votes.most_common(1)[0][0]}))
    return o


def room_box(o, U, V, main):
    a, b = np.array(o["body_center_m"]) @ U, np.array(o["body_center_m"]) @ V
    turned = round(((o["yaw_deg"] - main) % 360) / 90) % 2 == 1
    ha, hb = (o["depth_m"], o["width_m"]) if turned else (o["width_m"], o["depth_m"])
    return a - ha / 2, a + ha / 2, b - hb / 2, b + hb / 2


def set_box(o, box, U, V, main):
    a0, a1, b0, b1 = box; c = U * (a0 + a1) / 2 + V * (b0 + b1) / 2
    turned = round(((o["yaw_deg"] - main) % 360) / 90) % 2 == 1
    o["body_center_m"] = o["center_m"] = [round(float(x), 3) for x in c]
    o["width_m"], o["depth_m"] = [round(float(x), 3) for x in ((b1 - b0, a1 - a0) if turned else (a1 - a0, b1 - b0))]


def absorb(keep, part):
    keep["seen_in"] = sorted(set(keep["seen_in"]) | set(part["seen_in"])); keep["n_frames"] = len({s.split("#")[0] for s in keep["seen_in"]})
    keep["n_clips"] = len({s[0] for s in keep["seen_in"]}); keep["other_names"] = sorted((set(keep["other_names"]) | {part["name"]} | set(part["other_names"])) - {keep["name"]})


def consolidate(objs, U, V, main):
    """Folds fragments into the piece they belong to and joins the door panels of one wall into a single cabinet run."""
    few_views = len({s.split("#")[0] for o in objs for s in o["seen_in"]}) < 8  # a handful of photos rarely repeat an object
    solid = lambda o: few_views or o["n_frames"] >= 3 or (o["n_clips"] >= 2 and o["group"] not in ("chair", "stool"))  # seat fragments are common
    set_aside = [dict(o, set_aside="weak support") for o in objs if not o["clutter"] and not solid(o)]
    pieces = sorted([o for o in objs if o["clutter"] or solid(o)], key=lambda o: -(o["n_frames"] + 2 * o["n_clips"])); kept = []
    for p in pieces:
        P = room_box(p, U, V, main); area = max((P[1] - P[0]) * (P[3] - P[2]), 1e-3); host = None
        for k in kept:
            if k["clutter"] != p["clutter"] or not compatible(k["group"], p["group"]): continue
            K = room_box(k, U, V, main); ov = max(0, min(P[1], K[1]) - max(P[0], K[0])) * max(0, min(P[3], K[3]) - max(P[2], K[2]))
            if ov / area >= 0.55 and (K[1] - K[0]) * (K[3] - K[2]) >= area * 0.8: host = k; break
        if host: absorb(host, p)
        else: kept.append(p)
    changed = True
    while changed:
        changed = False; runs = [o for o in kept if o["group"] == "cabinet"]
        for i, x in enumerate(runs):
            for y in runs[i + 1:]:
                A, B = room_box(x, U, V, main), room_box(y, U, V, main); da, db = A[1] - A[0], A[3] - A[2]; ea, eb = B[1] - B[0], B[3] - B[2]
                along_b = max(da, ea) < 0.55 and abs((A[0] + A[1]) - (B[0] + B[1])) / 2 < 0.22 and max(A[2], B[2]) - min(A[3], B[3]) < 0.2
                along_a = max(db, eb) < 0.55 and abs((A[2] + A[3]) - (B[2] + B[3])) / 2 < 0.22 and max(A[0], B[0]) - min(A[1], B[1]) < 0.2
                if along_a or along_b:
                    absorb(x, y); set_box(x, (min(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), max(A[3], B[3])), U, V, main)
                    x["height_m"] = max(x["height_m"], y["height_m"]); x["name"] = "cabinet run"; kept.remove(y); changed = True; break
            if changed: break
    return kept, set_aside


def inside_share(o, outline, U, V, main):
    a0, a1, b0, b1 = room_box(o, U, V, main); poly = np.array(outline, np.float32); grid = [(a, b) for a in np.linspace(a0, a1, 7) for b in np.linspace(b0, b1, 7)]
    return np.mean([cv2.pointPolygonTest(poly, (float(a), float(b)), True) >= -0.08 for a, b in grid])


def orthogonal(contour, cell):
    pts = cv2.approxPolyDP(contour, 0.18 / cell, True)[:, 0, :].astype(float); E = []
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]; d = q - p
        E.append(["h", (p[1] + q[1]) / 2, abs(d[0])] if abs(d[0]) >= abs(d[1]) else ["v", (p[0] + q[0]) / 2, abs(d[1])])

    def merge(E):
        changed = True
        while changed and len(E) > 1:
            changed = False
            for i in range(len(E)):
                a, b = E[i], E[(i + 1) % len(E)]
                if a[0] == b[0]:
                    L = a[2] + b[2]; E[i] = [a[0], (a[1] * a[2] + b[1] * b[2]) / max(L, 1e-6), L]; del E[(i + 1) % len(E)]; changed = True; break
        return E

    E = merge(E)
    while len(E) > 4:
        i = int(np.argmin([e[2] for e in E]))
        if E[i][2] * cell >= 0.45: break
        del E[i]; E = merge(E)
    return [[(E[i] if E[i][0] == "v" else E[(i + 1) % len(E)])[1], (E[i] if E[i][0] == "h" else E[(i + 1) % len(E)])[1]] for i in range(len(E))]


def room_outline(floor, objs, cams, main):
    a = math.radians(main); U, V = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    uv = lambda p: np.c_[p @ U, p @ V]
    every = uv(np.concatenate([q for _, q, _u in floor])); lo = np.percentile(every, 0.2, axis=0) - 2.0; hi = np.percentile(every, 99.8, axis=0) + 2.0
    W, H = (np.ceil((hi - lo) / FLOOR_CELL).astype(int) + 1).tolist(); count = np.zeros((H, W), np.uint16)
    ij = lambda q: (np.clip(((q[:, 1] - lo[1]) / FLOOR_CELL).astype(int), 0, H - 1), np.clip(((q[:, 0] - lo[0]) / FLOOR_CELL).astype(int), 0, W - 1))
    for _, q, _u in floor:
        g = np.zeros((H, W), bool); g[ij(uv(q))] = True; count += ndi.binary_dilation(g, iterations=2)
    need = 2 if len(floor) >= 8 else 1; mask = count >= need
    # free space: the line from a camera to a wall point above the furniture is empty. Must agree across clips, so a window cannot open the room.
    carved = defaultdict(lambda: np.zeros((H, W), np.uint8))
    for (name, _q, up), cam in zip(floor, cams):
        if not len(up): continue
        c = (uv(np.array([cam]))[0] - lo) / FLOOR_CELL; e = (uv(up) - lo) / FLOOR_CELL; L = np.linalg.norm(e - c, axis=1, keepdims=True)
        e = c + (e - c) * np.clip(1 - (0.1 / FLOOR_CELL) / np.maximum(L, 1), 0, 1); tmp = carved[name[0]]
        for p in e.astype(int): cv2.line(tmp, (int(c[0]), int(c[1])), (int(p[0]), int(p[1])), 1, 1)
    if carved: mask |= sum(x.astype(np.uint8) for x in carved.values()) >= min(2, len(carved))
    for o in objs:
        if o["clutter"] or o["group"] not in ON_FLOOR or o["n_frames"] < 2: continue
        g = math.radians(o["yaw_deg"]); c = np.array(o["body_center_m"]); ax, ay = np.array([math.cos(g), math.sin(g)]), np.array([-math.sin(g), math.cos(g)])
        cs = uv(np.array([c + ax * sx * o["width_m"] / 2 + ay * sy * o["depth_m"] / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
        pix = np.round((cs - lo) / FLOOR_CELL).astype(np.int32); tmp = mask.astype(np.uint8); cv2.fillConvexPoly(tmp, pix, 1); mask = tmp.astype(bool)
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m8 = mask.astype(np.uint8); m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, disk(int(0.35 / FLOOR_CELL))); m8 = ndi.binary_fill_holes(m8).astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, disk(int(0.3 / FLOOR_CELL)))
    lab, n = ndi.label(m8)
    if n == 0: raise RuntimeError("no floor found")
    cam_pix = ij(uv(np.array(cams))); hits = Counter(int(x) for x in lab[cam_pix] if x > 0); sizes = ndi.sum(m8, lab, range(1, n + 1))
    keep = max(range(1, n + 1), key=lambda l: (hits.get(l, 0) > 0, sizes[l - 1]))
    room = (lab == keep).astype(np.uint8)
    cnt = max(cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    poly = orthogonal(cnt, FLOOR_CELL); out_uv = [[round(float(lo[0] + x * FLOOR_CELL), 3), round(float(lo[1] + y * FLOOR_CELL), 3)] for x, y in poly]
    return out_uv, U, V, float(room.sum() * FLOOR_CELL ** 2)


def draw(path, objs, cams, outline, U, V, title):
    ol = [U * a + V * b for a, b in outline]; pts = np.array(ol + [np.array(o["body_center_m"]) for o in objs if not o["clutter"]] + list(cams)); lo, hi = pts.min(0) - 1, pts.max(0) + 1
    sc = 130; im = Image.new("RGB", (int((hi[0] - lo[0]) * sc), int((hi[1] - lo[1]) * sc) + 60), "white"); d = ImageDraw.Draw(im)
    X = lambda p: ((p[0] - lo[0]) * sc, (hi[1] - p[1]) * sc + 60)
    try: font = ImageFont.truetype("arial.ttf", 13); big = ImageFont.truetype("arialbd.ttf", 21)
    except Exception: font = big = None
    d.polygon([X(q) for q in ol], fill=(247, 244, 237))
    for i, a in enumerate(ol):
        b = ol[(i + 1) % len(ol)]; d.line([X(a), X(b)], fill=(25, 25, 35), width=7); d.text(X((a + b) / 2), f"{np.linalg.norm(b - a):.2f} m", fill=(200, 30, 30), font=font)
    for o in objs:
        g = math.radians(o["yaw_deg"]); c = np.array(o["body_center_m"]); ax, ay = np.array([math.cos(g), math.sin(g)]), np.array([-math.sin(g), math.cos(g)])
        col = (190, 190, 190) if o["clutter"] else (30, 130, 60)
        d.polygon([X(c + ax * sx * max(o["width_m"], 0.08) / 2 + ay * sy * max(o["depth_m"], 0.08) / 2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))], outline=col, width=1 if o["clutter"] else 2)
        if not o["clutter"]: x, y = X(c); d.text((x - 25, y - 6), f"{o['group']} ({o['n_frames']})", fill=(15, 15, 15), font=font)
    for c in cams: x, y = X(c); d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(40, 90, 220))
    d.text((15, 12), title, fill=(0, 0, 0), font=big); d.text((15, 38), "green = furniture, grey = personal items left out, blue = cameras", fill=(70, 70, 70), font=font)
    d.line([X((lo[0] + 0.3, lo[1] + 0.3)), X((lo[0] + 1.3, lo[1] + 0.3))], fill=(0, 0, 0), width=4); d.text(X((lo[0] + 0.3, lo[1] + 0.52)), "1 m", fill=(0, 0, 0), font=font)
    im.save(path)


def main():
    al_path, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); al = json.load(open(al_path, encoding="utf-8"))["photos"]
    main_dir = main_direction(al); det, floor = load_detections(al); print(len(det), "detections,", len(al), "frames, main direction", main_dir, flush=True)
    single_ok = len(al) < 8; objs = []
    for idx in cluster(det):
        o = describe(idx, det, main_dir, single_ok); folder = al[o["photo"]]["folder"]; o["source_folder"] = folder
        min_frames = 1 if single_ok else 2
        if o["n_frames"] >= min_frames: objs.append(o)
    for o in objs: o["clutter"] = o["group"] in ("clutter", "other")
    a = math.radians(main_dir); U0, V0 = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    objs, set_aside = consolidate(objs, U0, V0, main_dir)
    cams = [np.array(v["t_m"], float) for v in al.values()]; outline, U, V, area = room_outline(floor, objs, cams, main_dir)
    inside = [o for o in objs if inside_share(o, outline, U, V, main_dir) >= 0.4]
    set_aside += [dict(o, set_aside="outside the room") for o in objs if not any(o is k for k in inside)]; objs = inside
    objs.sort(key=lambda o: (o["clutter"], o["group"], -o["n_frames"]))
    for i, o in enumerate(objs, 1): o["id"] = i
    furn = [o for o in objs if not o["clutter"]]
    print(len(set_aside), "set aside (kept in clean_map.json):", Counter(o["set_aside"] for o in set_aside).most_common())
    json.dump(dict(main_dir_deg=main_dir, axis_a=U.tolist(), axis_b=V.tolist(), outline_ab=outline, floor_area_m2=round(area, 2), needs_review=len(al) < 8, objects=objs, set_aside=set_aside,
                   cameras=[dict(photo=n, xy_m=v["t_m"], turn_deg=v["turn_deg"], folder=v["folder"]) for n, v in al.items()]),
              open(os.path.join(out, "clean_map.json"), "w", encoding="utf-8"), indent=1)
    draw(os.path.join(out, "clean_plan.png"), objs, cams, outline, U, V, f"Clean plan: {len(furn)} pieces, {area:.1f} m2 of floor")
    print(f"{len(objs)} objects: {len(furn)} furniture/decor + {len(objs) - len(furn)} personal items; outline {len(outline)} corners, {area:.1f} m2")
    print(Counter(o["group"] for o in objs).most_common())
    for o in furn: print(f"  {o['group']:10s} {o['name'][:30]:30s} {o['width_m']:.2f} x {o['depth_m']:.2f} x {o['height_m']:.2f}  frames {o['n_frames']}  clips {o['n_clips']}")


if __name__ == "__main__":
    main()
