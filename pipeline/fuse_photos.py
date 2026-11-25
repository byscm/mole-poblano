# Puts two photos of the same room into one map.
#  1) Alignment on the floor plan (turn + shift; both maps must already have the right scale, see known_sizes.py):
#     start from their common completed object (e.g. the dining table, both ways round) and refine with ICP on
#     table-height tops and on tall structure (walls, windows, heater...). Island-height tops are left OUT of the
#     alignment on purpose, so they can be used afterwards to check how far each photo put them.
#     With "landmarks": "auto" the reference pairs are found from the objects' qualities (object_tags.py:
#     color, DINOv3 look, size, foot position) - only distinctive objects, see object_match.auto_landmarks.
#  2) Objects: matched by qualities x position (both must agree), each pair at most once;
#     the same object seen in both photos is kept once, from the photo where it was CLOSER to the camera
#     (depth is much better near the camera). Objects seen in only one photo are kept as they are.
#  3) Optional extra surfaces measured from the points (e.g. a kitchen island that the detector missed).
# Usage: python -I fuse_photos.py <scene_A> <scene_B> <out_folder> <fuse_config.json>
#   The map is in scene A's frame (camera A at 0,0). Writes <out_folder>\map.json, fusion.json, alignment.png
import copy, itertools, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from room_layout import normals, measure_wall_distances
from draw_plan import footprint
import known_sizes as K
import object_match as OM
from scipy.optimize import linear_sum_assignment

MAX_RANGE = 9.0     # m: points farther than this from their camera are not used
SAME_OBJECT = 0.6   # m between body centres (after alignment) to be the same object
SAME_HEIGHT = 0.15  # m
SAME_SIZE = 1.6     # ratio of the larger floor side
MATCH_PLACE = 0.8   # m: position term exp(-(feet apart / MATCH_PLACE)^2)
MATCH_MIN = 0.3     # qualities x position needed to call two objects the same
MATCH_QUALITY = 0.5 # qualities alone needed too
INLIER = 0.10       # m


def features(folder, m):
    Q = K.load_points(folder, m)
    N = normals(Q)
    ok = np.isfinite(Q).all(-1) & np.isfinite(N).all(-1) & (np.linalg.norm(Q[..., :2], axis=-1) < MAX_RANGE)
    up = np.abs(N[..., 2] / (np.linalg.norm(N, axis=-1) + 1e-9))
    z = Q[..., 2]
    rng = np.random.default_rng(0)
    pick = lambda s, n=25000: (lambda P: P[rng.choice(len(P), min(n, len(P)), replace=False)])(Q[s][:, :2])
    return dict(top=pick(ok & (up > 0.9) & (z > 0.62) & (z < 0.82)),
                tall=pick(ok & (up < 0.5) & (z > 1.0) & (z < 2.3)),
                island=Q[ok & (up > 0.9) & (z > 0.86) & (z < 1.0)])


LANDMARK_WEIGHT = 0.5  # share of the alignment that the landmark pairs carry


def landmark(folder, m, oid, zlo, zhi):
    """Median floor position of an object's points between two heights (e.g. a pot's rim, a heater's body)."""
    Q = K.load_points(folder, m)
    P = Q[K.object_mask(folder, m, oid) & np.isfinite(Q).all(-1)]
    P = P[(P[:, 2] > zlo) & (P[:, 2] < zhi)]
    return np.median(P[:, :2], 0)


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def icp(A, B, R, t, iters=40, LA=None, LB=None):
    trees = {k: cKDTree(A[k]) for k in ("top", "tall")}
    for it in range(iters):
        thr = max(INLIER, 0.6 * (1 - it / 25))
        src, dst = [], []
        for k in ("top", "tall"):
            P = B[k] @ R.T + t
            d, j = trees[k].query(P)
            keep = d < thr
            src.append(B[k][keep]); dst.append(A[k][j[keep]])
        src, dst = np.vstack(src), np.vstack(dst)
        if len(src) < 100:
            return R, t, 0.0
        w = np.ones(len(src))
        if LA is not None and len(LA):  # landmark pairs carry LANDMARK_WEIGHT of the total weight
            src, dst = np.vstack([src, LB]), np.vstack([dst, LA])
            w = np.r_[w, np.full(len(LA), len(w) * LANDMARK_WEIGHT / (1 - LANDMARK_WEIGHT) / len(LA))]
        w = w / w.sum()
        ms, md = w @ src, w @ dst
        U, _, Vt = np.linalg.svd(((src - ms) * w[:, None]).T @ (dst - md))
        R = (U @ Vt).T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = (U @ Vt).T
        t = md - ms @ R.T
    score = np.mean([np.mean(trees[k].query(B[k] @ R.T + t)[0] < INLIER) for k in ("top", "tall")])
    return R, t, float(score)


def body(o):
    return footprint(o)[0]


def moved(o, R, t, theta_deg):
    """Object of photo B expressed in photo A's frame (keeping the 'visible face' convention around camera A)."""
    o = copy.deepcopy(o)
    c = body(o) @ R.T + t
    yaw = o.get("yaw_from_appearance", o["yaw_deg"]) + theta_deg
    o.pop("yaw_from_appearance", None)
    K.set_body(o, c, yaw, o["width_m"], o["depth_m"])
    return o


def rect_from_points(P):
    """Smallest-ish rectangle around floor points (PCA axes, 1-99 percentiles)."""
    c0 = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c0, full_matrices=False)
    ax = Vt[0]
    if ax[1] < 0:
        ax = -ax
    side = ax; front = np.array([side[1], -side[0]])  # footprint(): side = (cos g, sin g), front = (sin g, -cos g)
    u, v = (P - c0) @ side, (P - c0) @ front
    u0, u1 = np.percentile(u, [1, 99]); v0, v1 = np.percentile(v, [1, 99])
    c = c0 + side * (u0 + u1) / 2 + front * (v0 + v1) / 2
    return c, math.degrees(math.atan2(side[1], side[0])), u1 - u0, v1 - v0


def main():
    fa, fb, out, cfg_path = [os.path.abspath(x) for x in sys.argv[1:5]]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    os.makedirs(out, exist_ok=True)
    ma = json.load(open(os.path.join(fa, "map.json"), encoding="utf-8"))
    mb = json.load(open(os.path.join(fb, "map.json"), encoding="utf-8"))
    A, B = features(fa, ma), features(fb, mb)

    # ---- 1) alignment
    tagged = all(os.path.exists(os.path.join(f, "object_tags.npz")) for f in (fa, fb))
    TA, TB = (OM.load(fa), OM.load(fb)) if tagged else (None, None)
    marks = cfg.get("landmarks", [])
    starts = []  # (turn, point in A, same point in B)
    auto = None
    if marks == "auto":
        if not tagged:
            sys.exit("landmarks 'auto' needs object_tags.py run on both scenes")
        pairs_auto, Rt, distinct = OM.auto_landmarks(TA, TB)
        if not pairs_auto:
            sys.exit("no distinctive objects shared by the two photos: give landmarks by hand")
        auto = dict(pairs=[dict(A=a, B=b, A_name=TA[a]["o"]["name"], B_name=TB[b]["o"]["name"]) for a, b in pairs_auto],
                    distinctive=distinct)
        print("auto landmarks: " + ", ".join(f"A#{a} {TA[a]['o']['name']} = B#{b} {TB[b]['o']['name']}" for a, b in pairs_auto))
        # only DISTINCTIVE pairs pull the alignment; look-alikes (the 12 chairs) only confirmed it
        dset = {(d["A"], d["B"]) for d in distinct}
        pull = [(a, b) for a, b in pairs_auto if (a, b) in dset] or pairs_auto
        auto["pulling"] = [dict(A=a, B=b) for a, b in pull]
        LA = np.array([OM.anchor(TA[a]) for a, b in pull]); LB = np.array([OM.anchor(TB[b]) for a, b in pull])
        marks = [(a, b, None, None) for a, b in pull]
        R0, t0 = Rt
        starts.append((math.atan2(R0[1, 0], R0[0, 0]), t0, np.zeros(2)))
    else:
        LA = np.array([landmark(fa, ma, a, lo, hi) for a, b, lo, hi in marks]) if marks else None
        LB = np.array([landmark(fb, mb, b, lo, hi) for a, b, lo, hi in marks]) if marks else None
    if cfg.get("anchor"):  # a shared object given by hand: start from it, both ways round
        oa = next(o for o in ma["objects"] if o["id"] == cfg["anchor"][0])
        ob = next(o for o in mb["objects"] if o["id"] == cfg["anchor"][1])
        ga, gb = (math.radians(o.get("yaw_from_appearance", o["yaw_deg"])) for o in (oa, ob))
        for flip in (0.0, math.pi):
            starts.append((ga - gb + flip, body(oa), body(ob)))
    tries = []
    for th0, ca, cb in starts:
        for extra in np.radians([-10, 0, 10]):
            R0 = rot(th0 + extra)
            R, t, s = icp(A, B, R0, ca - cb @ R0.T, LA=LA, LB=LB)
            tries.append((s, R, t, math.degrees(th0 + extra)))
    tries.sort(key=lambda x: -x[0])
    for x in tries:
        print(f"  start {x[3]:+.0f}: turn {math.degrees(math.atan2(x[1][1, 0], x[1][0, 0])):.1f}, camera B ({x[2][0]:.2f}, {x[2][1]:.2f}), {x[0]*100:.0f} %")
    score, R, t, start = tries[0]
    theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    print(f"alignment: turn {theta:.1f} deg, camera B at ({t[0]:.2f}, {t[1]:.2f}), {score*100:.0f} % of points within {INLIER*100:.0f} cm"
          f"  (next best start: {tries[1][0]*100:.0f} %)")

    landmark_residuals = []
    if marks:
        for (a, b, lo, hi), pa, pb in zip(marks, LA, LB):
            landmark_residuals.append(dict(A=a, B=b, apart_m=round(float(np.linalg.norm(pb @ R.T + t - pa)), 2)))
            print(f"  landmark A#{a} / B#{b}: {landmark_residuals[-1]['apart_m']:.2f} m apart after alignment")

    checks = []
    for a, b in cfg.get("check_pairs", []):  # pairs NOT used to align: how far apart they end up
        if tagged and a in TA and b in TB:
            d = float(np.linalg.norm(OM.anchor(TB[b]) @ R.T + t - OM.anchor(TA[a])))
            checks.append(dict(A=a, B=b, feet_apart_m=round(d, 2)))
            print(f"  check (not used to align) A#{a} {TA[a]['o']['name']} / B#{b} {TB[b]['o']['name']}: feet {d:.2f} m apart")

    # ---- 2) objects: the same object once, from the photo where it is closer to its camera
    objs, used_b, merged = [], set(), []
    bm = {o["id"]: moved(o, R, t, theta) for o in mb["objects"]}
    dist_b = {o["id"]: float(np.linalg.norm(body(o))) for o in mb["objects"]}  # distance to camera B
    pairs = [tuple(p) for p in cfg.get("same", [])]
    match, why = {a: b for a, b in pairs}, {}
    if tagged:  # qualities x position, best overall assignment
        ia_list = [o["id"] for o in ma["objects"] if o["id"] in TA and o["id"] not in match]
        ib_list = [i for i in bm if i in TB and i not in match.values()]
        S = np.zeros((len(ia_list), len(ib_list)))
        info = {}
        for r, ia in enumerate(ia_list):
            for c, ib in enumerate(ib_list):
                q, parts = OM.quality(TA[ia], TB[ib])
                pa, pb = OM.anchor(TA[ia]), OM.anchor(TB[ib])
                d = float(np.linalg.norm(pb @ R.T + t - pa)) if pa is not None and pb is not None else 9.0
                place = math.exp(-(d / MATCH_PLACE) ** 2)
                S[r, c] = q * place
                info[(ia, ib)] = dict(quality=round(q, 2), place=round(place, 2), feet_apart_m=round(d, 2),
                                      **{k: round(v, 2) for k, v in parts.items()})
        rows, cols = linear_sum_assignment(-S)
        for r, c in zip(rows, cols):
            if S[r, c] >= MATCH_MIN and info[(ia_list[r], ib_list[c])]["quality"] >= MATCH_QUALITY:
                match[ia_list[r]] = ib_list[c]; why[ia_list[r]] = info[(ia_list[r], ib_list[c])]
    for o in ma["objects"]:
        cand = [match[o["id"]]] if o["id"] in match else []
        if not cand and not tagged:  # no qualities: closest body centre, similar height and size
            near = [(np.linalg.norm(body(bm[i]) - body(o)), i) for i in bm if i not in used_b
                    and abs(bm[i]["height_m"] - o["height_m"]) < SAME_HEIGHT
                    and max(bm[i]["width_m"], bm[i]["depth_m"]) / max(o["width_m"], o["depth_m"], 1e-3) < SAME_SIZE
                    and max(o["width_m"], o["depth_m"]) / max(bm[i]["width_m"], bm[i]["depth_m"], 1e-3) < SAME_SIZE]
            near = [x for x in near if x[0] < SAME_OBJECT]
            cand = [min(near)[1]] if near else []
        o2 = copy.deepcopy(o); o2["seen_in"] = ["A"]
        if cand:
            i = cand[0]; used_b.add(i)
            da = float(np.linalg.norm(body(o)))
            shift = float(np.linalg.norm(body(bm[i]) - body(o)))
            if dist_b[i] < da:
                o2 = copy.deepcopy(bm[i]); o2["id"] = o["id"]
            o2["seen_in"] = ["A", "B"]
            o2["fused"] = dict(other_id=i, kept="B" if dist_b[i] < da else "A", camera_distance_A=round(da, 2),
                               camera_distance_B=round(dist_b[i], 2), centres_apart_m=round(shift, 2),
                               matched_by=why.get(o["id"], "given by hand" if (o["id"], i) in pairs else "position"))
            merged.append((o["id"], i, o["name"], mb["objects"][[x["id"] for x in mb["objects"]].index(i)]["name"], o2["fused"]))
        objs.append(o2)
    nid = max(o["id"] for o in objs) + 1
    for i, o in bm.items():
        if i not in used_b and i not in cfg.get("skip_b", []) and dist_b[i] <= cfg.get("keep_b_within_m", 99):
            o = copy.deepcopy(o); o["seen_in"] = ["B"]; o["id_in_B"] = i; o["id"] = nid; nid += 1
            objs.append(o)

    # ---- 3) surfaces measured from the points (an island the detector missed): each photo separately, to compare
    surfaces = []
    for s in cfg.get("surfaces", []):
        found = {}
        for tag, F, Rt in (("A", A, (np.eye(2), np.zeros(2))), ("B", B, (R, t))):
            P = F["island"]
            P = P[np.linalg.norm(P[:, :2], axis=1) < s.get("max_camera_distance", {}).get(tag, MAX_RANGE)]
            if len(P) < 500:
                continue
            xy = P[:, :2] @ Rt[0].T + Rt[1]
            c, g, w, d = rect_from_points(xy)
            found[tag] = dict(centre=c, yaw=g, w=w, d=d, h=float(np.median(P[:, 2])), n=len(P),
                              camera_distance=float(np.median(np.linalg.norm(P[:, :2], axis=1))), xy=xy)
        if not found:
            continue
        keep = min(found, key=lambda k: found[k]["camera_distance"])
        f = found[keep]
        o = dict(id=nid, name=s["name"], type=s["name"], base_z_m=0.0, height_m=round(f["h"], 3), center_m=[0, 0],
                 yaw_deg=0.0, width_m=0.1, depth_m=0.1, seen_in=sorted(found), from_points_of=keep, image_box=[0, 0, 1, 1])
        K.set_body(o, f["centre"], f["yaw"], f["w"], f["d"])  # footprint(): the yaw angle is the direction of the width side
        nid += 1
        if s.get("replaces_A") and keep == "B":  # the detector's box for the same thing in A was far from camera A
            gone = next((x for x in objs if x["id"] == s["replaces_A"] and x["seen_in"] == ["A"]), None)
            if gone:
                objs.remove(gone)
                o["replaces"] = dict(id=gone["id"], name=gone["name"], centre=[round(float(x), 2) for x in body(gone)],
                                     size=[gone["width_m"], gone["depth_m"], gone["height_m"]])
                o["id"] = gone["id"]; nid -= 1
        objs.append(o)
        surfaces.append(dict(name=s["name"], id=o["id"], kept=keep, replaces=o.get("replaces"),
                             **{tag: dict(centre=[round(float(x), 2) for x in v["centre"]], size=[round(v["w"], 2), round(v["d"], 2)],
                                          height=round(v["h"], 3), camera_distance=round(v["camera_distance"], 2), points=v["n"])
                                for tag, v in found.items()}))
        if len(found) == 2:
            surfaces[-1]["A_vs_B_centres_m"] = round(float(np.linalg.norm(found["A"]["centre"] - found["B"]["centre"])), 2)

    fused = copy.deepcopy(ma)
    fused["objects"] = objs
    fused["fused_from"] = dict(A=os.path.basename(fa), B=os.path.basename(fb), camera_B_m=[round(float(x), 3) for x in t],
                               camera_B_turn_deg=round(theta, 1), alignment_inliers=round(score, 3),
                               landmarks=landmark_residuals,
                               camera_B_look=[round(float(x), 3) for x in R @ np.array(mb["cv_axes_in_floor"][2][:2], float)])
    fused["distances"] = [{"a": p["id"], "b": q["id"], "distance_m": round(math.dist(body(p), body(q)), 2)}
                          for p, q in itertools.combinations(objs, 2)]
    if fused.get("room"):
        measure_wall_distances(fused, fused["room"]["walls"])
    json.dump(fused, open(os.path.join(out, "map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    Image.open(os.path.join(fa, "photo.png")).save(os.path.join(out, "photo.png"))

    gaps = []
    by = {o["id"]: o for o in objs}
    for g in cfg.get("gaps", []):
        a = by[g["a"]] if isinstance(g["a"], int) else next(o for o in objs if o["name"] == g["a"])
        b = by[g["b"]] if isinstance(g["b"], int) else next(o for o in objs if o["name"] == g["b"])
        gaps.append(dict(g, fused=round(K.edge_gap(np.array(footprint(a)[1]), np.array(footprint(b)[1])), 2)))
        print(f"  gap {g['what']}: real {g['real']:.2f}  fused {gaps[-1]['fused']:.2f}")
    for a, b, na, nb, f in merged:
        mb_ = f["matched_by"]
        how = mb_ if isinstance(mb_, str) else (f"quality {mb_['quality']:.2f} (look {mb_['look']:.2f}, color {mb_['color']:.2f}, "
                                               f"height {mb_['height']:.2f}), feet {mb_['feet_apart_m']:.2f} m apart")
        print(f"  same object: A#{a} {na} = B#{b} {nb}  (kept {f['kept']}; {how})")
    for s in surfaces:
        print("  surface", json.dumps(s))
    json.dump(dict(alignment=fused["fused_from"], auto_landmarks=auto, checks=checks, tries=[dict(score=round(x[0], 3), start_turn=x[3]) for x in tries],
                   merged=[dict(A=a, B=b, A_name=na, B_name=nb, **f) for a, b, na, nb, f in merged],
                   surfaces=surfaces, gaps=gaps),
              open(os.path.join(out, "fusion.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    # ---- alignment picture: A's points grey, B's points (aligned) orange, cameras
    S, W, H = 80, 1000, 1000
    allp = np.vstack([A["tall"], A["top"], B["tall"] @ R.T + t, B["top"] @ R.T + t])
    lo = np.percentile(allp, 1, 0) - 0.5
    P = lambda p: ((p[0] - lo[0]) * S, H - (p[1] - lo[1]) * S)
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    for k, col in (("tall", (150, 150, 150)), ("top", (120, 200, 120))):
        for p in A[k][::4]: d.point(P(p), fill=col)
    for k, col in (("tall", (240, 140, 40)), ("top", (40, 120, 240))):
        for p in (B[k] @ R.T + t)[::4]: d.point(P(p), fill=col)
    for p, name in ((np.zeros(2), "camera A"), (t, "camera B")):
        x, y = P(p); d.ellipse([x - 6, y - 6, x + 6, y + 6], fill="black"); d.text((x + 8, y), name, fill="black")
    img.save(os.path.join(out, "alignment.png"))


if __name__ == "__main__":
    main()
