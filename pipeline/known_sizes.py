# Uses sizes the user measured with a tape:
#  1) fixes the map's scale with them (fix_scale.py --measure, median of all references)
#  2) "completes" objects of known size that the photo only shows in part (e.g. a dining table hidden by chairs):
#     a rectangle of the real size is slid around the visible part; a placement is rejected where the photo sees
#     THROUGH it (floor or things behind the table top, where the table should have hidden them), and rewarded
#     where the photo sees table top. Placements that the photo cannot tell apart (the extra part falls behind
#     occluders) all score the same: the middle one is used and the spread is reported.
# Usage: python -I known_sizes.py <scene_folder> <known_in_scenes.json>
# Writes: map.json (copy of the previous one in map_before_known.json), known_sizes.json in the scene folder
import itertools, json, math, os, shutil, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from room_layout import to_floor_points, measure_wall_distances
from draw_plan import footprint
import fix_scale

STEP = 0.02         # m, placement grid
YAW_RANGE = 15      # deg either side of the detected yaw
SEE_THROUGH = 0.12  # m: the photo sees something at least this far BELOW the top (vertically) -> no table there
ON_TOP = 0.04       # m: a point this close to the top height is table top
PLATEAU = 0.97      # placements scoring at least this fraction of the best are "equally good"


def load_points(folder, m):
    W, H = m["W"], m["H"]
    P = np.load(os.path.join(folder, "moge_points.npz"))["points"].reshape(H, W, 3).astype(np.float64)
    if np.nanmedian(P[..., 2]) < 0:
        P = P * np.array([1, -1, -1])
    return to_floor_points(m, P)


def object_mask(folder, m, oid):
    masks = np.load(os.path.join(folder, "object_masks.npz"))["masks"]
    idx = [o["id"] for o in m["objects"]].index(oid)
    return np.array(Image.fromarray(masks[idx].astype(np.uint8) * 255).resize((m["W"], m["H"]), Image.NEAREST)) > 127


def set_body(o, c, yaw, width, depth):
    """Store a box whose BODY centre is c, in the map's convention (center_m = middle of the face the camera sees)."""
    r = float(np.linalg.norm(c))
    p = c / r * max(r - depth / 2, 0.05)
    o["center_m"] = [round(float(p[0]), 3), round(float(p[1]), 3)] + list(o["center_m"][2:])
    o["yaw_deg"] = round(yaw % 360, 1)
    o.pop("yaw_from_appearance", None)
    o["width_m"], o["depth_m"] = round(width, 3), round(depth, 3)
    o["camera_distance_m"] = round(float(np.linalg.norm(p)), 3)


def top_height(folder, m, Q, o):
    """Height of a table's top, from the points of its mask (the box height fails when the legs are hidden)."""
    z = Q[object_mask(folder, m, o["id"]) & np.isfinite(Q).all(-1)][:, 2]
    z = z[(z > 0.3) & (z < o["base_z_m"] + o["height_m"] + 0.1)]
    return float(np.percentile(z, 80)) if len(z) > 50 else o["base_z_m"] + o["height_m"]


def complete(folder, m, Q, item):
    o = next(x for x in m["objects"] if x["id"] == item["id"])
    L, Wd = item["length"], item["width"]
    cam = np.array([0.0, 0.0, m["camera_height_m"]])
    own = object_mask(folder, m, o["id"])
    ok = np.isfinite(Q).all(-1)
    top = top_height(folder, m, Q, o)
    D = Q - cam
    down = ok & (D[..., 2] < -1e-6)
    t = np.where(down, (top - cam[2]) / np.where(down, D[..., 2], -1), np.nan)  # ray parameter where it crosses the top
    X = cam[:2] + t[..., None] * D[..., :2]                                     # where it crosses, on the floor plan
    through = down & (t < 1) & (Q[..., 2] < top - SEE_THROUGH)                  # seen below the top, inside its outline
    on_top = own & ok & (np.abs(Q[..., 2] - top) < ON_TOP)
    sup_xy, thr_xy = Q[on_top][:, :2], X[through]

    c0, _, _, g0 = footprint(o)
    near = lambda p: p[np.linalg.norm(p - c0, axis=1) < L + 1.5]
    sup_xy, thr_xy = near(sup_xy), near(thr_xy)
    if len(sup_xy) < 50:
        return dict(id=o["id"], skipped=f"only {len(sup_xy)} table-top points seen")

    results = []  # (score, yaw, long_along_side, centre)
    for dg in range(-YAW_RANGE, YAW_RANGE + 1):
        g = g0 + math.radians(dg)
        side, front = np.array([math.cos(g), math.sin(g)]), np.array([math.sin(g), -math.cos(g)])
        for long_side in (True, False):
            a_len, f_len = (L, Wd) if long_side else (Wd, L)
            su, sv = sup_xy @ side, sup_xy @ front
            tu, tv = thr_xy @ side, thr_xy @ front
            u_lo, v_lo = np.percentile(su, 2) - a_len, np.percentile(sv, 2) - f_len
            nu, nv = int((np.percentile(su, 98) - u_lo) / STEP) + 2, int((np.percentile(sv, 98) - v_lo) / STEP) + 2
            nu_c, nv_c = int(round(a_len / STEP)), int(round(f_len / STEP))
            Hs = np.histogram2d(su, sv, bins=(nu + nu_c, nv + nv_c), range=((u_lo, u_lo + (nu + nu_c) * STEP), (v_lo, v_lo + (nv + nv_c) * STEP)))[0]
            Ht = np.histogram2d(tu, tv, bins=Hs.shape, range=((u_lo, u_lo + Hs.shape[0] * STEP), (v_lo, v_lo + Hs.shape[1] * STEP)))[0]
            S = lambda Hh: np.pad(Hh.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            Ss, St = S(Hs), S(Ht)
            box = lambda Sx: Sx[nu_c:nu_c + nu, nv_c:nv_c + nv] - Sx[:nu, nv_c:nv_c + nv] - Sx[nu_c:nu_c + nu, :nv] + Sx[:nu, :nv]
            score = box(Ss) - box(St)
            for i, j in zip(*np.nonzero(score >= score.max() * PLATEAU)):
                cu, cv = u_lo + (i + nu_c / 2) * STEP, v_lo + (j + nv_c / 2) * STEP
                results.append((float(score[i, j]), math.degrees(g), long_side, cu * side + cv * front,
                                int(box(Ss)[i, j]), int(box(St)[i, j])))
    best = max(r[0] for r in results)
    plateau = [r for r in results if r[0] >= best * PLATEAU]
    # the photo cannot tell these apart: use the one in the middle, keep the spread
    yaw_mid = float(np.median([r[1] for r in plateau]))
    same = [r for r in plateau if abs(r[1] - yaw_mid) <= 1.0 and r[2] == max(set(x[2] for x in plateau), key=[x[2] for x in plateau].count)]
    centres = np.array([r[3] for r in same])
    mid = centres[np.argmin(np.linalg.norm(centres - np.median(centres, 0), axis=1))]
    pick = min(same, key=lambda r: np.linalg.norm(r[3] - mid))
    _, yaw, long_side, c, n_sup, n_thr = pick
    before = dict(center_m=o["center_m"], yaw_deg=o["yaw_deg"], width_m=o["width_m"], depth_m=o["depth_m"])
    if item.get("what"):  # the user told us what it is: the detector's name is kept aside
        o.setdefault("name_detected", o["name"]); o["name"] = item["what"]
    set_body(o, c, yaw, *((L, Wd) if long_side else (Wd, L)))
    if True:  # the top is measured on its own points (the box height fails e.g. when the legs are hidden): stand it on the floor
        before.update(base_z_m=o["base_z_m"], height_m=o["height_m"])
        o["base_z_m"], o["height_m"] = 0.0, round(top, 3)
    o["completed_from_known_size"] = dict(real_size_m=[L, Wd], before=before, top_points=n_sup, seen_through_points=n_thr,
                                          placement_spread_m=round(float(np.ptp(centres, 0).max()), 2) if len(centres) > 1 else 0.0)
    return dict(id=o["id"], what=item.get("what"), top_m=round(top, 3), before=before, yaw=round(yaw, 1), centre=[round(float(x), 2) for x in c],
                top_points=n_sup, seen_through=n_thr, spread_m=o["completed_from_known_size"]["placement_spread_m"],
                plateau_centres=[[round(float(x), 2) for x in p] for p in centres[:: max(1, len(centres) // 40)]],
                yaw_choices=sorted(set(round(r[1]) for r in plateau)))


def corners_of(o):
    return np.array(footprint(o)[1])


def edge_gap(A, B):
    """Shortest distance between two convex rectangles on the floor plan (0 if they overlap)."""
    def inside(p, poly):
        s = [(lambda u, w: u[0] * w[1] - u[1] * w[0])(poly[(k + 1) % 4] - poly[k], p - poly[k]) for k in range(4)]
        return all(x >= 0 for x in s) or all(x <= 0 for x in s)
    if any(inside(p, B) for p in A) or any(inside(p, A) for p in B):
        return 0.0
    def pseg(p, a, b):
        v = b - a; t = np.clip((p - a) @ v / max(v @ v, 1e-12), 0, 1)
        return float(np.linalg.norm(p - (a + t * v)))
    return min(min(pseg(p, Q[k], Q[(k + 1) % 4]) for p in P for k in range(4)) for P, Q in ((A, B), (B, A)))


def main():
    folder = os.path.abspath(sys.argv[1])
    cfg = json.load(open(sys.argv[2], encoding="utf-8"))[os.path.basename(folder)]
    path = os.path.join(folder, "map.json")
    keep = os.path.join(folder, "map_before_known.json")
    if os.path.exists(keep):  # rerun: start again from the map as it was before
        shutil.copy(keep, path)
    else:
        shutil.copy(path, keep)
    m0 = json.load(open(path, encoding="utf-8"))
    gaps_before = {(g["a"], g["b"]): edge_gap(corners_of(next(o for o in m0["objects"] if o["id"] == g["a"])),
                                              corners_of(next(o for o in m0["objects"] if o["id"] == g["b"])))
                   for g in cfg.get("gaps", [])}

    # a known table height is a scale reference measured on the table top's points (not on its box)
    extra = []
    Q0 = None
    for item in cfg.get("complete", []):
        if item.get("height"):
            Q0 = load_points(folder, m0) if Q0 is None else Q0
            o = next(x for x in m0["objects"] if x["id"] == item["id"])
            top = top_height(folder, m0, Q0, o)
            extra.append((f"table top {item['id']} ({top:.3f} m seen)", item["id"], item["height"] / top))
    if cfg.get("scale") or extra:
        sys.argv = ["fix_scale.py", folder] + (["--measure", cfg["scale"]] if cfg.get("scale") else [])
        fix_scale.main(extra)
    m = json.load(open(path, encoding="utf-8"))
    Q = load_points(folder, m)
    report = dict(scale=m.get("scale_applied"), completed=[], gaps=[])
    for item in cfg.get("complete", []):
        r = complete(folder, m, Q, item)
        report["completed"].append(r)
        print(f"  completed #{r['id']}: " + (r.get("skipped") or
              f"yaw {r['yaw']}, {r['top_points']} top points, {r['seen_through']} seen-through, placement spread {r['spread_m']} m"))

    objs = m["objects"]
    m["distances"] = [{"a": p["id"], "b": q["id"], "distance_m": round(math.dist(footprint(p)[0], footprint(q)[0]), 2)}
                      for p, q in itertools.combinations(objs, 2)]
    if m.get("room"):
        measure_wall_distances(m, m["room"]["walls"])
    json.dump(m, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    by = {o["id"]: o for o in objs}
    for g in cfg.get("gaps", []):
        now = edge_gap(corners_of(by[g["a"]]), corners_of(by[g["b"]]))
        report["gaps"].append(dict(g, before=round(gaps_before[(g["a"], g["b"])], 2), after=round(now, 2)))
        print(f"  gap {g['what']}: real {g['real']:.2f}  before {gaps_before[(g['a'], g['b'])]:.2f}  after {now:.2f}")
    json.dump(report, open(os.path.join(folder, "known_sizes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
