# Aligns MANY photos of one space with no hand-picked objects: SIFT points matched between every pair of photos,
# each match lifted to 3D with that photo's MoGe points (floor frame, metres), then a turn about the vertical +
# shift (+ a small scale, MoGe's scale differs a few % per photo) fitted with RANSAC. The pairs are chained to the
# reference photo along the strongest links (maximum spanning tree), so photos that don't overlap the reference
# still get placed through others.
# Run with the ComfyUI .venv python (needs cv2):  python align_photos.py <out_folder> <reference_scene> <scene> ...
# Writes <out_folder>/alignment.json: per photo {turn_deg, t_m, scale, via, inliers} (point in photo -> s R p + t in
# the reference frame), pairs.json (every pair's fit) and matches_<a>_<b>.png for the links used.
import json, math, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from known_sizes import load_points

MAX_RANGE = 9.0      # m from its camera: farther points are too rough
MIN_Z = 0.04         # m: points at/below the floor are skipped (a shiny floor mirrors the room into fake points)
INLIER_M = 0.12      # m in 3D after alignment
RANSAC_IT = 4000


def load(folder):
    m = json.load(open(os.path.join(folder, "map.json"), encoding="utf-8"))
    img = cv2.imread(os.path.join(folder, "photo.png"))
    if img.shape[1] != m["W"]:
        img = cv2.resize(img, (m["W"], m["H"]))
    P = load_points(folder, m)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02)
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    uv = np.array([k.pt for k in kp]).round().astype(int)
    xyz = P[uv[:, 1].clip(0, m["H"] - 1), uv[:, 0].clip(0, m["W"] - 1)]
    ok = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz[:, :2], axis=1) < MAX_RANGE) & (xyz[:, 2] > MIN_Z)
    return dict(name=os.path.basename(os.path.abspath(folder)), folder=folder, img=img, kp=[k for k, o in zip(kp, ok) if o],
                des=des[ok], xyz=xyz[ok])


def fit(a, b):
    """Turn + shift + scale (b -> a) on the xy plane, z by the median gap; returns dict with inliers."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    mm = bf.knnMatch(b["des"], a["des"], k=2)
    good = [p[0] for p in mm if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]
    if len(good) < 8:
        return dict(inliers=0, matches=len(good))
    B = b["xyz"][[g.queryIdx for g in good]]; A = a["xyz"][[g.trainIdx for g in good]]
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(RANSAC_IT):
        i, j = rng.choice(len(good), 2, replace=False)
        vb, va = B[j, :2] - B[i, :2], A[j, :2] - A[i, :2]
        if np.linalg.norm(vb) < 0.3: continue
        s = np.linalg.norm(va) / np.linalg.norm(vb)
        if not 0.8 < s < 1.25: continue
        th = math.atan2(va[1], va[0]) - math.atan2(vb[1], vb[0])
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        t = A[i, :2] - s * R @ B[i, :2]
        dz = A[i, 2] - s * B[i, 2]
        res = np.linalg.norm(np.c_[(s * B[:, :2] @ R.T + t) - A[:, :2], s * B[:, 2] + dz - A[:, 2]], axis=1)
        n = int((res < INLIER_M).sum())
        if n > best[0]:
            best = (n, res < INLIER_M)
    n, inl = best
    if n < 6:
        return dict(inliers=n, matches=len(good))
    # refine on the inliers (2D similarity, Umeyama)
    Pb, Pa = B[inl, :2], A[inl, :2]
    mb, ma = Pb.mean(0), Pa.mean(0)
    C = (Pa - ma).T @ (Pb - mb) / len(Pb)
    U, S, Vt = np.linalg.svd(C); D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt; s = float(np.trace(np.diag(S) @ D) / ((Pb - mb) ** 2).sum(1).mean())
    t = ma - s * R @ mb
    res = np.linalg.norm(s * Pb @ R.T + t - Pa, axis=1)
    return dict(inliers=int(n), matches=len(good), turn_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])), t_m=t.tolist(), scale=s,
                median_error_m=float(np.median(res)), _pairs=[(g.queryIdx, g.trainIdx) for g, k in zip(good, inl) if k])


def as_matrix(f):
    th = math.radians(f["turn_deg"]); s = f["scale"]
    M = np.eye(3); M[:2, :2] = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]); M[:2, 2] = f["t_m"]
    return M


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    scenes = [load(f) for f in sys.argv[2:]]
    for s in scenes: print(f"{s['name']}: {len(s['kp'])} usable points")
    n = len(scenes); fits = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                fits[(i, j)] = fit(scenes[i], scenes[j])  # j -> i
                f = fits[(i, j)]
                print(f"{scenes[j]['name']} -> {scenes[i]['name']}: {f['inliers']} of {f['matches']} matches agree"
                      + (f"  turn {f['turn_deg']:.1f}, shift ({f['t_m'][0]:.2f}, {f['t_m'][1]:.2f}), scale {f['scale']:.3f}, "
                         f"median error {f['median_error_m']*100:.0f} cm" if "turn_deg" in f else ""))
    # strongest links first (Prim from the reference)
    placed = {0: dict(M=np.eye(3), via=None, inliers=None)}
    while len(placed) < n:
        cand = [(fits[(i, j)]["inliers"], i, j) for i in placed for j in range(n) if j not in placed and "turn_deg" in fits[(i, j)]]
        if not cand or max(cand)[0] < 25:
            break
        k, i, j = max(cand)
        placed[j] = dict(M=placed[i]["M"] @ as_matrix(fits[(i, j)]), via=scenes[i]["name"], inliers=k)
        a, b = scenes[i], scenes[j]
        pairs = fits[(i, j)]["_pairs"][:80]
        vis = cv2.drawMatches(b["img"], b["kp"], a["img"], a["kp"], [cv2.DMatch(q, t, 0) for q, t in pairs], None,
                              matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.path.join(out, f"matches_{b['name']}_to_{a['name']}.jpg"), cv2.resize(vis, None, fx=0.6, fy=0.6))
    res = {}
    for j, p in placed.items():
        M = p["M"]; s = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
        res[scenes[j]["name"]] = dict(folder=os.path.abspath(scenes[j]["folder"]), turn_deg=round(math.degrees(math.atan2(M[1, 0], M[0, 0])), 2),
                                      t_m=[round(float(v), 3) for v in M[:2, 2]], scale=round(s, 4), via=p["via"], inliers=p["inliers"])
    missing = [s["name"] for k, s in enumerate(scenes) if k not in placed]
    json.dump(dict(reference=scenes[0]["name"], photos=res, not_placed=missing), open(os.path.join(out, "alignment.json"), "w"), indent=1)
    json.dump({f"{scenes[j]['name']}->{scenes[i]['name']}": {k: v for k, v in f.items() if k != "_pairs"} for (i, j), f in fits.items()},
              open(os.path.join(out, "pairs.json"), "w"), indent=1)
    for k, v in res.items(): print(f"PLACED {k}: turn {v['turn_deg']}, camera at {v['t_m']}, scale {v['scale']} (via {v['via']}, {v['inliers']} points)")
    if missing: print("NOT PLACED:", missing)


if __name__ == "__main__":
    main()
