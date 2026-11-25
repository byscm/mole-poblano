# LingBot-Map poses (pred.npz from lingbot_run.py) -> the alignment files the rest of the pipeline reads, same format as
# vggt_align.py. Scale comes from the MoGe / LingBot depth ratio, everything is placed in the reference photo's floor frame.
# MoGe pixels use the centre crop LingBot applies to portrait photos.
# Usage: python -I lingbot_align.py <pred.npz> <out_folder> <scene_folder> ...   (same order as the run)
# Writes alignment.json, alignment_good.json (without frames whose tilt or camera height disagree) and vggt_points.npz.
import json, math, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from known_sizes import load_points

MAX_TILT_DEG, MAX_HEIGHT_ERROR_M = 20.0, 0.6
pred, out, folders = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
d = np.load(pred)
E = d["extrinsic"].astype(np.float64)  # w2c, OpenCV
K = d["intrinsic"].astype(np.float64)
depth = d["depth"].astype(np.float32); depth = depth[..., 0] if depth.ndim == 4 else depth
dconf = d["depth_conf"].astype(np.float32)
names = [os.path.splitext(os.path.basename(str(p)))[0] for p in d["paths"]]
assert names == [os.path.basename(os.path.normpath(f)) for f in folders], (names, folders)
maps = [json.load(open(os.path.join(f, "map.json"), encoding="utf-8")) for f in folders]
n, h, w = depth.shape

scales = []
for k, (f, m) in enumerate(zip(folders, maps)):
    P = load_points(f, m)
    M = np.array(m["cv_axes_in_floor"], float).T
    cam = (P - np.array([0, 0, m["camera_height_m"]])) @ M
    nh = round(m["H"] * w / m["W"]); y0 = max(0, (nh - h) // 2)  # resize to width w, centre-crop height to h
    ys = (np.arange(h) + y0 + 0.5) * m["H"] / nh; xs = (np.arange(w) + 0.5) * m["W"] / w
    z_moge = cam[ys.astype(int)[:, None], xs.astype(int)[None, :], 2]
    good = np.isfinite(z_moge) & (z_moge > 0.3) & (z_moge < 8) & (dconf[k] > np.percentile(dconf[k], 40))
    r = z_moge[good] / depth[k][good]
    scales.append((float(np.median(r)), float(np.percentile(r, 75) / np.percentile(r, 25))))
    print(f"{names[k]}: MoGe/LingBot depth ratio {scales[-1][0]:.3f} (spread {scales[-1][1]:.2f})")
s = float(np.median([x[0] for x in scales]))

M0 = np.array(maps[0]["cv_axes_in_floor"], float).T; h0 = maps[0]["camera_height_m"]
R0, t0 = E[0][:, :3], E[0][:, 3]
res, pts, cols, who = {}, [], [], []
for k, (f, m) in enumerate(zip(folders, maps)):
    Rk, tk = E[k][:, :3], E[k][:, 3]
    Mk = np.array(m["cv_axes_in_floor"], float).T
    Q = M0 @ R0 @ Rk.T @ Mk.T
    turn = math.degrees(math.atan2(Q[1, 0], Q[0, 0]))
    tilt = math.degrees(math.acos(np.clip(Q[2, 2], -1, 1)))
    c_ref = M0 @ (s * (R0 @ (-Rk.T @ tk) + t0)) + np.array([0, 0, h0])
    res[names[k]] = dict(folder=os.path.abspath(f), turn_deg=round(turn, 2), t_m=[round(float(c_ref[0]), 3), round(float(c_ref[1]), 3)],
                         camera_height_from_vggt_m=round(float(c_ref[2]), 3), camera_height_own_m=m["camera_height_m"],
                         scale=1.0, tilt_error_deg=round(tilt, 2), depth_scale=round(scales[k][0] / s, 3))
    print(f"PLACED {names[k]}: turn {turn:.1f}, camera at ({c_ref[0]:.2f}, {c_ref[1]:.2f}), height {c_ref[2]:.2f} "
          f"(own {m['camera_height_m']:.2f}), tilt error {tilt:.1f} deg, depth_scale {scales[k][0] / s:.3f}")
    # this frame's confident pixels in the reference floor frame, for the plan tools
    v, u = np.mgrid[0:h:2, 0:w:2]; z = depth[k][::2, ::2]; ok = (dconf[k][::2, ::2] >= np.percentile(dconf[k], 50)) & (z > 0)
    rays = np.c_[(u[ok] + 0.5 - K[k, 0, 2]) / K[k, 0, 0], (v[ok] + 0.5 - K[k, 1, 2]) / K[k, 1, 1], np.ones(int(ok.sum()))]
    X = (rays * z[ok][:, None] - tk) @ Rk  # camera -> LingBot world
    pts.append((M0 @ (s * (R0 @ X.T + t0[:, None]))).T + np.array([0, 0, h0]))
    nh = round(m["H"] * w / m["W"]); y0 = max(0, (nh - h) // 2)
    rgb = np.asarray(Image.open(os.path.join(f, "photo.png")).convert("RGB").resize((w, nh)))[y0:y0 + h]
    cols.append(rgb[::2, ::2][ok]); who.append(np.full(len(X), k))

meta = dict(reference=names[0], metric_scale=s, source="LingBot-Map", extra_cameras={}, extra_names=[])
json.dump(dict(meta, photos=res), open(os.path.join(out, "alignment.json"), "w"), indent=1)
kept = {a: b for a, b in res.items() if b["tilt_error_deg"] <= MAX_TILT_DEG and abs(b["camera_height_from_vggt_m"] - b["camera_height_own_m"]) <= MAX_HEIGHT_ERROR_M}
json.dump(dict(meta, photos=kept), open(os.path.join(out, "alignment_good.json"), "w"), indent=1)
np.savez_compressed(os.path.join(out, "vggt_points.npz"), points=np.concatenate(pts), colors=np.concatenate(cols), photo=np.concatenate(who))
print(f"KEPT {len(kept)} of {len(res)} frames; dropped: {sorted(set(res) - set(kept)) or 'none'}")
