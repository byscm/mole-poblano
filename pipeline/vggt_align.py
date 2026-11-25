# Aligns MANY photos of one space with VGGT (all photos at once -> every camera's pose + depth), then puts each
# photo's own floor map into the reference photo's floor frame: a turn about the vertical + shift. VGGT has no real
# scale; it is fixed by comparing VGGT's depth with each photo's MoGe depth (metric, already used by the maps).
# The VGGT code + weights are the ones inside Orient Anything V2 (ORIENT_ANYTHING_DIR): nothing new
# to download.
# Run with the ComfyUI .venv python:  python vggt_align.py <out_folder> <reference_scene> <scene> ... [--extra <image_folder>]
#   --extra: more photos of the same space with no map (any orientation). They only help VGGT (more overlap -> better
#   poses for every camera, denser points); their cameras are reported too. Images then go in "pad" mode (no cropping).
# Writes <out_folder>/alignment.json: per photo {turn_deg, t_m (its camera on the reference plan), scale, tilt_error_deg,
#   depth_scale_spread}; and vggt_points.npz (all photos' points in the reference floor frame, for checking)
import json, math, os, sys
import numpy as np
import torch

OA = os.environ.get("ORIENT_ANYTHING_DIR", "orient-anything-v2")
sys.path.insert(0, OA)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from known_sizes import load_points


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    args = sys.argv[2:]
    extra = []
    if "--extra" in args:
        i = args.index("--extra"); ex = args[i + 1]; args = args[:i] + args[i + 2:]
        extra = sorted(os.path.join(ex, f) for f in os.listdir(ex) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    folders = args
    shapes = {(json.load(open(os.path.join(f, "map.json"), encoding="utf-8"))["W"] >= json.load(open(os.path.join(f, "map.json"), encoding="utf-8"))["H"]) for f in folders}
    mode = "pad" if extra or len(shapes) > 1 else "crop"  # mixed portrait / landscape photos: pad, never crop
    maps = [json.load(open(os.path.join(f, "map.json"), encoding="utf-8")) for f in folders]
    model = VGGT()
    sd = torch.load(os.path.join(OA, "weights", "rotmod_realrotaug_best.pt"), map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("model", sd); sd = sd.get("state_dict", sd)
    sd = {k[5:]: v for k, v in sd.items() if k.startswith("vggt.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("VGGT weights loaded; missing", len(missing), "unexpected", len(unexpected))
    model = model.cuda().eval()
    torch.cuda.empty_cache()
    imgs = load_and_preprocess_images([os.path.join(f, "photo.png") for f in folders] + extra, mode=mode).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(imgs)
    E, K = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
    E = E[0].float().cpu().numpy(); depth = pred["depth"][0, ..., 0].float().cpu().numpy(); dconf = pred["depth_conf"][0].float().cpu().numpy()
    wp = pred["world_points"][0].float().cpu().numpy(); wconf = pred["world_points_conf"][0].float().cpu().numpy()
    n, h, w = depth.shape
    print("images", imgs.shape, "depth", depth.shape)

    # scale: MoGe depth (metric) / VGGT depth on the same pixels, per photo
    scales = []
    for k, (f, m) in enumerate(zip(folders, maps)):
        P = load_points(f, m)  # floor frame, metres, at the map's W x H
        M = np.array(m["cv_axes_in_floor"], float).T  # columns: camera x, y, z axes in the floor frame
        cam = (P - np.array([0, 0, m["camera_height_m"]])) @ M  # back to camera coordinates
        if mode == "pad":  # the photo fills the width (landscape) or height, centred, rest is padding
            sc_ = max(m["W"], m["H"]) / w
            ox, oy = (w - m["W"] / sc_) / 2, (h - m["H"] / sc_) / 2
            ys = ((np.arange(h) + 0.5) - oy) * sc_; xs = ((np.arange(w) + 0.5) - ox) * sc_
        else:
            ys = (np.arange(h) + 0.5) * m["H"] / h; xs = (np.arange(w) + 0.5) * m["W"] / w
        iny = (ys >= 0) & (ys < m["H"]); inx = (xs >= 0) & (xs < m["W"])
        z_moge = np.full((h, w), np.nan)
        z_moge[np.ix_(iny, inx)] = cam[ys[iny].astype(int)[:, None], xs[inx].astype(int)[None, :], 2]
        good = np.isfinite(z_moge) & (z_moge > 0.3) & (z_moge < 8) & (dconf[k] > np.percentile(dconf[k], 40))
        r = z_moge[good] / depth[k][good]
        scales.append((float(np.median(r)), float(np.percentile(r, 75) / np.percentile(r, 25))))
        print(f"{os.path.basename(f)}: MoGe/VGGT depth ratio {scales[-1][0]:.3f} (spread {scales[-1][1]:.2f})")
    s = float(np.median([x[0] for x in scales]))

    M0 = np.array(maps[0]["cv_axes_in_floor"], float).T; h0 = maps[0]["camera_height_m"]
    R0, t0 = E[0][:, :3], E[0][:, 3]
    res = {}
    for k, (f, m) in enumerate(zip(folders, maps)):
        Rk, tk = E[k][:, :3], E[k][:, 3]
        Mk = np.array(m["cv_axes_in_floor"], float).T
        Q = M0 @ R0 @ Rk.T @ Mk.T  # photo k floor axes -> reference floor axes (should be a turn about z only)
        turn = math.degrees(math.atan2(Q[1, 0], Q[0, 0]))
        tilt = math.degrees(math.acos(np.clip(Q[2, 2], -1, 1)))
        Xk = -Rk.T @ tk  # camera k centre in VGGT world
        c_ref = M0 @ (s * (R0 @ Xk + t0)) + np.array([0, 0, h0])
        res[os.path.basename(f)] = dict(folder=os.path.abspath(f), turn_deg=round(turn, 2), t_m=[round(float(c_ref[0]), 3), round(float(c_ref[1]), 3)],
                                        camera_height_from_vggt_m=round(float(c_ref[2]), 3), camera_height_own_m=m["camera_height_m"],
                                        scale=1.0, tilt_error_deg=round(tilt, 2), depth_scale=round(scales[k][0] / s, 3))
        print(f"PLACED {os.path.basename(f)}: turn {turn:.1f}, camera at ({c_ref[0]:.2f}, {c_ref[1]:.2f}), height {c_ref[2]:.2f} "
              f"(own {m['camera_height_m']:.2f}), tilt error {tilt:.1f} deg")
    # the extra cameras: where they stand and where they look, on the reference plan
    xcam = {}
    for j, pth in enumerate(extra):
        k = len(folders) + j
        Rk, tk = E[k][:, :3], E[k][:, 3]
        c = M0 @ (s * (R0 @ (-Rk.T @ tk) + t0)) + np.array([0, 0, h0])
        fwd = M0 @ (R0 @ (Rk.T @ np.array([0, 0, 1.0])))
        xcam[os.path.basename(pth)] = dict(xyz_m=[round(float(v), 3) for v in c], look=[round(float(v), 3) for v in fwd])
    # all points in the reference floor frame, for a top-down check
    pts, cols, who = [], [], []
    for k in range(n):
        ok = (wconf[k] >= np.percentile(wconf[k], 50)) & (wconf[k] > 1.0001)  # >= : many ties at the floor value
        X = wp[k][ok]
        p = (M0 @ (s * (X @ R0.T + t0)).T).T + np.array([0, 0, h0])
        c = (imgs[k].permute(1, 2, 0).cpu().numpy()[ok] * 255).astype(np.uint8)
        pts.append(p); cols.append(c); who.append(np.full(len(p), k))
    np.savez_compressed(os.path.join(out, "vggt_points.npz"), points=np.concatenate(pts), colors=np.concatenate(cols), photo=np.concatenate(who))
    json.dump(dict(reference=os.path.basename(folders[0]), metric_scale=s, photos=res, extra_cameras=xcam,
                   extra_names=[os.path.basename(x) for x in extra]), open(os.path.join(out, "alignment.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
