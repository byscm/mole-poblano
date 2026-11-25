# Same job and output format as vggt_align.py, using MapAnything (Apache 2.0 weights): one joint pass over all photos
# gives every camera's pose and metric depth, then each photo's floor map is placed in the reference photo's floor frame.
# Run with the python of your map-anything venv:  python mapanything_align.py <out_folder> <reference_scene> <scene> ...
# Writes <out_folder>/alignment.json and vggt_points.npz (same names and fields, so merge_photos.py and the plan tools work).
import json, math, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from known_sizes import load_points
from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def main():
    out, folders = sys.argv[1], sys.argv[2:]; os.makedirs(out, exist_ok=True)
    maps = [json.load(open(os.path.join(f, "map.json"), encoding="utf-8")) for f in folders]
    model = MapAnything.from_pretrained("facebook/map-anything-apache").cuda().eval()
    views = load_images([os.path.join(f, "photo.png") for f in folders])
    t0 = time.time()
    preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16", apply_mask=True, mask_edges=True,
                        apply_confidence_mask=False)
    print(f"inference {time.time() - t0:.1f} s, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, {len(preds)} views", flush=True)
    T = [p["camera_poses"][0].float().cpu().numpy() for p in preds]  # OpenCV cam2world
    depth = [p["depth_z"][0, ..., 0].float().cpu().numpy() for p in preds]
    conf = [p["conf"][0].float().cpu().numpy() for p in preds]
    pts = [p["pts3d"][0].float().cpu().numpy() for p in preds]
    masks = [p["mask"][0].squeeze().cpu().numpy().astype(bool) for p in preds]
    cols = [(p["img_no_norm"][0].float().cpu().numpy() * 255).astype(np.uint8) for p in preds]
    print("depth shape", depth[0].shape, flush=True)

    scales = []  # MoGe depth / MapAnything depth on the same pixels (photo and prediction share the framing up to a small crop)
    for k, (f, m) in enumerate(zip(folders, maps)):
        P = load_points(f, m); M = np.array(m["cv_axes_in_floor"], float).T
        cam = (P - np.array([0, 0, m["camera_height_m"]])) @ M; h, w = depth[k].shape
        ys = ((np.arange(h) + 0.5) * m["H"] / h).astype(int); xs = ((np.arange(w) + 0.5) * m["W"] / w).astype(int)
        z = cam[ys[:, None], xs[None, :], 2]
        good = np.isfinite(z) & (z > 0.3) & (z < 8) & masks[k] & (depth[k] > 0.1)
        r = z[good] / depth[k][good]
        scales.append((float(np.median(r)), float(np.percentile(r, 75) / np.percentile(r, 25))))
        print(f"{os.path.basename(f)}: MoGe/MapAnything depth ratio {scales[-1][0]:.3f} (spread {scales[-1][1]:.2f})", flush=True)
    s = float(np.median([x[0] for x in scales]))  # keep the maps' scale so the result compares directly with VGGT's

    M0 = np.array(maps[0]["cv_axes_in_floor"], float).T; h0 = maps[0]["camera_height_m"]
    R0, C0 = T[0][:3, :3].T, T[0][:3, 3]
    to_ref = lambda X: (M0 @ (s * (R0 @ (X - C0).T))).T + np.array([0, 0, h0])
    res = {}
    for k, (f, m) in enumerate(zip(folders, maps)):
        Mk = np.array(m["cv_axes_in_floor"], float).T
        Q = M0 @ R0 @ T[k][:3, :3] @ Mk.T  # photo k floor axes -> reference floor axes
        turn = math.degrees(math.atan2(Q[1, 0], Q[0, 0])); tilt = math.degrees(math.acos(np.clip(Q[2, 2], -1, 1)))
        c = to_ref(T[k][:3, 3][None])[0]; n = os.path.basename(f)
        res[n] = dict(folder=os.path.abspath(f), turn_deg=round(turn, 2), t_m=[round(float(c[0]), 3), round(float(c[1]), 3)],
                      camera_height_from_vggt_m=round(float(c[2]), 3), camera_height_own_m=m["camera_height_m"], scale=1.0,
                      tilt_error_deg=round(tilt, 2), depth_scale=round(scales[k][0] / s, 3))
        print(f"PLACED {n}: turn {turn:.1f}, camera at ({c[0]:.2f}, {c[1]:.2f}), height {c[2]:.2f} (own {m['camera_height_m']:.2f}), tilt error {tilt:.1f} deg", flush=True)
    P_, C_, W_ = [], [], []
    for k in range(len(preds)):
        ok = masks[k] & (conf[k] >= np.percentile(conf[k], 50)); P_.append(to_ref(pts[k][ok])); C_.append(cols[k][ok]); W_.append(np.full(int(ok.sum()), k))
    np.savez_compressed(os.path.join(out, "vggt_points.npz"), points=np.concatenate(P_), colors=np.concatenate(C_), photo=np.concatenate(W_))
    json.dump(dict(reference=os.path.basename(folders[0]), metric_scale=s, method="MapAnything Apache", photos=res, extra_cameras={}, extra_names=[]),
              open(os.path.join(out, "alignment.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
