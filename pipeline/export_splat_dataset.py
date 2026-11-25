# Exports a room's frames as a nerfstudio-style dataset for a Gaussian-splat trainer (Brush), using the SAME cameras the
# 3D room is built and compared with (inventory_map.json cameras + each frame's map.json), so both are judged from the
# same viewpoints. Held-out photos go to transforms_test.json only: the splat never sees them while training.
# Usage: python -I export_splat_dataset.py <inventory_map.json> <aligned/vggt_points.npz> <out_folder> <held_out,photos,...>
#   -> <out>/images/*.png, transforms_train.json, transforms_test.json, points.ply (VGGT points as the starting cloud)
import json, os, shutil, sys
import numpy as np

MAP, PTS, OUT = (os.path.abspath(p) for p in sys.argv[1:4]); HOLD = set(sys.argv[4].split(","))
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
m = json.load(open(MAP, encoding="utf-8"))


def rot_z(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


train, test = [], []
for c in m["cameras"]:
    pm = json.load(open(os.path.join(c["folder"], "map.json"), encoding="utf-8"))
    ex, ey, ez = (np.array(v, float) for v in pm["cv_axes_in_floor"])
    R = np.eye(4); R[:3, :3] = np.stack([ex, -ey, -ez], 1)  # columns: camera x, y (up), z (back) -> OpenGL camera
    T = np.eye(4); T[:3, 3] = [c["xy_m"][0], c["xy_m"][1], 0]
    H = np.eye(4); H[2, 3] = pm["camera_height_m"] * c.get("shrink", 1.0)
    M = T @ rot_z(c["turn_deg"]) @ H @ R  # same matrix build_house.py / render_photo_views.py use
    name = c["photo"] + ".png"; shutil.copy(os.path.join(c["folder"], "photo.png"), os.path.join(OUT, "images", name))
    fr = dict(file_path="images/" + name, transform_matrix=M.tolist(), fl_x=pm["fx_px"], fl_y=pm["fx_px"],
              cx=pm["W"] / 2, cy=pm["H"] / 2, w=pm["W"], h=pm["H"])
    (test if c["photo"] in HOLD else train).append(fr)

d = np.load(PTS); P, C = d["points"], d["colors"]
rng = np.random.default_rng(0); k = rng.choice(len(P), min(len(P), 200000), replace=False); P, C = P[k], C[k]
C = C if C.max() > 1.5 else (C * 255)
with open(os.path.join(OUT, "points.ply"), "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, cc in zip(P, C.astype(np.uint8)): f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {cc[0]} {cc[1]} {cc[2]}\n")
for nm, fr in (("transforms_train.json", train), ("transforms_test.json", test)):
    json.dump(dict(camera_model="OPENCV", ply_file_path="points.ply", frames=fr), open(os.path.join(OUT, nm), "w"), indent=1)
print(len(train), "train,", len(test), "test (held out):", sorted(f["file_path"] for f in test))
