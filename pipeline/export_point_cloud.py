# Writes the scene's MoGe points as a colored PLY in the floor frame (metres, z up), the input SpatialLM expects.
# Usage: python -I export_point_cloud.py <scene_folder> [max_distance_m]
# Writes: <scene_folder>\point_cloud.ply
import json, math, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from room_layout import to_floor_points


def main():
    folder = os.path.abspath(sys.argv[1])
    max_dist = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    scene_map = json.load(open(os.path.join(folder, "map.json"), encoding="utf-8"))
    W, H = scene_map["W"], scene_map["H"]
    P = np.load(os.path.join(folder, "moge_points.npz"))["points"].reshape(H, W, 3).astype(np.float64)
    if np.nanmedian(P[..., 2]) < 0:
        P = P * np.array([1, -1, -1])
    Q = to_floor_points(scene_map, P).reshape(-1, 3)
    rgb = np.array(Image.open(os.path.join(folder, "photo.png")).convert("RGB")).reshape(-1, 3)
    top = (scene_map.get("room") or {}).get("ceiling_height_m") or 4.0
    keep = np.isfinite(Q).all(1) & (np.linalg.norm(Q[:, :2], axis=1) < max_dist) & (Q[:, 2] > -0.3) & (Q[:, 2] < top + 0.3)
    Q, rgb = Q[keep], rgb[keep].astype(np.uint8)
    # SpatialLM expects the room's walls along the x / y axes: turn the cloud so our longest wall is axis-aligned
    # (the turn is saved so run_spatiallm.py can turn its answer back)
    walls = (scene_map.get("room") or {}).get("walls", [])
    turn = 0.0
    if walls:
        w = max(walls, key=lambda w: w["length_m"])
        ang = math.atan2(w["end_m"][1] - w["start_m"][1], w["end_m"][0] - w["start_m"][0])
        turn = -(ang - round(ang / (math.pi / 2)) * math.pi / 2)
    c, s_ = math.cos(turn), math.sin(turn)
    Q = np.c_[Q[:, 0] * c - Q[:, 1] * s_, Q[:, 0] * s_ + Q[:, 1] * c, Q[:, 2]].astype(np.float32)
    json.dump(dict(turn_deg=math.degrees(turn)), open(os.path.join(folder, "point_cloud.json"), "w"))
    out = os.path.join(folder, "point_cloud.ply")
    rec = np.empty(len(Q), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = Q[:, 0], Q[:, 1], Q[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(out, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())
    print(f"{len(rec)} points -> {out} (turned {math.degrees(turn):.1f} deg)")


if __name__ == "__main__":
    main()
