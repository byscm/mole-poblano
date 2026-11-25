# Computes the pose (yaw, whether it lies down, fake-base cut) of every generated model of a map,
# comparing its silhouette seen from the real camera with the crop from the photo.
# Usage: python -I fit_poses.py <map_folder>
# Writes: generated/object_N_pose.json, and updates map.json
# (yaw_from_appearance, silhouette_score, needs_review)
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silhouette import fit_pose


def elevation(scene_map, o):
    h = scene_map["camera_height_m"] - (o.get("base_z_m", 0) + o["height_m"] / 2)
    return math.degrees(math.atan2(h, max(o["camera_distance_m"], 0.05)))


def main():
    scene = sys.argv[1]
    map_path = os.path.join(scene, "map.json")
    scene_map = json.load(open(map_path, encoding="utf-8"))
    for o in scene_map["objects"]:
        gen = os.path.join(scene, "generated")
        os.makedirs(gen, exist_ok=True)
        glb = o.get("model") or os.path.join(gen, f"object_{o['id']}.glb")
        if not os.path.exists(glb):
            continue
        p = fit_pose(glb, os.path.join(scene, o["crop"]), elevation(scene_map, o))
        json.dump(p, open(os.path.join(gen, f"object_{o['id']}_pose.json"), "w"), indent=1)
        # where the model's front faces in the floor frame (0 = towards -y)
        towards = math.degrees(math.atan2(-o["center_m"][1], -o["center_m"][0])) + 90
        o["yaw_from_appearance"] = round((towards + p["yaw"]) % 360, 1)
        o["silhouette_score"] = p["iou"]
        o["needs_review"] = p["iou"] < 0.6
        print(f"object {o['id']} {o['name']:28s} silhouette {p['iou']:.2f}  yaw {p['yaw']:3d}  lying {p['pitch']:4d}  "
              f"fake base {'yes' if p['base_cut'] is not None else 'no'}  elev {p['elev']}", flush=True)
    json.dump(scene_map, open(map_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
