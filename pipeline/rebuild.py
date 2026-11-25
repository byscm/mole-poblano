# Rebuild a scene folder after map.json was edited by hand (or by an AI agent): recomputes the object-to-object
# distances from the current positions, then rebuilds the Blender scene(s) and the plan. Nothing is re-detected.
# Usage:  python -I rebuild.py <map_folder> [--symbolic | --3d]
#   default: rebuilds the 3D scene (build_scene.py) when the folder has models, the symbolic one otherwise
#   --symbolic: only the boxes scene (fast)        --3d: only the scene with the models
# Writes: map.json (distances), plan.png, and symbolic_scene.blend + renders or scene.blend + render_3d.png/overlay.png
import json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe")


def run(title, cmd, folder):
    print(f"- {title} ...", flush=True)
    with open(os.path.join(folder, "_pipeline.log"), "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n===== {title}\n"); lf.flush()
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"  FAILED '{title}'. Check {os.path.join(folder, '_pipeline.log')}")


def main():
    args = sys.argv[1:]
    mode = "--symbolic" if "--symbolic" in args else "--3d" if "--3d" in args else None
    args = [a for a in args if a not in ("--symbolic", "--3d")]
    if not args or not os.path.exists(os.path.join(args[0], "map.json")):
        sys.exit("Usage: python -I rebuild.py <map_folder> [--symbolic | --3d]   (the folder must contain map.json)")
    out = os.path.abspath(args[0]); path = os.path.join(out, "map.json")
    scene_map = json.load(open(path, encoding="utf-8"))
    objs = scene_map["objects"]
    ids = [o["id"] for o in objs]
    if len(set(ids)) != len(ids):
        sys.exit(f"map.json has repeated object ids: {sorted(i for i in ids if ids.count(i) > 1)}")
    # same definition as detect_objects.py: centre to centre, in metres
    scene_map["distances"] = [{"a": p["id"], "b": q["id"], "distance_m": round(math.dist(p["center_m"][:2], q["center_m"][:2]), 3)}
                              for i, p in enumerate(objs) for q in objs[i + 1:]]
    for o in objs:  # the camera stands at the floor origin
        o["camera_distance_m"] = round(math.hypot(*o["center_m"][:2]), 3)
    json.dump(scene_map, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Map: {out}  ({len(objs)} objects, distances recomputed)")

    has_models = any(o.get("model") for o in objs) or os.path.isdir(os.path.join(out, "generated"))
    if mode == "--3d" and not has_models:
        sys.exit("No models in this folder (no generated\\ and no gallery models). Use --symbolic, or run photo_to_map.py without --symbolic.")
    if mode == "--symbolic" or (mode is None and not has_models):
        run("Symbolic scene in Blender", [BLENDER, "-b", "--factory-startup", "-P", os.path.join(HERE, "build_symbolic.py"), "--", out], out)
    else:
        run("Blender scene over the photo", [BLENDER, "-b", "--factory-startup", "-P", os.path.join(HERE, "build_scene.py"), "--", out], out)
    run("Plan with measurements", [sys.executable, "-I", os.path.join(HERE, "draw_plan.py"), out], out)
    print("Done: open plan.png" + (" and symbolic_overlay.png" if mode == "--symbolic" or not has_models else ", render_3d.png and overlay.png"))


if __name__ == "__main__":
    main()
