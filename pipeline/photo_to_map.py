# ONE COMMAND: photo -> map (what is there, where, how big, where it faces, distances)
#                     -> 3D model of each thing (from the gallery if there is an identical one; otherwise TRELLIS 2)
#                     -> Blender scene with the same camera, checked over the photo.
# All local (ComfyUI + Blender). Works for any object, whether or not it is in the gallery.
#
# Usage:  python -I photo_to_map.py <photo> [output_folder] [--port 8188] [--fov DEGREES] [--symbolic] [--measure "..."] [--no-3d]
#         python -I photo_to_map.py <folder_with_map> --measure "1-2=2.5"     (only fixes the scale and redraws)
#   output_folder: by default escenas\<photo_name>
#   --fov:       horizontal lens angle in degrees if you know it (read automatically from EXIF when present)
#   --symbolic:  each object as a box with its size, number and arrow (fast, no TRELLIS)
#   --measure:   a measurement you know, to fix the scale of the whole map:
#                "1-2=2.5" (distance), "3=1.8" (height), "3:width=1.2", "camera=1.6"; several with commas
#   --no-3d:     only the map, the guides and the plan, no Blender
#
# Result in the folder:
#   plan.png              top view with sizes, arrows, distances and a 1 m ruler
#   symbolic_overlay.png, symbolic_scene.blend   (with --symbolic)
#   guides.png            photo with number, name, height and distances of each object
#   map.json              all the data (positions in metres, sizes, yaws, distances between all)
#   overlay.png           the 3D scene drawn over your photo (to check it)
#   comparison.png        photo | 3D scene
#   scene.blend           the scene to open in Blender
#   crops\, generated\    background-free crops and 3D models of each object

import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe")
PY = os.environ.get("COMFY_PYTHON", os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\python.exe"))
PYV = os.environ.get("COMFY_VENV_PYTHON", os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe"))


def option(args, name, default=None):
    if name in args:
        k = args.index(name)
        v = args[k + 1]
        del args[k:k + 2]
        return v
    return default


def step(title, cmd, folder):
    t = time.time()
    print(f"- {title} ...", flush=True)
    with open(os.path.join(folder, "_pipeline.log"), "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n===== {title}\n")
        lf.flush()
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"  FAILED '{title}'. Check {os.path.join(folder, '_pipeline.log')}")
    print(f"  done ({time.time()-t:.0f} s)", flush=True)


def symbolic(out):
    step("Symbolic scene in Blender (boxes with size, number and arrow)",
         [BLENDER, "-b", "--factory-startup", "-P", os.path.join(HERE, "build_symbolic.py"), "--", out], out)


def main():
    args = sys.argv[1:]
    port = option(args, "--port", "8188")
    fov = option(args, "--fov")
    measure = option(args, "--measure")
    flags = {f: (f in args) for f in ("--no-3d", "--symbolic")}
    args = [a for a in args if a not in flags]
    no3d, symb = flags["--no-3d"], flags["--symbolic"]
    if not args:
        sys.exit("Usage: python -I photo_to_map.py <photo> [output_folder] [--symbolic] [--measure \"1-2=2.5\"] [--fov DEG] [--port 8188] [--no-3d]\n"
                 "       python -I photo_to_map.py <folder_with_map> --measure \"1-2=2.5\"   (only fixes the scale of an existing map)")
    t0 = time.time()
    if os.path.isdir(args[0]) and os.path.exists(os.path.join(args[0], "map.json")):
        # existing map: only fix the scale with the measurement and redraw everything
        out = os.path.abspath(args[0])
        if not measure:
            sys.exit("A folder with an existing map needs --measure (for example --measure \"1-2=2.5\").")
        print(f"Map: {out}\n")
        step(f"Scale from your measurement ({measure})", [PY, "-I", os.path.join(HERE, "fix_scale.py"), out, "--measure", measure], out)
        step("Redraw guides", [PY, "-I", os.path.join(HERE, "detect_objects.py"), "--redraw", out], out)
        symbolic(out)
        step("Plan with measurements", [PY, "-I", os.path.join(HERE, "draw_plan.py"), out], out)
        summary(out, True, t0)
        return
    photo = os.path.abspath(args[0])
    out = os.path.abspath(args[1]) if len(args) > 1 else os.path.join(ROOT, "escenas", os.path.splitext(os.path.basename(photo))[0])
    if os.path.exists(os.path.join(out, "map.json")):
        sys.exit(f"There is already a map in {out}. Use another output folder (nothing is overwritten).\n"
                 f"To only fix its scale: python -I photo_to_map.py \"{out}\" --measure \"1-2=2.5\"")
    os.makedirs(out, exist_ok=True)
    print(f"Photo: {photo}\nOutput: {out}\n")

    step("Map: what is there, where and how big (Qwen3-VL + SAM 3 + MoGe-2)",
         [PY, "-I", os.path.join(HERE, "detect_objects.py"), photo, out, port] + (["--fov", fov] if fov else []), out)
    step("Facing: which way each object looks (Orient Anything V2)", [PYV, "-I", os.path.join(HERE, "orientation.py"), out], out)
    if not symb:
        step("Gallery: use an existing model if there is an identical one (DINOv3)", [PYV, os.path.join(HERE, "gallery_match.py"), out], out)
    if measure:
        step(f"Scale from your measurement ({measure})", [PY, "-I", os.path.join(HERE, "fix_scale.py"), out, "--measure", measure], out)
        step("Redraw guides with the corrected scale", [PY, "-I", os.path.join(HERE, "detect_objects.py"), "--redraw", out], out)
    elif not (fov or json.load(open(os.path.join(out, "map.json"), encoding="utf-8"))["fov_source"] == "EXIF"):
        step("Scale from references of known size (people, gallery)", [PY, "-I", os.path.join(HERE, "fix_scale.py"), out], out)
        step("Redraw guides with the corrected scale", [PY, "-I", os.path.join(HERE, "detect_objects.py"), "--redraw", out], out)
    if symb:
        symbolic(out)
    elif not no3d:
        step("Generate in 3D what is not in the gallery (TRELLIS 2)", [PY, "-I", os.path.join(HERE, "generate_models.py"), out, port], out)
        step("Orientation of each model by comparing silhouettes", [PY, "-I", os.path.join(HERE, "fit_poses.py"), out], out)
        step("Blender scene over the photo", [BLENDER, "-b", "--factory-startup", "-P", os.path.join(HERE, "build_scene.py"), "--", out], out)
    step("Plan with measurements", [PY, "-I", os.path.join(HERE, "draw_plan.py"), out], out)
    summary(out, symb or no3d, t0)


def summary(out, no_models, t0):
    scene_map = json.load(open(os.path.join(out, "map.json"), encoding="utf-8"))
    print(f"\nLENS {scene_map['fov_deg']:.1f} degrees ({scene_map['fov_source']})   CAMERA {scene_map['camera_height_m']:.2f} m above the floor")
    if "scale_applied" in scene_map:
        e = scene_map["scale_applied"]
        print(f"SCALE corrected x{e['factor']} with {e.get('source', 'references')}: " + ", ".join(r["kind"] for r in e["references"]))
    else:
        print("SCALE not corrected (if you know a measurement, fix it with --measure)")
    for o in scene_map["objects"]:
        source = "gallery" if o.get("model") else ("-" if no_models else "generated")
        warning = "   ** check model **" if o.get("needs_review") and not no_models else ""
        yaw = o.get("yaw_from_appearance", o["yaw_deg"])
        print(f"  {o['id']:2d}. {o['name'][:28]:28s} {o['width_m']:.2f} x {o['depth_m']:.2f} x {o['height_m']:.2f} m  "
              f"at {o['camera_distance_m']:.2f} m  faces {yaw:5.1f} deg  model: {source}{warning}")
    extra = next((f for f in ("symbolic_overlay.png", "overlay.png", "guides.png") if os.path.exists(os.path.join(out, f))), "guides.png")
    print(f"\nTotal {time.time()-t0:.0f} s.  Open {os.path.join(out, 'plan.png')} and {os.path.join(out, extra)}")


if __name__ == "__main__":
    main()
