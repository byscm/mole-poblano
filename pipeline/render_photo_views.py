# Renders the finished room (its own materials and lights, Cycles) from the camera of each photo, so the 3D can be
# put NEXT TO the photo. Cameras: "fit" = the photo_<name> cameras saved by fit_to_photos.py in the blend;
# "map" = the starting cameras from inventory_map.json (before any fit).
# Usage: blender -b <room.blend> -P render_photo_views.py -- <inventory_map.json> <fit|map> <out_folder> <photo,photo,...> [scale=0.5]
import bpy, json, math, os, sys
from mathutils import Matrix, Vector

a = sys.argv[sys.argv.index("--") + 1:]
MAP, MODE, OUT = (os.path.abspath(a[0]), a[1], os.path.abspath(a[2])); PHOTOS = a[3].split(","); SCALE = float(a[4]) if len(a) > 4 else 0.5
os.makedirs(OUT, exist_ok=True)
m = json.load(open(MAP, encoding="utf-8")); cams = {c["photo"]: c for c in m["cameras"]}
sc = bpy.context.scene
for o in sc.objects:  # inside the room: every wall, the ceiling and the tops must show
    if o.type in ("MESH", "LIGHT"): o.hide_render = False
sc.render.engine = "CYCLES"; sc.cycles.samples = int(os.environ.get("SAMPLES", "64")); sc.cycles.use_denoising = True
try:
    sc.cycles.device = "GPU"; p = bpy.context.preferences.addons["cycles"].preferences; p.compute_device_type = "OPTIX"; p.get_devices()
    for d in p.devices: d.use = True
except Exception as e: print("GPU", e)
cd = bpy.data.cameras.new("view"); cam = bpy.data.objects.new("view", cd); sc.collection.objects.link(cam); sc.camera = cam
cd.sensor_fit = "VERTICAL"; cd.sensor_width = cd.sensor_height = 36.0; cd.clip_start = 0.05
for ph in PHOTOS:
    c = cams[ph]; pm = json.load(open(os.path.join(c["folder"], "map.json"), encoding="utf-8"))
    if MODE == "fit":
        src = bpy.data.objects["photo_" + ph]; cam.matrix_world = src.matrix_world; cd.lens = src.data.lens
    else:
        place = Matrix.Translation((*c["xy_m"], 0)) @ Matrix.Rotation(math.radians(c["turn_deg"]), 4, "Z")
        ex, ey, ez = (Vector(v) for v in pm["cv_axes_in_floor"])
        cam.matrix_world = place @ Matrix.Translation((0, 0, pm["camera_height_m"] * c.get("shrink", 1.0))) @ Matrix((ex, -ey, -ez)).transposed().to_4x4()
        cd.lens = pm["fx_px"] / pm["H"] * 36.0
    sc.render.resolution_x, sc.render.resolution_y = round(pm["W"] * SCALE), round(pm["H"] * SCALE)
    sc.render.filepath = os.path.join(OUT, f"{MODE}_{ph}.png"); bpy.ops.render.render(write_still=True)
    print("RENDERED", ph, flush=True)
